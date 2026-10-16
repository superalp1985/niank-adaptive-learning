"""Finite paired practice from an intact V22 runtime, never a weight-only fork."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import torch

from tracking_v22.run import dump
from tracking_v22.evaluate_stages import parameter_hash
from tracking_v23.evaluate import evaluate_checkpoint, file_hash
from tracking_v23.session import Session

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DEFAULT_SOURCE = REPO/"tracking_v22/results/patient_practice_20261016/events.sqlite"
RUNTIME_SOURCES = [
    "tracking_v17/world.py", "tracking_v22/__init__.py", "tracking_v22/world.py",
    "tracking_v22/neural.py", "tracking_v22/verified.py", "tracking_v22/run.py",
    "tracking_v22/evaluate_stages.py", "tracking_v23/__init__.py",
    "tracking_v23/vision.py", "tracking_v23/neural.py", "tracking_v23/session.py",
    "tracking_v23/evaluate.py", "tracking_v23/run.py",
]


def source_identity(path):
    path = Path(path).resolve()
    db = sqlite3.connect(path.as_uri()+"?mode=ro", uri=True)
    try:
        row = db.execute("SELECT step,sha256,payload FROM runtime WHERE id=1").fetchone()
        if row is None or hashlib.sha256(row[2]).hexdigest() != row[1]:
            raise RuntimeError("Source lacks a complete hash-verified runtime")
        counts = {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                  for table in ("facts", "events", "choices", "memory_events")}
        return dict(path=str(path), runtime_step=row[0], runtime_sha256=row[1],
                    database_sha256=file_hash(path), rows=counts)
    finally:
        db.close()


def snapshot(output, config, source):
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for relative in RUNTIME_SOURCES:
        current = REPO/relative
        hashes[relative] = file_hash(current)
        destination = output/"source"/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(current, destination)
    for current in ROOT.glob("*.md"):
        destination = output/"source"/"tracking_v23"/current.name
        shutil.copyfile(current, destination)
    for current in [*ROOT.glob("test_*.py"), *ROOT.glob("*.json"), ROOT/"audit_results.py"]:
        if current.exists():
            shutil.copyfile(current, output/"source"/"tracking_v23"/current.name)
    manifest = dict(config=config, source_runtime=source, sources=hashes,
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    sensory="native_1024x512_100Hz", physics_dt=.001, neural_dt=.0001,
                    input_dim=964 if config["arm"] == "active" else 576,
                    panorama="all pixels retained; head rotation rounded to one native pixel",
                    checkpoint_semantics="canonical physics solver cache reset in both arms")
    dump(output/"manifest.json", manifest)
    return manifest


def verify_manifest(output):
    manifest = json.loads((Path(output)/"manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["sources"].items():
        if file_hash(REPO/relative) != expected:
            raise RuntimeError("Resume requires the saved source version: "+relative)
    return manifest


def _save_agent_file(session, path):
    if not path.exists():
        temporary = path.with_suffix(".partial")
        torch.save(session.agent, temporary)
        temporary.replace(path)
    else:
        saved = torch.load(path, weights_only=False)
        if (parameter_hash(saved) != parameter_hash(session.agent)
                or saved.updates != session.agent.updates
                or saved.meta_updates != session.agent.meta_updates):
            raise RuntimeError("Existing model checkpoint differs from the committed runtime")
    return path


def _save_agent(session, relative):
    if session.step-session.origin_step != relative:
        raise ValueError("Milestone label must match the current continuation step")
    return _save_agent_file(session, session.output/f"learner_step_{relative:05d}.pt")


def run_arm(config, output, source, identity):
    output = Path(output)
    if (output/"COMPLETE.json").exists():
        raise RuntimeError("Completed arm must not be rerun")
    if (output/"manifest.json").exists():
        manifest = verify_manifest(output)
        if manifest["config"] != config or manifest["source_runtime"] != identity:
            raise RuntimeError("Resume configuration or V22 source changed")
    else:
        snapshot(output, config, identity)
    if (output/"events.sqlite").exists():
        session = Session.restore(output)
        if not hasattr(session, "origin_step"):
            row = session.db.execute("SELECT sha256 FROM runtime WHERE id=1").fetchone()
            if row[0] != identity["runtime_sha256"]:
                session.close()
                raise RuntimeError("Unrecognized incomplete continuation bootstrap")
            Session.initialize_fork(session, config)
            session.checkpoint()
    else:
        session = Session.fork(source, output, config)
        session.checkpoint()
    try:
        initial = _save_agent(session, 0) if session.step == session.origin_step else output/"learner_step_00000.pt"
        if not initial.exists():
            raise RuntimeError("Missing start milestone")
        stages = sorted(set([0, config["additional_steps"]//2, config["additional_steps"]]))
        stage_results = {}
        while True:
            relative = session.step-session.origin_step
            if relative <= config["additional_steps"] and (
                    relative % config["milestone_every"] == 0 or relative in stages):
                _save_agent(session, relative)
            # Probes are always scored against immutable milestone files.
            for stage in stages:
                checkpoint = output/f"learner_step_{stage:05d}.pt"
                if checkpoint.exists():
                    result = evaluate_checkpoint(checkpoint,
                        output/"stage_probes"/f"stage_{stage:05d}", config,
                        config["stage_probe_steps"], config["stage_scene_seed"])
                    stage_results[str(stage)] = result
            if relative >= config["additional_steps"]:
                break
            boundaries = [config["additional_steps"]]
            boundaries += [stage for stage in stages if stage > relative]
            boundaries += [((relative//config["milestone_every"])+1)*config["milestone_every"]]
            count = min(config["checkpoint_every"], min(boundaries)-relative)
            session.advance(count)
            session.checkpoint()
            relative = session.step-session.origin_step
            if relative % config["milestone_every"] == 0 or relative in stages:
                _save_agent(session, relative)
        if not (output/"training_only_summary.json").exists():
            dump(output/"training_only_summary.json", session.summary())
        session.learning_enabled = False
        settle_steps = config.get("settling_steps", max(config["block"], config["horizon"]))
        end = session.origin_step+config["additional_steps"]+settle_steps
        while session.step < end:
            session.advance(min(config["checkpoint_every"], end-session.step))
            session.checkpoint()
        if session.records:
            session.finish_block()
            session.checkpoint()
        if session.agent.trial is not None:
            raise RuntimeError("Fresh-window credit remains pending after the finite settling interval")
        final = output/"learner.pt"
        _save_agent_file(session, final)
        session.save_gif()
        summary = session.summary()
        dump(output/"summary.json", summary)
        final_probe = evaluate_checkpoint(final, output/"final_probe", config,
            config["final_probe_steps"], config["final_scene_seed"])
        diagnostics = {}
        if config["arm"] == "active" and config.get("final_diagnostics", True):
            for diagnostic in ("uniform", "head_hold"):
                diagnostics[diagnostic] = evaluate_checkpoint(final,
                    output/"diagnostics"/diagnostic, config, config["final_probe_steps"],
                    config["final_scene_seed"], diagnostic)
        result = dict(config=config, training=summary, stages=stage_results,
                      final_probe=final_probe, diagnostics=diagnostics,
                      source_runtime=identity)
        dump(output/"COMPLETE.json", result)
        print(json.dumps(dict(event="arm_complete", arm=config["arm"],
                              continuation=summary["continuation"])), flush=True)
        return result
    finally:
        session.close()


def run(config, output, source=DEFAULT_SOURCE, arm="both"):
    torch.set_num_threads(4)
    output, source = Path(output).resolve(), Path(source).resolve()
    if output.drive.upper() != "E:":
        raise ValueError("Experiment data must stay on E:")
    if (output/"COMPLETE.json").exists():
        raise RuntimeError("Completed paired configuration must not be rerun")
    identity = source_identity(source)
    results = {}
    arms = ("baseline", "active") if arm == "both" else (arm,)
    for name in arms:
        arm_config = dict(config, arm=name)
        path = output/name
        if arm == "both" and (path/"COMPLETE.json").exists():
            manifest = verify_manifest(path)
            if manifest["config"] != arm_config or manifest["source_runtime"] != identity:
                raise RuntimeError("Completed arm identity differs")
            results[name] = json.loads((path/"COMPLETE.json").read_text(encoding="utf-8"))
            continue
        results[name] = run_arm(arm_config, path, source, identity)
    if source_identity(source) != identity:
        raise RuntimeError("Source runtime changed during continuation")
    if arm == "both":
        dump(output/"COMPLETE.json", dict(config=config, source_runtime=identity,
                                        arms={key: str(output/key) for key in results}))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT/"extended_practice.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--arm", choices=("baseline", "active", "both"), default="both")
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")), args.output, args.source, args.arm)


if __name__ == "__main__":
    main()
