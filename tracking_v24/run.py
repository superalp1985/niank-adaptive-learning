"""Finite V24 forks; immutable milestones and source-checked runtime resume."""
import argparse
import json
from pathlib import Path
import shutil

import torch

from tracking_v22.evaluate_stages import parameter_hash
from tracking_v23.persistence import (
    atomic_json as dump, preserve_incomplete_bootstrap, read_complete,
)
from tracking_v23.run import RUNTIME_SOURCES as V23_SOURCES, source_identity
from tracking_v24.evaluate import evaluate, file_hash
from tracking_v24.session import Session

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DEFAULT_SOURCE = REPO/"tracking_v23/results/extended_practice_20261017/active/events.sqlite"
RUNTIME_SOURCES = V23_SOURCES+[
    "tracking_v24/__init__.py", "tracking_v24/credit.py", "tracking_v24/neural.py",
    "tracking_v24/session.py", "tracking_v24/evaluate.py", "tracking_v24/run.py",
]


def manifest(output, config, identity):
    output.mkdir(parents=True, exist_ok=True)
    path = output/"manifest.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["config"] != config or saved["source_runtime"] != identity:
            raise RuntimeError("Resume identity differs")
        if set(saved["sources"]) != set(RUNTIME_SOURCES):
            raise RuntimeError("Resume source list differs")
        for relative, expected in saved["sources"].items():
            if file_hash(REPO/relative) != expected:
                raise RuntimeError("Resume source changed: "+relative)
        return saved
    if (output/"events.sqlite").exists():
        raise RuntimeError("Existing runtime has no manifest; refusing to invent provenance")
    hashes = {}
    for relative in RUNTIME_SOURCES:
        hashes[relative] = file_hash(REPO/relative)
        destination = output/"source"/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/relative, destination)
    for source in ROOT.iterdir():
        if source.suffix in (".md", ".json", ".py"):
            shutil.copyfile(source, output/"source"/"tracking_v24"/source.name)
    value = dict(config=config, source_runtime=identity, sources=hashes,
                 torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                 sensory="native_1024x512_100Hz", physics_dt=.001, neural_dt=.0001)
    dump(path, value)
    return value


def save_agent(session, name):
    path = session.output/name
    if path.exists():
        existing = torch.load(path, weights_only=False)
        if (parameter_hash(existing) != parameter_hash(session.agent)
                or existing.updates != session.agent.updates
                or getattr(existing, "delayed_updates", 0)
                != getattr(session.agent, "delayed_updates", 0)):
            raise RuntimeError("Milestone differs from committed runtime")
    else:
        temporary = path.with_suffix(".partial")
        torch.save(session.agent, temporary)
        temporary.replace(path)
    return path


def valid_arm(result):
    config, summary = result["config"], result["training"]
    return (summary["new_steps"] == config["additional_steps"]+max(config["delay_horizons"])
            and summary["unsettled_learning_choices"] == 0
            and summary["delayed"]["pending"] == summary["delayed"]["ready"] == 0
            and result["final_probe"]["learned_parameters_unchanged"] is True
            and set(result["stages"]) == {"0", str(config["additional_steps"]//2),
                                         str(config["additional_steps"])})


def run_arm(config, output, source, identity):
    manifest(output, config, identity)
    complete = read_complete(output/"COMPLETE.json", valid_arm)
    if complete is not None:
        if complete["config"] != config or complete["source_runtime"] != identity:
            raise RuntimeError("Completed arm identity differs")
        return complete
    preserve_incomplete_bootstrap(output)
    if (output/"events.sqlite").exists():
        session = Session.restore(output)
        if not hasattr(session, "mechanism"):
            if session.step != identity["runtime_step"]:
                session.close()
                raise RuntimeError("Unrecognized incomplete bootstrap")
            session = Session.initialize_fork(session, config)
            session.checkpoint()
    else:
        session = Session.fork(source, output, config)
        session.checkpoint()
    stages = [0, config["additional_steps"]//2, config["additional_steps"]]
    stage_results = {}
    try:
        if session.step == session.origin_step:
            save_agent(session, "learner_step_00000.pt")
        if not (output/"learner_step_00000.pt").exists():
            raise RuntimeError("Missing immutable initial milestone")
        while True:
            relative = session.step-session.origin_step
            if relative <= config["additional_steps"] and (
                    relative % config["milestone_every"] == 0 or relative in stages):
                save_agent(session, f"learner_step_{relative:05d}.pt")
            for stage in stages:
                path = output/f"learner_step_{stage:05d}.pt"
                if path.exists():
                    stage_results[str(stage)] = evaluate(
                        path, output/"stage_probes"/f"stage_{stage:05d}", config,
                        config["stage_probe_steps"], config["stage_scene_seed"])
            if relative >= config["additional_steps"]:
                break
            next_boundary = min(
                [config["additional_steps"]]+[s for s in stages if s > relative]+[
                    (relative//config["milestone_every"]+1)*config["milestone_every"]])
            session.advance(min(config["checkpoint_every"], next_boundary-relative))
            session.checkpoint()
        if not (output/"training_only_summary.json").exists():
            dump(output/"training_only_summary.json", session.summary())
        session.learning_enabled = False
        session.admit_delayed = False
        endpoint = session.origin_step+config["additional_steps"]+max(config["delay_horizons"])
        while session.step < endpoint:
            session.advance(min(config["checkpoint_every"], endpoint-session.step))
            session.checkpoint()
        if session.records:
            session.finish_block()
        if session.mechanism == "delayed":
            session.flush_delayed()
        session.checkpoint()
        summary = session.summary()
        if (summary["unsettled_learning_choices"] or summary["delayed"]["pending"]
                or summary["delayed"]["ready"]):
            raise RuntimeError("Finite practice ended with uncredited learning")
        checkpoint = save_agent(session, "learner.pt")
        session.save_gif()
        dump(output/"summary.json", summary)
        final_probe = evaluate(checkpoint, output/"final_probe", config,
                               config["final_probe_steps"], config["final_scene_seed"])
        diagnostics = {}
        if config.get("final_diagnostics", True) and config["mechanism"] == "delayed":
            for condition in ("uniform", "head_hold", "delay_off"):
                diagnostics[condition] = evaluate(
                    checkpoint, output/"diagnostics"/condition, config,
                    config["final_probe_steps"], config["final_scene_seed"], condition)
        result = dict(config=config, source_runtime=identity, training=summary,
                      stages=stage_results, final_probe=final_probe, diagnostics=diagnostics)
        if not valid_arm(result):
            raise RuntimeError("Incomplete result")
        dump(output/"COMPLETE.json", result)
        print(json.dumps(dict(event="arm_complete", mechanism=config["mechanism"],
                              physical=summary["continuation"], delayed=summary["delayed"]["updates"])),
              flush=True)
        return result
    finally:
        session.close()


def run(config, output, source=DEFAULT_SOURCE):
    torch.set_num_threads(4)
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.drive.upper() != "E:":
        raise ValueError("Generated data must stay on E:")
    if (output/"COMPLETE.json").exists():
        raise RuntimeError("Completed pair must not be rerun")
    identity = source_identity(source)
    results = {}
    for mechanism in ("control", "delayed"):
        arm_config = dict(config, arm="active", mechanism=mechanism)
        results[mechanism] = run_arm(arm_config, output/mechanism, source, identity)
    if source_identity(source) != identity:
        raise RuntimeError("V23 source changed during continuation")
    dump(output/"COMPLETE.json", dict(config=config, source_runtime=identity,
                                    arms={key: str(output/key) for key in results}))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT/"practice.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")), args.output, args.source)
