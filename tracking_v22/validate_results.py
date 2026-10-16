"""Read-only checks against completed practice and frozen-observation data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3

import torch

from tracking_v22.evaluate_stages import parameter_hash
from tracking_v22.run import dump


def read_runtime(path):
    db = sqlite3.connect((path/"events.sqlite").resolve().as_uri()+"?mode=ro", uri=True)
    try:
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError(f"SQLite integrity failed: {path}")
        step, digest, blob = db.execute(
            "SELECT step,sha256,payload FROM runtime WHERE id=1").fetchone()
        if hashlib.sha256(blob).hexdigest() != digest:
            raise RuntimeError(f"Runtime digest differs: {path}")
        state = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)["state"]
        events = db.execute("SELECT count(*) FROM events").fetchone()[0]
        if state["step"] != step or events != step*state["bodies"]:
            raise RuntimeError(f"Incomplete event coverage: {path}")
        return state
    finally:
        db.close()


def validate(path):
    path = Path(path)
    complete = json.loads((path/"COMPLETE.json").read_text(encoding="utf-8"))
    manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    for relative, expected in manifest["sources"].items():
        if hashlib.sha256((path/"source"/relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Source snapshot changed: "+relative)
    training = read_runtime(path)
    agent = training["agent"]
    if agent.trial is not None or training["learning_enabled"]:
        raise RuntimeError("Training did not settle its last choice")
    if agent.accepted_updates != agent.updates:
        raise RuntimeError("Not all attempted local updates were accounted for")
    if training["step"] != complete["training"]["step"]:
        raise RuntimeError("Training summary differs from its runtime")
    saved_agent = torch.load(path/"learner.pt", map_location="cpu", weights_only=False)
    final_hash = parameter_hash(agent)
    if parameter_hash(saved_agent) != final_hash:
        raise RuntimeError("Final learner differs from complete runtime")
    frozen = read_runtime(path/"frozen_probe")
    if frozen["training"] or frozen["step"] != config["probe_steps"]:
        raise RuntimeError("Final frozen observation has wrong mode or duration")
    if parameter_hash(frozen["agent"]) != final_hash or any(frozen["choices"]):
        raise RuntimeError("Final frozen observation changed learned parameters")
    stages = []
    for stage in (0, 2048, 4096, 6144, 8192):
        stage_path = path/"stage_probes"/f"stage_{stage:05d}"
        item = json.loads((stage_path/"COMPLETE.json").read_text(encoding="utf-8"))
        source = path/f"learner_step_{stage:05d}.pt"
        original = torch.load(source, map_location="cpu", weights_only=False)
        observed = read_runtime(stage_path)
        if hashlib.sha256(source.read_bytes()).hexdigest() != item["source_sha256"]:
            raise RuntimeError(f"Stage source changed: {stage}")
        if observed["training"] or observed["step"] != 512 or item["observation_steps"] != 512:
            raise RuntimeError(f"Stage observation has wrong mode or duration: {stage}")
        expected = parameter_hash(original)
        if expected != item["parameter_sha256"] or expected != parameter_hash(observed["agent"]):
            raise RuntimeError(f"Frozen stage changed learned parameters: {stage}")
        stages.append(dict(step=stage, parameter_sha256=expected,
                           world_seed=observed["config"]["world_seed"]))
    result = dict(complete=True, training_step=training["step"],
        local_updates=agent.updates, settled_last_trial=True,
        final_frozen_parameters_unchanged=True, stage_checks=stages,
        frozen_world_seed=frozen["config"]["world_seed"],
        final_probe_shares_stage_scenes=all(
            row["world_seed"] == frozen["config"]["world_seed"] for row in stages),
        source_snapshots_unchanged=True, runtime_and_event_counts_verified=True)
    dump(path/"completion_audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    validate(parser.parse_args().run)
