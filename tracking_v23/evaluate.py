"""Resumable frozen probes; global training RNG is restored even on failure."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import time

import numpy as np
import torch

from tracking_v22.evaluate_stages import parameter_hash, reset_dynamic
from tracking_v23.persistence import atomic_json as dump, read_complete
from tracking_v23.session import Session


@contextmanager
def isolated_rng(seed):
    saved = (torch.get_rng_state(), torch.cuda.get_rng_state_all(),
             np.random.get_state(), random.getstate())
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        yield
    finally:
        torch.set_rng_state(saved[0])
        torch.cuda.set_rng_state_all(saved[1])
        np.random.set_state(saved[2])
        random.setstate(saved[3])


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            sha.update(part)
    return sha.hexdigest()


def scene_results(session):
    rows = session.db.execute("SELECT body,facts FROM events ORDER BY step,body")
    scenes = [dict(body=i, seed=session.config["world_seed"]+i*1009,
                   contact=0, safe_contact=0, collision=0, impact_events=0,
                   collision_seconds=0., progress=0., head_travel=0., frames=0)
              for i in range(session.bodies)]
    for body, blob in rows:
        info = json.loads(blob)
        item = scenes[body]
        for key in ("contact", "safe_contact", "collision"):
            item[key] += int(info[key])
        item["impact_events"] += info["impact_count"]
        item["collision_seconds"] += info["collision_duration"]
        item["progress"] += info["progress"]
        item["head_travel"] += info.get("head_travel", 0.)
        item["frames"] += 1
    return scenes


@torch.no_grad()
def input_dependence(agent, x):
    if not hasattr(agent.model, "attention"):
        return {}
    normal = agent.model.attention(x)
    no_visual = x.clone()
    no_visual[:, :512] = 0
    no_state = x.clone()
    no_state[:, 512:516] = 0
    no_state[:, 900:] = 0
    uniform_before = agent.model.force_uniform
    try:
        agent.model.force_uniform = False
        prediction = agent.model(x)
        agent.model.force_uniform = True
        uniform = agent.model(x)
    finally:
        agent.model.force_uniform = uniform_before
    no_detail = x.clone()
    no_detail[:, 516:900] = 0
    return dict(visual_attention_l1=float((normal-agent.model.attention(no_visual)).abs().mean()),
                state_attention_l1=float((normal-agent.model.attention(no_state)).abs().mean()),
                uniform_prediction_l1=float((prediction-uniform).abs().mean()),
                zero_detail_prediction_l1=float((agent.model(x)-agent.model(no_detail)).abs().mean()))


def valid_probe_result(result):
    return (result["learned_parameters_unchanged"] is True
            and isinstance(result["parameter_sha256"], str)
            and len(result["parameter_sha256"]) == 64
            and result["frozen"]["step"] == result["identity"]["steps"]
            and result["frozen"]["unsettled_learning_choices"] == 0
            and isinstance(result["scenes"], list)
            and len(result["scenes"]) == result["frozen"]["bodies"])


def evaluate_checkpoint(checkpoint, output, config, steps, scene_seed, diagnostic="normal"):
    checkpoint, output = Path(checkpoint), Path(output)
    if diagnostic not in ("normal", "uniform", "head_hold"):
        raise ValueError(diagnostic)
    if diagnostic != "normal" and config["arm"] != "active":
        raise ValueError("Active-vision diagnostics require the active arm")
    identity = dict(checkpoint_sha256=file_hash(checkpoint), steps=steps,
                    scene_seed=scene_seed, diagnostic=diagnostic, arm=config["arm"],
                    seed=config["probe_seed"])
    completed = output/"COMPLETE.json"
    result = read_complete(completed, valid_probe_result)
    if result is not None:
        if result["identity"] != identity:
            raise RuntimeError("Completed probe identity differs")
        return result
    with isolated_rng(config["probe_seed"]):
        source_agent = torch.load(checkpoint, weights_only=False)
        reset_dynamic(source_agent, config["bodies"])
        source_agent.trial = None
        if hasattr(source_agent.model, "force_uniform"):
            source_agent.model.force_uniform = diagnostic == "uniform"
        expected = parameter_hash(source_agent)
        probe_config = dict(config, world_seed=scene_seed, head_hold=diagnostic == "head_hold",
                            video_every=max(1, steps//24))
        if (output/"events.sqlite").exists():
            saved = json.loads((output/"identity.json").read_text(encoding="utf-8"))
            if saved != identity:
                raise RuntimeError("Interrupted probe identity differs")
            db = sqlite3.connect(output/"events.sqlite")
            try:
                has_runtime = db.execute("SELECT count(*) FROM runtime").fetchone()[0] > 0
            finally:
                db.close()
            if not has_runtime:
                # Preserve evidence from a failure before the very first checkpoint.
                output.rename(output.with_name(output.name+f".interrupted_{time.time_ns()}"))
        if (output/"events.sqlite").exists():
            session = Session.restore(output)
            if parameter_hash(session.agent) != expected:
                session.close()
                raise RuntimeError("Frozen resume parameter hash differs from source")
        else:
            output.mkdir(parents=True, exist_ok=True)
            dump(output/"identity.json", identity)
            session = Session(output, probe_config, training=False, learner=source_agent)
            try:
                session.checkpoint()
            except BaseException:
                session.close()
                raise
        del source_agent
        try:
            while session.step < steps:
                session.advance(min(config.get("probe_checkpoint_every", 128), steps-session.step))
                session.checkpoint()
            if parameter_hash(session.agent) != expected:
                raise RuntimeError("A frozen probe changed learned parameters")
            session.save_gif()
            result = dict(identity=identity, parameter_sha256=expected,
                          learned_parameters_unchanged=True, frozen=session.summary(),
                          scenes=scene_results(session),
                          input_dependence=input_dependence(session.agent, session.x),
                          limitations="Diagnostic interventions may change subsequent body choices "
                                      "and physical trajectories; scenes and horizons are matched.")
            dump(output/"summary.json", result["frozen"])
            dump(completed, result)
            return result
        finally:
            session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-directory", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--scene-seed", type=int, default=19231600)
    parser.add_argument("--diagnostic", choices=("normal", "uniform", "head_hold"), default="normal")
    args = parser.parse_args()
    from tracking_v23.run import verify_manifest
    config = verify_manifest(args.arm_directory)["config"]
    checkpoint = args.arm_directory/f"learner_step_{args.stage:05d}.pt"
    result = evaluate_checkpoint(checkpoint,
        args.arm_directory/"extra_probes"/f"{args.stage:05d}_{args.diagnostic}_{args.scene_seed}_{args.steps}",
        config, args.steps, args.scene_seed, args.diagnostic)
    print(json.dumps(dict(event="probe_complete", identity=result["identity"],
                          scenes=result["scenes"])), flush=True)


if __name__ == "__main__":
    main()
