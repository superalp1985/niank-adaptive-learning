"""Longer parallel practice; no reference-model agreement as an objective."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
from tracking_v17.run import dump
from tracking_v17.world import World, combinations
from tracking_v20.run import Pending, Session as PreviousSession
from tracking_v21.directory import Directory
from tracking_v21.learner import Agent


class SettlingPending(Pending):
    def __init__(self, worlds, directory, clock):
        super().__init__(worlds)
        self.directory, self.clock = directory, clock

    def observe(self, body, info, done):
        events = super().observe(body, info, done)
        for event in events:
            self.directory.settle(body, event["time"], event["target"], self.clock())
        return events


class ShiftWorld(World):
    """Held-back inertia, target speed and attack speed combinations."""

    def __init__(self, seed, combo, profile):
        self.profile = profile
        super().__init__(seed, combo, True)

    def reset(self):
        result = super().reset()
        self.target_direction *= (.75, 1.4)[self.profile]
        return result

    def step(self, action):
        self.body.velocity *= (.8, 1.18)[self.profile]
        return super().step(action)

    def _spawn_attack(self):
        count = len(self.projectiles)
        super()._spawn_attack()
        for body, _ in self.projectiles[count:]:
            body.velocity *= (1.3, .8)[self.profile]


class Session(PreviousSession):
    def __init__(self, output, mode, seed, config, training=True, shift=None):
        super().__init__(output, "random", seed, config, training=training,
                         heldout=not training)
        self.mode = mode
        self.agent = Agent(mode, "cuda", config["guidance"])
        if training and mode == "reversible":
            self.directory = Directory(self.worlds, config["memory_seed"], self.db,
                                       config["directory_capacity"])
            self.pending = SettlingPending(self.worlds, self.directory, lambda: self.step)
            self.agent.query = lambda: self.directory.query(
                self.memory.full_features, self.proprio, self.step)
        if shift is not None:
            options = combinations(True)
            self.envs = [ShiftWorld(seed + 1009 * i, options[i], shift)
                         for i in range(self.worlds)]
        self.shift = shift

    def summary(self):
        result = super().summary()
        result.update(negative_presentations=self.agent.negative_presentations,
                      shift_profile=self.shift,
                      evaluation="frozen_weights_no_episodic_correction"
                      if not self.training else "online_learning")
        if isinstance(self.directory, Directory):
            result["memory_mechanism"] = dict(
                cold_recalls=self.directory.cold_recalls,
                settled_trials=self.directory.settled_trials,
                enabled_trials=self.directory.enabled_trials,
                benefit_updates=self.directory.benefit_updates,
                evictions=int(self.directory.evictions.sum()),
                benefit_min=float(self.directory.benefits.min()),
                benefit_max=float(self.directory.benefits.max()))
        return result


def probe(session, output, config, shift=None):
    cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
    trial = None
    try:
        trial = Session(output, session.mode, config["world_seed"] + 900000 +
                        (0 if shift is None else 100000 * (shift + 1)),
                        config, training=False, shift=shift)
        trial.agent.load(session.agent.state())
        for _ in range((config["probe_steps"] if shift is None else config["shift_steps"]) //
                       config["rollout"]):
            trial.advance(config["rollout"])
        result = trial.summary()
        dump(trial.output / "summary.json", result)
        return result
    finally:
        if trial:
            trial.close()
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(gpu)


def prepare(config):
    output = ROOT / config["stage"]
    output.mkdir(exist_ok=False)
    files = sorted(ROOT.glob("*.py")) + sorted(ROOT.glob("*.md")) + [ROOT / "config.json"]
    files += [REPO / name for name in (
        "tracking_v20/run.py", "tracking_v20/learner.py", "tracking_v20/directory.py",
        "tracking_v19/run.py", "tracking_v19/learner.py", "tracking_v17/run.py",
        "tracking_v17/learner.py", "tracking_v17/world.py", "tracking_v14/learner.py")]
    sources = {}
    for source in files:
        relative = source.relative_to(REPO)
        sources[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    warm = REPO / config["warm_start"]
    shutil.copyfile(warm, output / "warm_start.pt")
    dump(output / "manifest.json", dict(config=config, sources=sources,
        warm_start_sha256=hashlib.sha256(warm.read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        new_experiences_per_arm=config["worlds"] * config["milestones"][-1],
        warm_start_kind="shared_V20_calibration_weights_fresh_optimizers_and_worlds"))
    return output


def worker(config, arm):
    output = ROOT / config["stage"]
    torch.manual_seed(config["model_seed"])
    torch.cuda.manual_seed_all(config["model_seed"])
    session = Session(output / arm, arm, config["world_seed"], config)
    session.agent.load(torch.load(output / "warm_start.pt", map_location="cuda",
                                  weights_only=True)["model"])
    try:
        for milestone in [0, *config["milestones"]]:
            while session.step < milestone:
                point = session.advance(config["rollout"], config["epochs"])
                if session.chunk % 8 == 0:
                    print(json.dumps(dict(event="progress", arm=arm, **point)), flush=True)
            if milestone:
                torch.save(dict(model=session.agent.state(), step=session.step),
                           session.output / f"checkpoint_{milestone}.pt")
            result = session.summary()
            result["probe"] = probe(session, session.output / f"probe_{milestone}", config)
            dump(session.output / f"result_{milestone}.json", result)
            print(json.dumps(dict(event="milestone", arm=arm, step=milestone,
                safe=result["probe"]["safe_contacts"], contacts=result["probe"]["contacts"],
                collisions=result["probe"]["collision_rate"])), flush=True)
        for shift in (0, 1):
            result = probe(session, session.output / f"final_shift_{shift}", config, shift)
            dump(session.output / f"final_shift_{shift}.json", result)
        torch.save(dict(model=session.agent.state()), session.output / "model.pt")
        dump(session.output / "summary.json", session.summary())
        dump(session.output / "COMPLETE.json", dict(arm=arm, steps=session.step))
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("random", "reversible"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("Use the existing CUDA environment on E:")
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if args.smoke:
        config.update(stage="integration_smoke", worlds=4, milestones=[256],
                      probe_steps=128, shift_steps=128, replay_budget=96)
    if args.arm:
        worker(config, args.arm)
        return
    output = prepare(config)
    children, handles = [], []
    try:
        for arm in config["arms"]:
            handle = (output / f"{arm}.log").open("w", encoding="utf-8")
            handles.append(handle)
            command = [sys.executable, "-B", "-m", "tracking_v21.run", "--arm", arm]
            if args.smoke:
                command.append("--smoke")
            children.append(subprocess.Popen(command, cwd=REPO,
                stdout=handle, stderr=subprocess.STDOUT, env=os.environ.copy(),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError("Worker failed; retain artifacts and diagnose")
            time.sleep(2)
        if any(child.returncode for child in children):
            raise RuntimeError("Worker failed; retain artifacts and diagnose")
        dump(output / "COMPLETE.json", dict(complete=True, config=config))
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
