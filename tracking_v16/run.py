"""Warm-started practice with actual whole-episode safe-contact feedback."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "tracking_v14"))
sys.path.insert(0, str(ROOT.parent / "tracking_runtime"))
from tracking_v14.run import dump
from tracking_v14.world import World as PhysicalWorld, LIMIT, combinations
from tracking_v15.run import Session as Previous


class World(PhysicalWorld):
    def __init__(self, *args, safe=False, **kwargs):
        self.safe = safe
        self.logged_base, self.logged_safe = [], []
        super().__init__(*args, **kwargs)

    def reset(self):
        self.ever_collided = False
        return super().reset()

    def observe(self):
        observation = super().observe()
        observation[30] = float(self.ever_collided)
        return observation

    def step(self, action):
        _, base_reward, done, info = super().step(action)
        self.ever_collided |= bool(info["collision"])
        safe_success = bool(info["success"] and not self.ever_collided)
        reward = base_reward
        if self.safe and info["success"]:
            reward += 16 if safe_success else -8
        self.logged_base.append(base_reward)
        self.logged_safe.append(safe_success)
        info["safe_success"], info["base_reward"] = safe_success, base_reward
        return self.observe(), reward, done, info


class Session(Previous):
    def __init__(self, output, mode, seed, worlds=48, device="cuda",
                 training=True, heldout=False, fixed_combos=None):
        super().__init__(output, "temporal", seed, worlds, device, training,
                         heldout, fixed_combos)
        self.mode = mode
        self.envs = [World(seed + 1009 * i, heldout=heldout,
                          combo=None if fixed_combos is None else fixed_combos[i],
                          safe=mode == "safe_contact") for i in range(worlds)]

    def advance(self, steps, update=True):
        for env in self.envs:
            env.logged_base, env.logged_safe = [], []
        point = super().advance(steps, update)
        path = self.output / f"transitions_{self.chunk - 1:05d}.npz"
        with np.load(path) as source:
            data = {key: source[key].copy() for key in source.files}
        data["base_reward"] = np.array([env.logged_base for env in self.envs],
                                      np.float32).T
        data["safe_success"] = np.array([env.logged_safe for env in self.envs], bool).T
        data["reward_mode"] = np.array(self.mode)
        np.savez_compressed(path, **data)
        return point


def probes(session, output, seed, windows):
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    results = {}
    try:
        for heldout, name in ((False, "seen"), (True, "heldout")):
            combos = combinations(heldout)
            trial = Session(output / name, session.mode,
                            seed + (900000 if heldout else 800000),
                            len(combos), session.device, training=False,
                            heldout=heldout, fixed_combos=combos)
            trial.agent.load(session.agent.state())
            for _ in range(windows):
                trial.advance(LIMIT, update=False)
            results[name] = trial.summary()
            trial.close()
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng)
    return results


def sources():
    paths = [ROOT / name for name in ("PROTOCOL.md", "pilot.json", "run.py", "test_contracts.py")]
    paths += [ROOT.parent / "tracking_v14" / name for name in
              ("world.py", "memory.py", "learner.py", "run.py")]
    paths += [ROOT.parent / "tracking_v15" / name for name in ("learner.py", "run.py")]
    return {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = json.loads((ROOT / "pilot.json").read_text(encoding="utf-8"))
    output = ROOT / config["stage"]
    output.mkdir()
    seed_path = ROOT / "seed" / "model.pt"
    checkpoint = torch.load(seed_path, map_location="cuda", weights_only=False)
    dump(output / "manifest.json", dict(config=config, source=sources(),
        warm_start_sha256=hashlib.sha256(seed_path.read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__))
    results = []
    for mode in config["modes"]:
        torch.manual_seed(config["model_seed"])
        torch.cuda.manual_seed_all(config["model_seed"])
        trial = Session(output / mode, mode, config["world_seed"], config["worlds"])
        trial.agent.load(checkpoint["model"])
        dump(output / mode / "result_0.json", dict(mode=mode, milestone=0,
             probes=probes(trial, output / mode / "probe_0", config["world_seed"],
                           config["probe_windows"])))
        for milestone in config["milestones"]:
            while trial.step < milestone:
                trial.advance(min(config["rollout"], milestone - trial.step))
            torch.save(dict(model=trial.agent.state(), step=trial.step,
                            optimizer_experts=[o.state_dict() for o in trial.agent.expert_optimizers],
                            optimizer_router=trial.agent.router_optimizer.state_dict(),
                            optimizer_value=trial.agent.value_optimizer.state_dict(),
                            optimizer_predictor=trial.agent.predictor_optimizer.state_dict(),
                            optimizer_risk=trial.agent.risk_optimizer.state_dict()),
                       output / mode / f"checkpoint_{milestone}.pt")
            result = trial.summary()
            result["milestone"] = milestone
            result["probes"] = probes(trial, output / mode / f"probe_{milestone}",
                                      config["world_seed"], config["probe_windows"])
            dump(output / mode / f"result_{milestone}.json", result)
            results.append(result)
            dump(output / "results.json", results)
            print(json.dumps(dict(event="milestone", mode=mode, step=milestone,
                                   success=result["probes"]["heldout"]["success"],
                                   impact=result["probes"]["heldout"]["impact"],
                                   collision=result["probes"]["heldout"]["collision_step_rate"])),
                  flush=True)
        trial.close()
        del trial
        torch.cuda.empty_cache()
    dump(output / "COMPLETE.json", dict(milestones=len(results), source=sources()))


if __name__ == "__main__":
    main()
