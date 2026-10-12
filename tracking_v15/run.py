"""V15 iteration reuses immutable V14 physical and archive behavior."""

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
from tracking_v14.run import Session as Previous, dump
from tracking_v14.world import LIMIT, combinations
from tracking_v15.learner import Agent, temporal_features


class Session(Previous):
    def __init__(self, output, mode, seed, worlds=48, device="cuda",
                 training=True, heldout=False, fixed_combos=None):
        super().__init__(output, "directory", seed, worlds, device, training,
                         heldout, fixed_combos)
        self.mode = mode
        self.agent = Agent(device=device, mode=mode)
        self.before = np.zeros((worlds, 32), np.float32)
        self.reset_flags = np.ones(worlds, bool)
        self.feature_step = -1
        self.saved_features, self.saved_ids = None, None

    def reset_world(self, index):
        super().reset_world(index)
        self.reset_flags[index] = True

    def _features(self, observations):
        if self.feature_step == self.step:
            return self.saved_features.copy(), self.saved_ids.copy()
        previous_features, ids = super()._features(observations)
        result = temporal_features(observations, self.before,
                                   previous_features[:, 32:], self.reset_flags)
        self.before = observations.copy()
        self.reset_flags[:] = False
        self.feature_step = self.step
        self.saved_features, self.saved_ids = result.copy(), ids.copy()
        return result, ids

    def advance(self, steps, update=True):
        self.agent.logged_corrections, self.agent.logged_risks = [], []
        point = super().advance(steps, update)
        path = self.output / f"transitions_{self.chunk - 1:05d}.npz"
        with np.load(path) as source:
            data = {key: source[key].copy() for key in source.files}
        data["correction"] = np.asarray(self.agent.logged_corrections, np.float32)
        data["risk_pred"] = np.asarray(self.agent.logged_risks, np.float32)
        np.savez_compressed(path, **data)
        self.agent.logged_corrections, self.agent.logged_risks = [], []
        return point

    def summary(self):
        return dict(**super().summary(), risk_presentations=self.agent.risk_presentations,
                    risk_updates=self.agent.risk_presentations // 768)


def probes(session, output, seed, windows, final=False):
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    results = {}
    try:
        conditions = [(False, "seen", False), (True, "heldout", False)]
        if final and session.mode == "risk_alloc":
            conditions.append((True, "heldout_no_risk", True))
        for heldout, name, disabled in conditions:
            combos = combinations(heldout)
            trial = Session(output / name, session.mode,
                            seed + (900000 if heldout else 800000),
                            len(combos), session.device, training=False,
                            heldout=heldout, fixed_combos=combos)
            trial.agent.load(session.agent.state())
            if disabled:
                trial.agent.mode = "temporal"
            for _ in range(windows):
                trial.advance(LIMIT, update=False)
            results[name] = trial.summary()
            trial.close()
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng)
    return results


def sources():
    paths = [ROOT / name for name in ("PROTOCOL.md", "pilot.json", "learner.py",
                                     "run.py", "test_contracts.py")]
    paths += [ROOT.parent / "tracking_v14" / name for name in
              ("world.py", "memory.py", "learner.py", "run.py")]
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
    dump(output / "manifest.json", dict(config=config, source=sources(),
                                        gpu=torch.cuda.get_device_name(),
                                        torch=torch.__version__))
    results = []
    for mode in config["modes"]:
        torch.manual_seed(config["model_seed"])
        torch.cuda.manual_seed_all(config["model_seed"])
        trial = Session(output / mode, mode, config["world_seed"], config["worlds"])
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
                                      config["world_seed"], config["probe_windows"],
                                      final=milestone == config["milestones"][-1])
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
