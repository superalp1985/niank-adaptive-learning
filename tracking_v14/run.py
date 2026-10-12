"""Continuous mixed-scene training and frozen probes."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "tracking_runtime"))
from learner import Agent
from memory import Directory
from world import LIMIT, combinations, World


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class Session:
    def __init__(self, output, mode, seed, worlds=48, device="cuda",
                 training=True, heldout=False, fixed_combos=None):
        output.mkdir(parents=True, exist_ok=True)
        self.output, self.mode, self.seed = output, mode, seed
        self.worlds, self.training, self.heldout = worlds, training, heldout
        self.device = device
        self.envs = [World(seed + 1009 * i, heldout=heldout,
                          combo=None if fixed_combos is None else fixed_combos[i])
                     for i in range(worlds)]
        self.memory = Directory(output / "events.sqlite", worlds, mode)
        self.agent = Agent(48, 4, device=device)
        self.rng = np.random.default_rng(seed + 991)
        self.step = 0
        self.episodes = 0
        self.chunk = 0
        self.curves = []
        self.combo_stats = {}
        self.total_reward = 0.
        self.start = time.perf_counter()

    def reset_world(self, index):
        self.envs[index].reset()
        self.episodes += 1

    def _features(self, observations):
        ids, evidence = self.memory.retrieve(observations)
        return np.concatenate((observations, evidence), axis=1), ids

    def advance(self, steps, update=True):
        rollout = {key: [] for key in
                   ("features", "actions", "groups", "action_logp", "route_logp",
                    "value", "reward", "done", "next_obs")}
        for _ in range(steps):
            observations = np.stack([env.observe() for env in self.envs])
            features, retrieved = self._features(observations)
            action_data = self.agent.act(features, greedy=not self.training)
            actions = action_data["actions"].detach().cpu().numpy().astype(np.int64)
            groups = action_data["groups"].detach().cpu().numpy().astype(np.int64)
            raw = np.stack([env.raster() for env in self.envs])
            next_obs, rewards, dones, infos = [], [], [], []
            for index, env in enumerate(self.envs):
                after, reward, done, info = env.step(int(actions[index]))
                next_obs.append(after)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
            next_obs = np.asarray(next_obs, np.float32)
            rewards = np.asarray(rewards, np.float32)
            dones = np.asarray(dones, bool)
            self.memory.append(self.step, observations, actions, next_obs,
                               raw)
            target = next_obs - observations
            with torch.no_grad():
                predicted = self.agent.predict(torch.as_tensor(features, device=self.device),
                                               torch.as_tensor(actions, device=self.device))
                prediction_error = ((predicted.cpu().numpy() - target) ** 2).mean(1)
            gain = -prediction_error + np.asarray([info["progress"] for info in infos])
            if self.training:
                self.memory.feedback(retrieved, gain)
            for index, info in enumerate(infos):
                key = str(tuple(info["combo"]))
                record = self.combo_stats.setdefault(key, dict(steps=0, contacts=0,
                    impacts=0, collisions=0, reward=0., episodes=0))
                record["steps"] += 1
                record["contacts"] += int(info["success"])
                record["impacts"] += int(info["impact"])
                record["collisions"] += int(info["collision"])
                record["reward"] += float(rewards[index])
                record.setdefault("prediction_error", 0.)
                record["prediction_error"] += float(prediction_error[index])
            rollout["features"].append(features.astype(np.float32))
            rollout["actions"].append(actions)
            rollout["groups"].append(groups)
            rollout["action_logp"].append(action_data["action_logp"].cpu().numpy())
            rollout["route_logp"].append(action_data["route_logp"].cpu().numpy())
            rollout["value"].append(action_data["value"].cpu().numpy())
            rollout["reward"].append(rewards)
            rollout["done"].append(dones)
            rollout["next_obs"].append(next_obs)
            for key, item in (
                ("combo", np.asarray([v["combo"] for v in infos], np.int8)),
                ("success", np.asarray([v["success"] for v in infos], bool)),
                ("contact", np.asarray([v["contact"] for v in infos], bool)),
                ("collision", np.asarray([v["collision"] for v in infos], bool)),
                ("impact", np.asarray([v["impact"] for v in infos], bool)),
                ("duration", np.asarray([v["duration"] for v in infos], np.int16)),
                ("episode", np.asarray([v["episode"] for v in infos], np.int32)),
                ("pred_error", prediction_error),
                ("retrieved", retrieved),
            ):
                rollout.setdefault(key, []).append(item)
            self.total_reward += float(rewards.sum())
            self.step += 1
            for index, done in enumerate(dones):
                if done:
                    self.combo_stats[str(tuple(infos[index]["combo"]))]["episodes"] += 1
                    self.reset_world(index)
        prediction_loss = None
        if update and self.training:
            data = {key: np.asarray(value) for key, value in rollout.items()}
            last_obs = np.stack([env.observe() for env in self.envs])
            last_features, _ = self._features(last_obs)
            prediction_loss = self.agent.update(data, last_features)
        data = {key: np.asarray(value) for key, value in rollout.items()}
        np.savez_compressed(self.output / f"transitions_{self.chunk:05d}.npz", **data)
        self.chunk += 1
        self.memory.db.commit()
        episodes = int(data["done"].sum())
        point = dict(step=self.step, experiences=self.step * self.worlds,
                     completed_episodes=episodes,
                     success=float(data["success"].sum() / max(episodes, 1)),
                     collision=float(data["collision"].mean()),
                     impact=float(data["impact"].sum() / max(episodes, 1)),
                     reward=float(data["reward"].mean()),
                     preupdate_mse=float(data["pred_error"].mean()),
                     prediction_loss=prediction_loss)
        self.curves.append(point)
        dump(self.output / "progress.json", point)
        return point

    def summary(self):
        elapsed = time.perf_counter() - self.start
        return dict(mode=self.mode, heldout=self.heldout, worlds=self.worlds,
                    actions=9, steps=self.step, new_experiences=self.step * self.worlds,
                    episodes=self.episodes,
                    parameters=self.agent.parameters(), memory_reads=self.memory.reads,
                    memory_reuses=self.memory.reuses, memory_fades=self.memory.fades,
                    total_reward=self.total_reward, updates=self.agent.updates,
                    predictor_updates=self.agent.predictor_updates,
                    update_presentations=self.agent.presentations,
                    elapsed_seconds=elapsed, curves=self.curves,
                    combo_stats=self.combo_stats,
                    started_episodes=self.episodes + self.worlds,
                    interrupted_episodes=self.worlds,
                    success=sum(v["contacts"] for v in self.combo_stats.values()) /
                            max(self.episodes, 1),
                    impact=sum(v["impacts"] for v in self.combo_stats.values()) /
                           max(self.episodes, 1),
                    collision_step_rate=sum(v["collisions"] for v in self.combo_stats.values()) /
                                        max(self.step * self.worlds, 1))

    def close(self):
        self.memory.close()


def probe(session, output, seed, heldout, episodes=8):
    combos = combinations(heldout)
    probe = Session(output, session.mode, seed, worlds=len(combos), device=session.device,
                    training=False, heldout=heldout, fixed_combos=combos)
    probe.agent.load(session.agent.state())
    for _ in range(episodes):
        probe.advance(LIMIT, update=False)
    result = probe.summary()
    probe.close()
    return result


def frozen_probes(session, output, seed, episodes):
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if session.device == "cuda" else None
    results = {}
    try:
        for heldout, name in ((False, "seen"), (True, "heldout")):
            results[name] = probe(session, output / name,
                                  seed + (900000 if heldout else 800000),
                                  heldout, episodes)
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng)
    return results


def source_manifest():
    paths = [ROOT / name for name in
             ("PROTOCOL.md", "PLAN.md", "pilot.json", "world.py", "memory.py",
              "learner.py", "run.py", "test_contracts.py")]
    return {str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def main(config_path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; use the bundled E training runtime")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = ROOT / config["stage"]
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "manifest.json", dict(config=config, source=source_manifest(),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__))
    results = []
    for mode in config["modes"]:
        torch.manual_seed(config["model_seed"])
        torch.cuda.manual_seed_all(config["model_seed"])
        session = Session(output / mode, mode, config["world_seed"],
                          config["worlds"], "cuda")
        initial = frozen_probes(session, output / mode / "probe_0",
                                config["world_seed"], config["probe_episodes_per_combo"])
        dump(output / mode / "result_0.json", dict(mode=mode, milestone=0, probes=initial))
        for milestone in config["milestones"]:
            target_steps = milestone
            while session.step < target_steps:
                session.advance(min(config["rollout"], target_steps - session.step))
            result = session.summary()
            result["milestone"] = milestone
            torch.save(dict(model=session.agent.state(),
                            optimizer_experts=[o.state_dict() for o in session.agent.expert_optimizers],
                            optimizer_router=session.agent.router_optimizer.state_dict(),
                            optimizer_value=session.agent.value_optimizer.state_dict(),
                            optimizer_predictor=session.agent.predictor_optimizer.state_dict(),
                            step=session.step), output / mode / f"checkpoint_{milestone}.pt")
            result["probes"] = frozen_probes(
                session, output / mode / ("probe_" + str(milestone)),
                config["world_seed"],
                config["probe_episodes_per_combo"])
            dump(output / mode / f"result_{milestone}.json", result)
            results.append(result)
            dump(output / "results.json", results)
            print(json.dumps(dict(event="milestone", mode=mode, step=milestone,
                                   heldout_episodes=result["probes"]["heldout"]["episodes"],
                                   heldout_success=result["probes"]["heldout"]["success"],
                                   seconds=result["elapsed_seconds"])), flush=True)
        session.close()
        del session
        torch.cuda.empty_cache()
    dump(output / "COMPLETE.json", dict(configurations=len(config["modes"]),
                                        milestones=len(results),
                                        source=source_manifest()))


if __name__ == "__main__":
    main(sys.argv[1])
