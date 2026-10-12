"""Continuous CUDA practice; no shared optimization path across modules."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

import core
from core import ROOT


def dump(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


class FactualReplay:
    def __init__(self):
        self.items = []
        self.cursor = 0

    def add(self, features, reward):
        item = (features.detach().clone(), reward.detach().clone())
        if len(self.items) < 256:
            self.items.append(item)
        else:
            self.items[self.cursor] = item
        self.cursor = (self.cursor + 1) % 256

    def sample(self, rng, count=128):
        times = rng.integers(len(self.items), size=count)
        rows = rng.integers(len(self.items[0][0]), size=count)
        return tuple(torch.stack([self.items[t][i][r] for t, r in zip(times, rows)])
                     for i in range(2))


class Practice:
    def __init__(self, output, mode, seed, count, groups, training=True,
                 heldout=False, reads_enabled=True):
        output.mkdir(parents=True, exist_ok=True)
        self.output, self.mode, self.count = output, mode, count
        self.training, self.reads_enabled = training, reads_enabled
        self.env, self.memory = core.Stream(count, seed, heldout), core.Memory(count)
        self.archive, self.plans = core.Archive(output / "events.sqlite"), core.Plans()
        self.model = core.LocalPredictor(groups).cuda()
        self.planner, self.selector = core.Value(0).cuda(), core.Value(groups).cuda()
        self.optimizers = [torch.optim.Adam(expert.parameters(), lr=0.001)
                           for expert in self.model.experts]
        self.plan_optimizer = torch.optim.Adam(self.planner.parameters(), lr=0.001)
        self.route_optimizer = torch.optim.Adam(self.selector.parameters(), lr=0.001)
        self.rng, self.replay_rng = np.random.default_rng(seed + 918), np.random.default_rng(seed + 771)
        self.replay = FactualReplay()
        self.last_loss, self.ema_loss = np.zeros(count, np.float32), np.zeros(count, np.float32)
        self.total = dict(new_experiences=0, planner_replay_presentations=0,
                          predictor_macs=0, planning_macs=0, routing_macs=0,
                          local_update_macs=0, local_optimizer_elements=0)
        self.usage = np.zeros(groups, np.int64)
        self.history, self.decisions, self.outcomes, self.curves = [], [], [], []
        self.segment, self.seconds = 0, 0.
        self.planner_macs, self.selector_macs = core.macs(self.planner), core.macs(self.selector)
        self.expert_macs = core.macs(self.model.experts[0])
        self.expert_parameters = sum(p.numel() for p in self.model.experts[0].parameters())
        self.planner_parameters = sum(p.numel() for p in self.planner.parameters())
        self.selector_parameters = sum(p.numel() for p in self.selector.parameters())
        self.all_parameters = sum(p.numel() for p in self.model.parameters()) + self.planner_parameters + self.selector_parameters

    def advance(self, until):
        started = time.perf_counter()
        if self.env.step % 4 or until % 4:
            raise ValueError("Advance and save only at complete operation windows")
        self.model.train(self.training)
        rows_np, rows_t = np.arange(self.count), torch.arange(self.count, device="cuda")
        eye = torch.eye(self.model.groups, device="cuda")
        while self.env.step < until:
            step = self.env.step
            image, sensor = self.env.observe()
            if step % 4 == 0:
                self.memory.append(step, image, self.archive.append(step, image))
                state = core.query(self.memory, sensor, self.last_loss, self.ema_loss, step)
                features, valid = self.plans.features(self.memory, state, step, self.mode,
                                                      reads_enabled=self.reads_enabled)
                with torch.no_grad():
                    scores = self.planner(features)
                    choice = core.select(scores, valid, self.rng, 0.20 if self.training else 0)
                choice_t = torch.as_tensor(choice, device="cuda")
                selected_features = features[rows_t, choice_t]
                before_live, directory_ids = self.memory.live.copy(), self.memory.ids.copy()
                read_counts = self.memory.apply(choice, self.plans, step, self.archive)
                accumulated = np.zeros(self.count, np.float32)
                self.total["planning_macs"] += self.count * len(self.plans.items) * self.planner_macs
                self.decisions.append(dict(step=step, choices=choice, reads=read_counts,
                                           live=before_live, ids=directory_ids,
                                           plan_features=selected_features.cpu().numpy(),
                                           estimated_score=scores[rows_t, choice_t].cpu().numpy()))
            state = torch.as_tensor(core.query(self.memory, sensor, self.last_loss, self.ema_loss, step),
                                    device="cuda")
            route_features = torch.cat((state[:, None].expand(-1, self.model.groups, -1),
                                        eye[None].expand(self.count, -1, -1)), -1)
            with torch.no_grad():
                route_scores = self.selector(route_features)
                group = core.select(route_scores, torch.ones_like(route_scores, dtype=torch.bool),
                                    self.rng, 0.15 if self.training else 0)
            group_t = torch.as_tensor(group, device="cuda")
            numerical = self.memory.numerical(step, image, sensor)
            body, pair, context = self.env.body.copy(), self.env.active.copy(), self.env.context.copy()
            cue_slots = [((step // 48) * 48 // 4 + i) % core.CANDIDATES for i in range(2)]
            both_cues = self.memory.live[:, cue_slots].all(1)
            with torch.set_grad_enabled(self.training):
                states, contacts = self.model(numerical, group_t)
                actions = core.choose(states.detach(), contacts.detach(),
                                      torch.as_tensor(sensor, device="cuda")).cpu().numpy()
                if self.training:
                    exploring = self.rng.random(self.count) < 0.10
                    actions = np.where(exploring, self.rng.integers(3, size=self.count), actions)
                actual, contact, cost, regret = self.env.execute(actions)
                action_t = torch.as_tensor(actions, device="cuda")
                loss, mse = core.loss_for(states, contacts, action_t,
                                          torch.as_tensor(actual, device="cuda"),
                                          torch.as_tensor(contact, device="cuda"))
            if not torch.isfinite(loss).all():
                raise FloatingPointError("Nonfinite factual prediction loss")
            prediction = states.detach()[rows_t, action_t].cpu().numpy()
            mse_np, loss_np = mse.detach().cpu().numpy(), loss.detach().cpu().numpy()
            accumulated += (-cost - 0.5 * mse_np) / 4
            self.last_loss = loss_np
            self.ema_loss = 0.95 * self.ema_loss + 0.05 * loss_np
            np.add.at(self.usage, group, 1)
            self.total["new_experiences"] += self.count
            self.total["predictor_macs"] += self.count * self.expert_macs
            self.total["routing_macs"] += self.count * self.model.groups * self.selector_macs
            if self.training:
                core.update_groups(self.model, self.optimizers, loss, group_t)
                core.update_value(self.selector, self.route_optimizer,
                                  route_features[rows_t, group_t], -torch.as_tensor(loss_np, device="cuda"))
                self.total["local_update_macs"] += 2 * self.count * self.expert_macs + 3 * self.count * self.selector_macs
                self.total["local_optimizer_elements"] += len(np.unique(group)) * self.expert_parameters + self.selector_parameters
            self.history.append(dict(step=step, body=body, pair=pair, context=context,
                                     action=actions.astype(np.int8), actual=actual,
                                     predicted=prediction, contact=contact, cost=cost,
                                     regret=regret, mse=mse_np, group=group.astype(np.int8),
                                     both_cues=both_cues))
            if self.env.step % 4 == 0:
                reward = accumulated - 0.002 * read_counts
                self.outcomes.append(dict(step=step - 3, completed_at=step, reward=reward.copy()))
                if self.training:
                    self.replay.add(selected_features, torch.as_tensor(reward, device="cuda"))
                    replay_features, targets = self.replay.sample(self.replay_rng)
                    core.update_value(self.planner, self.plan_optimizer, replay_features, targets)
                    self.total["planner_replay_presentations"] += len(targets)
                    self.total["local_update_macs"] += 3 * len(targets) * self.planner_macs
                    self.total["local_optimizer_elements"] += self.planner_parameters
            if self.env.step % 256 == 0:
                recent = self.history[-256:]
                point = dict(step=self.env.step, success=float(np.mean([v["regret"] <= core.TOLERANCE for v in recent])),
                             mse=float(np.mean([v["mse"] for v in recent])), reads=self.archive.reads)
                self.curves.append(point)
                dump(self.output / "progress.json", point)
        self.seconds += time.perf_counter() - started

    def summarize(self):
        arrays = {k: np.asarray([v[k] for v in self.history]) for k in self.history[0]}
        decisions = {k: np.asarray([v[k] for v in self.decisions]) for k in self.decisions[0]}
        outcomes = {k: np.asarray([v[k] for v in self.outcomes]) for k in self.outcomes[0]}
        critical = ((arrays["step"] % 48 >= 20) & (arrays["step"] % 48 < 24)
                    & (arrays["step"] >= 48))
        kinds = self.plans.array[decisions["choices"], 0]
        result = dict(
            mode=self.mode, step=self.env.step, worlds=self.count, training=self.training,
            reads_enabled=self.reads_enabled, parameters=self.all_parameters,
            active_predictor_parameters=self.expert_parameters,
            success=float((arrays["regret"] <= core.TOLERANCE).mean()),
            critical_success=float((arrays["regret"][critical] <= core.TOLERANCE).mean()),
            mse=float(arrays["mse"].mean()), critical_mse=float(arrays["mse"][critical].mean()),
            cost=float(arrays["cost"].mean()), regret=float(arrays["regret"].mean()),
            retrievals=self.archive.reads,
            both_cues_at_change=float(arrays["both_cues"][arrays["step"] % 48 == 20].mean()),
            operation_counts={name: int((kinds == i).sum()) for i, name in enumerate(("keep", "retrieve", "fade"))},
            paired_retrievals=int(((kinds == 1) & (self.plans.array[decisions["choices"], 2] >= 0)).sum()),
            group_execution_counts=self.usage.tolist(), totals=copy.deepcopy(self.total),
            estimated_total_macs=sum(self.total[k] for k in ("predictor_macs", "planning_macs", "routing_macs", "local_update_macs")),
            curves=self.curves, seconds=self.seconds)
        for name, payload in (("transitions", arrays), ("decisions", decisions), ("outcomes", outcomes)):
            np.savez_compressed(self.output / f"{name}_{self.segment:02d}.npz", **payload)
        dump(self.output / f"result_{self.env.step}.json", result)
        self.history, self.decisions, self.outcomes = [], [], []
        self.segment += 1
        return result

    def save(self):
        if self.env.step % 4:
            raise RuntimeError("Save only complete operation windows")
        self.archive.db.commit()
        excluded = ("output", "model", "planner", "selector", "optimizers", "replay",
                    "plan_optimizer", "route_optimizer", "archive")
        runtime = {k: v for k, v in self.__dict__.items() if k not in excluded}
        torch.save(dict(
            model=self.model.state_dict(), planner=self.planner.state_dict(), selector=self.selector.state_dict(),
            optimizers=[opt.state_dict() for opt in self.optimizers],
            plan_optimizer=self.plan_optimizer.state_dict(), route_optimizer=self.route_optimizer.state_dict(),
            replay=dict(items=self.replay.items, cursor=self.replay.cursor),
            runtime=runtime, cpu_rng=torch.get_rng_state(), gpu_rng=torch.cuda.get_rng_state(),
            archive_reads=self.archive.reads), self.output / "resume.pt")

    def load(self, path):
        saved = torch.load(path, weights_only=False)
        for name in ("model", "planner", "selector"):
            getattr(self, name).load_state_dict(saved[name])
        for optimizer, state in zip(self.optimizers, saved["optimizers"]):
            optimizer.load_state_dict(state)
        self.plan_optimizer.load_state_dict(saved["plan_optimizer"])
        self.route_optimizer.load_state_dict(saved["route_optimizer"])
        self.__dict__.update(saved["runtime"])
        self.replay.items, self.replay.cursor = saved["replay"]["items"], saved["replay"]["cursor"]
        self.archive.reads = saved["archive_reads"]
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["gpu_rng"])
        expected = self.env.step // 4 * self.count
        found = self.archive.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        if found != expected:
            raise RuntimeError("Resume requires the matching complete external database")

    def close(self):
        self.archive.close()


def probe(session, output, seed, steps, final=False):
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    results = {}
    conditions = [("seen", False, True), ("composition", True, True)]
    if final:
        conditions.append(("composition_no_reads", True, False))
    try:
        for name, heldout, reads_enabled in conditions:
            probe_seed = seed + (900000 if heldout else 800000)
            test = Practice(output / name, session.mode, probe_seed, 64, session.model.groups,
                            training=False, heldout=heldout, reads_enabled=reads_enabled)
            for module in ("model", "planner", "selector"):
                getattr(test, module).load_state_dict(getattr(session, module).state_dict())
            test.advance(steps)
            results[name] = test.summarize()
            test.close()
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng)
    return results


def main(config_path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = ROOT / config["stage"]
    output.mkdir()
    files = [ROOT / "core.py", ROOT / "run.py", ROOT / "PROTOCOL.md", ROOT / "pilot.json",
             ROOT.parent / "allocation_v11" / "core.py",
             ROOT.parent / "allocation_v10" / "core.py",
             ROOT.parent / "retention_v5" / "experiment.py"]
    source = {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    dump(output / "manifest.json", dict(config=config, sha256=source,
                                       gpu=torch.cuda.get_device_name(), torch=torch.__version__))
    results = []
    for mode in config["modes"]:
        torch.manual_seed(config["model_seed"])
        torch.cuda.manual_seed_all(config["model_seed"])
        session = Practice(output / mode, mode, config["world_seed"], config["worlds"], config["groups"])
        for milestone in config["milestones"]:
            session.advance(milestone)
            result = session.summarize()
            session.save()
            result["probes"] = probe(session, output / mode / f"probe_{milestone}",
                                     config["world_seed"], config["probe_steps"],
                                     milestone == config["milestones"][-1])
            dump(output / mode / f"result_{milestone}.json", result)
            results.append(result)
            dump(output / "results.json", results)
            emit("stage_complete", mode=mode, step=milestone,
                 composition=result["probes"]["composition"]["success"],
                 critical=result["probes"]["composition"]["critical_success"],
                 reads=result["probes"]["composition"]["retrievals"])
        session.close()
        del session
        torch.cuda.empty_cache()
    dump(output / "COMPLETE.json", dict(configurations=len(config["modes"]), stages=len(results)))


if __name__ == "__main__":
    main(sys.argv[1])
