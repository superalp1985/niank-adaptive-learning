"""CUDA practice with resumable worlds, directory, optimizers and RNG."""

from __future__ import annotations

import core
from core import ROOT, HORIZON, OPS, Stream, Archive, Memory, GroupTrace, Predictor, Controller
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def dump(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


class Replay:
    def __init__(self, count):
        self.items = []
        self.cursor = 0
        self.count = count

    def add(self, batch, actions, actual, contact):
        item = ({k: v.detach().clone() for k, v in batch.items()},
                actions.detach().clone(), actual.detach().clone(), contact.detach().clone())
        if len(self.items) < 32:
            self.items.append(item)
        else:
            self.items[self.cursor] = item
        self.cursor = (self.cursor + 1) % 32

    def sample(self, rows, rng):
        times = rng.integers(len(self.items), size=len(rows))
        batch = {k: torch.stack([self.items[t][0][k][r] for t, r in zip(times, rows)])
                 for k in self.items[0][0]}
        rest = [torch.stack([self.items[t][i][r] for t, r in zip(times, rows)]) for i in range(1, 4)]
        return batch, *rest


class Practice:
    def __init__(self, model, controller, output, seed, count, delay, training=True, heldout=False):
        self.model, self.controller = model, controller
        self.output, self.training, self.count = output, training, count
        self.env = Stream(count, seed, heldout, delay)
        self.memory, self.groups = Memory(count), GroupTrace(count, model.groups)
        self.archive = Archive(output / "events.sqlite")
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.policy_optimizer = torch.optim.Adam(controller.parameters(), lr=0.0005)
        self.rng = np.random.default_rng(seed + 918)
        self.replay_rng = np.random.default_rng(seed + 771)
        self.replay = Replay(count)
        self.admitted = np.full(count, 2, np.int64)
        self.boost = np.zeros(count, bool)
        self.last_loss = np.zeros(count, np.float32)
        self.ema_loss = np.zeros(count, np.float32)
        self.replay_count = np.zeros(count, np.int64)
        self.last_op = np.zeros(count, np.int64)
        self.records, self.pending, self.history, self.decisions, self.credits = [], [], [], [], []
        self.observation = None
        self.total = dict(new_experiences=0, replay_presentations=0, predictor_macs=0,
                          estimated_learning_macs=0, controller_macs=0,
                          predictor_optimizer_elements=0, controller_optimizer_elements=0,
                          matured_decisions=0, expired_record_credits=0)
        self.usage = np.zeros(model.groups, np.int64)
        self.initial_groups = model.w1.detach().cpu().clone()
        self.seconds = 0.
        self.curves = []
        self.segment = 0
        self.controller_macs = sum(m.in_features * m.out_features *
                                   (core.CANDIDATES if name.startswith("pointer") else 1)
                                   for name, m in controller.named_modules() if isinstance(m, torch.nn.Linear))
        self.model_count = sum(p.numel() for p in model.parameters())
        self.controller_count = sum(p.numel() for p in controller.parameters())

    def policy_inputs(self, sensor):
        data = list(core.inputs(self.memory, self.env.step, sensor, self.admitted,
                               self.boost, self.last_loss, self.ema_loss, self.replay_count,
                               self.last_op, self.model.groups, self.controller.mode))
        if not self.training:
            data[2][:, 2] = False
        return data

    def learn(self, batch, actions, actual, contact, admitted, boost):
        states, contacts, _ = self.model(batch, admitted, boost)
        loss, _ = core.loss_for(states, contacts, actions, actual, contact)
        self.backward(loss.mean())

    def backward(self, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite factual loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
        self.optimizer.step()
        self.total["predictor_optimizer_elements"] += self.model_count

    def account_forward(self, rows, boosted):
        macs = self.model.forward_macs(rows, boosted)
        self.total["predictor_macs"] += macs
        self.total["estimated_learning_macs"] += (3 if self.training else 1) * macs

    def advance(self, until):
        started = time.perf_counter()
        self.model.train(self.training)
        self.controller.train(self.training)
        while self.env.step < until:
            step = self.env.step
            image, sensor = self.observation if self.observation is not None else self.env.observe()
            self.observation = None
            if step % 4 == 0:
                ids = self.archive.append(step, image)
                self.memory.append(step, image, ids)
                data = self.policy_inputs(sensor)
                with torch.set_grad_enabled(self.training):
                    op, pointer, logp, value, entropy = self.controller.act(*data, greedy=not self.training)
                self.total["controller_macs"] += self.count * self.controller_macs
                ops, pointers = op.detach().cpu().numpy(), pointer.detach().cpu().numpy()
                identities = self.memory.ids[np.arange(self.count), pointers].copy()
                self.decisions.append(dict(step=step, ops=ops.copy(), pointers=pointers.copy(),
                                           identities=identities, live=self.memory.live.copy(),
                                           directory_ids=self.memory.ids.copy()))
                self.memory.apply(ops, pointers, step, self.archive)
                self.admitted += (ops == 4).astype(np.int64) - (ops == 6).astype(np.int64)
                self.boost, self.last_op = ops == 3, ops.copy()
                charge = 0.01 * (ops == 1) + 0.01 * (ops == 2) + 0.02 * (ops == 4)
                self.pending.append(dict(step=step, ops=ops.copy(), identities=identities,
                                         uses=np.zeros((self.count, self.model.groups), np.float32),
                                         ret=np.zeros(self.count, np.float32)))
                if self.training:
                    self.records.append([logp, value, entropy,
                                         torch.zeros(self.count, device="cuda")])
                rows = np.flatnonzero(ops == 2)
                if self.training and len(rows):
                    rb, ra, rs, rc = self.replay.sample(rows, self.replay_rng)
                    self.learn(rb, ra, rs, rc, torch.as_tensor(self.admitted[rows], device="cuda"),
                               torch.as_tensor(self.boost[rows], device="cuda"))
                    self.replay_count[rows] += 1
                    self.total["replay_presentations"] += len(rows)
                    self.account_forward(len(rows), int(self.boost[rows].sum()))
            batch = self.memory.tensors(step, image, sensor)
            batch["group_features"] = torch.as_tensor(
                self.groups.features(step, self.controller.mode), device="cuda")
            admitted = torch.as_tensor(self.admitted, device="cuda")
            boost = torch.as_tensor(self.boost, device="cuda")
            body, pair, context = self.env.body.copy(), self.env.active.copy(), self.env.context.copy()
            with torch.set_grad_enabled(self.training):
                states, contacts, route = self.model(batch, admitted, boost)
                actions = core.choose(states.detach(), contacts.detach(), batch["sensor"]).cpu().numpy()
                if self.training:
                    explore = self.rng.random(self.count) < 0.1
                    actions = np.where(explore, self.rng.integers(3, size=self.count), actions)
                actual, contact, cost, regret = self.env.execute(actions)
                ta = torch.as_tensor(actions, device="cuda")
                ts = torch.as_tensor(actual, device="cuda")
                tc = torch.as_tensor(contact, device="cuda")
                loss, mse = core.loss_for(states, contacts, ta, ts, tc)
            self.last_loss = loss.detach().cpu().numpy()
            self.ema_loss = 0.95 * self.ema_loss + 0.05 * self.last_loss
            mse_np = mse.detach().cpu().numpy()
            prediction = states.detach()[torch.arange(self.count, device="cuda"), ta].cpu().numpy()
            reward = -cost - 0.5 * mse_np - 0.002 * (1 + self.boost.astype(int)) - 0.0002 * self.admitted
            if step % 4 == 0:
                reward -= charge
            if self.training:
                self.records[-1][3] = self.records[-1][3] + 0.97**(step % 4) * torch.as_tensor(reward, device="cuda")
                self.backward(loss.mean())
                self.replay.add(batch, ta, ts, tc)
            self.total["new_experiences"] += self.count
            self.memory.feedback(self.last_loss, (self.last_op == 2) & (step % 4 == 0))
            selected = route.detach().cpu().numpy()
            for slot in range(2):
                rows = np.arange(self.count) if slot == 0 else np.flatnonzero(self.boost)
                np.add.at(self.pending[-1]["uses"], (rows, selected[rows, slot]), 1)
                np.add.at(self.usage, selected[rows, slot], 1)
            for record in self.pending:
                record["ret"] += 0.97**(step - record["step"]) * reward
            if self.pending and step - self.pending[0]["step"] + 1 == HORIZON:
                finished = self.pending.pop(0)
                self.memory.credit(finished["ops"], finished["identities"], finished["ret"])
                self.groups.credit(finished["uses"], finished["ret"], step)
                self.total["matured_decisions"] += self.count
                self.credits.append(dict(step=finished["step"], completed_at=step,
                                         returns=finished["ret"].copy(),
                                         identities=finished["identities"], ops=finished["ops"],
                                         uses=finished["uses"]))
            self.account_forward(self.count, int(self.boost.sum()))
            self.history.append(dict(step=step, body=body, pair=pair, context=context,
                                     action=actions.astype(np.int8), actual=actual, predicted=prediction,
                                     contact=contact, cost=cost, regret=regret, mse=mse_np,
                                     admitted=self.admitted.copy(), boost=self.boost.copy(),
                                     route=selected.astype(np.int16)))
            if self.training and self.env.step % 32 == 0:
                self.observation = self.env.observe()
                with torch.no_grad():
                    _, bootstrap, _ = self.controller.distributions(*self.policy_inputs(self.observation[1])[:3])
                core.old.update_controller(self.policy_optimizer, self.records, bootstrap, "delayed")
                self.total["controller_macs"] += self.count * self.controller_macs
                self.total["controller_optimizer_elements"] += self.controller_count
                self.records = []
            if self.env.step % 256 == 0:
                recent = self.history[-256:]
                point = dict(step=self.env.step,
                             success=float(np.mean([v["regret"] <= core.TOLERANCE for v in recent])),
                             mse=float(np.mean([v["mse"] for v in recent])),
                             reads=self.archive.reads, admitted=float(self.admitted.mean()))
                self.curves.append(point)
                dump(self.output / "progress.json", point)
        self.seconds += time.perf_counter() - started

    def summarize(self):
        arrays = {key: np.asarray([v[key] for v in self.history]) for key in self.history[0]}
        decision = {key: np.asarray([v[key] for v in self.decisions]) for key in self.decisions[0]}
        critical = ((arrays["step"] % 48 >= self.env.delay)
                    & (arrays["step"] % 48 < self.env.delay + 4) & (arrays["step"] >= 48))
        result = dict(
            mode=self.controller.mode, training=self.training, delay=self.env.delay,
            step=self.env.step, worlds=self.count, segment=self.segment,
            success=float((arrays["regret"] <= core.TOLERANCE).mean()),
            critical_success=float((arrays["regret"][critical] <= core.TOLERANCE).mean()) if critical.any() else None,
            cost=float(arrays["cost"].mean()), regret=float(arrays["regret"].mean()),
            mse=float(arrays["mse"].mean()),
            critical_mse=float(arrays["mse"][critical].mean()) if critical.any() else None,
            operation_counts={name: int((decision["ops"] == i).sum()) for i, name in enumerate(OPS)},
            retrievals=self.archive.reads, returns=self.memory.returns,
            expired_record_credits=self.memory.expired_credit,
            unresolved_credit_windows=len(self.pending) * self.count,
            total_parameters=self.model_count + self.controller_count,
            predictor_parameters=self.model_count, controller_parameters=self.controller_count,
            used_groups=int((self.usage > 0).sum()), group_execution_counts=self.usage.tolist(),
            changed_input_groups=int((self.model.w1.detach().cpu() != self.initial_groups).flatten(1).any(1).sum()),
            mean_active_predictor_parameters=float(self.model.active_parameters(arrays["boost"]).mean()),
            estimated_total_macs=self.total["estimated_learning_macs"] +
            (3 if self.training else 1) * self.total["controller_macs"],
            totals=copy.deepcopy(self.total), curves=self.curves, seconds=self.seconds,
        )
        # Payloads are separate compressed segments; no duplicated complete prefixes.
        np.savez_compressed(self.output / f"transitions_{self.segment:02d}.npz", **arrays)
        np.savez_compressed(self.output / f"decisions_{self.segment:02d}.npz", **decision)
        if self.credits:
            np.savez_compressed(self.output / f"credits_{self.segment:02d}.npz",
                                **{k: np.asarray([v[k] for v in self.credits]) for k in self.credits[0]})
        dump(self.output / f"result_{self.env.step}.json", result)
        self.history, self.decisions, self.credits = [], [], []
        self.segment += 1
        return result

    def save(self, path):
        if self.records:
            raise RuntimeError("Save only after a controller update boundary")
        self.archive.db.commit()
        runtime = {k: v for k, v in self.__dict__.items()
                   if k not in ("model", "controller", "archive", "optimizer", "policy_optimizer", "output")}
        torch.save(dict(model=self.model.state_dict(), controller=self.controller.state_dict(),
                        optimizer=self.optimizer.state_dict(), policy_optimizer=self.policy_optimizer.state_dict(),
                        runtime=runtime, cpu_rng=torch.get_rng_state(), gpu_rng=torch.cuda.get_rng_state(),
                        archive_reads=self.archive.reads), path)

    def load(self, path):
        state = torch.load(path, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.controller.load_state_dict(state["controller"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        self.__dict__.update(state["runtime"])
        self.archive.reads = state["archive_reads"]
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["gpu_rng"])

    def close(self):
        self.archive.close()


def probe(model, controller, output, seed, delay):
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    results = {}
    try:
        for name, held in (("seen", False), ("composition", True)):
            target = output / name
            target.mkdir(parents=True)
            session = Practice(model, controller, target, seed + (900000 if held else 800000),
                               64, delay, training=False, heldout=held)
            session.advance(240)
            results[name] = session.summarize()
            session.close()
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
    source = {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [ROOT / "core.py", ROOT / "run.py", ROOT / "PROTOCOL.md",
                        ROOT.parent / "allocation_v10" / "core.py",
                        ROOT.parent / "retention_v5" / "experiment.py"]}
    dump(output / "manifest.json", dict(config=config, sha256=source, gpu=torch.cuda.get_device_name(),
                                       torch=torch.__version__))
    all_results = []
    for delay in config["delays"]:
        for mode in config["modes"]:
            target = output / f"delay{delay}_{mode}"
            target.mkdir()
            torch.manual_seed(config["model_seed"])
            torch.cuda.manual_seed_all(config["model_seed"])
            model, controller = Predictor(config["groups"]).cuda(), Controller(mode).cuda()
            session = Practice(model, controller, target, config["world_seed"], config["worlds"], delay)
            for milestone in config["milestones"]:
                session.advance(milestone)
                result = session.summarize()
                session.save(target / "resume.pt")
                result["probes"] = probe(model, controller, target / f"probe_{milestone}", config["world_seed"], delay)
                dump(target / f"result_{milestone}.json", result)
                all_results.append(result)
                dump(output / "results.json", all_results)
                emit("stage_complete", mode=mode, delay=delay, step=milestone,
                     composition=result["probes"]["composition"]["success"],
                     critical=result["probes"]["composition"]["critical_success"])
            session.close()
            del session, model, controller
            torch.cuda.empty_cache()
    dump(output / "COMPLETE.json", dict(runs=len(config["delays"]) * len(config["modes"]),
                                        stages=len(all_results)))


if __name__ == "__main__":
    main(sys.argv[1])
