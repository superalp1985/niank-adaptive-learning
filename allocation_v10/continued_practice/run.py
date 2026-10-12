"""Batched CUDA runner; write compact logs and all artifacts to E:."""

from __future__ import annotations

import core
from core import (
    ROOT, OPS, MODES, Stream, Archive, Memory, Predictor, Controller,
    controller_inputs, choose, loss_for, update_controller, TOLERANCE,
)
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch


def dump(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def counts(module):
    return sum(p.numel() for p in module.parameters())


class Replay:
    def __init__(self, batch, count, length=64):
        self.length, self.used, self.cursor = length, 0, 0
        self.data = {k: torch.zeros((length, *v.shape), device=v.device, dtype=v.dtype) for k, v in batch.items()}
        self.action = torch.zeros((length, count), device="cuda", dtype=torch.long)
        self.actual = torch.zeros((length, count, 2), device="cuda")
        self.contact = torch.zeros((length, count), device="cuda")

    def add(self, batch, action, actual, contact):
        for key, val in batch.items():
            self.data[key][self.cursor].copy_(val.detach())
        self.action[self.cursor].copy_(action)
        self.actual[self.cursor].copy_(actual)
        self.contact[self.cursor].copy_(contact)
        self.cursor = (self.cursor + 1) % self.length
        self.used = min(self.length, self.used + 1)

    def sample(self, rows, generator):
        times = torch.randint(self.used, (len(rows),), device="cuda", generator=generator)
        return ({k: v[times, rows] for k, v in self.data.items()},
                self.action[times, rows], self.actual[times, rows], self.contact[times, rows])


def train_predictor(model, optimizer, batch, admitted, boost, actions, actual, contact):
    optimizer.zero_grad(set_to_none=True)
    states, contacts, _ = model(batch, admitted, boost)
    loss, _ = loss_for(states, contacts, actions, actual, contact)
    mean = loss.mean()
    if not torch.isfinite(mean):
        raise FloatingPointError("Nonfinite replay loss")
    mean.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
    optimizer.step()


def simulate(model, controller, output, seed, steps, count, *, training, heldout=False,
             milestones=()):
    model.train(training)
    controller.train(training)
    env, memory = Stream(count, seed, heldout), Memory(count)
    archive = Archive(output / "events.sqlite")
    predictor_optimizer = torch.optim.Adam(model.parameters(), lr=0.001) if training else None
    controller_optimizer = torch.optim.Adam(controller.parameters(), lr=0.0005) if training else None
    replay = None
    sample_generator = torch.Generator(device="cuda").manual_seed(seed + 771)
    action_rng = np.random.default_rng(seed + 918)
    admitted = np.full(count, 2, np.int64)
    boost = np.zeros(count, bool)
    last_loss = np.zeros(count, np.float32)
    ema_loss = np.zeros(count, np.float32)
    replay_count = np.zeros(count, np.int64)
    last_op = np.zeros(count, np.int64)
    records, curves, checkpoints = [], [], []
    current_record = None
    totals = dict(main_presentations=0, replay_presentations=0, forward_macs=0,
                  estimated_train_macs=0, optimizer_elements=0, controller_forward_macs=0,
                  controller_optimizer_elements=0, selected_groups=0, controller_updates=0)
    arrays = {
        "body": np.zeros((steps, count, 5), np.float32),
        "actual": np.zeros((steps, count, 2), np.float32),
        "predicted": np.zeros((steps, count, 2), np.float32),
        "context": np.zeros((steps, count), np.float32),
        "pair": np.zeros((steps, count, 2), np.int8),
        "action": np.zeros((steps, count), np.int8),
        "contact": np.zeros((steps, count), np.float32),
        "cost": np.zeros((steps, count), np.float32),
        "regret": np.zeros((steps, count), np.float32),
        "mse": np.zeros((steps, count), np.float32),
        "admitted": np.zeros((steps, count), np.int16),
        "boost": np.zeros((steps, count), bool),
        "op": np.full((steps // 4, count), -1, np.int8),
        "pointer": np.zeros((steps // 4, count), np.int8),
        "pointer_id": np.zeros((steps // 4, count), np.int64),
        "memory_before": np.zeros((steps // 4, count, core.CANDIDATES), bool),
        "entropy": np.zeros(steps // 32, np.float32),
        "cumulative_estimated_macs": np.zeros(steps, np.int64),
    }
    start = time.perf_counter()
    predictor_count, controller_count = counts(model), counts(controller)
    # Dense/controller bookkeeping is deliberately reported separately.
    controller_macs_per_row = sum(
        m.in_features * m.out_features * (core.CANDIDATES if name.startswith("pointer") else 1)
        for name, m in controller.named_modules() if isinstance(m, torch.nn.Linear)
    )
    try:
        for step in range(steps):
            image, sensor = env.observe()
            if step % 4 == 0:
                ids = archive.append(step, image)
                memory.append(step, image, ids)
                inputs = controller_inputs(
                    memory, step, sensor, admitted, boost, last_loss, ema_loss,
                    replay_count, last_op, model.groups,
                )
                if not training:
                    inputs[2][:, 2] = False
                if records and step % 32 == 0:
                    with torch.no_grad():
                        _, bootstrap, _ = controller.distributions(*inputs[:3])
                    _, entropy = update_controller(controller_optimizer, records, bootstrap, controller.mode)
                    arrays["entropy"][step // 32 - 1] = entropy
                    totals["controller_updates"] += 1
                    totals["controller_optimizer_elements"] += controller_count
                    totals["controller_forward_macs"] += count * controller_macs_per_row
                    records = []
                with nullcontext() if training else torch.no_grad():
                    op, pointer, logp, value, entropy = controller.act(*inputs, greedy=not training)
                totals["controller_forward_macs"] += count * controller_macs_per_row
                ops, pointers = op.detach().cpu().numpy(), pointer.detach().cpu().numpy()
                arrays["op"][step // 4] = ops
                arrays["pointer"][step // 4] = pointers
                arrays["pointer_id"][step // 4] = memory.ids[np.arange(count), pointers]
                arrays["memory_before"][step // 4] = memory.live
                memory.apply(ops, pointers, step, archive)
                admitted += (ops == 4).astype(np.int64) - (ops == 6).astype(np.int64)
                boost = ops == 3
                last_op = ops.copy()
                charge = 0.01 * (ops == 1) + 0.01 * (ops == 2) + 0.02 * (ops == 4)
                reward = -torch.as_tensor(charge, device="cuda", dtype=torch.float32)
                if training:
                    current_record = [logp, value, entropy, reward]
                    records.append(current_record)
                selected = np.flatnonzero(ops == 2)
                if training and len(selected):
                    rows = torch.as_tensor(selected, device="cuda")
                    rb, ra, rs, rc = replay.sample(rows, sample_generator)
                    train_predictor(model, predictor_optimizer, rb,
                                    torch.as_tensor(admitted[selected], device="cuda"),
                                    torch.as_tensor(boost[selected], device="cuda"), ra, rs, rc)
                    replay_count[selected] += 1
                    totals["replay_presentations"] += len(selected)
                    macs = model.forward_macs(len(selected), boost[selected].sum())
                    totals["forward_macs"] += macs
                    totals["estimated_train_macs"] += 3 * macs
                    totals["optimizer_elements"] += predictor_count
            batch = memory.tensors(step, image, sensor)
            ta = torch.as_tensor(admitted, device="cuda")
            tb = torch.as_tensor(boost, device="cuda")
            with nullcontext() if training else torch.no_grad():
                states, contacts, route = model(batch, ta, tb)
                actions = choose(states.detach(), contacts.detach(), batch["sensor"]).cpu().numpy()
                if training:
                    explore = action_rng.random(count) < 0.1
                    random_actions = action_rng.integers(3, size=count)
                    actions = np.where(explore, random_actions, actions)
                arrays["body"][step] = env.body
                arrays["context"][step] = env.context
                arrays["pair"][step] = env.active
                actual, contact, cost, regret = env.execute(actions)
                action_t = torch.as_tensor(actions, device="cuda")
                actual_t = torch.as_tensor(actual, device="cuda")
                contact_t = torch.as_tensor(contact, device="cuda")
                factual_loss, mse = loss_for(states, contacts, action_t, actual_t, contact_t)
            last_loss = factual_loss.detach().cpu().numpy()
            mse_np = mse.detach().cpu().numpy()
            arrays["predicted"][step] = states.detach()[torch.arange(count, device="cuda"), action_t].cpu().numpy()
            ema_loss = 0.95 * ema_loss + 0.05 * last_loss
            if training:
                step_reward = -cost - 0.5 * mse_np - 0.002 * (1 + boost.astype(int)) - 0.0002 * admitted
                current_record[3] = current_record[3] + 0.97**(step % 4) * torch.as_tensor(step_reward, device="cuda")
                predictor_optimizer.zero_grad(set_to_none=True)
                loss = factual_loss.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite factual loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                predictor_optimizer.step()
                if replay is None:
                    replay = Replay(batch, count)
                replay.add(batch, action_t, actual_t, contact_t)
                totals["main_presentations"] += count
                totals["optimizer_elements"] += predictor_count
            memory.feedback(last_loss, (last_op == 2) & (step % 4 == 0))
            macs = model.forward_macs(count, boost.sum())
            totals["forward_macs"] += macs
            totals["estimated_train_macs"] += (3 if training else 1) * macs
            totals["selected_groups"] += count + int(boost.sum())
            arrays["cumulative_estimated_macs"][step] = totals["estimated_train_macs"] + (3 if training else 1) * totals["controller_forward_macs"]
            for key, value_np in (
                ("actual", actual), ("action", actions), ("contact", contact),
                ("cost", cost), ("regret", regret), ("mse", mse_np),
                ("admitted", admitted), ("boost", boost),
            ):
                arrays[key][step] = value_np
            if (step + 1) % 128 == 0 or step + 1 == steps:
                window = slice(max(0, step + 1 - 128), step + 1)
                point = dict(
                    step=step + 1, presentations=(step + 1) * count,
                    success=float((arrays["regret"][window] <= TOLERANCE).mean()),
                    regret=float(arrays["regret"][window].mean()),
                    cost=float(arrays["cost"][window].mean()),
                    mse=float(arrays["mse"][window].mean()),
                    admitted=float(admitted.mean()), reads=archive.reads,
                    replay=int(replay_count.sum()),
                    estimated_macs=totals["estimated_train_macs"] + 3 * totals["controller_forward_macs"],
                    seconds=round(time.perf_counter() - start, 2),
                )
                curves.append(point)
                dump(output / "progress.json", point)
                if training:
                    emit("progress", run=output.name, **point)
            if step + 1 in milestones and step + 1 < steps:
                # Evaluation RNG must not change the subsequent training prefix.
                cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
                stage = output / f"checkpoint_{step + 1}"
                stage.mkdir()
                torch.save({"model": model.state_dict(), "controller": controller.state_dict()}, stage / "model.pt")
                for name, held in (("seen", False), ("composition", True)):
                    probe = stage / name
                    probe.mkdir()
                    result = simulate(model, controller, probe, seed + (900000 if held else 800000),
                                      128, 64, training=False, heldout=held)
                    checkpoints.append(dict(step=step + 1, probe=name, result=result))
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(gpu_rng)
                model.train()
                controller.train()
                dump(output / "checkpoints.json", checkpoints)
        if records:
            # Bootstrap the observed continuation state, not an artificial terminal.
            image, sensor = env.observe()
            inputs = controller_inputs(memory, steps, sensor, admitted, boost,
                                       last_loss, ema_loss, replay_count, last_op, model.groups)
            with torch.no_grad():
                _, bootstrap, _ = controller.distributions(*inputs[:3])
            _, entropy = update_controller(controller_optimizer, records, bootstrap, controller.mode)
            arrays["entropy"][-1] = entropy
            totals["controller_updates"] += 1
            totals["controller_optimizer_elements"] += controller_count
            totals["controller_forward_macs"] += count * controller_macs_per_row
        result = dict(
            mode=controller.mode, training=training, heldout=heldout,
            environment_seed=seed, steps=steps, worlds=count,
            success=float((arrays["regret"] <= TOLERANCE).mean()),
            last_success=float((arrays["regret"][-min(256, steps):] <= TOLERANCE).mean()),
            cost=float(arrays["cost"].mean()), regret=float(arrays["regret"].mean()),
            mse=float(arrays["mse"].mean()), last_mse=float(arrays["mse"][-min(256, steps):].mean()),
            total_parameters=predictor_count + controller_count,
            predictor_parameters=predictor_count, controller_parameters=controller_count,
            mean_active_predictor_parameters=float(model.active_parameters(arrays["boost"]).mean()),
            group_reserve=model.groups, final_accessible_groups=float(admitted.mean()),
            operation_counts={name: int((arrays["op"] == i).sum()) for i, name in enumerate(OPS)},
            retrievals=archive.reads, faded_references=memory.fade_count,
            restored_references=memory.returns,
            unresolved_references=memory.censored + int((memory.faded >= 0).sum()),
            seconds=time.perf_counter() - start, totals=totals, curves=curves,
        )
        result["estimated_total_macs"] = totals["estimated_train_macs"] + (3 if training else 1) * totals["controller_forward_macs"]
        np.savez_compressed(output / "transitions.npz", **arrays)
        dump(output / "result.json", result)
        if training:
            torch.save({"model": model.state_dict(), "controller": controller.state_dict(),
                        "groups": model.groups, "mode": controller.mode}, output / "model.pt")
        return result
    finally:
        archive.close()


def main(config_path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback disabled")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = ROOT / config["stage"]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    hashes = {}
    for name in ("core.py", "run.py", "test_contracts.py", "PROTOCOL.md"):
        source = ROOT / name
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copyfile(source, output / name)
    dependency = ROOT.parent / "retention_v5" / "experiment.py"
    hashes["physics_dependency"] = hashlib.sha256(dependency.read_bytes()).hexdigest()
    shutil.copyfile(dependency, output / "physics_dependency.py")
    manifest = dict(config=config, sha256=hashes, gpu=torch.cuda.get_device_name(),
                    torch=torch.__version__, device="cuda")
    dump(output / "manifest.json", manifest)
    emit("started", **manifest)
    results = []
    for item in config["runs"]:
        mode, seed, world_seed, groups = item["mode"], item["seed"], item["world_seed"], item["groups"]
        target = output / f"g{groups}_s{seed}_w{world_seed}_{mode}"
        target.mkdir()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = Predictor(groups).cuda()
        controller = Controller(mode).cuda()
        result = simulate(model, controller, target, world_seed, config["steps"], config["worlds"],
                          training=True, milestones=config.get("milestones", ()))
        result["model_seed"] = seed
        for name, held in (("seen", False), ("composition", True)):
            probe = target / name
            probe.mkdir()
            result[name] = simulate(model, controller, probe, world_seed + (900000 if held else 800000),
                                    128, 64, training=False, heldout=held)
        dump(target / "result.json", result)
        results.append(result)
        dump(output / "results.json", results)
        emit("result", run=target.name, success=result["success"],
             last_success=result["last_success"],
             seen=result["seen"]["success"], composition=result["composition"]["success"],
             seconds=round(result["seconds"], 1), parameters=result["total_parameters"])
        del model, controller
        torch.cuda.empty_cache()
    dump(output / "COMPLETE.json", {"runs": len(results)})
    emit("complete", runs=len(results))


if __name__ == "__main__":
    main(sys.argv[1])
