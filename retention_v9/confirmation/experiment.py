"""Repeated closed-loop practice with revisable memory and factual feedback."""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)
for name in ("TEMP", "TMP", "TMPDIR", "TORCH_HOME", "XDG_CACHE_HOME"):
    os.environ[name] = str(CACHE)
sys.path.insert(0, str(ROOT.parent))

import argparse
from contextlib import closing
import hashlib
import json
import shutil
import sqlite3
import time

import numpy as np
import torch
from torch.nn import functional as F

from retention_v5.experiment import Episodes, TOLERANCE, archive, dump, emit, generate, physics
from retention_v6.experiment import load_observations, seed_all
from retention_v8.experiment import OutcomeRetentionModel

MODES = ("fixed", "periodic", "recurrent", "consolidated")
CAPACITY = 4
BASE_LIFETIME = 6
CONTEXTS = (0.0, 1.0, 0.0, 0.5, 1.0, 0.0, 1.0, 0.5)


def checkpoint(seed):
    return ROOT.parent / "retention_v8" / "confirmation" / f"u2048_c32_s{seed}_quartet_delta" / "model.pt"


def facts(data):
    rows = np.arange(len(data))
    material = data.images[rows, data.cue_times[:, 0], 0, 1, 1] > 0
    direction = data.images[rows, data.cue_times[:, 1], 1, 2, 2] > 0
    return np.where(material, 0.9, 0.25).astype(np.float32), np.where(direction, 0.18, -0.18).astype(np.float32)


def worlds(count, seed, scenario="practice"):
    data = generate(count, seed, scenario=scenario)
    for row, identity in enumerate(data.ids):
        rng = np.random.default_rng(int(identity) + 73000000)
        data.body[row] = (
            rng.uniform(0.15, 0.85), rng.uniform(-0.65, 0.65),
            rng.uniform(0.15, 0.85), rng.uniform(0.4, 0.8),
            -1.0 if row % 2 else 1.0,
        )
    data.context[:] = 0
    data.states, data.contacts, data.costs = physics(data.body, *facts(data), data.context[:, 0])
    return data


def observation_batch(images, body, context, device="cuda"):
    return {
        "images": torch.as_tensor(images, device=device, dtype=torch.float32),
        "body": torch.as_tensor(body, device=device, dtype=torch.float32),
        "context": torch.as_tensor(context, device=device, dtype=torch.float32),
    }


def plan(output, body):
    states = output["states"]
    actions = states.new_tensor((-1.0, 0.0, 1.0))
    cost = ((states[..., 0].clamp(0, 1) - body[:, 2:3]) ** 2
            + 0.025 * states[..., 1] ** 2 + 0.005 * actions**2
            + 0.6 * output["contacts"].sigmoid())
    return cost.argmin(-1)


def factual_loss(output, actions, actual_states, actual_contacts):
    """The learner receives only the outcome of its executed action."""
    rows = torch.arange(len(actions), device=actions.device)
    state = output["states"][rows, actions]
    contact = output["contacts"][rows, actions]
    error = (state - actual_states).square().mean(-1)
    return 4.0 * error + 0.2 * F.binary_cross_entropy_with_logits(
        contact, actual_contacts, reduction="none"
    ), error


def factual_branches(output, actions, actual_states, actual_contacts):
    count, steps = output["states"].shape[:2]
    selected = output["states"].gather(2, actions[:, None, None, None].expand(-1, steps, 1, 2)).squeeze(2)
    contacts = output["contacts"].gather(2, actions[:, None, None].expand(-1, steps, 1)).squeeze(2)
    return (4.0 * (selected - actual_states[:, None]).square().mean(-1)
            + 0.2 * F.binary_cross_entropy_with_logits(
                contacts, actual_contacts[:, None].expand(-1, steps), reduction="none"))


class Directory:
    def __init__(self, images, utility, mode):
        self.mode = mode
        self.count, self.steps = images.shape[:2]
        self.trace = np.zeros((self.count, self.steps), np.float32)
        self.strength = np.zeros_like(self.trace)
        self.last_visit = np.full(self.count, -1, np.int64)
        self.error_scale = np.full(self.count, 0.01, np.float32)
        self.indices = np.sort(np.argsort(utility, axis=1)[:, -CAPACITY:], axis=1)
        self.mask = np.ones_like(self.indices, dtype=bool)
        self.expiry = np.full_like(self.indices, BASE_LIFETIME)
        self.cached = np.zeros_like(images)
        self._cache(images)

    def _cache(self, images, rows=None):
        if rows is None:
            rows = np.arange(self.count)
        self.cached[rows] = 0
        for local, row in enumerate(rows):
            slots = self.indices[row][self.mask[row]]
            self.cached[row, slots] = images[local, slots]

    def override(self, device="cuda"):
        return (torch.as_tensor(self.indices, device=device), torch.as_tensor(self.mask, device=device))

    def active_map(self):
        result = np.zeros((self.count, self.steps), dtype=bool)
        for row in range(self.count):
            result[row, self.indices[row][self.mask[row]]] = True
        return result

    def request(self, step, observed_error):
        if self.mode == "fixed":
            flags = np.zeros(self.count, bool)
        elif self.mode == "periodic":
            flags = np.full(self.count, (step + 1) % 8 == 0, bool)
        else:
            flags = ((observed_error > np.maximum(0.005, 1.5 * self.error_scale))
                     | (step - self.last_visit >= 16) | ~self.mask.any(1))
        self.error_scale = 0.9 * self.error_scale + 0.1 * observed_error
        return flags

    def practice(self, step, slot_gain, loss):
        if self.mode == "fixed":
            return
        self.trace *= 0.985
        self.strength *= 0.985
        for row in range(self.count):
            slots = self.indices[row][self.mask[row]]
            gains = slot_gain[row][self.mask[row]]
            self.trace[row, slots] = 0.75 * self.trace[row, slots] + 0.25 * gains
            evidence = np.clip(gains / (loss[row] + 0.005), 0, 2)
            self.strength[row, slots] += evidence
            reinforced = evidence > 0.05
            extension = (np.minimum(48, 4 * self.strength[row, slots]).astype(np.int64)
                         if self.mode == "consolidated" else np.zeros(len(slots), np.int64))
            positions = np.flatnonzero(self.mask[row])
            self.expiry[row, positions[reinforced]] = step + BASE_LIFETIME + extension[reinforced]

    def recover(self, step, rows, images, gains, full_loss):
        for local, row in enumerate(rows):
            self.trace[row] = 0.5 * self.trace[row] + 0.5 * gains[local]
            evidence = np.clip(gains[local] / (full_loss[local] + 0.005), 0, 2)
            self.strength[row] += evidence
            self.indices[row] = np.sort(np.argsort(self.trace[row])[-CAPACITY:])
            self.mask[row] = True
            extension = (np.minimum(48, 4 * self.strength[row, self.indices[row]]).astype(np.int64)
                         if self.mode == "consolidated" else np.zeros(CAPACITY, np.int64))
            self.expiry[row] = step + BASE_LIFETIME + extension
            self.last_visit[row] = step
        self._cache(images, rows)

    def fade(self, step):
        if self.mode == "fixed":
            return
        self.mask &= self.expiry > step
        live = self.active_map()
        self.cached *= live[..., None, None, None]


@torch.no_grad()
def slot_contributions(model, batch, directory, actions, actual_states, actual_contacts, current_loss):
    count = len(actions)
    expanded = {key: value.repeat_interleave(CAPACITY, dim=0) for key, value in batch.items()}
    indices, masks = directory.override(str(actions.device))
    indices = indices.repeat_interleave(CAPACITY, dim=0)
    masks = masks.repeat_interleave(CAPACITY, dim=0)
    positions = torch.arange(CAPACITY, device=actions.device).repeat(count)
    masks[torch.arange(count * CAPACITY, device=actions.device), positions] = False
    output = model(expanded, "quartet_delta", active_override=(indices, masks))
    loss, _ = factual_loss(
        output, actions.repeat_interleave(CAPACITY),
        actual_states.repeat_interleave(CAPACITY, dim=0),
        actual_contacts.repeat_interleave(CAPACITY),
    )
    return loss.reshape(count, CAPACITY) - current_loss[:, None]


def load_model(seed, device="cuda"):
    model = OutcomeRetentionModel().to(device)
    model.load_state_dict(torch.load(checkpoint(seed), map_location=device, weights_only=True))
    # Planning uses predicted physics; the old action-logit head is not trained here.
    for parameter in model.action.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def frozen_probe(model, dataset):
    model.eval()
    body, momentum, force = dataset.body, *facts(dataset)
    pieces = []
    for context_value in (0.0, 0.5, 1.0):
        context = np.full((len(dataset), 1), context_value, np.float32)
        states, contacts, costs = physics(body, momentum, force, context[:, 0])
        batch = observation_batch(dataset.images, body, context)
        indices = torch.arange(dataset.images.shape[1], device="cuda")[None].expand(len(dataset), -1)
        output = model(batch, "quartet_delta", active_override=(indices, torch.ones_like(indices, dtype=torch.bool)))
        actions = plan(output, batch["body"]).cpu().numpy()
        regret = costs[np.arange(len(dataset)), actions] - costs.min(-1)
        predicted = output["states"][np.arange(len(dataset)), actions].cpu().numpy()
        pieces.append({
            "context": context_value, "success": float((regret <= TOLERANCE).mean()),
            "regret": float(regret.mean()), "state_mse": float(((predicted - states[np.arange(len(dataset)), actions]) ** 2).mean()),
            "left_success": float((regret[body[:, 4] < 0] <= TOLERANCE).mean()),
            "right_success": float((regret[body[:, 4] > 0] <= TOLERANCE).mean()),
        })
    return {"contexts": pieces, **{
        key: float(np.mean([piece[key] for piece in pieces]))
        for key in ("success", "regret", "state_mse", "left_success", "right_success")
    }}


def summarize(records):
    return {
        "success": float(records["success"].mean()),
        "regret": float(records["regret"].mean()),
        "state_mse": float(records["state_mse"].mean()),
        "actual_cost": float(records["actual_cost"].mean()),
        "retrieval_fraction": float(records["retrieval"].mean()),
        "archive_records_read": int(records["retrieval"].sum() * 18),
        "reference_drops": int(records["dropped_count"].sum()),
        "reference_returns": int(records["returned_count"].sum()),
        "fine_drop_events": int(records["fine_dropped"].sum()),
        "fine_return_events": int(records["fine_returned"].sum()),
        "mean_active_references": float(records["active_count"].mean()),
    }


def store_practice(path, ids, records):
    """Observed past facts are archived separately from initial visual histories."""
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "CREATE TABLE practice (world INTEGER, step INTEGER, body BLOB, context REAL,"
            " action INTEGER, next_state BLOB, contact REAL, prediction BLOB, queried INTEGER,"
            " active_indices BLOB, active_mask BLOB, PRIMARY KEY(world, step))"
        )
        connection.executemany(
            "INSERT INTO practice VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ((int(identity), step, records["body"][step, row].tobytes(),
              float(records["context"][step, row, 0]), int(records["action"][step, row]),
              records["actual_state"][step, row].tobytes(), float(records["actual_contact"][step, row]),
              records["prediction"][step, row].tobytes(), int(records["retrieval"][step, row]),
              records["indices"][step, row].tobytes(), records["mask"][step, row].tobytes())
             for step in range(len(records["action"])) for row, identity in enumerate(ids)),
        )
        connection.commit()


def practice_run(base, path, mode, seed, steps, output, final_probe=None):
    seed_all(seed)
    model = load_model(seed)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0003, weight_decay=0.0001)
    initial_body = base.body.copy()
    body = initial_body.copy()
    momentum, force = facts(base)
    images, init_io = load_observations(path, base.ids, base.images.shape[1])
    initial_context = np.zeros((len(base), 1), np.float32)
    with torch.no_grad():
        initial = model(observation_batch(images, body, initial_context), "quartet_delta")
    directory = Directory(images, initial["utility"].cpu().numpy(), mode)
    del images, initial
    count = len(base)
    rows = np.arange(count)
    generator = np.random.default_rng(int(base.ids[0]) + 40000000)
    goals = generator.uniform(0.1, 0.9, (steps // 8 + 1, count)).astype(np.float32)
    records = {name: [] for name in (
        "body", "context", "action", "prediction", "actual_state", "actual_contact",
        "success", "regret", "state_mse", "actual_cost", "all_action_costs",
        "retrieval", "active_count", "indices", "mask", "expiry", "strength", "trace",
        "dropped_count", "returned_count", "fine_dropped", "fine_returned",
        "fine_active", "replay_improvement", "weight_update_index",
    )}
    seen_active = directory.active_map()
    dropped_before = np.zeros_like(seen_active)
    started = time.perf_counter()
    before_probe = frozen_probe(model, final_probe) if final_probe is not None else None
    for step in range(steps):
        context = np.full((count, 1), CONTEXTS[(step // 16) % len(CONTEXTS)], np.float32)
        body[:, 2] = goals[step // 8]
        before_body = body.copy()
        old_indices, old_masks = directory.indices.copy(), directory.mask.copy()
        active_before = directory.active_map()
        batch = observation_batch(directory.cached, body, context)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output_batch = model(batch, "quartet_delta", active_override=directory.override())
        actions = plan(output_batch, batch["body"]).detach()
        action_array = actions.cpu().numpy()

        # Only the environment and scorer receive every counterfactual outcome.
        all_states, all_contacts, costs = physics(body, momentum, force, context[:, 0])
        actual_state = all_states[rows, action_array]
        actual_contact = all_contacts[rows, action_array]
        state_target = torch.from_numpy(actual_state).cuda()
        contact_target = torch.from_numpy(actual_contact).cuda()
        loss, error = factual_loss(output_batch, actions, state_target, contact_target)
        current_loss = loss.detach()
        error_array = error.detach().cpu().numpy()
        regret = costs[rows, action_array] - costs.min(-1)
        prediction = output_batch["states"][rows, actions].detach().cpu().numpy()

        # Record the prediction before feedback can change weights or references.
        immutable = {
            "body": before_body, "context": context, "action": action_array,
            "prediction": prediction, "actual_state": actual_state,
            "actual_contact": actual_contact, "success": regret <= TOLERANCE,
            "regret": regret, "state_mse": error_array, "actual_cost": costs[rows, action_array],
            "all_action_costs": costs, "indices": old_indices, "mask": old_masks,
            "active_count": old_masks.sum(1), "fine_active": active_before[rows, base.cue_times[:, 1]],
            "weight_update_index": np.full(count, step, np.int64),
        }
        for key, value in immutable.items():
            records[key].append(np.array(value, copy=True))

        gains = slot_contributions(model, batch, directory, actions, state_target, contact_target, current_loss)
        live = torch.as_tensor(directory.mask, device="cuda")
        predicted_value = output_batch["utility"].gather(1, torch.as_tensor(directory.indices, device="cuda"))
        value_loss = ((predicted_value - gains.detach()).square() * live).sum() / live.sum().clamp_min(1)
        total = loss.mean() + value_loss
        directory.practice(step, gains.cpu().numpy(), current_loss.cpu().numpy())
        requested = directory.request(step, error_array)
        selected = np.flatnonzero(requested)
        replay_improvement = np.zeros(count, np.float32)
        if len(selected):
            recovered_images, _ = load_observations(path, base.ids[selected], base.images.shape[1])
            recovered_batch = observation_batch(recovered_images, before_body[selected], context[selected])
            recovered = model(recovered_batch, "quartet_delta", replay=True)
            replay_loss, _ = factual_loss(recovered["replay"], actions[selected], state_target[selected], contact_target[selected])
            omitted_loss = factual_branches(recovered["omitted"], actions[selected], state_target[selected], contact_target[selected])
            contributions = omitted_loss - replay_loss.detach()[:, None]
            aux_value = (recovered["utility"] - contributions.detach()).square().mean()
            total = total + 0.5 * replay_loss.sum() / count + aux_value * len(selected) / count
            replay_improvement[selected] = (current_loss[selected] - replay_loss.detach()).cpu().numpy()
            directory.recover(step, selected, recovered_images, contributions.cpu().numpy(), replay_loss.detach().cpu().numpy())
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite practice update")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        directory.fade(step)
        after = directory.active_map()
        dropped = active_before & ~after
        returned = after & ~active_before & dropped_before
        dropped_before |= dropped
        seen_active |= after
        for key, value in {
            "retrieval": requested, "expiry": directory.expiry.copy(), "strength": directory.strength.copy(),
            "trace": directory.trace.copy(),
            "dropped_count": dropped.sum(1), "returned_count": returned.sum(1),
            "fine_dropped": dropped[rows, base.cue_times[:, 1]],
            "fine_returned": returned[rows, base.cue_times[:, 1]],
            "replay_improvement": replay_improvement,
        }.items():
            records[key].append(value)
        body[:, :2] = actual_state
        if (step + 1) % 32 == 0:
            recent = np.stack(records["success"][-32:])
            emit("practice", mode=mode, seed=seed, step=step + 1, total_steps=steps,
                 success=float(recent.mean()), active=float(directory.mask.sum(1).mean()),
                 seconds=round(time.perf_counter() - started, 1))
    arrays = {name: np.stack(value) for name, value in records.items()}
    np.savez_compressed(output / "transitions.npz", ids=base.ids, cue_times=base.cue_times, **arrays)
    store_practice(output / "practice.sqlite", base.ids, arrays)
    torch.save(model.state_dict(), output / "model.pt")
    result = {
        "mode": mode, "seed": seed, "steps": steps, "worlds": count,
        "actual_transitions": steps * count,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "seconds": time.perf_counter() - started, "prequential": summarize(arrays),
        "first_32": summarize({k: v[:32] for k, v in arrays.items()}),
        "last_32": summarize({k: v[-32:] for k, v in arrays.items()}),
        "initial_database_io": init_io, "initial_probe": before_probe,
        "final_probe": frozen_probe(model, final_probe) if final_probe is not None else None,
        "phases": [
            {"start": start, "context": float(arrays["context"][start, 0, 0]),
             **summarize({k: v[start:start + 16] for k, v in arrays.items()})}
            for start in range(0, steps, 16)
        ],
        "archive_bytes_read": int(arrays["retrieval"].sum() * 18 * 432 + init_io["bytes"]),
        "cached_omission_decoder_rows": int(steps * count * CAPACITY),
        "archive_omission_decoder_rows": int(arrays["retrieval"].sum() * 18),
        "factual_feedback": "chosen action only, recorded and scored before updates",
    }
    dump(output / "result.json", result)
    return result


def run(stage):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no implicit CPU fallback")
    torch.set_num_threads(2)
    output = ROOT / stage
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    for name in ("experiment.py", "PROTOCOL.md", "test_contracts.py"):
        shutil.copyfile(ROOT / name, output / name)
    dependencies = {}
    for version in ("retention_v5", "retention_v6", "retention_v7", "retention_v8"):
        path = ROOT.parent / version / "experiment.py"
        dependencies[version] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copyfile(path, output / f"dependency_{version}.py")
    dev = stage.startswith("development")
    seeds = (11,) if dev else (11, 23, 37)
    count = 32 if dev else 64
    budgets = (128,) if dev else (128, 512)
    offset = 3100000 if dev else 3300000
    manifest = {
        "stage": stage, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependencies": dependencies, "modes": MODES, "worlds": count, "budgets": budgets,
        "seeds": seeds, "checkpoints": {seed: hashlib.sha256(checkpoint(seed).read_bytes()).hexdigest() for seed in seeds},
        "base_lifetime": BASE_LIFETIME, "practice_decay": 0.985, "gain_trace_rate": 0.25,
        "retrieval_error_multiplier": 1.5, "retrieval_floor": 0.005, "curiosity_gap": 16,
        "data": {},
    }
    datasets = {}
    for ordinal, seed in enumerate(seeds):
        base = worlds(count, offset + ordinal * 10000)
        path = output / f"observations_s{seed}.sqlite"
        archive(path, base)
        probe = None if dev else worlds(256, offset + 100000 + ordinal * 10000, scenario="frozen_probe")
        if probe is not None:
            archive(output / f"frozen_probe_s{seed}.sqlite", probe)
        datasets[seed] = (base, path, probe)
        manifest["data"][seed] = {
            "world_ids": [int(base.ids.min()), int(base.ids.max())], "world_sha256": base.fingerprint(),
            "probe_ids": None if probe is None else [int(probe.ids.min()), int(probe.ids.max())],
            "probe_sha256": None if probe is None else probe.fingerprint(),
        }
    dump(output / "manifest.json", manifest)
    emit("started", stage=stage, budgets=budgets, seeds=seeds, worlds=count, source=manifest["sha256"])
    results = []
    for steps in budgets:
        for seed in seeds:
            base, path, probe = datasets[seed]
            for mode in MODES:
                target = output / f"t{steps}_s{seed}_{mode}"
                target.mkdir()
                result = practice_run(base, path, mode, seed, steps, target, probe)
                results.append(result)
                dump(output / "results.json", results)
                emit("result", mode=mode, seed=seed, steps=steps, prequential=result["prequential"])
    dump(output / "COMPLETE.json", {"runs": len(results)})
    emit("completed", runs=len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "confirmation"), required=True)
    run(parser.parse_args().stage)
