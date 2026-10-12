"""Causal active-memory retention experiment in a reduced physical task."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)
for name in ("TEMP", "TMP", "TMPDIR", "TORCH_HOME", "XDG_CACHE_HOME"):
    os.environ[name] = str(CACHE)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
import math
import random
import shutil
import sqlite3
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODES = ("full", "fifo", "surprise", "value", "anti_value", "none")
VOLUMES = ((64, 64), (512, 8), (512, 32), (2048, 8))
ACTIONS = np.asarray((-1.0, 0.0, 1.0), dtype=np.float32)
CAPACITY = 4
READS = 2
TOLERANCE = 0.02


def emit(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, **values}), flush=True)


def dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


@dataclass
class Episodes:
    ids: np.ndarray
    images: np.ndarray
    context: np.ndarray
    body: np.ndarray
    states: np.ndarray
    contacts: np.ndarray
    costs: np.ndarray
    cue_times: np.ndarray
    scenario: str

    def __len__(self) -> int:
        return len(self.ids)

    def take(self, indices: Any) -> "Episodes":
        return Episodes(
            *(value[indices] for value in (
                self.ids, self.images, self.context, self.body, self.states,
                self.contacts, self.costs, self.cue_times,
            )),
            self.scenario,
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256(self.scenario.encode())
        for value in (
            self.ids, self.images, self.context, self.body, self.states,
            self.contacts, self.costs, self.cue_times,
        ):
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()

    def tensors(self, device: str = "cuda") -> dict[str, torch.Tensor]:
        return {
            name: torch.from_numpy(getattr(self, name)).to(device)
            for name in (
                "images", "context", "body", "states", "contacts", "costs", "cue_times"
            )
        }


def physics(
    body: np.ndarray, momentum: np.ndarray, force: np.ndarray, risk: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position, velocity, goal, delta_time, wall = body.T
    next_velocity = (
        momentum[:, None] * velocity[:, None]
        + 0.35 * ACTIONS[None]
        + risk[:, None] * force[:, None]
    )
    raw_position = position[:, None] + delta_time[:, None] * next_velocity
    contacts = np.where(wall[:, None] > 0, raw_position >= 1.0, raw_position <= 0.0)
    next_position = raw_position.clip(0.0, 1.0)
    states = np.stack((next_position, next_velocity), axis=-1).astype(np.float32)
    costs = (
        (next_position - goal[:, None]) ** 2
        + 0.025 * next_velocity**2
        + 0.005 * ACTIONS[None] ** 2
        + 0.6 * contacts
    ).astype(np.float32)
    return states, contacts.astype(np.float32), costs


def generate(
    count: int, seed: int, *, delay: int = 18, scenario: str = "train"
) -> Episodes:
    """Each episode has its own seed, so training prefixes are exactly nested."""
    images = np.zeros((count, delay, 3, 6, 6), np.float32)
    body = np.zeros((count, 5), np.float32)
    context = np.zeros((count, 1), np.float32)
    cue_times = np.zeros((count, 2), np.int64)
    momenta = np.zeros(count, np.float32)
    forces = np.zeros(count, np.float32)
    ids = np.arange(seed, seed + count, dtype=np.int64)
    for index, identity in enumerate(ids):
        rng = np.random.default_rng(int(identity))
        material = int(rng.integers(0, 2))
        fine = int(rng.integers(0, 2))
        risk = float(rng.random() < 0.35)
        context[index, 0] = risk
        momenta[index] = (0.25, 0.90)[material]
        forces[index] = (-0.18, 0.18)[fine]
        wall = -1.0 if scenario == "left_layout" else 1.0
        velocity = float(rng.uniform(0.35, 1.10)) * wall
        position = float(rng.uniform(0.25, 0.80))
        goal = float(rng.uniform(0.05, 0.95))
        dt = float(rng.uniform(0.35, 0.90))
        body[index] = (position, velocity, goal, dt, wall)
        cues = rng.choice(np.arange(1, 5), size=2, replace=False)
        cue_times[index] = cues
        for step in range(delay):
            texture = rng.uniform(0.15, 1.0, size=(6, 6)).astype(np.float32)
            if scenario == "texture_shift":
                texture = rng.choice((0.0, 1.0), size=(6, 6)).astype(np.float32)
            images[index, step, 2] = texture
        # Cues have fixed visual shape, variable physical value, and random timing.
        images[index, cues[0], 0, 1:4, 1:4] = 0.4 if material else -0.4
        images[index, cues[1], 1, 2, 2:4] = 0.12 if fine else -0.12
    states, contacts, costs = physics(body, momenta, forces, context[:, 0])
    return Episodes(ids, images, context, body, states, contacts, costs, cue_times, scenario)


def archive(path: Path, dataset: Episodes) -> None:
    """Store observations and body context; never store prediction labels here."""
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "CREATE TABLE episodes (id INTEGER PRIMARY KEY, context BLOB, body BLOB)"
        )
        connection.execute(
            "CREATE TABLE events (episode INTEGER, time INTEGER, observation BLOB,"
            " PRIMARY KEY (episode, time))"
        )
        connection.executemany(
            "INSERT INTO episodes VALUES (?, ?, ?)",
            ((int(identity), dataset.context[i].tobytes(), dataset.body[i].tobytes())
             for i, identity in enumerate(dataset.ids)),
        )
        connection.executemany(
            "INSERT INTO events VALUES (?, ?, ?)",
            ((int(identity), step, dataset.images[i, step].tobytes())
             for i, identity in enumerate(dataset.ids)
             for step in range(dataset.images.shape[1])),
        )
        connection.execute("PRAGMA user_version = 5")
        connection.commit()


def select_records(priority: torch.Tensor, capacity: int) -> torch.Tensor:
    """Equivalent to keeping a bounded priority buffer for each stream prefix."""
    selected = priority.topk(min(capacity, priority.shape[1]), dim=1).indices
    return selected.sort(dim=1).values


class RetentionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Sequential(nn.Linear(108, 48), nn.GELU(), nn.Linear(48, 32))
        self.body = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 32))
        self.read_key = nn.Linear(32, 16)
        self.read_query = nn.Linear(32, 16)
        self.value = nn.Sequential(nn.Linear(33, 32), nn.GELU(), nn.Linear(32, 1))
        self.fuse = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 48), nn.GELU())
        self.action = nn.Linear(48, 3)
        self.state = nn.Linear(48, 6)
        self.contact = nn.Linear(48, 3)

    def decode(self, query: torch.Tensor, memory: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.fuse(torch.cat((query, memory), -1))
        return {
            "action": self.action(hidden),
            "states": self.state(hidden).reshape(*hidden.shape[:-1], 3, 2),
            "contacts": self.contact(hidden),
            "activity": hidden.abs().mean(-1),
        }

    def forward(
        self, batch: dict[str, torch.Tensor], mode: str, *, replay: bool = False
    ) -> dict[str, Any]:
        images, context = batch["images"], batch["context"]
        count, steps = images.shape[:2]
        events = self.visual(images.flatten(2))
        query = self.body(torch.cat((batch["body"], context), -1))
        contemporaneous = context[:, None].expand(-1, steps, -1)
        utility = self.value(torch.cat((events.detach(), contemporaneous), -1)).squeeze(-1)
        read_scores = (
            self.read_key(events) * self.read_query(query)[:, None]
        ).sum(-1) / 4.0
        if mode == "full":
            indices = torch.arange(steps, device=images.device).expand(count, -1)
        elif mode == "fifo":
            priority = torch.arange(steps, device=images.device).float().expand(count, -1)
            indices = select_records(priority, CAPACITY)
        elif mode == "surprise":
            previous = torch.cat((torch.zeros_like(images[:, :1]), images[:, :-1]), 1)
            priority = (images - previous).square().mean((2, 3, 4))
            indices = select_records(priority, CAPACITY)
        elif mode in ("value", "anti_value"):
            indices = select_records(utility if mode == "value" else -utility, CAPACITY)
        elif mode == "none":
            indices = torch.empty((count, 0), dtype=torch.long, device=images.device)
        else:
            raise ValueError(mode)
        if indices.shape[1]:
            chosen_scores = read_scores.gather(1, indices)
            if mode == "full":
                read_indices = indices
            else:
                selection = chosen_scores.topk(min(READS, indices.shape[1]), dim=1).indices
                read_indices = indices.gather(1, selection)
                chosen_scores = read_scores.gather(1, read_indices)
            weights = chosen_scores.softmax(-1)
            values = events.gather(1, read_indices[..., None].expand(-1, -1, events.shape[-1]))
            memory = (weights[..., None] * values).sum(1)
        else:
            read_indices = indices
            memory = torch.zeros_like(events[:, 0])
        result: dict[str, Any] = self.decode(query, memory)
        result.update(indices=indices, read_indices=read_indices, utility=utility)
        if replay:
            # Only training sees this archive branch; evaluation returns above.
            archive_weights = read_scores.softmax(-1)
            archive_memory = (archive_weights[..., None] * events).sum(1)
            result["replay"] = self.decode(query, archive_memory)
            with torch.no_grad():
                # Stop-gradient counterfactuals train the value predictor without
                # injecting prediction labels into the active-memory answer.
                omitted = (
                    archive_memory[:, None] - archive_weights[..., None] * events
                ) / (1.0 - archive_weights[..., None]).clamp_min(1e-5)
                result["omitted"] = self.decode(
                    query[:, None].expand(-1, steps, -1), omitted
                )
        return result


def environment_loss(
    predictions: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    costs = batch["costs"]
    target = (costs - costs.min(-1, keepdim=True).values <= TOLERANCE).float()
    target = target / target.sum(-1, keepdim=True)
    states, contacts = batch["states"], batch["contacts"]
    if predictions["action"].ndim == 3:
        target = target[:, None].expand_as(predictions["action"])
        states = states[:, None].expand_as(predictions["states"])
        contacts = contacts[:, None].expand_as(predictions["contacts"])
    action = -(target * predictions["action"].log_softmax(-1)).sum(-1)
    state = (predictions["states"] - states).square().mean((-1, -2))
    contact = F.binary_cross_entropy_with_logits(
        predictions["contacts"], contacts, reduction="none"
    ).mean(-1)
    total = action + 4.0 * state + 0.2 * contact + 0.001 * predictions["activity"]
    return total, {"action": action, "state": state, "contact": contact}


def training_loss(
    output: dict[str, Any], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    main, components = environment_loss(output, batch)
    replay, _ = environment_loss(output["replay"], batch)
    counterfactual, _ = environment_loss(output["omitted"], batch)
    delta = counterfactual.detach() - replay.detach()[:, None]
    targets = (delta / 0.03).softmax(-1)
    value_loss = -(targets * output["utility"].log_softmax(-1)).sum(-1).mean()
    total = main.mean() + 0.5 * replay.mean() + 0.1 * value_loss
    return total, {
        "loss": float(total.detach()),
        "state_mse": float(components["state"].mean().detach()),
        "action_loss": float(components["action"].mean().detach()),
        "contact_bce": float(components["contact"].mean().detach()),
        "value_loss": float(value_loss.detach()),
        "counterfactual_spread": float(delta.std(-1).mean()),
    }


@torch.no_grad()
def evaluate(
    model: RetentionModel, mode: str, dataset: Episodes, *, device: str = "cuda"
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    arrays: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "success", "regret", "state_mse", "collision", "fine_retained",
            "coarse_retained", "risk", "nuisance", "contact_correct",
            "contact_truth", "selected_action",
        )
    }
    for start in range(0, len(dataset), 128):
        batch = dataset.take(slice(start, start + 128)).tensors(device)
        output = model(batch, mode, replay=False)
        action = output["action"].argmax(-1)
        cost = batch["costs"].gather(1, action[:, None]).squeeze(-1)
        regret = cost - batch["costs"].min(-1).values
        contacts = output["contacts"].sigmoid() >= 0.5
        truth = batch["contacts"] >= 0.5
        retained = output["indices"]
        fine = (retained == batch["cue_times"][:, 1:2]).any(1)
        coarse = (retained == batch["cue_times"][:, :1]).any(1)
        nuisance = retained.shape[1] - fine.int() - coarse.int()
        values = {
            "success": regret <= TOLERANCE,
            "regret": regret,
            "state_mse": (output["states"] - batch["states"]).square().mean((1, 2)),
            "collision": batch["contacts"].gather(1, action[:, None]).squeeze(-1),
            "fine_retained": fine,
            "coarse_retained": coarse,
            "risk": batch["context"][:, 0] > 0.5,
            "nuisance": nuisance,
            "contact_correct": contacts == truth,
            "contact_truth": truth,
            "selected_action": action,
        }
        for name, value in values.items():
            arrays[name].append(value.cpu().numpy())
    combined = {name: np.concatenate(value) for name, value in arrays.items()}
    risk = combined["risk"]
    contact_truth, contact_correct = combined["contact_truth"], combined["contact_correct"]
    tpr = float(contact_correct[contact_truth].mean()) if contact_truth.any() else None
    tnr = float(contact_correct[~contact_truth].mean()) if (~contact_truth).any() else None
    metrics = {
        "episodes": len(dataset),
        "success": float(combined["success"].mean()),
        "regret": float(combined["regret"].mean()),
        "state_mse": float(combined["state_mse"].mean()),
        "collision": float(combined["collision"].mean()),
        "risk_success": float(combined["success"][risk].mean()) if risk.any() else None,
        "fine_retained_risk": float(combined["fine_retained"][risk].mean()) if risk.any() else None,
        "fine_retained_safe": float(combined["fine_retained"][~risk].mean()) if (~risk).any() else None,
        "coarse_retained": float(combined["coarse_retained"].mean()),
        "nuisance_records": float(combined["nuisance"].mean()),
        "contact_positive_fraction": float(contact_truth.mean()),
        "contact_tpr": tpr,
        "contact_tnr": tnr,
        "contact_balanced_accuracy": (tpr + tnr) / 2.0 if tpr is not None and tnr is not None else None,
        "event_records_scored": int(dataset.images.shape[1]),
        "active_records": dataset.images.shape[1] if mode == "full" else 0 if mode == "none" else CAPACITY,
        "read_records": dataset.images.shape[1] if mode == "full" else 0 if mode == "none" else READS,
    }
    return metrics, combined


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subset(batch: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: value[indices] for name, value in batch.items()}


def fit_cohort(
    train: Episodes,
    development: Episodes,
    *,
    repeats: int,
    seed: int,
    output: Path,
    modes: tuple[str, ...],
) -> dict[str, RetentionModel]:
    output.mkdir(parents=True)
    tensors = train.tensors()
    models, optimizers = {}, {}
    for mode in modes:
        seed_all(seed)
        model = RetentionModel().cuda()
        models[mode] = model
        optimizers[mode] = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
    generator = torch.Generator(device="cuda").manual_seed(seed + 983)
    history: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    updates = 0
    started = time.perf_counter()
    checkpoints = sorted(set((1, max(1, repeats // 4), max(1, repeats // 2), repeats)))
    for repeat in range(1, repeats + 1):
        order = torch.randperm(len(train), device="cuda", generator=generator)
        sums = {mode: {} for mode in modes}
        batches = 0
        for start in range(0, len(train), 64):
            batch = subset(tensors, order[start:start + 64])
            batches += 1
            updates += 1
            for mode, model in models.items():
                model.train()
                optimizer = optimizers[mode]
                optimizer.zero_grad(set_to_none=True)
                output_batch = model(batch, mode, replay=True)
                loss, parts = training_loss(output_batch, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{mode}: nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                for name, value in parts.items():
                    sums[mode][name] = sums[mode].get(name, 0.0) + value
        if repeat in checkpoints:
            for mode, model in models.items():
                metrics, _ = evaluate(model, mode, development)
                record = {
                    "repeat": repeat, "presentations": repeat * len(train),
                    "updates": updates,
                    "train": {name: value / batches for name, value in sums[mode].items()},
                    "development": metrics,
                }
                history[mode].append(record)
            dump(output / "history.json", history)
            emit(
                "checkpoint", unique=len(train), repeats=repeat, seed=seed,
                seconds=round(time.perf_counter() - started, 1),
                success={mode: round(history[mode][-1]["development"]["success"], 4) for mode in modes},
            )
    summary = {
        "unique_episodes": len(train), "repeats": repeats,
        "presentations": len(train) * repeats, "updates": updates,
        "seed": seed, "parameters": sum(p.numel() for p in next(iter(models.values())).parameters()),
        "seconds": time.perf_counter() - started,
        "train_fingerprint": train.fingerprint(),
        "development": {mode: history[mode][-1]["development"] for mode in modes},
        "replay_extra_reads": int(len(train) * repeats * train.images.shape[1]),
        "counterfactual_decoder_evaluations": int(len(train) * repeats * train.images.shape[1]),
    }
    dump(output / "development.json", summary)
    for mode, model in models.items():
        torch.save(model.state_dict(), output / f"{mode}.pt")
    return models


def run(stage: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required. No implicit CPU fallback.")
    torch.set_num_threads(2)
    # Sources and all evaluation plans are fixed before results are opened.
    output = ROOT / stage
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite existing run: {output}")
    output.mkdir()
    shutil.copyfile(Path(__file__), output / "source_snapshot.py")
    manifest = {
        "stage": stage, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "modes": MODES, "volumes": VOLUMES,
        "seeds": (11,) if stage == "discovery" else (11, 23, 37),
        "capacity": CAPACITY, "reads": READS,
        "truth": "Independent physical outcome cost; all near-optimal actions pass.",
        "retention": "Evaluation priority never sees outcome or cue-time metadata.",
        "feedback": "Evaluation scores precede learning; frozen tests do not update.",
    }
    dump(output / "frozen_manifest.json", manifest)
    emit("started", **manifest)
    pool_seed = 310000 if stage == "discovery" else 510000
    development_seed = 410000 if stage == "discovery" else 610000
    pool = generate(512 if stage == "discovery" else 2048, pool_seed)
    development = generate(256, development_seed, scenario="development")
    dump(output / "data_manifest.json", {
        "train": pool.fingerprint(), "development": development.fingerprint(),
        "train_ids": [int(pool.ids.min()), int(pool.ids.max())],
        "development_ids": [int(development.ids.min()), int(development.ids.max())],
    })
    archive(output / "observations.sqlite", pool)
    conditions = ((512, 8),) if stage == "discovery" else VOLUMES
    final_records = []
    for unique, repeats in conditions:
        for seed in manifest["seeds"]:
            target = output / f"u{unique}_r{repeats}_s{seed}"
            models = fit_cohort(
                pool.take(slice(0, unique)), development,
                repeats=repeats, seed=seed, output=target, modes=MODES,
            )
            if stage == "discovery":
                continue
            record: dict[str, Any] = {
                "unique": unique, "repeats": repeats, "presentations": unique * repeats,
                "seed": seed, "scenarios": {},
            }
            for scenario, test_seed, delay in (
                ("ordinary", 710000, 18),
                ("long_delay", 720000, 34),
                ("left_layout", 730000, 18),
                ("texture_shift", 740000, 34),
            ):
                test = generate(512, test_seed, delay=delay, scenario=scenario)
                record["scenarios"][scenario] = {}
                for mode, model in models.items():
                    metrics, arrays = evaluate(model, mode, test)
                    record["scenarios"][scenario][mode] = metrics
                    np.savez_compressed(
                        target / f"predictions_{scenario}_{mode}.npz",
                        ids=test.ids, **arrays,
                    )
            dump(target / "final.json", record)
            final_records.append(record)
            dump(output / "final_results.json", final_records)
            emit(
                "final_cohort", unique=unique, repeats=repeats, seed=seed,
                ordinary={mode: round(record["scenarios"]["ordinary"][mode]["success"], 4) for mode in MODES},
            )
            del models
    emit("completed", stage=stage, cohorts=len(conditions) * len(manifest["seeds"]))
    dump(output / "COMPLETE.json", {"stage": stage, "cohorts": len(conditions) * len(manifest["seeds"])})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("discovery", "confirmation"), required=True)
    run(parser.parse_args().stage)
