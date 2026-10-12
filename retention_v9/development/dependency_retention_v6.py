"""Frozen-checkpoint state-conditioned memory and actual SQLite recovery."""

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
import random
import shutil
import sqlite3
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from retention_v5.experiment import (
    CAPACITY, READS, TOLERANCE, Episodes, archive, dump, emit,
    environment_loss, generate, physics, select_records,
)

MODES = ("state_nogate", "state_gate")
VOLUMES = ((512, 32), (2048, 8))
SEEDS = (11, 23, 37)


class StateRetentionModel(nn.Module):
    def __init__(self, gated: bool) -> None:
        super().__init__()
        self.gated = gated
        self.channel_visual = nn.Sequential(nn.Linear(36, 16), nn.GELU())
        self.visual = nn.Linear(48, 32)
        self.body = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 32))
        self.read_key = nn.Linear(32, 16)
        self.read_query = nn.Linear(32, 16)
        self.value = nn.Sequential(nn.Linear(33, 32), nn.GELU(), nn.Linear(32, 1))
        self.fuse = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 48), nn.GELU())
        self.action = nn.Linear(48, 3)
        self.state = nn.Linear(48, 6)
        self.contact = nn.Linear(48, 3)
        # Build shared modules first, so paired seeds share their initial weights.
        if gated:
            self.gate = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(images.flatten(3), dim=-1, eps=1e-5) * 6.0
        return self.visual(self.channel_visual(normalized).flatten(2))

    def decode(self, query: torch.Tensor, memory: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.fuse(torch.cat((query, memory), -1))
        return {
            "action": self.action(hidden),
            "states": self.state(hidden).reshape(*hidden.shape[:-1], 3, 2),
            "contacts": self.contact(hidden),
            "activity": hidden.abs().mean(-1),
        }

    def forward(
        self, batch: dict[str, torch.Tensor], mode: str, *,
        replay: bool = False, active_override: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if (mode == "state_gate") != self.gated:
            raise ValueError("Mode must match constructed gate")
        images, context = batch["images"], batch["context"]
        steps = images.shape[1]
        events = self.encode(images)
        query = self.body(torch.cat((batch["body"], context), -1))
        contemporaneous = context[:, None].expand(-1, steps, -1)
        utility = self.value(torch.cat((events.detach(), contemporaneous), -1)).squeeze(-1)
        read_scores = (self.read_key(events) * self.read_query(query)[:, None]).sum(-1) / 4.0
        indices = select_records(utility, CAPACITY) if active_override is None else active_override
        chosen_scores = read_scores.gather(1, indices)
        read_choice = chosen_scores.topk(min(READS, indices.shape[1]), dim=1).indices
        read_indices = indices.gather(1, read_choice)
        weights = read_scores.gather(1, read_indices).softmax(-1)
        values = events.gather(1, read_indices[..., None].expand(-1, -1, events.shape[-1]))
        memory = (weights[..., None] * values).sum(1)
        gate = self.gate(query).sigmoid() if self.gated else torch.ones_like(query[:, :1])
        result: dict[str, Any] = self.decode(query, memory * gate)
        result.update(indices=indices, read_indices=read_indices, utility=utility,
                      gate=gate.squeeze(-1))
        if replay:
            archive_weights = read_scores.softmax(-1)
            archive_memory = (archive_weights[..., None] * events).sum(1)
            result["replay"] = self.decode(query, archive_memory * gate)
            result["single_events"] = self.decode(
                query[:, None].expand(-1, steps, -1), events * gate[:, None, :]
            )
            with torch.no_grad():
                result["empty"] = self.decode(query, torch.zeros_like(archive_memory))
        return result


def training_loss(output: dict[str, Any], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    main, components = environment_loss(output, batch)
    replay, _ = environment_loss(output["replay"], batch)
    singles, _ = environment_loss(output["single_events"], batch)
    empty, _ = environment_loss(output["empty"], batch)
    delta = empty.detach()[:, None] - singles.detach()
    target = (delta / 0.03).softmax(-1)
    value_loss = -(target * output["utility"].log_softmax(-1)).sum(-1).mean()
    total = main.mean() + 0.5 * replay.mean() + 0.25 * singles.mean() + 0.1 * value_loss
    return total, {
        "loss": float(total.detach()),
        "action_loss": float(components["action"].mean().detach()),
        "state_mse": float(components["state"].mean().detach()),
        "value_loss": float(value_loss.detach()),
        "gate_mean": float(output["gate"].mean().detach()),
    }


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def with_context(dataset: Episodes, risk: float, flip: bool = False) -> Episodes:
    result = dataset.take(slice(None))
    result.images = result.images.copy()
    result.context = np.full_like(result.context, risk)
    rows = np.arange(len(result))
    if flip:
        result.images[rows, result.cue_times[:, 1], 1] *= -1
    coarse = result.images[rows, result.cue_times[:, 0], 0, 1, 1] > 0
    fine = result.images[rows, result.cue_times[:, 1], 1, 2, 2] > 0
    momentum = np.where(coarse, 0.9, 0.25).astype(np.float32)
    force = np.where(fine, 0.18, -0.18).astype(np.float32)
    result.states, result.contacts, result.costs = physics(
        result.body, momentum, force, result.context[:, 0]
    )
    return result


def load_observations(
    path: Path, ids: np.ndarray, steps: int, *,
    active_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Read image bytes from a read-only archive, optionally just live references."""
    images = np.zeros((len(ids), steps, 3, 6, 6), np.float32)
    records = byte_count = 0
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        for row, identity in enumerate(ids):
            if active_indices is None:
                fetched = connection.execute(
                    "SELECT time, observation FROM events WHERE episode=? ORDER BY time",
                    (int(identity),),
                ).fetchall()
                expected = steps
            else:
                selected = [int(step) for step in active_indices[row]]
                slots = ",".join("?" for _ in selected)
                fetched = connection.execute(
                    f"SELECT time, observation FROM events WHERE episode=? AND time IN ({slots}) ORDER BY time",
                    (int(identity), *selected),
                ).fetchall()
                expected = len(selected)
            if len(fetched) != expected:
                raise ValueError(f"Incomplete archive for episode {identity}")
            for step, blob in fetched:
                images[row, int(step)] = np.frombuffer(blob, np.float32).reshape(3, 6, 6)
                records += 1
                byte_count += len(blob)
    return images, {"records": records, "bytes": byte_count}


@torch.no_grad()
def metrics(model, mode: str, dataset: Episodes, active_override=None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    device = next(model.parameters()).device
    values = []
    for start in range(0, len(dataset), 128):
        batch = dataset.take(slice(start, start + 128)).tensors(str(device))
        output = model(batch, mode, active_override=None if active_override is None else active_override[start:start + 128])
        action = output["action"].argmax(-1)
        regret = batch["costs"].gather(1, action[:, None]).squeeze(-1) - batch["costs"].min(-1).values
        values.append({
            "success": (regret <= TOLERANCE).cpu().numpy(),
            "regret": regret.cpu().numpy(),
            "state_mse": (output["states"] - batch["states"]).square().mean((1, 2)).cpu().numpy(),
            "fine_retained": (output["indices"] == batch["cue_times"][:, 1:2]).any(1).cpu().numpy(),
            "fine_read": (output["read_indices"] == batch["cue_times"][:, 1:2]).any(1).cpu().numpy(),
            "coarse_retained": (output["indices"] == batch["cue_times"][:, :1]).any(1).cpu().numpy(),
            "gate": output["gate"].cpu().numpy(),
            "risk": (batch["context"][:, 0] > 0.5).cpu().numpy(),
            "velocity": output["states"][..., 1].cpu().numpy(),
            "action": action.cpu().numpy(),
            "active_indices": output["indices"].cpu().numpy(),
        })
    joined = {key: np.concatenate([row[key] for row in values]) for key in values[0]}
    summary = {
        key: float(joined[key].mean())
        for key in ("success", "regret", "state_mse", "fine_retained", "fine_read", "coarse_retained", "gate")
    }
    for label, mask in (("risk", joined["risk"]), ("safe", ~joined["risk"])):
        for key in ("success", "fine_retained", "fine_read", "gate"):
            summary[f"{label}_{key}"] = float(joined[key][mask].mean()) if mask.any() else None
    summary.update(episodes=len(dataset), event_records_scored=dataset.images.shape[1],
                   active_records=CAPACITY, read_records=READS)
    return summary, joined


@torch.no_grad()
def detail_probe(model, mode: str, base: Episodes) -> dict[str, Any]:
    result = {}
    for context in (0.0, 1.0):
        first, second = with_context(base, context), with_context(base, context, True)
        _, a = metrics(model, mode, first)
        _, b = metrics(model, mode, second)
        truth = second.states[..., 1] - first.states[..., 1]
        effect = b["velocity"] - a["velocity"]
        result[str(int(context))] = {
            "effect_gain": float((effect * truth).sum() / (truth**2).sum()) if context else None,
            "effect_mse": float(((effect - truth)**2).mean()),
            "predicted_effect_mae": float(np.abs(effect).mean()),
            "truth_effect_mae": float(np.abs(truth).mean()),
            "fine_retained": float((a["fine_retained"].mean() + b["fine_retained"].mean()) / 2),
            "fine_read": float((a["fine_read"].mean() + b["fine_read"].mean()) / 2),
        }
    return result


@torch.no_grad()
def intervention(model, mode: str, base: Episodes, archive_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    device = str(next(model.parameters()).device)
    safe = with_context(base, 0.0)
    _, initial = metrics(model, mode, safe)
    active = initial["active_indices"]
    risk = with_context(base, 1.0)
    stale_images, stale_io = load_observations(archive_path, base.ids, base.images.shape[1], active_indices=active)
    stale_dataset = risk.take(slice(None))
    stale_dataset.images = stale_images
    override = torch.from_numpy(active).to(device)
    stale, a = metrics(model, mode, stale_dataset, override)
    # The overwritten tensor cannot supply forgotten observations. Recovery needs SQLite.
    restored, recovery_io = load_observations(archive_path, base.ids, base.images.shape[1])
    recovered_dataset = risk.take(slice(None))
    recovered_dataset.images = restored
    recovered, b = metrics(model, mode, recovered_dataset)
    forgotten = ~initial["fine_retained"]
    result = {
        "stale_success": stale["success"], "recovered_success": recovered["success"],
        "stale_regret": stale["regret"], "recovered_regret": recovered["regret"],
        "stale_state_mse": stale["state_mse"], "recovered_state_mse": recovered["state_mse"],
        "stale_fine_retained": float(initial["fine_retained"].mean()),
        "recovered_fine_retained": recovered["fine_retained"],
        "safe_gate": float(initial["gate"].mean()), "risk_gate": recovered["gate"],
        "forgotten_episodes": int(forgotten.sum()),
        "forgotten_stale_success": float(a["success"][forgotten].mean()) if forgotten.any() else None,
        "forgotten_recovered_success": float(b["success"][forgotten].mean()) if forgotten.any() else None,
        "stale_database_reads": stale_io, "recovery_database_reads": recovery_io,
        "stale_real_observation_records": CAPACITY,
        "recovery_real_observation_records": base.images.shape[1],
        "stale_slot_computations_including_blanks": base.images.shape[1],
    }
    arrays = {
        "ids": base.ids, "forgotten": forgotten, "stale_success": a["success"],
        "recovered_success": b["success"], "stale_regret": a["regret"],
        "recovered_regret": b["regret"], "old_active": active,
        "new_active": b["active_indices"],
    }
    return result, arrays


def fit(train: Episodes, development: Episodes, mode: str, seed: int, repeats: int, output: Path) -> StateRetentionModel:
    seed_all(seed)
    model = StateRetentionModel(gated=mode == "state_gate").cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
    source = train.tensors()
    generator = torch.Generator(device="cuda").manual_seed(seed + 197)
    history = []
    started = time.perf_counter()
    checkpoints = sorted(set((1, max(1, repeats // 4), max(1, repeats // 2), repeats)))
    for repeat in range(1, repeats + 1):
        order = torch.randperm(len(train), device="cuda", generator=generator)
        sums: dict[str, float] = {}
        batches = 0
        for start in range(0, len(train), 64):
            batches += 1
            batch = {name: value[order[start:start + 64]] for name, value in source.items()}
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output_batch = model(batch, mode, replay=True)
            loss, parts = training_loss(output_batch, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + value
        if repeat in checkpoints:
            dev, _ = metrics(model, mode, development)
            history.append({"repeat": repeat, "presentations": repeat * len(train),
                            "updates": repeat * batches, "development": dev,
                            "train": {key: value / batches for key, value in sums.items()}})
            dump(output / "history.json", history)
            emit("checkpoint", mode=mode, unique=len(train), repeats=repeat, seed=seed,
                 success=dev["success"], seconds=round(time.perf_counter() - started, 1))
    dump(output / "training.json", {
        "mode": mode, "seed": seed, "unique": len(train), "repeats": repeats,
        "presentations": len(train) * repeats, "parameters": sum(p.numel() for p in model.parameters()),
        "train_fingerprint": train.fingerprint(), "seconds": time.perf_counter() - started,
        "extra_training_event_replays": len(train) * repeats * train.images.shape[1],
        "single_event_decoder_evaluations": len(train) * repeats * train.images.shape[1],
    })
    torch.save(model.state_dict(), output / "model.pt")
    return model


def run(stage: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    torch.set_num_threads(2)
    output = ROOT / stage
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    for source in (Path(__file__), ROOT / "PROTOCOL.md", ROOT / "test_contracts.py"):
        shutil.copyfile(source, output / source.name)
    dependency = ROOT.parent / "retention_v5" / "experiment.py"
    shutil.copyfile(dependency, output / "dependency_v5.py")
    development_stage = stage.startswith("development")
    offset = 1000000 if development_stage else 1200000
    pool = generate(512 if development_stage else 2048, offset)
    development = generate(256, offset + 10000, scenario="development")
    detail = generate(256, offset + 20000, delay=26, scenario="detail")
    recovery = generate(256, offset + 30000, delay=26, scenario="recovery")
    archive(output / "observations.sqlite", pool)
    archive(output / "recovery.sqlite", recovery)
    data = {"train": pool, "development": development, "detail": detail, "recovery": recovery}
    conditions = ((512, 32),) if development_stage else VOLUMES
    seeds = (11,) if development_stage else SEEDS
    manifest = {
        "stage": stage, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_sha256": hashlib.sha256(dependency.read_bytes()).hexdigest(),
        "modes": MODES, "volumes": conditions, "seeds": seeds,
        "data": {name: {"sha256": value.fingerprint(), "ids": [int(value.ids.min()), int(value.ids.max())]}
                 for name, value in data.items()},
        "final_ids": None if development_stage else [offset + 40000, offset + 40511],
        "truth": "Physical action cost; any near-optimal action passes.",
        "selection": "Final IDs fixed across optimization seeds and never used during development.",
    }
    dump(output / "manifest.json", manifest)
    emit("started", **manifest)
    results = []
    for unique, repeats in conditions:
        for seed in seeds:
            for mode in MODES:
                target = output / f"u{unique}_r{repeats}_s{seed}_{mode}"
                target.mkdir()
                model = fit(pool.take(slice(0, unique)), development, mode, seed, repeats, target)
                result = json.loads((target / "training.json").read_text(encoding="utf-8"))
                result["development"], _ = metrics(model, mode, development)
                result["detail"] = detail_probe(model, mode, detail)
                result["recovery"], arrays = intervention(model, mode, recovery, output / "recovery.sqlite")
                np.savez_compressed(target / "recovery_predictions.npz", **arrays)
                if not development_stage:
                    final = generate(512, offset + 40000, scenario="final")
                    result["final"], arrays = metrics(model, mode, final)
                    np.savez_compressed(target / "final_predictions.npz", ids=final.ids, **arrays)
                dump(target / "result.json", result)
                results.append(result)
                dump(output / "results.json", results)
                emit("result", mode=mode, seed=seed, unique=unique,
                     development=result["development"], detail=result["detail"], recovery=result["recovery"])
                del model
    dump(output / "COMPLETE.json", {"models": len(results)})
    emit("completed", models=len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "confirmation"), required=True)
    run(parser.parse_args().stage)
