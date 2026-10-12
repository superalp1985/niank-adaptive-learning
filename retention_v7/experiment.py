"""Event-specific state use, paired physical histories and signed retention value."""

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
import hashlib
import json
import shutil
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from retention_v5.experiment import (
    CAPACITY, READS, TOLERANCE, Episodes, archive, dump, emit,
    environment_loss, generate, select_records,
)
from retention_v6.experiment import (
    StateRetentionModel, load_observations, seed_all, with_context,
)

MODES = ("ordinary_plain", "ordinary_event", "paired_plain", "paired_event")
USE_COST = 0.005
VOLUMES = ((512, 16), (2048, 4), (2048, 16))
SEEDS = (11, 23, 37)


class EventRetentionModel(StateRetentionModel):
    def __init__(self, gated: bool) -> None:
        super().__init__(False)
        self.event_gated = gated
        self.value = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        if gated:
            self.event_gate = nn.Sequential(nn.Linear(64, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(
        self, batch: dict[str, torch.Tensor], mode: str, *,
        replay: bool = False, active_override: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        if mode.endswith("event") != self.event_gated:
            raise ValueError("Mode does not match gate")
        events = self.encode(batch["images"])
        steps = events.shape[1]
        query = self.body(torch.cat((batch["body"], batch["context"]), -1))
        queries = query[:, None].expand(-1, steps, -1)
        features = torch.cat((events, queries), -1)
        utility = self.value(features.detach()).squeeze(-1)
        gates = self.event_gate(features).sigmoid() if self.event_gated else torch.ones_like(events[..., :1])
        scored_events = events * gates
        read_scores = (self.read_key(events) * self.read_query(query)[:, None]).sum(-1) / 4.0
        if active_override is None:
            indices = select_records(utility, CAPACITY)
            active_mask = utility.gather(1, indices) > USE_COST
        else:
            indices, active_mask = active_override
        chosen = read_scores.gather(1, indices).masked_fill(~active_mask, -1e9)
        choice = chosen.topk(min(READS, indices.shape[1]), dim=1).indices
        read_indices = indices.gather(1, choice)
        read_mask = active_mask.gather(1, choice)
        chosen_read = chosen.gather(1, choice)
        weights = chosen_read.softmax(-1) * read_mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        values = scored_events.gather(1, read_indices[..., None].expand(-1, -1, events.shape[-1]))
        memory = (weights[..., None] * values).sum(1)
        result: dict[str, Any] = self.decode(query, memory)
        result.update(indices=indices, active_mask=active_mask, read_indices=read_indices,
                      read_mask=read_mask, utility=utility, gates=gates.squeeze(-1))
        if replay:
            archive_weights = read_scores.softmax(-1)
            result["replay"] = self.decode(query, (archive_weights[..., None] * scored_events).sum(1))
            result["single_events"] = self.decode(queries, scored_events)
            with torch.no_grad():
                result["empty"] = self.decode(query, torch.zeros_like(memory))
        return result


def training_loss(output: dict[str, Any], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    main, components = environment_loss(output, batch)
    replay, _ = environment_loss(output["replay"], batch)
    singles, _ = environment_loss(output["single_events"], batch)
    empty, _ = environment_loss(output["empty"], batch)
    target = empty.detach()[:, None] - singles.detach()
    value_loss = (output["utility"] - target).square().mean()
    total = main.mean() + 0.5 * replay.mean() + 0.25 * singles.mean() + value_loss
    return total, {
        "loss": float(total.detach()), "value_loss": float(value_loss.detach()),
        "state_mse": float(components["state"].mean().detach()),
        "active_count": float(output["active_mask"].float().sum(1).mean()),
    }


@torch.no_grad()
def metrics(model, mode: str, dataset: Episodes, active_override=None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    device = str(next(model.parameters()).device)
    values = []
    for start in range(0, len(dataset), 128):
        batch = dataset.take(slice(start, start + 128)).tensors(device)
        override = None if active_override is None else tuple(x[start:start + 128] for x in active_override)
        output = model(batch, mode, active_override=override)
        action = output["action"].argmax(-1)
        regret = batch["costs"].gather(1, action[:, None]).squeeze(-1) - batch["costs"].min(-1).values
        fine = batch["cue_times"][:, 1:2]
        coarse = batch["cue_times"][:, :1]
        values.append({
            "success": (regret <= TOLERANCE).cpu().numpy(), "regret": regret.cpu().numpy(),
            "state_mse": (output["states"] - batch["states"]).square().mean((1, 2)).cpu().numpy(),
            "fine_retained": ((output["indices"] == fine) & output["active_mask"]).any(1).cpu().numpy(),
            "fine_read": ((output["read_indices"] == fine) & output["read_mask"]).any(1).cpu().numpy(),
            "coarse_retained": ((output["indices"] == coarse) & output["active_mask"]).any(1).cpu().numpy(),
            "active_count": output["active_mask"].sum(1).cpu().numpy(),
            "read_count": output["read_mask"].sum(1).cpu().numpy(),
            "fine_gate": output["gates"].gather(1, fine).squeeze(-1).cpu().numpy(),
            "risk": (batch["context"][:, 0] > 0.5).cpu().numpy(),
            "velocity": output["states"][..., 1].cpu().numpy(),
            "action": action.cpu().numpy(), "active_indices": output["indices"].cpu().numpy(),
            "active_mask": output["active_mask"].cpu().numpy(),
        })
    joined = {key: np.concatenate([x[key] for x in values]) for key in values[0]}
    summary = {key: float(joined[key].mean()) for key in (
        "success", "regret", "state_mse", "fine_retained", "fine_read",
        "coarse_retained", "active_count", "read_count", "fine_gate",
    )}
    for label, mask in (("risk", joined["risk"]), ("safe", ~joined["risk"])):
        for key in ("success", "fine_retained", "fine_read", "fine_gate", "active_count"):
            summary[f"{label}_{key}"] = float(joined[key][mask].mean()) if mask.any() else None
    summary.update(episodes=len(dataset), event_records_scored=dataset.images.shape[1])
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
            "fine_gate": float((a["fine_gate"].mean() + b["fine_gate"].mean()) / 2),
        }
    return result


def masked_archive(path: Path, ids: np.ndarray, steps: int, indices: np.ndarray, masks: np.ndarray):
    # Variable-length references use the same label-free SQLite reader.
    pieces = []
    io = {"records": 0, "bytes": 0}
    for identity, selected, mask in zip(ids, indices, masks):
        live = selected[mask]
        if len(live):
            image, counts = load_observations(path, np.asarray([identity]), steps, active_indices=live[None])
            pieces.append(image)
            for key in io:
                io[key] += counts[key]
        else:
            pieces.append(np.zeros((1, steps, 3, 6, 6), np.float32))
    return np.concatenate(pieces), io


@torch.no_grad()
def recovery_probe(model, mode: str, base: Episodes, path: Path):
    device = str(next(model.parameters()).device)
    _, initial = metrics(model, mode, with_context(base, 0.0))
    indices, masks = initial["active_indices"], initial["active_mask"]
    stale = with_context(base, 1.0)
    stale.images, stale_io = masked_archive(path, base.ids, base.images.shape[1], indices, masks)
    override = (torch.from_numpy(indices).to(device), torch.from_numpy(masks).to(device))
    a_summary, a = metrics(model, mode, stale, override)
    recovered = with_context(base, 1.0)
    recovered.images, recovery_io = load_observations(path, base.ids, base.images.shape[1])
    b_summary, b = metrics(model, mode, recovered)
    forgotten = ~initial["fine_retained"]
    result = {
        "stale_success": a_summary["success"], "recovered_success": b_summary["success"],
        "stale_regret": a_summary["regret"], "recovered_regret": b_summary["regret"],
        "stale_state_mse": a_summary["state_mse"], "recovered_state_mse": b_summary["state_mse"],
        "safe_fine_retained": float(initial["fine_retained"].mean()),
        "recovered_fine_retained": b_summary["fine_retained"],
        "forgotten_episodes": int(forgotten.sum()),
        "forgotten_stale_success": float(a["success"][forgotten].mean()) if forgotten.any() else None,
        "forgotten_recovered_success": float(b["success"][forgotten].mean()) if forgotten.any() else None,
        "stale_database_reads": stale_io, "recovery_database_reads": recovery_io,
        "full_slot_computations_per_episode": base.images.shape[1],
    }
    arrays = {"ids": base.ids, "forgotten": forgotten, "stale_success": a["success"],
              "recovered_success": b["success"], "stale_regret": a["regret"],
              "recovered_regret": b["regret"], "old_indices": indices, "old_masks": masks,
              "new_indices": b["active_indices"], "new_masks": b["active_mask"]}
    return result, arrays


def exposure_data(base: Episodes, paired: bool) -> Episodes:
    # Repeated ordinary rows and paired rows share body, images and batch schedule.
    first = with_context(base, 0.0) if paired else base
    second = with_context(base, 1.0) if paired else base
    names = ("ids", "images", "context", "body", "states", "contacts", "costs", "cue_times")
    values = [np.concatenate((getattr(first, name), getattr(second, name))) for name in names]
    return Episodes(*values, "paired" if paired else "ordinary")


def fit(base, development, mode, seed, cycles, output):
    seed_all(seed)
    model = EventRetentionModel(mode.endswith("event")).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
    train = exposure_data(base, mode.startswith("paired"))
    source = train.tensors()
    generator = torch.Generator(device="cuda").manual_seed(seed + 197)
    checkpoints = sorted(set((1, max(1, cycles // 4), max(1, cycles // 2), cycles)))
    history = []
    started = time.perf_counter()
    for cycle in range(1, cycles + 1):
        order = torch.randperm(len(base), device="cuda", generator=generator)
        sums = {}
        batches = 0
        for start in range(0, len(base), 64):
            ids = order[start:start + 64]
            ids = torch.cat((ids, ids + len(base)))
            batch = {name: value[ids] for name, value in source.items()}
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, parts = training_loss(model(batch, mode, replay=True), batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            batches += 1
            for key, value in parts.items():
                sums[key] = sums.get(key, 0) + value
        if cycle in checkpoints:
            dev, _ = metrics(model, mode, development)
            history.append({"cycle": cycle, "presentations": cycle * len(train),
                            "updates": cycle * batches, "development": dev,
                            "train": {key: value / batches for key, value in sums.items()}})
            dump(output / "history.json", history)
            emit("checkpoint", mode=mode, seed=seed, unique=len(base), cycle=cycle,
                 success=dev["success"], active=dev["active_count"],
                 seconds=round(time.perf_counter() - started, 1))
    result = {
        "mode": mode, "seed": seed, "unique": len(base), "cycles": cycles,
        "presentations": len(train) * cycles, "updates": cycles * batches,
        "distinct_body_context_pairs": len(base) * 2 if mode.startswith("paired") else len(base),
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": time.perf_counter() - started, "train_fingerprint": train.fingerprint(),
        "auxiliary_event_decoder_calls": cycles * len(train) * train.images.shape[1],
        "archive_replay_event_encodings": cycles * len(train) * train.images.shape[1],
    }
    dump(output / "training.json", result)
    torch.save(model.state_dict(), output / "model.pt")
    return model, result


def run(stage):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(2)
    output = ROOT / stage
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    for name in ("experiment.py", "PROTOCOL.md", "test_contracts.py"):
        shutil.copyfile(ROOT / name, output / name)
    dependencies = {}
    for version in ("retention_v5", "retention_v6"):
        path = ROOT.parent / version / "experiment.py"
        dependencies[version] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copyfile(path, output / f"dependency_{version}.py")
    dev = stage != "confirmation"
    offset = 1400000 if dev else 1600000
    datasets = {
        "train": generate(512 if stage == "development" else 2048, offset),
        "development": generate(256, offset + 10000, scenario="development"),
        "detail": generate(256, offset + 20000, delay=26, scenario="detail"),
        "recovery": generate(256, offset + 30000, delay=26, scenario="recovery"),
    }
    if not dev:
        datasets["final"] = generate(512, offset + 40000, scenario="final")
    conditions = ((512, 16),) if dev else VOLUMES
    if stage == "development_volume":
        conditions = ((512, 64), (2048, 16))
    modes = MODES if dev else ("paired_plain", "paired_event")
    seeds = (11,) if dev else SEEDS
    manifest = {
        "stage": stage, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependencies": dependencies, "modes": modes, "volumes": conditions,
        "seeds": seeds, "use_cost": USE_COST,
        "high_volume_extra_controls": [] if dev else ["ordinary_plain", "ordinary_event"],
        "data": {name: {"sha256": data.fingerprint(), "ids": [int(data.ids.min()), int(data.ids.max())]}
                 for name, data in datasets.items()},
        "final_ids": None if dev else [offset + 40000, offset + 40511],
    }
    dump(output / "manifest.json", manifest)
    for name in datasets:
        archive(output / f"{name}.sqlite", datasets[name])
    emit("started", **manifest)
    results = []
    for unique, cycles in conditions:
        for seed in seeds:
            cohort_modes = MODES if not dev and (unique, cycles) == (2048, 16) else modes
            for mode in cohort_modes:
                target = output / f"u{unique}_c{cycles}_s{seed}_{mode}"
                target.mkdir()
                model, result = fit(datasets["train"].take(slice(0, unique)), datasets["development"], mode, seed, cycles, target)
                result["development"], _ = metrics(model, mode, datasets["development"])
                result["detail"] = detail_probe(model, mode, datasets["detail"])
                result["recovery"], arrays = recovery_probe(model, mode, datasets["recovery"], output / "recovery.sqlite")
                np.savez_compressed(target / "recovery_predictions.npz", **arrays)
                if not dev:
                    final = datasets["final"]
                    result["final"], arrays = metrics(model, mode, final)
                    np.savez_compressed(target / "final_predictions.npz", ids=final.ids, **arrays)
                dump(target / "result.json", result)
                results.append(result)
                dump(output / "results.json", results)
                emit("result", **result)
                del model
    dump(output / "COMPLETE.json", {"models": len(results)})
    emit("completed", models=len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "development_volume", "confirmation"), required=True)
    run(parser.parse_args().stage)
