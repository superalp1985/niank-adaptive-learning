"""Learn conditional event value and measured physical outcome differences."""

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
import shutil
import time

import numpy as np
import torch

from retention_v5.experiment import (
    Episodes, archive, dump, emit, environment_loss, generate,
)
from retention_v6.experiment import seed_all, with_context
from retention_v7.experiment import (
    EventRetentionModel, detail_probe, metrics, recovery_probe,
)

MODES = ("quartet_point", "quartet_delta")
VOLUMES = ((512, 8), (2048, 2), (2048, 8), (2048, 32))
SEEDS = (11, 23, 37)


class OutcomeRetentionModel(EventRetentionModel):
    def __init__(self):
        super().__init__(True)

    def forward(self, batch, mode, *, replay=False, active_override=None):
        result = super().forward(batch, "paired_event", replay=replay, active_override=active_override)
        if replay:
            events = self.encode(batch["images"])
            query = self.body(torch.cat((batch["body"], batch["context"]), -1))
            queries = query[:, None].expand(-1, events.shape[1], -1)
            gates = self.event_gate(torch.cat((events, queries), -1)).sigmoid()
            scored = events * gates
            scores = (self.read_key(events) * self.read_query(query)[:, None]).sum(-1) / 4.0
            with torch.no_grad():
                weights = scores.softmax(-1)
                memory = (weights[..., None] * scored).sum(1)
                omitted = (memory[:, None] - weights[..., None] * scored) / (1.0 - weights[..., None]).clamp_min(1e-5)
                result["omitted"] = self.decode(queries, omitted)
        return result


def quartet_data(base: Episodes) -> Episodes:
    groups = [with_context(base, context, flip) for context, flip in (
        (0.0, False), (0.0, True), (1.0, False), (1.0, True),
    )]
    names = ("ids", "images", "context", "body", "states", "contacts", "costs", "cue_times")
    return Episodes(*(np.concatenate([getattr(g, name) for g in groups]) for name in names), "quartet")


def outcome_difference_loss(predictions, batch):
    prediction = predictions["states"].reshape(4, -1, 3, 2)
    truth = batch["states"].reshape(4, -1, 3, 2)
    predicted_delta = torch.stack((prediction[1] - prediction[0], prediction[3] - prediction[2]))
    actual_delta = torch.stack((truth[1] - truth[0], truth[3] - truth[2]))
    return (predicted_delta - actual_delta).square().mean()


def training_loss(output, batch, mode):
    main, components = environment_loss(output, batch)
    replay, _ = environment_loss(output["replay"], batch)
    omitted, _ = environment_loss(output["omitted"], batch)
    target = omitted.detach() - replay.detach()[:, None]
    value_loss = (output["utility"] - target).square().mean()
    delta = outcome_difference_loss(output, batch) + 0.5 * outcome_difference_loss(output["replay"], batch)
    total = main.mean() + 0.5 * replay.mean() + value_loss
    if mode == "quartet_delta":
        total = total + 8.0 * delta
    return total, {
        "loss": float(total.detach()), "value_loss": float(value_loss.detach()),
        "difference_loss": float(delta.detach()),
        "state_mse": float(components["state"].mean().detach()),
        "active_count": float(output["active_mask"].float().sum(1).mean()),
    }


def fit(base, development, mode, seed, cycles, output):
    seed_all(seed)
    model = OutcomeRetentionModel().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
    source = quartet_data(base).tensors()
    generator = torch.Generator(device="cuda").manual_seed(seed + 197)
    checkpoints = sorted(set((1, max(1, cycles // 4), max(1, cycles // 2), cycles)))
    history = []
    started = time.perf_counter()
    for cycle in range(1, cycles + 1):
        order = torch.randperm(len(base), device="cuda", generator=generator)
        sums = {}
        batches = 0
        for start in range(0, len(base), 32):
            indices = order[start:start + 32]
            indices = torch.cat([indices + group * len(base) for group in range(4)])
            batch = {name: value[indices] for name, value in source.items()}
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, parts = training_loss(model(batch, mode, replay=True), batch, mode)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            batches += 1
            for key, value in parts.items():
                sums[key] = sums.get(key, 0) + value
        if cycle in checkpoints:
            dev, _ = metrics(model, mode, development)
            history.append({"cycle": cycle, "presentations": cycle * len(base) * 4,
                            "updates": cycle * batches, "development": dev,
                            "train": {key: value / batches for key, value in sums.items()}})
            dump(output / "history.json", history)
            emit("checkpoint", mode=mode, seed=seed, unique=len(base), cycle=cycle,
                 success=dev["success"], active=dev["active_count"],
                 seconds=round(time.perf_counter() - started, 1))
    result = {
        "mode": mode, "seed": seed, "unique": len(base), "cycles": cycles,
        "presentations": len(base) * cycles * 4, "updates": cycles * batches,
        "distinct_physical_experiences": len(base) * 4,
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": time.perf_counter() - started,
        "train_fingerprint": quartet_data(base).fingerprint(),
        "counterfactual_decoder_evaluations": cycles * len(base) * 4 * base.images.shape[1],
        "duplicated_training_event_encodings": cycles * len(base) * 4 * base.images.shape[1],
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
    sources = ("experiment.py", "PROTOCOL.md", "test_contracts.py")
    for name in sources:
        shutil.copyfile(ROOT / name, output / name)
    dependencies = {}
    for version in ("retention_v5", "retention_v6", "retention_v7"):
        path = ROOT.parent / version / "experiment.py"
        dependencies[version] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copyfile(path, output / f"dependency_{version}.py")
    dev = stage == "development"
    offset = 1900000 if dev else 2100000
    datasets = {
        "train": generate(512 if dev else 2048, offset),
        "development": generate(256, offset + 10000, scenario="development"),
        "detail": generate(256, offset + 20000, delay=26, scenario="detail"),
        "recovery": generate(256, offset + 30000, delay=26, scenario="recovery"),
    }
    if not dev:
        datasets.update(
            final=generate(512, offset + 40000, scenario="final"),
            left_layout=generate(512, offset + 50000, scenario="left_layout"),
            texture_shift=generate(512, offset + 60000, delay=34, scenario="texture_shift"),
        )
    conditions = ((512, 32),) if dev else VOLUMES
    seeds = (11,) if dev else SEEDS
    manifest = {
        "stage": stage, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependencies": dependencies, "modes": MODES, "volumes": conditions, "seeds": seeds,
        "value_target": "omitted physical loss minus archive physical loss, stop-gradient",
        "difference_coefficient": 8.0,
        "data": {name: {"sha256": data.fingerprint(), "ids": [int(data.ids.min()), int(data.ids.max())]}
                 for name, data in datasets.items()},
        "final_ids": None if dev else [offset + 40000, offset + 40511],
    }
    dump(output / "manifest.json", manifest)
    for name, data in datasets.items():
        archive(output / f"{name}.sqlite", data)
    emit("started", **manifest)
    results = []
    for unique, cycles in conditions:
        for seed in seeds:
            for mode in MODES:
                target = output / f"u{unique}_c{cycles}_s{seed}_{mode}"
                target.mkdir()
                model, result = fit(datasets["train"].take(slice(0, unique)), datasets["development"], mode, seed, cycles, target)
                result["development"], _ = metrics(model, mode, datasets["development"])
                result["detail"] = detail_probe(model, mode, datasets["detail"])
                result["recovery"], arrays = recovery_probe(model, mode, datasets["recovery"], output / "recovery.sqlite")
                np.savez_compressed(target / "recovery_predictions.npz", **arrays)
                if not dev:
                    for name in ("final", "left_layout", "texture_shift"):
                        result[name], arrays = metrics(model, mode, datasets[name])
                        np.savez_compressed(target / f"{name}_predictions.npz", ids=datasets[name].ids, **arrays)
                dump(target / "result.json", result)
                results.append(result)
                dump(output / "results.json", results)
                emit("result", **result)
                del model
    dump(output / "COMPLETE.json", {"models": len(results)})
    emit("completed", models=len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "confirmation"), required=True)
    run(parser.parse_args().stage)
