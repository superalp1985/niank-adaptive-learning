"""CUDA-only learning-volume experiment on a connectome-derived rate RNN.

This is neither the upstream LIF model nor a whole-brain learning simulation.
Activity L1 is a representation proxy; every recurrent update uses a dense kernel.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
ACTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
HELDOUT_PAIR = (0, 2)
OBJECTIVES = ("task_only", "task_predict", "task_predict_consistency", "full")
SCENARIOS = ("in_distribution", "unseen_pair", "long_chain", "long_unseen_pair")
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Config:
    matrix_path: str
    output_dir: str
    pilot: bool = False
    unique_counts: tuple[int, ...] = (64, 256, 1024)
    repetitions: tuple[int, ...] = (1, 4, 16)
    seeds: tuple[int, ...] = (1103, 2207, 3301)
    architectures: tuple[str, ...] = ("connectome", "rewired", "gru")
    objectives: tuple[str, ...] = OBJECTIVES
    data_seed: int = 801017
    batch_size: int = 64
    validation_per_scenario: int = 256
    test_per_scenario: int = 512
    learning_rate: float = 0.003
    train_noise: float = 0.03
    eval_noise: float = 0.03
    prediction_weight: float = 0.5
    consistency_weight: float = 1.0
    variance_weight: float = 1.0
    variance_floor: float = 0.1
    activity_weight: float = 0.001
    leak: float = 0.5
    gru_hidden: int = 0
    evaluation_batch_size: int = 256
    save_predictions: bool = True


def transition(position: int, action: int) -> int:
    """Independent environment truth: four translations on a 4 by 4 torus."""
    dx, dy = ACTIONS[action]
    return ((position // 4 + dy) % 4) * 4 + (position % 4 + dx) % 4


def rollout(position: int, actions: tuple[int, ...]) -> tuple[int, ...]:
    result = []
    for action in actions:
        position = transition(position, action)
        result.append(position)
    return tuple(result)


def has_heldout_pair(actions: tuple[int, ...]) -> bool:
    return any(pair == HELDOUT_PAIR for pair in zip(actions, actions[1:]))


@dataclass
class Dataset:
    starts: np.ndarray
    actions: np.ndarray
    lengths: np.ndarray
    states: np.ndarray
    scenario: str

    def __len__(self) -> int:
        return len(self.starts)

    def take(self, indices: Any) -> "Dataset":
        return Dataset(self.starts[indices], self.actions[indices], self.lengths[indices],
                       self.states[indices], self.scenario)

    def identities(self) -> list[tuple[int, ...]]:
        return [(int(start), *map(int, actions[:length]))
                for start, actions, length in zip(self.starts, self.actions, self.lengths)]

    def fingerprint(self) -> str:
        data = np.column_stack((self.starts, self.lengths, self.actions, self.states))
        return hashlib.sha256(data.astype("<i8").tobytes()).hexdigest()


def make_pool(lengths: tuple[int, ...], unseen: bool, scenario: str) -> Dataset:
    records = [(start, actions) for length in lengths
               for actions in itertools.product(range(4), repeat=length)
               if has_heldout_pair(actions) == unseen for start in range(16)]
    starts = np.asarray([row[0] for row in records], dtype=np.int64)
    steps = np.asarray([len(row[1]) for row in records], dtype=np.int64)
    actions = np.full((len(records), max(lengths)), -1, dtype=np.int64)
    states = actions.copy()
    for i, (start, sequence) in enumerate(records):
        actions[i, :len(sequence)] = sequence
        states[i, :len(sequence)] = rollout(start, sequence)
    return Dataset(starts, actions, steps, states, scenario)


def make_splits(config: Config) -> tuple[Dataset, dict[str, Dataset], dict[str, Dataset]]:
    """One fixed split and nested training order shared by all model seeds."""
    rng = np.random.default_rng(config.data_seed)
    train = None
    validation, test = {}, {}
    for scenario, lengths, unseen in (
        (SCENARIOS[0], (3, 4), False),
        (SCENARIOS[1], (3, 4), True),
        (SCENARIOS[2], (5, 6), False),
        (SCENARIOS[3], (5, 6), True),
    ):
        pool = make_pool(lengths, unseen, scenario)
        order = rng.permutation(len(pool))
        offset = max(config.unique_counts) if scenario == SCENARIOS[0] else 0
        required = offset + config.validation_per_scenario + config.test_per_scenario
        if required > len(pool):
            raise ValueError(f"{scenario}: {required} requested from only {len(pool)} identities")
        if offset:
            train = pool.take(order[:offset])
        v_end = offset + config.validation_per_scenario
        validation[scenario] = pool.take(order[offset:v_end])
        test[scenario] = pool.take(order[v_end:required])
    assert train is not None
    datasets = [train, *validation.values(), *test.values()]
    seen: set[tuple[int, ...]] = set()
    for dataset in datasets:
        identities = dataset.identities()
        if len(set(identities)) != len(identities) or seen.intersection(identities):
            raise AssertionError("Duplicate identity or cross-split leakage")
        seen.update(identities)
    return train, validation, test


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite square post-by-pre matrix")
    scale = np.abs(matrix).sum(axis=1, keepdims=True)
    return matrix / np.maximum(scale, 1e-12)


def signed_degree(matrix: np.ndarray) -> dict[str, list[int]]:
    return {f"{sign}_{axis}": np.sum(mask, axis=dim).tolist()
            for sign, mask in (("positive", matrix > 0), ("negative", matrix < 0))
            for axis, dim in (("incoming", 1), ("outgoing", 0))}


def load_matrices(location: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(location, allow_pickle=False) as data:
        matrices = {name: np.array(data[key], dtype=np.float32)
                    for name, key in (("connectome", "matrix"), ("rewired", "rewired_matrix"))}
        ids = np.array(data["ids"], dtype=np.int64)
    for name, matrix in matrices.items():
        row_normalize(matrix)
        if matrix.shape[0] != len(ids):
            raise ValueError(f"{name} and ids have different shapes")
    if signed_degree(matrices["connectome"]) != signed_degree(matrices["rewired"]):
        raise ValueError("Rewired matrix must preserve signed incoming and outgoing degrees")
    for sign in (1, -1):
        first = matrices["connectome"][matrices["connectome"] * sign > 0]
        second = matrices["rewired"][matrices["rewired"] * sign > 0]
        if not np.array_equal(np.sort(first), np.sort(second)):
            raise ValueError("Rewired matrix must preserve the signed weight multiset")
    return matrices, {
        "nodes": len(ids),
        "edges": int(np.count_nonzero(matrices["connectome"])),
        "positive_edges": int(np.sum(matrices["connectome"] > 0)),
        "negative_edges": int(np.sum(matrices["connectome"] < 0)),
        "signed_degree_preserved": True,
        "signed_weight_multiset_preserved_before_row_normalization": True,
        "retained_topology_fraction": float(np.sum((matrices["connectome"] != 0)
                                                   & (matrices["rewired"] != 0))
                                              / max(1, np.count_nonzero(matrices["connectome"]))),
    }


class RateRNN(nn.Module):
    def __init__(self, matrix: np.ndarray, leak: float = 0.5):
        super().__init__()
        weight = row_normalize(matrix)
        self.hidden_size = weight.shape[0]
        self.leak = leak
        post, pre = np.nonzero(weight)
        magnitude = torch.from_numpy(np.abs(weight[post, pre]))
        self.register_buffer("post", torch.from_numpy(post).long())
        self.register_buffer("pre", torch.from_numpy(pre).long())
        self.register_buffer("edge_sign", torch.from_numpy(np.sign(weight[post, pre])))
        # Inverse softplus preserves each observed edge's initial magnitude.
        self.edge_raw = nn.Parameter(torch.log(torch.expm1(magnitude.clamp_min(1e-8))))
        self.initial = nn.Linear(16, self.hidden_size)
        self.action_input = nn.Linear(4, self.hidden_size, bias=False)
        self.recurrent_bias = nn.Parameter(torch.zeros(self.hidden_size))
        self.readout = nn.Linear(self.hidden_size, 16)

    def recurrent_weight(self) -> torch.Tensor:
        magnitudes = F.softplus(self.edge_raw)
        weight = magnitudes.new_zeros(self.hidden_size, self.hidden_size)
        weight = weight.index_put((self.post, self.pre), magnitudes * self.edge_sign)
        # Keep rows bounded throughout learning; missing rows remain exactly zero.
        return weight / weight.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)

    def forward(self, frame: torch.Tensor, actions: torch.Tensor,
                lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(self.initial(frame))
        weight = self.recurrent_weight()
        action_features = self.action_input(F.one_hot(actions.clamp_min(0), 4).float())
        states = []
        for step in range(actions.shape[1]):
            candidate = torch.tanh(F.linear(hidden, weight, self.recurrent_bias)
                                   + action_features[:, step])
            candidate = (1.0 - self.leak) * hidden + self.leak * candidate
            hidden = torch.where((step < lengths).unsqueeze(1), candidate, hidden)
            states.append(hidden)
        trajectory = torch.stack(states, dim=1)
        return self.readout(trajectory), trajectory


class PlainGRU(nn.Module):
    """Ordinary learned recurrent baseline; its parameter count is reported."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.initial = nn.Linear(16, hidden_size)
        self.recurrent = nn.GRUCell(4, hidden_size)
        self.readout = nn.Linear(hidden_size, 16)

    def forward(self, frame: torch.Tensor, actions: torch.Tensor,
                lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(self.initial(frame))
        features = F.one_hot(actions.clamp_min(0), 4).float()
        states = []
        for step in range(actions.shape[1]):
            candidate = self.recurrent(features[:, step], hidden)
            hidden = torch.where((step < lengths).unsqueeze(1), candidate, hidden)
            states.append(hidden)
        trajectory = torch.stack(states, dim=1)
        return self.readout(trajectory), trajectory


def create_model(architecture: str, matrices: dict[str, np.ndarray], config: Config) -> nn.Module:
    if architecture == "gru":
        target = np.count_nonzero(matrices["connectome"]) + 38 * len(matrices["connectome"]) + 16
        matched = max(1, round((-51 + math.sqrt(51**2 + 12 * (target - 16))) / 6))
        return PlainGRU(config.gru_hidden or matched)
    return RateRNN(matrices[architecture], leak=config.leak)


def masks(lengths: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(steps, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1), positions < (lengths - 1).unsqueeze(1)


def task_logits(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return logits[torch.arange(len(lengths), device=lengths.device), lengths - 1]


def loss_terms(logits: torch.Tensor, hidden: torch.Tensor, states: torch.Tensor,
               lengths: torch.Tensor, objective: str, config: Config,
               second_hidden: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    valid, intermediate = masks(lengths, logits.shape[1])
    final_targets = states[torch.arange(len(lengths), device=lengths.device), lengths - 1]
    task = F.cross_entropy(task_logits(logits, lengths), final_targets)
    zero = task.new_zeros(())
    prediction = (F.cross_entropy(logits[intermediate], states[intermediate])
                  if objective != "task_only" else zero)
    consistency, variance = zero, zero
    if objective in ("task_predict_consistency", "full"):
        if second_hidden is None:
            raise ValueError("Consistency objectives require an independent augmented view")
        first, second = hidden[valid], second_hidden[valid]
        consistency = F.mse_loss(first, second)
        variance = 0.5 * (
            F.relu(config.variance_floor - (first.var(dim=0, unbiased=False) + 1e-4).sqrt()).mean()
            + F.relu(config.variance_floor - (second.var(dim=0, unbiased=False) + 1e-4).sqrt()).mean()
        )
    activity = hidden[valid].abs().mean()
    total = (task + config.prediction_weight * prediction
             + config.consistency_weight * consistency + config.variance_weight * variance)
    if objective == "full":
        total = total + config.activity_weight * activity
    return {"total": total, "task": task, "prediction": prediction,
            "consistency": consistency, "variance": variance, "activity_l1_proxy": activity}


def seed_model(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(dataset: Dataset, device: torch.device) -> dict[str, torch.Tensor]:
    return {name: torch.as_tensor(getattr(dataset, name), device=device)
            for name in ("starts", "actions", "lengths", "states")}


def augmented_frame(starts: torch.Tensor, noise: float,
                    generator: torch.Generator) -> torch.Tensor:
    frame = F.one_hot(starts, 16).float()
    if noise:
        frame = frame + noise * torch.randn(frame.shape, device=frame.device, generator=generator)
    return frame


def json_write(location: Path, value: Any) -> None:
    temporary = location.with_suffix(location.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                         encoding="utf-8")
    os.replace(temporary, location)


@torch.no_grad()
def evaluate(model: nn.Module, dataset: Dataset, device: torch.device, config: Config,
             prediction_path: Path | None = None) -> dict[str, Any]:
    model.eval()
    tensors = to_device(dataset, device)
    noise_generator = torch.Generator(device=device).manual_seed(config.data_seed + 709)
    outputs, step_outputs, targets, all_ce, all_brier = [], [], [], [], []
    step_ce, step_brier, activity_total, states_count = 0.0, 0.0, 0.0, 0
    intermediate_correct, intermediate_count, step_correct, step_count = 0, 0, 0, 0
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for offset in range(0, len(dataset), config.evaluation_batch_size):
        batch = {key: value[offset:offset + config.evaluation_batch_size]
                 for key, value in tensors.items()}
        frame = augmented_frame(batch["starts"], config.eval_noise, noise_generator)
        logits, hidden = model(frame, batch["actions"], batch["lengths"])
        valid, intermediate = masks(batch["lengths"], logits.shape[1])
        selected = task_logits(logits, batch["lengths"])
        reference = batch["states"][torch.arange(len(frame), device=device), batch["lengths"] - 1]
        probabilities = selected.softmax(dim=-1)
        final_ce = F.cross_entropy(selected, reference, reduction="none")
        final_brier = (probabilities - F.one_hot(reference, 16)).square().sum(dim=-1)
        step_probabilities = logits[valid].softmax(dim=-1)
        step_reference = batch["states"][valid]
        step_ce += float(F.cross_entropy(logits[valid], step_reference, reduction="sum"))
        step_brier += float((step_probabilities - F.one_hot(step_reference, 16)).square().sum())
        step_predictions = logits.argmax(dim=-1)
        step_correct += int((step_predictions[valid] == step_reference).sum())
        step_count += int(valid.sum())
        intermediate_correct += int((step_predictions[intermediate]
                                     == batch["states"][intermediate]).sum())
        intermediate_count += int(intermediate.sum())
        activity_total += float(hidden[valid].abs().sum())
        states_count += int(valid.sum()) * hidden.shape[-1]
        outputs.append(probabilities.cpu().numpy())
        targets.append(reference.cpu().numpy())
        step_outputs.append(step_predictions.cpu().numpy())
        all_ce.append(final_ce.cpu().numpy())
        all_brier.append(final_brier.cpu().numpy())
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    probabilities = np.concatenate(outputs)
    reference = np.concatenate(targets)
    prediction = probabilities.argmax(axis=-1)
    next_predictions = np.concatenate(step_outputs)
    ce = np.concatenate(all_ce)
    brier = np.concatenate(all_brier)
    complete_correct = np.all((next_predictions == dataset.states) | (dataset.states < 0), axis=1)
    if prediction_path is not None:
        np.savez_compressed(prediction_path, starts=dataset.starts, actions=dataset.actions,
                            lengths=dataset.lengths, states=dataset.states, truth=reference,
                            probabilities=probabilities, prediction=prediction,
                            step_predictions=next_predictions, final_ce=ce, final_brier=brier)
    return {
        "questions": len(dataset), "correct": int(np.sum(prediction == reference)),
        "accuracy": float(np.mean(prediction == reference)),
        "final_ce": float(ce.mean()), "final_brier": float(brier.mean()),
        "all_steps_correct": int(complete_correct.sum()),
        "all_steps_accuracy": float(complete_correct.mean()),
        "next_state_accuracy": step_correct / step_count,
        "next_state_ce": step_ce / step_count,
        "next_state_brier": step_brier / step_count,
        "intermediate_accuracy": intermediate_correct / intermediate_count,
        "activity_l1_proxy": activity_total / states_count,
        "evaluation_seconds": elapsed,
        "length_counts": {str(length): int(np.sum(dataset.lengths == length))
                          for length in sorted(set(map(int, dataset.lengths)))},
        "predictions_file": prediction_path.name if prediction_path else None,
    }


def train_series(architecture: str, objective: str, unique: int, seed: int,
                 matrices: dict[str, np.ndarray], train: Dataset,
                 validation: dict[str, Dataset], test: dict[str, Dataset],
                 config: Config, output: Path, device: torch.device) -> list[dict[str, Any]]:
    seed_model(seed)
    model = create_model(architecture, matrices, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    training = train.take(slice(0, unique))
    data = to_device(training, device)
    order_generator = torch.Generator(device="cpu").manual_seed(seed + 1009)
    primary_noise = torch.Generator(device=device).manual_seed(seed + 2017)
    secondary_noise = torch.Generator(device=device).manual_seed(seed + 3011)
    updates, training_seconds, optimization_steps, records = 0, 0.0, [], []
    cost_accumulators = {name: 0.0 for name in
                         ("total", "task", "prediction", "consistency", "variance", "activity_l1_proxy")}
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_edges = int(np.count_nonzero(matrices[architecture])) if architecture != "gru" else None
    tag = f"{architecture}__{objective}__u{unique}__seed{seed}"
    cuda_peak = 0
    for repeat in range(1, max(config.repetitions) + 1):
        model.train()
        permutation = torch.randperm(unique, generator=order_generator)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        epoch_loss = {name: 0.0 for name in cost_accumulators}
        for offset in range(0, unique, config.batch_size):
            indices = permutation[offset:offset + config.batch_size].to(device)
            batch = {key: value[indices] for key, value in data.items()}
            first_frame = augmented_frame(batch["starts"], config.train_noise, primary_noise)
            logits, hidden = model(first_frame, batch["actions"], batch["lengths"])
            second_hidden = None
            if objective in ("task_predict_consistency", "full"):
                second_frame = augmented_frame(batch["starts"], config.train_noise, secondary_noise)
                _, second_hidden = model(second_frame, batch["actions"], batch["lengths"])
            terms = loss_terms(logits, hidden, batch["states"], batch["lengths"],
                               objective, config, second_hidden)
            if not bool(torch.isfinite(terms["total"])):
                raise FloatingPointError(f"Non-finite loss: {tag} repeat={repeat} update={updates}")
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            for name, value in terms.items():
                contribution = float(value.detach()) * len(indices)
                epoch_loss[name] += contribution
                cost_accumulators[name] += contribution
        torch.cuda.synchronize(device)
        training_seconds += time.perf_counter() - started
        cuda_peak = max(cuda_peak, torch.cuda.max_memory_allocated(device))
        optimization_steps.append({"repeat": repeat, "updates": updates,
                                   **{key: value / unique for key, value in epoch_loss.items()}})
        if repeat not in config.repetitions:
            continue
        identifier = f"{tag}__r{repeat}"
        metrics = {"train": evaluate(model, training, device, config)}
        for split, datasets in (("validation", validation), ("test", {} if config.pilot else test)):
            metrics[split] = {}
            for scenario, dataset in datasets.items():
                prediction_path = (output / "predictions" / f"{identifier}__{split}__{scenario}.npz"
                                   if config.save_predictions else None)
                metrics[split][scenario] = evaluate(model, dataset, device, config, prediction_path)
        views = 2 if objective in ("task_predict_consistency", "full") else 1
        sequence_steps = int(training.lengths.sum()) * repeat
        record = {
            "id": identifier, "architecture": architecture, "objective": objective,
            "seed": seed, "unique_examples": unique, "repetitions": repeat,
            "presentations": unique * repeat, "optimizer_updates": updates,
            "final_target_labels": unique * repeat,
            "intermediate_target_labels": (int((training.lengths - 1).sum()) * repeat
                                           if objective != "task_only" else 0),
            "forward_views_per_presentation": views,
            "forward_sequence_steps": sequence_steps * views,
            "padded_forward_sequence_steps": unique * repeat * training.actions.shape[1] * views,
            "trainable_parameters": parameter_count, "trainable_recurrent_edges": trainable_edges,
            "hidden_size": model.hidden_size,
            "training_seconds_cumulative": training_seconds, "cuda_peak_allocated_bytes": cuda_peak,
            "mean_training_losses": {key: value / (unique * repeat)
                                     for key, value in cost_accumulators.items()},
            "history": list(optimization_steps), "metrics": metrics,
            "dense_kernels": True, "activity_proxy_is_paid_compute": False,
        }
        json_write(output / "records" / f"{identifier}.json", record)
        records.append(record)
        print(json.dumps({"event": "checkpoint", "id": identifier, "updates": updates,
                          "train_seconds": round(training_seconds, 3),
                          "training_accuracy": round(metrics["train"]["accuracy"], 4),
                          "training_ce": round(metrics["train"]["final_ce"], 4),
                          "validation": {k: round(v["accuracy"], 4)
                                         for k, v in metrics["validation"].items()},
                          "test": {k: round(v["accuracy"], 4)
                                   for k, v in metrics["test"].items()}}), flush=True)
    torch.save({"model": model.state_dict(), "architecture": architecture, "objective": objective,
                "unique_examples": unique, "seed": seed, "repetitions": max(config.repetitions),
                "config": asdict(config)}, output / "models" / f"{tag}.pt")
    return records


def dataset_manifest(train: Dataset, validation: dict[str, Dataset],
                     test: dict[str, Dataset], config: Config) -> dict[str, Any]:
    def item(dataset: Dataset) -> dict[str, Any]:
        return {"count": len(dataset), "sha256": dataset.fingerprint(),
                "length_counts": {str(length): int(np.sum(dataset.lengths == length))
                                  for length in sorted(set(map(int, dataset.lengths)))}}
    return {"training": {str(size): item(train.take(slice(0, size))) for size in config.unique_counts},
            "validation": {key: item(value) for key, value in validation.items()},
            "test": {key: item(value) for key, value in test.items()},
            "identity_disjoint": True, "fixed_split_shared_by_model_seeds": True,
            "heldout_ordered_pair": list(HELDOUT_PAIR)}


def summary_records(records: list[dict[str, Any]], pilot: bool) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["architecture"], record["objective"], record["unique_examples"],
               record["repetitions"])
        grouped.setdefault(key, []).append(record)
    summary = []
    for key, group in grouped.items():
        entry = dict(zip(("architecture", "objective", "unique_examples", "repetitions"), key))
        entry["seeds"] = [row["seed"] for row in group]
        entry["presentations"] = group[0]["presentations"]
        entry["metrics"] = {}
        for split in (("validation",) if pilot else ("validation", "test")):
            entry["metrics"][split] = {}
            for scenario in SCENARIOS:
                entry["metrics"][split][scenario] = {
                    metric: {"mean": float(np.mean(values)), "min": float(np.min(values)),
                             "max": float(np.max(values)), "by_seed": values}
                    for metric in ("accuracy", "final_ce", "final_brier", "all_steps_accuracy",
                                   "next_state_accuracy", "next_state_ce", "activity_l1_proxy")
                    for values in [[float(row["metrics"][split][scenario][metric]) for row in group]]
                }
        summary.append(entry)
    return summary


def run(config: Config) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is prohibited")
    if any(value <= 0 for value in (*config.unique_counts, *config.repetitions)):
        raise ValueError("Positive unique counts and repetition counts required")
    if any(name not in OBJECTIVES for name in config.objectives):
        raise ValueError("Unknown objective")
    if not 0 < config.leak <= 1:
        raise ValueError("leak must be in (0, 1]")
    output = Path(config.output_dir)
    if (output / "FROZEN.json").exists():
        raise FileExistsError(f"An experiment already exists at {output}; use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("records", "predictions", "models", "source"):
        (output / folder).mkdir(exist_ok=True)
    device = torch.device("cuda")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    matrices, matrix_metadata = load_matrices(config.matrix_path)
    train, validation, test = make_splits(config)
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in ("learning_experiment.py", "test_learning.py")}
    for name in source_hashes:
        (output / "source" / name).write_bytes((ROOT / name).read_bytes())
    manifest = {
        "schema_version": SCHEMA_VERSION, "config": asdict(config), "source_hashes": source_hashes,
        "matrix_sha256": hashlib.sha256(Path(config.matrix_path).read_bytes()).hexdigest(),
        "matrix_metadata": matrix_metadata, "datasets": dataset_manifest(train, validation, test, config),
        "selection": "All prespecified conditions are reported; test feedback never selects or tunes a model.",
        "device": torch.cuda.get_device_name(device), "pytorch": torch.__version__,
        "limitations": [
            "Connectome-derived differentiable rate model, not original LIF or whole-brain learning.",
            "Learnable signed edges with fixed topology and row-L1 normalization at every forward.",
            "Torus translation actions commute; heldout pairs do not prove temporal-order understanding.",
            "Model seeds share a fixed split; seed variation is not independent task-world replication.",
            "Intermediate-state supervision supplies extra labels and adds computation.",
            "Two augmented views add forward/backward work; presentations alone do not equal training compute.",
            "Activity L1 is not measured compute, sparsity, energy, or acceleration; kernels are dense.",
            "No FAME, Qwen parameter routing, biological plasticity, or language generalization tested.",
        ],
    }
    json_write(output / "FROZEN.json", manifest)
    print(json.dumps({"event": "frozen", "output": str(output),
                      "device": manifest["device"], "config": asdict(config)}), flush=True)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for unique in config.unique_counts:
            for seed in config.seeds:
                for architecture in config.architectures:
                    for objective in config.objectives:
                        records.extend(train_series(architecture, objective, unique, seed, matrices,
                                                    train, validation, test, config, output, device))
                        json_write(output / "progress.json", {
                            "status": "running", "completed_records": len(records),
                            "last_record": records[-1]["id"], "elapsed_seconds": time.perf_counter() - started})
        for name, digest in source_hashes.items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Frozen source changed during experiment: {name}")
        if hashlib.sha256(Path(config.matrix_path).read_bytes()).hexdigest() != manifest["matrix_sha256"]:
            raise RuntimeError("Frozen connectome artifact changed during experiment")
        result = {"status": "complete", "manifest": manifest, "records": records,
                  "summary": summary_records(records, config.pilot),
                  "elapsed_seconds": time.perf_counter() - started}
        json_write(output / "results.json", result)
        json_write(output / "progress.json", {"status": "complete", "completed_records": len(records),
                                             "elapsed_seconds": result["elapsed_seconds"]})
        print(json.dumps({"event": "complete", "records": len(records),
                          "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
        return result
    except BaseException as error:
        json_write(output / "failure.json", {"type": type(error).__name__, "message": str(error),
                                            "completed_records": len(records),
                                            "elapsed_seconds": time.perf_counter() - started})
        raise


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(ROOT / "connectome_reduced.npz"))
    parser.add_argument("--output-dir")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--unique", type=int, nargs="+")
    parser.add_argument("--repetitions", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--architectures", choices=("connectome", "rewired", "gru"), nargs="+")
    parser.add_argument("--objectives", choices=OBJECTIVES, nargs="+")
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--no-predictions", action="store_true")
    args = parser.parse_args()
    return Config(
        matrix_path=args.matrix,
        output_dir=args.output_dir or str(ROOT / ("learning_pilot" if args.pilot else "learning_run")),
        pilot=args.pilot,
        unique_counts=tuple(sorted(set(args.unique or ([1024] if args.pilot else [64, 256, 1024])))),
        repetitions=tuple(sorted(set(args.repetitions or ([1, 4, 16] if args.pilot else [1, 4, 16])))),
        seeds=tuple(args.seeds or ([1103] if args.pilot else [1103, 2207, 3301])),
        architectures=tuple(args.architectures or (["connectome", "gru"] if args.pilot
                                                   else ["connectome", "rewired", "gru"])),
        objectives=tuple(args.objectives or (["task_predict"] if args.pilot else list(OBJECTIVES))),
        learning_rate=args.learning_rate, save_predictions=not args.no_predictions,
    )


if __name__ == "__main__":
    run(parse_args())
