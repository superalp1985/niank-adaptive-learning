"""Online vector directory with explicit, independently charged maintenance.

The public ingestion API accepts only observed vectors and external record IDs.
CUDA is mandatory outside explicitly opted-in small CPU contract tests. Query
refinement reads only vectors belonging to the selected directory cells.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ObserveResult:
    added: int
    observed: int
    cells: int
    new_cell: int | None
    work: float
    seconds: float


@dataclass(frozen=True)
class RouteResult:
    candidate_ids: np.ndarray
    selected_cells: np.ndarray
    best_id: int
    similarity: float
    work: float
    seconds: float

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_ids)

    @property
    def cost(self) -> float:
        return self.work


@dataclass(frozen=True)
class MaintenanceResult:
    mode: str
    executed: bool
    visited: int
    changed: int
    budget: int | None
    work: float
    seconds: float


class OnlineDirectory:
    """Grow at most one cell per observe call; maintain only on explicit calls.

    ``route(query, k)`` probes k cells, not k records. ``full`` and
    ``incremental`` maintenance score against frozen pre-operation centers,
    then update affected center sums once. They do not iterate to convergence.
    Empty cells remain dormant with stable IDs and are excluded from queries.
    Later novel observations can reuse a dormant cell.
    """

    def __init__(
        self,
        dim: int,
        max_cells: int = 32,
        novelty: float = 0.7,
        device: str = "cuda",
        allow_cpu_test: bool = False,
        read_weight: float = 1.0,
    ) -> None:
        if int(dim) != dim or dim < 1 or int(max_cells) != max_cells or max_cells < 1:
            raise ValueError("dim and max_cells must be positive integers.")
        if not math.isfinite(novelty) or not -1 <= novelty <= 1:
            raise ValueError("novelty must lie in [-1, 1].")
        if not math.isfinite(read_weight) or read_weight < 0:
            raise ValueError("read_weight must be finite and nonnegative.")
        self.device = torch.device(device)
        if self.device.type == "cpu":
            if not allow_cpu_test:
                raise RuntimeError("CPU is allowed only with allow_cpu_test=True.")
        elif self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU fallback is disabled.")
        self.dim = int(dim)
        self.max_cells = int(max_cells)
        self.novelty = float(novelty)
        self.read_weight = float(read_weight)
        self.n = 0
        self.cell_count = 0
        self._capacity = 0
        self._keys = torch.empty((0, self.dim), device=self.device)
        self._record_ids = np.empty(0, dtype=np.int64)
        self._labels = np.empty(0, dtype=np.int64)
        self._id_to_position: dict[int, int] = {}
        self._members: list[set[int]] = [set() for _ in range(self.max_cells)]
        self._centers = torch.zeros((self.max_cells, self.dim), device=self.device)
        self._sums = torch.zeros_like(self._centers)
        self._counts = np.zeros(self.max_cells, dtype=np.int64)
        self._offsets = np.zeros(1, dtype=np.int64)
        self._maintenance_cursor = 0
        self.work = {name: 0.0 for name in ("build", "query", "maintenance", "audit")}
        self.seconds = {name: 0.0 for name in self.work}
        self.work_breakdown = {
            name: self._blank_cost() for name in self.work
        }
        self.growth_curve: list[dict[str, int | float | None]] = []
        self.maintenance_calls = {"stale": 0, "full": 0, "incremental": 0}

    @staticmethod
    def _blank_cost() -> dict[str, float]:
        return {"arithmetic": 0.0, "reads": 0.0, "writes": 0.0, "index_ops": 0.0}

    @staticmethod
    def _add(cost: dict[str, float], **increments: float) -> None:
        for name, value in increments.items():
            cost[name] += float(value)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _finish(self, phase: str, cost: dict[str, float], started: float) -> tuple[float, float]:
        self._sync()
        seconds = time.perf_counter() - started
        total = (
            cost["arithmetic"] + cost["index_ops"]
            + self.read_weight * (cost["reads"] + cost["writes"])
        )
        self.work[phase] += total
        self.seconds[phase] += seconds
        for key, value in cost.items():
            self.work_breakdown[phase][key] += value
        return total, seconds

    @property
    def keys(self) -> torch.Tensor:
        return self._keys[: self.n]

    @property
    def counts(self) -> np.ndarray:
        return self._counts[: self.cell_count].copy()

    @property
    def offsets(self) -> np.ndarray:
        return self._offsets.copy()

    @property
    def record_cluster(self) -> np.ndarray:
        return self._labels[: self.n].copy()

    @property
    def record_ids(self) -> np.ndarray:
        return self._record_ids[: self.n].copy()

    @property
    def centroids(self) -> torch.Tensor:
        return self._centers[: self.cell_count]

    @property
    def active_cells(self) -> np.ndarray:
        return np.flatnonzero(self._counts[: self.cell_count] > 0)

    def _ensure_capacity(self, required: int, cost: dict[str, float]) -> None:
        if required <= self._capacity:
            return
        capacity = max(64, required, 2 * self._capacity)
        keys = torch.empty((capacity, self.dim), device=self.device)
        ids = np.empty(capacity, dtype=np.int64)
        labels = np.empty(capacity, dtype=np.int64)
        if self.n:
            keys[: self.n].copy_(self._keys[: self.n])
            ids[: self.n] = self._record_ids[: self.n]
            labels[: self.n] = self._labels[: self.n]
        self._add(
            cost, reads=self.n * (self.dim + 2),
            writes=self.n * (self.dim + 2), index_ops=1,
        )
        self._keys, self._record_ids, self._labels = keys, ids, labels
        self._capacity = capacity

    def _refresh_centers(self, cells: np.ndarray, cost: dict[str, float]) -> None:
        if not len(cells):
            return
        indices = torch.as_tensor(cells, device=self.device)
        sums = self._sums.index_select(0, indices)
        norms = torch.linalg.vector_norm(sums, dim=1)
        previous = self._centers.index_select(0, indices)
        centers = torch.where(
            (norms > 1e-12)[:, None], sums / norms.clamp_min(1e-12)[:, None], previous
        )
        self._centers.index_copy_(0, indices, centers)
        self._add(
            cost, arithmetic=3 * len(cells) * self.dim + len(cells),
            reads=4 * len(cells) * self.dim + len(cells),
            writes=3 * len(cells) * self.dim + len(cells),
        )

    def _refresh_offsets(self, cost: dict[str, float]) -> None:
        self._offsets = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(self._counts[: self.cell_count])]
        )
        self._add(
            cost, arithmetic=self.cell_count, reads=self.cell_count,
            writes=self.cell_count + 1,
        )

    def _score_centers(
        self, vectors: torch.Tensor, cost: dict[str, float]
    ) -> tuple[torch.Tensor, np.ndarray]:
        active = self.active_cells
        index = torch.as_tensor(active, device=self.device)
        centers = self._centers.index_select(0, index)
        scores = vectors @ centers.T
        rows = vectors.shape[0]
        self._add(
            cost, arithmetic=2 * rows * self.dim * len(active),
            reads=rows * self.dim + 2 * len(active) * self.dim + len(active),
            writes=rows * len(active) + len(active) * self.dim,
        )
        return scores, active

    def observe(self, keys: np.ndarray, record_ids: np.ndarray) -> ObserveResult:
        self._sync()
        started = time.perf_counter()
        cost = self._blank_cost()
        source = np.array(keys, dtype=np.float32, copy=True)
        ids = np.asarray(record_ids)
        if source.ndim != 2 or source.shape[1] != self.dim or len(source) == 0:
            raise ValueError("keys must have nonempty shape [rows, dim].")
        if ids.shape != (len(source),) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("record_ids must contain one integer per vector.")
        ids = ids.astype(np.int64, copy=True)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Duplicate record IDs within an observation batch.")
        if any(int(rid) in self._id_to_position for rid in ids):
            raise ValueError("An observed record ID cannot be inserted twice.")
        if not np.isfinite(source).all() or np.any(np.linalg.norm(source, axis=1) <= 0):
            raise ValueError("Observed vectors must be finite and nonzero.")
        rows = len(source)
        self._add(
            cost, arithmetic=4 * rows * self.dim,
            reads=4 * rows * self.dim + 3 * rows,
            writes=2 * rows * self.dim + rows,
            index_ops=2 * rows + rows * math.log2(max(2, rows)),
        )
        self._ensure_capacity(self.n + rows, cost)
        new_cell = None
        with torch.no_grad():
            observed = F.normalize(torch.as_tensor(source, device=self.device), dim=1)
            self._add(
                cost, arithmetic=3 * rows * self.dim,
                reads=2 * rows * self.dim, writes=2 * rows * self.dim,
            )
            if self.cell_count == 0:
                self.cell_count = 1
                new_cell = 0
                self._centers[0] = observed[0]
                labels = torch.zeros(rows, device=self.device, dtype=torch.int64)
                worst_similarity = None
                self._add(cost, reads=self.dim, writes=self.dim + rows)
            else:
                scores, active = self._score_centers(observed, cost)
                best_scores, local_labels = scores.max(dim=1)
                worst_index = int(torch.argmin(best_scores).item())
                worst_similarity = float(best_scores[worst_index].item())
                active_gpu = torch.as_tensor(active, device=self.device)
                labels = active_gpu[local_labels]
                dormant = np.flatnonzero(self._counts[: self.cell_count] == 0)
                can_grow = len(dormant) > 0 or self.cell_count < self.max_cells
                self._add(
                    cost, arithmetic=rows * len(active) + rows,
                    reads=rows * len(active) + 3 * rows + self.cell_count,
                    writes=3 * rows, index_ops=self.cell_count,
                )
                if worst_similarity < self.novelty and can_grow:
                    if len(dormant):
                        new_cell = int(dormant[0])
                        self._sums[new_cell].zero_()
                    else:
                        new_cell = self.cell_count
                        self.cell_count += 1
                    self._centers[new_cell] = observed[worst_index]
                    added_scores = observed @ self._centers[new_cell]
                    labels = torch.where(added_scores > best_scores, new_cell, labels)
                    self._add(
                        cost, arithmetic=2 * rows * self.dim + rows,
                        reads=rows * self.dim + 2 * self.dim + 3 * rows,
                        writes=self.dim + 2 * rows,
                    )
            begin, end = self.n, self.n + rows
            self._keys[begin:end].copy_(observed)
            label_cpu = labels.cpu().numpy()
            self._record_ids[begin:end] = ids
            self._labels[begin:end] = label_cpu
            for position, rid, cell in zip(range(begin, end), ids, label_cpu):
                self._id_to_position[int(rid)] = position
                self._members[int(cell)].add(position)
            self._sums.index_add_(0, labels, observed)
            self._counts[: self.cell_count] += np.bincount(
                label_cpu, minlength=self.cell_count
            )
            self.n = end
            self._add(
                cost, arithmetic=rows * self.dim + rows + self.cell_count,
                reads=3 * rows * self.dim + 5 * rows,
                writes=2 * rows * self.dim + 5 * rows + self.cell_count,
                index_ops=2 * rows,
            )
            self._refresh_centers(np.unique(label_cpu), cost)
            self._refresh_offsets(cost)
        work, seconds = self._finish("build", cost, started)
        self.growth_curve.append(
            {
                "observed": self.n, "batch_rows": rows, "cells": self.cell_count,
                "new_cell": new_cell, "worst_prior_cosine": worst_similarity,
            }
        )
        return ObserveResult(rows, self.n, self.cell_count, new_cell, work, seconds)

    def route(self, query: np.ndarray, k: int) -> RouteResult:
        if not self.n:
            raise RuntimeError("Cannot query an empty directory.")
        if int(k) != k or k < 1:
            raise ValueError("k is a positive count of directory cells.")
        self._sync()
        started = time.perf_counter()
        cost = self._blank_cost()
        source = np.array(query, dtype=np.float32, copy=True)
        if source.shape != (self.dim,) or not np.isfinite(source).all():
            raise ValueError("Query shape or values are invalid.")
        if np.linalg.norm(source) <= 0:
            raise ValueError("Query vector cannot be zero.")
        with torch.no_grad():
            q = F.normalize(torch.as_tensor(source, device=self.device), dim=0)
            self._add(
                cost, arithmetic=6 * self.dim,
                reads=4 * self.dim, writes=2 * self.dim,
            )
            scores, active = self._score_centers(q[None], cost)
            score_cpu = scores[0].cpu().numpy()
            ranked = np.argsort(-score_cpu, kind="stable")[: min(int(k), len(active))]
            selected_cells = active[ranked]
            positions = np.asarray(
                sorted(position for cell in selected_cells for position in self._members[int(cell)]),
                dtype=np.int64,
            )
            count = len(positions)
            candidate_gpu = torch.as_tensor(positions, device=self.device)
            candidate_keys = self._keys.index_select(0, candidate_gpu)
            similarities = candidate_keys @ q
            winner = int(torch.argmax(similarities).item())
            similarity = float(similarities[winner].item())
            best_id = int(self._record_ids[positions[winner]])
            candidate_ids = self._record_ids[positions].copy()
            self._add(
                cost, arithmetic=2 * count * self.dim + count,
                reads=2 * count * self.dim + 5 * count + len(active),
                writes=count * self.dim + 3 * count,
                index_ops=len(active) * math.log2(max(2, len(active)))
                + count * math.log2(max(2, count)) + len(selected_cells),
            )
        work, seconds = self._finish("query", cost, started)
        return RouteResult(
            candidate_ids, selected_cells.copy(), best_id, similarity, work, seconds
        )

    def maintain(self, mode: str, budget: int | None = None) -> MaintenanceResult:
        aliases = {
            "stale": "stale", "full": "full", "periodic": "full",
            "incremental": "incremental", "bounded": "incremental",
        }
        if mode not in aliases:
            raise ValueError("Maintenance mode must be stale, full, or incremental.")
        mode = aliases[mode]
        if budget is not None and (int(budget) != budget or budget < 0):
            raise ValueError("budget must be a nonnegative record count.")
        if mode == "incremental" and budget is None:
            raise ValueError("Incremental maintenance requires a record budget.")
        budget = int(budget) if budget is not None else None
        self.maintenance_calls[mode] += 1
        if mode == "stale" or not self.n or budget == 0:
            return MaintenanceResult(mode, False, 0, 0, budget, 0.0, 0.0)
        if mode == "full" and budget is not None and budget < self.n:
            return MaintenanceResult(mode, False, 0, 0, budget, 0.0, 0.0)
        self._sync()
        started = time.perf_counter()
        cost = self._blank_cost()
        count = self.n if mode == "full" else min(int(budget), self.n)
        if mode == "full":
            positions = np.arange(self.n, dtype=np.int64)
        else:
            positions = (np.arange(count, dtype=np.int64) + self._maintenance_cursor) % self.n
            self._maintenance_cursor = (self._maintenance_cursor + count) % self.n
        previous = self._labels[positions].copy()
        with torch.no_grad():
            indices = torch.as_tensor(positions, device=self.device)
            observed = self._keys.index_select(0, indices)
            scores, active = self._score_centers(observed, cost)
            nearest = scores.argmax(dim=1).cpu().numpy()
            current = active[nearest]
            changed_mask = current != previous
            changed_positions = positions[changed_mask]
            changed_old = previous[changed_mask]
            changed_new = current[changed_mask]
            changed = len(changed_positions)
            self._add(
                cost, arithmetic=count * len(active) + 2 * count,
                reads=count * self.dim + count * len(active) + 5 * count,
                writes=count * self.dim + 4 * count, index_ops=count,
            )
            if changed:
                changed_index = torch.as_tensor(changed_positions, device=self.device)
                moved = self._keys.index_select(0, changed_index)
                old_gpu = torch.as_tensor(changed_old, device=self.device)
                new_gpu = torch.as_tensor(changed_new, device=self.device)
                self._sums.index_add_(0, old_gpu, -moved)
                self._sums.index_add_(0, new_gpu, moved)
                for position, old, new in zip(changed_positions, changed_old, changed_new):
                    self._members[int(old)].remove(int(position))
                    self._members[int(new)].add(int(position))
                self._labels[changed_positions] = changed_new
                self._counts[: self.cell_count] -= np.bincount(
                    changed_old, minlength=self.cell_count
                )
                self._counts[: self.cell_count] += np.bincount(
                    changed_new, minlength=self.cell_count
                )
                touched = np.unique(np.concatenate([changed_old, changed_new]))
                dormant = touched[self._counts[touched] == 0]
                if len(dormant):
                    dormant_gpu = torch.as_tensor(dormant, device=self.device)
                    self._sums.index_fill_(0, dormant_gpu, 0.0)
                self._refresh_centers(touched, cost)
                self._add(
                    cost, arithmetic=3 * changed * self.dim + 2 * changed,
                    reads=5 * changed * self.dim + 6 * changed,
                    writes=4 * changed * self.dim + 5 * changed
                    + len(dormant) * self.dim,
                    index_ops=2 * changed + len(touched) * math.log2(max(2, len(touched))),
                )
            self._refresh_offsets(cost)
        work, seconds = self._finish("maintenance", cost, started)
        return MaintenanceResult(mode, True, count, changed, budget, work, seconds)

    def snapshot(self) -> dict[str, object]:
        """Return an auditable copy; packing/transfer costs are in work['audit']."""
        self._sync()
        started = time.perf_counter()
        cost = self._blank_cost()
        packed = np.asarray(
            [
                position for cell in range(self.cell_count)
                for position in sorted(self._members[cell])
            ],
            dtype=np.int64,
        )
        self._add(
            cost, reads=6 * self.n + 2 * self.cell_count * self.dim,
            writes=6 * self.n + 2 * self.cell_count * self.dim,
            index_ops=sum(
                len(cell) * math.log2(max(2, len(cell)))
                for cell in self._members[: self.cell_count]
            ),
        )
        state: dict[str, object] = {
            "observed": self.n,
            "cells": self.cell_count,
            "record_ids": self.record_ids,
            "membership": self.record_cluster,
            "counts": self.counts,
            "offsets": self.offsets,
            "packed_record_ids": self._record_ids[packed].copy(),
            "packed_positions": packed,
            "centroids": self.centroids.detach().cpu().numpy().copy(),
            "running_sums": self._sums[: self.cell_count].detach().cpu().numpy().copy(),
            "active_cells": self.active_cells,
            "maintenance_cursor": self._maintenance_cursor,
            "maintenance_calls": dict(self.maintenance_calls),
        }
        self._finish("audit", cost, started)
        state["work"] = dict(self.work)
        state["seconds"] = dict(self.seconds)
        state["work_breakdown"] = {
            phase: dict(breakdown) for phase, breakdown in self.work_breakdown.items()
        }
        return state
