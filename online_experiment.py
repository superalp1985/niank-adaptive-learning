"""Sequential observations, independently scored answers, and charged maintenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F

from online_benchmark import Answer, BranchingWorld, Fact, OnlineSession, Query, make_branching_world
from online_directory import OnlineDirectory, RouteResult


ROOT = Path(__file__).resolve().parent
DEVELOPMENT_SEEDS = (911, 919, 929)
VALIDATION_SEEDS = (1009, 1013)
TEST_SEEDS = (1103, 1109, 1117, 1123, 1129)
STREAMS = ("mixed", "shift", "return")
STAGES = 24
BATCH_SIZE = 64
QUESTIONS_PER_STAGE = 6


def save_json(location: Path, value: object) -> None:
    temporary = location.with_suffix(location.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(location)


def source_hashes() -> dict[str, str]:
    sources = [
        ROOT / "online_experiment.py", ROOT / "online_directory.py",
        ROOT / "ONLINE_PROTOCOL.md",
        *sorted((ROOT / "online_benchmark").glob("*.py")),
    ]
    return {str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sources}


@dataclass(frozen=True)
class Method:
    name: str
    kind: str
    probes: int = 1


def candidate_methods() -> list[Method]:
    return [Method("dense", "dense"), Method("symbolic", "symbolic")] + [
        Method(f"{kind}_{probes}", kind, probes)
        for kind in ("stale", "periodic", "incremental")
        for probes in (1, 2, 4)
    ]


@dataclass(frozen=True)
class StreamTask:
    world: BranchingWorld
    batches: tuple[tuple[Fact, ...], ...]
    questions: tuple[tuple[Query, ...], ...]
    stream: str
    dim: int


def make_stream(
    seed: int, stream: str, stages: int = STAGES,
    questions_per_stage: int = QUESTIONS_PER_STAGE, dim: int = 48,
) -> StreamTask:
    if stream not in STREAMS or stages < 1 or questions_per_stage < 1:
        raise ValueError("Invalid stream settings.")
    base = make_branching_world(seed, n_entities=256, n_relations=4, dim=dim, query_count=1)
    address_keys = {(fact.source, fact.relation): fact.key for fact in base.facts}
    rng = np.random.default_rng(np.random.SeedSequence((seed, 8521)))
    # Opaque IDs carry no insertion order, target, entity or relation encoding.
    ids = rng.choice(2**31 - 1001, size=stages * BATCH_SIZE, replace=False) + 1001
    facts: list[Fact] = []
    batches: list[tuple[Fact, ...]] = []
    queries: list[tuple[Query, ...]] = []
    learned_problems: list[tuple[int, tuple[int, int], int]] = []
    registry: dict[tuple[int, int], list[Fact]] = {}

    def append_fact(source: int, relation: int, target: int) -> Fact:
        observed = len(facts)
        fact = Fact(
            int(ids[observed]), int(source), int(relation), int(target), observed,
            float(np.clip(rng.lognormal(0, .3), .5, 2)),
            address_keys[(int(source), int(relation))],
        )
        facts.append(fact)
        registry.setdefault((fact.source, fact.relation), []).append(fact)
        return fact

    for stage in range(stages):
        block: list[Fact] = []
        fresh: list[tuple[int, tuple[int, int], int]] = []
        for episode in range(8):
            phase = min(2, 3 * stage // stages)
            if stream == "mixed":
                topic = int(rng.integers(4))
            elif stream == "shift":
                topic = phase
            else:
                topic = 0 if phase in (0, 2) else 1
            start = int(rng.integers(16 * topic, 16 * topic + 16))
            relation0, relation1 = (int(r) for r in rng.integers(4, size=2))
            middles = rng.choice(np.arange(64, 192), size=2, replace=False)
            for middle in middles:
                block.append(append_fact(start, relation0, int(middle)))
                for terminal in rng.choice(256, size=2, replace=False):
                    block.append(append_fact(int(middle), relation1, int(terminal)))
            fresh.append((start, (relation0, relation1), stage))
        for _ in range(16):
            block.append(append_fact(
                int(rng.integers(192, 256)), int(rng.integers(4)), int(rng.integers(256))
            ))
        if len(block) != BATCH_SIZE:
            raise AssertionError("Observation batch must contain 64 facts.")
        learned_problems.extend(fresh)
        batches.append(tuple(block))
        choices = [
            fresh[0],
            fresh[1],
            learned_problems[0],
            learned_problems[min(1, len(learned_problems) - 1)],
        ]
        choices.extend(
            learned_problems[int(rng.integers(len(learned_problems)))]
            for _ in range(max(0, questions_per_stage - len(choices)))
        )
        stage_queries = []
        for index, (start, relations, born) in enumerate(choices[:questions_per_stage]):
            terminals = {
                second.target
                for first in registry.get((start, relations[0]), ())
                for second in registry.get((first.target, relations[1]), ())
            }
            if not terminals:
                raise AssertionError("Generated observed problem has no valid path.")
            stage_queries.append(Query(
                id=f"{seed}:{stream}:stage{stage}:q{index}",
                start=start,
                relations=relations,
                goals=frozenset(terminals),
                as_of=len(facts) - 1,
                problem_id=f"{seed}:{stream}:{start}:{relations}:{born}",
                noise=.08,
                seed=int(rng.integers(2**32)),
                goal_description="any entity reachable through these two relations",
            ))
        queries.append(tuple(stage_queries))
    # The environment owns the encoder. Policies see only its current vector.
    world = BranchingWorld(facts, (), encoder=base._encoder, seed=seed)
    return StreamTask(world, tuple(batches), tuple(queries), stream, dim)


class ObservedSqlite:
    """Only observe() can add facts; no full-world registry is held here."""

    def __init__(self, location: Path, symbolic_index: bool = False) -> None:
        location.parent.mkdir(parents=True, exist_ok=True)
        if location.exists():
            raise FileExistsError(f"Refusing to reuse a populated experiment DB: {location}")
        self.location = location
        self.connection = sqlite3.connect(location)
        self.connection.execute(
            "CREATE TABLE facts (id INTEGER PRIMARY KEY, source INTEGER, relation INTEGER,"
            " target INTEGER, observed INTEGER, cost REAL)"
        )
        if symbolic_index:
            self.connection.execute("CREATE INDEX by_address ON facts(source, relation, id)")
        self.connection.commit()
        self.symbolic_index = symbolic_index
        self.n = 0
        self.work = {"write": 0.0, "read": 0.0}
        self.seconds = {"write": 0.0, "read": 0.0}
        self.read_ids: list[int] = []

    def observe(self, facts: Iterable[Fact]) -> float:
        batch = tuple(facts)
        started = time.perf_counter()
        with self.connection:
            self.connection.executemany(
                "INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?)",
                [(f.id, f.source, f.relation, f.target, f.observed_at, f.read_cost)
                 for f in batch],
            )
        cost = sum(
            12 + math.log2(max(2, self.n + i + 1))
            + (math.log2(max(2, self.n + i + 1)) + 4 if self.symbolic_index else 0)
            for i in range(len(batch))
        )
        self.n += len(batch)
        self.work["write"] += cost
        self.seconds["write"] += time.perf_counter() - started
        return cost

    def read(self, rid: int, as_of: int) -> tuple[int, int, int, int]:
        started = time.perf_counter()
        row = self.connection.execute(
            "SELECT source, relation, target, observed, cost FROM facts WHERE id = ?",
            (int(rid),),
        ).fetchone()
        self.seconds["read"] += time.perf_counter() - started
        if row is None or row[3] > as_of:
            raise KeyError(f"Unobserved record {rid} at time {as_of}")
        self.work["read"] += math.log2(max(2, self.n)) + 10 + row[4]
        self.read_ids.append(int(rid))
        return tuple(int(value) for value in row[:4])

    def lookup(self, source: int, relation: int, as_of: int) -> int | None:
        if not self.symbolic_index:
            raise RuntimeError("Symbolic lookup requires its explicitly charged index.")
        started = time.perf_counter()
        row = self.connection.execute(
            "SELECT id FROM facts WHERE source=? AND relation=? AND observed<=?"
            " ORDER BY id LIMIT 1", (source, relation, as_of),
        ).fetchone()
        self.seconds["read"] += time.perf_counter() - started
        self.work["read"] += math.log2(max(2, self.n)) + 6
        return None if row is None else int(row[0])

    def close(self) -> None:
        self.connection.close()


class DenseObservedKeys:
    def __init__(self, dim: int, device: str, allow_cpu_test: bool = False) -> None:
        self.device = torch.device(device)
        if self.device.type == "cpu" and not allow_cpu_test:
            raise RuntimeError("CPU requires explicit small-test opt-in.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; no CPU fallback.")
        self.dim = dim
        self.n = 0
        self.capacity = 0
        self.keys = torch.empty((0, dim), device=self.device)
        self.ids = np.empty(0, dtype=np.int64)
        self.work = {"build": 0.0, "query": 0.0, "maintenance": 0.0}
        self.build_seconds = 0.0

    def observe(self, keys: np.ndarray, ids: np.ndarray) -> float:
        started = time.perf_counter()
        rows = len(keys)
        if np.asarray(keys).shape != (rows, self.dim) or not np.isfinite(keys).all():
            raise ValueError("Invalid dense keys.")
        if np.any(np.linalg.norm(keys, axis=1) <= 0) or len(np.unique(ids)) != rows:
            raise ValueError("Invalid dense observations.")
        required = self.n + rows
        copy_work = 0.0
        if required > self.capacity:
            new_capacity = max(64, required, 2 * self.capacity)
            storage = torch.empty((new_capacity, self.dim), device=self.device)
            id_storage = np.empty(new_capacity, dtype=np.int64)
            storage[:self.n].copy_(self.keys[:self.n])
            id_storage[:self.n] = self.ids[:self.n]
            copy_work = 2 * self.n * (self.dim + 1)
            self.keys, self.ids, self.capacity = storage, id_storage, new_capacity
        observed = F.normalize(
            torch.as_tensor(np.array(keys, dtype=np.float32, copy=True), device=self.device),
            dim=1,
        )
        self.keys[self.n:required].copy_(observed)
        self.ids[self.n:required] = ids
        self.n = required
        work = 18 * rows * self.dim + 9 * rows + rows * math.log2(max(2, rows)) + copy_work
        self.work["build"] += work
        self.build_seconds += time.perf_counter() - started
        return work

    def route(self, query: np.ndarray, k: int = 1) -> RouteResult:
        started = time.perf_counter()
        q = F.normalize(
            torch.as_tensor(np.array(query, dtype=np.float32, copy=True), device=self.device),
            dim=0,
        )
        similarities = self.keys[:self.n] @ q
        position = int(similarities.argmax().item())
        similarity = float(similarities[position].item())
        work = 12 * self.dim + 3 * self.n * self.dim + 2 * self.n
        self.work["query"] += work
        return RouteResult(self.ids[:self.n].copy(), np.asarray([], dtype=np.int64),
                           int(self.ids[position]), similarity, work,
                           time.perf_counter() - started)


def _method_router(method: Method, dim: int, device: str, cpu_test: bool):
    if method.kind in ("dense", "symbolic"):
        return DenseObservedKeys(dim, device, cpu_test)
    if method.kind not in ("stale", "periodic", "incremental"):
        raise ValueError(method.kind)
    return OnlineDirectory(
        dim, max_cells=32, novelty=.7, device=device, allow_cpu_test=cpu_test
    )


def execute_answer(
    query: Query, vector_provider, router, store: ObservedSqlite, method: Method,
) -> tuple[Answer, dict[str, object]]:
    """No world, target set, reference or score is consulted by this executor."""
    current = query.start
    support: list[int] = []
    hops = []
    route_work = 0.0
    db_before = store.work["read"]
    started = time.perf_counter()
    for step, relation in enumerate(query.relations):
        if method.kind == "symbolic":
            rid = store.lookup(current, relation, query.as_of)
            hop = {"source": current, "relation": relation, "selected_record": rid,
                   "candidate_count": None, "similarity": None, "route_work": 0.0}
        else:
            qvector = vector_provider(current, relation, step)
            route_work += len(qvector) * 8
            result = router.route(qvector, method.probes)
            route_work += result.work
            rid = result.best_id
            hop = {"source": current, "relation": relation, "selected_record": rid,
                   "candidate_count": result.candidate_count,
                   "similarity": result.similarity, "route_work": result.work}
        if rid is None:
            break
        source, found_relation, target, observed = store.read(rid, query.as_of)
        # Retrieved payloads are followed as returned. Independent verification
        # later rejects wrong addresses; there is no dense repair or oracle retry.
        support.append(int(rid))
        current = target
        hop["payload"] = {"source": source, "relation": found_relation,
                          "target": target, "observed": observed}
        hops.append(hop)
    answer = Answer(int(current), tuple(support))
    return answer, {
        "route_work": route_work,
        "database_read_work": store.work["read"] - db_before,
        "answer_work": 16 * len(support) + 8,
        "seconds": time.perf_counter() - started,
        "hops": hops,
    }


def run_method(
    task: StreamTask, method: Method, output: Path, *, device: str = "cuda",
    allow_cpu_test: bool = False,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    router = _method_router(method, task.dim, device, allow_cpu_test)
    store = ObservedSqlite(output / f"{method.name}.sqlite", method.kind == "symbolic")
    session = OnlineSession(task.world.facts, feedback_enabled=False)
    stage_rows = []
    answer_work = 0.0
    encoder_work = 0.0
    records_written = 0
    observed_ids: set[int] = set()
    started = time.perf_counter()
    try:
        with (output / f"{method.name}.answers.jsonl").open("x", encoding="utf-8") as log:
            for stage, (batch, questions) in enumerate(zip(task.batches, task.questions), 1):
                stage_begin = time.perf_counter()
                store.observe(batch)
                observable = [session.observe_fact(fact) for fact in batch]
                observed_ids.update(item.id for item in observable)
                router.observe(
                    np.asarray([item.key for item in observable], dtype=np.float32),
                    np.asarray([item.id for item in observable], dtype=np.int64),
                )
                stage_correct = 0
                stage_terminal = 0
                stage_query_work = 0.0
                for query in questions:
                    session.ask(query)
                    answer, costs = execute_answer(
                        query.observable(),
                        lambda current, relation, step, q=query: task.world.query_vector(
                            current, relation, q, step
                        ),
                        router, store, method,
                    )
                    session.commit_answer(
                        query.id, answer,
                        route_cost=costs["route_work"] + costs["answer_work"],
                        database_cost=costs["database_read_work"],
                        inference_seconds=costs["seconds"],
                    )
                    score = session.score(query.id).score
                    stage_correct += int(score.full_correct)
                    stage_terminal += int(score.terminal_correct)
                    stage_query_work += (
                        costs["route_work"] + costs["database_read_work"] + costs["answer_work"]
                    )
                    answer_work += costs["answer_work"]
                    encoder_work += 0 if method.kind == "symbolic" else task.dim * 8 * len(costs["hops"])
                    references = task.world.reference(query)
                    alternatives = references[:2]
                    row = {
                        "method": method.name, "seed": task.world.seed, "stream": task.stream,
                        "stage": stage, "observed_facts": len(observed_ids),
                        "query_id": query.id,
                        "question": task.world.render_question(query),
                        "answer": task.world.render_answer(query, answer),
                        "committed": asdict(answer), "score": asdict(score),
                        "valid_answer_count": len(references),
                        "valid_alternatives": [
                            task.world.render_answer(query, Answer(proof.terminal, proof.support_ids))
                            for proof in alternatives
                        ],
                        "costs": costs,
                    }
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
                    records_written += 1
                maintenance = None
                if method.kind == "periodic" and stage % 4 == 0:
                    maintenance = router.maintain("full", router.n)
                elif method.kind == "incremental":
                    maintenance = router.maintain("incremental", BATCH_SIZE)
                stage_rows.append({
                    "stage": stage, "facts": store.n, "questions": len(questions),
                    "correct": stage_correct, "accuracy": stage_correct / len(questions),
                    "terminal_correct": stage_terminal,
                    "query_work": stage_query_work,
                    "cumulative_build_work": router.work["build"],
                    "cumulative_maintenance_work": router.work["maintenance"],
                    "cumulative_sql_write_work": store.work["write"],
                    "maintenance": asdict(maintenance) if maintenance else None,
                    "seconds": time.perf_counter() - stage_begin,
                })
                log.flush()
        components = {
            "build": router.work["build"],
            "query": router.work["query"],
            "maintenance": router.work["maintenance"],
            "encoder": encoder_work,
            "answer": answer_work,
            "sql_write": store.work["write"],
            "sql_read": store.work["read"],
        }
        summary = {
            "method": asdict(method), "seed": task.world.seed, "stream": task.stream,
            "questions": records_written,
            "correct": sum(row["correct"] for row in stage_rows),
            "accuracy": sum(row["correct"] for row in stage_rows) / records_written,
            "total_work": sum(components.values()), "components": components,
            "seconds": time.perf_counter() - started,
            "database_bytes": store.location.stat().st_size,
            "database_seconds": store.seconds,
            "stages": stage_rows,
            "observed_rows": store.n,
            "full_answer_evaluation": "environment proof constraints, never dense agreement",
        }
        save_json(output / f"{method.name}.summary.json", summary)
        return summary
    finally:
        store.close()


def aggregate(summaries: list[dict[str, object]]) -> dict[str, object]:
    questions = sum(item["questions"] for item in summaries)
    correct = sum(item["correct"] for item in summaries)
    streams = {
        stream: [row for row in summaries if row["stream"] == stream]
        for stream in STREAMS
    }
    return {
        "questions": questions, "correct": correct,
        "accuracy": correct / questions,
        "total_work": sum(item["total_work"] for item in summaries),
        "seconds": sum(item["seconds"] for item in summaries),
        "streams": {
            stream: sum(item["correct"] for item in rows) / sum(item["questions"] for item in rows)
            for stream, rows in streams.items() if rows
        },
        "worlds": summaries,
    }


def select_methods(validation: dict[str, dict[str, object]]) -> dict[str, object]:
    dense = validation["dense"]
    selected, gates = ["dense", "symbolic"], {}
    for kind in ("stale", "periodic", "incremental"):
        candidates = [f"{kind}_{p}" for p in (1, 2, 4)]
        passing = [
            name for name in candidates
            if validation[name]["accuracy"] >= dense["accuracy"] - .01 - 1e-12
            and all(validation[name]["streams"][stream] >= accuracy - .03 - 1e-12
                    for stream, accuracy in dense["streams"].items())
        ]
        if passing:
            chosen = min(passing, key=lambda name: validation[name]["total_work"])
        else:
            chosen = min(candidates, key=lambda name: (-validation[name]["accuracy"],
                                                      validation[name]["total_work"]))
        selected.append(chosen)
        gates[kind] = bool(passing)
    return {"selected_names": selected, "quality_gate": gates}


def run_split(
    output: Path, seeds: Iterable[int], methods: list[Method], *,
    stages: int = STAGES, questions_per_stage: int = QUESTIONS_PER_STAGE,
    device: str = "cuda", allow_cpu_test: bool = False,
) -> dict[str, dict[str, object]]:
    output.mkdir(parents=True, exist_ok=True)
    summaries = {method.name: [] for method in methods}
    for seed in seeds:
        for stream in STREAMS:
            task = make_stream(seed, stream, stages, questions_per_stage)
            world_output = output / f"{seed}_{stream}"
            for method in methods:
                result = run_method(
                    task, method, world_output, device=device, allow_cpu_test=allow_cpu_test
                )
                summaries[method.name].append(result)
                save_json(output / "progress.json", {
                    "seed": seed, "stream": stream, "method": method.name,
                    "accuracy": result["accuracy"], "seconds": result["seconds"],
                })
    result = {name: aggregate(rows) for name, rows in summaries.items()}
    save_json(output / "results.json", result)
    return result


def pilot(output: str | Path, device: str = "cuda") -> dict[str, dict[str, object]]:
    return run_split(
        Path(output), (901,),
        [Method("dense", "dense"), Method("stale_2", "stale", 2),
         Method("periodic_2", "periodic", 2), Method("incremental_2", "incremental", 2),
         Method("symbolic", "symbolic")],
        stages=4, questions_per_stage=4, device=device,
    )


def develop_and_select(output: str | Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required, CPU fallback disabled.")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / "protocol_sources.json", source_hashes())
    methods = candidate_methods()
    run_split(root / "development", DEVELOPMENT_SEEDS, methods)
    validation = run_split(root / "validation", VALIDATION_SEEDS, methods)
    selected = select_methods(validation)
    selected["source_hashes"] = source_hashes()
    selected["validation_seeds"] = VALIDATION_SEEDS
    selected["test_seeds"] = TEST_SEEDS
    save_json(root / "selection.json", selected)
    return selected


def frozen_test(output: str | Path) -> dict[str, dict[str, object]]:
    root = Path(output)
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    if selection["source_hashes"] != source_hashes():
        raise RuntimeError("Sources changed after validation selection.")
    names = set(selection["selected_names"])
    methods = [method for method in candidate_methods() if method.name in names]
    return run_split(root / "test", TEST_SEEDS, methods)
