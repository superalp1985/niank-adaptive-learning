"""Branching proofs and an explicit observe/answer/feedback event protocol.

The benchmark world and session belong to the trusted experiment controller.
Policies receive only ObservableRecord, AskView, query vectors, and Feedback.
The private truth registry and evaluation events must never be passed to a
policy. This API is a data boundary, not a Python security sandbox.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class Fact:
    id: int
    source: int
    relation: int
    target: int
    observed_at: int
    read_cost: float = 1.0
    key: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(value, int) or value < 0 for value in (
            self.id, self.source, self.relation, self.target, self.observed_at
        )):
            raise ValueError("fact IDs, entities, relations, and times must be nonnegative ints")
        if not math.isfinite(self.read_cost) or self.read_cost < 0:
            raise ValueError("read_cost must be finite and nonnegative")
        if not isinstance(self.key, tuple) or not all(math.isfinite(x) for x in self.key):
            raise ValueError("key must be a finite immutable tuple")

    def observable(self) -> "ObservableRecord":
        return ObservableRecord(self.id, self.key, self.observed_at, self.read_cost)


@dataclass(frozen=True, slots=True)
class ObservableRecord:
    id: int
    key: tuple[float, ...]
    observed_at: int
    read_cost: float


@dataclass(frozen=True, slots=True)
class Query:
    id: str
    start: int
    relations: tuple[int, ...]
    goals: frozenset[int]
    as_of: int
    problem_id: str = ""
    noise: float = 0.08
    seed: int = 0
    simple_path: bool = False
    goal_description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("query id must be a nonempty string")
        if not isinstance(self.start, int) or self.start < 0:
            raise ValueError("query start must be a nonnegative int")
        if (
            not isinstance(self.relations, tuple)
            or not self.relations
            or not all(isinstance(r, int) and r >= 0 for r in self.relations)
        ):
            raise ValueError("relations must be a nonempty tuple of nonnegative ints")
        if (
            not isinstance(self.goals, frozenset)
            or not self.goals
            or not all(isinstance(g, int) and g >= 0 for g in self.goals)
        ):
            raise ValueError("goals must be a nonempty frozenset of nonnegative ints")
        if not isinstance(self.as_of, int) or self.as_of < 0:
            raise ValueError("as_of must be a nonnegative int")
        if not math.isfinite(self.noise) or self.noise < 0 or self.seed < 0:
            raise ValueError("noise and seed must be finite and nonnegative")
        if not self.problem_id:
            identity = (self.start, self.relations, sorted(self.goals), self.as_of,
                        self.simple_path)
            digest = hashlib.sha256(json.dumps(identity).encode("ascii")).hexdigest()[:24]
            object.__setattr__(self, "problem_id", digest)

    @property
    def goal(self) -> int:
        if len(self.goals) != 1:
            raise ValueError("this query has multiple accepted terminal entities")
        return next(iter(self.goals))

    def observable(self) -> "ObservableQuery":
        return ObservableQuery(
            self.id, self.start, self.relations, self.as_of,
            self.simple_path, self.goal_text,
        )

    @property
    def goal_text(self) -> str:
        if self.goal_description:
            return self.goal_description
        destinations = ", ".join(f"Entity {goal:04d}" for goal in sorted(self.goals))
        return f"one of the requested destinations: {destinations}"


@dataclass(frozen=True, slots=True)
class ObservableQuery:
    id: str
    start: int
    relations: tuple[int, ...]
    as_of: int
    simple_path: bool
    goal_description: str


@dataclass(frozen=True, slots=True)
class Answer:
    terminal: int | None
    support_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.terminal is not None and (
            not isinstance(self.terminal, int) or self.terminal < 0
        ):
            raise ValueError("terminal must be a nonnegative int or None for abstention")
        if not isinstance(self.support_ids, tuple):
            raise ValueError("support_ids must be an immutable tuple")
        if not all(isinstance(rid, int) for rid in self.support_ids):
            raise ValueError("support_ids must contain ints")


@dataclass(frozen=True, slots=True)
class Proof:
    terminal: int
    support_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ProofScore:
    terminal_correct: bool
    evidence_valid: bool
    full_correct: bool
    reason: str


class BranchingWorld:
    """Trusted, independent truth graph. No directory or learner is stored here."""

    def __init__(
        self,
        facts: Iterable[Fact],
        queries: Iterable[Query] = (),
        *,
        encoder: Any = None,
        seed: int = 0,
    ) -> None:
        self.facts = tuple(facts)
        registry = {fact.id: fact for fact in self.facts}
        if len(registry) != len(self.facts):
            raise ValueError("fact IDs must be unique")
        self._facts = MappingProxyType(registry)
        self.queries = tuple(queries)
        if len({q.id for q in self.queries}) != len(self.queries):
            raise ValueError("query presentation IDs must be unique")
        self.seed = seed
        self._encoder = encoder
        self._outgoing: dict[tuple[int, int], tuple[Fact, ...]] = {}
        buckets: dict[tuple[int, int], list[Fact]] = {}
        for fact in self.facts:
            buckets.setdefault((fact.source, fact.relation), []).append(fact)
        self._outgoing = {key: tuple(values) for key, values in buckets.items()}

    def fact(self, record_id: int) -> Fact:
        return self._facts[record_id]

    def visible_facts(self, as_of: int) -> tuple[Fact, ...]:
        return tuple(fact for fact in self.facts if fact.observed_at <= as_of)

    def score(
        self, query: Query, answer: Answer, visible: Iterable[Fact | int]
    ) -> ProofScore:
        """Validate the supplied proof directly; never call reference()."""
        terminal_correct = answer.terminal in query.goals
        observed = set()
        for item in visible:
            rid = item.id if isinstance(item, Fact) else item
            if isinstance(item, Fact) and self._facts.get(rid) != item:
                raise ValueError("visible record content differs from immutable truth")
            observed.add(rid)
        if answer.terminal is None:
            return ProofScore(False, False, False, "abstained")
        if len(answer.support_ids) != len(query.relations):
            return ProofScore(terminal_correct, False, False, "wrong_length")
        current = query.start
        visited = {current}
        for rid, relation in zip(answer.support_ids, query.relations):
            fact = self._facts.get(rid)
            if fact is None:
                return ProofScore(terminal_correct, False, False, "unknown_record")
            if fact.observed_at > query.as_of:
                return ProofScore(terminal_correct, False, False, "future_record")
            if rid not in observed:
                return ProofScore(terminal_correct, False, False, "unobserved_record")
            if fact.source != current:
                return ProofScore(terminal_correct, False, False, "disconnected")
            if fact.relation != relation:
                return ProofScore(terminal_correct, False, False, "wrong_relation")
            if query.simple_path and fact.target in visited:
                return ProofScore(terminal_correct, False, False, "cycle")
            current = fact.target
            visited.add(current)
        if current != answer.terminal:
            return ProofScore(terminal_correct, False, False, "terminal_mismatch")
        return ProofScore(
            terminal_correct, True, terminal_correct,
            "valid" if terminal_correct else "wrong_goal",
        )

    def reference(self, query: Query, max_proofs: int = 10000) -> tuple[Proof, ...]:
        """Evaluation-only enumeration, unsuitable as an inference API."""
        proofs: list[Proof] = []

        def visit(entity: int, step: int, path: tuple[int, ...], seen: frozenset[int]):
            if step == len(query.relations):
                if entity in query.goals:
                    if len(proofs) >= max_proofs:
                        raise ValueError("proof enumeration exceeded its explicit limit")
                    proofs.append(Proof(entity, path))
                return
            for fact in self._outgoing.get((entity, query.relations[step]), ()):
                if fact.observed_at > query.as_of:
                    continue
                if query.simple_path and fact.target in seen:
                    continue
                visit(fact.target, step + 1, path + (fact.id,), seen | {fact.target})

        visit(query.start, 0, (), frozenset((query.start,)))
        return tuple(proofs)

    def query_vector(
        self, current_entity: int, relation: int, query: Query, step: int
    ) -> np.ndarray:
        """Reuse the frozen controlled encoder, without reading branch targets."""
        if self._encoder is None:
            raise ValueError("this hand-authored world has no vector encoder")
        legacy = _legacy_benchmark()
        fixed_query = legacy.Query(
            query.id, query.start, query.relations, "online", query.noise, query.seed
        )
        return self._encoder.query_vector(current_entity, relation, fixed_query, step)

    def render_question(self, query: Query) -> str:
        chain = ", then ".join(f"relation {r}" for r in query.relations)
        constraint = " Use a simple path." if query.simple_path else ""
        return (
            f"At time {query.as_of}, start at Entity {query.start:04d} and follow "
            f"{chain} to reach {query.goal_text}. Return one valid terminal "
            f"entity and its ordered supporting records.{constraint}"
        )

    def render_answer(self, query: Query, answer: Answer) -> str:
        if answer.terminal is None:
            return "No answer was committed."
        parts = [f"The terminal entity is Entity {answer.terminal:04d}."]
        for step, rid in enumerate(answer.support_ids, 1):
            fact = self._facts.get(rid)
            if fact is None:
                parts.append(f"{step}. Record {rid} does not exist.")
            else:
                parts.append(
                    f"{step}. Record {rid}: Entity {fact.source:04d} -- relation "
                    f"{fact.relation} --> Entity {fact.target:04d}, observed at "
                    f"time {fact.observed_at}."
                )
        return " ".join(parts)


def _legacy_benchmark():
    """Read the existing frozen module by path without changing its source."""
    name = "_frozen_memory_efficiency_benchmark"
    if name not in sys.modules:
        source = (
            Path(__file__).resolve().parents[2]
            / "memory_efficiency_sim_20261011" / "benchmark.py"
        )
        spec = importlib.util.spec_from_file_location(name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load frozen benchmark from {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def make_branching_world(
    seed: int,
    n_entities: int = 64,
    n_relations: int = 4,
    dim: int = 48,
    query_count: int = 12,
) -> BranchingWorld:
    """Generate independent branch alternatives atop frozen vector geometry.

    Each generated problem has at least two legal paths with distinct terminal
    entities. Its public goal is any terminal reachable by the relation chain;
    the evaluator derives the accepted set from the independent graph. Keys
    encode only (source, relation), not branch targets, times, or record IDs.
    The fixed encoder is not semantic learning.
    """
    if seed < 0 or n_entities < 8 or n_relations < 2 or query_count < 1:
        raise ValueError("seed >= 0, entities >= 8, relations >= 2, queries >= 1 required")
    base = _legacy_benchmark().make_world(seed, n_entities, n_relations, dim)
    rng = np.random.default_rng(np.random.SeedSequence((seed, 301)))
    edges = [
        (rid // n_relations, rid % n_relations, int(target))
        for rid, target in enumerate(base.values)
    ]
    specifications = []
    for index in range(query_count):
        length = 2 + index % 2
        selected = rng.choice(n_entities, size=1 + 2 * length, replace=False)
        start = int(selected[0])
        relations = tuple(int(r) for r in rng.integers(n_relations, size=length))
        goals, inserted = [], []
        for branch in range(2):
            current = start
            for step, relation in enumerate(relations):
                target = int(selected[1 + branch * length + step])
                inserted.append(len(edges))
                edges.append((current, relation, target))
                current = target
            goals.append(current)
        specifications.append((start, relations, frozenset(goals), inserted))

    n_records = len(edges)
    ids = rng.choice(2**31 - 1001, size=n_records, replace=False) + 1001
    times = rng.permutation(n_records)
    costs = np.clip(rng.lognormal(0, 0.55, n_records), 0.35, 4.0)
    facts = []
    for index, (source, relation, target) in enumerate(edges):
        key = base.keys[source * n_relations + relation]
        facts.append(Fact(
            int(ids[index]), source, relation, target, int(times[index]),
            float(costs[index]), tuple(float(x) for x in key),
        ))
    queries = []
    outgoing: dict[tuple[int, int], list[Fact]] = {}
    for fact in facts:
        outgoing.setdefault((fact.source, fact.relation), []).append(fact)
    for index, (start, relations, _, inserted) in enumerate(specifications):
        as_of = max(int(times[position]) for position in inserted)
        reachable = {start}
        for relation in relations:
            reachable = {
                fact.target
                for entity in reachable
                for fact in outgoing.get((entity, relation), ())
                if fact.observed_at <= as_of
            }
        problem_id = hashlib.sha256(f"{seed}:problem:{index}".encode("ascii")).hexdigest()[:24]
        queries.append(Query(
            id=f"presentation-{index:06d}", start=start, relations=relations,
            goals=frozenset(reachable), as_of=as_of, problem_id=problem_id,
            seed=int(rng.integers(2**32)),
            goal_description="any terminal reachable by this relation chain",
        ))
    return BranchingWorld(facts, queries, encoder=base, seed=seed)


class ProtocolError(RuntimeError):
    pass


class EventKind(str, Enum):
    OBSERVE_FACT = "ObserveFact"
    ASK = "Ask"
    COMMIT_ANSWER = "CommitAnswer"
    SCORE = "Score"
    FEEDBACK = "Feedback"
    UPDATE = "Update"


@dataclass(frozen=True, slots=True)
class AskView:
    query: ObservableQuery
    visible_records: tuple[ObservableRecord, ...]


@dataclass(frozen=True, slots=True)
class Feedback:
    """Whole-question reward and incurred costs, with no gold path diagnostics."""
    query_id: str
    problem_id: str
    full_correct: bool
    abstained: bool
    route_cost: float
    database_cost: float
    maintenance_cost: float
    inference_seconds: float

    @property
    def total_work(self) -> float:
        return self.route_cost + self.database_cost + self.maintenance_cost


@dataclass(frozen=True, slots=True)
class Event:
    sequence: int
    kind: EventKind
    query_id: str | None = None
    fact_id: int | None = None
    score: ProofScore | None = None
    feedback: Feedback | None = None
    result: Any = None


@dataclass(frozen=True, slots=True)
class TrainingCounts:
    unique_questions: int
    presentations: int
    feedback_events: int
    updates: int
    presentations_per_problem: tuple[int, ...]


@dataclass
class _Pending:
    query: Query
    observed_ids: frozenset[int]
    state: EventKind = EventKind.ASK
    answer: Answer | None = None
    score: ProofScore | None = None
    feedback: Feedback | None = None
    read_work: float = 0.0
    route_cost: float = 0.0
    database_cost: float = 0.0
    maintenance_cost: float = 0.0
    inference_seconds: float = 0.0


class FeedbackLearner(Protocol):
    def update(self, feedback: Feedback) -> Any: ...


class OnlineSession:
    """Trusted event driver with per-question immutable observation snapshots.

    Passing a full fact registry here does not make it visible: only
    observe_fact exposes keys and only read_fact exposes an observed payload.
    Policies must be passed AskView and Feedback, never the session itself.
    """

    def __init__(self, facts: Iterable[Fact], *, feedback_enabled: bool = True):
        self._world = BranchingWorld(facts)
        self._observed: dict[int, Fact] = {}
        self._pending: dict[str, _Pending] = {}
        self._events: list[Event] = []
        self._presentations: Counter[str] = Counter()
        self._feedback_count = 0
        self._update_count = 0
        self._costs = {
            "observe": 0.0, "route": 0.0, "database": 0.0,
            "maintenance": 0.0, "update": 0.0,
        }
        self._update_seconds = 0.0
        self._event_time = -1
        self.feedback_enabled = feedback_enabled

    @property
    def events(self) -> tuple[Event, ...]:
        """Evaluator-only event log; includes scores before feedback delivery."""
        return tuple(self._events)

    @property
    def costs(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self._costs))

    @property
    def total_work(self) -> float:
        return sum(self._costs.values())

    @property
    def update_seconds(self) -> float:
        return self._update_seconds

    @property
    def training_counts(self) -> TrainingCounts:
        return TrainingCounts(
            len(self._presentations), sum(self._presentations.values()),
            self._feedback_count, self._update_count,
            tuple(sorted(self._presentations.values())),
        )

    @staticmethod
    def _cost(value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("costs and durations must be finite and nonnegative")
        return float(value)

    def _event(self, kind: EventKind, **kwargs) -> Event:
        event = Event(len(self._events), kind, **kwargs)
        self._events.append(event)
        return event

    def _require(self, query_id: str, state: EventKind) -> _Pending:
        pending = self._pending.get(query_id)
        if pending is None or pending.state is not state:
            raise ProtocolError(f"{query_id}: operation requires completed {state.value}")
        return pending

    def observe_fact(self, fact: Fact, *, observation_cost: float = 0.0) -> ObservableRecord:
        cost = self._cost(observation_cost)
        if self._world._facts.get(fact.id) != fact:
            raise ProtocolError("observation does not match the immutable fact registry")
        if fact.id in self._observed:
            raise ProtocolError("a fact can be observed only once")
        if fact.observed_at < self._event_time:
            raise ProtocolError("observations and questions must not move time backwards")
        self._event_time = fact.observed_at
        self._observed[fact.id] = fact
        self._costs["observe"] += cost
        self._event(EventKind.OBSERVE_FACT, fact_id=fact.id)
        return fact.observable()

    def ask(self, query: Query) -> AskView:
        if query.id in self._pending:
            raise ProtocolError("each presentation requires a new query ID")
        if query.as_of < self._event_time:
            raise ProtocolError("a question cannot precede facts or questions already seen")
        self._event_time = query.as_of
        visible = tuple(
            fact for fact in self._observed.values() if fact.observed_at <= query.as_of
        )
        self._pending[query.id] = _Pending(query, frozenset(f.id for f in visible))
        self._presentations[query.problem_id] += 1
        self._event(EventKind.ASK, query_id=query.id)
        return AskView(query.observable(), tuple(f.observable() for f in visible))

    def read_fact(self, record_id: int, query_id: str | None = None) -> Fact:
        if record_id not in self._observed:
            raise ProtocolError("cannot read a fact before its observation event")
        pending = None
        if query_id is not None:
            pending = self._require(query_id, EventKind.ASK)
            if record_id not in pending.observed_ids:
                raise ProtocolError("fact is outside this question's observation snapshot")
        fact = self._observed[record_id]
        self._costs["database"] += fact.read_cost
        if pending is not None:
            pending.read_work += fact.read_cost
        return fact

    def commit_answer(
        self, query_id: str, answer: Answer, *, route_cost: float = 0.0,
        database_cost: float = 0.0, maintenance_cost: float = 0.0,
        inference_seconds: float = 0.0,
    ) -> Event:
        """Commit once. database_cost is additional to charged read_fact calls."""
        pending = self._require(query_id, EventKind.ASK)
        if not isinstance(answer, Answer):
            raise TypeError("answer must be an immutable Answer")
        route, database, maintenance, seconds = map(
            self._cost, (route_cost, database_cost, maintenance_cost, inference_seconds)
        )
        pending.answer = answer
        pending.route_cost = route
        pending.database_cost = pending.read_work + database
        pending.maintenance_cost = maintenance
        pending.inference_seconds = seconds
        self._costs["route"] += route
        self._costs["database"] += database
        self._costs["maintenance"] += maintenance
        pending.state = EventKind.COMMIT_ANSWER
        return self._event(EventKind.COMMIT_ANSWER, query_id=query_id)

    def score(self, query_id: str) -> Event:
        pending = self._require(query_id, EventKind.COMMIT_ANSWER)
        pending.score = self._world.score(
            pending.query, pending.answer, pending.observed_ids
        )
        pending.state = EventKind.SCORE
        return self._event(EventKind.SCORE, query_id=query_id, score=pending.score)

    def feedback(self, query_id: str) -> Event:
        if not self.feedback_enabled:
            raise ProtocolError("feedback is disabled for this frozen evaluation session")
        pending = self._require(query_id, EventKind.SCORE)
        pending.feedback = Feedback(
            query_id, pending.query.problem_id, pending.score.full_correct,
            pending.answer.terminal is None, pending.route_cost, pending.database_cost,
            pending.maintenance_cost, pending.inference_seconds,
        )
        pending.state = EventKind.FEEDBACK
        self._feedback_count += 1
        return self._event(EventKind.FEEDBACK, query_id=query_id, feedback=pending.feedback)

    def update(
        self, query_id: str, learner: FeedbackLearner, *, update_cost: float = 0.0
    ) -> Event:
        """Consume one feedback update attempt, including attempts that fail.

        A learner may mutate before raising, so retries cannot be made atomic.
        Failed attempts remain charged and are recorded before re-raising.
        """
        pending = self._require(query_id, EventKind.FEEDBACK)
        cost = self._cost(update_cost)
        self._costs["update"] += cost
        self._update_count += 1
        pending.state = EventKind.UPDATE
        started = time.perf_counter()
        try:
            result = learner.update(pending.feedback)
        except BaseException as exc:
            self._event(
                EventKind.UPDATE, query_id=query_id,
                result={"error_type": type(exc).__name__},
            )
            raise
        finally:
            self._update_seconds += time.perf_counter() - started
        return self._event(EventKind.UPDATE, query_id=query_id, result=result)


def training_schedule(
    queries: Iterable[Query], unique_count: int, repeats: int
) -> list[Query]:
    """Present U problems in time order, repeating each R times at its snapshot.

    Repeats are contiguous so causal time never moves backwards. Repetition
    does not change query noise, seed, snapshot, or unique-problem identity.
    """
    if unique_count < 1 or repeats < 1:
        raise ValueError("unique_count and repeats must be positive")
    selected = tuple(queries)[:unique_count]
    if len(selected) != unique_count:
        raise ValueError("not enough unique problems")
    if len({query.problem_id for query in selected}) != unique_count:
        raise ValueError("duplicate problems cannot count toward U")
    return [
        replace(query, id=f"{query.problem_id}:presentation:{repeat}")
        for query in sorted(selected, key=lambda item: item.as_of)
        for repeat in range(repeats)
    ]
