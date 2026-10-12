# Online Benchmark Contract

The package uses the standard library and NumPy. It imports no GPU framework.
Existing V1/V2 sources are read only: their controlled vector encoder is loaded
from the sibling frozen benchmark without editing that source.

## Controller And Policy Boundary

`BranchingWorld` and `OnlineSession` belong to the trusted experiment controller.
Policies receive `ObservableRecord`, `AskView`, a current query vector, and
delivered `Feedback`. They must not receive the truth registry, `reference()`,
session event log, or scores before feedback delivery.

An observable record contains an opaque ID, vector key, observation time, and
read cost. Its source, relation, and target are payload fields obtained through
a charged observed-store read. Query vectors are produced by the frozen
controlled encoder; this is not learned semantic understanding.

## Proof Semantics

`Answer(terminal, support_ids)` contains one terminal and an ordered immutable
record-ID tuple. `terminal=None` is abstention.

`world.score(query, answer, visible_ids)` checks immutable facts, observation
visibility, timestamps, edge continuity, relation order, proof length, optional
simple-path constraints, and terminal membership. It never calls `reference()`.
Different valid paths and valid terminals are accepted.

`make_branching_world(seed, ...)` inserts at least two legal alternatives per
problem. Its public question asks for any terminal reachable by the relation
chain. The evaluator derives all accepted terminals from its truth graph.
Keys for branches with the same source/relation are identical, so a retrieval
policy may need charged branch exploration when a selected branch dead-ends.
Hand-authored queries can restrict `goals`; by default the rendered query names
those destination constraints.

## Event Order

Call `observe_fact(fact)`, then `ask(query)`, `commit_answer(query_id, answer)`,
`score(query_id)`, `feedback(query_id)`, and `update(query_id, learner)`.
Several committed questions may await feedback. An answer cannot be rewritten
after commit; each feedback and update can be consumed once.

Observation and question times cannot move backwards. At Ask, observed IDs are
frozen: later observations cannot repair the proof. Controllers must also keep
policy inference tied to this snapshot. `read_fact(id, query_id)` enforces its
snapshot and charges the fact's read cost.

`feedback_enabled=False` allows frozen scoring but rejects feedback and updates.
Feedback contains only whole-question correctness, abstention, incurred costs,
and inference duration. It contains no gold path or edge-level diagnostics.

`update` consumes an attempt before invoking the learner. An exception is
re-raised after recording and charging the failed attempt, because learner
mutations cannot be rolled back safely. The update counter counts attempts.

## Counts And Costs

`training_schedule(queries, unique_count=U, repeats=R)` selects U distinct
problem IDs, sorts them by snapshot time, and presents each R times
contiguously. Query seed/noise/snapshot stay unchanged across repetitions.
`session.training_counts` reports unique problems, presentations, feedback
events, update attempts, and presentations per problem.

`read_fact` charges database work immediately. The `database_cost` supplied to
`commit_answer` is additional work, not a replacement for those charges.
`session.costs` separates observation, route, database, maintenance, and update
work. `total_work` sums them; callers must supply their actual charged
observation, routing, maintenance, and learning work.

## CPU Verification

Run from the containing V3 experiment directory:

```text
python -m unittest discover -s online_benchmark -p test_online_benchmark.py -v
```

On 2026-10-11, all 29 tests passed. Coverage includes multiple proofs/terminals,
wrong goals, cycles, unknown/future facts, canonical-reference independence,
immutable snapshots, chronological events, frozen feedback, update causality,
replay rejection, U x R accounting, and a literal cost ledger.
