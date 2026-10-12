# Frozen online-memory protocol

Date: 2026-10-11. Scope: functional vector retrieval and explicit index
maintenance. This does not simulate physical brain dynamics or train language
understanding. Existing frozen experiments are unchanged.

## Environment and independent answer verification

The stream contains 24 batches of 64 observed facts, at most 1536 facts.
Every batch contains complete branching two-hop paths and irrelevant facts.
Each source/relation address can have multiple targets. Record IDs are opaque;
keys encode only the observable source/relation, never a target or proof ID.
The fixed encoder is a controlled representation, not learned knowledge.

After each batch, ask six complete questions: two recent, two old, two sampled
from observed problems. The question asks for any terminal reachable through
its stated relation sequence and the ordered evidence. The target condition
is therefore observable, not a hidden class constraint. The trusted verifier
checks every edge's source, relation, observation time, continuity and terminal.
It accepts all valid paths, rather than agreement with dense retrieval.

Three streams use the same generation rules: mixed topics, topic shift, and
return to the early topic. SQLite receives facts only as observations arrive.
No inference policy receives the world, reference paths or unobserved values.
After an answer is committed, the controller scores it. This track has no
learned updater; its maintenance schedules are fixed in advance.

## Methods and costs

- Dense: GPU cosine scoring of all observed keys, then read the chosen payload.
- Stale: growing directory without reassignment.
- Periodic: same directory, full reassignment after every fourth stage.
- Incremental: same directory, reassignment of the next 64 observed records
  after every stage. Record visitation is round-robin, not gold-directed.
- Symbolic index diagnostic: SQL source/relation lookup, because the symbolic
  query is explicit. This strong conventional baseline exposes the limits of
  interpreting vector retrieval savings as general reasoning improvements.

Stale, periodic and incremental probe counts are 1, 2 or 4, selected only on
validation. Periodic/incremental maintenance happens after that stage's answers
are scored, so it can affect only later stages. Maintenance is not conditional
on the correctness of the current answer.

All methods pay for observed-key validation/normalization/storage and actual
SQLite writes. The symbolic method pays for its additional SQL index. Dense
does not pay for an unused directory. Reads, candidate scoring, gather traffic,
directory insertion, index sorting, maintenance and SQL payload decoding are
included. Work units and wall time are reported separately. Reference
enumeration, scoring and log serialization are evaluation overhead.

## Selection and sealed evaluation

Development seeds: 911, 919, 929. Validation seeds: 1009, 1013.
Final test seeds: 1103, 1109, 1117, 1123, 1129.

Pilot seed 901 is for runtime and implementation checks only. The pilot may
use fewer stages/questions and is never included in a scientific aggregate.

For each maintenance family, select the lowest total-work probe count whose
validation overall full-answer accuracy is within 1 percentage point of dense
and whose per-stream accuracy is within 3 points of dense. If none qualify,
keep the highest-accuracy configuration and explicitly record failed selection.
Freeze selected names and source hashes before reading test results.

Report all test seeds, per-stage learning curves, per-stream accuracy, complete
answer and at least two accepted alternatives when available, and total cost
including index maintenance. A useful vector method must satisfy the same
quality limits and save at least 20% total work versus dense. Beating ordinary
symbolic indexing requires a separate direct comparison; failure is admissible.
No test-set tuning, early stopping after a favorable seed, or success guarantee.
