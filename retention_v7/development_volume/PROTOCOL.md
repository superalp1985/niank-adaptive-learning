# V7: state-specific event use with signed empirical retention value

Date: 2026-10-12. V5 and V6 are preserved.

## Failure motivating this experiment

V6's whole-memory scalar gate did not distinguish irrelevant fine information.
V7 tests event-level use and learning from identical observations in both current
risk states. This screen does not claim a novel general algorithm.

## Factorial comparison

Four modes share all modules except the optional 1,057-parameter event gate:

- `ordinary_plain`: ordinary state histories, no event gate.
- `ordinary_event`: ordinary state histories, learned per-event gate.
- `paired_plain`: both contexts for each body history, no event gate.
- `paired_event`: both contexts, learned per-event gate.

The gate sees the encoded event and the encoded current body/context. It is
trained through action, next-state, and contact outcomes only. It does not see
cue identities or future information.

All four modes replace V5's relative softmax value target with a signed loss
gain: empty-memory loss minus single-event loss. The value predictor sees the
event and current body/context, with both representations detached in this
auxiliary branch. Its regression target is detached. This mechanism is common
to all four modes, so this screen cannot isolate it from V5.

## Retention

Keep at most four records and read at most two. A record with predicted physical
loss improvement <= 0.005 is not an active reference. This stated cost is a
research setting, not a human importance label or a measured hardware cost.
All candidates are still encoded and scored. No speedup is claimed.

## Exposure accounting

Batch size is 128 outcome contexts. An ordinary batch repeats each body history
twice in its original context; a paired batch presents that same body history
once in each context. Both have equal body count, outcome-context presentations,
optimization updates, and auxiliary branch calls.

Development: 512 distinct body histories, 16 cycles, 16,384 outcome-context
presentations, optimization seed 11. Four modes.

After this low-volume screen failed to produce state-selective retention,
an unchanged-mechanism training-volume screen was scheduled before its results:
512 x 64 and 2,048 x 16, both 65,536 presentations, all four modes, seed 11.
It uses the same development probe IDs and no final set. This is a development
iteration, not independent confirmation. The original frozen source is retained.

Confirmation, only after development selection is frozen:

- 512 histories x 16 cycles x 2 contexts/copies = 16,384 presentations.
- 2,048 histories x 4 cycles x 2 contexts/copies = 16,384 presentations.
- 2,048 histories x 16 cycles x 2 contexts/copies = 65,536 presentations.
- Seeds 11, 23, 37, fresh shared IDs and fixed final episodes.

The high-volume condition examines whether gains need more training, rather than
inferring emergence from a single action success threshold.

## Evaluation

Physical action success accepts all actions within 0.02 of the optimal cost.
Frozen predictions precede feedback. Labels and cue-time metadata do not enter
forward computation. Final datasets are not generated during development.

Fine-sign intervention holds body, coarse cue and texture fixed. True velocity
effect is zero in safe states and 0.36 in risky states. Report response gain,
effect MSE, safe spurious response, and active/read fine retention.

Recovery changes safe to risky context. Old references alone are read from
SQLite for stale evaluation; recovery reads actual raw images from SQLite and
rebuilds references. Count bytes and rows. Report paired outcomes and outcomes
restricted to histories where the fine reference had been forgotten. Models
and the archive remain unchanged during probes.

## Limits

This is a reduced visual/physical memory task with synchronous GPU batches. It
is neither a full fly-brain simulation nor Qwen precision routing. The loss
value comes from a learned decoder and can itself be miscalibrated. Recovery
currently scans all archived records. Retention of a cue alone does not prove
it is used correctly. Optimization seeds are not independent environment worlds.
