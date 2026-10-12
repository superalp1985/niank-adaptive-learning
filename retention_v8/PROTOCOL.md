# V8 development: outcome differences and conditional omission value

Date: 2026-10-12. Proposed while V7's fixed training-volume screen is running.
V7 will be retained whether it succeeds or fails. No V8 final set is opened.

## Motivation

Low-volume V7 can score above 90% on actions while responding weakly to a force
sign that has a known physical effect. The body's distribution can partially
predict the action, so action success alone cannot establish detail learning.
An irrelevant signal can also appear useful when judged alone by an incomplete
decoder. V8 tests two changes and isolates only the second:

1. Common to both modes: empirical event value is now loss with an event omitted
   from the archive read minus loss with that read intact. This evaluates a
   contribution in the presence of other information. Targets are detached.
2. `quartet_delta` additionally learns the difference between predicted outcomes
   and actual physical outcome differences for matched observations.
   `quartet_point` sees exactly the same quartet and has only pointwise losses.

All models have the same learned event gate and complete current-body value
inputs. No parameter receives a human label for importance or irrelevance.
The same 0.005 research use cost, four-reference maximum and two-read maximum
remain. All candidate events are encoded; recovery is a complete SQLite scan.

## Matched exposure

Each unique body history supplies four physical experiences: original and
force-sign-reversed observations, each in context zero and context one.
Body, coarse cue and texture are unchanged. Actual next-state/contact/cost labels
are recomputed by the environment for each experience.

All four versions are observed, predicted, then used for learning. Differences
are a training relation between measured physical outcomes, not a preset
instruction to set a gate to zero or one.

Both modes share source histories, batches, presentations, initialization,
parameter count and update count. Auxiliary omission decoding adds work and is
recorded. This is software GPU batch computation, not biological simultaneity.

## Fixed development screen

512 histories x 32 quartet cycles = 65,536 outcome-context presentations.
Seed 11. Fresh development IDs: train 1,900,000; ordinary 1,910,000;
fine intervention 1,920,000; recovery 1,930,000.

No independent final dataset exists in the development run. The development
screen assesses action success, signed fine response gain (ideal 1), safe
spurious effect (ideal 0), state-selective retention, and real archive recovery.
Failure is preserved and does not trigger a rerun of the same configuration.

## Confirmation, conditional on development evidence

Freeze both modes; run three optimization seeds (11,23,37) and:

- 512 histories x 8 quartet cycles = 16,384 presentations.
- 2,048 histories x 2 quartet cycles = 16,384 presentations.
- 2,048 histories x 8 quartet cycles = 65,536 presentations.
- 2,048 histories x 32 quartet cycles = 262,144 presentations.

Fresh train/development/detail/recovery/final ID ranges. Final IDs shared across
models. Include left-layout and texture-shift transfer sets. This is a limited
task finding; it cannot establish general intelligence or a biological theory.
