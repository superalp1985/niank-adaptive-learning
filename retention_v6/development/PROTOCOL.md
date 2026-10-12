# V6: state-conditioned use and archive recovery

Date: 2026-10-12. This is a new experiment; V5 remains frozen.

V5 showed that future-value retention can preserve a weak physically important
cue, but a post-hoc intervention also showed spurious use of that cue when the
current state made it irrelevant. V6 tests the next hypothesis:

> Retention and use must be conditioned on the current body state. When the
> state changes, the archived observation should be eligible for recovery even
> after its old active reference was forgotten.

## Two learned mechanisms

`state_nogate` learns the same event value rank as V5 and reads two of the four
highest-value records. It has no state gate after the read.

`state_gate` uses the same bounded value buffer, then multiplies the memory
read by a learned gate conditioned on body position, velocity, goal, wall and
current risk context. The gate is trained only through physical action/state
loss; no gate label is supplied.

Both variants have the same encoders, decoder, replay branch, value learning
signal, capacity, optimizer, and update schedule. The gate adds a small
explicit module; its parameter count is reported and not hidden.

## Recovery test

For each new episode, first expose the observations with risk context zero and
save the selected active references. Then switch the current context to one
without changing the archived observations:

- stale: keep the old active references;
- recover: re-score the archived observations under the new state and rebuild
  the active references.

The environment truth is recomputed from the same body and visual cues under
the new context. The prediction is scored before any feedback. This is a
read-only frozen-checkpoint diagnostic; it never changes model weights.

## Training volume

Two equal-presentation comparisons are run:

- 512 unique episodes x 32 repeats = 16,384 presentations;
- 2,048 unique episodes x 8 repeats = 16,384 presentations.

Development uses seed 11 and 512 x 32, with separate development, detail and
recovery episodes. It does not generate the final set. Confirmation, if reached,
uses three new optimization runs (11, 23, 37), both volumes and fresh episode
ranges. Final IDs are identical across all models for paired comparison.
Source, dependency, protocol and dataset hashes are recorded before training.

The recovery archive is an actual SQLite file. Stale evaluation reads only four
old active records into an otherwise empty image tensor; recovery reads the
original observation bytes from SQLite and rebuilds references under the changed
context. Rows and byte counts are recorded. The current implementation still
executes its fixed-shape encoder on blank stale slots; this overhead is disclosed.
No outcome, contact, cost or cue-time metadata is stored in the archive.

## Metrics

Report physical action success, regret, next-state MSE, fine-cue retention,
risk-context intervention response gain, safe-context spurious response, stale
versus recovered action success, and records read. Equal-cost actions pass.

The archive remains the source of specific observations. Forgetting removes an
active reference only; the SQLite archive is not deleted. Full slot scoring is
still charged as a research cost. No claim of measured hardware speedup is
made.
