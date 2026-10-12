# V5: useful forgetting, physical response, and learning volume

Date: 2026-10-12. New independent experiment; previous outputs are preserved.

## Correction to the previous pilot

The previous contact target was also present in the same-step input. Its 100%
contact accuracy is invalid evidence for prediction. Its directory values were
static learned parameters, not observed episode records. The dual-rate class
kept an unused inherited GRU; its reported parameter count included that GRU.
Action accuracy compared with one heuristic choice, rather than scoring all
legal actions by their physical outcome. Successive designs also reused an
opened test set. Previous numbers remain exploratory and cannot support a
confirmed generalization or biological-mechanism claim.

## New functional hypothesis

At fixed active memory capacity, retaining information that helps future
prediction and physical response can outperform keeping recent or surprising
information. Small visual details may be worth keeping when their physical
consequences matter. Large unpredictable nuisance signals may be worth dropping.
This is a testable hypothesis, not a presumed result.

The value estimator is inspired by the user's description of FAME's role.
It is not an implementation of FAME's full theory and makes no claim of global
algorithmic novelty.

## Reduced physical environment

A small visual scene announces two material properties at random early times:
momentum retention and a weak signed force. The force cue occupies only two
pixels and has lower amplitude than irrelevant textures. Later distractors are
bright and unpredictable. Material details disappear before the decision.

The body observes position, velocity, goal, and a persistent internal risk
context. That context determines whether the weak force affects later dynamics.
At the query, three candidate accelerations cause:

    v_next = momentum * v + 0.35 * action + risk * weak_force
    x_next = x + dt * v_next

Truth is generated from these physical equations. This is a delayed-calibration
screening task with visual and bodily signals, not a complete fly body, learned
brain dynamics, or a replacement for FlyGym.

Neither the material values nor future contact/action labels enter the query.
Two layouts (right and left wall) and longer delays are reported separately.
The final stress split also changes irrelevant texture statistics.

## Real memory and archive

Values are encodings of observed events with record IDs and timestamps.
Priority scores depend only on each observed event and contemporaneous internal
context. Keeping the top scores in a prefix is equivalent to a bounded online
priority buffer; future observations and query labels cannot change past ranks.
The active buffer contains four records. Reads mix two chosen records.
Full memory reads all records. Observed records are also archived on E: in an
SQLite database. Forgetting removes an active reference; it does not delete
the archived observation.

For this first causal retention comparison, all retained records are searched.
Hierarchical coarse lookup is deferred until retention quality is established.
The buffer read count is not claimed as a measured GPU speedup.

## Learned value

During training, use the same prediction head to measure the change in
environment-scored loss when a single observed event is available versus an
empty memory. Detached loss differences teach a value scorer conditioned on
event content and internal risk context. A shared single-event decoder branch
is trained on physical truth to give small cues a stronger learning signal.

At evaluation, priority uses only that learned scorer, never evaluation truth.
The full archive branch is training replay only; it does not repair the final
buffer answer. Its compute is charged and every variant receives the same
training-only auxiliary branch. No model is distilled from another model's
answers.

Variants have exactly the same learned architecture:

- full: retain and read all observed events;
- fifo: keep the latest four records;
- surprise: keep the largest visual changes;
- value: keep four learned future-value records;
- anti_value: deliberately keep the four lowest-value records;
- none: no episodic read, to quantify the need for remembered information.

All variants train action quality, physical-state prediction, contact prediction,
and the value estimator. Selection uses development data only.

## Volume controls

Nested training pools use independent episode seeds; count unique episodes,
presentations, repeats, and optimizer updates separately.

- 64 unique x 64 repeats = 4,096 presentations;
- 512 unique x 8 repeats = 4,096 presentations;
- 512 unique x 32 repeats = 16,384 presentations;
- 2,048 unique x 8 repeats = 16,384 presentations.

Batch size 64 gives equal updates at each equal presentation budget.
Three optimization seeds are used. Each CUDA batch contains independent
episodes in parallel. Hypotheses use independent parameters and optimizers.

## Metrics and constraints

Primary outcome: fraction of actions whose environment cost is within 0.02
of the minimum possible cost. Multiple equally good actions pass.

Also report action regret, physical-state MSE, future contact balanced accuracy,
collision rate under the chosen action, fine-cue retention in risk contexts,
nuisance occupancy, records read/scored, and training curves. Contact truth is
not an observation. Predictions are scored before any feedback.

## Preserved failed iteration

`discovery/` used unbalanced physical sampling and leave-one-event-out value
targets. All six models chose the same constant deceleration action: 70.70%
success, exactly matching that constant's development baseline. This is a
failed task discrimination and is not evidence of learned response.

`discovery_v2/` balances the generating environment's best action over the
three choices. It uses per-channel visual encoders and event-versus-empty
training value, with 512 unique episodes x 32 repetitions. These changes use
development results only; final seeds remain unopened.

`discovery_v2` at 32 repeats: full 95.70%, value 94.92%, FIFO 85.94%,
surprise 91.80%, anti-value 86.33%, no memory 87.11%. However, value retained
the weak force cue in only 17.72% of risk contexts. This is a partial success
for coarse retention and a failure to protect important fine details.

`discovery_v3/` adds magnitude-invariant per-channel visual encoding to test
whether weak details are lost before the value scorer can judge them. This
transformation treats all three observed channels by the same formula; it does
not use ground-truth cue type or evaluation outcomes.

Use new training/development/final episode seeds. Freeze source hashes and all
configurations before final evaluation. Preserve intentional failures and
ordinary failures. Do not infer emergence from a thresholded score alone.

Storage, caches, logs, snapshots, and checkpoints stay under this E: directory.
Python starts as a hidden worker with bounded logs, not an interactive inference
CLI. This experiment uses software scheduling, not a brain-dynamics equivalence.

## Post-hoc independent diagnostic

After confirmation, freeze all saved checkpoints and use 256 new episode IDs
810000..810255, with 26-step delay. Hold body, coarse cue, and distractors fixed,
flip the small cue sign, and separately set the observed risk context to zero
or one. Ground-truth velocity change is respectively zero or 0.36 in magnitude.
Report effect gain, error, spurious safe-state response, and detail retention.
This is a read-only failure diagnostic, not another selection set. No model is
retrained after opening it.
