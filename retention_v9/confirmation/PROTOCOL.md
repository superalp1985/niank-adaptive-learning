# V9: recurring practice, fading references and evidence-driven relearning

Date: 2026-10-12. User correction: no detail is permanently labeled useless.
Retention grades are revisable results of repeated practice, not fixed classes.

## Question

After a reference fades, do actual prediction failures followed by archive
relearning help future actions? Does repeated measured contribution lengthen
retention, and can that retention decline again under changing practice?

This is a new reduced closed-loop experiment. It does not simulate biological
parallel dynamics or Qwen parameter routing.

## Shared initialization and worlds

Use the already completed V8 `quartet_delta` checkpoints at 2,048 histories x 32
cycles, seeds 11/23/37. No checkpoint is chosen by its score. This isolates the
new practice mechanism from another round of pretraining.

New worlds have independent initial bodies, visual histories and hidden material/
force. Initial bodies are sampled without balancing by the correct action.
An action changes the next position and velocity. Goals change every 8 steps.
Both wall orientations are present from the start, unlike the V8 pretraining.
Context changes every 16 steps, repeatedly revisiting previous conditions.
Context values belong to the simulator, never to the memory importance labels.

Each world has its own raw observation archive and bounded active references.
The controller receives current body/context and cached active image records.
All other original images are read through SQLite only when retrieval occurs.
Truth arrays remain in the environment and evaluator, outside controller input.

## Practice loop

1. Predict next states and contacts from the current active references.
2. Choose the action minimizing predicted physical cost.
3. Execute it in the simulator; score the prediction and action before updating.
4. Reveal only that action's actual next state and contact to the learner.
5. Optionally read the raw archive and learn again from the observed transition.
6. Estimate event contribution as omission loss minus full-read loss, with
   detached targets. Adjust a decaying evidence trace and reference lifetime.
7. Update predictor weights by backpropagation. Continue from the actual new body.

Alternative action outcomes are available only to the evaluator for regret and
near-optimal-action scoring. The learner never receives those counterfactual
labels. A predictor's own counterfactual estimates are not simulator truth.

## Four controls

- `fixed`: keep the initial references. Predictor weights still learn from each
  action's outcome. No later archive reads or directory updates.
- `periodic`: every 8 steps reread and relearn; references use the latest measured
  evidence without accumulated lifetime extension.
- `recurrent`: prediction-error-triggered rereading plus an exploration visit
  after 16 steps without retrieval. Evidence revises grades after every visit.
- `consolidated`: same retrieval trigger and empirical evidence as recurrent;
  accumulated positive contributions lengthen reference lifetime, with decay.

The error trigger uses the already observed chosen-action error and a decaying
past-error scale. It does not see future truth or signal identity. Relearning
cannot change the score or chosen action of the transition that triggered it.

No algorithm sees cue-time metadata. It is used only by the evaluator to count
which reference faded or returned. Rank competition and lifetime limits are
declared research choices, not natural constants or learned innate importance.

All modes have the same model and active capacity four/read capacity two. Their
archive reads and auxiliary training differ by policy and are counted, not
presented as equal-compute comparisons.

## Frozen stages

Development: 32 parallel worlds, 128 steps, seed 11, all four modes.
No final probe set is generated.

Confirmation after development review: 64 parallel worlds, 128 and 512 steps,
three paired initialization/environment seeds, all four modes. Stream outcomes
are scored before learning (prequential evaluation). This is explicitly an
online-learning score, not a frozen test accuracy.

New final probe bodies/images are held out. They are scored with frozen
checkpoints before and after the whole stream, with fresh references, separately
from the stream result. Final results do not select hyperparameters.

## Records and limits

Persist each transition's body, action, predicted and actual chosen next state,
contact, regret, recovery flag, references before/after, fading/restoration
events and reference lifetimes. Raw images remain label-free in SQLite.
Record source/dependency/checkpoint hashes and physical/data seeds.

The real chosen-action experience is also stored in each run's `practice.sqlite`.
This table contains already observed outcomes and predictions. It does not
contain the unexecuted action outcomes used exclusively by the scoring audit.
The raw visual archive remains a separate label-free SQLite file.

The implemented trace has 0.985 per-step decay, 0.25 active-evidence update rate,
and a six-step base lifetime. Positive measured contribution relative to current
factual loss can extend lifetime, capped at 48 extra steps for consolidation.
These are provisional algorithm choices. No biological optimality or permanent
classification follows from them. Inactive details can return on a later scan.

Report complete phase curves, early versus late repeated contexts, lost-reference
errors, read counts and held-out changes. Repeated contexts are not independent
tasks. Policy trajectories diverge after different actions, so paired initial
worlds do not imply identical later observations.

Archive recovery is still a full scan of a world's 18 records. This experiment
tests functional learning dynamics, not hardware acceleration or general
intelligence. Information can remain in weights or directory metadata even
when its raw reference disappears; fading references is not complete erasure.
