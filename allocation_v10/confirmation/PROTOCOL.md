# V10: feedback-driven common allocation

Frozen initial protocol, 2026-10-12. This is a new local mechanism experiment,
not a reproduction of a published architecture and not proof of novelty.

## Task and learner boundary

Reuse the V5 physical transition function. Train on 14 material/force pairs;
reserve (0,0) and (3,3) combinations for frozen composition probes. Both factors
individually occur in training. Every 32 steps the environment selects a new pair,
displays material at phase 0 and force at phase 4, then switches both at phase 8.
The starting pair is also visible through the initial flashes. Texture frames
continue every 4 steps. Position and velocity continue from executed outcomes.
Goals change each cycle. Body, risk context and periodic phase are observed.

No learner receives pair indices, hidden physics constants, unexecuted outcomes
or evaluator regret. The model predicts three actions but only the executed
action receives supervised next-state/contact feedback. Physical exploration is
10% uniformly random during learning, 0% during frozen evaluation.
Success means realized cost is within 0.02 of the best simulator action.
Controller reward uses only realized cost, pre-update factual prediction error,
and declared resource charges, never success or regret.

## Memory and parameters

All nonzero event images are stored as lossless int8 blobs in SQLite. Active
memory contains at most 3 references; candidate metadata covers the latest
8 events. Older data remains queryable in the database but is outside this
initial controller's candidate window. New events enter automatically and
evict the oldest live reference when full. The controller learns retrieval
and explicit fading pointers from timestamps/live flags, without seeing
missing raw image values. Fading erases the cached raw image, not model weights.

The physical predictor has a shared encoder and a bank of hidden-unit groups
within two parameter matrices. A learned router selects 1 group, or 2 after a
compute operation. All groups are reserved at initialization and count toward
total parameters. Admission/retirement changes the per-world accessible prefix,
initially 2, not allocated tensor size. It is reversible capacity admission,
NOT actual dynamic creation/deletion of parameters. Groups share learned
weights across worlds, so admission does not guarantee an untrained group.
This limited first step must be reported before attempting actual growth.

Small/large reserves use 9/78 groups, width 96, shared width 64. Count exact
parameters including controller. Sparse selected matrix multiplies are real,
but PyTorch gradients and Adam updates use dense parameter storage. Record
forward MAC estimates separately from dense optimizer element updates.
This round does not implement reduced precision or Qwen QKV sparsity.

## Controllers and learning

Every 4 physical steps choose one of keep, retrieve, relearn, compute, admit,
fade, retire. A separate learned pointer chooses retrieve/fade candidates.
Compute grants a second group for 4 steps. Admission increases the prefix by
one; retirement decreases it by one, down to 2. Invalid operations are masked.
Five percent uniform exploration is mixed into the policy.

Three controllers:

* independent: separate memory/capacity networks with fixed equal arbitration;
  branches lack shared recovery state; delayed returns.
* immediate: joint network; current 4-step factual return; recovery state hidden.
* delayed: joint network; 32-step returns and bootstrapped value; recovery state.

Recovery state records observed time absent, losses accumulated while absent,
replays while absent, and revisits. These are correlates, not causal estimates
of an individual detail's value. Unresolved absences are censored.

Predictor: Adam 0.001, one update per step, clip 2. Controller: Adam 0.0005,
one update per 32 steps, clip 2, discount 0.97 per physical step, entropy 0.025.
Retain the last 64 factual transitions per world; relearn selects a random
past transition from the same world's buffer, at most once per decision.
Replay occurs before the following physical action. Retrieval supplies raw
observations but does not itself add an optimizer step.

Reward per step:
`-realized_cost - 0.5 * pre_feedback_state_MSE
 -0.002 * number_of_selected_groups
 -0.0002 * accessible_groups`.
Retrieval costs 0.01; replay 0.01; admission 0.02, charged at decision time.
All components are separately reported. These are fixed engineering cost
weights, not biological laws or measured hardware latency.

## Execution and accounting

Development: three modes, one paired model/world seed, 128 worlds x 256 steps.
After diagnostics freeze either this implementation or one justified revision.
First formal rung: 128 worlds x 2048 steps = 262,144 new transitions per run.
Use separate initialization/world seeds; pair conditions across methods.
Do not tune using held-out combinations. Additional 1M/8M parameter or exposure
rungs follow observed failure modes, not a blind full sweep.

Record main/replay presentations, selected group counts, resident and active
parameters, controller work, forward/training MAC estimates, optimizer element
updates, SQLite reads, actual scores, curves and source hashes. Training MAC
estimates use 3x forward MACs and do not claim measured FLOPs or speedup.
Same-experience comparisons have equal caps, not identical realized replay cost.
Compare interpolated performance at shared estimated MAC budgets separately.

Statistical replication is a whole training run. Shared-weight worlds and
overlapping controller windows are not independent algorithm replicates.
Frozen probes are inference-only with feedback used for metrics, never updates.
Engineering tests check chosen-outcome isolation, preserved raw memories,
parameter routing, time/order constraints and cost accounting.
