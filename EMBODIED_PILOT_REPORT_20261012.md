# Embodied Multi-Signal Pilot

Date: 2026-10-12

Audit correction (2026-10-12): the previous contact target was included in the
same-step proprioceptive input, so its contact accuracy cannot support a
prediction claim. The directory holds static parameters rather than observed
event memory. The dual-rate parameter count includes an unused inherited GRU,
and successive designs reused an opened test set. Action accuracy measures
agreement with a heuristic, not all environment-valid actions. Numbers below
are preserved exploratory measurements; superiority and physical-generalization
interpretations require a new independently scored experiment. See
`retention_v5/PROTOCOL.md`.

This pilot is a functional screening experiment. It is not a biological
simulation and does not claim that a synchronous PyTorch loop is equivalent to
brain dynamics.

## Question

Can an online time-indexed directory route a small fraction of recurrent state
while a separate fast path handles immediate action and a slower path predicts
physical state?

The experiment uses:

- an egocentric 12x12 three-channel visual signal;
- proprioception: position, velocity, target-relative position, and contact;
- five actions: stay, up, down, left, right;
- environment-generated next-state and contact targets;
- fixed train, validation, and held-out moving-target test streams;
- two random training seeds: 11 and 23.

The test labels come from the environment, not from a dense model. Accuracy is
actual task accuracy on the generated reaction labels.

## Models

`DenseGRU` processes all recurrent state every step.

`DirectoryRNN` adds 48 trainable directory slots and selects 4 slots per step.
The selected values enter one recurrent state. Its route fraction is
`4/48 = 8.33%`; this is a routing sparsity proxy, not a claim of an equal
wall-clock FLOP reduction because the current pilot still computes all slot
scores.

`DualRateDirectoryRNN` separates:

- fast state: action head only;
- slow state: directory-routed state prediction;
- shared visual/proprioceptive encoders;
- contact prediction from both paths.

This is the third algorithm iteration. The first version let the action head
read both fast and slow state and showed a stronger action/prediction conflict.

## Fixed pilot results

All runs use 256 training episodes, 64 validation episodes, 96 test episodes,
20 steps, 8 epochs, and the same generated split.

| Model | Objective | Params | Action accuracy | State MSE | Contact accuracy | Route fraction |
|---|---:|---:|---:|---:|---:|---:|
| DenseGRU h=96 | task-only | 78,124 | 63.70% | 0.18588 | 43.80% | 100% |
| DenseGRU h=96 | multi | 78,124 | 59.87% | **0.04446** | 96.56% | 100% |
| DirectoryRNN h=96 | task-only | 100,813 | 62.03% | 0.21014 | 43.93% | 8.33% |
| DirectoryRNN h=96 | multi | 100,813 | 54.30% | 0.05488 | 97.81% | 8.33% |
| DualRateDirectoryRNN h=96 | task-only | 143,101 | 62.71% | 0.28775 | 65.55% | 8.33% |
| **DualRateDirectoryRNN h=96** | **multi** | **143,101** | **64.51%** | 0.05188 | **100%** | **8.33%** |
| DenseGRU h=152 (capacity control) | task-only | 142,300 | 64.01% | 0.18324 | 58.93% | 100% |
| DenseGRU h=152 (capacity control) | multi | 142,300 | 62.76% | **0.04696** | 96.69% | 100% |

Values are averages over seeds 11 and 23. The full per-seed records and
training curves are in the corresponding `SUMMARY.json` files.

## Interpretation

1. Multi-objective learning is not automatically better for immediate action.
   In the single-state models it improves state and contact prediction while
   reducing action accuracy.
2. Splitting fast reaction and slow directory prediction is a useful
   mechanism-level improvement. At nearly equal parameter count, the
   dual-rate model has a small action advantage over the dense capacity control
   (`64.51%` vs `62.76%`) and keeps a sparse route.
3. The dense capacity control is still better at state MSE
   (`0.04696` vs `0.05188`). The directory mechanism has not yet shown a
   prediction-quality win.
4. The action improvement is small and seed-dependent. It is evidence for a
   promising algorithmic direction, not a final generalization claim.
5. The 8.33% route number must not be reported as measured runtime reduction.
   The current implementation still performs a full query-to-slot score
   matrix; reducing that cost is a separate implementation experiment.

## Reproducible artifacts

- `embodied_pilot.py`: environment, models, loss, training, and evaluation.
- `embodied_pilot_v1_20261012/`: dense/directory ablations.
- `embodied_pilot_v2_dual_rate_20261012/`: first dual-rate version.
- `embodied_pilot_v3_fast_action_20261012/`: current dual-rate version.
- `embodied_pilot_v4_capacity_dense_20261012/`: nearly parameter-matched dense
  control.
- `online_benchmark/`: 29 protocol tests, all passing.

## Next experiment

The next change should be one factor at a time:

1. Keep the current dual-rate split.
2. Add a held-out environment stream with target switches and obstacle layouts
   unseen during training, reported separately from the current moving-target
   test.
3. Add an explicit directory maintenance rule: timestamp, state similarity,
   prediction error, usage, and bounded forgetting of low-use slots.
4. Replace full slot scoring with a coarse route index so the 8.33% route is
   also reflected in measured computation.
5. Only after this pilot is stable, write an adapter for a mature fruit-fly
   body/environment package. The adapter must preserve visual, contact,
   proprioceptive, and action channels instead of converting the task back into
   text QA.
