# V14 engineering record

Nine behavioral contracts passed before scientific training, including physical
contacts, split coverage, archive ownership, preaction raster persistence,
selected-expert gradients, terminal GAE and fixed probe scene assignment.
The first missing-implementation test pass failed as expected. Two runner
contracts then exposed world-step accounting and missing fixed-scene support;
these were repaired before training.

The runner originally contained external 4% random actions incompatible with
the stored PPO action log probability. This was removed before any scientific
run. Exploration comes only from the sampled expert and routing distributions.
Episode statistics include all rollout transitions, not only the last frame.

Runtime: existing E: Python training environment, PyTorch 2.7.1+cu128,
RTX 4070 Laptop GPU. Pymunk 7.1.0 installed into the E: tracking_runtime
directory; no changes to the shared training environment.
Tensor inputs are 32 visible sensor features and 16 explicitly historical
transition features. Raster images are stored, not used to train perception.
This is idealized visible-feature control, not end-to-end camera learning.

Checkpoint files are learning states and optimizer snapshots, not exact
mid-trajectory resume snapshots of Pymunk bodies. Do not claim exact continuation
from them. Frozen probes restore training RNG after completion and never fit
models or update record utility from probe feedback.

The first rowwise audit was stopped before completion after excessive runtime:
its two per-record array assertions dominate the cost for roughly one million
transitions. Source is preserved as audit_rowwise_snapshot.py; it produced no
valid audit receipt. The final auditor batch-loads rows with the same ownership,
time, sensor/action/next-state, reward/contact/terminal and raster-integrity
checks. This changes archive verification only; no training is repeated.
