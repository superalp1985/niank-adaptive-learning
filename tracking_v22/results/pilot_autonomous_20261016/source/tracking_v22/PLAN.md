# V22: continuous internal state, spikes, and visual pursuit

Approved direction: `../SUBJECTIVE_STATE_DESIGN_NOTES_ZH.md`.
Started 2026-10-16. This is an implementation and experiment plan, not a result.

## Constraints

- Work only on E:, reuse the existing CUDA environment and Pymunk runtime.
- Preserve V21 and all earlier results. Every changed algorithm gets a new run.
- Physical 360-degree RGB panoramas, 1024 x 512, at 100 Hz.
- Pymunk physics <= 1 ms; LIF integration 0.1 ms, checked against 0.05 ms.
- Independent clocks in seconds. No interpolation of old 128-ray observations.
- Agents receive RGB and body sensors, never object coordinates or identities.
- Parallel mixed scenes. Record contacts, collisions, attacks and learning curves.
- "Subjective" means functional continuous internal state, not consciousness.
- Learning authority is limited to local models, memory selection and parameters.
- Facts stay recoverable in SQLite. Active forgetting is reversible.
- Archive complete outputs, verify remote hashes before deleting originals.

## Interface

`tracking_v22/world.py`: `World(seed, combo=None, heldout=False)`.
`observe()` returns uint8 RGB [512,1024,3] and float32 proprioception [8]:
body-frame vx/1.5, vy/1.5, angular speed/pi, touch, attack impact,
energy [0,1], integrity [0,1], target-contact sensor.
`step(action)` advances 0.01 seconds and returns `(observation, reward, done, info)`.
Actions: 0 stop, 1..8 body-relative directions at multiples of pi/4.
Reward is diagnostic only, not the autonomous learner's objective.
Info includes timestamp, combo, contact, safe_contact, collision, impact,
spawned, progress, distance and visibility audit. Distance is audit-only.
Full runtime serialization must preserve world random and physical states.

`tracking_v22/neural.py`: native RGB feature extraction preserving 2D marker
structure; LIF membrane, spikes and decaying eligibility; trainable continuous
latent state; local action/outcome predictors and contextual learning controller.

`tracking_v22/run.py`: synchronous batched GPU practice, SQLite event/fact
storage, independent trial directories, complete checkpoints and summaries.

## Tasks and status

1. [complete] New physical/visual environment, real-input snapshots, tests.
2. [complete; review underway] Spiking state and local autonomous learning, causal tests.
3. [complete] Runtime, persistence/recovery, sensor and timing checks.
4. [in progress] Concurrent-scene practice; failure-driven mechanism iteration.
5. [pending] Review, report, GitHub archive and remote verification.

## Algorithm starting point

The initial architecture uses learned visual features, a continuous latent
internal state and LIF temporal traces. A disclosed visual pursuit prior may
provide initial motor competence; it must read pixels only and be shared by
all comparisons. A learned residual policy and action-consequence model
adjust behavior from physical outcomes. Internal energy, integrity, surprise
and latent state modulate the value of consequences.

A contextual controller chooses observe, recent learning, or memory recall.
Its credit uses subsequently observed prediction improvement and physical
consequences, with explicit delayed attribution and update cost. Stored
experiences can leave active memory and later return. No reference policy
agreement is counted as correctness.

We test a shared-sensor fixed-learning arm against autonomous learning, then
change a mechanism only after seeing its failure. Comparisons record practice
time, fresh frames, action count and optimizer steps separately. No promise
of generalization or universal self-improvement follows from a short run.

## Evidence and stopping

Required before claiming a run is complete: sensory previews, timing check,
parameter-change evidence, deterministic continuation check, per-phase
physical metrics, choices and prediction losses, frozen unseen-combination
probe and an explicit limitation statement. Runs are finite and checkpointed;
the broader research direction remains open.
