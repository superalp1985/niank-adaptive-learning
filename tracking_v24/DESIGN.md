# V24: delayed observation consequences

Approved direction: continue practice and connect observation to later action.
This version starts both branches from the complete final V23 active runtime.
It preserves the V23 code and original database.

## Mechanism

The control branch continues V23 unchanged. The delayed branch adds three
physical consequence readouts, at 32, 128 and 512 sensor frames. Each predicts
energy change, integrity change, collision time fraction, impact count, contact
count, safe contact count and terminal occurrence for the selected joint action.
Only bodily sensors and experienced physical events supply these targets.
World reward, distance, target coordinates and simulator progress are excluded.

The readouts share the existing attentive encoder. Their supervised losses can
therefore change attention and visual representation. A learned delayed value
chooses head hold/left/right conditional on the body action selected by V23.
The body action at the current observation retains the existing selection rule;
changed observations and shared learning can change later body choices.
New readout weights are zero at initialization, so the initial policy is exact.

For each horizon, value is
`3*energy_need*dE + 5*integrity_need*dI - 2*integrity_need*collision_fraction
- 5*integrity_need*impacts + 3*energy_need*contacts + energy_need*safe_contacts
- 2*integrity_need*terminal`.
Energy and integrity needs are the disclosed V23 functions of current body state.
Horizon weights are 0.2, 0.3, 0.5. Delayed policy gain rises to 0.8 over 128
auxiliary updates. Values are clipped before action use, not before storage.
These are designed preferences, not a discovered intrinsic objective.

Windows end at their horizon or at the actual episode terminal, before reset.
No outcome after reset is assigned to an earlier body episode. Multiple later
actions intervene: these are experienced returns, not causal estimates of the
initial gaze action's isolated contribution.

## Budget and interpretation

Each branch: eight bodies, 32768 additional 10 ms frames per body, plus 512
frames without new learning choices to settle outstanding credit. Native images
remain 1024x512 at 100 Hz; physics 1 ms; spike threshold grid 0.1 ms.
The world remains a software 2.5D approximation.

Keep the original learning scheduler, revisable memory, exploration and all
negative updates. Additional delayed updates are counted separately. Equal
exposure is not equal optimization work. This is an exploratory mechanism
comparison, not a compute-matched claim or proof of a new theory.

Frozen common scenes at 0, 16384 and 32768 additional steps. A separate final
scene set measures final behavior, with uniform-attention, fixed-head and
disabled-delayed-policy interventions. Frozen observations cannot select
weights, change hyperparameters or feed training.

Primary evidence: per-quarter changes in safe contacts, collisions, impacts,
deaths; individual bodies; actual attention/head behavior; realized delayed
targets and learning losses. A changed attention distribution alone is not
success. More minutes do not establish sufficient training.

## Persistence and publication

Preserve worlds, RNG, optimizers, memory, original pending windows and all new
delayed windows in complete runtime checkpoints. Log delayed input batches,
pre-reset body states, matured targets and every auxiliary update in SQLite.
Save runtime-source hashes before practice and require them for resume.
Completed configurations must not be rerun.

All data and caches stay on E. Finish by publishing code, report and a complete
immutable archive; read all remote archive parts back and verify hashes before
selective cleanup.
