# V14: mixed-scene marked-object pursuit

Date: 2026-10-13. The user authorized implementation and algorithm iteration.
The withdrawn V13 sequence is not resumed. This task replaces the one-dimensional
cue task; its scores are not comparable to V12.

## Functional target

A circular mobile body must contact a visibly marked target while avoiding
solid obstacles and incoming projectiles. Targets can be stationary, moving
linearly, or turning. Physical motion and collisions use Pymunk 7.1.0.
This is a reduced tracking-controller study, not a deployed robot, camera
recognizer, or biological dynamics simulation.

Perception is explicitly idealized: measured relative positions of the marked
target, nearby obstacles and visible projectiles, body velocity, and boundary
range are available. Target/projectile velocities are inferred from successive
observations, never supplied as hidden physics parameters. Raw raster views and
physical observations are archived in SQLite. No scenario identity, future
attack schedule, hidden target program or best action enters learning.

## Concurrent scene distribution

Three layout families: open, scattered circles, and a barrier with a gap.
Three target motions: stationary, straight with wall bounce, and curved.
Three attack regimes: none, aimed, and crossing.
All 27 combinations exist. Hold out the 9 combinations satisfying
`(layout + motion + attack) % 3 == 0`; train on the other 18.
Each factor and each pair of factors occurs in both splits.

48 independent bodies continuously sample training combinations on reset.
There is no shared A-B-A-B phase. Each trajectory retains its time order.
Physics is CPU scheduled; CUDA batches neural inference and training.
This is not simultaneous physical execution of every parameter.

## Local learning and the directory question

Four independent action experts, a routing policy, a state-value estimator,
and a factual next-observation predictor have distinct parameters and optimizers.
PPO updates selected expert/router log probabilities from detached advantages.
The value estimator fits trajectory-derived targets independently. Generalized
advantage estimation includes value predictions at actual successor states;
these are estimates, not observed facts or oracle labels.
The predictor fits only executed-action next observations.
No gradient crosses module boundaries, memory lookup or the simulator.

Two matched configurations:

- `recent`: retrieve a relevant executed transition from the last 8 records
  belonging to the same body.
- `directory`: keep a 64-entry coarse temporal index per body, with 8 active
  records and cold records recoverable from SQLite. Coarse keys use observed
  target bearing, obstacle proximity and projectile presence. Select one
  neighboring-key record by observable-state similarity and a revisable utility
  estimate. Utility combines factual progress and predictor error on reuse; fading,
  recovery and repeated practice can change it.

Both expose the same retrieved-transition descriptor to the policies.
A memory record is explicitly tagged as past evidence; it is never substituted
for a current observation. No other body's past observation enters a body's
current sensor vector. All real experiences remain in SQLite.
This is a project-specific directory comparison, not a claim that PPO, replay,
multitask training or nearest-neighbor retrieval is newly invented.

## Frozen initial run

One exploratory initialization, model seed 141, world seed 641000.
48 worlds, 128-step time limit, rollouts of 64 steps, 4 PPO epochs.
Continuous milestones: 1,024 / 4,096 / 8,192 steps per body:
49,152 / 196,608 / 393,216 new experiences per configuration.
Learning counts, updates, parameters and wall time are recorded separately.
Each checkpoint receives frozen seen/held-out probes with independent seeds;
probe feedback never updates any learning module or selects hyperparameters.
Initial untrained probes are included. Probe weights and utility estimates are
frozen; observed records can still enter memory as past evidence. Each fixed
scene receives 1,024 steps rather than exactly 8 completed episodes; report
completed and interrupted episodes explicitly. Later algorithm revisions require a
new directory and fresh declared evaluation seeds; preserve failures.

Reward combines actual decrease in target distance, a time/effort cost,
actual obstacle collision, actual projectile impact, and target contact.
Contact is successful only if no projectile impact occurs in the same step.
Projectile impact terminates an episode; obstacle contact does not.
Collision count is per physical step, not per internal solver callback.
No simulator evaluation of unexecuted actions is used for training.

## Evidence and limits

Track every trajectory and finished episode: target-contact success, collision
steps, projectile impacts, duration, physical reward, and preupdate prediction
error, separately by scene combination. Report training curves and frozen
held-out development throughout learning. Any exposed evaluation split becomes
development data for subsequent revisions, never a fresh confirmatory test.

Test collision/contact correctness, sensor boundaries, held-out pair coverage,
archive recovery and ownership, selected-expert gradients, and rollout terminal
handling before the scientific run. Independently recompute episode totals
from raw transitions before reporting.

Upload code, report and complete snapshots using tools/archive_snapshot.py;
read back SHA-256 for every part before cleanup_verified.ps1 deletes bulk data.
All caches, dependencies, logs and temporary files stay on E:.

## Prior-work boundary

The multi-scene motivation and caveats are in
adaptation_v13/RESEARCH_REVIEW_20261013_ZH.md.
Pymunk official API: https://www.pymunk.org/en/latest/pymunk.html
PPO: Schulman et al., 2017, https://arxiv.org/abs/1707.06347
GAE: Schulman et al., 2015, https://arxiv.org/abs/1506.02438
No published benchmark or prior training result is reproduced here.
