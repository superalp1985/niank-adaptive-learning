# V17: continuous 360-degree compound-eye practice

Date: 2026-10-13. User approved continued experimentation and explicitly
requires 360-degree compound vision. All outputs and caches remain on E:.

## Design

The existing Pymunk engine supplies actual motion/contact physics. Replace
object-coordinate measurements with 128 equiangular RGB ommatidia, nearest
surface visibility, occlusion and illumination changes. These are planar
360-degree engineering eyes, not anatomically reconstructed insect eyes or
full spherical 3D cameras. No depth, object coordinates, target identity,
attack schedule or scene ID is a policy input.

A green/white striped marked object is the contact target. Unmarked and
partially similar green objects are distractors. All classes share one RGB
image; there are no semantic target/obstacle/projectile sensor channels.
Only body velocity, orientation change and factual collision/impact flags
are additional inputs. Ground-truth visibility is isolated in audit metrics.

Six layout families, four target-motion families and four attack regimes
produce 96 combinations; train the 72 where sum(combo) % 4 != 0 and develop
on the other 24. Every trained combination starts with a parallel body.
Each body's scene persists 1024 steps (122.88 simulated seconds), then
samples another training scene. Target contact respawns a new target in the
same world; impacts do not terminate practice. Each transition has its
body, time and segment recorded. Physical discontinuity clears temporal
images and sector memory, without resetting learned parameters.

Attacks have geometric waiting times, varying speeds and aimed, crossing
or burst trajectories. There is no periodic release rule. Layouts include
scattered obstacles, gaps, staggered gates, U-shaped traps and moving
obstacles. Target motion includes random turns. Distractors and occlusion
are present in both splits.

## Local algorithm

Each of four independently trained circular-convolution action experts has
a learned sector attention distribution. A separate router, value module,
and executed-action RGB predictor have independent parameters/optimizers.
All intermodule inputs/targets are detached. PPO remains the policy-update
method; this experiment does not claim PPO, compound eyes, attention,
prediction or memory are new.

Past RGB is explicitly separate from current RGB and carries per-sector
age/gate. Proprioceptive rotation aligns history to the current eye frame.
No remembered image replaces the current image.

Two equal-parameter arms:

- fixed: sector history decays with a fixed age schedule.
- revisable: measure, BEFORE updating, the difference between next-RGB
  prediction error with history and with ALL sector history erased.
  Exponential local evidence modifies each sector's gate. Actual next
  images alone supply labels. This is a predictor-based omission comparison,
  not an unexecuted-action outcome or proof of causal behavioral benefit.
  Missing/changed details can be retained again on later observations.

Both input conditions also train the same predictor, with their actual
next image, so baseline predictions are not deliberately untrained.
The gain is localized by output eye after erasing all history together:
it is NOT an isolated causal omission of a single sector, because
neighboring circular filters can contribute. History here is an online
bounded trace. SQLite preserves raw details but V17 does not yet retrieve
cold database memories into the eye trace.

Both predictors see current RGB, aligned prior RGB, gated sector history,
age/gate and body state. Both train from the same actual executed action.
This particular retention rule in this continuous task is the hypothesis;
limited prior-work review cannot establish historical novelty.

Lock means the chosen expert's maximum-attention eye lies on the visibly
marked target. This is an operational attention metric, NOT proof of
recognition, stable identity, or hidden-target localization. Also measure
attention confidence, visibility-conditioned lock fraction, loss and
reacquisition delay. Pixel/control outcomes must improve jointly before
claiming useful tracking. Avoidance is actual collision/impact frequency,
and contact/safe contact are physical outcomes. Report attacked scenes
separately; total impact frequency includes peaceful scenes.

## Frozen run

Model seed 171, world seed 671000, 72 parallel bodies, 128-step rollouts,
3 PPO epochs, gamma .995, GAE lambda .97. Milestones:
2048 / 8192 / 16384 steps per body, respectively
147456 / 589824 / 1179648 actual training transitions PER ARM.
Both arms start untrained; V16 coordinate policies are incompatible.
At the final milestone each body accumulates 32.768 simulated minutes,
distributed over 16 two-minute segments. Shared weights see all bodies.
These durations still do not represent long biological development.

Frozen development readouts use 24 fixed held-out combinations, 512 steps
each at the initial and all milestone weights. No parameter or utility
feedback learning occurs in probes; observation history can update.
Save preupdate predictions, errors and actions without rewriting them.
Probe feedback remains a separate dataset. No seed search.
A sensor-removal check at final weights may test input use, but is not
an isolated proof of the proposed mechanism.

Raw before/after RGB and sensor states are SQLite records; NPZ contains
action/outcome/preupdate telemetry, not redundant RGB arrays.
Keep source hashes, all intermediate curves, failures and final weights.
Independently audit SQL row counts, NPZ outcomes, chronology, visibility
denominators, attack-scene metrics and per-segment contact history.
Publish complete snapshots with tools/archive_snapshot.py. Only delete
verified data with the existing SHA-256 cleanup workflow.

## Prior work checked

FlyGym official vision tutorial:
https://flygym.readthedocs.io/latest/tutorials/vision_basics.html
It documents ommatidial rendering/processing. We reuse the general principle,
not its published training task. The current 2D ring sacrifices elevation,
real optics and body morphology to examine the retention/learning rule.
The prior multiple-scene literature check remains in adaptation_v13.

## Implementation plan

- [x] world.py: occluded RGB ring; continuous contact/impact physics;
      randomized broad concurrent scenes, sensor/metric boundary.
- [x] learner.py: independent circular modules and action-conditioned
      predictor; motion-aligned revisable sector history.
- [x] run.py: continuous sessions, SQLite raw experience, matched arms,
      frozen milestone readouts, source/model provenance.
- [ ] test_contracts.py: rear visibility, seam/occlusion/distractor color,
      rotation, no privileged policy input, factual local update,
      nonterminal hits/contacts, random attack times, index ownership.
- [ ] Execute declared CUDA run once; preserve failed hypotheses.
- [ ] Audit, report trajectories/process, push code and full verified
      archive; clean only confirmed remote-backed bulk files.
