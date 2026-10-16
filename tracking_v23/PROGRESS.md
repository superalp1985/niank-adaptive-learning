# V23 progress

## Decisions before results

- Keep the existing checkout and use branch `codex/v23-active-vision`; the
  starting tree was clean at `b18298e`. Preserve all V22 sources and results.
- Finite budget: 32768 new steps per body per arm, eight parallel bodies.
  This is four times the prior training interval, with total practice about
  five times the prior training interval. It is not a claim of enough data.
- Upgrade in a new directory, retaining the existing physical simulation,
  fine-clock spikes, prior, memory and local credit scheme.
- Add head actions, conditional spatial attention and a head detail channel
  together. This tests their combined usefulness; final frozen interventions
  help diagnose dependence but cannot fully separate learning contributions.
- Head dynamics model a bounded sensor direction, not fly neck biomechanics.
  Panoramic rotation is an exact native-pixel permutation.
- Preserve baseline dynamic state at the continuation boundary. Recomputing
  sensory drive there would alter the source state, so only add new active
  input channels.

## Completed checks

- Source repository and remote main agree at `b18298e`.
- Source database hash:
  `9f6b104c938d15907c154fed2150e39055f7d3f722df0cb1ccf1f261f4e67ce7`.
- New record auditor was exercised against the existing source: 66048 physical
  rows, 65800 learning windows and 128 learning decisions agree; all 32 negative
  trials retained. The source database was unchanged.
- Runtime timing sample on the current machine: 8-body native retina about
  0.022 seconds/frame; 8-world simulation/render about 0.063 seconds/frame.
  This is a small scheduling estimate, not an experiment result.

## Current status

Core implementation committed as `884a08f`. Twenty-four focused tests passed
in 20.651 seconds. A real-source smoke check continued both arms for 16 frames
and checked exact V23 checkpoint continuation, 248 migrated pending windows,
and unchanged source bytes. A native active sensor image was visually inspected.

Independent review found core requirements met, with two recovery issues:
non-atomic JSON completion/manifest writes, and a partially written first
database backup potentially blocking resume. Both are being fixed before
long practice. Fix commit `cdfbd7d` uses atomic JSON publication, validates
completion markers, validates temporary database backups before publication,
and preserves interrupted evidence. Sixteen persistence/runtime tests passed
in 23.890 seconds. The focused fix review passed, with no new blocking issue.

Real-source learning smoke: 256 new steps per body per arm, 12 new predictor
updates and 3 settled future-credit choices per arm. New active records are
964-dimensional, all three head-output groups changed, exact resumed state
matches, and the source database hash stayed unchanged. Mean attention change
was nonzero but near uniform; this is integration evidence, not useful behavior.

Full paired practice is now starting with the frozen implementation above.
No full new-practice results yet.

## Additional observation decision during training

The baseline midpoint greedy probe had zero contacts, collision fraction
0.07080 versus initial 0.11499, and unchanged 3 impact events. Training uses
stochastic actions whereas this probe is greedy. Add a separate final
1024-step frozen-weight stochastic probe for each arm on the same independent
final scenes. This is an exploratory diagnostic added after observing the
midpoint, not a replacement for the planned greedy results or a training change.
One focused freeze/restore test passed in 2.759 seconds; independent review
found no blocking issue. No predictor/state/spike/meta updates are permitted.

Added process analysis separates each 8192-step interval and individual
bodies. Attention/head charts exclude final settling steps and use a minimum
multiplier color range of 0.9 to 1.1 so tiny changes are not visually exaggerated.
