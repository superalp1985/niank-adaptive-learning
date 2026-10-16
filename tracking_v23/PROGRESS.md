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

Implementation and behavior tests are underway. No new practice results yet.
An independent implementation review will precede the long paired run.
