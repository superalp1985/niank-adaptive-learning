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

## Implementation checks before practice

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

Full paired practice used the frozen implementation above. The following
observations record its progression; completion is recorded at the end.

Baseline completed all 32768 new training steps/body plus 64 settling steps.
Equal 8192-step practice intervals recorded 26/30/34/27 contacts and
20/23/32/20 safe contacts. Collision frame fractions were
0.049469/0.034531/0.039825/0.039413; impact events were 139/137/135/155.
The common-scene frozen probes had 0/0/1 contacts. The final independent
scene seed had 3 contacts, 2 safe, 0.018677 collision fraction and 14 impacts.
It has no initial-model result on that final seed, so 3 contacts alone do not
establish improvement there.

Active practice followed baseline in the same run. Initial common-scene
physical metrics match baseline exactly, including progress and collision.
Early observation at 2621 new steps (5248 decision/body samples) measured
mean attention entropy 4.155634 versus uniform 4.158883, mean consecutive
attention L1 change 0.010675 and head-limit occupancy 0.269817.
These are interim activity measures, not evidence of useful active vision.
The predictor has 101954 parameters in baseline and 184942 in active,
excluding additional state/spike mechanisms.

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

Active practice completed its first three 8192-step intervals: contacts
37/35/32, safe contacts 28/25/25, collision fractions
0.072449/0.040588/0.062057 and impacts 141/141/129.
The third interval settled 128 learning decisions: 72 pause, 31 recent and
25 recall. Its 56 learning trials retained all 23 negative-gain attempts;
193 predictor updates were credited. More experience is not automatically
more optimization, and these outcomes do not show monotonic improvement.

Attention analysis now distinguishes pre-action sampled motion from a new
outward command at a head limit, and includes all eight bodies.
`ACTIVE_VISION_RESEARCH_ZH.md` records primary-paper references and candidate
longer-horizon observation credit. It is not a change to the running algorithm.

## Completed practice and observations

Both arms completed 32768 new steps per body and 64 settling steps; the paired
process and supplemental frozen-exploration process exited successfully.
Active training contacts across four intervals were 37/35/32/23 and safe
contacts 28/25/25/18. New training totals were baseline 117 contacts, 95 safe,
4.08096% collision frames, 566 impact events; active 127 contacts, 96 safe,
6.44455% collision frames, 548 impact events.

Attention became differentiated and heads physically moved, but the middle
checkpoint's behavioral gains partly receded. In the independent final
greedy observations, baseline had 3/2 contacts/safe contacts, active 3/1,
active uniform attention 3/3 and active fixed head 3/1. Supplemental frozen
training-exploration observations had baseline 4/4 and active 3/1, with all
learned parameters unchanged. These results do not establish useful learned
attention or head control, and total practice remains under seven minutes
per body including history.

The full record audit covered 657408 physical rows, 656912 consequence
windows and 1280 settled learning choices including duplicated history.
All frozen parameter checks passed, all negative learning trials were kept,
the 14 runtime source hashes per arm matched and the V22 source was unchanged.
The Chinese report and compact measurements are complete. Archive and
selective cleanup receipts are tracked in STATUS.md; no running algorithm
source was changed during analysis.

## Publication complete

The immutable snapshot contains 595 files and 15 ZIP parts. GitHub workflow
38966723767 succeeded; all Release asset metadata matched and every part was
read back in full with matching size and SHA-256. Selective cleanup removed
12 observation databases and verified upload staging, freeing 2186486266
bytes including duplicates. Both main databases, all 12 weights and the V22
continuation source remain. All 583 retained snapshot files matched before
these publication-status updates. The temporary upload branch was deleted;
Release data and all receipts remain available. See STATUS.md.
