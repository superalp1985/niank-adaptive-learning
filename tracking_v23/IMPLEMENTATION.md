# V23 implementation

Implementation is confined to `tracking_v23`. V22 source and completed data are
unchanged. The full 32768-step paired experiment is running at
`results/extended_practice_20261017`: baseline practice and observations are
complete; active practice is still in progress. Final results belong in the
result report, rather than being inferred from implementation checks.

## Runtime

Run from the experiment repository with:

```powershell
$env:TEMP='E:\niank-temp'
$env:TMP='E:\niank-temp'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH="$PWD\tracking_runtime"
& 'E:\WangcaiOfficeAssistant\WangcaiDocumentWriter\.venv-train\Scripts\python.exe' `
  -m tracking_v23.run --output tracking_v23/results/extended_practice_20261017
```

The default config runs baseline then active vision, sequentially. Each starts
at the complete source runtime step 8256, receives 32768 additional training
steps per each of 8 bodies, then 64 steps to settle delayed learning credit.
`training_only_summary.json` ends at 41024; final runtime ends at 41088.
`--arm baseline` or `--arm active` can execute one arm. Reusing the same command
resumes incomplete work; completed configurations are rejected. A paired resume
skips a completed arm only after checking its manifest and source identity.

Checkpoints occur every 1024 new steps; model milestones are at 0, 8192, 16384,
24576 and 32768. Stage probes use 512 steps at 0/16384/32768; the final independent
scene set uses 1024 steps. Active final diagnostics force uniform attention or
head hold for the same 1024-step scenes. Probes use isolated RNG and cannot
update predictor, internal-state, spike or meta parameters.

`python -m tracking_v23.evaluate --arm-directory PATH --stage 32768` runs a
separate reproducible probe. `--diagnostic uniform` and `--diagnostic head_hold`
are available for the active arm.

## Mechanism and migration

- `HeadWorld`: 27 commands, body `a % 9`, head `a // 9`; hold/left/right, bounded
  yaw +/-pi/2, speed pi radians/second, integrated on the 1 ms physical grid.
  This is a virtual sensor joint, without neck mass, anatomy, torque or energy.
- The native 1024x512 RGB panorama is circularly shifted by head yaw rounded to
  a native pixel. Every pixel survives. Only RGB plus reported head sensors
  enter the policy. Exact inverse pixel rotation before the V22 retina keeps
  all 64 global sectors and outcome directions in body coordinates.
- Head-centered detail uses the forward 256x256 native crop, pooled to 3x8x16.
  It gives head orientation a computational consequence; pooling is not an
  increase in sensor resolution.
- Predictor input is `[global512, head4, detail384, body8, latent24, spikes32]`.
  The final 32 spike channels retain their eligibility-trace interpretation.
- Trainable spatial scores depend on raw-image-derived sectors and bodily,
  latent and spike state. Neutral initialization is uniform. Multipliers are
  `0.25 + 0.75*64*softmax(score)`, so the global periphery never disappears.
- Every old encoder column and head output is explicitly mapped. Each old
  body output is copied to all three head choices; inserted weights are zero.
  Adam step counts, moments and parameter-group options are copied with the
  same mapping. Historical missing head/detail observations are zeros.
- Old replay facts remain untouched and migrate lazily. Pending observations,
  unfinished records and pending trial-before models migrate immediately.
  Full worlds, RNG, dynamic state, other optimizers and memory are retained.
  Continuation explicitly reenables learning after V22's settled checkpoint.
- Baseline fork preserves delayed sensory/drive state verbatim. The active
  fork adds current head/detail to the saved input without a new neural tick.
  Episode death resets use the new width and reset head state.
- Error evaluation batches the exact last surviving matured record per body.
  This preserves mathematical semantics but GPU GEMM grouping may introduce
  small floating-point differences from the old per-row implementation.
- SQLite stores all physical events and sampled attention distributions on
  action decisions. Summary metrics include new-experience deltas, head travel,
  motion fraction, attention entropy/change and per-scene outcomes.

## Validation

Commands use the environment above and `torch.set_num_threads(4)`.

```text
python -m unittest tracking_v23.test_vision tracking_v23.test_neural tracking_v23.test_runtime tracking_v23.test_persistence -v
python -m tracking_v23.smoke_source --output tracking_v23/validation/actual_source_learning_20261017 --steps 256
```

The initial focused run passed 14/14 vision and neural tests in 3.935 s.
The first runtime run passed 8/8 tests in 56.427 s. Adding two early-interruption
tests exposed a Windows SQLite connection leak: the connection context manager
committed but did not close, preventing preservation of the interrupted probe
directory. Explicit `close()` fixed it; the targeted regression passed in
2.159 s. Final combined test and actual-source results are recorded below.

Final combined run: **24 tests passed, 20.651 s, exit 0**. This includes both
early-interruption regressions; no tests were weakened or skipped.

Actual V22 source smoke: **passed for both arms**, 16 additional frames per body
per arm, including an exact save/restore comparison after frame 8. All 248
pending windows survived; input widths were 576 and 964. The original database
remained SHA-256
`9f6b104c938d15907c154fed2150e39055f7d3f722df0cb1ccf1f261f4e67ce7`.
Evidence is `validation/actual_source_20261017/CHECKS.json`, native PNGs and
per-arm GIFs. The inspected native image showed the target cross and green
distractors across the full panorama. This short smoke ends before a new
64-frame learning block; its uniform attention is expected and is not a
claim of learned benefit.

The suite covers complete pixel coverage, body-coordinate invariance, bounded
independent head motion, old predictions and Adam moments, learned attention
and detail dependence, peripheral floor, pending-trial migration, lazy replay,
death resets, unchanged baseline delayed state, exact V23 save/resume, frozen
weights, training RNG isolation, interruption recovery, finite paired budgets,
source preservation, completion rejection and code-manifest validation.

### Independent review follow-up

The review found two P2 interruption risks: JSON completion markers could be
truncated by direct writes, and SQLite backup could expose an unfinished final
database. `persistence.py` now writes JSON to a unique sibling, flushes and
syncs it, then atomically replaces the destination. Complete markers must parse
and have the required result content; invalid markers are retained under a
unique evidence name and recovery uses the existing manifest and committed
runtime. Invalid manifests are preserved and rejected; provenance is never
silently regenerated after a run exists.

Database backup now uses a unique sibling, validates SQLite integrity and the
runtime SHA-256, closes all connections, and only then publishes the file.
Interrupted temporary files remain available. A legacy incomplete bootstrap
database is preserved in an evidence directory and rebuilt only if no practice
milestone or progress evidence exists; damaged practiced state is not replaced.

Focused revalidation: **16 persistence/runtime tests passed, 23.890 s, exit 0**.
This covers interrupted JSON writes/publication, truncated arm/pair/probe
markers, interrupted database copying, conservative manifest failure and both
new and legacy bootstrap recovery. No implementation tests failed in this
follow-up. Vision/neural algorithms were unchanged.

Extended real-source validation:
`validation/actual_source_learning_20261017/CHECKS.json` records **256 new steps
per body per arm**, 12 predictor updates and 3 fresh-window credit settlements
in each arm. Baseline credit gains were approximately +0.199, +0.127, -0.032;
active gains were -0.090, +0.194, +0.024. Every update was retained. Both arms'
last 16 frames resumed exactly, including all Adam state tensors and physical
positions. The active encoder Adam moment is 128x964; all four new facts have
width 964, and all three gaze groups in both prediction heads changed weights.
All 248 source pending windows remained available and the source database hash
was unchanged. This validates actual learning continuation, not superiority of
the new mechanism. Native PNGs and 256-step GIFs accompany the JSON evidence.

## Practical limits

The head prior is identical for all three choices; head preference can emerge
only from exploration and learned future consequences. Neutral frozen ties
choose hold until learned output values distinguish head actions.

Attention activity, parameter changes and head travel do not establish useful
behavior. The matched scene outcomes and uniform/head-hold interventions must
be inspected. Diagnostic trajectories naturally diverge after an intervention.
The two arms also differ in parameter count and detail representation, so this
does not separately identify the causal contributions of those changes.

All probe GIFs are reduced display previews; `actual_input_*.png` are native
sensor images. Actual-source smoke databases are temporary; compact sensor
images/GIFs and their JSON audit are retained in `validation`.
