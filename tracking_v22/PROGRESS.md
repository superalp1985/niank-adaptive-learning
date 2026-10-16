# SDD ledger - plan: tracking_v22/PLAN.md

Spec: SUBJECTIVE_STATE_DESIGN_NOTES_ZH.md and the user's approved instructions.

Ruling: use an isolated new branch and experiment module in the existing E:
checkout. The native worktree tool is tied to the C: chat folder; duplicating
the environment or archive is unnecessary. Older experiment modules stay intact.

Ruling: implement the physical world and renderer with one dedicated agent,
while the coordinator owns the independent learning modules and runner.
The subagent-driven-development skill requires this delegation.

| Tasks | Shared interface / self-consistency | Finding |
|---|---|---|
| 1 / 2 | RGB [512,1024,3], proprio [8], 100 Hz | Fixed explicitly above |
| 1 / 3 | World step and complete physical checkpoint | Restore must be tested |
| 2 / 3 | Local state, optimizers, RNG and delayed controller credit | All checkpointed |
| 3 / 4 | Same sensors, split training and probe worlds | Frozen probes isolated |
| 4 / 5 | Immutable source/config/output directories | Archive after runs |
| 1 | Fine physics plus independently rendered frames | No old-frame expansion |
| 2 | Stateful learning plus delayed facts | No hidden coordinates as inputs |
| 3 | Full recovery and bounded recordings | SQLite transactions |
| 4 | New runs for changed mechanisms | No overwriting failed attempts |
| 5 | Upload before deletion | Existing archive tooling |

Task 1: complete. Native RGB/sensor audit at E:/niank-data/v22/sensory_audit_20261016.
Independent review found solver-cache and subframe-event issues; both fixed.
Ruling: canonical checkpoints deliberately reset unsaved Pymunk numerical
contact caches in BOTH live and resumed branches. This is reproducible but
not numerically identical to an uninterrupted warm-cache simulation.

Task 2: implemented, five tests pass; independent review underway.
Task 3: complete. Sixteen world/neural/runtime checks pass. Full runtime
continuation reproduces positions, weights, choices and membranes exactly.
Fine-clock audit: .1ms versus .05ms yielded .2873% spike-count difference
for the tested varying drive; this is limited numerical evidence.
Task 4: started. Fixed-schedule pilot running in
E:/niank-data/v22/pilot_fixed_20261016 with immutable source snapshot.
Task 5: pending.

User steering during the pilot: autonomous learning needs more trial-and-error
and adaptation time; early negative feedback must not decide eventual success.
Ruling: abandon the proposed one-window rollback BEFORE running that variant.
V22-B will retain early unsuccessful updates, collect12 completed learning
trials before LR adaptation, adjust only every4 learning trials, keep30%
meta-controller exploration, and extend practice to8192 steps per body.
The fixed reflex fades gradually only after96 retained local updates.
Acceptance means retention, not proof that the update is beneficial.

Both short pilots completed. Frozen probes: fixed2 contacts/1446 collision
frames/5 impact frames; autonomous3 contacts/1453 collision frames/5 impact
frames, each4096 body frames. Too small for endpoint ranking.
Task1-3 files plus21 tests complete; scoped review found no remaining blocker
for the planned8192-step configuration. Archive pilots before extended practice.
Outputs moved intact from E:/niank-data/v22 into tracking_v22/results so the
existing repository archive tool can cover all files without duplicating them.

2026-10-16 completion update:
- Pilot archive research-v22-spiking-pilots-20261016 was uploaded and fully
  read back with matching hashes. Verified pilot databases/staging cleaned.
- Task 2 review complete. Full suite now23 tests, passing in8.728 seconds.
- Task 4 complete for this finite iteration:8192 training steps/body,
  64 settlement steps,1024 frozen steps,8 bodies.
- Five common-scene frozen observations at0/2048/4096/6144/8192,512 steps each,
  verified all learned parameter hashes unchanged. No duplicate completed runs.
- 359 local updates retained,32 negative trials,29 recalls,7 cold returns.
  Initial-to-final frozen contacts1/3/1/1/0: no stable pursuit gain established.
- Source/runtime/event integrity, final settlement and frozen-parameter checks
  passed on the actual completed data; completion_audit.json records results.
- PATIENT_PRACTICE_REPORT_ZH.md records practice limits, shared final probe
  scenes, hand-engineered prior fade and next distinctions to investigate.
- Task 5: result documentation complete; full archive publication/readback and
  selective cleanup use research-v22-patient-practice-20261016. Its archive
  receipt, rather than this pre-publication note, determines completion.

Task 5 complete. Archive research-v22-patient-practice-20261016 contains189
files,485467344 source bytes,92447214 ZIP bytes in3 parts. GitHub Actions
38926083951 succeeded; all parts fully read back with matching SHA-256.
Verified cleanup removed only6 observation databases (370434048 bytes)
and disposable upload staging (275613992 bytes). Main runtime/facts and
all weights remain on E:. Temporary upload branch removed; release retained.
Direct GitHub access reset connections; publication used the existing local
SOCKS proxy. Python readback disabled automatic HTTP proxy discovery while
using that same SOCKS route. No machine-wide network settings changed.
