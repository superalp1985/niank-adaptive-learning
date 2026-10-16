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
