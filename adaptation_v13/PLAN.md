# V13 Process Learning Implementation Plan

> WITHDRAWN, NOT EXECUTED, 2026-10-13.
> The user requested research before experiments. The old sequential protocol
> below is superseded. No implementation or training is authorized by this
> historical checklist. Current state: STATUS.md.

> Execution: inline in the current task. The user has approved the direction;
> do not create subagents, repeat approval requests, or rerun V12.

**Goal:** Observe whether local learning produces better choices through
initial practice, changed dynamics, and returning environments.

**Architecture:** Inherit V12 raw memory, four independent predictors and
local gradient helpers. Add a 33-plan class/record selector, factual pending
credit, continuous reversal stream and process-first reporting.

**Tech Stack:** Existing E: CUDA Python, PyTorch 2.3.1, NumPy, SQLite, Matplotlib.

**Spec:** `adaptation_v13/PROTOCOL.md`.

## Global Constraints

All data/cache/logs on E:. Preserve V12 sources and immutable archive.
Only factual executed outcomes train networks. No gradient crosses modules.
No fixed entropy target, annealed exploration, biological-dynamics claim,
seed sweep, or selected best checkpoint. Upload/readback before deletion.

## Tasks

- [ ] `test_contracts.py`: write and observe failing contracts for missing V13
  `Stream`, `Plans.probabilities`, `Pending.add/observe`, `Practice.save/load`.
  Probability test sums plan mass by operation for unequal candidate counts;
  credit test feeds eight known windows and checks start/end time and target.
  Continuation test compares uninterrupted and restored CUDA state exactly.
- [ ] `core.py`: reuse V12 `Memory` and `LocalPredictor`; `Stream.observe`
  switches visible risk and future cue mapping at the declared schedule.
  `Plans.features` builds detached class and record inputs;
  `Plans.probabilities` returns legal normalized sampling probabilities and
  greedy/exploration diagnostics.
- [ ] `run.py`: `Practice.advance(until)` observes, plans, predicts, acts,
  records then updates. `Pending.observe` emits only complete factual spans.
  Save raw SQLite and compressed per-phase records; checkpoint at each phase.
  No resets at environment changes. Freeze source hashes before training.
- [ ] Run `test_contracts.py`, fix engineering errors before formal data.
  Execute `run.py pilot.json` once for all three continuous configurations.
- [ ] `analyze.py`: independently audit raw physics, window rewards, plan
  legality/probability, source hashes and directory reconstruction. Emit
  `AUDIT.json`, `SUMMARY.json`, all-cycle curves and `RESULTS.md`; plot process
  trajectories and write Chinese interpretation including negative findings.
- [ ] Commit/push small artifacts. Prepare/publish `adaptation_v13` snapshot,
  verify remote SHA-256, run scoped cleanup, record receipts and push status.
