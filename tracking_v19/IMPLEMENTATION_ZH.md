# V19 Implementation Plan

> For agentic workers: use the existing execution and verification workflow;
> do not rerun completed configurations.

**Goal:** Test whether revisable disagreement protection changes the learning
path under an equal local-memory access budget.

**Architecture:** Reuse the V17 world and detached policy modules. Introduce
local input interventions, three bootstrapped predictors and two memory scores.

**Tech Stack:** Existing E-drive CUDA PyTorch, NumPy, Pymunk, SQLite.

**Spec:** `tracking_v19/PROTOCOL_ZH.md`

## Constraints

All outputs on E. Two arms, identical common seed checkpoint, 72 bodies,
1024 shared calibration steps and 8192 additional steps per arm.
Preserve raw experience and failures. No online test-label optimization.

## Deliverables

- [x] Read V18, identify scientific limits, check HSAM and Anthropic sources.
- [x] Register protocol, budget, initial weights and operational definitions.
- [x] `learner.py` and `test_memory_contracts.py`: input-group coordinates,
  recoverable traces, 64-eye budget, stable bootstrap and detached predictors.
- [x] `run.py`: raw experience and input-intervention evidence, common
  calibration, independent arms, phase observations and source snapshots.
- [x] Focused tests: memory changes only after the factual next frame,
  read-cache does not rotate twice, probes do not train weights, raw replay
  reconstructs the exact accessed memory.
- [x] Complete both finite training budgets and all registered observations.
- [x] `audit.py` and `analyze.py`: independently recompute evidence, access
  decisions and metrics, describe all phase/scene tradeoffs.
- [x] Report results and record the limits of the forgetting/relearning claim.
- [ ] Upload complete archive, verify every remote part,
  clean verified bulk files and update the research entry point.
