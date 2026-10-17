# V24 Delayed Observation Implementation Plan

> **For agentic workers:** use superpowers:executing-plans in this session.
> User authorization already covers this experiment and publication.

**Goal:** Test further practice and delayed physical observation credit from one
complete V23 active state.

**Architecture:** Reuse the V23 runtime and physical predictor. Add a persistent
multi-horizon consequence buffer and zero-initialized readouts sharing learned
attention. Preserve a mechanism control with identical additional exposure.

**Tech Stack:** Existing Python, CUDA PyTorch, NumPy, SQLite, Pillow.

**Spec:** [DESIGN.md](DESIGN.md).

## Global Constraints

- Preserve completed versions and source runtime bytes.
- All generated files, temporary files and caches stay on E.
- No privileged world state supplies training targets.
- Episode terminals flush delayed targets before reset.
- Freeze all learned parameters during diagnostic observations.
- Preserve failures; no rollback or selection by frozen performance.
- Publish, remote-read verify, then selectively clean.

## Task 1: Multi-horizon physical credit

Files: `tracking_v24/credit.py`, `tracking_v24/test_credit.py`.
Interface: `ConsequenceWindows.start(step,x,actions,before,episodes)`,
`advance(step,after,events,terminals)` returning matured dictionaries.

- [x] Test delayed maturation, exact physical sums, terminal truncation and
  serializable partially matured windows, then implement these behaviors.
- [x] Test several bodies with different episode boundaries.

## Task 2: Shared attentive readouts and runtime

Files: `tracking_v24/neural.py`, `tracking_v24/session.py`,
`tracking_v24/test_runtime.py`.
Interfaces: `DelayedLearner.upgrade(agent,config)`,
`train_delayed(rows)`, `Session.fork/restore/advance`.

- [x] Test zero-initialized policy preservation, attention gradients, optimizer
  preservation and frozen action selection before implementing readouts.
- [x] Test complete-state resume with pending delayed windows and terminal
  targets, then implement logging and runtime hooks.
- [x] Preserve original body-action selection; report auxiliary steps separately.

## Task 3: Finite paired practice

Files: `tracking_v24/run.py`, `tracking_v24/evaluate.py`,
`tracking_v24/practice.json`, `tracking_v24/audit.py`.

- [x] Save runtime sources and source identity, refuse completed reruns and
  resume only matching configurations.
- [x] Use immutable milestone weights for frozen observations; test parameter
  freeze and training RNG isolation.
- [x] Execute both finite branches, preserve all updates, settle pending credit.
- [x] Audit complete physical records and all new delayed targets.

## Task 4: Report and storage

Files: `tracking_v24/analyze.py`, `tracking_v24/REPORT_V24_ZH.md`,
`tracking_v24/STATUS.md`, versioned archive receipts.

- [x] Report quarter-by-quarter and frozen behavior, mechanisms and limits.
- [ ] Commit/push code and report, prepare and publish complete archive.
- [ ] Verify remote parts, clean only selected verified duplicates, retain
  source/continuation runtimes and milestone weights.
