# V14 Implementation Plan

Use superpowers:executing-plans inline; user authorization already includes
implementation, experiments, iterations, upload and verified cleanup.
Do not dispatch subagents.

Goal: learn mixed-scene obstacle/attack avoidance and marked-target contact.
Architecture: Pymunk worlds, local CUDA modules, factual temporal directory,
SQLite observations, independently audited episode results.
Spec: PROTOCOL.md. Data/cache/dependencies stay on E:.

- [ ] Write behavioral contracts in test_contracts.py; confirm missing implementation.
- [ ] Implement world.py: visible sensors, physical contact, 18/9 split, continuous worlds.
- [ ] Implement memory.py: raw storage, per-body ownership, recent/directory retrieval.
- [ ] Implement learner.py: independent expert/router/value/predictor updates and GAE.
- [ ] Run contracts and a separately labeled engineering smoke with no efficacy claim.
- [ ] Freeze pilot.json and source hashes, run two continuous configurations once.
- [ ] Audit transitions, produce learning curves and scene-wise report.
- [ ] Identify next algorithm change from the process; never select a lucky seed.
- [ ] Publish code and full archive, verify all remote hashes, then clean bulk files.
