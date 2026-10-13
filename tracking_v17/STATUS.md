# V17 status

2026-10-13.

- Implementation: 360-degree planar RGB ommatidia with nearest-surface occlusion.
- Passed: 10 physical/sensor/local-learning/recording contracts.
- CUDA integration: 64 engineering transitions; not scientific results.
- Completed: both declared arms, 1,179,648 new physical transitions per arm.
- Passed: 2,482,176 raw training/probe row comparisons, chronology and metrics.
- Report: REPORT_V17_ZH.md and process/actual-eye figures.
- Post-run defect: revisable utility receives next-frame gains in the old
  frame, then rotates again. See POSTRUN_FRAME_AUDIT_ZH.md. This invalidates
  a clean inference about correctly aligned adaptive retention.
- Pending: complete GitHub archive/readback and verified cleanup.

Current versus historical image channels are separate. Oracle visibility
and coordinates appear only in outcome telemetry, never policy features.
Attention lock is a limited operational metric; identity-preserving tracking
through occlusion has not been established.

Training source, configuration and results remain immutable. Source hashes
match. Initial model tensors match across arms. Fixed-arm sensor removal
reduces contact rate from 7.609 to .448 per body minute.
