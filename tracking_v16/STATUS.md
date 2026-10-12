# V16 status

2026-10-13. V15 fully archived/read-back and verified cleanup before V16 data.
Whole-episode contact feedback contracts and CUDA smoke passed.
Training warm starts share the V15 temporal final checkpoint, with fresh
worlds, memory/replay and optimizers. Source manifest is frozen.

Both more_practice and safe_contact completed with identical additional volume.
All training/probe records passed independent audit, including base physical
reward and the revised whole-episode feedback. Report and post hoc failure
diagnostics are complete.

Full release research-v16-safe-contact-20261013 has nine read-back verified
parts. Verified cleanup recovered 580,330,935 bytes of bulk data and
1,006,702,907 bytes of staging files. The two final model.pt files and
seed/model.pt remain as the small working set; all checkpoints exist remotely.
Receipts: archives/research-v16-safe-contact-20261013.

Mixed outcome: safe_contact improved collision-free target-contact episodes,
but increased projectile impacts and collision-contact steps relative to
more_practice. No universal winner or completed real-robot capability.
Next research question: obstacle recovery and revisable local path memory.

The immutable full archive keeps the initial plot; use the corrected main-branch
PROCESS_CURVES_CORRECTED.png to avoid clipped second-arm values.
