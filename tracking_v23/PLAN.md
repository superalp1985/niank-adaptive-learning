# V23: extended practice and active vision

User direction: more real practice; learn visual attention and independent head
movement. Continue functional subjective state, fine-clock spikes, reversible
memory and physical consequence learning. No claim of consciousness.

## Global constraints

- Preserve V22 code and completed data. Work only in tracking_v23.
- Start from the complete V22 patient_practice_20261016 runtime checkpoint,
  including worlds, optimizer, memory, pending windows and RNG.
- Two arms: unchanged mechanism continued; active vision continued.
- Each arm receives 32768 NEW steps per body, eight simultaneous bodies,
  native 1024x512 RGB at 100 Hz, 1 ms physics and 0.1 ms spike grid.
- New practice is 4 times the previous 8192 training steps. This remains a
  limited learning budget, not evidence that training is sufficient.
- Active vision preserves all-around perception. Head rotation does not
  reveal space outside an already complete panoramic field.
- Raw observations must influence trainable attention inside the predictor.
  Attention cannot be a visualization disconnected from learning.
- Head-relative visual cues must be converted to body-relative motor cues.
- Head actions are learned through experienced future consequences, without
  privileged target coordinates or scripted target-facing commands.
- Keep all updates, including short-term harmful ones. Credit uses later
  fresh windows. Frozen evaluation does not train or influence selection.
- Physical contact, collision, impact and learning process are primary.
  Baseline agreement and changing attention alone are not success metrics.
- Everything, including caches and temporary data, stays on E.
- Upload complete results, verify remote bytes, then selectively clean.

## Design

Use 27 joint actions: action % 9 is the existing body command;
action // 9 selects head hold, left or right. The head has bounded relative yaw,
finite angular speed and a reported motion state. Body orientation remains
independent. A forward native-image detail channel gives head orientation
a computational consequence while preserving the global panoramic pathway.

The predictor uses trainable spatial attention over the 64 existing angular
feature sectors, conditioned on internal and bodily state. Preserve a nonzero
peripheral contribution. A shared encoder predicts nine future consequences
for each of 27 joint actions. The original body reflex remains disclosed,
identical across the three head choices. Head choice is not given a target
direction prior.

Preserve the final 32 spike features in every input and replay record.
Map legacy model columns, outputs and Adam moments explicitly. Legacy facts
with no head/detail observations receive zeros for those unavailable channels;
do not fabricate historical visual detail. Preserve the source database.

## Execution

1. Implement and test independent head physics, visual coordinates, learned
   attention, compatible model/optimizer migration and complete runtime forks.
2. Verify actual sensor images and continuation/reset behavior. Run finite
   paired practice with checkpoints every 1024 steps and weights every 8192.
3. Observe frozen agents at start, +16384 and +32768 on the same held-out
   physical scenes, plus a separate final scene set. Report paired scene
   results, head travel, attention entropy/change and dependence on inputs.
4. Audit learning records and parameter changes; report failures and remaining
   confounds. Archive, remote-read verify, and clean only verified duplicates.

## Limitations fixed before results

This is a two-arm mechanism exploration, not a causal separation of attention,
head motion and extra parameters. Local feature pooling is not sensor resolution.
The world is 2.5D with engineered color/morphology features and a disclosed body
reflex. More data cannot guarantee improvement. Repeated checkpoint resets
canonicalize unpersisted physics solver caches in both arms.

## Progress

- Design and budget: complete.
- Implementation, behavior tests and independent review: complete.
- Baseline extended practice and planned frozen observations: complete.
- Active extended practice and observations: in progress.
- Final analysis, audit, full-result upload and verified cleanup: pending.
