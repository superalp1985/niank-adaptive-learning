# V9 confirmation decision

Date: 2026-10-12. No final probes or confirmation streams have been opened.

All four development conditions completed without execution failures.
Recurring rereads produce actual repeated fading/return events. Consolidation
reduced fine-reference drops from 112 to 73, but did not improve aggregate action
performance in this one small run. Keep all four controls in confirmation.

Freeze model, learning rate, retrieval thresholds, decay, lifetime and evidence
rules. Confirmation uses three existing initialization seeds and three new world
pools, both 128-step and 512-step budgets. Report entire learning trajectories,
fixed-probe before/after scores and all conditions. No post-hoc best-seed choice.

After development, only logging was expanded: serialize evidence traces and store
every real chosen-action experience in a separate SQLite practice table.
No algorithm behavior changed. Initial raw images stay in label-free databases.
Chosen-action outcomes are legitimate past experience; unexecuted action truths
are not stored in the learner's experience database.
