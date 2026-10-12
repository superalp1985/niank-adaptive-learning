# V16: more practice versus whole-episode safe-contact feedback

Date: 2026-10-13. Authorized algorithm iteration after V15's negative risk result.
V15's temporal final checkpoint is the shared warm start. Its hash is recorded.
New physical worlds and optimizers are explicitly created: this is not exact
mid-trajectory continuation. The warm start has 393,216 prior new experiences.
Both configurations receive another 393,216, with readouts at
1024/4096/8192 new steps per body.

## Question

V15 learned target contact but often touched obstacles during the episode.
Can genuine whole-episode consequences improve safe target contact, beyond
merely giving the same architecture more practice?

Common: V14 physical task and 18/9 split, V15 temporal+directory observation,
four independent action experts, risk module trained but unused in the policy,
48 worlds, 64-step rollouts, 4 PPO epochs. Model seed 161, world seed 661000.
Both start from identical V15 temporal weights and new optimizers.

- `more_practice`: original physical reward.
- `safe_contact`: original reward plus 16 for target contact with no obstacle
  collision anywhere in the episode, minus 8 for target contact after any
  collision. A same-step projectile impact still makes contact unsuccessful.

Reward modification uses only already executed physical outcomes. No hidden
best action or unexecuted simulator trajectory is used. A cumulative prior
collision flag is observable proprioceptive history: sensor slot30 becomes
ever-collided in BOTH arms. Both therefore have identical state information.
This differs from V15; V15 versus V16 cannot isolate only learning quantity.
The contemporaneous more_practice arm isolates the changed reward within V16.

Keep V15's gamma=0.98 and GAE lambda=0.95 in both arms. Episode feedback still
propagates through actual trajectories; it can be weak for early events.
There is no learned global gradient across modules.

Primary operational observation: fraction of completed episodes with marked
target contact and no obstacle collision in the full episode. Also report
ordinary contact, impacts, timeouts, collision steps and per-scene results.
Do not call a reward increase evidence unless physical safety also improves.

All probe feedback remains excluded from weight/utility fitting. V14/V15's
combination family is already development data; fresh declared samples are not
a new task-family confirmation. Record initial warm-start probes, intermediate
curves, final checkpoint and failures. No favorable-seed search.

Complete raw observations and executed actions remain in SQLite.
Audit base physical reward and revised episode reward separately from recorded
contact/collision history. Publish and verify all parts before deleting bulk data.
All data, caches and checkpoint copies stay on E:.
