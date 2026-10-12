# V15: observed temporal changes and factual risk allocation

Date: 2026-10-13. Follow-up to V14 after reviewing its development curves.
No old scientific run is repeated. V14 is archived before V15 produces data.

The physical task, 18/9 combination split and simulator are unchanged.
The held-out combination family has already been inspected in V14: V15 is
development iteration, not independent confirmation of a broad generalization
claim. Seeds for new world/probe samples change by declared version, not by
searching for a favorable result.

## Failure to address

V14 reached targets frequently, but retained obstacle contact and projectile
impacts. Its action path had no explicit immediate collision/impact prediction,
and old retrieved transitions were not necessarily current motion evidence.
The narrow question is whether current temporal observations plus a separately
trained factual risk estimator improve physical outcomes.

## Common temporal input

Both arms use the same 80-dimensional input: 32 current measured sensor values,
32 differences from that same body's immediately preceding observation,
16 retrieved past-transition features. Differences reset to zero on episode
reset; other bodies' observations are inaccessible. Object slots are sorted by
current proximity, so slot differences are only approximate motion estimates,
not perfect tracked identities. This limitation is reported.

Both arms have identical action experts, routing, state-value, delta predictor,
risk estimator, independent optimizers and learning budgets. Raw raster inputs
remain archived rather than becoming a trained visual encoder.

## Factual risk allocation

The risk estimator outputs two logits per action: next-step obstacle contact
and projectile impact. Its targets are only the actual executed action's
observed next-step events. Nonexecuted actions do not receive labels.
Natural-frequency replay keeps 16,384 executed transitions; four minibatches
of 768 samples are presented after each rollout's policy updates.
No positive oversampling or simulator counterfactuals.

Arms:

- `temporal`: risk estimator trains, but its outputs do not alter action logits.
- `risk_alloc`: subtract `2*p(collision) + 4*p(impact)` from each expert's action
  logits before action sampling. The correction is detached. It is a declared
  fixed interface gain, not itself a learned global optimization.

During PPO updates, corrections saved at action time are reused, so changing
the estimator cannot silently change the policy denominator. Risk learns after
policy updates; value targets and prediction updates remain detached.
Frozen probes fit nothing, including utility and risk estimates.
The risk system predicts immediate events only; it does not guarantee safe
paths, long-term safety, calibrated probabilities or global optimality.

## Budget and evaluation

Model seed 151, world seed 651000, 48 bodies, 64-step rollouts, 4 PPO epochs.
Continuous milestones 1024/4096/8192 steps per body, same budget as V14.
Both arms start with matching parameters. Probe samples are fresh and fixed
across checkpoints and arms. Track contact success, collision steps, projectile
impact, observed-object motion failures, prediction and risk errors, every scene
and the entire learning process. Do not select the best checkpoint.
At the last risk_alloc checkpoint, also run a frozen held-out probe with the
correction disabled and the same exogenous probe seeds. This is an inference
intervention, not another training configuration.

Separate the benefit of contact from safety. A useful safe-contact measure is
the fraction of completed episodes with target contact and no obstacle collision
anywhere in that episode. Report it beside ordinary contact and impacts; do not
retroactively redefine V14's declared primary metric.

Archive and independently audit all transitions including failed variants.
All data/cache/dependencies stay on E:. Upload full archive, read back every
part's SHA-256, then clean only verified local files.

## Prior work boundary

Model-based risk and modular safety decisions have extensive prior work;
this is a project-specific learning-interface iteration, not a claim of having
invented risk prediction or a new theorem. See V14 PPO and GAE sources.
The learned risk module is intentionally not an oracle safety shield.
