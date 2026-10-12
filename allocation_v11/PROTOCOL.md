# V11: resource-addressed delayed practice feedback

Date: 2026-10-12. Exploratory mechanism change, not a claim of novelty or efficacy.

## Hypothesis

V10 averaged recovery history across different experiences. V11 retains mutable
feedback for each database record and each matrix group, and feeds it to the
corresponding pointer/router. Compare this with pooling those same statistics
before selection. Both versions collect identical fields and have identical
parameters, operation sets and presentation limits.

At decision t, record the selected retrieval/fade record identity and groups
actually executed over the next four steps. After 32 real steps, associate the
discounted factual return with those resources:

    G[t] = sum(j=0..31, 0.97**j * reward[t+j])
    trace[id] <- 0.8 * trace[id] + 0.2 * G[t]

These are observed associations, not causal estimates of a detail's value.
Intervening decisions affect G. Expired identities never credit replacement
records. All traces can change with further practice. No immutable useful/useless
labels, no unexecuted counterfactual labels, no regret in either training target.

## Environment and controls

- Reuse V10 physical law, image/body modalities and 3 active memory slots.
- Extend cycles to 48 steps and addressable directory to 16 events (64 steps).
- Two observable flashes at steps 0 and 4; physical change starts at 8 or 20.
  Both cues are outside automatic FIFO coverage by step 20, but remain queryable.
- Cyclic phase remains observable. No hidden unannounced rule changes.
- Compare `pooled` and `addressed`, with 9 reserved matrix groups and identical
  initial predictor/controller weights and external random seeds.
- Readouts use executed physical outcomes; success uses evaluator-only regret
  <=0.02. Report full stream AND first four steps after physical change, where
  recovering an earlier cue should matter most.
- Train 64 worlds for 1024 then 4096 total steps, without resetting optimizer,
  worlds, replay, directory or RNG. A continuation contract checks split vs
  uninterrupted execution. The first stage is not rerun.
- Frozen probes use separate seeds, no gradients or replay learning, with seen
  combinations and held-out (0,0)/(3,3). Historical feedback can still update
  runtime traces. Held-out probes never modify training state or tune the method.
- One model/world seed per condition: descriptive exploration, no significance
  or broad generalization claims. Larger capacity follows evidence about routing.

The new directory/group features change V10's mechanism; this is not a direct
cross-version score comparison. Capacity remains preallocated; admission is still
a prefix restriction, not true tensor growth. Routing is by matrix groups, not
individual coefficients. No Qwen precision experiment or biological dynamics
simulation is claimed.

## Accounting and storage

Save selected actions, resource identities, delayed returns, physical transitions,
raw SQLite images, learning curves and resume states on E:. Count new experiences,
replay presentations, actual group use, controller/route MAC estimates, optimizer
elements and SQLite reads separately. MAC estimates are not hardware speedups.
Archive all results, including failures, verify remote hashes, then remove bulk
local copies while retaining final continuation checkpoints.

## Relation to prior work

The existing research plan already identifies delayed feedback, memory utility
learning and dynamic expansion as established directions (see the repository's
NOVEL_ALGORITHM_PLAN_20261012_ZH.md). V11 does not rerun those papers' experiments.
The narrow candidate difference here is attaching
the same mutable delayed record to retrieval/fading and routed matrix groups
within ongoing physical practice. Its utility is unknown.

If addressed feedback does not improve the delayed consequence window, do not
respond with a blind parameter sweep. Inspect whether the directory actually
recovers old cues, whether resource statistics mature in time, and whether
training volume changes those behaviors. Preserve negative results.
