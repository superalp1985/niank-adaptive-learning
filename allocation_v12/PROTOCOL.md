# V12: local factual learning with bounded retrieval plans

Date: 2026-10-12. One exploratory run, three configurations. No efficacy or
novelty claim in advance. User approved the next-step direction after V11.

## Question and local updates

V11's frozen policy never retrieved. Can a state-conditioned estimate of the
actual consequences of a bounded retrieval plan learn when to recover old raw
records, while the numerical predictor receives no directory summary?

There is no end-to-end objective across the modules in this version.
Four independent predictor groups have separate Adam states and learn only the
executed physical action's observed next state/contact. A group selector learns
the selected group's observed prediction loss. A plan scorer learns only the
selected plan's four-step factual reward, with detached inputs and targets.
Neither scorer differentiates through the database or predictor.

This implements local learning and coordination, not guaranteed local optima or
global optimality. A differentiable scalar loss can have a full-network gradient
from initialization; that gradient does not certify a global minimum. Likewise,
search over our finite plan list maximizes an estimated score in that list only.
Local solutions need not compose into a global solution. Exhaustive search is
not the only optimization method; its guarantees require a complete feasible
search and a reliable objective.

## Candidate mechanism

- Raw observations remain in SQLite. Sixteen directory entries, three raw cache
  slots, four physical steps per internal decision.
- Each record has three unsigned channel-presence bits. They are independent
  of the sign/magnitude of our task cues. These bits only enter plan selection.
  No summary, hidden simulator pair, or evaluator regret enters predictions.
- Enumerate keep, each single retrieval, each unordered pair retrieval, and each
  single fade. Invalid plans are masked. Up to two raw reads per decision.
  This is cheap candidate scoring, not execution of all hypothetical worlds.
- Restored records receive a new cache access time while preserving their
  original event timestamp. Fading remains reversible; no permanent detail label.
- Score plans jointly from current body/phase/error and candidate metadata,
  using nonlinear interactions. Regress the executed plan's four-step reward:
  mean(-physical_cost - 0.5 * preupdate_state_MSE) - 0.002 * raw_reads.
  No counterfactual action labels or bootstrap from unexecuted plans.
- Keep a bounded replay of factual operation outcomes. Each decision uses one
  minibatch of 128 presentations, counted separately from new experiences.
  Predictor groups get no extra replay presentations.
- Exploration: 0.20 uniform valid-plan exploration during training, 0.15
  uniform group exploration, 0.10 physical-action exploration. Frozen probes
  use greedy choices and never learn.

This is related to contextual bandits, memory utility learning and decoupled
local learning, rather than a new proof of those principles. The narrow unknown
is whether joint one/two-record plans, sign-free summaries, and separate factual
update paths solve this project's previously unused directory.

## Task and controls

Reuse the continuous physical simulator, with a newly declared observable task
variant: 48-step cycles, delay 20, fixed risk 0.75, and random cue order between
steps 0 and 4 in each world/cycle. A visible body impulse and new goal at step 20
create a consequence window before the learner has observed the changed law.
The impulse is in the sensor; no hidden unannounced change is introduced.
These changes are identical across V12 configurations. Do not compare V12
numbers directly with V11 or attribute a cross-version difference to learning.

Three matched configurations:

1. `state_content`: current-state query plus per-record presence bits.
2. `state_time`: same query and parameters, content bits zeroed only for plans.
3. `no_query_content`: same parameters and candidates, current-state query
   zeroed only for plans; ages/live/content still available.

All have identical initial predictor/selector/scorer parameters and exogenous
world seeds. Actions can cause legitimate trajectory differences. Each has four
independent groups with the same capacity; no false claim of dynamic growth,
coefficient-level sparsity, real brain dynamics or Qwen precision execution.

Train 64 worlds continuously through 1024, 4096 and 16384 steps:
65,536 -> 262,144 -> 1,048,576 new experiences/model. Do not reset optimizers,
worlds, operation replay, directory or RNG between readouts.
Frozen probes use seen pairs and held-out (0,0)/(3,3), with a fixed separate seed.
Also run a declared final intervention that disables database reads; it is an
ablation of retrieval use, not a new training run.
No held-out probe is used to choose hyperparameters. One model/world seed:
descriptive exploration, not significance or a broad generalization claim.

Report factual action success (evaluator-only regret <= 0.02), MSE, physical
cost, first four steps after the change, both-cue availability, real database
reads, operation outcomes/replay, group use, parameter and MAC accounting.
More reads without better physical behavior do not establish success.

## Implementation and acceptance

1. Write boundary contracts first: summaries cannot disclose cue sign, restoring
   pairs really reads SQLite and survives the next insertion, invalid plans are
   masked, gradients stay in the chosen group/local scorer, split continuation
   matches uninterrupted CUDA practice.
2. Implement `core.py` and `run.py` in this directory only. Preserve old runs.
3. Run contracts once, then the three declared configurations with continuous
   milestones. Any engineering failure stops training until fixed; incomplete
   attempts are recorded separately.
4. Independently recompute physical results, plan rewards, read counts and
   candidate legality from saved raw records/trajectories before reporting.
5. Upload code/report and complete raw snapshot, read back all part hashes,
   then clean verified bulk files. Data/cache/logs/staging stay on E:.

If retrieval remains unused or its removal does not hurt, preserve the negative
result. Inspect operation ranking and input access, rather than cycling seeds
or claiming that local learning proves global optimality.

## Primary prior work

- Belilovsky et al., Greedy Layerwise Learning Can Scale To ImageNet (2019):
  https://proceedings.mlr.press/v97/belilovsky19a.html
- Belilovsky et al., Decoupled Greedy Learning of CNNs (2020):
  https://proceedings.mlr.press/v119/belilovsky20a.html
- Goodfellow et al., Deep Learning, chapter 8, optimization versus generalization:
  https://www.deeplearningbook.org/contents/optimization.html
- Memory utility and routing overlap: repository
  `NOVEL_ALGORITHM_PLAN_20261012_ZH.md`. No reproduction of prior paper tasks.
