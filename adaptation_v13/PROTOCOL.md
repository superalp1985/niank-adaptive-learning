# V13: better choices through continuing embodied practice

> WITHDRAWN BEFORE IMPLEMENTATION, 2026-10-13.
> The user rejected the sequential A-B-A-B training premise and requested
> literature research first. This file preserves the superseded draft only.
> No V13 training, tests, simulation or source implementation was run.
> Do not execute this protocol or treat its budgets as approved.
> Read RESEARCH_REVIEW_20261013_ZH.md and STATUS.md.

Date: 2026-10-13. The user approved the next algorithm step and clarified:
the target is progressively better choices, not fewer choices or a headline
final score. This version tests a small software core for continuing embodied
learning, not biological dynamics or a finished robot.

## Question and declared changes

Can separating operation-class choice from record choice, with factual
feedback spanning later practice, improve adaptation across repeated changes?
We reuse V12's independently updated predictor groups and raw SQLite memory.
No old experiment is rerun. All V13 arms share the new stream and networks.

Directory plans are keep, retrieve one of 16 records, fade one of 16 records:
33 plans rather than 153. V12's unused pair primitive is removed in every arm.
A class scorer estimates the realized value of keep/retrieve/fade.
A record scorer estimates a deviation from the class estimate saved at choice.
Class and record scorers train independently on detached factual targets.
Numerical prediction receives raw images only, never directory summaries.

Directory context contains cached unsigned presence, ages, occupancy, and
past prediction-error summaries, but no current body query. Routing and physical
prediction still receive body sensors. Record metadata adds age/live/existence
and three unsigned presence bits. Raw identity is logged, never a label.

Three arms, identical initial networks and exogenous world seeds:

- `flat_short`: select from all 33 combined class+record estimates, with
  20% uniform legal-plan exploration; four-step factual target.
- `class_short`: select class first, then record; 20% uniform legal-class
  exploration and 20% uniform legal-record exploration; four-step target.
- `class_span`: same hierarchical selection, 32-step factual target.

All arms score the same candidates and train both heads. Class choice changes
both exploitation and exploration, so this is a package comparison, not an
isolated claim about exploration alone. Keep has a dummy zero-metadata plan.
Exploration is fixed throughout, not annealed.

Each four-step window records mean(-physical_cost - 0.5 * preupdate_MSE)
minus 0.002 per raw read. The class target is the mean factual window reward
over the declared horizon minus the predecision EMA reward baseline.
The record target subtracts the saved predecision class prediction.
The baseline may reduce nonstationarity; it is not a causal counterfactual or
a proof that predictor changes have been separated from memory value.
The 32-step label includes intervening actions and their read costs.
No future labels, simulator-optimal actions or unexecuted outcomes train any
module. Pending unfinished spans remain censored and saved for continuation.

Four independent predictors learn executed next-state/contact outcomes.
Their selector learns the selected group's factual prediction loss.
Plan replay is bounded to 256 factual batches and samples 128 presentations
per complete window once mature labels exist. Predictor replay is absent.
All costs, matured labels and replay presentations are counted separately.

## Continuous world

64 parallel worlds; 48-step cycles with cue flashes at steps 0 and 4,
random cue order, and a physical change/visible velocity impulse at step 20.
All 16 pairs of four inertia and four force cues are sampled independently.
No two-pair diagonal held-out set is used to claim independent composition.

Four 3,072-step phases: A, B, A, B, for 12,288 steps and 786,432 physical
transitions per arm. Networks, optimizers, memory, bodies, factual replay,
pending credit and RNG continue without phase resets.

A: cue indices use the ordinary inertia/force mapping, observed risk 0.35.
B: both cue mappings reverse (index becomes 3-index), observed risk 0.95.
The risk field is an actual observable environmental property used in physics,
not a private phase label. At a switch the old active law persists until its
normal phase-20 replacement. A learner need not infer a new mapping on its
first encounter; its task is to adapt through actual sensory consequences.
No phase number or mapping index enters the learner. Cue values and physical
outcomes supply evidence over practice.

Each action is predicted and scored before feedback, then learned.
Physical action exploration is 0.10; group exploration 0.15, both constant.
No end-of-run frozen exam selects parameters or the next checkpoint.
One training seed: descriptive process evidence, not significance or a
general biological learning claim.

## Primary process observations

Keep all cycle-level and 96-step curves, not just final values.
For each of the four phases compare its first eight and last eight cycles:

1. Preupdate next-state MSE; contact frequency; goal-distance squared;
   speed/action effort; actual physical cost.
2. Evaluator-only action regret and regret<=0.02 rate as secondary diagnostics.
3. All-action and non-exploratory-action outcomes, to separate forced trials
   from greedy practice. Never compare a hidden alternative to train a learner.
4. Operation fractions, raw reads, fade/retrieve identity lifecycles, cue
   availability and group-use entropy. These describe behavior, not a value
   judgment: concentrated choices alone do not establish improvement.
5. After each change, time until a rolling four-cycle cost and MSE both reach
   within 10% of that phase's eventual last-eight-cycle mean for four
   consecutive cycles. This is a retrospective descriptive recovery measure,
   not a training threshold. Report 'not reached' explicitly.
6. Compare initial and repeat A/B early-window costs/errors and recovery.
   More accumulated training is a possible explanation; not proof of memory
   savings independent of training volume.

Report improvements and deterioration together, with no composite 'biology
score'. A low final error with failed re-adaptation is a failed process result.
Entropy decreases without outcome improvements are not success.

## Prior work and scope

Learning progress as an exploration signal and developmental robotics already
exist: Oudeyer, Kaplan & Hafner (2007), author manuscript
https://www.pyoudeyer.com/ims.pdf ; Forestier et al. (2022),
https://www.jmlr.org/beta/papers/v23/21-0808.html .
We borrow the process perspective; V13 does not reproduce their curricula or
claim that this perspective itself is new.

Multi-step factual credit is established in reinforcement learning:
Sutton & Barto, Reinforcement Learning, second edition, chapters 7 and 12,
https://www.incompleteideas.net/book/bookdraft2018mar21.pdf .
Plasticity loss in long learning streams is documented by Dohare et al. (2024),
https://www.nature.com/articles/s41586-024-07711-7 .
V13 is too short to establish or refute that general phenomenon.

The experimental contribution is only this project's concrete revision of
class/record coordination, centered factual feedback and its measured
continuous adaptation. Hierarchy, delayed credit and local learning are not
claimed as individually new algorithms. No global-optimum guarantee.

## Engineering and storage

CUDA batch execution in the existing E: environment. Contract checks before
the run: class exploration independent of record multiplicity; legal plans;
no future credit; saved predecision baseline; sign-free summary isolation;
local gradients; exact split continuation including SQLite.
Independent audit recomputes physics, preupdate error, feedback windows,
record identities and memory transitions, and verifies source hashes.
No in-run algorithm changes. Archive all data, verify remote parts, then clean
local bulk files while retaining final resume states and source/report/curves.
