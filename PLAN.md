# V3: learning volume, online directories, and fly connectivity

Date: 2026-10-11. This is a new experiment; the previous frozen runs are unchanged.

## Research principle

Unknown results are worth testing. Failure is an admissible result, not a reason
to conceal a run, tune on the test set, or keep searching until a preferred
mechanism wins. Hardware optimization is deferred; actual runtime remains an
operational measurement, not the scientific selection objective.

## Authorized scope

The user authorized further algorithm iteration, multi-objective learning,
learning-volume controls, online memory organization, and downloading a relevant
fruit-fly-brain project. Everything new is stored on E:, including downloaded
data, runtime caches, temporary files, logs, and experiment outputs.

## Model source

- Repository: `eonsystemspbc/fly-brain`
- Commit: `a3db62f9436074e485c0278290c2164ed6150808`
- Local source: `E:/byte级模型/v6/niank_v9/external/fly-brain/a3db62f9436074e485c0278290c2164ed6150808`
- Data: `2025_Completeness_783.csv`, `2025_Connectivity_783.parquet`.
- Upstream license: GPL-2.0; the included original Shiu implementation has a
  separate MIT license. Preserve both and attribution.
- These are measured-connectivity-derived synaptic weights and a LIF simulator,
  not pretrained language-model weights. No language knowledge is assumed.

## Distinct tracks

1. Validate a bounded whole-connectome CUDA activity experiment using the
   upstream model classes. Inspect propagation and cost, not answer agreement
   as a quality metric. Any later reduced model is explicitly marked reduced.
2. Learning volume: cross unique training examples with repetition count;
   report both, number of updates, continuous prediction error, and task
   accuracy. Match total presentations to distinguish diversity from repetition.
3. Online memory: alternate observation, questions, committed answers, feedback,
   and maintenance. Compare stale assignments, periodic reindexing, and bounded
   incremental reassignment on identical streams.
4. Multi-objective learning: separate task correctness, prediction,
   information association, and compute cost. Auxiliary improvements alone do
   not establish task improvement. Every added objective has an ablation.
5. Capacity: compare nested connection-derived networks where feasible, match
   randomized controls at each size, and separate parameter count from distinct
   experiences, presentations, and optimization steps.

## Quality and controls

- Truth comes from the environment, never from dense-model output.
- Questions allowing multiple valid answers use a constraint verifier, not
  exact comparison with one canonical answer or proof.
- Score a question before feedback; later observations cannot repair earlier
  results. Future data are inaccessible to the learner.
- Disjoint training, validation, and final test data. Freeze selection before
  opening final test results; later ideas require new independent tests.
- For biological structure claims, use matched randomized connectivity and
  ordinary non-biological baselines. Topology, dynamics, and learning are
  separate factors.
- Cost includes building, updating, searching, moving records, training, and
  extra exploration. Physical layout optimization is deferred.
- Preserve all seeds and failed variants. Do not infer phase transitions from
  exact-match jumps alone; show continuous loss curves too.
- FAME remains outside this round unless its existing full definition is
  explicitly implemented in a separate protocol.
- The user explicitly accepts physical prediction and reaction tasks in place
  of knowledge QA and authorizes increasing parameter count as required.
- Centralized computer scheduling is not treated as physically equivalent to
  distributed brain dynamics. Discrete-time models test functional hypotheses.

## Implementation ledger

- [x] Resolve and download the source and connection weights to E:.
- [x] Validate every downloaded Git blob.
- [x] Check CUDA/runtime with bounded process lifetime and persistent logs.
- [ ] Implement and validate multiple-answer/event-stream benchmark contracts.
- [ ] Implement and validate online directory maintenance.
- [x] Run the bounded whole-connectome dynamics experiment.
- [x] Freeze and run learning-volume and multi-objective comparisons.
- [x] Run the first embodied visual/contact/state/action pilot.
- [x] Add a dual-rate fast-reaction/slow-directory iteration.
- [x] Add an approximately parameter-matched dense capacity control.
- [x] Record per-seed embodied pilot results and failure modes.
- [x] Audit old pilot leakage, static-slot memory, and unused-parameter limitations.
- [x] Run V5 observed-event memory, archival storage, and value forgetting.
- [x] Compare four learning-volume conditions with three optimization seeds.
- [x] Preserve three discovery iterations and intentional anti-value failures.
- [x] Add new layout/delay/texture splits and independent fine-cue intervention.
- [ ] Add an unseen-layout test stream and explicit directory maintenance.
- [ ] Replace full slot scoring with a measured coarse route index.
- [ ] Audit outputs and report successes, failures, and remaining hypotheses.
