# V12: bounded retrieval plans and local factual learning

Three continuous configurations. Readouts use actual physical outcomes,
not agreement with another model. One seed: descriptive exploration.
All networks have 126,918 parameters, including scorers; one active
predictor group has 30,985 parameters. No end-to-end module gradient.

## Held-out combinations

| Mode | New experiences | Success | Critical four steps | MSE | Reads | Both cues at change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| state_content | 65,536 | 79.26% | 81.93% | 0.022163 | 454 | 10.00% |
| state_content | 262,144 | 86.82% | 82.23% | 0.016481 | 1,671 | 13.44% |
| state_content | 1,048,576 | 79.33% | 83.20% | 0.020447 | 760 | 3.75% |
| state_time | 65,536 | 72.48% | 79.98% | 0.023939 | 0 | 0.00% |
| state_time | 262,144 | 85.41% | 89.84% | 0.017902 | 7,284 | 0.00% |
| state_time | 1,048,576 | 81.22% | 91.70% | 0.022413 | 1,241 | 2.19% |
| no_query_content | 65,536 | 75.35% | 85.35% | 0.028085 | 7,168 | 0.00% |
| no_query_content | 262,144 | 84.56% | 88.09% | 0.015514 | 2,068 | 0.00% |
| no_query_content | 1,048,576 | 90.21% | 88.67% | 0.014000 | 2,103 | 0.00% |

## Final frozen intervention

Enabled minus disabled raw reads. Same environment seed and weights;
different actions can cause subsequent physical trajectories to diverge.
This estimates the effect of allowing the trained retrieval policy;
it does not isolate causal value for each individual database detail.

| Mode | Success difference | Critical difference | MSE difference |
| --- | ---: | ---: | ---: |
| no_query_content | +16.02 pp | +7.42 pp | -0.011832 |
| state_content | +2.97 pp | +3.32 pp | -0.002114 |
| state_time | +3.84 pp | +1.95 pp | +0.000067 |

## Final retrieval and local-value diagnostics

Current-cycle cues are records written at phase 0 or 4 of this cycle.
Other-event reads may include earlier cues; this classification does
not label those records permanently useless. Value errors compare
only the executed plan's estimate and factual four-step outcome.

| Mode | Current-cycle cue reads / all reads | Reads at change | Value RMSE | Largest group share |
| --- | ---: | ---: | ---: | ---: |
| no_query_content | 1,451 / 2,103 | 107 | 0.065838 | 39.62% |
| state_content | 338 / 760 | 92 | 0.080944 | 69.84% |
| state_time | 113 / 1,241 | 162 | 0.083757 | 99.99% |

## Learning and accounting

| Mode | Experiences | Operation replay presentations | Estimated total MAC | Seconds |
| --- | ---: | ---: | ---: | ---: |
| state_content | 1,048,576 | 524,288 | 173,774,209,024 | 676.1 |
| state_time | 1,048,576 | 524,288 | 173,774,209,024 | 953.7 |
| no_query_content | 1,048,576 | 524,288 | 173,774,209,024 | 1312.3 |

## Audit and limits

Independent physical/access/operation-target audit: 24 directories, 3,468,288 transitions, 867,072 operation windows.
Source hashes match the run manifest.

The environment differs from V11: fixed risk, random cue order, visible
impulse at the change. No direct cross-version mechanism attribution.
Search covers 153 metadata plans, not all physical/parameter choices.
Local Adam updates do not certify local minima or global optimality.
Backpropagation still runs inside each local network; separation is
at module boundaries, not a biologically local synapse rule.
No guaranteed monotonic generalization with more training.

Summary bits never disclose cue sign; raw records alone enter numerical
predictions. The tiny scorer evaluates many plans, so total planning MAC
is not necessarily below predictor MAC. No hardware speedup claim.

Complete data, databases and final resume states are archived in Release.
Resume states require the matching external SQLite databases.

See PROTOCOL.md for prior work, exact candidate differences and budgets.
Interpretation in REPORT_V12_ZH.md is written after all matched readouts.
