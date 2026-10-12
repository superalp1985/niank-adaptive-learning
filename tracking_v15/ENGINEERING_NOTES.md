# V15 engineering and interpretation

Four temporal/risk boundary contracts and the CUDA smoke passed before training.
Two configurations completed exactly 393,216 new transitions each, plus all
frozen probes and the final no-risk intervention. No training was repeated.

The independent auditor's first V15 invocation stopped at a 2.21e-6 reward
reconstruction discrepancy. The simulator computes distance using double
precision, while archived measured coordinates are float32. Recomputing their
norm in float32 introduces avoidable rounding. The auditor now calculates the
norm in float64 with the same 2e-6 absolute / 2e-5 relative tolerance.
All raw action/observation/next-state identities, physical rewards, terminal
events and memory ownership then passed. No training source or data was edited.

Fixed risk allocation was not an improvement over temporal control. Its own
frozen no-risk intervention suggests limited actual use of the risk output,
but does not establish a general benefit. Preserve the negative finding.

Final checkpoint files contain weights and optimizer state, not exact
mid-trajectory simulator/replay resumption. A later warm start must explicitly
start fresh worlds and optimizers or restore every required component;
do not pretend it continues the same trajectory.
