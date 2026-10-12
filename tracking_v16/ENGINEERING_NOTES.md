# V16 engineering boundaries

Two physical whole-episode feedback contracts passed before training.
World initialization/reset clears cumulative collision history. This same
history is visible to both arms; per-step collision labels remain separate.
Raw raster/observations, actual actions, original physical reward and modified
training reward are archived separately. Safety success is independently
recomputed from the full observed episode.

The warm start is the V15 temporal 8192-step model, copied to seed/model.pt
before verified V15 cleanup. Its SHA-256 is recorded. Fresh optimizers,
fresh body worlds and empty memory/replay are intentional; this is additional
practice from learned weights, not exact continuation.

Do not conflate the common cumulative-history change with training-volume
benefits: compare the two V16 arms to isolate only whole-episode feedback.
This is goal-feedback engineering; it does not establish a novel RL theorem.

The first plot used the first arm's y-axis limits and clipped higher values in
the second arm. Raw data and report tables were correct. The immutable archive
retains that initial plot. PROCESS_CURVES_CORRECTED.png and the revised report
generator, saved in main, use the full range of both arms. No scientific run
or numerical result changed.
