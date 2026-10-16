"""Frozen common-scene observations of successive training checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from tracking_v22.run import Session, dump


def parameter_hash(agent):
    sha = hashlib.sha256()
    for module in (agent.model, agent.state_map, agent.state_readout, agent.meta):
        for tensor in module.state_dict().values():
            sha.update(tensor.detach().cpu().numpy().tobytes())
    for tensor in (agent.state_tau, agent.spike_weights):
        sha.update(tensor.detach().cpu().numpy().tobytes())
    return sha.hexdigest()


def reset_dynamic(agent, bodies):
    agent.trial = None
    agent.latent.zero_()
    agent.pretrace.zero_()
    agent.error.zero_()
    agent.last_sensory = agent.last_drive = None
    agent.spikes = type(agent.spikes)(bodies, agent.spike_dim, device="cuda")


class VisualProbe(Session):
    def __init__(self, *args, **kwargs):
        self.sensor_video = []
        super().__init__(*args, **kwargs)

    def install_observations(self, observations, event_bins=None):
        super().install_observations(observations, event_bins)
        if self.step % 32 == 0:
            self.sensor_video.append(Image.fromarray(observations[0][0]))


def evaluate(training_path, stages, steps):
    training_path = Path(training_path)
    config = json.loads((training_path/"manifest.json").read_text())["config"]
    config = dict(config, world_seed=921600)
    torch.set_num_threads(4)
    results = []
    for stage in stages:
        source = training_path/f"learner_step_{stage:05d}.pt"
        if not source.exists():
            raise FileNotFoundError(source)
        output = training_path/"stage_probes"/f"stage_{stage:05d}"
        if (output/"COMPLETE.json").exists():
            results.append(json.loads((output/"COMPLETE.json").read_text()))
            continue
        np.random.seed(23001)
        torch.manual_seed(23001)
        torch.cuda.manual_seed_all(23001)
        agent = torch.load(source, weights_only=False)
        reset_dynamic(agent, config["bodies"])
        expected = parameter_hash(agent)
        if (output/"events.sqlite").exists():
            trial = VisualProbe.restore(output)
            expected = parameter_hash(trial.agent)
        else:
            trial = VisualProbe(output, config, training=False, learner=agent)
            trial.checkpoint()
        try:
            while trial.step < steps:
                trial.advance(min(256, steps-trial.step))
                trial.checkpoint()
            if parameter_hash(trial.agent) != expected:
                raise RuntimeError("Frozen stage changed a learned parameter")
            summary = trial.summary()
            video = trial.sensor_video
            if video:
                video[0].save(output/"actual_sensor.gif", save_all=True,
                              append_images=video[1:], duration=320, loop=0)
            result = dict(training_step=stage, observation_steps=steps,
                          source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          parameter_sha256=expected, learned_parameters_unchanged=True,
                          frozen=summary)
            dump(output/"COMPLETE.json", result)
            results.append(result)
            print(json.dumps(dict(event="stage_complete", training_step=stage,
                                  contacts=summary["contact"],
                                  collision_fraction=summary["collision_fraction"],
                                  impacts=summary["impact_events"],
                                  prior_fraction=summary["prior_fraction"])), flush=True)
        finally:
            trial.close()
    dump(training_path/"stage_observations.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", type=int, required=True)
    parser.add_argument("--steps", type=int, default=512)
    args = parser.parse_args()
    evaluate(args.training, args.stages, args.steps)
