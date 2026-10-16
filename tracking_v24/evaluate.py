"""Frozen physical observations, isolated from continued learning and its RNG."""
import json
from pathlib import Path

import torch

from tracking_v22.evaluate_stages import parameter_hash, reset_dynamic
from tracking_v23.evaluate import (
    file_hash, input_dependence, isolated_rng, scene_results, valid_probe_result,
)
from tracking_v23.persistence import atomic_json as dump, read_complete
from tracking_v23.session import Session


def evaluate(checkpoint, output, config, steps, scene_seed, diagnostic="normal"):
    checkpoint, output = Path(checkpoint), Path(output)
    if diagnostic not in ("normal", "uniform", "head_hold", "delay_off"):
        raise ValueError(diagnostic)
    identity = dict(checkpoint_sha256=file_hash(checkpoint), steps=steps,
                    scene_seed=scene_seed, diagnostic=diagnostic,
                    mechanism=config["mechanism"], seed=config["probe_seed"])
    complete = read_complete(output/"COMPLETE.json", valid_probe_result)
    if complete is not None:
        if complete["identity"] != identity:
            raise RuntimeError("Completed probe identity differs")
        return complete
    with isolated_rng(config["probe_seed"]):
        agent = torch.load(checkpoint, weights_only=False)
        reset_dynamic(agent, config["bodies"])
        agent.model.force_uniform = diagnostic == "uniform"
        if hasattr(agent, "force_no_delayed"):
            agent.force_no_delayed = diagnostic == "delay_off"
        expected = parameter_hash(agent)
        probe_config = dict(config, world_seed=scene_seed,
                            head_hold=diagnostic == "head_hold", video_every=max(1, steps//24))
        if (output/"events.sqlite").exists():
            if json.loads((output/"identity.json").read_text()) != identity:
                raise RuntimeError("Interrupted probe identity differs")
            session = Session.restore(output)
            if parameter_hash(session.agent) != expected:
                session.close()
                raise RuntimeError("Frozen resume parameter hash differs")
        else:
            output.mkdir(parents=True, exist_ok=True)
            dump(output/"identity.json", identity)
            session = Session(output, probe_config, training=False, learner=agent)
            session.checkpoint()
        del agent
        try:
            while session.step < steps:
                session.advance(min(config["probe_checkpoint_every"], steps-session.step))
                session.checkpoint()
            if parameter_hash(session.agent) != expected:
                raise RuntimeError("Frozen observation changed learned parameters")
            session.save_gif()
            result = dict(identity=identity, parameter_sha256=expected,
                          learned_parameters_unchanged=True, frozen=session.summary(),
                          scenes=scene_results(session),
                          input_dependence=input_dependence(session.agent, session.x))
            dump(output/"summary.json", result["frozen"])
            dump(output/"COMPLETE.json", result)
            return result
        finally:
            session.close()
