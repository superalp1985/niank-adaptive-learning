"""Frozen parameters with the continuing agent's stochastic action policy."""
import argparse
import json
from pathlib import Path

import torch

from tracking_v22.evaluate_stages import parameter_hash, reset_dynamic
from tracking_v23.evaluate import file_hash, isolated_rng, scene_results
from tracking_v23.persistence import atomic_json, preserve_incomplete_bootstrap, read_complete
from tracking_v23.run import verify_manifest
from tracking_v23.session import Session


class ExploringFrozenSession(Session):
    def advance(self, count):
        if self.learning_enabled or self.agent.trial is not None:
            raise RuntimeError("Exploration observation must not train or settle a trial")
        previous = self.training
        # The inherited flag also enables settle_trial, which is a no-op when
        # trial is None. learning_enabled=False prevents any new learning trial.
        self.training = True
        try:
            return super().advance(count)
        finally:
            self.training = previous


def evaluate(path, steps=1024):
    path = Path(path).resolve()
    config = verify_manifest(path)["config"]
    checkpoint = path/"learner.pt"
    identity = dict(checkpoint_sha256=file_hash(checkpoint), steps=steps,
                    scene_seed=config["final_scene_seed"], seed=config["probe_seed"],
                    arm=config["arm"], action_policy="unchanged_training_exploration",
                    evaluator_sha256=file_hash(Path(__file__)))
    output = path/"exploring_frozen_probe"
    completed = read_complete(output/"COMPLETE.json", lambda result:
        result["learned_parameters_unchanged"] and result["frozen"]["step"] == steps)
    if completed is not None:
        if completed["identity"] != identity:
            raise RuntimeError("Completed exploration probe differs")
        return completed
    with isolated_rng(config["probe_seed"]):
        agent = torch.load(checkpoint, weights_only=False)
        reset_dynamic(agent, config["bodies"])
        expected = parameter_hash(agent)
        preserve_incomplete_bootstrap(output)
        if (output/"events.sqlite").exists():
            if json.loads((output/"identity.json").read_text(encoding="utf-8")) != identity:
                raise RuntimeError("Interrupted exploration probe differs")
            session = ExploringFrozenSession.restore(output)
            if parameter_hash(session.agent) != expected:
                session.close()
                raise RuntimeError("Restored parameters differ")
        else:
            output.mkdir(parents=True, exist_ok=True)
            atomic_json(output/"identity.json", identity)
            probe_config = dict(config, world_seed=config["final_scene_seed"],
                                video_every=max(1, steps//24))
            session = ExploringFrozenSession(output, probe_config, training=False, learner=agent)
            try:
                session.checkpoint()
            except BaseException:
                session.close()
                raise
        try:
            counters = (session.agent.updates, session.agent.meta_updates,
                        session.agent.local_updates, session.agent.state_updates)
            while session.step < steps:
                session.advance(min(config["probe_checkpoint_every"], steps-session.step))
                session.checkpoint()
            actual_counters = (session.agent.updates, session.agent.meta_updates,
                               session.agent.local_updates, session.agent.state_updates)
            if parameter_hash(session.agent) != expected or actual_counters != counters:
                raise RuntimeError("Exploration probe changed learned parameters or update counters")
            session.save_gif()
            result = dict(identity=identity, learned_parameters_unchanged=True,
                          parameter_sha256=expected, frozen=session.summary(),
                          scenes=scene_results(session),
                          limitation="One matched set of eight scenes; stochastic trajectories "
                                     "differ from greedy evaluation. No parameter learning.")
            atomic_json(output/"COMPLETE.json", result)
            return result
        finally:
            session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arms", nargs="+", type=Path)
    parser.add_argument("--steps", type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(4)
    for arm in args.arms:
        result = evaluate(arm, args.steps)
        print(json.dumps(dict(arm=str(arm), scenes=result["scenes"],
                              unchanged=result["learned_parameters_unchanged"])))
