"""Small actual-source continuation check; retain only audit images and JSON."""
import argparse
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from tracking_v22.evaluate_stages import parameter_hash
from tracking_v23.persistence import atomic_json as dump
from tracking_v23.run import DEFAULT_SOURCE, ROOT, source_identity
from tracking_v23.session import Session, backup_database


def check(output, source=DEFAULT_SOURCE, steps=256):
    if steps < 128 or steps % 64:
        raise ValueError("Learning smoke needs at least 128 steps, in complete 64-step blocks")
    output = Path(output)
    if output.exists():
        raise RuntimeError("Use a fresh validation output directory")
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    config = json.loads((ROOT/"extended_practice.json").read_text())
    identity = source_identity(source)
    results = {}
    with tempfile.TemporaryDirectory(dir="E:/niank-temp") as directory:
        root = Path(directory)
        for arm in ("baseline", "active"):
            session = Session.fork(source, root/arm, dict(config, arm=arm, video_every=max(1, steps//32)))
            try:
                origin = session.step
                pending = sum(len(queue) for queue in session.pending)
                updates = session.agent.updates
                meta_updates = session.agent.meta_updates
                initial_trial = session.agent.trial is not None
                old_heads = [head.weight.detach().clone() for head in session.agent.model.heads]
                session.advance(steps-16)
                session.checkpoint()
                backup_database(session.output/"events.sqlite", root/(arm+"_resume")/"events.sqlite")
                session.advance(16)
                expected = parameter_hash(session.agent)
                optimizer = copy.deepcopy(session.agent.optimizer.state_dict())
                positions = [tuple(world.body.position) for world in session.worlds]
                error = session.agent.error.clone()
                session.save_gif()
                target = output/arm
                target.mkdir()
                for file in session.output.glob("actual_*"):
                    shutil.copyfile(file, target/file.name)
                facts = []
                for blob, in session.db.execute("SELECT batch FROM facts WHERE step>?", (origin,)):
                    with np.load(io.BytesIO(blob), allow_pickle=False) as arrays:
                        facts.append(arrays["a0"].shape[1])
                credits = [json.loads(row[0]) for row in session.db.execute(
                    "SELECT outcome FROM choices WHERE step>?", (origin,))]
                groups = 3 if arm == "active" else 1
                head_changes = [(head.weight-old).reshape(groups, -1).norm(dim=1).detach().cpu().tolist()
                                for head, old in zip(session.agent.model.heads, old_heads)]
                state = session.agent.optimizer.state[session.agent.model.encoder[0].weight]
                results[arm] = dict(origin_step=origin, additional_smoke_steps=steps,
                                    pending_windows_migrated=pending,
                                    input_width=session.x.shape[1],
                                    parameter_sha256=expected,
                                    initial_source_trial=initial_trial,
                                    new_predictor_updates=session.agent.updates-updates,
                                    new_meta_updates=session.agent.meta_updates-meta_updates,
                                    settled_credits=credits,
                                    new_fact_widths=facts,
                                    encoder_adam_moment_shape=list(state["exp_avg"].shape),
                                    head_group_weight_changes=head_changes,
                                    vision=session.summary()["vision"])
            finally:
                session.close()
            resumed = Session.restore(root/(arm+"_resume"))
            try:
                resumed.advance(16)
                assert parameter_hash(resumed.agent) == expected
                assert [tuple(world.body.position) for world in resumed.worlds] == positions
                assert torch.equal(resumed.agent.error, error)
                actual = resumed.agent.optimizer.state_dict()
                assert actual["param_groups"] == optimizer["param_groups"]
                for key, state in optimizer["state"].items():
                    for field, value in state.items():
                        assert torch.equal(actual["state"][key][field], value)
                results[arm]["exact_checkpoint_continuation"] = True
            finally:
                resumed.close()
    assert source_identity(source) == identity
    report = dict(source=identity, source_unchanged=True, arms=results,
                  scope=f"{steps} extra frames/body/arm only; no full practice experiment")
    dump(output/"CHECKS.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--steps", type=int, default=256)
    args = parser.parse_args()
    check(args.output, args.source, args.steps)
