"""Small actual-source continuation check; retain only audit images and JSON."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import torch

from tracking_v22.evaluate_stages import parameter_hash
from tracking_v22.run import dump
from tracking_v23.run import DEFAULT_SOURCE, ROOT, source_identity
from tracking_v23.session import Session, backup_database


def check(output, source=DEFAULT_SOURCE):
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
            session = Session.fork(source, root/arm, dict(config, arm=arm, video_every=2))
            try:
                origin = session.step
                pending = sum(len(queue) for queue in session.pending)
                session.advance(8)
                session.checkpoint()
                backup_database(session.output/"events.sqlite", root/(arm+"_resume")/"events.sqlite")
                session.advance(8)
                expected = parameter_hash(session.agent)
                positions = [tuple(world.body.position) for world in session.worlds]
                error = session.agent.error.clone()
                session.save_gif()
                target = output/arm
                target.mkdir()
                for file in session.output.glob("actual_*"):
                    shutil.copyfile(file, target/file.name)
                results[arm] = dict(origin_step=origin, additional_smoke_steps=16,
                                    pending_windows_migrated=pending,
                                    input_width=session.x.shape[1],
                                    parameter_sha256=expected,
                                    initial_source_trial=session.agent.trial is not None,
                                    vision=session.summary()["vision"])
            finally:
                session.close()
            resumed = Session.restore(root/(arm+"_resume"))
            try:
                resumed.advance(8)
                assert parameter_hash(resumed.agent) == expected
                assert [tuple(world.body.position) for world in resumed.worlds] == positions
                assert torch.equal(resumed.agent.error, error)
                results[arm]["exact_checkpoint_continuation"] = True
            finally:
                resumed.close()
    assert source_identity(source) == identity
    report = dict(source=identity, source_unchanged=True, arms=results,
                  scope="16 extra frames/body/arm only; no full practice experiment")
    dump(output/"CHECKS.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    check(args.output, args.source)
