"""Run the second declared arm concurrently, without repeating the first."""

import hashlib
import json
from pathlib import Path

import torch

from tracking_v17.run import Session, probe, dump, ROOT


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads((ROOT / "pilot.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "pilot/manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["source"].items():
        assert hashlib.sha256((ROOT.parent / name).read_bytes()).hexdigest() == digest
    mode = "revisable"
    output = ROOT / "pilot"
    dump(output / "execution_schedule.json", dict(
        reason="Two declared arms in separate processes; no data rerun",
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        running_first_arm="fixed", concurrent_second_arm=mode,
        expected_parent_exit="Original sequential driver encounters existing second-arm folder after fixed completes.",
        scientific_sources_unchanged=True))
    torch.manual_seed(config["model_seed"])
    torch.cuda.manual_seed_all(config["model_seed"])
    session = Session(output / mode, mode, config["world_seed"], config["worlds"])
    torch.save(session.agent.state(), session.output / "initial.pt")
    initial = probe(session, session.output / "probe_0", config["world_seed"] + 900000,
                    config["probe_steps"])
    dump(session.output / "result_0.json", dict(milestone=0, probes=initial))
    for milestone in config["milestones"]:
        while session.step < milestone:
            point = session.advance(min(config["rollout"], milestone - session.step),
                                    config["epochs"])
            if session.chunk % 8 == 0:
                print(json.dumps(dict(event="progress", mode=mode, **point)), flush=True)
        torch.save(dict(model=session.agent.state(),
                        optimizers=[o.state_dict() for o in session.agent.optimizers],
                        step=session.step, memory=session.memory.__dict__,
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()),
                   session.output / f"checkpoint_{milestone}.pt")
        result = session.summary()
        result["probes"] = probe(session, session.output / f"probe_{milestone}",
                                 config["world_seed"] + 900000, config["probe_steps"])
        dump(session.output / f"result_{milestone}.json", result)
        print(json.dumps(dict(event="milestone", mode=mode, step=milestone,
            contact=result["probes"]["contact_per_minute"],
            lock=result["probes"]["visible_lock"],
            collision=result["probes"]["collision_rate"],
            impact=result["probes"]["attacked_impact_rate"])), flush=True)
    torch.save(dict(model=session.agent.state()), session.output / "model.pt")
    removal = probe(session, session.output / "probe_no_eyes",
                    config["world_seed"] + 900000, config["probe_steps"], True)
    dump(session.output / "sensor_removal.json", removal)
    session.close()
    dump(session.output / "WORKER_COMPLETE.json", dict(
        steps=config["milestones"][-1], experiences=config["worlds"] * config["milestones"][-1],
        source=manifest["source"]))


def collect():
    output = ROOT / "pilot"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    results = []
    for mode in manifest["config"]["modes"]:
        for step in manifest["config"]["milestones"]:
            results.append(json.loads((output / mode / f"result_{step}.json").read_text(encoding="utf-8")))
        assert (output / mode / "model.pt").exists()
        assert (output / mode / "sensor_removal.json").exists()
    dump(output / "results.json", results)
    dump(output / "COMPLETE.json", dict(
        results=len(results), source=manifest["source"],
        execution="Concurrent isolated processes; original sequential parent stops on second-arm directory guard"))
    print(json.dumps(dict(collected=len(results))), flush=True)


if __name__ == "__main__":
    main()
