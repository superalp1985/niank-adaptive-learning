"""Continue trained weights in a new experience stream, without rerunning prefixes."""

import core
from core import ROOT, MODES, Predictor, Controller
from run import counts, dump, emit, simulate
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    output = ROOT / "continued_practice"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    runs = [
        dict(mode=mode, seed=23, world_seed=440100, groups=groups)
        for groups in (9, 78) for mode in MODES
    ]
    hashes = {}
    for name in ("core.py", "run.py", "test_contracts.py", "PROTOCOL.md", "continue_learning.py"):
        source = ROOT / name
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copyfile(source, output / name)
    dependency = ROOT.parent / "retention_v5" / "experiment.py"
    hashes["physics_dependency"] = hashlib.sha256(dependency.read_bytes()).hexdigest()
    shutil.copyfile(dependency, output / "physics_dependency.py")
    manifest = dict(
        config=dict(stage="continued_practice", runs=runs, steps=6144, worlds=128),
        sha256=hashes, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        source_stage="confirmation", source_world_seed=420100, source_model_seed=23,
        previous_main_presentations=262144, additional_main_presentations=786432,
        cumulative_main_presentations=1048576,
        optimizer_restart=True, directory_and_world_restart=True,
        limitation="Further practice on new worlds, NOT an isolated training-volume-only comparison.",
    )
    dump(output / "manifest.json", manifest)
    results = []
    for item in runs:
        mode, groups = item["mode"], item["groups"]
        source = ROOT / "confirmation" / f"g{groups}_s23_w420100_{mode}" / "model.pt"
        if not source.exists():
            raise FileNotFoundError(source)
        random.seed(10023)
        np.random.seed(10023)
        torch.manual_seed(10023)
        torch.cuda.manual_seed_all(10023)
        model, controller = Predictor(groups).cuda(), Controller(mode).cuda()
        state = torch.load(source, map_location="cuda", weights_only=True)
        model.load_state_dict(state["model"])
        controller.load_state_dict(state["controller"])
        target = output / f"g{groups}_s23_w440100_{mode}"
        target.mkdir()
        dump(target / "initialization.json", dict(
            checkpoint=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            optimizer_restart=True, previous_main_presentations=262144,
        ))
        result = simulate(model, controller, target, 440100, 6144, 128, training=True)
        result.update(model_seed=23, previous_main_presentations=262144, cumulative_main_presentations=1048576)
        for name, held in (("seen", False), ("composition", True)):
            probe = target / name
            probe.mkdir()
            result[name] = simulate(model, controller, probe, 440100 + (900000 if held else 800000),
                                    128, 64, training=False, heldout=held)
        dump(target / "result.json", result)
        results.append(result)
        dump(output / "results.json", results)
        emit("continued_result", groups=groups, mode=mode, last=result["last_success"],
             seen=result["seen"]["success"], composition=result["composition"]["success"])
        del model, controller
        torch.cuda.empty_cache()
    dump(output / "COMPLETE.json", dict(runs=len(results)))
    emit("complete", runs=len(results))


if __name__ == "__main__":
    main()
