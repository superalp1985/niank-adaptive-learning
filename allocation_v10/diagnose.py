"""Post-run diagnostics; these are exploratory, not new primary endpoints."""

import json
from pathlib import Path
import sqlite3
import sys
import zlib

import numpy as np
import torch

from core import ROOT, Stream, Memory, Predictor, Controller, choose, loss_for, TOLERANCE
from run import dump


def diagnostics(stage):
    folder = ROOT / stage
    results = []
    learned_groups = []
    for target in sorted(folder.glob("g*")):
        if not (target / "result.json").exists():
            continue
        result = json.loads((target / "result.json").read_text())
        if "composition" not in result:
            continue
        if stage == "confirmation":
            torch.manual_seed(result["model_seed"])
            initial = Predictor(result["group_reserve"])
            checkpoint = torch.load(target / "model.pt", map_location="cpu", weights_only=True)
            changed = (checkpoint["model"]["w1"] - initial.w1.detach()).abs().flatten(1).amax(1) > 0
            learned_groups.append(dict(
                run=target.name, groups=result["group_reserve"],
                groups_with_changed_input_weights=int(changed.sum()),
                changed_indices=torch.nonzero(changed, as_tuple=True)[0].tolist(),
            ))
        for probe in ("seen", "composition"):
            with np.load(target / probe / "transitions.npz") as data:
                phase = np.arange(len(data["action"])) % 32
                moving = np.abs(data["body"][:, :, 1]) > 0.3
                late = phase >= 16
                after_change = (phase >= 8) & (phase <= 12)
                results.append(dict(
                    run=target.name, groups=result["group_reserve"], mode=result["mode"], probe=probe,
                    zero_action_rate=float((data["action"] == 1).mean()),
                    moving_fraction=float(moving.mean()),
                    after_change_success=float((data["regret"][after_change] <= TOLERANCE).mean()),
                    late_cycle_success=float((data["regret"][late] <= TOLERANCE).mean()),
                    no_exploration_operations=result[probe]["operation_counts"],
                    active_parameters=result[probe]["mean_active_predictor_parameters"],
                    mean_capacity=float(data["admitted"].mean()),
                ))
    dump(folder / "DIAGNOSTICS.json", results)
    if learned_groups:
        dump(folder / "LEARNED_GROUPS.json", learned_groups)
    print(json.dumps({"stage": stage, "probes": len(results)}))


class ReadOnlyArchive:
    def __init__(self, path):
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("PRAGMA query_only=ON")

    def get(self, identity):
        raw = self.connection.execute("SELECT image FROM events WHERE id=?", (int(identity),)).fetchone()[0]
        return np.frombuffer(zlib.decompress(raw), np.int8).reshape(3, 6, 6).copy()


@torch.no_grad()
def replay_checkpoint(target):
    result = json.loads((target / "result.json").read_text())
    checkpoint = torch.load(target / "model.pt", map_location="cpu", weights_only=True)
    model = Predictor(result["group_reserve"]).eval()
    model.load_state_dict(checkpoint["model"])
    errors = []
    for probe in ("seen", "composition"):
        with np.load(target / probe / "transitions.npz") as datafile:
            data = {k: datafile[k] for k in datafile.files}
        steps, count = data["action"].shape
        memory = Memory(count)
        archive = ReadOnlyArchive(target / probe / "events.sqlite")
        for step in range(steps):
            image = np.zeros((count, 3, 6, 6), np.int8)
            if step % 4 == 0:
                identities = step * count + np.arange(count)
                image = np.stack([archive.get(identity) for identity in identities])
                memory.append(step, image, identities)
                memory.apply(data["op"][step // 4], data["pointer"][step // 4], step, archive)
            phase = step % 32
            sensor = np.concatenate((
                data["body"][step], data["context"][step, :, None],
                np.full((count, 1), np.sin(2 * np.pi * phase / 32), np.float32),
                np.full((count, 1), np.cos(2 * np.pi * phase / 32), np.float32),
            ), 1)
            batch = memory.tensors(step, image, sensor, "cpu")
            states, contacts, _ = model(
                batch, torch.as_tensor(data["admitted"][step], dtype=torch.long),
                torch.as_tensor(data["boost"][step]),
            )
            actions = torch.as_tensor(data["action"][step], dtype=torch.long)
            predicted = states[torch.arange(count), actions].numpy()
            error = float(np.max(np.abs(predicted - data["predicted"][step])))
            if error > 3e-5:
                raise AssertionError((target.name, probe, step, error))
            errors.append(error)
        archive.connection.close()
    return {"run": target.name, "replayed_predictions": len(errors) * count, "max_abs_error": max(errors)}


if __name__ == "__main__":
    torch.set_num_threads(2)
    diagnostics(sys.argv[1])
    if "--checkpoint" in sys.argv:
        folder = ROOT / sys.argv[1]
        checks = [
            replay_checkpoint(next(folder.glob(f"g{groups}_s23_*_delayed")))
            for groups in (9, 78)
        ]
        dump(folder / "CHECKPOINT_AUDIT.json", checks)
        print(json.dumps(checks))
