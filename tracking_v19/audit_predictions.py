"""Replay raw visual traces and sample saved predictor forward calculations."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import zlib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
from tracking_v17.learner import Predictor
from tracking_v17.run import dump

EYES, MEMBERS = 128, 3


def _checkpoints(path, summary):
    probe = re.fullmatch(r"probe_(\d+)_(frozen|adaptive)", path.name)
    if probe:
        step = int(probe.group(1))
        return {0: path.parent / ("initial.pt" if step == 0 else f"checkpoint_{step}.pt")}
    result = {0: path / "initial.pt"}
    if summary["training"]:
        for checkpoint in path.glob("checkpoint_*.pt"):
            step = int(checkpoint.stem.split("_")[1])
            if 0 < step < summary["step"]:
                result[step] = checkpoint
    return result


def _load_predictors(checkpoint, device):
    state = torch.load(checkpoint, map_location=device, weights_only=True)["model"]["predictors"]
    predictors = []
    for member in range(MEMBERS):
        prefix = f"{member}."
        model = Predictor().to(device)
        model.load_state_dict({key[len(prefix):]: value for key, value in state.items()
                               if key.startswith(prefix)})
        model.eval()
        predictors.append(model)
    return predictors


@torch.no_grad()
def _recompute(predictors, full, proprio, actions, starts, after, floor, device):
    x = torch.from_numpy(full).to(device)
    prop = torch.from_numpy(proprio).to(device)
    action = torch.as_tensor(actions, device=device, dtype=torch.long)
    erased = x.clone()
    support = []
    for body, start in enumerate(starts):
        erased[body, 6:11, [(start + i) % EYES for i in range(16)]] = 0
        support.append([(start + i) % EYES for i in range(-4, 20)])
    normal = torch.stack([model(x, prop, action) for model in predictors], 1)
    omitted = torch.stack([model(erased, prop, action) for model in predictors], 1)
    label = torch.from_numpy(after).to(device).permute(0, 2, 1).float() / 255
    normal_local, omitted_local, disagreement = [], [], []
    for body, eyes in enumerate(support):
        target = label[body][:, eyes]
        complete = normal[body][:, :, eyes]
        missing = omitted[body][:, :, eyes]
        normal_local.append((complete - target).square().mean((1, 2)))
        omitted_local.append((missing - target).square().mean((1, 2)))
        disagreement.append(complete.var(0, unbiased=False).mean())
    normal_local = torch.stack(normal_local)
    omitted_local = torch.stack(omitted_local)
    values = dict(
        normal_local=normal_local,
        omitted_local=omitted_local,
        scores=(omitted_local - normal_local) / (omitted_local + normal_local + floor),
        prediction_disagreement=torch.stack(disagreement))
    return {key: value.cpu().numpy() for key, value in values.items()}


def audit_session(path, checkpoints=None, device="cuda", bodies=4,
                  atol=2e-6, score_atol=2e-5, rtol=2e-4):
    """Replay selected bodies from raw time zero, then audit each anchor's row."""
    path = Path(path).resolve()
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    checkpoints = (_checkpoints(path, summary) if checkpoints is None else
                   {int(step): Path(value) for step, value in checkpoints.items()})
    if not checkpoints:
        raise ValueError("At least one saved-weight anchor is required")
    count = min(int(bodies), int(summary["worlds"]))
    if count < 1 or min(checkpoints) < 0 or max(checkpoints) >= summary["step"]:
        raise ValueError("Requested bodies or anchor are outside this session")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass device='cpu' for forward audit")
    for checkpoint in checkpoints.values():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    floor = float(summary["floor"])
    if not np.isfinite(floor) or floor <= 0:
        raise AssertionError("Invalid recorded evidence floor")
    previous = np.zeros((count, EYES, 3), np.float32)
    history = np.zeros_like(previous)
    age = np.zeros((count, EYES), np.float32)
    old_segment = np.full(count, -1, np.int64)
    records = []
    max_error = {}
    database = sqlite3.connect(f"{(path / 'events.sqlite').as_uri()}?mode=ro", uri=True)
    stream = None
    try:
        stream = database.execute(
            "SELECT body,time,segment,rgb,proprio,action,next_rgb FROM events "
            "WHERE body<? AND time<=? ORDER BY time,body", (count, max(checkpoints)))
        for step in range(max(checkpoints) + 1):
            rows = stream.fetchmany(count)
            if len(rows) != count or [(r[0], r[1]) for r in rows] != [
                    (body, step) for body in range(count)]:
                raise AssertionError(f"{path.name}: missing raw observation at step {step}")
            segment = np.array([row[2] for row in rows], np.int64)
            rgb = np.stack([np.frombuffer(zlib.decompress(row[3]), np.uint8)
                            .reshape(EYES, 3) for row in rows])
            prop = np.stack([np.frombuffer(row[4], np.float32) for row in rows])
            if prop.shape != (count, 6):
                raise AssertionError("Invalid stored proprioception")
            shifts = np.rint(prop[:, 2] * 64).astype(int)
            for body, shift in enumerate(shifts):
                if segment[body] != old_segment[body]:
                    previous[body] = history[body] = 0
                    age[body] = 0
                elif shift:
                    previous[body] = np.roll(previous[body], -shift, axis=0)
                    history[body] = np.roll(history[body], -shift, axis=0)
                    age[body] = np.roll(age[body], -shift)
            current = rgb.astype(np.float32) / 255
            if summary.get("eye_disabled", False):
                current.fill(0)
            gate = np.exp(-age / 32)
            full = np.concatenate(
                (current, previous, history * gate[:, :, None], gate[:, :, None],
                 np.minimum(age / 128, 1)[:, :, None]), axis=2).transpose(0, 2, 1).copy()
            if step in checkpoints:
                evidence = database.execute(
                    "SELECT body,start,normal_errors,omitted_errors,scores,disagreement "
                    "FROM evidence WHERE time=? AND body<? ORDER BY body", (step, count)
                ).fetchall()
                if len(evidence) != count or [row[0] for row in evidence] != list(range(count)):
                    raise AssertionError("Missing sampled evidence")
                starts = [row[1] for row in evidence]
                expected_starts = [((step + 3 * body) % 8) * 16 for body in range(count)]
                if starts != expected_starts:
                    raise AssertionError("Unexpected intervention group")
                observed = {
                    key: np.stack([np.frombuffer(row[column], np.float32) for row in evidence])
                    for key, column in (("normal_local", 2), ("omitted_local", 3), ("scores", 4))}
                observed["prediction_disagreement"] = np.asarray([row[5] for row in evidence])
                after = np.stack([np.frombuffer(zlib.decompress(row[6]), np.uint8)
                                  .reshape(EYES, 3) for row in rows])
                checkpoint = checkpoints[step]
                predictors = _load_predictors(checkpoint, device)
                actual = _recompute(predictors, full, prop, [row[5] for row in rows],
                                    starts, after, floor, device)
                errors = {}
                for key, expected in observed.items():
                    if actual[key].shape != expected.shape:
                        raise AssertionError(f"Invalid stored {key} shape")
                    error = float(np.max(np.abs(actual[key] - expected)))
                    errors[key] = error
                    max_error[key] = max(max_error.get(key, 0.), error)
                    np.testing.assert_allclose(
                        actual[key], expected, rtol=rtol,
                        atol=score_atol if key == "scores" else atol,
                        err_msg=f"{path.name} step={step} {key}")
                records.append(dict(
                    session=str(path), anchor=step, nrows=count, max_error=errors,
                    checkpoint=str(checkpoint),
                    checkpointsha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
                del predictors
            # These three states depend only on raw observations, not access masks.
            changed = np.abs(current - previous).mean(2) > .07
            keep = np.where(changed, .96 * gate, .65 * gate)
            history = (keep[:, :, None] * history +
                       (1 - keep[:, :, None]) * current).astype(np.float32)
            age = np.where(changed, 0, age + 1).astype(np.float32)
            previous = current.copy()
            old_segment = segment
    finally:
        if stream is not None:
            stream.close()
        database.close()
    return dict(session=str(path), anchors=records, prediction_rows=len(records) * count,
                reconstructed_rows=(max(checkpoints) + 1) * count,
                max_error=max_error, device=device,
                tolerance=dict(atol=atol, score_atol=score_atol, rtol=rtol))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    output = ROOT / config["stage"]
    if not (output / "COMPLETE.json").is_file():
        raise RuntimeError("Finish the registered experiment before the prediction audit")
    config = json.loads((output / "manifest.json").read_text(encoding="utf-8"))["config"]
    paths = [output / "calibration"]
    for arm in config["arms"]:
        paths.append(output / arm)
        paths.extend(sorted((output / arm).glob("probe_*")))
    results = [audit_session(path, device=args.device) for path in paths]
    result = dict(
        sessions=results, all_prediction_rows=sum(row["prediction_rows"] for row in results),
        all_reconstructed_rows=sum(row["reconstructed_rows"] for row in results),
        scope="First four bodies at saved-weight anchors; sampled forward verification only.")
    dump(output / "PREDICTION_AUDIT.json", result)
    print(json.dumps(dict(sessions=len(results), rows=result["all_prediction_rows"])), flush=True)


if __name__ == "__main__":
    main()
