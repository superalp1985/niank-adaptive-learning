"""Inspect sector gain magnitude before spatial averaging can cancel signs."""

import json
import sqlite3
import zlib

import numpy as np
import torch

from tracking_v17.run import ROOT, dump


def main():
    results = {}
    for mode in ("fixed", "revisable"):
        db = sqlite3.connect(ROOT / "pilot" / mode / "events.sqlite")
        cursor = db.execute("SELECT time,retention_gain FROM events ORDER BY time,body")
        phases = [dict(cells=0, absolute_sum=0., signed_sum=0., above_001=0, above_01=0)
                  for _ in range(3)]
        while rows := cursor.fetchmany(8192):
            clocks = np.array([r[0] for r in rows])
            gains = np.frombuffer(b"".join(zlib.decompress(r[1]) for r in rows),
                                  np.float32).reshape(len(rows), 128)
            for index, (start, end) in enumerate(((0, 2048), (2048, 8192), (8192, 16384))):
                local = gains[(clocks >= start) & (clocks < end)]
                if not local.size:
                    continue
                p = phases[index]
                p["cells"] += local.size
                p["absolute_sum"] += float(np.abs(local).sum(dtype=np.float64))
                p["signed_sum"] += float(local.sum(dtype=np.float64))
                p["above_001"] += int((np.abs(local) > .001).sum())
                p["above_01"] += int((np.abs(local) > .01).sum())
        db.close()
        for p in phases:
            for field in ("absolute_sum", "signed_sum", "above_001", "above_01"):
                p[field + "_per_cell"] = p[field] / p["cells"]
        checkpoint = torch.load(ROOT / "pilot" / mode / "checkpoint_16384.pt",
                                map_location="cpu", weights_only=False)
        utility = checkpoint["memory"]["utility"]
        age = checkpoint["memory"]["age"]
        fixed = np.exp(-age / 32)
        adjustable = np.clip(fixed * np.clip(1 + utility * 12, .1, 1.9), 0, 1)
        results[mode] = dict(phases=phases, final_snapshot=dict(
            absolute_sector_utility_mean=float(np.abs(utility).mean()),
            absolute_sector_utility_max=float(np.abs(utility).max()),
            absolute_gate_change_mean=float(np.abs(adjustable - fixed).mean()),
            absolute_gate_change_max=float(np.abs(adjustable - fixed).max())),
            note=("Post-hoc descriptive only; V17 next-frame utility rotation defect remains. "
                  "Final checkpoint follows a 1024-step physical reset, so its cleared utility "
                  "does not measure the utility throughout learning."))
    dump(ROOT / "pilot/RETENTION_DIAGNOSTICS.json", results)
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
