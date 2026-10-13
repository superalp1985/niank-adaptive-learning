"""Break target traces at respawns rather than drawing teleportation as motion."""

from pathlib import Path
import sqlite3
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    for row, mode in enumerate(("fixed", "revisable")):
        directory = ROOT / "pilot" / mode / "probe_16384"
        with np.load(sorted(directory.glob("transitions_*.npz"))[0]) as d:
            body = d["body_position"][:, 0].copy()
            target = d["target_position"][:, 0].copy()
            goals = d["goal_id"][:, 0]
        target[1:][goals[1:] != goals[:-1]] = np.nan
        axes[row, 0].plot(body[:, 0], body[:, 1], color="#267f91", label="body")
        axes[row, 0].plot(target[:, 0], target[:, 1], color="#bf415c", label="target")
        axes[row, 0].set(xlim=(-4, 4), ylim=(-4, 4), title=mode + ": first 128 steps")
        axes[row, 0].set_aspect("equal")
        axes[row, 0].legend()
        db = sqlite3.connect(directory / "events.sqlite")
        rows = db.execute("SELECT rgb FROM events WHERE body=0 ORDER BY time").fetchall()
        db.close()
        image = np.stack([np.frombuffer(zlib.decompress(v[0]), np.uint8).reshape(128, 3)
                          for v in rows])
        axes[row, 1].imshow(image, aspect="auto", extent=(0, 360, len(image), 0))
        axes[row, 1].set(xlabel="Compound-eye azimuth (degrees)", ylabel="Time step",
                         title=mode + ": actual 360-degree RGB")
    fig.savefig(ROOT / "ACTUAL_EYES_CORRECTED.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
