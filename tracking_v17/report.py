"""Report every milestone rather than selecting a successful checkpoint."""

import json
from pathlib import Path
import sqlite3
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    results = json.loads((ROOT / "pilot/results.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    fields = [("contact_per_minute", "Physical contacts / body minute"),
              ("visible_lock", "Attention lock given visible target"),
              ("collision_rate", "Collision steps / all steps"),
              ("attacked_impact_rate", "Impact steps / attacked-scene steps"),
              ("prediction_mse", "RGB prediction error before updates"),
              ("safe_contact_fraction", "Safe fraction among contacts")]
    for mode, color in (("fixed", "#267f91"), ("revisable", "#bf415c")):
        initial = json.loads((ROOT / f"pilot/{mode}/result_0.json").read_text())
        points = [(0, initial["probes"])] + [(v["step"], v["probes"])
                                            for v in results if v["mode"] == mode]
        for ax, (field, label) in zip(axes.flat, fields):
            ax.plot([v[0] for v in points], [v[1][field] for v in points],
                    "o-", label=mode, color=color)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Actual steps / body")
            ax.grid(alpha=.2)
    axes.flat[0].legend()
    fig.savefig(ROOT / "PROCESS_CURVES.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    for row, mode in enumerate(("fixed", "revisable")):
        directory = ROOT / f"pilot/{mode}/probe_16384"
        files = sorted(directory.glob("transitions_*.npz"))
        with np.load(files[0]) as d:
            body, target = d["body_position"][:, 0], d["target_position"][:, 0]
        axes[row, 0].plot(body[:, 0], body[:, 1], color="#267f91", label="body")
        axes[row, 0].plot(target[:, 0], target[:, 1], color="#bf415c", label="target")
        axes[row, 0].set(xlim=(-4, 4), ylim=(-4, 4), title=mode + ": first 128 steps")
        axes[row, 0].set_aspect("equal")
        axes[row, 0].legend()
        db = sqlite3.connect(directory / "events.sqlite")
        data = db.execute("SELECT rgb FROM events WHERE body=0 ORDER BY time").fetchall()
        image = np.stack([np.frombuffer(zlib.decompress(v[0]), np.uint8).reshape(128, 3)
                          for v in data])
        db.close()
        axes[row, 1].imshow(image, aspect="auto", extent=(0, 360, len(image), 0))
        axes[row, 1].set(xlabel="Compound-eye azimuth (degrees)",
                         ylabel="Time step", title=mode + ": actual 360-degree RGB")
    fig.savefig(ROOT / "ACTUAL_EYES.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
