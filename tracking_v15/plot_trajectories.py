"""Plot the first recorded episode of a predeclared barrier/attack scene."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

ROOT = Path(__file__).resolve().parent


def episode(mode):
    folder = ROOT / "pilot" / mode / "probe_8192" / "heldout"
    parts = []
    for path in sorted(folder.glob("transitions_*.npz")):
        with np.load(path) as data:
            parts.append({key: data[key][:, 6].copy()
                          for key in ("features", "episode", "done", "success", "impact")})
    full = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    mask = full["episode"] == 0
    return {key: value[mask] for key, value in full.items()}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, mode in zip(axes, ("temporal", "risk_alloc")):
        data = episode(mode)
        features = data["features"]
        position = features[:, 26:28] * 6
        target = position + features[:, 2:4] * 6
        ax.plot(position[:, 0], position[:, 1], color="#147d92", lw=2, label="Body")
        ax.scatter(position[0, 0], position[0, 1], marker="s", color="#147d92", s=45)
        ax.scatter(position[-1, 0], position[-1, 1], marker="o", color="#147d92", s=45)
        ax.plot(target[:, 0], target[:, 1], color="#24a05b", linestyle="--", label="Marked target")
        ax.scatter(target[-1, 0], target[-1, 1], color="#24a05b", marker="*", s=140)
        obstacles = set()
        for row, body in zip(features, position):
            for index in range(3):
                offset = 8 + 3 * index
                radius = row[offset + 2]
                if radius > 0:
                    point = row[offset:offset + 2] * 6 + body
                    obstacles.add((round(float(point[0]), 2), round(float(point[1]), 2),
                                   round(float(radius), 2)))
        for x, y, radius in obstacles:
            ax.add_patch(Circle((x, y), radius, color="#737b7a", alpha=0.65))
        projectiles = []
        for row, body in zip(features, position):
            for index in range(3):
                offset = 17 + 3 * index
                if row[offset + 2] > 0:
                    projectiles.append(row[offset:offset + 2] * 6 + body)
        if projectiles:
            points = np.asarray(projectiles)
            ax.scatter(points[:, 0], points[:, 1], c="#bd503c", s=6, alpha=0.3,
                       label="Observed projectile positions")
        outcome = ("contact" if data["success"].any() else
                   "impact" if data["impact"].any() else "timeout")
        ax.set_title(f"{mode}: first episode, {outcome}, {len(features)} steps", fontsize=10)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=7, frameon=False)
        ax.grid(alpha=0.15)
    fig.suptitle("Actual recorded trajectories: barrier, stationary target, aimed attacks",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "ACTUAL_TRAJECTORIES.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
