"""Plot the declared held-out readouts without selecting a best checkpoint."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def main():
    rows = json.loads((ROOT / "pilot" / "SUMMARY.json").read_text())["rows"]
    modes = ("state_content", "state_time", "no_query_content")
    colors = ("#006d77", "#b24724", "#6e4b8b")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    panels = (("success", "Actual action success (%)", 100),
              ("critical", "Critical four-step success (%)", 100),
              ("cost", "Mean physical cost (lower is better)", 1))
    for ax, (metric, title, scale) in zip(axes, panels):
        for mode, color in zip(modes, colors):
            selected = sorted((row for row in rows if row["mode"] == mode
                               and row["probe"] == "composition"),
                              key=lambda row: row["experiences"])
            ax.plot([row["experiences"] for row in selected],
                    [scale * row[metric] for row in selected],
                    color=color, marker="o", linewidth=1.6, markersize=4, label=mode)
        ax.set_xscale("log", base=4)
        ax.set_xticks((65536, 262144, 1048576), ("65,536", "262,144", "1,048,576"))
        ax.set_xlabel("New experiences / model", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("V12 held-out combinations | one seed | frozen weights | descriptive only", fontsize=11)
    path = ROOT / "TRAINING_CURVES.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(str(path), flush=True)


if __name__ == "__main__":
    main()
