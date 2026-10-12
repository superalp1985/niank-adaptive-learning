"""Descriptive operation coverage; no extra training or counterfactual labels."""

import json
from pathlib import Path
import sys

import numpy as np

from core import ROOT, Plans
from run import dump


def main(stage):
    output = ROOT / stage
    assert (output / "COMPLETE.json").exists()
    plan_types = Plans().array[:, 0]
    entries = []
    for path in sorted(output.rglob("decisions_*.npz")):
        suffix = path.stem.split("_")[-1]
        with np.load(path) as decisions, np.load(path.with_name(f"outcomes_{suffix}.npz")) as outcomes:
            kind = plan_types[decisions["choices"]]
            reward, estimate = outcomes["reward"], decisions["estimated_score"]
            groups = {}
            for index, name in enumerate(("keep", "retrieve", "fade")):
                selected = kind == index
                count = int(selected.sum())
                groups[name] = dict(
                    count=count, fraction=count / kind.size,
                    mean_reward=float(reward[selected].mean()) if count else None,
                    estimate_bias=float((estimate[selected] - reward[selected]).mean()) if count else None,
                )
            entries.append(dict(path=str(path.relative_to(ROOT)),
                                first_step=int(decisions["step"][0]),
                                last_step=int(decisions["step"][-1]),
                                operations=groups))
    dump(output / "OPERATION_COVERAGE.json", dict(
        interpretation="Descriptive selected-operation statistics, not matched causal treatment effects.",
        exploration="Uniform over legal plans, not over keep/retrieve/fade classes.",
        candidates=dict(keep=1, retrieve=136, fade=16),
        entries=entries,
    ))
    print(json.dumps(dict(segments=len(entries), output="OPERATION_COVERAGE.json")), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
