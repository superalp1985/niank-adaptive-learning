"""Export larger induced connectome controls; no task data are inspected."""

import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fly_connectome import ROOT, SOURCE, load_upstream, prepare_reduced


def main():
    load_upstream()
    import benchmark

    ids = pd.read_csv(SOURCE / "data" / "2025_Completeness_783.csv").iloc[:, 0].to_numpy(np.int64)
    table = pq.read_table(
        SOURCE / "data" / "2025_Connectivity_783.parquet",
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    pre = table.column(0).to_numpy().astype(np.int64, copy=False)
    post = table.column(1).to_numpy().astype(np.int64, copy=False)
    weight = table.column(2).to_numpy().astype(np.float32)
    indices = {int(value): i for i, value in enumerate(ids)}
    seeds = np.array([indices[int(i)] for i in benchmark.get_experiment("sugar")["neu_exc"]])
    results = []
    for size in (512, 1024):
        stem = f"connectome_{size}"
        result = prepare_reduced(pre, post, weight, ids, seeds, size, stem)
        result["sha256"] = hashlib.sha256((ROOT / f"{stem}.npz").read_bytes()).hexdigest()
        results.append(result)
        print(json.dumps({"event": "capacity_export", **result}), flush=True)
    (ROOT / "capacity_exports.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
