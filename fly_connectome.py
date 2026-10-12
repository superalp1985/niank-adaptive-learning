"""Prepare a reduced structural control and run upstream whole-graph LIF dynamics.

The imported upstream simulator remains GPL-2.0-or-later. No pretrained task
knowledge or biological learning is claimed by the reduced rate-network export.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
COMMIT = "a3db62f9436074e485c0278290c2164ed6150808"
SOURCE = ROOT.parents[1] / "external" / "fly-brain" / COMMIT


def emit(stage, **values):
    print(json.dumps({"stage": stage, **values}), flush=True)


def load_upstream():
    sys.path.insert(0, str(SOURCE / "code"))
    spec = importlib.util.spec_from_file_location(
        "upstream_fly_torch", SOURCE / "code" / "run_pytorch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rewire(matrix, seed=711, swaps_per_edge=12):
    """Signed double-edge swaps preserve both signed in- and out-degree."""
    rng = np.random.default_rng(seed)
    result = matrix.copy()
    swaps = 0
    for sign in (1, -1):
        rows, cols = np.nonzero(matrix * sign > 0)
        occupied = set(zip(*np.nonzero(result)))
        for _ in range(len(rows) * swaps_per_edge):
            a, b = rng.integers(0, len(rows), 2)
            r1, c1, r2, c2 = int(rows[a]), int(cols[a]), int(rows[b]), int(cols[b])
            if r1 == r2 or c1 == c2:
                continue
            if (r1, c2) in occupied or (r2, c1) in occupied:
                continue
            v1, v2 = result[r1, c1], result[r2, c2]
            result[r1, c1] = result[r2, c2] = 0
            result[r1, c2], result[r2, c1] = v1, v2
            occupied.remove((r1, c1))
            occupied.remove((r2, c2))
            occupied.add((r1, c2))
            occupied.add((r2, c1))
            cols[a], cols[b] = c2, c1
            swaps += 1
    for sign in (1, -1):
        for axis in (0, 1):
            np.testing.assert_array_equal(
                (matrix * sign > 0).sum(axis), (result * sign > 0).sum(axis)
            )
    np.testing.assert_array_equal(np.sort(matrix.ravel()), np.sort(result.ravel()))
    return result, swaps


def prepare_reduced(pre, post, weight, neuron_ids, seeds, size=256, artifact_stem="connectome_reduced"):
    n = len(neuron_ids)
    selected = list(map(int, seeds))
    # Select by observed outgoing connection mass, without task labels.
    for _ in range((size + 63) // 64):
        mass = np.bincount(
            post, weights=np.abs(weight) * np.isin(pre, selected), minlength=n
        )
        mass[selected] = -1
        take = min(size - len(selected), 64)
        if take <= 0:
            break
        chosen = np.argsort(-mass, kind="stable")[:take]
        selected.extend(chosen.tolist())
    if len(selected) < size:
        mass = np.bincount(post, weights=np.abs(weight), minlength=n)
        mass[selected] = -1
        selected.extend(np.argsort(-mass, kind="stable")[:size-len(selected)].tolist())
    selected = np.asarray(selected, dtype=np.int64)
    mapping = np.full(n, -1, dtype=np.int64)
    mapping[selected] = np.arange(len(selected))
    keep = (mapping[pre] >= 0) & (mapping[post] >= 0)
    matrix = np.zeros((size, size), dtype=np.float32)
    np.add.at(matrix, (mapping[post[keep]], mapping[pre[keep]]), weight[keep])
    randomized, swaps = rewire(matrix)
    np.savez_compressed(
        ROOT / f"{artifact_stem}.npz",
        ids=neuron_ids[selected],
        source_indices=selected,
        matrix=matrix,
        rewired_matrix=randomized,
    )
    details = {
        "nodes": size,
        "nonzero_edges": int(np.count_nonzero(matrix)),
        "positive_edges": int(np.count_nonzero(matrix > 0)),
        "negative_edges": int(np.count_nonzero(matrix < 0)),
        "accepted_double_edge_swaps": swaps,
        "edge_overlap_fraction": float(
            np.count_nonzero((matrix != 0) & (randomized != 0))
            / np.count_nonzero(matrix)
        ),
        "selection": "official sugar neurons then strongest outgoing mass, at most 64 per round",
        "control": "signed degree-preserving double-edge swaps; weight multiset preserved",
        "not_preserved": "individual weighted degrees and full-brain dynamics",
        "scope": "induced structural prior, not a complete brain or trained model",
        "source_commit": COMMIT,
    }
    (ROOT / f"{artifact_stem}.json").write_text(json.dumps(details, indent=2))
    return details


def main():
    started = time.perf_counter()
    assert torch.cuda.is_available(), "CUDA is required; no CPU fallback."
    torch.set_num_threads(4)
    upstream = load_upstream()
    import benchmark as upstream_benchmark
    neurons = pd.read_csv(SOURCE / "data" / "2025_Completeness_783.csv").iloc[:, 0]
    neuron_ids = neurons.to_numpy(dtype=np.int64)
    table = pq.read_table(
        SOURCE / "data" / "2025_Connectivity_783.parquet",
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    pre = table.column(0).to_numpy().astype(np.int64, copy=False)
    post = table.column(1).to_numpy().astype(np.int64, copy=False)
    weight = table.column(2).to_numpy().astype(np.float32)
    del table
    n = len(neuron_ids)
    assert pre.min() >= 0 and post.min() >= 0 and pre.max() < n and post.max() < n
    id_to_index = {int(value): i for i, value in enumerate(neuron_ids)}
    sugar = upstream_benchmark.get_experiment("sugar")
    stimulated = np.array([id_to_index[int(i)] for i in sugar["neu_exc"]], dtype=np.int64)
    reduced = prepare_reduced(pre, post, weight, neuron_ids, stimulated)
    emit("reduced_ready", **reduced)
    index = torch.from_numpy(np.stack([post, pre]))
    sparse = torch.sparse_coo_tensor(index, torch.from_numpy(weight), (n, n)).coalesce()
    sparse = sparse.to_sparse_csr().cuda()
    del index
    torch.cuda.synchronize()
    emit("full_graph_ready", neurons=n, stored_connections=len(weight), nonzero=sparse._nnz())
    stimulated_t = torch.as_tensor(stimulated, device="cuda")
    rates = torch.zeros(3, n, device="cuda")
    rates[1, stimulated_t] = 100.0
    rates[2, stimulated_t] = 200.0
    results = []
    for seed in (811, 812, 813):
        model = upstream.TorchModel(
            batch=3, size=n, dt=upstream.DT, params=upstream.MODEL_PARAMS,
            weights=sparse, exc_indices=stimulated_t, device="cuda",
        )
        state = model.state_init()
        totals = torch.zeros(3, n, device="cuda")
        window_counts = []
        generator = torch.Generator(device="cuda").manual_seed(seed)
        begin = time.perf_counter()
        with torch.no_grad():
            for step in range(1000):
                state = model(rates, *state, generator=generator)
                totals += state[2]
                if (step + 1) % 100 == 0:
                    window_counts.append(totals.sum(1).cpu().tolist())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        counts = totals.cpu().numpy()
        for b, rate in enumerate((0, 100, 200)):
            active = counts[b] > 0
            downstream = active.copy()
            downstream[stimulated] = False
            results.append({
                "seed": seed, "rate_hz": rate, "simulated_ms": 100,
                "spikes": int(counts[b].sum()),
                "active_neurons": int(active.sum()),
                "active_unstimulated_neurons": int(downstream.sum()),
                "cumulative_spikes_per_10ms": [v[b] for v in window_counts],
                "batch_elapsed_seconds": elapsed,
            })
        np.savez_compressed(ROOT / f"lif_counts_{seed}.npz", ids=neuron_ids, counts=counts)
        emit("seed_done", seed=seed, seconds=elapsed, records=results[-3:])
    assert all(r["spikes"] == 0 for r in results if r["rate_hz"] == 0)
    report = {
        "source_commit": COMMIT, "neurons": n, "input_connection_rows": len(weight),
        "stored_sparse_connections": sparse._nnz(), "gpu": torch.cuda.get_device_name(0),
        "model": "unmodified upstream TorchModel LIF dynamics",
        "dt_ms": upstream.DT, "results": results, "reduced": reduced,
        "whole_run_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "scope": "activity propagation only; no task training or accuracy claim",
        "source_hashes": {
            name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
            for name in ("code/run_pytorch.py", "data/2025_Connectivity_783.parquet")
        },
    }
    (ROOT / "lif_results.json").write_text(json.dumps(report, indent=2))
    emit("finished", elapsed_seconds=report["whole_run_seconds"])


if __name__ == "__main__":
    main()
