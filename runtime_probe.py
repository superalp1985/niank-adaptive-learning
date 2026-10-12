"""Bounded environment and dataset probe; no model training."""

import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / "external" / "fly-brain" / "a3db62f9436074e485c0278290c2164ed6150808"


def mark(stage, **values):
    print(json.dumps({"stage": stage, **values}), flush=True)


mark("start", timestamp=time.time())
import pyarrow.parquet as pq
mark("pyarrow_imported")
import numpy as np
import torch
mark("torch_imported", version=torch.__version__)
assert torch.cuda.is_available(), "This experiment requires CUDA."
mark("cuda_ready", gpu=torch.cuda.get_device_name(0))
schema = pq.read_schema(SOURCE / "data" / "2025_Connectivity_783.parquet")
meta = pq.read_metadata(SOURCE / "data" / "2025_Connectivity_783.parquet")
mark("dataset", columns=str(schema), rows=meta.num_rows)
value = torch.ones(8, device="cuda").sum().item()
mark("finished", cuda_sum=value)
