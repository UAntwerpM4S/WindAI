import shutil
import numpy as np
import torch
from anemoi.utils.checkpoints import load_metadata, replace_metadata

CKPT = "/mnt/weatherloss/WindPower/training/WindAI/VanillaPowerGT/checkpoint/630c9ccef176477c85eb935ad26435f6/inference-anemoi-by_time-epoch_029-step_200000.ckpt"
GRAPH = "/mnt/weatherloss/WindPower/graphs/WindAI/WindAIgraph.pt"
BACKUP = CKPT + ".bak"

shutil.copy2(CKPT, BACKUP)

g = torch.load(GRAPH, weights_only=False)
idx = g["data"]["indices_connected_nodes"].numpy().squeeze().astype(np.int64)

metadata, arrays = load_metadata(CKPT, supporting_arrays=True)

arrays.setdefault("data", {})["grid_indices"] = idx

paths = metadata.setdefault("supporting_arrays_paths", {})
data_paths = paths.setdefault("data", {})

# Derive the path prefix from an existing entry
sample = next(iter(data_paths.values()))
prefix = sample["path"].rsplit("/", 1)[0]  # e.g. "inference-last/anemoi-metadata/data"
print("path prefix:", prefix)

data_paths["grid_indices"] = {
    "path": f"{prefix}/grid_indices.numpy",
    "shape": list(idx.shape),
    "dtype": str(idx.dtype),
}

replace_metadata(CKPT, metadata, arrays)
print("done")
