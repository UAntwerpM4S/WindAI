#!/usr/bin/env python3
"""patch.py over every checkpoint in a directory: clear dataset.variables_metadata, in place.

Same operation, same env, just a loop. Supporting arrays are read and written straight back, so
latitudes/longitudes/cutout_mask/grid_indices are untouched -- rank_checkpoints.py checks for
exactly those before it will run a checkpoint, and this must not disturb them.
"""

from pathlib import Path

from anemoi.utils.checkpoints import load_metadata, replace_metadata

ROOT = Path("/mnt/weatherloss/WindPower/training/WPDistr/NoWeightPowerFinetune")
GLOB = "**/*.ckpt"

paths = sorted(ROOT.glob(GLOB))
print(f"{len(paths)} checkpoints under {ROOT}\n")

for i, p in enumerate(paths, 1):
    try:
        metadata, arrays = load_metadata(p, supporting_arrays=True)
    except Exception as e:
        # not every .ckpt carries anemoi metadata; one of those must not stop the loop
        print(f"{i:4d} skipped  {p.relative_to(ROOT)}  ({type(e).__name__})")
        continue

    if not metadata.get("dataset", {}).get("variables_metadata"):
        print(f"{i:4d} clean    {p.relative_to(ROOT)}")
        continue

    metadata["dataset"]["variables_metadata"] = {}
    replace_metadata(p, metadata, arrays)
    print(f"{i:4d} patched  {p.relative_to(ROOT)}")

print("\nDone.")
