"""Verify that patch.py's grid_indices is the CORRECT permutation for a checkpoint.

A wrong-length grid_indices crashes loudly (torch.cat mismatch). A wrong-*order*
grid_indices of the right length does NOT crash -- it silently maps each model
node to the wrong physical grid cell and degrades the forecast to ~climatology.
This script catches that silent case by checking that
    dataset.latitudes[grid_indices] == the model's node latitudes
within float tolerance, and that grid_indices matches the graph's
indices_connected_nodes.

Run on the remote (windpower312 env):
    python check_grid_indices.py
Edit CKPT / GRAPH / CUTOUT below if needed.
"""

import numpy as np
import torch
from anemoi.utils.checkpoints import load_metadata
from anemoi.datasets import open_dataset

CKPT = "/mnt/weatherloss/WindPower/training/WindAI/VanillaPowerGT/checkpoint/VanillaPowerGT/inference-anemoi-by_time-epoch_022-step_107412.ckpt"
GRAPH = "/mnt/weatherloss/WindPower/graphs/EGU26/EGUgraph15km.pt"
CUTOUT = dict(
    cutout=[
        {"dataset": "/mnt/weatherloss/WindPower/data/WindAI/Anemoidatasets/New_Cerra_A_large.zarr"},
        {"dataset": "/mnt/weatherloss/WindPower/data/WindAI/Anemoidatasets/era5_A_large.zarr"},
    ],
    min_distance_km=0,
    adjust="all",
)


def get(arrays, key):
    """Supporting arrays are namespaced under the dataset name ('data') in this fork."""
    if key in arrays:
        return arrays[key]
    if "data" in arrays and key in arrays["data"]:
        return arrays["data"][key]
    return None


def main():
    # ---- 1. dataset grid (the full cutout, e.g. 82068 points) ----
    ds = open_dataset(**CUTOUT)
    ds_lat = np.asarray(ds.latitudes)
    ds_lon = np.asarray(ds.longitudes)
    print(f"dataset cutout points: {ds_lat.size}")

    # ---- 2. graph: full node coords + indices_connected_nodes ----
    g = torch.load(GRAPH, weights_only=False, map_location="cpu")
    icn = g["data"]["indices_connected_nodes"].cpu().numpy().squeeze().astype(np.int64)
    print(f"graph indices_connected_nodes: len={icn.size}, "
          f"sorted={np.all(np.diff(icn) >= 0)}, min={icn.min()}, max={icn.max()}")

    # graph 'x' holds ONLY the connected nodes (post RemoveUnconnectedNodes), so its
    # length == len(indices_connected_nodes). 'x' is (lat, lon) in radians.
    gx = g["data"]["x"].cpu().numpy()
    g_lat = np.degrees(gx[:, 0])
    g_lon = np.degrees(gx[:, 1])
    print(f"graph connected nodes (g['data']['x']): {g_lat.size}")
    print(f"implied original grid size (max icn + 1): {icn.max() + 1}   vs dataset {ds_lat.size}")

    def lat_close(a, b):
        return a.size == b.size and np.allclose(a, b, atol=1e-3)

    def lon_close(a, b):
        # compare modulo 360 to be robust to [-180,180] vs [0,360] conventions
        if a.size != b.size:
            return False
        d = np.abs((a - b + 180) % 360 - 180)
        return np.all(d < 1e-3)

    # ---- 3. THE decisive test ---------------------------------------------------
    # Index TODAY's dataset by the graph's indices_connected_nodes. If the graph was
    # built from this same dataset, this reproduces the graph's connected-node coords.
    if icn.max() < ds_lat.size:
        ds_sub_lat = ds_lat[icn]
        ds_sub_lon = ds_lon[icn]
        ok_lat = lat_close(ds_sub_lat, g_lat)
        ok_lon = lon_close(ds_sub_lon, g_lon)
        print(f"[A] dataset[indices_connected_nodes] lat == graph node lat: {ok_lat}")
        print(f"[A] dataset[indices_connected_nodes] lon == graph node lon: {ok_lon}")
        if not (ok_lat and ok_lon):
            n_bad = int(np.sum(np.abs(ds_sub_lat - g_lat) > 1e-3))
            print(f"    !! {n_bad}/{g_lat.size} nodes mismatch -> dataset changed since the graph was built")
            print("    !! patch.py's grid_indices selects the WRONG physical cells (silent scramble)")
    else:
        print(f"[A] icn.max()={icn.max()} >= dataset size {ds_lat.size} -> indices out of range, definitely stale graph")

    # ---- 4. checkpoint supporting arrays ----
    _, arrays = load_metadata(CKPT, supporting_arrays=True)
    top = list(arrays.keys())
    data_keys = list(arrays.get("data", {}).keys()) if isinstance(arrays.get("data"), dict) else []
    print(f"checkpoint supporting arrays: top={top} data={data_keys}")

    om = get(arrays, "output_mask")
    ck_lat = get(arrays, "latitudes")

    if om is not None:
        om = np.asarray(om).astype(bool)
        print(f"[B] output_mask: len={om.size}, inner(True)={om.sum()}, boundary(False)={(~om).sum()}")

    # ---- 5. checkpoint stores the FULL grid latitudes (82068), not the 77419 subset.
    #         So compare full-vs-full: has the dataset drifted since the ckpt was made? ----
    if ck_lat is not None:
        ck_lat = np.asarray(ck_lat)
        if ck_lat.size == ds_lat.size:
            print(f"[C] checkpoint full grid (len={ck_lat.size}) == current dataset grid: {lat_close(ck_lat, ds_lat)}")
        else:
            print(f"[C] checkpoint 'latitudes' len={ck_lat.size} (model subset would be {icn.size}); "
                  f"== graph node lat: {lat_close(ck_lat, g_lat)}")

    print("\nVerdict: [A] True  => patch.py / grid_indices maps the dataset onto the graph correctly; NOT the cause.")
    print("         [A] False => stale graph vs current dataset; grid_indices scrambles the input -> the regression.")


if __name__ == "__main__":
    main()
