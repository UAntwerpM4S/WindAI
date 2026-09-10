"""Does the graph file patch.py reads == the graph baked into the trained model?

check_grid_indices.py proved patch.py's graph is consistent with the DATASET.
It did NOT prove it's the SAME graph the model was trained on. If EGUgraph15km.pt
was regenerated after training (e.g. re-running the EGUgraph.yaml recipe, or a
later run rebuilding it), patch.py injects indices_connected_nodes from a graph
the model never saw -> same node count (no crash) but wrong node ORDER -> silent
spatial scramble -> ~climatology RMSE.

The model stores its own graph in `._graph_data` (see anemoi/models/models/base.py).
This compares the model's data-node coords against the graph file's. If they match
elementwise, patch.py is reading the model's graph and is fully exonerated.

Run on the remote (windpower312 env):
    python check_graph_matches_model.py
"""

import numpy as np
import torch

CKPT = "/mnt/weatherloss/WindPower/training/WindAI/VanillaPowerGT/checkpoint/VanillaPowerGT/inference-anemoi-by_time-epoch_022-step_107412.ckpt"
GRAPH = "/mnt/weatherloss/WindPower/graphs/EGU26/EGUgraph15km.pt"
NODE = "data"


def model_node_coords(m, name=NODE):
    """Decode the model's persisted node coords from buffer latlons_<name> = [sin(x), cos(x)]."""
    target = f"latlons_{name}"
    buf = None
    for bname, b in m.named_buffers():
        if bname.split(".")[-1] == target:
            buf = b
            break
    if buf is None:
        names = [n for n, _ in m.named_buffers()]
        raise RuntimeError(f"buffer {target!r} not found. available buffers:\n{names}")
    sc = buf.detach().cpu().numpy()
    ndim = sc.shape[1] // 2
    sin_v, cos_v = sc[:, :ndim], sc[:, ndim:]
    rad = np.arctan2(sin_v, cos_v)            # (N, 2) -> (lat, lon) in radians
    return np.degrees(rad[:, 0]), np.degrees(rad[:, 1])


def file_node_coords(g, name=NODE):
    x = g[name].x
    x = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return np.degrees(x[:, 0]), np.degrees(x[:, 1])


def main():
    # ---- model's own node coords (what it was actually trained on) ----
    m = torch.load(CKPT, weights_only=False, map_location="cpu")
    m_lat, m_lon = model_node_coords(m)
    print(f"model graph nodes: {m_lat.size}")

    # ---- the graph file patch.py reads ----
    g = torch.load(GRAPH, weights_only=False, map_location="cpu")
    f_lat, f_lon = file_node_coords(g)
    f_icn = g[NODE]["indices_connected_nodes"].cpu().numpy().squeeze().astype(np.int64)
    print(f"file  graph nodes: {f_lat.size}  (icn max {f_icn.max()})")

    # ---- compare ----
    if m_lat.size != f_lat.size:
        print(f"[X] SIZE DIFF model {m_lat.size} vs file {f_lat.size} -> DIFFERENT graphs. patch.py graph is NOT the model's.")
        return

    lat_ok = np.allclose(m_lat, f_lat, atol=1e-4)
    lon_ok = np.all(np.abs((m_lon - f_lon + 180) % 360 - 180) < 1e-4)
    print(f"[1] model node lat == file node lat: {lat_ok}")
    print(f"[1] model node lon == file node lon: {lon_ok}")

    if lat_ok and lon_ok:
        print("\n=> patch.py reads the SAME graph the model trained on. Fully exonerated.")
    else:
        n_bad = int(np.sum(np.abs(m_lat - f_lat) > 1e-4))
        print(f"\n=> MISMATCH on {n_bad}/{m_lat.size} nodes. The .pt differs from the model's graph:")
        print("   patch.py's grid_indices scrambles the model's nodes -> THIS is the regression.")
        print("   Fix: rebuild the checkpoint's grid_indices from the graph the model was trained on")
        print("   (or retrain/patch so the file and the model agree).")


if __name__ == "__main__":
    main()
