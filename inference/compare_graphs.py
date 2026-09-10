"""Diff two anemoi graph .pt files (e.g. an 0.8.1 build vs the current 0.9.4 build).

Goal: decide whether the graphs-0.9.4 build differs from the old 0.8.1 build that
the 1.5 m/s model used. Node positions come from the dataset and should match; the
interesting differences are in EDGES (KNN/cutoff/RestrictEdgeLength) and in which
nodes survive RemoveUnconnectedNodes.

Usage:
    # build the OLD graph in an env with anemoi-graphs==0.8.1:
    #   anemoi-graphs create EGUgraph.yaml /tmp/graph_081.pt
    python compare_graphs.py /tmp/graph_081.pt /mnt/weatherloss/WindPower/graphs/EGU26/EGUgraph15km.pt
"""

import sys
import numpy as np
import torch


def load(p):
    return torch.load(p, weights_only=False, map_location="cpu")


def node_summary(g, name):
    ns = g[name]
    n = ns.num_nodes
    extras = {}
    for k in ("cutout", "boundary", "indices_connected_nodes", "area_weight"):
        if k in ns:
            v = ns[k].cpu().numpy().squeeze()
            extras[k] = v
    return n, extras


def edge_stats(g, rel):
    if rel not in g.edge_types:
        return None
    es = g[rel]
    ei = es.edge_index.cpu().numpy()
    n_edges = ei.shape[1]
    # degree per target
    tgt = ei[1]
    deg = np.bincount(tgt)
    out = {
        "n_edges": int(n_edges),
        "n_targets_with_edges": int((deg > 0).sum()),
        "deg_min": int(deg[deg > 0].min()) if (deg > 0).any() else 0,
        "deg_max": int(deg.max()) if deg.size else 0,
        "deg_mean": float(deg[deg > 0].mean()) if (deg > 0).any() else 0.0,
    }
    for attr in ("edge_length", "edge_dirs"):
        if attr in es:
            a = es[attr].cpu().numpy()
            out[f"{attr}_mean"] = float(a.mean())
            out[f"{attr}_std"] = float(a.std())
            out[f"{attr}_min"] = float(a.min())
            out[f"{attr}_max"] = float(a.max())
    return out


def main(p_old, p_new):
    ga, gb = load(p_old), load(p_new)
    print(f"OLD: {p_old}\nNEW: {p_new}\n")

    print("=== NODES ===")
    for name in sorted(set(ga.node_types) | set(gb.node_types)):
        na, ea = node_summary(ga, name) if name in ga.node_types else (None, {})
        nb, eb = node_summary(gb, name) if name in gb.node_types else (None, {})
        print(f"  {name}: old={na}  new={nb}  {'SAME' if na == nb else '*** DIFF ***'}")
        for k in sorted(set(ea) | set(eb)):
            va, vb = ea.get(k), eb.get(k)
            if k in ("cutout", "boundary") and va is not None and vb is not None:
                print(f"     {k}: old sum={int(va.sum())}  new sum={int(vb.sum())}")
            elif k == "indices_connected_nodes" and va is not None and vb is not None:
                same = va.shape == vb.shape and np.array_equal(va, vb)
                print(f"     {k}: old len={va.size} max={va.max()}  new len={vb.size} max={vb.max()}  "
                      f"{'identical' if same else '*** DIFFERENT ***'}")

    print("\n=== EDGES ===")
    rels = sorted(set(ga.edge_types) | set(gb.edge_types))
    for rel in rels:
        sa, sb = edge_stats(ga, rel), edge_stats(gb, rel)
        print(f"  {rel}:")
        keys = sorted(set(sa or {}) | set(sb or {}))
        for k in keys:
            va = sa.get(k) if sa else None
            vb = sb.get(k) if sb else None
            flag = ""
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and vb:
                if abs(va - vb) / (abs(vb) + 1e-9) > 0.02:
                    flag = "  *** DIFF >2% ***"
            print(f"     {k:24s} old={va}  new={vb}{flag}")

    print("\nInterpretation: large edge-count / degree / edge_dirs differences => the 0.9.4 build")
    print("is geometrically different from the 0.8.1 build the 1.5 m/s model used.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python compare_graphs.py OLD_GRAPH.pt NEW_GRAPH.pt")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
