"""Steps 3-4: candidate generation and per-point labels.

generate_candidates() uses only GPS + road network, so the API can call it on any
trip. add_labels() uses the reference route and is for training/evaluation only:
never call it inside the prediction pipeline.

Usage: python candidates.py "<Student folder>" <out_dir> [radius_m] [max_k]
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import load_trip
from geo import to_metres, project_onto_segments


def arc_geometry(d):
    """Arc endpoints in metres: ax, ay, bx, by arrays indexed by arc_id."""
    nx_, ny_ = to_metres(d["nodes"]["lon"], d["nodes"]["lat"])
    a = d["arcs"]["from_node"].values
    b = d["arcs"]["to_node"].values
    return nx_[a], ny_[a], nx_[b], ny_[b]


def generate_candidates(d, radius_m=50.0, max_k=16):
    """All arcs within radius_m of each GPS point, keeping the max_k nearest.

    Returns one row per (GPS point, candidate arc) with the agreed columns:
    trip_id, obs_idx, timestamp, x, y, arc_id, dist_m, offset_m, arc_len_m
    """
    tr = d["track"]
    px, py = to_metres(tr["lon"], tr["lat"])
    ax, ay, bx, by = arc_geometry(d)

    # Spatial index over arc lines, so we don't compare every point with every arc
    lines = shapely.linestrings(np.stack([np.c_[ax, ay], np.c_[bx, by]], axis=1))
    tree = shapely.STRtree(lines)
    pts = shapely.points(px, py)
    obs_idx, arc_id = tree.query(pts, predicate="dwithin", distance=radius_m)

    dist, offset, seg_len = project_onto_segments(
        px[obs_idx], py[obs_idx], ax[arc_id], ay[arc_id], bx[arc_id], by[arc_id])

    c = pd.DataFrame({
        "trip_id": d["trip_id"],
        "obs_idx": obs_idx,
        "timestamp": tr["t"].values[obs_idx],
        "x": px[obs_idx],
        "y": py[obs_idx],
        "arc_id": arc_id,
        "dist_m": dist,
        "offset_m": offset,
        "arc_len_m": seg_len,
    })
    # keep the max_k nearest per point (two-way roads give 2 arcs per road)
    c = c.sort_values(["obs_idx", "dist_m", "arc_id"])
    c = c[c.groupby("obs_idx").cumcount() < max_k].reset_index(drop=True)
    return c


def align_route(d):
    """Give every GPS point a reference arc from the route.

    The route is an ordered list of arcs, shorter than the GPS track. We pick, for
    each point, a position in the route so that positions never go backwards in
    time and the total point-to-arc distance is as small as possible
    (dynamic programming, like a simple DTW alignment).
    Returns true_arc and true_dist_m per GPS point.
    """
    tr = d["track"]
    px, py = to_metres(tr["lon"], tr["lat"])
    ax, ay, bx, by = arc_geometry(d)
    r = d["route"]["arc_id"].values
    rax, ray, rbx, rby = ax[r], ay[r], bx[r], by[r]

    n, m = len(px), len(r)
    back = np.empty((n, m), dtype=np.int32)  # best previous route position
    dp = None
    for i in range(n):
        cost, _, _ = project_onto_segments(np.full(m, px[i]), np.full(m, py[i]),
                                           rax, ray, rbx, rby)
        if dp is None:
            dp = cost
            back[i] = np.arange(m)
            continue
        # best total cost of reaching any position <= j at the previous point
        best_idx = np.zeros(m, dtype=np.int32)
        run_min = np.minimum.accumulate(dp)
        changed = np.r_[True, run_min[1:] < run_min[:-1]]
        best_idx = np.maximum.accumulate(np.where(changed, np.arange(m), 0))
        dp = cost + run_min
        back[i] = best_idx

    # trace back the best path
    pos = np.empty(n, dtype=np.int32)
    pos[-1] = int(np.argmin(dp))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    true_arc = r[pos]
    true_dist, _, _ = project_onto_segments(px, py, rax[pos], ray[pos], rbx[pos], rby[pos])
    return pd.DataFrame({"trip_id": d["trip_id"], "obs_idx": np.arange(n),
                         "route_pos": pos, "true_arc": true_arc, "true_dist_m": true_dist})


def add_labels(cands, truth):
    """label = 1 where the candidate arc is the point's reference arc."""
    c = cands.merge(truth[["obs_idx", "true_arc"]], on="obs_idx", how="left")
    c["label"] = (c["arc_id"] == c["true_arc"]).astype(np.int8)
    return c.drop(columns="true_arc")


def miss_rate(cands, truth):
    """% of GPS points whose reference arc is not among their candidates."""
    hit = cands.loc[cands["label"] == 1, "obs_idx"].unique()
    return 100.0 * (1 - len(hit) / len(truth))


if __name__ == "__main__":
    root, out = sys.argv[1], sys.argv[2]
    radius = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
    max_k = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    os.makedirs(out, exist_ok=True)

    for part in ["train", "student_test"]:
        pdir = os.path.join(root, part)
        all_c, all_t, rows = [], [], []
        for tid in sorted(os.listdir(pdir)):
            t0 = time.time()
            d = load_trip(os.path.join(pdir, tid))
            truth = align_route(d)
            c = add_labels(generate_candidates(d, radius, max_k), truth)
            rows.append({"trip_id": tid, "n_gps": len(truth), "n_cand_rows": len(c),
                         "cands_per_point": round(len(c) / len(truth), 1),
                         "miss_rate_pct": round(miss_rate(c, truth), 2),
                         "median_true_dist_m": round(truth["true_dist_m"].median(), 1),
                         "seconds": round(time.time() - t0, 1)})
            all_c.append(c)
            all_t.append(truth)
            print(part, rows[-1], flush=True)
        pd.concat(all_c).to_parquet(os.path.join(out, f"candidates_{part}.parquet"), index=False)
        pd.concat(all_t).to_parquet(os.path.join(out, f"point_labels_{part}.parquet"), index=False)
        pd.DataFrame(rows).to_csv(os.path.join(out, f"candidate_summary_{part}.csv"), index=False)
