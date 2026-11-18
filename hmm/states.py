"""Part A, step 2: HMM observations and hidden states for one trip.

Observations  = the GPS points, in time order (t = 0 .. n-1).
Hidden states = at point t, the candidate arcs near that point (the roads the car
                could be on). Different points have different numbers of states.

build_states(d, cands) groups the candidate table by GPS point, so the decoder can
ask "what are the candidates of point t?" in O(1). It keeps, for every point:
  rows[t]   : row numbers in `cands` (used later to look up classifier probabilities)
  arcs[t]   : candidate arc_ids
  dist[t]   : GPS-to-arc distance in metres   (for the Gaussian emission)
  offset[t] : distance along the arc from its start node (for transitions)
Uses only GPS + road network (no labels), so it is safe to call inside the API.

Usage (quick check): python hmm/states.py "<Student folder>" [split/trip_id ...]
"""
import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from load_data import load_trip
from geo import to_metres, trip_origin
from candidates import generate_candidates


def build_states(d, cands):
    """Per-point candidate lists for one trip. cands = generate_candidates(d, ...)."""
    tr = d["track"]
    n = len(tr)
    t = tr["t"].to_numpy(dtype=float)
    if np.any(np.diff(t) < 0):
        raise ValueError(f"trip {d['trip_id']}: GPS timestamps are not in time order")

    # GPS positions in the same metres as the arcs (needed for straight-line distances).
    x, y = to_metres(tr["lon"], tr["lat"], trip_origin(d))

    # generate_candidates sorts by obs_idx, so each point's candidates are one block.
    cands = cands.reset_index(drop=True)
    obs = cands["obs_idx"].to_numpy()
    if np.any(np.diff(obs) < 0):
        raise ValueError("candidates must be sorted by obs_idx")
    start = np.searchsorted(obs, np.arange(n), side="left")
    end = np.searchsorted(obs, np.arange(n), side="right")

    arc_id = cands["arc_id"].to_numpy()
    dist = cands["dist_m"].to_numpy()
    offset = cands["offset_m"].to_numpy()

    rows = [np.arange(s, e) for s, e in zip(start, end)]
    return {
        "trip_id": d["trip_id"],
        "n_points": n,
        "t": t,                      # timestamp of each GPS point (s)
        "x": x, "y": y,              # GPS position in metres
        "n_cands": end - start,      # number of hidden states at each point
        "rows": rows,
        "arcs": [arc_id[r] for r in rows],
        "dist": [dist[r] for r in rows],
        "offset": [offset[r] for r in rows],
    }


def describe(trip_folder, radius_m=50.0, max_k=16):
    """Print how many states each point has and what the time steps look like."""
    d = load_trip(trip_folder)
    cands = generate_candidates(d, radius_m, max_k)
    s = build_states(d, cands)
    k = s["n_cands"]
    dt = np.diff(s["t"])

    # Sanity checks: the per-point lists must give back exactly the candidate table.
    assert k.sum() == len(cands)
    assert all(len(np.unique(a)) == len(a) for a in s["arcs"])  # no arc twice at a point
    i = int(np.argmax(k))
    assert np.array_equal(s["arcs"][i], cands.loc[cands.obs_idx == i, "arc_id"].to_numpy())

    print(f"\n=== trip {s['trip_id']} ===")
    print(f"GPS points (observations)   : {s['n_points']:,}")
    print(f"candidate rows (all states) : {len(cands):,}")
    print(f"states per point            : min {k.min()}  median {np.median(k):.0f}  "
          f"mean {k.mean():.1f}  max {k.max()}")
    print(f"points with 0 candidates    : {(k == 0).sum()}")
    print(f"points at the max_k cap     : {(k == max_k).sum()}")
    print(f"time step dt (s)            : min {dt.min():.0f}  median {np.median(dt):.0f}  "
          f"max {dt.max():.0f}")
    print(f"steps with dt == 0          : {(dt == 0).sum()}")
    print(f"steps with dt > 60 s        : {(dt > 60).sum()}")
    print(f"example, point {i}: arcs {s['arcs'][i][:5].tolist()} ...  "
          f"dist_m {np.round(s['dist'][i][:5], 1).tolist()} ...")


if __name__ == "__main__":
    root = sys.argv[1]  # path to "Student folder"
    trips = sys.argv[2:] or ["train/00000046", "train/00000013", "train/00000090"]
    for tp in trips:
        describe(os.path.join(root, *tp.split("/")))
