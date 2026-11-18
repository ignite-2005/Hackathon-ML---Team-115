"""Part A, step 4: transition scores between candidates of consecutive GPS points.

Idea (Newson & Krumm 2009): between two GPS fixes a car drives along the roads, and that
driving distance should be close to the straight-line distance between the two fixes.
So a move from (arc a, offset oa) at point t-1 to (arc b, offset ob) at point t gets

    d_route = distance driven along the roads from oa on a to ob on b
    d_gc    = straight-line distance between the two GPS points
    log_t   = -|d_route - d_gc| / beta

d_route has three cases:
    same arc (a == b)            : ob - oa         (more than NEG_TOL_M backwards = impossible)
    a ends where b starts        : (len_a - oa) + ob
    otherwise                    : (len_a - oa) + SP(end of a, start of b) + ob
The second case is the third with SP = 0, so the code handles both the same way.
SP = shortest road distance (Dijkstra). If there is no path within the cutoff
(dt * MAX_SPEED + SLACK metres), the move is impossible: log_t = -inf.

Usage (quick check): python hmm/transitions.py "<Student folder>" [split/trip_id ...]
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)
from load_data import load_trip
from candidates import generate_candidates, MAX_MEDIAN_TRUE_DIST_M
from graph import build_graph
from states import build_states
from emissions import estimate_sigma

PARAMS_PATH = os.path.join(REPO, "outputs", "hmm", "estimated_params.json")

MAX_SPEED_MS = 50.0   # fastest believable speed (180 km/h)
SLACK_M = 100.0       # extra metres on the cutoff for GPS noise and short dt
# How far "backwards" along the same arc we still accept as GPS noise. Was 10 m (the
# brief's value); on train, 1,746 true moves jumped back more than 10 m, 60% of them GPS
# jitter (see check_backward.py). At 30 m (~4 sigma) only 8 remain. The |d_route - d_gc|
# penalty still punishes going backwards, so the decoder won't drive arcs the wrong way.
NEG_TOL_M = 30.0


class ShortestPaths:
    """Shortest road distances from a node, with a cache.

    Consecutive GPS points are usually on the same few roads, so the same start nodes
    come up again and again. Each Dijkstra run is stored as {reachable node: metres}.
    """

    def __init__(self, graph):
        self.graph = graph
        self.cache = {}      # node -> (limit used, {node: distance})
        self.n_runs = 0

    def from_node(self, u, limit):
        hit = self.cache.get(u)
        if hit is None or hit[0] < limit:   # not cached, or cached with a smaller cutoff
            dist = dijkstra(self.graph, directed=True, indices=u, limit=limit)
            reach = np.flatnonzero(np.isfinite(dist))   # nodes reached within the limit
            hit = (limit, dict(zip(reach.tolist(), dist[reach].tolist())))
            self.cache[u] = hit
            self.n_runs += 1
        return hit[1]


def log_transitions(g, sp, arcs_a, off_a, arcs_b, off_b, d_gc, dt, beta,
                    neg_tol=NEG_TOL_M, max_speed=MAX_SPEED_MS, slack=SLACK_M):
    """Matrix log_t[i, j] for moving from candidate i (point t-1) to candidate j (point t).

    g = build_graph(d), sp = ShortestPaths(g["graph"]).
    """
    cutoff = dt * max_speed + slack
    targets = g["arc_from"][arcs_b]                  # start node of every candidate at t

    # SP[i, j] = road distance from the end of arc a_i to the start of arc b_j
    SP = np.full((len(arcs_a), len(arcs_b)), np.inf)
    for i, u in enumerate(g["arc_to"][arcs_a]):
        reach = sp.from_node(u, cutoff)
        SP[i] = [reach.get(v, np.inf) for v in targets]
    SP[SP > cutoff] = np.inf                         # cache may hold a longer search

    rest_of_a = g["arc_len"][arcs_a] - off_a         # metres left on arc a after oa
    d_route = rest_of_a[:, None] + SP + off_b[None, :]

    # same arc: just the distance moved along it (small negative = GPS noise)
    same = arcs_a[:, None] == arcs_b[None, :]
    along = off_b[None, :] - off_a[:, None]
    d_route = np.where(same, np.where(along >= -neg_tol, along, np.inf), d_route)

    return -np.abs(d_route - d_gc) / beta            # inf -> -inf (impossible move)


def true_transitions(d, labels, gap_s=60, neg_tol=NEG_TOL_M):
    """One row per step between the TRUE arcs of consecutive points (labels: tuning/eval only).

    gap = |d_route - d_gc| (log_t with beta = 1). When the true move comes out impossible
    (gap = inf), `reason` says why:
      same_arc_backward : same arc, but ob is more than neg_tol metres behind oa
      no_path_in_cutoff : different arcs, no road path within dt * MAX_SPEED + SLACK
      other             : anything else (should never happen; kept as a check)
    Steps over gap_s seconds, and steps where a true arc is not a candidate, are skipped.
    """
    g = build_graph(d)
    sp = ShortestPaths(g["graph"])
    s = build_states(d, generate_candidates(d))
    truth = labels["true_arc"].to_numpy()
    rows = []
    for t in range(1, s["n_points"]):
        dt = s["t"][t] - s["t"][t - 1]
        if dt > gap_s:
            continue
        ia = np.flatnonzero(s["arcs"][t - 1] == truth[t - 1])
        ib = np.flatnonzero(s["arcs"][t] == truth[t])
        if len(ia) == 0 or len(ib) == 0:
            continue  # true arc not among the candidates
        a, b = truth[t - 1], truth[t]
        oa, ob = s["offset"][t - 1][ia[0]], s["offset"][t][ib[0]]
        d_gc = np.hypot(s["x"][t] - s["x"][t - 1], s["y"][t] - s["y"][t - 1])
        gap = -log_transitions(g, sp, np.array([a]), np.array([oa]),
                               np.array([b]), np.array([ob]), d_gc, dt, beta=1.0,
                               neg_tol=neg_tol)[0, 0]

        reason, sp_any = "", np.nan
        if not np.isfinite(gap):
            if a == b and ob - oa < -neg_tol:
                reason = "same_arc_backward"
            elif a != b:
                reason = "no_path_in_cutoff"
                # is there a path at all, ignoring the cutoff? (separate run, not cached)
                sp_any = dijkstra(g["graph"], indices=g["arc_to"][a])[g["arc_from"][b]]
            else:
                reason = "other"
        rows.append({"trip_id": d["trip_id"], "obs_idx": t, "dt": dt, "d_gc": d_gc,
                     "arc_a": a, "arc_b": b, "oa": oa, "ob": ob,
                     "len_a": g["arc_len"][a], "gap": gap, "reason": reason,
                     "sp_no_cutoff": sp_any})
    return pd.DataFrame(rows)


def estimate_beta(tt):
    """beta = median |d_route - d_gc| over the possible TRUE transitions / ln 2
    (Newson & Krumm 2009). tt = true_transitions() rows from TRAIN trips only."""
    finite = tt.loc[np.isfinite(tt["gap"]), "gap"]
    return float(finite.median() / np.log(2))


def train_true_transitions(root, labels, neg_tol=NEG_TOL_M):
    """true_transitions() for every TRAIN trip whose labels are reliable."""
    out = []
    for tid, lab in labels.groupby("trip_id"):
        if lab["true_dist_m"].median() > MAX_MEDIAN_TRUE_DIST_M:
            continue
        out.append(true_transitions(load_trip(os.path.join(root, "train", tid)),
                                    lab.sort_values("obs_idx"), neg_tol=neg_tol))
    return pd.concat(out, ignore_index=True)


def report_impossible(tt, n_examples=5):
    """Print why TRUE transitions come out impossible, with examples."""
    bad = tt[~np.isfinite(tt["gap"])]
    print(f"\nimpossible TRUE transitions: {len(bad):,} of {len(tt):,} "
          f"({100 * len(bad) / len(tt):.2f}%)")
    print(bad["reason"].value_counts().to_string())

    cols = ["trip_id", "obs_idx", "dt", "d_gc", "oa", "ob", "len_a"]
    pd.set_option("display.width", 200)
    for reason, grp in bad.groupby("reason"):
        print(f"\n--- {reason}: examples ---")
        print(grp[cols + (["sp_no_cutoff"] if reason == "no_path_in_cutoff" else [])]
              .sample(min(n_examples, len(grp)), random_state=0).round(1).to_string(index=False))

    back = bad[bad["reason"] == "same_arc_backward"]
    if len(back):
        moved_back = back["oa"] - back["ob"]
        print(f"\nsame_arc_backward: backward jump (oa - ob) m: median {moved_back.median():.1f}, "
              f"p90 {moved_back.quantile(0.9):.1f}, max {moved_back.max():.1f}")
        print(f"  d_gc m: median {back['d_gc'].median():.1f}; "
              f"points almost still (d_gc < 2 m): {100 * (back['d_gc'] < 2).mean():.0f}%")
        for tol in (20, 30, 50):
            print(f"  still impossible if NEG_TOL_M = {tol} m: {(moved_back > tol).sum()}")
        print("  trips with most:", back["trip_id"].value_counts().head(5).to_dict())
    nopath = bad[bad["reason"] == "no_path_in_cutoff"]
    if len(nopath):
        print(f"\nno_path_in_cutoff: no path at all {np.isinf(nopath['sp_no_cutoff']).sum()}, "
              f"path exists but beyond cutoff {np.isfinite(nopath['sp_no_cutoff']).sum()}")


def describe(trip_folder, beta, gap_s=60):
    """How many moves are possible, how often a whole step is impossible, and speed."""
    d = load_trip(trip_folder)
    t0 = time.perf_counter()
    g = build_graph(d)
    sp = ShortestPaths(g["graph"])
    s = build_states(d, generate_candidates(d))

    n_pairs = n_finite = n_dead = n_steps = n_gap = n_empty = 0
    for t in range(1, s["n_points"]):
        dt = s["t"][t] - s["t"][t - 1]
        if dt > gap_s:
            n_gap += 1
            continue
        if s["n_cands"][t - 1] == 0 or s["n_cands"][t] == 0:
            n_empty += 1
            continue
        d_gc = np.hypot(s["x"][t] - s["x"][t - 1], s["y"][t] - s["y"][t - 1])
        lt = log_transitions(g, sp, s["arcs"][t - 1], s["offset"][t - 1],
                             s["arcs"][t], s["offset"][t], d_gc, dt, beta)
        n_steps += 1
        n_pairs += lt.size
        n_finite += np.isfinite(lt).sum()
        n_dead += not np.isfinite(lt).any()
    secs = time.perf_counter() - t0

    print(f"\n=== trip {s['trip_id']} ===  (beta = {beta:.2f} m, {secs:.1f} s)")
    print(f"steps scored                 : {n_steps:,}  (skipped: {n_gap} gaps > {gap_s} s, "
          f"{n_empty} next to a point with no candidates)")
    print(f"candidate pairs scored       : {n_pairs:,}")
    print(f"possible moves (finite log_t): {100 * n_finite / n_pairs:.1f}%")
    print(f"steps where EVERY move is -inf (chain must break): {n_dead}")
    print(f"Dijkstra runs / cached nodes : {sp.n_runs:,} / {len(sp.cache):,}")


if __name__ == "__main__":
    root = sys.argv[1]  # path to "Student folder"
    labels = pd.read_parquet(os.path.join(REPO, "outputs", "candidates", "point_labels_train.parquet"))
    t0 = time.perf_counter()
    tt = train_true_transitions(root, labels)            # train only
    beta = estimate_beta(tt)
    print(f"beta estimated on train true transitions: {beta:.2f} m "
          f"({len(tt):,} true transitions; {time.perf_counter() - t0:.0f} s)")
    report_impossible(tt)

    # Save the train-estimated starting values so later steps can read them.
    sigma = estimate_sigma(labels)
    os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)
    with open(PARAMS_PATH, "w") as f:
        json.dump({"sigma_m": round(sigma, 3), "beta_m": round(beta, 3),
                   "neg_tol_m": NEG_TOL_M, "max_speed_ms": MAX_SPEED_MS, "slack_m": SLACK_M,
                   "estimated_on": "train (label_quality_ok trips only)",
                   "n_true_transitions": int(len(tt))}, f, indent=2)
    print(f"\nsaved {os.path.relpath(PARAMS_PATH, REPO)}: sigma {sigma:.2f} m, beta {beta:.2f} m")
    trips = sys.argv[2:] or ["train/00000046", "train/00000013", "student_test/00000056"]
    for tp in trips:
        describe(os.path.join(root, *tp.split("/")), beta)
