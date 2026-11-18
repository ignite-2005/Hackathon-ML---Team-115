"""Part A, step 6: breaks and edge cases around Viterbi.

decode_points(d, ...) = the full per-point decoder:
  1. Skip stationary points (optional, skip_k > 0): walk through the track and keep a
     point only if it is at least skip_k * sigma metres from the last KEPT point.
     When the car is stopped the GPS jumps around (~7 m/s on trip 77) without telling us
     the direction, so decoding every jittery fix makes Viterbi flip between the two
     directions of a road. Points with no candidates are never kept (nothing to decode).
  2. Chain breaks: Viterbi runs on the kept points only. A new chain starts when the REAL
     time gap between two kept points (largest dt of the original points in between) is
     over gap_s, or a point with no candidates lies between them, or every move is
     impossible. (The dt between kept points can be long just because we skipped a stop,
     so it is not used for the gap rule.)
  3. Skipped points: get whichever decoded arc around them is closer to their GPS fix:
     the arc of the previous kept point or of the next kept point (if it is among their
     candidates), within the same chain.
  4. Points with no candidates stay unmatched (-1), or, if fill_s > 0, take the arc of the
     nearest matched point in time when that point is at most fill_s seconds away.
Everything that happened is counted in `stats` for the report.

Usage (quick check): python hmm/edge_cases.py "<Student folder>" [split/trip_id ...]
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)
from load_data import load_trip
from candidates import generate_candidates, align_route
from graph import build_graph
from states import build_states
from emissions import gaussian_log_emissions
from transitions import PARAMS_PATH
from viterbi import viterbi, nearest_arcs
from metrics import route_from_points, continuity, route_prf, stopped_mask


def select_points(states, min_move_m):
    """Indices of the points to decode (always the first decodable point of a stretch)."""
    has = states["n_cands"] > 0
    keep, last = [], None
    for t in range(states["n_points"]):
        if not has[t]:
            last = None            # after a point with no candidates, keep the next one
            continue
        if last is None or min_move_m <= 0 or np.hypot(
                states["x"][t] - states["x"][last], states["y"][t] - states["y"][last]) >= min_move_m:
            keep.append(t)
            last = t
    return np.array(keep, dtype=int)


def subset(states, log_e, idx):
    """States and emissions for only the points in idx (same structure as build_states)."""
    sub = {"trip_id": states["trip_id"], "n_points": len(idx),
           "t": states["t"][idx], "x": states["x"][idx], "y": states["y"][idx],
           "n_cands": states["n_cands"][idx]}
    for key in ("rows", "arcs", "dist", "offset"):
        sub[key] = [states[key][i] for i in idx]
    return sub, [log_e[i] for i in idx]


def break_reasons(states, idx, gap_s):
    """For each kept point: why a new chain must start there ('' = no reason)."""
    t = states["t"]
    no_cand = np.r_[0, np.cumsum(states["n_cands"] == 0)]  # no-cand points before index i
    out = [""]
    for a, b in zip(idx[:-1], idx[1:]):
        if no_cand[b] - no_cand[a + 1] > 0:
            out.append("no_candidates")
        elif np.max(np.diff(t[a:b + 1])) > gap_s:
            out.append("time_gap")
        else:
            out.append("")
    return out


def decode_points(d, sigma, beta, w=1.0, gap_s=60.0, skip_k=0.0, fill_s=0.0,
                  cands=None, log_e=None, g=None, trans_kw=None):
    """Arc per GPS point (-1 = unmatched) and counts of every edge case.

    log_e: emissions per point (default: Gaussian with sigma). Pass classifier
    emissions here later. cands / g can be passed in to avoid rebuilding them.
    """
    t0 = time.perf_counter()
    g = g or build_graph(d)
    cands = generate_candidates(d) if cands is None else cands
    s = build_states(d, cands)
    log_e = log_e or gaussian_log_emissions(s, sigma)
    n = s["n_points"]

    # 1-2. decode the kept points
    idx = select_points(s, skip_k * sigma)
    sub, sub_e = subset(s, log_e, idx)
    reasons = break_reasons(s, idx, gap_s)
    choice, breaks = viterbi(sub, sub_e, g, beta, w, gap_s, break_before=reasons,
                             trans_kw=trans_kw)

    arcs = np.full(n, -1)
    chain_id = np.full(n, -1)
    kept_arc = np.array([sub["arcs"][k][c] if c >= 0 else -1 for k, c in enumerate(choice)])
    arcs[idx] = kept_arc
    new_chain = np.r_[True, [r != "" for r in reasons[1:]]] if len(idx) else np.array([], bool)
    chain_of_kept = np.cumsum(new_chain) - 1
    chain_id[idx] = chain_of_kept

    # 3. skipped points between two kept points: closer of the two neighbouring arcs
    n_skipped = 0
    for k in range(len(idx) - 1):
        a, b = idx[k], idx[k + 1]
        same_chain = chain_of_kept[k] == chain_of_kept[k + 1]
        for t in range(a + 1, b):
            if s["n_cands"][t] == 0:
                continue
            n_skipped += 1
            options = [kept_arc[k]] + ([kept_arc[k + 1]] if same_chain else [])
            best, best_dist = options[0], np.inf
            for arc in options:
                hit = np.flatnonzero(s["arcs"][t] == arc)
                if len(hit) and s["dist"][t][hit[0]] < best_dist:
                    best, best_dist = arc, s["dist"][t][hit[0]]
            arcs[t] = best
    # skipped points after the last kept point (end of the trip)
    if len(idx):
        for t in range(idx[-1] + 1, n):
            if s["n_cands"][t] > 0:
                arcs[t] = kept_arc[-1]
                n_skipped += 1

    # 4. points with no candidates
    no_cand = np.flatnonzero(s["n_cands"] == 0)
    n_filled = 0
    if fill_s > 0 and len(no_cand):
        matched = np.flatnonzero(arcs >= 0)
        if len(matched):
            for t in no_cand:
                j = np.searchsorted(matched, t)
                near = [m for m in (matched[j - 1] if j > 0 else None,
                                    matched[j] if j < len(matched) else None) if m is not None]
                m = min(near, key=lambda m: abs(s["t"][m] - s["t"][t]))
                if abs(s["t"][m] - s["t"][t]) <= fill_s:
                    arcs[t] = arcs[m]
                    n_filled += 1

    stats = {"points": n, "kept": len(idx), "skipped": n_skipped,
             "no_cand_points": len(no_cand), "no_cand_filled": n_filled,
             "unmatched": int((arcs < 0).sum()),
             "chains": int(new_chain.sum() + breaks.get("all_moves_impossible", 0)),
             **{f"break_{k}": v for k, v in breaks.items()},
             "secs": round(time.perf_counter() - t0, 2)}
    return {"arcs": arcs, "states": s, "log_e": log_e, "g": g, "stats": stats}


def score(d, arcs, g, x, y):
    """Point and route metrics for one arc-per-point prediction (labels for scoring only)."""
    truth = align_route(d)["true_arc"].to_numpy()
    still = stopped_mask(x, y)
    ok = arcs == truth
    route = route_from_points(arcs)
    conn, pairs = continuity(route, g["arc_from"], g["arc_to"])
    tp, n_pred, n_ref = route_prf(route, d["route"]["arc_id"].to_numpy())
    return {"correct": int(ok.sum()), "n": len(ok),
            "correct_stopped": int(ok[still].sum()), "n_stopped": int(still.sum()),
            "correct_moving": int(ok[~still].sum()), "n_moving": int((~still).sum()),
            "conn": conn, "pairs": pairs, "tp": tp, "npred": n_pred, "nref": n_ref}


def rates(c):
    """Counts (one trip or summed) -> percentages."""
    p, r = c["tp"] / c["npred"], c["tp"] / c["nref"]
    return {"top1": 100 * c["correct"] / c["n"],
            "top1_stopped": 100 * c["correct_stopped"] / c["n_stopped"] if c["n_stopped"] else np.nan,
            "top1_moving": 100 * c["correct_moving"] / c["n_moving"],
            "continuity": 100 * c["conn"] / c["pairs"],
            "precision": 100 * p, "recall": 100 * r, "f1": 100 * 2 * p * r / (p + r)}


CONFIGS = {  # name: (skip_k, fill_s)
    "nearest-arc": None,
    "viterbi": (0.0, 0.0),
    "viterbi+skip2sigma": (2.0, 0.0),
    "viterbi+skip3sigma": (3.0, 0.0),
}

if __name__ == "__main__":
    root = sys.argv[1]  # path to "Student folder"
    with open(PARAMS_PATH) as f:
        p = json.load(f)
    sigma, beta = p["sigma_m"], p["beta_m"]
    print(f"sigma {sigma} m, beta {beta} m, w 1.0, gap 60 s")
    pd.set_option("display.width", 250)

    tdir = os.path.join(root, "student_test")
    rows = []
    for tid in sorted(os.listdir(tdir)):
        d = load_trip(os.path.join(tdir, tid))
        cands, g = generate_candidates(d), build_graph(d)
        for name, cfg in CONFIGS.items():
            if cfg is None:
                s = build_states(d, cands)
                arcs, st = nearest_arcs(s, gaussian_log_emissions(s, sigma)), {}
            else:
                res = decode_points(d, sigma, beta, skip_k=cfg[0], fill_s=cfg[1], cands=cands, g=g)
                arcs, s, st = res["arcs"], res["states"], res["stats"]
            rows.append({"trip_id": tid, "method": name, **st, **score(d, arcs, g, s["x"], s["y"])})
    df = pd.DataFrame(rows)

    count_cols = ["correct", "n", "correct_stopped", "n_stopped", "correct_moving", "n_moving",
                  "conn", "pairs", "tp", "npred", "nref", "kept", "skipped", "secs"]
    tot = df.groupby("method", sort=False)[count_cols].sum()
    summary = pd.DataFrame({m: rates(tot.loc[m]) for m in tot.index}).T
    summary["kept_%"] = 100 * tot["kept"] / tot["n"]
    summary["secs"] = tot["secs"]
    print("\nstudent_test, all 10 trips pooled:")
    print(summary.round(1).to_string())

    per = df.join(pd.DataFrame([rates(r) for _, r in df.iterrows()]))
    print("\nTop-1 per trip:")
    print(per.pivot(index="trip_id", columns="method", values="top1")[list(CONFIGS)].round(1).to_string())
    out = os.path.join(REPO, "outputs", "hmm", "edge_cases_student_test.csv")
    per.round(3).to_csv(out, index=False)
    print(f"saved {os.path.relpath(out, REPO)}")

    # no-candidate case: train trip 2, unmatched left as -1 vs filled from neighbours
    for tp in sys.argv[2:] or ["train/00000002"]:
        d = load_trip(os.path.join(root, *tp.split("/")))
        cands, g = generate_candidates(d), build_graph(d)
        print(f"\n{tp}:")
        for fill_s in (0.0, 5.0, 30.0):
            res = decode_points(d, sigma, beta, fill_s=fill_s, cands=cands, g=g)
            r = rates(score(d, res["arcs"], g, res["states"]["x"], res["states"]["y"]))
            print(f"  fill_s={fill_s:>4}: {res['stats']}\n            Top-1 {r['top1']:.1f}%  "
                  f"continuity {r['continuity']:.1f}%  route P/R/F1 "
                  f"{r['precision']:.1f}/{r['recall']:.1f}/{r['f1']:.1f}")
