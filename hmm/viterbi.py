"""Part A, step 5: Viterbi decoding, i.e. the most likely sequence of arcs for a trip.

In log space, for every point t and every candidate j of t:
    delta_t(j) = max_i [ delta_{t-1}(i) + log_t(i, j) ]  +  w * log_e_t(j)
    back_t(j)  = the i that gave that max
delta_t(j) is the score of the best path that ends in candidate j at point t.
At the end of a chain we take the best final candidate and follow the back pointers
backwards to recover the whole path.

A chain has to stop (and a new one start) when Viterbi cannot continue:
  - the time gap to the next point is over gap_s seconds
  - the next point has no candidates at all
  - every move from t-1 to t is impossible (all log_t = -inf)
Each finished chain is backtracked on its own. Step 6 adds what to do with the points
left unmatched; here they are just marked -1 and counted.

Usage (quick check): python hmm/viterbi.py "<Student folder>" [split/trip_id ...]
"""
import os
import sys
import json
import time
from collections import Counter
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
from transitions import ShortestPaths, log_transitions, PARAMS_PATH
from metrics import route_from_points, continuity, route_prf


def viterbi(states, log_e, g, beta, w=1.0, gap_s=60.0, sp=None, break_before=None,
            trans_kw=None, precomputed=None):
    """Best candidate index per point (-1 = unmatched) + counts of why chains broke.

    states = build_states(...), log_e = one emission array per point (step 3),
    g = build_graph(d), beta = transition scale (m), w = weight on the emissions.
    break_before (optional, step 6): one string per point; non-empty = start a new chain
    at that point for that reason. When given, it replaces the dt > gap_s check (used
    when some points were skipped, so the dt between kept points is not the real gap).
    trans_kw (optional): neg_tol / max_speed / slack for log_transitions (from hmm_config.json).
    precomputed (optional, for tuning): {t: |d_route - d_gc| matrix for the step t-1 -> t}.
    log_t is then just -matrix / beta, so trying many beta / w values needs no new Dijkstra.
    """
    n = states["n_points"]
    sp = sp or ShortestPaths(g["graph"])
    choice = np.full(n, -1)          # chosen candidate index at each point
    breaks = Counter()
    chain = []                       # (t, back pointers into t-1) for the current chain
    delta = None                     # scores of the current chain at its last point

    def finish_chain():
        """Backtrack the current chain and write its choices."""
        if not chain:
            return
        j = int(np.argmax(delta))
        for t, back in reversed(chain):
            choice[t] = j
            if back is not None:
                j = int(back[j])
        chain.clear()

    for t in range(n):
        if states["n_cands"][t] == 0:                # nothing to match this point to
            if chain:
                breaks["no_candidates"] += 1
            finish_chain()
            delta = None
            continue

        if delta is not None:
            prev = chain[-1][0]
            dt = states["t"][t] - states["t"][prev]
            if break_before is not None:
                reason = break_before[t]
            else:
                reason = "time_gap" if dt > gap_s else ""
            if reason:
                breaks[reason] += 1
            else:
                if precomputed is not None and prev == t - 1:
                    lt = -precomputed[t] / beta
                else:
                    d_gc = np.hypot(states["x"][t] - states["x"][prev],
                                    states["y"][t] - states["y"][prev])
                    lt = log_transitions(g, sp, states["arcs"][prev], states["offset"][prev],
                                         states["arcs"][t], states["offset"][t], d_gc, dt, beta,
                                         **(trans_kw or {}))
                scores = delta[:, None] + lt           # [i, j]: come from i, go to j
                back = np.argmax(scores, axis=0)       # best i for every j
                best = scores[back, np.arange(len(back))]
                if np.isfinite(best).any():
                    delta = best + w * log_e[t]
                    delta -= delta[np.isfinite(delta)].max()   # keep numbers small
                    chain.append((t, back))
                    continue
                breaks["all_moves_impossible"] += 1
            finish_chain()

        # start a new chain at point t
        delta = w * log_e[t]
        chain.append((t, None))

    finish_chain()
    return choice, breaks


def chosen_arcs(states, choice):
    """Candidate index per point -> arc_id per point (-1 stays -1)."""
    return np.array([states["arcs"][t][c] if c >= 0 else -1 for t, c in enumerate(choice)])


def nearest_arcs(states, log_e):
    """Emission-only baseline: the best-scoring candidate at each point (-1 if none)."""
    return np.array([a[np.argmax(e)] if len(a) else -1
                     for a, e in zip(states["arcs"], log_e)])


def decode(d, sigma, beta, w=1.0):
    """Nearest-arc and Viterbi arc per point for one trip, plus what's needed to score them."""
    t0 = time.perf_counter()
    g = build_graph(d)
    s = build_states(d, generate_candidates(d))
    log_e = gaussian_log_emissions(s, sigma)
    choice, breaks = viterbi(s, log_e, g, beta, w)
    return {"g": g, "states": s, "nearest": nearest_arcs(s, log_e),
            "viterbi": chosen_arcs(s, choice), "breaks": breaks,
            "secs": time.perf_counter() - t0}


def score_trip(d, res):
    """Counts for every metric (summable across trips), labels used for scoring only."""
    g, s = res["g"], res["states"]
    truth = align_route(d)["true_arc"].to_numpy()
    ref = d["route"]["arc_id"].to_numpy()
    row = {"trip_id": d["trip_id"], "points": s["n_points"],
           "no_cand_points": int((s["n_cands"] == 0).sum()),
           "chains": 1 + sum(res["breaks"].values()),
           "breaks": ";".join(f"{k}={v}" for k, v in res["breaks"].items()),
           "secs": round(res["secs"], 2)}
    for name in ("nearest", "viterbi"):
        arcs = res[name]
        route = route_from_points(arcs)
        conn, pairs = continuity(route, g["arc_from"], g["arc_to"])
        tp, n_pred, n_ref = route_prf(route, ref)
        row.update({f"{name}_correct": int((arcs == truth).sum()),   # unmatched count as wrong
                    f"{name}_conn": conn, f"{name}_pairs": pairs,
                    f"{name}_tp": tp, f"{name}_npred": n_pred, f"{name}_nref": n_ref})
    return row


def add_rates(df):
    """Turn the summed counts into percentages (works for per-trip rows and totals)."""
    for name in ("nearest", "viterbi"):
        df[f"{name}_top1"] = 100 * df[f"{name}_correct"] / df["points"]
        df[f"{name}_continuity"] = 100 * df[f"{name}_conn"] / df[f"{name}_pairs"]
        p = df[f"{name}_tp"] / df[f"{name}_npred"]
        r = df[f"{name}_tp"] / df[f"{name}_nref"]
        df[f"{name}_precision"], df[f"{name}_recall"] = 100 * p, 100 * r
        df[f"{name}_f1"] = 100 * 2 * p * r / (p + r)
    return df


SHOW = ["trip_id", "points", "no_cand_points", "chains",
        "nearest_top1", "viterbi_top1", "nearest_continuity", "viterbi_continuity",
        "nearest_precision", "viterbi_precision", "nearest_recall", "viterbi_recall",
        "nearest_f1", "viterbi_f1"]


if __name__ == "__main__":
    root = sys.argv[1]  # path to "Student folder"
    with open(PARAMS_PATH) as f:   # written by: python hmm/transitions.py "<Student folder>"
        p = json.load(f)
    sigma, beta = p["sigma_m"], p["beta_m"]
    print(f"sigma {sigma} m, beta {beta} m, w 1.0  (estimated on train)")
    pd.set_option("display.width", 250)

    # student_test: per-trip table + overall totals, saved
    tdir = os.path.join(root, "student_test")
    rows = []
    for tid in sorted(os.listdir(tdir)):
        d = load_trip(os.path.join(tdir, tid))
        rows.append(score_trip(d, decode(d, sigma, beta)))
    df = pd.DataFrame(rows)
    total = df.drop(columns=["trip_id", "breaks"]).sum().to_dict()  # pooled counts
    total.update({"trip_id": "TOTAL", "breaks": ""})
    df = add_rates(pd.concat([df, pd.DataFrame([total])], ignore_index=True))
    out = os.path.join(REPO, "outputs", "hmm", "viterbi_student_test.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.round(2).to_csv(out, index=False)
    print("\nstudent_test (TOTAL = pooled over all points / arcs):")
    print(df[SHOW].round(1).to_string(index=False))
    print(f"saved {os.path.relpath(out, REPO)}")

    # extra trips given on the command line (default: train 2, which has no-candidate points)
    for tp in sys.argv[2:] or ["train/00000002"]:
        d = load_trip(os.path.join(root, *tp.split("/")))
        res = decode(d, sigma, beta)
        r = add_rates(pd.DataFrame([score_trip(d, res)]))
        print(f"\n{tp}: breaks {dict(res['breaks'])}, "
              f"unmatched by Viterbi {(res['viterbi'] < 0).sum()}")
        print(r[SHOW].round(1).to_string(index=False))
