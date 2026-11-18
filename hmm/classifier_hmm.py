"""Classifier emissions + Viterbi: probabilities, decoder tuning and the final comparison.

1. probas  : Random-forest probability for every candidate.
             train = OUT-OF-FOLD (Person B's 5 GroupKFold folds: each trip is scored by a
                     model that never saw it), so decoder tuning on train is not optimistic.
             test  = model fitted on all 80 train trips.
2. tune    : grid over beta (transition scale), w (emission weight) and the gap threshold,
             for both emission types, on TRAIN only (trips 33 and 54 left out: their labels
             are unreliable). Picked by point Top-1.
3. compare : on student_test, per trip: nearest arc, classifier only (argmax p),
             Gaussian + Viterbi, classifier + Viterbi; Top-1 and route P/R/F1 / continuity.

Transitions are precomputed once per trip as |d_route - d_gc| (they do not depend on
beta, w or the emissions), so each grid point only re-runs Viterbi.

Usage (repo root):
  python hmm/classifier_hmm.py probas  "<Student folder>" --params untuned|tuned
  python hmm/classifier_hmm.py tune    "<Student folder>" --params tuned
  python hmm/classifier_hmm.py compare "<Student folder>" --params tuned
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tuning"))
from src.candidate_features import FEATURE_COLUMNS
from src.train_models import make_folds, read_partition, score_candidates
from load_data import load_trip
from candidates import generate_candidates, align_route
from graph import build_graph
from states import build_states
from emissions import gaussian_log_emissions, classifier_log_emissions
from transitions import ShortestPaths, log_transitions
from viterbi import viterbi, chosen_arcs
from route import build_route
from metrics import continuity, route_prf

DATA = REPO / "data" / "hmm"
OUT = REPO / "outputs" / "hmm"
BAD_LABEL_TRIPS = {"00000033", "00000054"}
FEATURE_OPTIONS = {"radius_m": 50.0, "max_k": 16, "max_heading_gap_s": 5.0}
MAX_GAP_S = 120.0          # largest gap threshold tried; transitions precomputed up to this


def rf_params(which):
    """Person B's untuned Random forest, or the Optuna best (models/best_params.json)."""
    from hpo import UNTUNED
    if which == "untuned":
        return dict(UNTUNED)
    return json.loads((REPO / "models" / "best_params.json").read_text())["params"]


# ---------------------------------------------------------------- 1. probabilities
def make_probas(root, which):
    from hpo import make_model
    params = rf_params(which)
    train, truth, _ = read_partition(root, "train", FEATURE_OPTIONS)
    test, _, _ = read_partition(root, "student_test", FEATURE_OPTIONS)
    oof = np.full(len(train), np.nan)
    for k, (fit_mask, val_mask, _) in enumerate(make_folds(train, truth, 5)):
        model = make_model(params)
        model.fit(train.loc[fit_mask, FEATURE_COLUMNS], train.loc[fit_mask, "label"])
        oof[val_mask] = score_candidates(model, train.loc[val_mask])
        print(f"  fold {k + 1}/5 done", flush=True)
    assert not np.isnan(oof).any()
    model = make_model(params)
    model.fit(train[FEATURE_COLUMNS], train["label"])
    DATA.mkdir(parents=True, exist_ok=True)
    keep = ["trip_id", "obs_idx", "arc_id"]
    train[keep].assign(p=oof).to_parquet(DATA / f"proba_train_oof_{which}.parquet", index=False)
    test[keep].assign(p=score_candidates(model, test)).to_parquet(
        DATA / f"proba_test_{which}.parquet", index=False)
    print(f"saved {DATA.relative_to(REPO)}/proba_train_oof_{which}.parquet and proba_test_{which}.parquet")


# ---------------------------------------------------------------- per-trip inputs
def trip_inputs(d, proba, sigma):
    """States, both emission types, graph and precomputed transition costs for one trip."""
    g = build_graph(d)
    cands = generate_candidates(d, FEATURE_OPTIONS["radius_m"], FEATURE_OPTIONS["max_k"])
    s = build_states(d, cands)
    # align probabilities by (obs_idx, arc_id), never by row order
    p = cands[["obs_idx", "arc_id"]].merge(proba, on=["obs_idx", "arc_id"], how="left")["p"]
    assert p.notna().all(), "a candidate has no probability"
    e_cls = classifier_log_emissions(s, p.to_numpy())
    e_gau = gaussian_log_emissions(s, sigma)

    # transition costs depend only on GPS + roads, so they are cached on disk per trip
    cache = DATA / "costs" / f"{d['trip_id']}.pkl"
    if cache.exists():
        costs = pd.read_pickle(cache)
    else:
        sp, costs = ShortestPaths(g["graph"]), {}
        for t in range(1, s["n_points"]):
            dt = s["t"][t] - s["t"][t - 1]
            if dt > MAX_GAP_S or s["n_cands"][t] == 0 or s["n_cands"][t - 1] == 0:
                continue
            d_gc = np.hypot(s["x"][t] - s["x"][t - 1], s["y"][t] - s["y"][t - 1])
            costs[t] = -log_transitions(g, sp, s["arcs"][t - 1], s["offset"][t - 1],
                                        s["arcs"][t], s["offset"][t], d_gc, dt, beta=1.0)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(costs, cache)
    truth = align_route(d)["true_arc"].to_numpy()
    return {"d": d, "g": g, "s": s, "e": {"gaussian": e_gau, "classifier": e_cls},
            "costs": costs, "truth": truth, "ref": d["route"]["arc_id"].to_numpy(),
            "argmax_p": np.array([a[np.argmax(e)] if len(a) else -1 for a, e in zip(s["arcs"], e_cls)]),
            "nearest": np.array([a[np.argmax(e)] if len(a) else -1 for a, e in zip(s["arcs"], e_gau)])}


def load_inputs(root, split, which, sigma, trips=None):
    proba = pd.read_parquet(DATA / (f"proba_train_oof_{which}.parquet" if split == "train"
                                    else f"proba_test_{which}.parquet"))
    out = {}
    for tid in sorted(os.listdir(root / split)):
        if (trips is not None and tid not in trips) or (split == "train" and tid in BAD_LABEL_TRIPS):
            continue
        t0 = time.perf_counter()
        out[tid] = trip_inputs(load_trip(root / split / tid), proba[proba.trip_id == tid], sigma)
        print(f"  {split} {tid}: {time.perf_counter() - t0:.1f}s", flush=True)
    return out


def decode(inp, emission, beta, w, gap_s):
    choice, breaks = viterbi(inp["s"], inp["e"][emission], inp["g"], beta, w, gap_s,
                             precomputed=inp["costs"])
    return chosen_arcs(inp["s"], choice)


def counts(inp, arcs):
    """Summable counts for Top-1 and route metrics (route = collapsed + short loops cut)."""
    g = inp["g"]
    route = build_route(arcs, g["arc_len"])
    conn, pairs = continuity(route, g["arc_from"], g["arc_to"])
    tp, n_pred, n_ref = route_prf(route, inp["ref"])
    return {"correct": int((arcs == inp["truth"]).sum()), "n": len(arcs), "conn": conn,
            "pairs": pairs, "tp": tp, "npred": n_pred, "nref": n_ref}


def rates(c):
    p, r = c["tp"] / c["npred"], c["tp"] / c["nref"]
    return {"top1": 100 * c["correct"] / c["n"], "continuity": 100 * c["conn"] / c["pairs"],
            "precision": 100 * p, "recall": 100 * r, "f1": 100 * 2 * p * r / (p + r)}


def pooled(inputs, arcs_by_trip):
    tot = None
    for tid, inp in inputs.items():
        c = counts(inp, arcs_by_trip[tid])
        tot = c if tot is None else {k: tot[k] + c[k] for k in tot}
    return rates(tot)


# ---------------------------------------------------------------- 2. decoder tuning
BETAS = [1.0, 1.7, 3.0, 5.0, 10.0, 20.0, 50.0]
WS = {"classifier": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
      "gaussian": [0.25, 0.5, 1.0, 2.0, 4.0]}          # gaussian w = (7.35 / sigma)^2
GAPS = [15.0, 30.0, 60.0, 120.0]


def tune(root, which, sigma):
    inputs = load_inputs(root, "train", which, sigma)
    rows = []
    for emission in ("classifier", "gaussian"):
        for beta in BETAS:                                   # stage 1: beta x w at gap 60 s
            for w in WS[emission]:
                t0 = time.perf_counter()
                r = pooled(inputs, {tid: decode(i, emission, beta, w, 60.0) for tid, i in inputs.items()})
                rows.append({"emission": emission, "beta": beta, "w": w, "gap_s": 60.0, **r})
                print(f"{emission:10s} beta {beta:5.1f} w {w:5.2f} gap 60: Top-1 {r['top1']:.2f}  "
                      f"F1 {r['f1']:.2f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        best = max((x for x in rows if x["emission"] == emission), key=lambda x: x["top1"])
        for gap in GAPS:                                     # stage 2: gap at the best beta, w
            if gap == 60.0:
                continue
            r = pooled(inputs, {tid: decode(i, emission, best["beta"], best["w"], gap)
                                for tid, i in inputs.items()})
            rows.append({"emission": emission, "beta": best["beta"], "w": best["w"], "gap_s": gap, **r})
            print(f"{emission:10s} beta {best['beta']:5.1f} w {best['w']:5.2f} gap {gap:5.0f}: "
                  f"Top-1 {r['top1']:.2f}  F1 {r['f1']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(OUT / f"decoder_tuning_train_{which}.csv", index=False)
    best = df.loc[df.groupby("emission")["top1"].idxmax()]
    print("\nbest per emission (train, OOF probabilities, trips 33/54 excluded):")
    print(best.round(2).to_string(index=False))
    (OUT / f"decoder_best_{which}.json").write_text(json.dumps(
        {e: {"beta_m": float(r.beta), "w": float(r.w), "gap_threshold_s": float(r.gap_s),
             "train_top1": float(r.top1), "train_route_f1": float(r.f1)}
         for e, r in best.set_index("emission").iterrows()}, indent=2) + "\n")
    print(f"saved {OUT.relative_to(REPO)}/decoder_tuning_train_{which}.csv, decoder_best_{which}.json")


# ---------------------------------------------------------------- 3. test comparison
def compare(root, which, sigma, beta0):
    best = json.loads((OUT / f"decoder_best_{which}.json").read_text())
    inputs = load_inputs(root, "student_test", which, sigma)
    methods = {
        "nearest arc": lambda i: i["nearest"],
        "classifier only": lambda i: i["argmax_p"],
        "gaussian+viterbi (untuned)": lambda i: decode(i, "gaussian", beta0, 1.0, 60.0),
        "gaussian+viterbi (tuned)": lambda i: decode(i, "gaussian", best["gaussian"]["beta_m"],
                                                     best["gaussian"]["w"], best["gaussian"]["gap_threshold_s"]),
        "classifier+viterbi (untuned)": lambda i: decode(i, "classifier", beta0, 1.0, 60.0),
        "classifier+viterbi (tuned)": lambda i: decode(i, "classifier", best["classifier"]["beta_m"],
                                                       best["classifier"]["w"], best["classifier"]["gap_threshold_s"]),
    }
    rows = []
    for tid, inp in inputs.items():
        for name, fn in methods.items():
            rows.append({"trip_id": tid, "method": name, **counts(inp, fn(inp))})
    df = pd.DataFrame(rows)
    per = df.join(pd.DataFrame([rates(r) for _, r in df.iterrows()]))
    tot = df.groupby("method", sort=False).sum(numeric_only=True)
    summary = pd.DataFrame({m: rates(tot.loc[m]) for m in tot.index}).T
    pd.set_option("display.width", 250)
    print(f"\nstudent_test, 10 trips pooled ({which} random forest; untuned decoder = beta {beta0}, w 1, gap 60):")
    print(summary.round(1).to_string())
    top = per.pivot(index="trip_id", columns="method", values="top1")[list(methods)]
    print("\nTop-1 per trip:")
    print(top.round(1).to_string())
    worse = top[top["classifier+viterbi (tuned)"] < top["classifier only"]]
    print("\ntrips where classifier+viterbi (tuned) did NOT beat classifier only on Top-1:",
          list(worse.index) or "none")
    f1 = per.pivot(index="trip_id", columns="method", values="f1")[list(methods)]
    worse_f1 = f1[f1["classifier+viterbi (tuned)"] < f1["classifier only"]]
    print("... on route F1:", list(worse_f1.index) or "none")
    per.round(3).to_csv(OUT / f"hmm_comparison_student_test_{which}.csv", index=False)
    summary.round(3).to_csv(OUT / f"hmm_comparison_student_test_{which}_summary.csv")
    print(f"saved {OUT.relative_to(REPO)}/hmm_comparison_student_test_{which}.csv (+ _summary.csv)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["probas", "tune", "compare"])
    ap.add_argument("student_folder", type=Path)
    ap.add_argument("--params", choices=["untuned", "tuned"], default="tuned")
    a = ap.parse_args()
    est = json.loads((REPO / "outputs" / "hmm" / "estimated_params.json").read_text())
    if a.cmd == "probas":
        make_probas(a.student_folder, a.params)
    elif a.cmd == "tune":
        tune(a.student_folder, a.params, est["sigma_m"])
    else:
        compare(a.student_folder, a.params, est["sigma_m"], est["beta_m"])
