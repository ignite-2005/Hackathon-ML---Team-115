"""Part A, step 3: emission scores, i.e. how well each candidate arc explains its GPS point.

Two versions, kept side by side so they can be compared in the report:
  baseline  : log_e = -0.5 * (dist_m / sigma)^2      (Gaussian GPS noise, Newson & Krumm 2009)
  classifier: log_e = log(p_classifier + 1e-9)       (Person B's predict_proba per candidate row)
Both return one array per GPS point, aligned with states["arcs"][t].

The Gaussian's constant -log(sigma * sqrt(2*pi)) is dropped: it is the same for every
candidate, so it never changes which path Viterbi picks.

Usage (quick check): python hmm/emissions.py "<Student folder>" [split/trip_id ...]
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)
from load_data import load_trip
from candidates import generate_candidates, align_route, MAX_MEDIAN_TRUE_DIST_M
from states import build_states

EPS = 1e-9  # stops log(0) = -inf when the classifier gives a probability of exactly 0


def estimate_sigma(point_labels):
    """GPS noise sigma (m) from TRAIN labels: 1.4826 * median distance to the true arc.

    1.4826 * median is a robust standard-deviation estimate (Newson & Krumm 2009);
    unlike the plain std it is not blown up by the few points that are 100+ m off.
    Trips whose labels are unreliable (median distance > 50 m) are left out.
    """
    trip_median = point_labels.groupby("trip_id")["true_dist_m"].transform("median")
    ok = point_labels[trip_median <= MAX_MEDIAN_TRUE_DIST_M]
    return float(1.4826 * ok["true_dist_m"].median())


def gaussian_log_emissions(states, sigma):
    """Baseline: candidates closer to the GPS point score higher."""
    return [-0.5 * (dist / sigma) ** 2 for dist in states["dist"]]


def classifier_log_emissions(states, proba):
    """Final: log of the classifier probability of each candidate.

    proba = model.predict_proba(features)[:, 1], one value per row of the candidate
    table that was passed to build_states, in the same row order.
    """
    proba = np.asarray(proba, dtype=float)
    if len(proba) != sum(len(r) for r in states["rows"]):
        raise ValueError("proba must have one value per candidate row")
    return [np.log(proba[r] + EPS) for r in states["rows"]]


def describe(trip_folder, sigma):
    """Emission-only matching: pick the best-scoring candidate at each point, no HMM.

    This is the 'no transitions' baseline that Viterbi has to beat. Labels are used
    here only to score it (evaluation), never to compute the emissions.
    """
    d = load_trip(trip_folder)
    cands = generate_candidates(d)
    s = build_states(d, cands)
    log_e = gaussian_log_emissions(s, sigma)
    truth = align_route(d)["true_arc"].to_numpy()

    has = s["n_cands"] > 0
    best = np.array([a[np.argmax(e)] if len(a) else -1 for a, e in zip(s["arcs"], log_e)])
    # ties: the top score is shared by 2+ arcs (a two-way road gives both directions
    # the same distance). Emissions alone cannot choose the direction.
    ties = np.array([len(e) > 1 and np.isclose(np.sort(e)[-1], np.sort(e)[-2]) for e in log_e])
    true_in_ties = np.array([
        len(e) > 0 and truth[i] in a[np.isclose(e, e.max())]
        for i, (a, e) in enumerate(zip(s["arcs"], log_e))])

    # mechanical check of the classifier path: probability 1 gives log_e ~ 0
    ones = classifier_log_emissions(s, np.ones(len(cands)))
    assert all(np.allclose(x, 0.0, atol=1e-8) for x in ones)

    print(f"\n=== trip {s['trip_id']} ===  (sigma = {sigma:.2f} m)")
    print(f"points with candidates            : {has.sum():,} / {s['n_points']:,}")
    print(f"log_e range                       : {min(e.min() for e in log_e if len(e)):.1f} .. 0")
    print(f"best emission == true arc (Top-1) : {100 * np.mean(best[has] == truth[has]):.1f}%")
    print(f"true arc among the tied best      : {100 * np.mean(true_in_ties[has]):.1f}%")
    print(f"points where the best is a tie    : {100 * np.mean(ties[has]):.1f}%")


if __name__ == "__main__":
    root = sys.argv[1]  # path to "Student folder"
    labels = pd.read_parquet(os.path.join(REPO, "outputs", "candidates", "point_labels_train.parquet"))
    sigma = estimate_sigma(labels)  # train only
    print(f"sigma estimated from train labels: {sigma:.2f} m")
    trips = sys.argv[2:] or ["train/00000046", "train/00000013", "student_test/00000056"]
    for tp in trips:
        describe(os.path.join(root, *tp.split("/")), sigma)
