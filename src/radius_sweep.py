"""Candidate miss rate and candidate count vs search radius (TRAIN partition only).

Used to choose the candidate radius without looking at test data.
Usage: python radius_sweep.py "<Student folder>" <out_dir>
"""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import load_trip
from candidates import (generate_candidates, align_route, add_labels,
                        MAX_MEDIAN_TRUE_DIST_M)

RADII = [10, 20, 30, 40, 50, 75, 100]
MAX_K = 16

if __name__ == "__main__":
    root, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    tdir = os.path.join(root, "train")

    rows = []
    for tid in sorted(os.listdir(tdir)):
        d = load_trip(os.path.join(tdir, tid))
        truth = align_route(d)
        if truth["true_dist_m"].median() > MAX_MEDIAN_TRUE_DIST_M:
            continue  # same quality rule as candidates.py
        for r in RADII:
            c = add_labels(generate_candidates(d, r, MAX_K), truth)
            hit = c.loc[c["label"] == 1, "obs_idx"].nunique()
            rows.append({"trip_id": tid, "radius_m": r, "n_gps": len(truth),
                         "n_hit": hit, "n_cand_rows": len(c)})
        print(tid, flush=True)

    df = pd.DataFrame(rows)
    s = df.groupby("radius_m")[["n_gps", "n_hit", "n_cand_rows"]].sum()
    s["miss_rate_pct"] = 100 * (1 - s["n_hit"] / s["n_gps"])
    s["cands_per_point"] = s["n_cand_rows"] / s["n_gps"]
    s = s.round(3)
    s.to_csv(os.path.join(out, "radius_sweep_train.csv"))
    print(s[["miss_rate_pct", "cands_per_point"]])

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(s.index, s["miss_rate_pct"], "o-", color="tab:red")
    ax1.set_xlabel("candidate search radius (m)")
    ax1.set_ylabel("candidate miss rate (%)", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(s.index, s["cands_per_point"], "s--", color="tab:blue")
    ax2.set_ylabel("candidates per GPS point", color="tab:blue")
    plt.title(f"Radius vs miss rate and candidate count (train, max_k={MAX_K})")
    fig.tight_layout()
    plt.savefig(os.path.join(out, "radius_sweep_train.png"), dpi=130)
