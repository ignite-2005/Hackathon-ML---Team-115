"""Diagnose the TRUE transitions that come out impossible as 'same_arc_backward'.

For each case (point t moved more than NEG_TOL_M backwards along its labelled arc):
  (1) does the reverse arc (to_node -> from_node) exist?
  (2) is the GPS heading over points t-5 .. t+5 opposite to the labelled arc's direction?
Verdict per case:
  reference_direction : reverse arc exists AND heading is opposite -> label has the wrong direction
  gps_jitter          : heading matches the arc -> car drives along it, one fix jumped back
  opposite_no_reverse : heading opposite but no reverse arc (one-way road driven 'backwards')
  unclear             : car (almost) still over the window, or moving sideways to the arc
Uses labels, so this is TRAIN analysis only.

Usage: python hmm/check_backward.py "<Student folder>"
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
from geo import to_metres, trip_origin
from candidates import arc_geometry
from transitions import train_true_transitions

WINDOW = 5         # points either side of t
MIN_MOVE_M = 5.0   # less GPS movement than this over the window = car (almost) still
COS_CLEAR = 0.5    # |cos| above this = clearly along / against the arc (within 60 degrees)


def classify_trip(d, cases):
    """Add has_reverse, cos_heading, move_m and verdict to one trip's backward cases."""
    tr = d["track"]
    x, y = to_metres(tr["lon"], tr["lat"], trip_origin(d))
    ax, ay, bx, by = arc_geometry(d)
    arcs = d["arcs"]
    pairs = set(zip(arcs["from_node"], arcs["to_node"]))

    out = []
    for _, c in cases.iterrows():
        a, t = int(c["arc_a"]), int(c["obs_idx"])
        f, to = arcs.at[a, "from_node"], arcs.at[a, "to_node"]
        has_reverse = (to, f) in pairs

        lo, hi = max(t - WINDOW, 0), min(t + WINDOW, len(x) - 1)
        gx, gy = x[hi] - x[lo], y[hi] - y[lo]            # GPS movement over the window
        vx, vy = bx[a] - ax[a], by[a] - ay[a]            # labelled arc direction
        move = np.hypot(gx, gy)
        cos = (gx * vx + gy * vy) / (move * np.hypot(vx, vy)) if move > 0 else 0.0

        if move < MIN_MOVE_M or abs(cos) <= COS_CLEAR:
            verdict = "unclear"
        elif cos > COS_CLEAR:
            verdict = "gps_jitter"
        elif has_reverse:
            verdict = "reference_direction"
        else:
            verdict = "opposite_no_reverse"
        out.append({"has_reverse": has_reverse, "cos_heading": cos,
                    "move_m": move, "verdict": verdict})
    return pd.concat([cases.reset_index(drop=True), pd.DataFrame(out)], axis=1)


if __name__ == "__main__":
    root = sys.argv[1]  # path to "Student folder"
    labels = pd.read_parquet(os.path.join(REPO, "data", "candidates", "point_labels_train.parquet"))
    tt = train_true_transitions(root, labels)
    back = tt[tt["reason"] == "same_arc_backward"]

    res = pd.concat([classify_trip(load_trip(os.path.join(root, "train", tid)), grp)
                     for tid, grp in back.groupby("trip_id")], ignore_index=True)

    print(f"same_arc_backward cases: {len(res):,}")
    print(f"(1) reverse arc exists      : {res['has_reverse'].sum():,} ({100 * res['has_reverse'].mean():.0f}%)")
    print(f"(2) heading opposite to arc : {(res['cos_heading'] < -COS_CLEAR).sum():,}   "
          f"matches arc: {(res['cos_heading'] > COS_CLEAR).sum():,}")
    print("\nverdict:")
    print(res["verdict"].value_counts().to_string())

    per_trip = pd.crosstab(res["trip_id"], res["verdict"])
    per_trip["total"] = per_trip.sum(axis=1)
    print("\nper trip:")
    print(per_trip.sort_values("total", ascending=False).to_string())

    out_dir = os.path.join(REPO, "outputs", "hmm")
    os.makedirs(out_dir, exist_ok=True)
    res.to_csv(os.path.join(out_dir, "backward_cases_train.csv"), index=False)
    per_trip.to_csv(os.path.join(out_dir, "backward_cases_per_trip_train.csv"))
    print(f"\nsaved: outputs/hmm/backward_cases_train.csv, backward_cases_per_trip_train.csv")
