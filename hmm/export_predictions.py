"""Export per-point predictions for src/final_map.py (section 6 maps).

Runs each trip through api.pipeline.Pipeline, so the predictions come from exactly what
the API serves: the tuned Random forest (models/rf_tuned.joblib) + the HMM settings in
hmm/hmm_config.json. Nothing is retrained or re-tuned here, and the .route file is
never read.

Output CSV, one row per GPS point:
  trip_id, obs_idx, pred_arc_id (classifier + HMM), clf_arc_id (classifier only)
Unmatched points (-1, no candidate arc within the search radius) are written as BLANK.
final_map.py drops blanks, but it would read -1 as an index, i.e. "the last arc of the
network", and draw a wrong line.

Usage (repo root):
  python hmm/export_predictions.py "..\\Student folder\\student_test" [trip_id ...]
                                   [--out outputs/maps/predictions_student_test.csv]
"""
import argparse
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from api.pipeline import Pipeline, list_trips


def export(trips_dir, trip_ids, out):
    pipe = Pipeline()                      # loads the model + HMM config once
    print(f"model: {pipe.model_name} ({pipe.model_path})")
    frames = []
    for tid in trip_ids:
        pred = pipe.predict(os.path.join(trips_dir, tid))
        df = pd.DataFrame({"trip_id": tid,
                           "obs_idx": range(len(pred["point_arcs"])),
                           "pred_arc_id": pred["point_arcs"],
                           "clf_arc_id": pred["classifier_point_arcs"]})
        for col in ("pred_arc_id", "clf_arc_id"):
            # -1 = unmatched -> missing value; Int64 keeps whole numbers and writes blanks
            df[col] = df[col].astype("Int64").mask(df[col] < 0)
        frames.append(df)
        print(f"  {tid}: {len(df):,} points, unmatched HMM {df['pred_arc_id'].isna().sum()}, "
              f"classifier {df['clf_arc_id'].isna().sum()}, {pred['stats']['pipeline_secs']} s")
    result = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    result.to_csv(out, index=False)
    print(f"saved {out} ({len(result):,} rows, {len(trip_ids)} trips)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export per-point predictions for final_map.py")
    ap.add_argument("trips_dir", help="folder with one sub-folder per trip, e.g. student_test")
    ap.add_argument("trip_ids", nargs="*", help="trips to export (default: all in trips_dir)")
    ap.add_argument("--out", default=os.path.join("outputs", "maps", "predictions_student_test.csv"))
    args = ap.parse_args()
    if "evaluator" in os.path.basename(os.path.normpath(args.trips_dir)).lower():
        sys.exit("refusing to export evaluator trips: they are for the final API run only")
    trips = args.trip_ids or list_trips(args.trips_dir)
    missing = [t for t in trips if t not in list_trips(args.trips_dir)]
    if missing:
        sys.exit(f"trip(s) not found in {args.trips_dir}: {missing}")
    export(args.trips_dir, trips, args.out)
