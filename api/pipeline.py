"""PathSnap inference pipeline used by the API (Person D).

predict(folder)   : track + road network -> arc per GPS point and the driven route.
                    Runs Person C's decoder (hmm/decode.py) with the model named in
                    hmm/hmm_config.json (the Optuna-tuned Random forest, models/rf_tuned.joblib)
                    as HMM emissions. It never opens the .route file.
evaluate(folder,p): server-side scoring against the trip's reference .route. The results
                    are only numbers. The reference route itself is never returned to the UI.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "src"), os.path.join(REPO, "hmm")):
    if p not in sys.path:
        sys.path.insert(0, p)

from decode import (decode_trip, load_config, load_model, load_trip_no_labels,  # noqa: E402
                    classifier_proba)                                             # (hmm/)
from metrics import route_from_points, continuity, route_prf, prf   # noqa: E402  (hmm/)
from load_data import load_trip                                    # noqa: E402  (src/)
from candidates import align_route, generate_candidates            # noqa: E402  (src/)

# Optional override, e.g. PATHSNAP_MODEL=results/person_b_final/best_model.joblib (untuned)
MODEL_OVERRIDE = os.environ.get("PATHSNAP_MODEL")


class Pipeline:
    """Loads the model and HMM config once, then decodes any number of trips."""

    def __init__(self, model_path=MODEL_OVERRIDE):
        self.config = load_config()
        self.model_path = model_path or self.config["model_path"]
        self.bundle = load_model(self.model_path)
        self.model_name = self.bundle["model_name"]
        self.feature_options = self.bundle["feature_options"]
        # the classifier and the decoder must search candidates the same way
        for key in ("radius_m", "max_k"):
            if self.feature_options[key] != self.config[key]:
                raise ValueError(f"{key} differs between the model bundle and hmm_config.json")

    def predict(self, folder):
        """Decode one trip folder. Reads only .track, .nodes and .arcs."""
        t0 = time.perf_counter()
        out = decode_trip(folder, self.bundle, self.config)
        out["stats"]["pipeline_secs"] = round(time.perf_counter() - t0, 2)

        # Classifier-only prediction (no HMM): the highest-probability candidate per point,
        # with the same candidates and the same model as the decoder.
        d = load_trip_no_labels(folder)
        cands = generate_candidates(d, self.config["radius_m"], self.config["max_k"])
        clf = np.full(len(d["track"]), -1, dtype=int)
        if len(cands):
            r = cands[["obs_idx", "arc_id"]].assign(p=classifier_proba(d, cands, self.bundle))
            best = r.loc[r.groupby("obs_idx", sort=False)["p"].idxmax()]   # ties -> nearest
            clf[best["obs_idx"].to_numpy()] = best["arc_id"].to_numpy()
        out["classifier_point_arcs"] = clf.tolist()
        out["model_name"] = self.model_name
        return out

    @staticmethod
    def gps(folder):
        """GPS points as [[lat, lon], ...] for drawing (input data, not labels)."""
        tr = load_trip_no_labels(folder)["track"]
        return tr[["lat", "lon"]].round(6).to_numpy().tolist()

    @staticmethod
    def evaluate(folder, pred):
        """Server-side metrics vs the reference route. Returns None if no .route exists."""
        trip_id = os.path.basename(os.path.normpath(folder))
        if not os.path.exists(os.path.join(folder, f"{trip_id}.route")):
            return None
        d = load_trip(folder)
        truth = align_route(d)["true_arc"].to_numpy()
        ref = d["route"]["arc_id"].to_numpy()
        arc_from = d["arcs"]["from_node"].to_numpy()
        arc_to = d["arcs"]["to_node"].to_numpy()

        def score(point_arcs, route=None):
            a = np.asarray(point_arcs)
            route = route_from_points(a) if route is None else np.asarray(route)
            conn, pairs = continuity(route, arc_from, arc_to)
            p, r, f1 = prf(*route_prf(route, ref))
            return {
                "top1_accuracy": round(float((a == truth).mean()), 4),  # unmatched = wrong
                "route_continuity": round(conn / pairs, 4) if pairs else None,
                "route_precision": round(p, 4),
                "route_recall": round(r, 4),
                "route_f1": round(f1, 4),
                "unmatched_points": int((a < 0).sum()),
            }

        return {
            "hmm": score(pred["point_arcs"], pred["route"]),
            "classifier_only": score(pred["classifier_point_arcs"]),
            "n_points": int(len(truth)),
            "n_reference_arcs": int(len(ref)),
        }


def list_trips(trips_dir):
    """Trip IDs in a folder (each trip is a sub-folder holding <id>.track etc.)."""
    if not os.path.isdir(trips_dir):
        return []
    return sorted(t for t in os.listdir(trips_dir)
                  if os.path.isfile(os.path.join(trips_dir, t, f"{t}.track")))


if __name__ == "__main__":
    # python api/pipeline.py "<trips folder>" [trip_id ...] [--out results.csv]
    import argparse
    ap = argparse.ArgumentParser(description="Run the API pipeline offline on a trips folder")
    ap.add_argument("trips_dir")
    ap.add_argument("trip_ids", nargs="*")
    ap.add_argument("--out", help="CSV file for the per-trip results")
    args = ap.parse_args()
    root = args.trips_dir
    trips = args.trip_ids or list_trips(root)
    pipe = Pipeline()
    rows = []
    for tid in trips:
        folder = os.path.join(root, tid)
        pred = pipe.predict(folder)
        ev = pipe.evaluate(folder, pred)
        rows.append({"trip_id": tid, "points": len(pred["point_arcs"]),
                     "secs": pred["stats"]["pipeline_secs"],
                     **{f"hmm_{k}": v for k, v in ev["hmm"].items()},
                     **{f"clf_{k}": v for k, v in ev["classifier_only"].items()}})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    # pooled over all points (weighted by trip length), like the UI summary
    w = df["points"]
    total = {"trip_id": "POOLED", "points": int(w.sum()), "secs": round(df["secs"].sum(), 2)}
    for col in df.columns:
        if col.endswith(("top1_accuracy", "route_continuity", "route_precision",
                         "route_recall", "route_f1")):
            total[col] = round(float((df[col] * w).sum() / w.sum()), 4)
        elif col.endswith("unmatched_points"):
            total[col] = int(df[col].sum())
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"saved {args.out}")
