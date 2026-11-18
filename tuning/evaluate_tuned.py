"""Tuned vs untuned Random forest, both measured the same way on the FULL data.

For each of {untuned (Person B), tuned (models/best_params.json)}:
  - 5-fold GroupKFold CV on all 80 train trips (Person B's folds, no subsampling):
    mean candidate Macro F1 (+ std), mean Top-1, mean log loss
  - refit on all 80 train trips, then student_test: Top-1, Macro F1, log loss,
    fit seconds, scoring time (median of 3 passes over the test candidates, in ms)
The tuned model is saved as models/rf_tuned.joblib in Person B's bundle format, so
predict_trip() / decode_trip() can load it like best_model.joblib.

Usage (repo root): python tuning/evaluate_tuned.py "<Student folder>"
"""
import json
import sys
import time
from pathlib import Path
from statistics import median

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tuning"))
from src.candidate_features import FEATURE_COLUMNS
from src.train_models import evaluate, make_folds, read_partition, score_candidates
from hpo import UNTUNED, FEATURE_OPTIONS, make_model, OUT_DIR, BEST_PARAMS


def measure(params, train, truth, test, test_truth, folds):
    cv = []
    for fit_mask, val_mask, val_truth in folds:
        m = make_model(params)
        m.fit(train.loc[fit_mask, FEATURE_COLUMNS], train.loc[fit_mask, "label"])
        val = train.loc[val_mask]
        cv.append(evaluate(val, val_truth, score_candidates(m, val), probabilistic=True))
    cv = pd.DataFrame(cv)

    model = make_model(params)
    t0 = time.perf_counter()
    model.fit(train[FEATURE_COLUMNS], train["label"])
    fit_s = time.perf_counter() - t0
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        scores = score_candidates(model, test)
        times.append(1000 * (time.perf_counter() - t0))
    te = evaluate(test, test_truth, scores, probabilistic=True)
    row = {"cv_macro_f1_mean": cv["candidate_macro_f1"].mean(),
           "cv_macro_f1_std": cv["candidate_macro_f1"].std(),
           "cv_top1_mean": cv["top1_accuracy"].mean(), "cv_log_loss_mean": cv["log_loss"].mean(),
           "test_top1": te["top1_accuracy"], "test_macro_f1": te["candidate_macro_f1"],
           "test_log_loss": te["log_loss"], "fit_seconds": fit_s, "inference_ms": median(times)}
    return row, model


if __name__ == "__main__":
    root = Path(sys.argv[1])
    train, truth, _ = read_partition(root, "train", FEATURE_OPTIONS)
    test, test_truth, _ = read_partition(root, "student_test", FEATURE_OPTIONS)
    folds = list(make_folds(train, truth, 5))
    tuned = json.loads(BEST_PARAMS.read_text())["params"]

    rows = {}
    for name, params in (("untuned (Person B)", UNTUNED), ("tuned (Optuna)", tuned)):
        print(f"measuring {name}: {params}", flush=True)
        rows[name], model = measure(params, train, truth, test, test_truth, folds)
    joblib.dump({"model_name": "Random forest (tuned)", "estimator": model,
                 "feature_columns": FEATURE_COLUMNS, "feature_options": FEATURE_OPTIONS,
                 "params": tuned}, REPO / "models" / "rf_tuned.joblib", compress=3)

    df = pd.DataFrame(rows).T
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.round(5).to_csv(OUT_DIR / "tuned_vs_untuned.csv")
    pd.set_option("display.width", 250)
    print(df.round(4).to_string())
    print("saved outputs/tuning/tuned_vs_untuned.csv and models/rf_tuned.joblib")
