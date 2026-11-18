"""Compare map-matching classifiers and export a fitted scoring pipeline."""

import argparse
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.candidate_features import FEATURE_COLUMNS, KEY, build_features
from src.candidates import MAX_MEDIAN_TRUE_DIST_M, add_labels, align_route
from src.load_data import load_trip

SEED = 115
BASELINE = "Nearest arc"


def model_candidates(jobs):
    def pipeline(estimator, scale=False):
        steps = [SimpleImputer(strategy="median", keep_empty_features=True)]
        if scale:
            steps.append(StandardScaler())
        return make_pipeline(*steps, estimator)

    return {
        "Logistic regression": pipeline(LogisticRegression(max_iter=300), scale=True),
        "Gaussian naive Bayes": pipeline(GaussianNB(), scale=True),
        "Random forest": pipeline(
            RandomForestClassifier(
                n_estimators=64, max_depth=14, min_samples_leaf=20,
                max_samples=0.15, n_jobs=jobs, random_state=SEED,
            )
        ),
        "Histogram gradient boosting": pipeline(
            HistGradientBoostingClassifier(
                max_iter=100, max_leaf_nodes=31, early_stopping=False, random_state=SEED,
            )
        ),
        "Neural network": pipeline(
            MLPClassifier(
                hidden_layer_sizes=(32, 16), batch_size=2048, max_iter=20,
                random_state=SEED,
            ),
            scale=True,
        ),
    }


def resolve_student_folder(root):
    """Resolve a dataset root or a folder containing one extracted dataset."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset path does not exist or is not a directory: {root}")
    if (root / "train").is_dir() and (root / "student_test").is_dir():
        return root

    matches = []
    for directory, folders, _ in os.walk(root):
        if {"train", "student_test"}.issubset(folders):
            matches.append(Path(directory))
            folders.clear()
        else:
            folders[:] = [name for name in folders if not name.startswith(".")]

    if len(matches) == 1:
        return matches[0]
    if matches:
        choices = "\n".join(str(path) for path in sorted(matches))
        raise ValueError(f"Multiple datasets found. Pass one of these folders:\n{choices}")
    raise ValueError(
        f"No folder under {root} contains both train and student_test. "
        "Extract the student dataset and pass its containing folder."
    )


def read_partition(root, partition, feature_options, *, exclude_flagged_train=False):
    folders = sorted(
        folder for folder in (root / partition).iterdir()
        if folder.is_dir() and not folder.name.startswith(".")
    )
    frames, truths, excluded = [], [], []
    for folder in folders:
        trip = load_trip(folder)
        truth = align_route(trip)
        distance = float(truth["true_dist_m"].median())
        if exclude_flagged_train and partition == "train" and distance > MAX_MEDIAN_TRUE_DIST_M:
            excluded.append({"trip_id": trip["trip_id"], "median_true_dist_m": distance})
            continue
        features = build_features(trip, **feature_options)
        arcs = trip["arcs"].iloc[features["arc_id"].to_numpy(dtype=np.intp)]
        features[["from_node", "to_node"]] = arcs[["from_node", "to_node"]].to_numpy()
        rows = add_labels(features, truth)[
            KEY + ["arc_id", "label", "timestamp", "from_node", "to_node"] + FEATURE_COLUMNS
        ]
        rows = rows.astype(dict.fromkeys(FEATURE_COLUMNS, np.float32))
        frames.append(rows)
        truths.append(truth[KEY + ["true_arc"]])
    if not frames:
        raise ValueError(f"No eligible trip directories found in {root / partition}")
    return pd.concat(frames, ignore_index=True), pd.concat(truths, ignore_index=True), excluded


def score_candidates(estimator, candidates):
    """Return one positive-class probability per candidate, in input order."""
    if candidates.empty:
        return np.empty(0, dtype=float)
    matrix = candidates[FEATURE_COLUMNS]
    if not matrix.dtypes.eq(np.float32).all():
        matrix = matrix.astype(np.float32)
    positive = np.flatnonzero(estimator.classes_ == 1).item()
    return estimator.predict_proba(matrix)[:, positive]


def predict_trip(bundle, trip):
    """Return candidate geometry and probabilities without consulting a reference route."""
    rows = build_features(trip, **bundle["feature_options"])
    rows["match_probability"] = score_candidates(bundle["estimator"], rows)
    return rows


def winner_positions(rows, scores):
    # Rows arrive in distance and arc-ID order within each observation.
    ranked = rows[KEY].reset_index(drop=True).assign(score=np.asarray(scores))
    if ranked.empty:
        return np.empty(0, dtype=np.intp)
    return ranked.groupby(KEY, sort=False)["score"].idxmax().to_numpy(dtype=np.intp)


def direct_route_continuity(winners, max_gap_s):
    """Share of eligible consecutive GPS pairs on the same or directly adjacent arcs."""
    previous = winners.shift()
    dt = winners["timestamp"] - previous["timestamp"]
    eligible = (
        winners["trip_id"].eq(previous["trip_id"])
        & winners["obs_idx"].sub(previous["obs_idx"]).eq(1)
        & dt.gt(0) & dt.le(max_gap_s)
    )
    count = int(eligible.sum())
    connected = winners["arc_id"].eq(previous["arc_id"]) | winners["from_node"].eq(previous["to_node"])
    return (int((eligible & connected).sum()) / count if count else np.nan), count


def evaluate(rows, truth, scores, *, probabilistic, max_gap_s=5.0):
    positions = winner_positions(rows, scores)
    labels = rows["label"].to_numpy()
    hits = int(labels[positions].sum())
    count = len(truth)
    continuity, eligible_pairs = direct_route_continuity(rows.iloc[positions], max_gap_s)
    result = {
        "direct_route_continuity": continuity,
        "continuity_pairs": eligible_pairs,
        "top1_accuracy": hits / count,
        "top1_f1": 2 * hits / (count + len(positions)),
        "candidate_coverage": int(labels.sum()) / count,
        "candidate_f1": np.nan,
        "candidate_macro_f1": np.nan,
        "candidate_precision": np.nan,
        "candidate_recall": np.nan,
        "candidate_accuracy": np.nan,
        "tn": 0, "fp": 0, "fn": 0, "tp": 0,
        "log_loss": np.nan,
    }
    if len(rows):
        if probabilistic:
            hard = scores >= 0.5
            result["log_loss"] = log_loss(labels, scores, labels=[0, 1])
        else:
            hard = np.zeros(len(rows), dtype=bool)
            hard[positions] = True
        tn, fp, fn, tp = (int(value) for value in confusion_matrix(labels, hard, labels=[0, 1]).ravel())

        def ratio(numerator, denominator):
            return numerator / denominator if denominator else 0.0

        positive_f1 = ratio(2 * tp, 2 * tp + fp + fn)
        negative_f1 = ratio(2 * tn, 2 * tn + fp + fn)
        result.update(
            candidate_f1=positive_f1,
            candidate_macro_f1=(positive_f1 + negative_f1) / 2,
            candidate_precision=ratio(tp, tp + fp),
            candidate_recall=ratio(tp, tp + fn),
            candidate_accuracy=ratio(tp + tn, len(rows)),
            tn=tn, fp=fp, fn=fn, tp=tp,
        )
    return result


def make_folds(train, truth, n_splits):
    # Split the full trip list so observations with no candidate still count.
    trips = truth["trip_id"].unique()
    if len(trips) < n_splits:
        raise ValueError(f"Need {n_splits} training trips, found {len(trips)}")
    for fit, val in GroupKFold(n_splits=n_splits).split(trips, groups=trips):
        fit_mask = train["trip_id"].isin(trips[fit]).to_numpy()
        if train.loc[fit_mask, "label"].nunique() != 2:
            raise ValueError("A training fold is missing one candidate class.")
        val_mask = train["trip_id"].isin(trips[val]).to_numpy()
        yield fit_mask, val_mask, truth[truth["trip_id"].isin(trips[val])]


def score(estimator, rows):
    return -rows["dist_m"].to_numpy() if estimator is None else score_candidates(estimator, rows)


def compare(train, train_truth, test, test_truth, *, n_splits, jobs, max_gap_s=5.0):
    folds = list(make_folds(train, train_truth, n_splits))
    models = {BASELINE: None, **model_candidates(jobs)}
    cv_rows = []
    for name, estimator in models.items():
        print(f"Cross-validating {name}", flush=True)
        for fold, (fit_mask, val_mask, val_truth) in enumerate(folds, start=1):
            print(f"  Fold {fold}/{n_splits}", flush=True)
            started = perf_counter()
            if estimator is not None:
                estimator.fit(train.loc[fit_mask, FEATURE_COLUMNS], train.loc[fit_mask, "label"])
            fit_seconds = perf_counter() - started if estimator is not None else 0.0
            val = train.loc[val_mask]
            metrics = evaluate(val, val_truth, score(estimator, val),
                               probabilistic=estimator is not None, max_gap_s=max_gap_s)
            cv_rows.append({
                "model": name, "fold": fold,
                "validation_trips": " ".join(sorted(val_truth["trip_id"].unique())),
                "fit_seconds": fit_seconds, **metrics,
            })
            print(f"    Finished in {perf_counter() - started:.1f}s", flush=True)

    cv = pd.DataFrame(cv_rows)
    summary = cv.groupby("model", sort=False).agg(
        cv_f1_mean=("candidate_f1", "mean"),
        cv_f1_std=("candidate_f1", "std"),
        cv_macro_f1_mean=("candidate_macro_f1", "mean"),
        cv_macro_f1_std=("candidate_macro_f1", "std"),
        cv_top1_mean=("top1_accuracy", "mean"),
        cv_fit_seconds_mean=("fit_seconds", "mean"),
    )
    selected = summary.drop(index=BASELINE).sort_values(
        ["cv_top1_mean", "cv_macro_f1_mean"], ascending=False, kind="stable"
    ).index[0]

    test_rows, exported_scores = [], {}
    for name, estimator in models.items():
        print(f"Final training and test scoring: {name}", flush=True)
        started = perf_counter()
        if estimator is not None:
            estimator.fit(train[FEATURE_COLUMNS], train["label"])
        fit_seconds = perf_counter() - started if estimator is not None else 0.0
        train_metrics = evaluate(train, train_truth, score(estimator, train),
                                 probabilistic=estimator is not None, max_gap_s=max_gap_s)
        timings = []
        for _ in range(3):
            start = perf_counter()
            scores = score(estimator, test)
            timings.append(1000 * (perf_counter() - start))
        metrics = evaluate(test, test_truth, scores,
                           probabilistic=estimator is not None, max_gap_s=max_gap_s)
        test_rows.append({
            "model": name, **{f"test_{key}": value for key, value in metrics.items()},
            "train_macro_f1": train_metrics["candidate_macro_f1"],
            "train_top1_accuracy": train_metrics["top1_accuracy"],
            "fit_seconds": fit_seconds,
            "inference_ms": median(timings), "selected_by_cv": name == selected,
        })
        if name in (BASELINE, selected):
            exported_scores[name] = scores
    summary = summary.join(pd.DataFrame(test_rows).set_index("model")).reset_index()
    return cv, summary, selected, models[selected], exported_scores


def write_comparison(output, cv, summary, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cv.to_csv(output / "cv_folds.csv", index=False)
    summary.to_csv(output / "model_comparison.csv", index=False)
    panels = [
        ("cv_macro_f1_mean", "CV candidate macro F1"),
        ("test_top1_accuracy", "Test Top-1 accuracy"),
        ("test_candidate_macro_f1", "Test candidate macro F1"),
        ("test_direct_route_continuity", "Test direct route continuity"),
        ("test_log_loss", "Test log loss"),
        ("inference_ms", "Test scoring time (ms)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 7), layout="constrained")
    for ax, (column, title) in zip(axes.flat, panels):
        rows = summary.dropna(subset=[column])
        errors = rows["cv_macro_f1_std"].fillna(0) if column == "cv_macro_f1_mean" else None
        colors = np.where(rows["selected_by_cv"], "#15616d", "#8e9aaf")
        ax.barh(rows["model"], rows[column], xerr=errors, color=colors)
        ax.invert_yaxis()
        ax.set_title(title)
        if column.endswith("f1") or column in {"cv_macro_f1_mean", "test_top1_accuracy", "test_direct_route_continuity"}:
            ax.set_xlim(0, 1.05)
    fig.savefig(output / "model_comparison.png", dpi=160)
    plt.close(fig)

    report = [
        "## Model comparison results", "",
        f"Selected using training cross-validation: **{selected}**.", "",
        "| Model | CV macro F1 (mean +/- SD) | Test Top-1 | Test macro F1 | Test precision | Test recall | Test continuity | Test log loss | Fit (s) | Score (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        loss = "N/A (hard choice)" if pd.isna(row.test_log_loss) else f"{row.test_log_loss:.4f}"
        report.append(
            f"| {row.model} | {row.cv_macro_f1_mean:.4f} +/- {row.cv_macro_f1_std:.4f} | "
            f"{row.test_top1_accuracy:.4f} | {row.test_candidate_macro_f1:.4f} | "
            f"{row.test_candidate_precision:.4f} | {row.test_candidate_recall:.4f} | "
            f"{row.test_direct_route_continuity:.4f} | {loss} | "
            f"{row.fit_seconds:.2f} | {row.inference_ms:.2f} |"
        )
    report.extend([
        "", f"Test candidate coverage was {summary['test_candidate_coverage'].iloc[0]:.2%}.",
        "", "![Model comparison plots](model_comparison.png)",
    ])
    (output / "section3_results.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def write_results(output, train_truth, test_truth, test, cv, summary,
                  selected, estimator, scores, options, excluded):
    output.mkdir(parents=True, exist_ok=True)
    write_comparison(output, cv, summary, selected)
    joblib.dump({
        "model_name": selected, "estimator": estimator, "feature_columns": FEATURE_COLUMNS,
        "feature_options": options,
    }, output / "best_model.joblib", compress=3)

    test[KEY + ["arc_id"]].assign(match_probability=scores[selected]).to_parquet(
        output / "test_candidate_probabilities.parquet", index=False
    )
    predictions = test_truth.copy()
    for name, values in scores.items():
        column = "nearest_arc" if name == BASELINE else "predicted_arc"
        winners = test.iloc[winner_positions(test, values)][KEY + ["arc_id"]]
        predictions = predictions.merge(
            winners.rename(columns={"arc_id": column}), on=KEY, how="left", validate="one_to_one",
        )
        predictions[column] = predictions[column].astype("Int64")
    predictions.to_csv(output / "test_predictions.csv", index=False)
    for column in ("nearest_arc", "predicted_arc"):
        predictions[f"{column}_correct"] = predictions[column].eq(predictions["true_arc"]).fillna(False)
    predictions.groupby("trip_id").agg(
        observations=("obs_idx", "size"),
        predicted_observations=("predicted_arc", "count"),
        nearest_accuracy=("nearest_arc_correct", "mean"),
        model_accuracy=("predicted_arc_correct", "mean"),
    ).to_csv(output / "test_trip_accuracy.csv")

    metadata = {
        "selected_model": selected,
        "selection": "Mean trip-grouped CV Top-1 accuracy, then mean CV candidate macro F1",
        "seed": SEED,
        "estimator_parameters": estimator.steps[-1][1].get_params(),
        "training_trips": sorted(train_truth["trip_id"].unique()),
        "test_trips": sorted(test_truth["trip_id"].unique()),
        "excluded_training_trips": excluded,
        "feature_options": options,
        "feature_columns": FEATURE_COLUMNS,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student_folder", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/person_b"))
    parser.add_argument("--radius-m", type=float, default=50.0)
    parser.add_argument("--max-k", type=int, default=16)
    parser.add_argument("--max-heading-gap-s", type=float, default=5.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=2, help="Parallel workers for Random Forest")
    parser.add_argument("--exclude-flagged-train", action="store_true",
                        help="Optional sensitivity run excluding low-quality training labels")
    args = parser.parse_args()
    if not (
        np.isfinite(args.radius_m) and args.radius_m > 0
        and np.isfinite(args.max_heading_gap_s) and args.max_heading_gap_s > 0
        and args.max_k >= 1 and args.folds >= 2 and args.jobs >= 1
    ):
        parser.error("Use positive radius, candidate cap, heading gap and jobs, and at least two folds.")

    options = {"radius_m": args.radius_m, "max_k": args.max_k, "max_heading_gap_s": args.max_heading_gap_s}
    try:
        data_root = resolve_student_folder(args.student_folder)
    except ValueError as error:
        parser.error(str(error))
    print(f"Using dataset at {data_root}", flush=True)
    train, train_truth, excluded = read_partition(
        data_root, "train", options, exclude_flagged_train=args.exclude_flagged_train
    )
    test, test_truth, _ = read_partition(data_root, "student_test", options)
    if set(train_truth["trip_id"]) & set(test_truth["trip_id"]):
        raise ValueError("Train and test contain overlapping trip IDs.")
    comparison = compare(
        train, train_truth, test, test_truth, n_splits=args.folds, jobs=args.jobs,
        max_gap_s=args.max_heading_gap_s,
    )
    write_results(args.output, train_truth, test_truth, test, *comparison, options, excluded)
    print(comparison[1].to_string(index=False))
    print(f"Selected {comparison[2]}. Results written to {args.output}.")


if __name__ == "__main__":
    main()
