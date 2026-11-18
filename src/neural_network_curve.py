"""Measure a neural network learning curve on a held-out training fold."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline

from src.candidate_features import FEATURE_COLUMNS
from src.train_models import make_folds, model_candidates, read_partition, resolve_student_folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student_folder", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/person_b_final"))
    args = parser.parse_args()
    root = resolve_student_folder(args.student_folder)
    train, truth, _ = read_partition(root, "train", {})
    fit, validation, validation_truth = next(make_folds(train, truth, 5))

    pipeline = model_candidates(jobs=1)["Neural network"]
    preprocessing = make_pipeline(*(step for _, step in pipeline.steps[:-1]))
    model = pipeline.steps[-1][1]
    x_train = preprocessing.fit_transform(train.loc[fit, FEATURE_COLUMNS])
    x_validation = preprocessing.transform(train.loc[validation, FEATURE_COLUMNS])
    y_train = train.loc[fit, "label"].to_numpy()
    y_validation = train.loc[validation, "label"].to_numpy()

    rows = []
    for epoch in range(1, model.max_iter + 1):
        model.partial_fit(x_train, y_train, classes=np.array([0, 1]))
        validation_loss = log_loss(y_validation, model.predict_proba(x_validation), labels=[0, 1])
        rows.append({"epoch": epoch, "train_objective": model.loss_, "validation_log_loss": validation_loss})
        print(f"MLP epoch {epoch}/{model.max_iter} - validation log loss {validation_loss:.4f}", flush=True)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(rows)
    history["validation_trips"] = " ".join(sorted(validation_truth["trip_id"].unique()))
    history.to_csv(output / "neural_network_learning_curve.csv", index=False)

    write_plot(history, output)


def write_plot(history, output):
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.plot(history["epoch"], history["train_objective"], label="Training objective")
    ax.plot(history["epoch"], history["validation_log_loss"], label="Held-out fold log loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Neural network learning curve, grouped fold 1")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output / "neural_network_learning_curve.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
