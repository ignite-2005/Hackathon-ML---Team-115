"""Check grouped evaluation and the classifier-to-decoder handoff."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import joblib
import pandas as pd

from src.candidate_features import build_features
from src.load_data import load_trip
from src.train_models import compare, predict_trip, read_partition, write_results


def write_trip(root, partition, trip_id, *, reverse=False, gap=False):
    folder = root / partition / trip_id
    folder.mkdir(parents=True)
    longitudes = [0.0001, 0.0003, 0.0005]
    if reverse:
        longitudes.reverse()
    track = "".join(f"{lon} 0 {idx}\n" for idx, lon in enumerate(longitudes))
    if gap:
        track += "0.0100 0 90\n"
    files = {
        "track": track, "nodes": "0 0\n0.001 0\n", "arcs": "0 1\n1 0\n",
        "route": "1\n" if reverse else "0\n",
    }
    for extension, content in files.items():
        (folder / f"{trip_id}.{extension}").write_text(content, encoding="utf-8")
    return folder


class ModelComparisonTest(unittest.TestCase):
    def test_grouped_comparison_and_candidate_miss(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(3):
                write_trip(root, "train", f"{i:08d}", reverse=(i == 1))
            trip = load_trip(write_trip(root, "student_test", "00000099", gap=True))
            features = build_features(trip)
            self.assertEqual(set(features["obs_idx"]), {0, 1, 2})
            forward = features.loc[features["arc_id"] == 0, "heading_diff_deg"]
            reverse = features.loc[features["arc_id"] == 1, "heading_diff_deg"]
            self.assertTrue((forward < 1).all())
            self.assertTrue((reverse > 179).all())

            train, train_truth, excluded = read_partition(root, "train", {})
            test, test_truth, _ = read_partition(root, "student_test", {})
            self.assertEqual(int(test["label"].sum()), 3)
            comparison = compare(train, train_truth, test, test_truth, n_splits=2, jobs=1)
            cv, summary, model, _, _ = comparison
            self.assertEqual(len(summary), 6)
            self.assertEqual(int(summary["selected_by_cv"].sum()), 1)
            self.assertTrue((summary["test_candidate_coverage"] == 0.75).all())
            self.assertTrue((summary["test_top1_accuracy"] <= 0.75).all())
            selected = cv.loc[cv["model"].eq(model), "validation_trips"]
            self.assertEqual(sorted(selected.str.split().explode()), sorted(train["trip_id"].unique()))

            output = root / "results"
            write_results(output, train_truth, test_truth, test, *comparison, {}, excluded)
            self.assertTrue((output / "model_comparison.png").stat().st_size > 0)
            predictions = pd.read_csv(output / "test_predictions.csv", dtype={"trip_id": str})
            self.assertEqual(len(predictions), 4)
            self.assertTrue(pd.isna(predictions.loc[3, "predicted_arc"]))
            trip.pop("route")
            scored = predict_trip(joblib.load(output / "best_model.joblib"), trip)
            self.assertEqual(len(scored), 6)
            self.assertTrue(scored["match_probability"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
