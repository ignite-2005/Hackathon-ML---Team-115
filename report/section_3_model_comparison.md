# 3. Model comparison

## Features and labels

Each row describes one directed road arc near a GPS observation. The default candidate search keeps up to 16 arcs within 50 m. The model uses distance to the arc, directed heading difference, GPS speed, arc length, fractional projection offset, distance rank, candidate count, and distance above the nearest arc. Speed and heading use adjacent steps up to five seconds apart. Heading is missing when movement is stationary or undefined.

Person A's monotonic alignment matches each GPS point to an arc on the reference route. A candidate receives label 1 when it matches that arc. The reference is used for training and evaluation. The feature builder and inference accept only the track and road network.

## Evaluation

Use `train` for five-fold `GroupKFold` grouped by trip. The training set omits trips whose median GPS-to-reference distance exceeds Person A's 50 m label-quality threshold. Fit preprocessing within each fold. Compare all models on the separate `student_test` partition. Select the classifier by mean CV Top-1 accuracy, with mean CV candidate F1 as the tiebreaker.

Candidate F1 uses a 0.5 probability threshold. Candidate log loss uses the positive-class probability. The nearest-arc baseline makes a hard choice, so it has no log loss. Top-1 chooses one arc per observation using the highest score. Ties favor the nearest arc, then the smallest arc ID. Candidate misses and zero-candidate points remain in the Top-1 denominator. Top-1 arc F1 counts a wrong arc as a false positive and its missed reference arc as a false negative. Scoring time is the median of three full test-batch passes and includes the classifier's preprocessing. Candidate generation and feature engineering are outside that timer.

| Model | How it works and why it is included | What to examine |
| --- | --- | --- |
| Nearest arc | Picks the closest candidate and provides a geometry-only baseline. | Parallel roads and opposite arcs can have almost identical distances. |
| Logistic regression | Learns a linear log-odds function with scaled inputs. It gives fast probability estimates. | Heading may matter more at higher speeds, which a linear model without interaction terms may miss. |
| Gaussian naive Bayes | Uses a per-class Gaussian likelihood for each feature and combines the likelihoods. It gives a lightweight generative baseline. | Distance and distance rank are correlated. Compare candidate log loss with ranking accuracy. |
| Random forest | Averages independent trees trained on bootstrap samples, with randomized feature choices. It can learn motion and geometry interactions. | Compare grouped CV variance and scoring cost. |
| Histogram gradient boosting | Fits trees sequentially to reduce classification loss. Histograms make training practical on many candidate rows. | Compare log loss and Top-1 with the forest and check whether better probabilities improve arc picks. |
| Neural network | A small multilayer perceptron learns nonlinear combinations of scaled features. | Check CV variability, convergence warnings, and scoring time. |

Use the generated `section3_results.md` for the measured table and plot. Compare each trip with the nearest-arc baseline and use candidate coverage to explain the best possible Top-1 accuracy under this candidate search.

## Reproduce and hand off

Run from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p test_person_b.py
python -m src.train_models ".\Student folder"
```

The input path should contain `train` and `student_test`. Use `--output` to set a separate result directory for each run. The default is `results/person_b`.

- `model_comparison.csv`, `cv_folds.csv`, `model_comparison.png`, `section3_results.md`, and `test_trip_accuracy.csv` support the report.
- `best_model.joblib` contains the fitted classifier pipeline and feature-generation settings. Load it with the same library versions used for training.
- `test_candidate_probabilities.parquet` contains `trip_id, obs_idx, arc_id, match_probability` for Viterbi. Each value is a binary classifier probability for one arc. `test_predictions.csv` retains points with zero candidates, with a blank predicted arc.
- For API inference, call `src.train_models.predict_trip(bundle, trip)` with the loaded bundle and a trip dictionary containing the road network and track. It returns candidate geometry and probabilities, including the original observation indexes and timestamps.
- `run_metadata.json` records trip IDs, feature settings, and excluded training trips. `cv_folds.csv` records the validation trip IDs for each fold so tuning can use the same groups.
