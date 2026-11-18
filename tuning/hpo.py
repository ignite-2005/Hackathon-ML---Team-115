"""Part B: Bayesian hyperparameter optimisation of Person B's selected model (Random forest).

Objective : mean candidate Macro F1 over the same 5 GroupKFold folds (grouped by trip) that
            Person B used, on all 80 train trips (Person B's final setup). student_test is
            never touched here.
Also logged per trial (trial.set_user_attr): mean CV Top-1, mean CV log loss, fit seconds.
Search    : Optuna TPESampler(seed=42); MedianPruner stops a trial early when its running
            mean Macro F1 after a fold is below the median of earlier trials at that fold.
Storage   : sqlite:///tuning/study.db, so a stopped run can be resumed with the same command.
Trial 0 is Person B's untuned configuration, so the study contains the baseline.

Usage (from the repo root):
  python tuning/hpo.py run "<Student folder>" --trials 50     # run / resume the search
  python tuning/hpo.py report                                  # CSV, plots, best_params.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.candidate_features import FEATURE_COLUMNS
from src.train_models import (SEED, evaluate, make_folds, model_candidates, read_partition,
                              score_candidates)

# Speed-up (allowed by the brief): each fold's model is FITTED on a random 25% of the GPS
# points of every training trip (with all candidates of those points), but always SCORED on
# the full validation fold. The final tuned model is refitted on all the data.
FIT_FRACTION = 0.25
STUDY_NAME = "random_forest_macro_f1_fit25"   # the first, full-data study stays in study.db
                                              # as "random_forest_macro_f1" (6 trials, stopped)
STORAGE = f"sqlite:///{(REPO / 'tuning' / 'study.db').as_posix()}"
OUT_DIR = REPO / "outputs" / "tuning"
BEST_PARAMS = REPO / "models" / "best_params.json"
FEATURE_OPTIONS = {"radius_m": 50.0, "max_k": 16, "max_heading_gap_s": 5.0}  # Person B's

# Person B's untuned Random forest (src/train_models.py), queued as trial 0
UNTUNED = {"n_estimators": 64, "max_depth": 14, "min_samples_leaf": 20,
           "max_features": "sqrt", "max_samples": 0.15, "class_weight": None}


def make_model(params):
    """Person B's pipeline (median imputer + Random forest) with these hyperparameters."""
    pipe = model_candidates(jobs=-1)["Random forest"]       # n_jobs=-1 inside the model
    pipe.set_params(**{f"randomforestclassifier__{k}": v for k, v in params.items()},
                    randomforestclassifier__random_state=SEED)
    return pipe


def suggest(trial):
    """The search space. 8 features, so max_features sqrt=2, log2=3, 0.5=4, 0.75=6, 1.0=8."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=10),
        "max_depth": trial.suggest_int("max_depth", 6, 32),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 200, log=True),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.75, 1.0]),
        "max_samples": trial.suggest_float("max_samples", 0.05, 0.4),
        # ~1 positive per candidate set (about 1 in 8 rows): let the search decide on weighting
        "class_weight": trial.suggest_categorical("class_weight",
                                                  [None, "balanced", "balanced_subsample"]),
    }


def point_sample(train, fraction, seed=42):
    """Row mask keeping a random `fraction` of GPS points per trip (all their candidates)."""
    points = train[["trip_id", "obs_idx"]].drop_duplicates()
    keep = points.sample(frac=fraction, random_state=seed)
    return train.set_index(["trip_id", "obs_idx"]).index.isin(
        keep.set_index(["trip_id", "obs_idx"]).index)


def objective_factory(train, folds):
    in_sample = point_sample(train, FIT_FRACTION)
    def objective(trial):
        params = suggest(trial)
        f1s, top1s, losses, fit_s = [], [], [], []
        for k, (fit_mask, val_mask, val_truth) in enumerate(folds):
            fit_mask = fit_mask & in_sample                 # fit on the sample, score on all
            model = make_model(params)
            t0 = time.perf_counter()
            model.fit(train.loc[fit_mask, FEATURE_COLUMNS], train.loc[fit_mask, "label"])
            fit_s.append(time.perf_counter() - t0)
            val = train.loc[val_mask]
            m = evaluate(val, val_truth, score_candidates(model, val), probabilistic=True)
            f1s.append(m["candidate_macro_f1"])
            top1s.append(m["top1_accuracy"])
            losses.append(m["log_loss"])
            trial.report(float(np.mean(f1s)), step=k)        # running mean after this fold
            if trial.should_prune():
                trial.set_user_attr("folds_done", k + 1)
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_top1_mean", float(np.mean(top1s)))
        trial.set_user_attr("cv_log_loss_mean", float(np.mean(losses)))
        trial.set_user_attr("cv_macro_f1_std", float(np.std(f1s, ddof=1)))
        trial.set_user_attr("fit_seconds_mean", float(np.mean(fit_s)))
        trial.set_user_attr("folds_done", len(folds))
        return float(np.mean(f1s))
    return objective


def get_study():
    return optuna.create_study(
        study_name=STUDY_NAME, storage=STORAGE, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))


def run(student_folder, n_trials):
    train, truth, excluded = read_partition(Path(student_folder), "train", FEATURE_OPTIONS)
    assert not excluded and truth["trip_id"].nunique() == 80   # all 80 trips, as Person B
    folds = list(make_folds(train, truth, 5))
    study = get_study()
    if len(study.trials) == 0:
        study.enqueue_trial(UNTUNED)                            # trial 0 = untuned baseline
    done = len([t for t in study.trials if t.state.is_finished()])
    print(f"study has {done} finished trials; running {n_trials} more", flush=True)

    def log(study, trial):
        a = trial.user_attrs
        val = "pruned" if trial.value is None else f"macro F1 {trial.value:.4f}"
        print(f"trial {trial.number:3d}: {val}  top1 {a.get('cv_top1_mean', float('nan')):.4f}  "
              f"folds {a.get('folds_done')}  {trial.duration.total_seconds():.0f}s  "
              f"best so far {study.best_value:.4f}  {trial.params}", flush=True)

    study.optimize(objective_factory(train, folds), n_trials=n_trials, n_jobs=1, callbacks=[log])


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

    study = get_study()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv(OUT_DIR / "hpo_trials.csv", index=False)

    for name, fn in (("optimization_history", plot_optimization_history),
                     ("param_importances", plot_param_importances)):
        ax = fn(study)
        ax.figure.set_size_inches(9, 5)
        ax.figure.tight_layout()
        ax.figure.savefig(OUT_DIR / f"{name}.png", dpi=150)
        plt.close(ax.figure)

    best = study.best_trial
    base = study.trials[0]
    BEST_PARAMS.parent.mkdir(exist_ok=True)
    BEST_PARAMS.write_text(json.dumps({
        "model": "Random forest", "params": best.params, "trial": best.number,
        "cv_macro_f1_mean": best.value, **best.user_attrs,
        "objective": "mean 5-fold GroupKFold candidate macro F1, all 80 train trips "
                     f"(fold models fitted on {int(FIT_FRACTION * 100)}% of GPS points per trip, "
                     "scored on the full validation fold)",
        "sampler": "TPESampler(seed=42)", "pruner": "MedianPruner(n_startup_trials=5, n_warmup_steps=1)",
        "n_trials": len(df), "n_complete": int((df["state"] == "COMPLETE").sum()),
        "n_pruned": int((df["state"] == "PRUNED").sum()),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"trials: {len(df)} (complete {(df['state'] == 'COMPLETE').sum()}, "
          f"pruned {(df['state'] == 'PRUNED').sum()})")
    print(f"untuned (trial 0): macro F1 {base.value:.4f}, top1 {base.user_attrs['cv_top1_mean']:.4f}")
    print(f"best (trial {best.number}): macro F1 {best.value:.4f}, "
          f"top1 {best.user_attrs['cv_top1_mean']:.4f}, params {best.params}")
    print(f"saved {OUT_DIR.relative_to(REPO)}/hpo_trials.csv, optimization_history.png, "
          f"param_importances.png and {BEST_PARAMS.relative_to(REPO)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("student_folder")
    r.add_argument("--trials", type=int, default=50)
    sub.add_parser("report")
    args = p.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if args.cmd == "run":
        run(args.student_folder, args.trials)
    else:
        report()
