"""Optuna-tuned nested-CV classifier comparison on the operational cohort (Sec. 4.4, App. C).

Families, folds, trials, seeds and search spaces come from ``[prediction.comparison]``,
``[prediction.search_space]`` and ``[prediction.fixed_params]`` of the active profile.
The long class is MaxDelayMinutes >= ``probability.duration_cutoff_min``; the cohort
(``inputs.incident_cohort``, rows with Eligible = true) must match
``probability.duration_counts``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from bbconfig import cfg  # noqa: E402

PREDICTION_FILE = ROOT / "05_duration_model_library.py"
SEED = int(cfg.get("prediction.comparison.seed"))
FAMILIES = tuple(str(value) for value in cfg.get("prediction.comparison.families"))
# Family index used in the TPE sampler seed (the published run evaluated eight families).
SAMPLER_ORDER = tuple(str(value) for value in cfg.get("prediction.comparison.sampler_order"))
# Estimator class, seed keyword and thread keyword per family (code; the values come from config).
ESTIMATORS = {
    "Logistic Regression": (LogisticRegression, "random_state", None),
    "KNN": (KNeighborsClassifier, None, "n_jobs"),
    "Random Forest": (RandomForestClassifier, "random_state", "n_jobs"),
    "GBDT": (GradientBoostingClassifier, "random_state", None),
    "Extra Trees": (ExtraTreesClassifier, "random_state", "n_jobs"),
    "XGBoost": (XGBClassifier, "random_state", "n_jobs"),
    "LightGBM": (LGBMClassifier, "random_state", "n_jobs"),
    "CatBoost": (CatBoostClassifier, "random_seed", "thread_count"),
}


def output_dir() -> Path:
    return cfg.results_dir / cfg.get("paths.ml") / cfg.get("prediction.outputs.comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-folds", type=int, default=int(cfg.get("prediction.comparison.outer_folds")))
    parser.add_argument("--inner-folds", type=int, default=int(cfg.get("prediction.comparison.inner_folds")))
    parser.add_argument("--trials", type=int, default=int(cfg.get("prediction.comparison.trials")))
    parser.add_argument("--threads", type=int, default=int(cfg.get("parallel.ml_threads")))
    parser.add_argument("--output-dir", type=Path, default=output_dir())
    args = parser.parse_args()
    if min(args.outer_folds, args.inner_folds, args.trials, args.threads) < 1:
        parser.error("folds, trials, and threads must be positive")
    args.output_dir = args.output_dir.resolve()
    return args


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def cohort_counts() -> tuple[int, int]:
    """(cohort size, long count) from probability.duration_counts (paper: 303, 95)."""
    counts = {int(key): int(value) for key, value in cfg.get("probability.duration_counts").items()}
    return sum(counts.values()), counts[max(counts)]


def load_operational_cohort(prediction, cutoff: int | None = None):
    """Eligible cohort rows; target = 1 iff MaxDelayMinutes >= cutoff (default: Sec. 4.4 cutoff)."""
    cutoff = int(cfg.get("probability.duration_cutoff_min")) if cutoff is None else int(cutoff)
    data, duration, identifiers, source_audit = prediction.load_data()
    cases = pd.read_csv(cfg.path("inputs.incident_cohort"))
    eligible = cases["Eligible"].astype(str).str.lower().isin(["true", "1"])
    eligible_ids = set(pd.to_numeric(cases.loc[eligible, "IncidentID"]).astype(int))
    selected = identifiers["IncidentID"].astype(int).isin(eligible_ids).to_numpy()
    data = data.loc[selected].reset_index(drop=True)
    identifiers = identifiers.loc[selected].reset_index(drop=True)
    target = (np.asarray(duration)[selected] >= cutoff).astype(int)
    size, long_count = cohort_counts()
    if len(data) != size:
        raise AssertionError(f"unexpected cohort size {len(data)} (probability.duration_counts: {size})")
    if cutoff == int(cfg.get("probability.duration_cutoff_min")) and int(target.sum()) != long_count:
        raise AssertionError(f"unexpected long count {int(target.sum())} (probability.duration_counts: {long_count})")
    guard = {int(c): int(n) for c, n in cfg.get("prediction.guards.long_counts_cohort")}
    if cutoff in guard and int(target.sum()) != guard[cutoff]:
        raise AssertionError(f"Long{cutoff}: {int(target.sum())} != prediction.guards.long_counts_cohort {guard}")
    return data, target, identifiers, source_audit


def _value(value):
    return None if value == "none" else value


def suggest_parameters(trial: optuna.Trial, family: str) -> dict[str, object]:
    """Suggest the family's hyperparameters in the order of [prediction.search_space.<family>]."""
    parameters = {}
    for name, spec in cfg.get(f"prediction.search_space.{family}").items():
        kind = spec["type"]
        if kind == "int":
            parameters[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
        elif kind == "float":
            parameters[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]),
                                                   log=bool(spec.get("log", False)))
        elif kind == "categorical":
            parameters[name] = trial.suggest_categorical(name, [_value(v) for v in spec["choices"]])
        else:
            raise ValueError(f"unknown search-space type {kind!r} for {family}.{name}")
    return parameters


def make_estimator(family: str, parameters: dict[str, object], threads: int):
    estimator_class, seed_keyword, thread_keyword = ESTIMATORS[family]
    keywords = dict(parameters)
    keywords.update(cfg.get(f"prediction.fixed_params.{family}"))
    if seed_keyword:
        keywords[seed_keyword] = SEED
    if thread_keyword:
        keywords[thread_keyword] = threads
    return estimator_class(**keywords)


def suggest_estimator(trial: optuna.Trial, family: str, threads: int):
    return make_estimator(family, suggest_parameters(trial, family), threads)


def candidate(prediction, family: str, estimator) -> dict[str, object]:
    return {
        "Name": family,
        "Family": family,
        "Configuration": family,
        "FeatureSet": "full",
        "Specification": prediction.FEATURE_SETS["full"],
        "Estimator": estimator,
    }


def tune_fold(
    prediction,
    data: pd.DataFrame,
    target: np.ndarray,
    outer_train: np.ndarray,
    family: str,
    fold: int,
    inner_folds: int,
    trials: int,
    threads: int,
    sampler_order: tuple[str, ...] = SAMPLER_ORDER,
):
    inner = StratifiedKFold(inner_folds, shuffle=True, random_state=SEED + fold)
    splits = list(inner.split(outer_train, target[outer_train]))

    def objective(trial: optuna.Trial) -> float:
        estimator = suggest_estimator(trial, family, threads)
        model_candidate = candidate(prediction, family, estimator)
        probability = prediction.oof_probabilities(
            data, target, outer_train, model_candidate, splits
        )
        return float(np.mean((target[outer_train] - probability) ** 2))

    sampler = optuna.samplers.TPESampler(seed=SEED + 1000 * fold + sampler_order.index(family))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    estimator = suggest_estimator(study.best_trial, family, threads)
    return estimator, study


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    threshold = float(cfg.get("prediction.comparison.label_threshold"))
    prediction = load_module("comparison_optuna_prediction", PREDICTION_FILE)
    data, target, identifiers, source_audit = load_operational_cohort(prediction)
    outer = StratifiedKFold(args.outer_folds, shuffle=True, random_state=SEED)
    outer_splits = list(outer.split(data, target))
    reference = np.full(len(target), np.nan)
    for train, test in outer_splits:
        reference[test] = target[train].mean()

    summaries = []
    fold_rows = []
    selection_rows = []
    oof_rows = []
    trial_rows = []
    for family in FAMILIES:
        probability = np.full(len(target), np.nan)
        fold_assignment = np.zeros(len(target), dtype=int)
        for fold, (outer_train, outer_test) in enumerate(outer_splits, 1):
            estimator, study = tune_fold(
                prediction, data, target, outer_train, family, fold,
                args.inner_folds, args.trials, args.threads,
            )
            fitted = prediction.build_pipeline(
                candidate(prediction, family, estimator)
            ).fit(data.iloc[outer_train], target[outer_train])
            fold_probability = fitted.predict_proba(data.iloc[outer_test])[:, 1]
            probability[outer_test] = fold_probability
            fold_assignment[outer_test] = fold
            fold_metric = prediction.probability_metrics(
                target[outer_test], fold_probability, reference[outer_test]
            )
            fold_rows.append({"Model": family, "Fold": fold, **fold_metric})
            selection_rows.append({
                "Model": family,
                "Fold": fold,
                "InnerBrierScore": float(study.best_value),
                "BestParameters": json.dumps(study.best_params, ensure_ascii=False),
                "CompletedTrials": len(study.trials),
            })
            for trial in study.trials:
                trial_rows.append({
                    "Model": family,
                    "Fold": fold,
                    "Trial": trial.number,
                    "State": trial.state.name,
                    "BrierScore": trial.value,
                    "Parameters": json.dumps(trial.params, ensure_ascii=False),
                })
            print(
                f"{family} fold {fold}/{args.outer_folds}: "
                f"inner Brier={study.best_value:.4f}",
                flush=True,
            )
        metrics = prediction.probability_metrics(target, probability, reference)
        labels = (probability >= threshold).astype(int)
        summaries.append({
            "Model": family,
            "N": len(target),
            "Long": int(target.sum()),
            "Short": int((target == 0).sum()),
            **metrics,
            "F1": float(f1_score(target, labels, zero_division=0)),
            "Precision": float(precision_score(target, labels, zero_division=0)),
            "Recall": float(recall_score(target, labels, zero_division=0)),
            "Accuracy": float(accuracy_score(target, labels)),
        })
        family_oof = identifiers.copy()
        family_oof.insert(0, "Model", family)
        family_oof["OuterFold"] = fold_assignment
        family_oof["Target"] = target
        family_oof["Probability"] = probability
        family_oof["ReferenceProbability"] = reference
        oof_rows.append(family_oof)
        pd.DataFrame(summaries).sort_values(
            ["BrierScore", "LogLoss", "ROC_AUC"], ascending=[True, True, False]
        ).to_csv(
            args.output_dir / "model_comparison_checkpoint.csv",
            index=False,
            encoding="utf-8-sig",
        )
        family_oof.to_csv(
            args.output_dir / f"oof_{family.lower().replace(' ', '_')}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(f"{family} complete: outer OOF Brier={metrics['BrierScore']:.4f}", flush=True)

    summary = pd.DataFrame(summaries).sort_values(
        ["BrierScore", "LogLoss", "ROC_AUC"], ascending=[True, True, False]
    )
    summary.to_csv(args.output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(
        args.output_dir / "outer_fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(selection_rows).to_csv(
        args.output_dir / "outer_fold_selections.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(trial_rows).to_csv(
        args.output_dir / "optuna_trials.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(oof_rows, ignore_index=True).to_csv(
        args.output_dir / "nested_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    audit = {
        "schema": "long30-303-optuna-nested-comparison-v1",
        "profile": cfg.name,
        "N": int(len(target)),
        "Long": int(target.sum()),
        "Short": int((target == 0).sum()),
        "cutoff_min": int(cfg.get("probability.duration_cutoff_min")),
        "features": prediction.FEATURE_SETS["full"],
        "excluded_features": ["FirstLine", "FirstDirection"],
        "outer_cv": f"StratifiedKFold({args.outer_folds}, shuffle=True, random_state={SEED})",
        "inner_cv": f"StratifiedKFold({args.inner_folds}, shuffle=True, random_state={SEED}+outer_fold)",
        "optimizer": "Optuna TPESampler",
        "trials_per_family_per_outer_fold": args.trials,
        "selection_metric": "minimum inner-OOF Brier score",
        "reported_performance": "pooled outer-fold OOF only",
        "source_audit": source_audit,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary[[
        "Model", "N", "Long", "Short", "BrierScore", "LogLoss", "ROC_AUC",
        "PR_AUC", "ECE10", "F1", "Precision", "Recall", "Accuracy",
    ]].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
