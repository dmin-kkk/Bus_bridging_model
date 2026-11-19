"""Nested-CV LightGBM using incident type as its only predictor."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from bbconfig import cfg  # noqa: E402

PREDICTION_FILE = ROOT / "05_duration_model_library.py"
OPTIMIZER_FILE = ROOT / "05_duration_model_comparison.py"
ML_DIR = cfg.results_dir / cfg.get("paths.ml")
ABLATION_RESULTS = ML_DIR / cfg.get("prediction.outputs.no_incident_type")
OUTPUT = ML_DIR / cfg.get("prediction.outputs.incident_type_only")
SEED = int(cfg.get("prediction.comparison.seed"))
OUTER_FOLDS = int(cfg.get("prediction.comparison.outer_folds"))
INNER_FOLDS = int(cfg.get("prediction.comparison.inner_folds"))
TRIALS = int(cfg.get("prediction.comparison.trials"))
FAMILY = str(cfg.get("prediction.comparison.selected_family"))
LABEL_THRESHOLD = float(cfg.get("prediction.comparison.label_threshold"))
# code (not in paper): the published cutoff/ablation runs seeded TPE with the family index
# in the six reported families (the model comparison used the eight-family sampler_order).
REPORTED_FAMILIES = tuple(str(v) for v in cfg.get("prediction.comparison.families"))
INCIDENT_TYPE_FEATURE = "ItemTaxonomyRoot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=int(cfg.get("parallel.ml_threads")))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    prediction = load_module("type_only_prediction", PREDICTION_FILE)
    optimizer = load_module("type_only_optimizer", OPTIMIZER_FILE)
    data, target, identifiers, source_audit = optimizer.load_operational_cohort(
        prediction
    )
    prediction.FEATURE_SETS["full"] = {
        "categorical": [INCIDENT_TYPE_FEATURE],
        "numeric": [],
    }
    data = data[[INCIDENT_TYPE_FEATURE]].copy()

    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    outer_splits = list(outer.split(data, target))
    probability = np.full(len(target), np.nan)
    reference = np.full(len(target), np.nan)
    fold_assignment = np.zeros(len(target), dtype=int)
    selection_rows = []

    for fold, (outer_train, outer_test) in enumerate(outer_splits, 1):
        estimator, study = optimizer.tune_fold(
            prediction,
            data,
            target,
            outer_train,
            FAMILY,
            fold,
            INNER_FOLDS,
            TRIALS,
            args.threads,
            REPORTED_FAMILIES,
        )
        model = prediction.build_pipeline(
            optimizer.candidate(prediction, FAMILY, estimator)
        )
        model.fit(data.iloc[outer_train], target[outer_train])
        probability[outer_test] = model.predict_proba(data.iloc[outer_test])[:, 1]
        reference[outer_test] = target[outer_train].mean()
        fold_assignment[outer_test] = fold
        selection_rows.append(
            {
                "OuterFold": fold,
                "InnerBrierScore": float(study.best_value),
                "BestParameters": json.dumps(
                    study.best_params, ensure_ascii=False
                ),
                "CompletedTrials": len(study.trials),
            }
        )
        print(
            f"Incident type only: fold {fold}/{OUTER_FOLDS}, "
            f"inner Brier={study.best_value:.6f}",
            flush=True,
        )

    if np.isnan(probability).any():
        raise RuntimeError("incomplete outer-fold OOF probabilities")
    metrics = prediction.probability_metrics(target, probability, reference)
    labels = (probability >= LABEL_THRESHOLD).astype(int)
    type_only = {
        "Specification": "Incident type only",
        "N": len(target),
        "Long": int(target.sum()),
        "Short": int((target == 0).sum()),
        **metrics,
        "F1": float(f1_score(target, labels, zero_division=0)),
        "Precision": float(precision_score(target, labels, zero_division=0)),
        "Recall": float(recall_score(target, labels, zero_division=0)),
        "Accuracy": float(accuracy_score(target, labels)),
    }

    comparison = pd.read_csv(
        ABLATION_RESULTS / "lightgbm_incident_type_ablation_metrics.csv"
    )
    comparison = pd.concat(
        [comparison, pd.DataFrame([type_only])[comparison.columns]],
        ignore_index=True,
    )
    comparison.to_csv(
        OUTPUT / "lightgbm_incident_type_three_way_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    oof = identifiers.copy()
    oof["OuterFold"] = fold_assignment
    oof["Target"] = target
    oof["Probability"] = probability
    oof["ReferenceProbability"] = reference
    oof.to_csv(
        OUTPUT / "lightgbm_incident_type_only_oof.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(selection_rows).to_csv(
        OUTPUT / "lightgbm_incident_type_only_outer_selections.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit = {
        "schema": "lightgbm-incident-type-only-v1",
        "N": len(target),
        "Long": int(target.sum()),
        "Short": int((target == 0).sum()),
        "model": FAMILY,
        "retained_features": [INCIDENT_TYPE_FEATURE],
        "row_exclusions": 0,
        "outer_cv": f"StratifiedKFold({OUTER_FOLDS}, shuffle=True, random_state={SEED})",
        "inner_cv": f"StratifiedKFold({INNER_FOLDS}, shuffle=True, random_state={SEED}+outer_fold)",
        "trials_per_outer_fold": TRIALS,
        "selection_metric": "minimum inner-OOF Brier score",
        "reported_performance": "pooled outer-fold OOF only",
        "source_audit": source_audit,
    }
    (OUTPUT / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(comparison.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
