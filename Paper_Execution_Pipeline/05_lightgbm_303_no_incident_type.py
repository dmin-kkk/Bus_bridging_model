"""Nested-CV LightGBM ablation excluding only the incident-type feature."""
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

PREDICTION_FILE = ROOT / "05_prediction_model.py"
OPTIMIZER_FILE = ROOT / "05_compare_models_long30_303_optuna.py"
ML_DIR = cfg.results_dir / cfg.get("paths.ml")
BASELINE_RESULTS = ML_DIR / cfg.get("prediction.outputs.comparison")
OUTPUT = ML_DIR / cfg.get("prediction.outputs.no_incident_type")
SEED = int(cfg.get("prediction.comparison.seed"))
OUTER_FOLDS = int(cfg.get("prediction.comparison.outer_folds"))
INNER_FOLDS = int(cfg.get("prediction.comparison.inner_folds"))
TRIALS = int(cfg.get("prediction.comparison.trials"))
FAMILY = str(cfg.get("prediction.comparison.selected_family"))
LABEL_THRESHOLD = float(cfg.get("prediction.comparison.label_threshold"))
# code (not in paper): the published cutoff/ablation runs seeded TPE with the family index
# in the six reported families (the model comparison used the eight-family sampler_order).
REPORTED_FAMILIES = tuple(str(v) for v in cfg.get("prediction.comparison.families"))
EXCLUDED_FEATURE = "ItemTaxonomyRoot"


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

    prediction = load_module("no_type_prediction", PREDICTION_FILE)
    optimizer = load_module("no_type_optimizer", OPTIMIZER_FILE)
    data, target, identifiers, source_audit = optimizer.load_operational_cohort(
        prediction
    )
    full_specification = prediction.FEATURE_SETS["full"]
    ablated_specification = {
        "categorical": [
            feature
            for feature in full_specification["categorical"]
            if feature != EXCLUDED_FEATURE
        ],
        "numeric": list(full_specification["numeric"]),
    }
    prediction.FEATURE_SETS["full"] = ablated_specification
    data = data.drop(columns=[EXCLUDED_FEATURE])

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
            f"No incident type: fold {fold}/{OUTER_FOLDS}, "
            f"inner Brier={study.best_value:.6f}",
            flush=True,
        )

    if np.isnan(probability).any():
        raise RuntimeError("incomplete outer-fold OOF probabilities")
    metrics = prediction.probability_metrics(target, probability, reference)
    labels = (probability >= LABEL_THRESHOLD).astype(int)
    ablated = {
        "Specification": "Without incident type",
        "N": len(target),
        "Long": int(target.sum()),
        "Short": int((target == 0).sum()),
        **metrics,
        "F1": float(f1_score(target, labels, zero_division=0)),
        "Precision": float(precision_score(target, labels, zero_division=0)),
        "Recall": float(recall_score(target, labels, zero_division=0)),
        "Accuracy": float(accuracy_score(target, labels)),
    }

    baseline_frame = pd.read_csv(BASELINE_RESULTS / "model_comparison.csv")
    baseline_row = baseline_frame[baseline_frame["Model"].eq(FAMILY)]
    if len(baseline_row) != 1:
        raise ValueError("expected one baseline LightGBM result")
    baseline = baseline_row.iloc[0].to_dict()
    baseline["Specification"] = "Full model"
    comparison_columns = [
        "Specification", "N", "Long", "Short", "BrierScore", "LogLoss",
        "ROC_AUC", "PR_AUC", "ECE10", "F1", "Precision", "Recall",
        "Accuracy",
    ]
    comparison = pd.DataFrame([baseline, ablated])[comparison_columns]
    comparison.to_csv(
        OUTPUT / "lightgbm_incident_type_ablation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    oof = identifiers.copy()
    oof["OuterFold"] = fold_assignment
    oof["Target"] = target
    oof["Probability"] = probability
    oof["ReferenceProbability"] = reference
    oof.to_csv(
        OUTPUT / "lightgbm_no_incident_type_oof.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(selection_rows).to_csv(
        OUTPUT / "lightgbm_no_incident_type_outer_selections.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit = {
        "schema": "lightgbm-303-incident-type-ablation-v1",
        "N": len(target),
        "Long": int(target.sum()),
        "Short": int((target == 0).sum()),
        "model": FAMILY,
        "retained_features": (
            ablated_specification["categorical"]
            + ablated_specification["numeric"]
        ),
        "excluded_feature": EXCLUDED_FEATURE,
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
