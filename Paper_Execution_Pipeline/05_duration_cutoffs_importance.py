"""Evaluate LightGBM at 30/35/40 min and plot grouped Brier importance."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import t as student_t
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
from bbconfig import cfg  # noqa: E402
VISUALIZATION_CODE = PROJECT / "Visualization_Code"
if str(VISUALIZATION_CODE) not in sys.path:
    sys.path.insert(0, str(VISUALIZATION_CODE))

from plot_style import (  # noqa: E402
    GRID_COLOR,
    GRID_LINESTYLE,
    GRID_LINEWIDTH,
    apply_global_style,
)


PREDICTION_FILE = ROOT / "05_duration_model_library.py"
OPTUNA_FILE = ROOT / "05_duration_model_comparison.py"
ML_DIR = cfg.results_dir / cfg.get("paths.ml")
LONG30_RESULTS = ML_DIR / cfg.get("prediction.outputs.comparison")
OUTPUT = ML_DIR / cfg.get("prediction.outputs.thresholds")
FINAL_FIGURES = cfg.results_dir / cfg.get("paths.figures")

SEED = int(cfg.get("prediction.comparison.seed"))
CUTOFFS = tuple(int(value) for value in cfg.get("probability.cutoff_alternatives_min"))  # Table tab:cutoff
MAIN_CUTOFF = int(cfg.get("probability.duration_cutoff_min"))  # Sec. 4.4
OUTER_FOLDS = int(cfg.get("prediction.comparison.outer_folds"))
INNER_FOLDS = int(cfg.get("prediction.comparison.inner_folds"))
TRIALS = int(cfg.get("prediction.comparison.trials"))
REPEATS = int(cfg.get("prediction.permutation.repeats"))
FAMILY = str(cfg.get("prediction.comparison.selected_family"))
LABEL_THRESHOLD = float(cfg.get("prediction.comparison.label_threshold"))
# code (not in paper): the published cutoff/ablation runs seeded TPE with the family index
# in the six reported families (the model comparison used the eight-family sampler_order).
REPORTED_FAMILIES = tuple(str(v) for v in cfg.get("prediction.comparison.families"))
OOF_TOLERANCE = 1e-12  # comparison tolerance of the stored-OOF reconstruction
OPTIMIZER = None       # 05_duration_model_comparison module (set in main)
FILE_STEM = ""         # output stem built from family, cohort size and cutoff (set in main)
FIGSIZE = (10.0, 7.2)
FEATURES = [
    "Station",
    "ModelCorridor",
    "ItemTaxonomyRoot",
    "Hour",
    "WeekdayCategory",
    "MonthCategory",
    "RainMm",
    "TemperatureC",
]
DISPLAY = {
    "Station": "Station",
    "ModelCorridor": "Corridor",
    "ItemTaxonomyRoot": "Incident type",
    "Hour": "Hour",
    "WeekdayCategory": "Day of week",
    "MonthCategory": "Month",
    "RainMm": "Rainfall",
    "TemperatureC": "Temperature",
}


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


def load_cohort(prediction):
    data, duration, identifiers, source_audit = prediction.load_data()
    cases = pd.read_csv(cfg.path("inputs.incident_cohort"))
    eligible = cases["Eligible"].astype(str).str.lower().isin(["true", "1"])
    eligible_ids = set(
        pd.to_numeric(cases.loc[eligible, "IncidentID"], errors="raise").astype(int)
    )
    selected = identifiers["IncidentID"].astype(int).isin(eligible_ids).to_numpy()
    data = data.loc[selected].reset_index(drop=True)
    identifiers = identifiers.loc[selected].reset_index(drop=True)
    duration = np.asarray(duration, dtype=float)[selected]
    counts = {int(k): int(v) for k, v in cfg.get("probability.duration_counts").items()}
    if len(data) != sum(counts.values()):
        raise AssertionError(f"unexpected operational cohort: N={len(data)} (probability.duration_counts)")
    expected = {int(k): int(v) for k, v in cfg.get("prediction.guards.long_counts_cohort")}
    expected.setdefault(MAIN_CUTOFF, counts[max(counts)])
    observed = {cutoff: int((duration >= cutoff).sum()) for cutoff in CUTOFFS}
    if any(observed[c] != expected[c] for c in observed if c in expected):
        raise AssertionError(f"unexpected Long counts: {observed} (expected {expected})")
    missing_features = set(FEATURES) - set(data.columns)
    if missing_features:
        raise AssertionError(f"missing model features: {sorted(missing_features)}")
    return data, duration, identifiers, source_audit


def lightgbm_from_parameters(parameters: dict[str, object], threads: int):
    return OPTIMIZER.make_estimator(FAMILY, parameters, threads)


def stored_long30_selections() -> dict[int, dict[str, object]]:
    rows = pd.read_csv(LONG30_RESULTS / "outer_fold_selections.csv")
    rows = rows[rows["Model"].eq(FAMILY)].copy()
    if len(rows) != OUTER_FOLDS:
        raise ValueError(f"expected {OUTER_FOLDS} stored {FAMILY} outer-fold selections")
    return {
        int(row.Fold): {
            "parameters": json.loads(row.BestParameters),
            "inner_brier": float(row.InnerBrierScore),
        }
        for row in rows.itertuples(index=False)
    }


def evaluate_cutoff(
    prediction,
    optimizer,
    data: pd.DataFrame,
    duration: np.ndarray,
    identifiers: pd.DataFrame,
    cutoff: int,
    threads: int,
):
    target = (duration >= cutoff).astype(int)
    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    splits = list(outer.split(data, target))
    reference = np.full(len(target), np.nan)
    probability = np.full(len(target), np.nan)
    fold_assignment = np.zeros(len(target), dtype=int)
    selection_rows: list[dict[str, object]] = []
    permutation_rows: list[dict[str, object]] = []
    stored = stored_long30_selections() if cutoff == MAIN_CUTOFF else None

    for fold, (train, test) in enumerate(splits, 1):
        reference[test] = target[train].mean()
        if stored is not None:
            parameters = stored[fold]["parameters"]
            inner_brier = stored[fold]["inner_brier"]
        else:
            estimator, study = optimizer.tune_fold(
                prediction,
                data,
                target,
                train,
                FAMILY,
                fold,
                INNER_FOLDS,
                TRIALS,
                threads,
                REPORTED_FAMILIES,
            )
            parameters = dict(study.best_params)
            inner_brier = float(study.best_value)
            if estimator.get_params()["n_estimators"] != parameters["n_estimators"]:
                raise AssertionError("tuned LightGBM parameters were not reconstructed")

        estimator = lightgbm_from_parameters(parameters, threads)
        model = prediction.build_pipeline(
            optimizer.candidate(prediction, FAMILY, estimator)
        )
        model.fit(data.iloc[train], target[train])
        test_data = data.iloc[test].reset_index(drop=True)
        test_target = target[test]
        fold_probability = model.predict_proba(test_data)[:, 1]
        probability[test] = fold_probability
        fold_assignment[test] = fold
        selection_rows.append({
            "CutoffMinutes": cutoff,
            "OuterFold": fold,
            "InnerBrierScore": inner_brier,
            "BestParameters": json.dumps(parameters, ensure_ascii=False),
        })

        if cutoff == MAIN_CUTOFF:
            baseline_brier = brier_score_loss(test_target, fold_probability)
            for feature_position, feature in enumerate(FEATURES):
                rng = np.random.default_rng(
                    SEED + fold * 10_000 + feature_position * 100
                )
                values = test_data[feature].to_numpy(copy=True)
                for repeat in range(1, REPEATS + 1):
                    permuted = test_data.copy()
                    permuted[feature] = rng.permutation(values)
                    permuted_probability = model.predict_proba(permuted)[:, 1]
                    permuted_brier = brier_score_loss(
                        test_target, permuted_probability
                    )
                    permutation_rows.append({
                        "OuterFold": fold,
                        "Feature": feature,
                        "DisplayFeature": DISPLAY[feature],
                        "Repeat": repeat,
                        "BaselineBrierScore": baseline_brier,
                        "PermutedBrierScore": permuted_brier,
                        "BrierScoreIncrease": permuted_brier - baseline_brier,
                    })
        print(
            f"{FAMILY} cutoff={cutoff} fold={fold}/{OUTER_FOLDS} "
            f"inner_Brier={inner_brier:.6f}",
            flush=True,
        )

    if np.isnan(probability).any():
        raise RuntimeError(f"incomplete OOF probabilities for cutoff {cutoff}")
    if cutoff == MAIN_CUTOFF:
        stored_oof = pd.read_csv(
            LONG30_RESULTS / f"oof_{FAMILY.lower().replace(' ', '_')}.csv").sort_values(
            "IncidentID"
        )
        reconstructed = pd.DataFrame({
            "IncidentID": identifiers["IncidentID"].astype(int),
            "Probability": probability,
        }).sort_values("IncidentID")
        difference = float(np.max(np.abs(
            reconstructed["Probability"].to_numpy(float)
            - stored_oof["Probability"].to_numpy(float)
        )))
        if difference > OOF_TOLERANCE:
            raise AssertionError(
                f"Long30 reconstructed OOF probabilities differ by {difference}"
            )

    metrics = prediction.probability_metrics(target, probability, reference)
    label = (probability >= LABEL_THRESHOLD).astype(int)
    summary = {
        "CutoffMinutes": cutoff,
        "N": len(target),
        "Long": int(target.sum()),
        "Short": int((target == 0).sum()),
        **metrics,
        "F1": float(f1_score(target, label, zero_division=0)),
        "Precision": float(precision_score(target, label, zero_division=0)),
        "Recall": float(recall_score(target, label, zero_division=0)),
        "Accuracy": float(accuracy_score(target, label)),
    }
    oof = identifiers[["IncidentID", "Station"]].copy()
    oof["CutoffMinutes"] = cutoff
    oof["OuterFold"] = fold_assignment
    oof["Target"] = target
    oof["Probability"] = probability
    oof["ReferenceProbability"] = reference
    return summary, oof, selection_rows, permutation_rows


def summarize_permutation(raw: pd.DataFrame) -> pd.DataFrame:
    fold_values = (
        raw.groupby(["Feature", "DisplayFeature", "OuterFold"], as_index=False)
        ["BrierScoreIncrease"]
        .mean()
    )
    summary = (
        fold_values.groupby(["Feature", "DisplayFeature"], as_index=False)
        .agg(
            BrierScoreIncreaseMean=("BrierScoreIncrease", "mean"),
            BrierScoreIncreaseFoldSD=("BrierScoreIncrease", "std"),
            OuterFolds=("OuterFold", "nunique"),
        )
    )
    # 95% t interval across outer folds; t rounded to 3 decimals as in the paper run (2.776 for 5 folds).
    t_value = np.round(student_t.ppf(0.975, summary["OuterFolds"] - 1), 3)
    summary["BrierScoreIncreaseCI95"] = (
        t_value
        * summary["BrierScoreIncreaseFoldSD"]
        / np.sqrt(summary["OuterFolds"])
    )
    return summary.sort_values(
        "BrierScoreIncreaseMean", ascending=False
    ).reset_index(drop=True)


def plot_permutation(summary: pd.DataFrame) -> None:
    plot = summary.sort_values("BrierScoreIncreaseMean", ascending=True)
    colors = np.where(
        plot["BrierScoreIncreaseMean"].ge(0), "#2F6DAE", "#A7A7A7"
    )
    fig, axis = plt.subplots(figsize=FIGSIZE)
    axis.barh(
        plot["DisplayFeature"],
        plot["BrierScoreIncreaseMean"],
        xerr=plot["BrierScoreIncreaseCI95"],
        color=colors,
        edgecolor="#333333",
        linewidth=0.75,
        error_kw={"ecolor": "#333333", "elinewidth": 1.2, "capsize": 3},
    )
    axis.axvline(0, color="#333333", linewidth=0.9)
    axis.set_xlabel("Increase in Brier score after permutation")
    axis.set_ylabel("Feature")
    axis.set_axisbelow(True)
    axis.grid(
        axis="x",
        linestyle=GRID_LINESTYLE,
        linewidth=GRID_LINEWIDTH,
        color=GRID_COLOR,
        alpha=0.9,
    )
    axis.grid(False, axis="y")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.14, top=0.97)
    figure_stem = FILE_STEM
    with plt.rc_context({"savefig.bbox": None}):
        for directory, prefix in ((OUTPUT, ""), (FINAL_FIGURES, "02_ml__")):
            directory.mkdir(parents=True, exist_ok=True)
            fig.savefig(directory / f"{prefix}{figure_stem}.png", dpi=300)
    plt.close(fig)


def main() -> int:
    global OPTIMIZER, FILE_STEM
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FINAL_FIGURES.mkdir(parents=True, exist_ok=True)
    apply_global_style()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    prediction = load_module("lightgbm303_prediction", PREDICTION_FILE)
    optimizer = load_module("lightgbm303_optimizer", OPTUNA_FILE)
    OPTIMIZER = optimizer
    data, duration, identifiers, source_audit = load_cohort(prediction)
    n = len(data)
    FILE_STEM = f"grouped_permutation_importance_{FAMILY.lower()}_n{n}_long{MAIN_CUTOFF}"

    summaries = []
    oof_rows = []
    selection_rows: list[dict[str, object]] = []
    permutation_rows: list[dict[str, object]] = []
    for cutoff in CUTOFFS:
        summary, oof, selections, permutation = evaluate_cutoff(
            prediction,
            optimizer,
            data,
            duration,
            identifiers,
            cutoff,
            args.threads,
        )
        summaries.append(summary)
        oof_rows.append(oof)
        selection_rows.extend(selections)
        permutation_rows.extend(permutation)

    summary = pd.DataFrame(summaries)
    summary.to_csv(
        OUTPUT / f"{FAMILY.lower()}_{n}_threshold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(oof_rows, ignore_index=True).to_csv(
        OUTPUT / f"{FAMILY.lower()}_{n}_threshold_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(selection_rows).to_csv(
        OUTPUT / f"{FAMILY.lower()}_{n}_threshold_outer_selections.csv",
        index=False,
        encoding="utf-8-sig",
    )

    raw = pd.DataFrame(permutation_rows)
    permutation_summary = summarize_permutation(raw)
    raw.to_csv(
        OUTPUT / f"{FILE_STEM}_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    permutation_summary.to_csv(
        OUTPUT / f"{FILE_STEM}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_permutation(permutation_summary)

    audit = {
        "schema": "lightgbm-303-thresholds-grouped-permutation-v1",
        "N": n,
        "cutoffs": list(CUTOFFS),
        "model": FAMILY,
        "features": FEATURES,
        "excluded_features": ["FirstLine", "FirstDirection"],
        "outer_cv": f"StratifiedKFold({OUTER_FOLDS}, shuffle=True, random_state={SEED})",
        "inner_cv": f"StratifiedKFold({INNER_FOLDS}, shuffle=True, random_state={SEED}+outer_fold)",
        "trials_per_outer_fold": TRIALS,
        "selection_metric": "minimum inner-OOF Brier score",
        "long30_parameters": "reused from the stored LightGBM nested Optuna run",
        "long35_long40_parameters": "independently Optuna-tuned by cutoff",
        "permutation_cutoff_minutes": MAIN_CUTOFF,
        "permutation_unit": "one original pre-one-hot feature column",
        "permutation_importance": (
            "permuted outer-test-fold Brier score minus baseline Brier score"
        ),
        "permutation_repeats_per_feature_per_fold": REPEATS,
        "permutation_error_bars": (
            f"95% t interval across {OUTER_FOLDS} outer-fold mean importances"
        ),
        "source_audit": source_audit,
    }
    (OUTPUT / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary[[
        "CutoffMinutes", "N", "Long", "Short", "BrierScore", "LogLoss",
        "ROC_AUC", "PR_AUC", "ECE10", "F1", "Precision", "Recall",
        "Accuracy",
    ]].to_string(index=False), flush=True)
    print(permutation_summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
