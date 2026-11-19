"""Train and evaluate the paper's long-disruption probability model.

The script is intentionally self-contained.  TripID is used only to define
the regular 8xxx passenger-service cohort and is removed before model fitting.
The Long30, Long35, and Long40 outcomes are compared with fully nested
stratified CV. Inner-CV tuning minimizes Brier score; the outer CV is used
only for evaluation. No classification threshold is tuned or used.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bbconfig import cfg  # noqa: E402

warnings.filterwarnings(
    "ignore",
    message=(
        "X does not have valid feature names, but LGBMClassifier was fitted "
        "with feature names"
    ),
    category=UserWarning,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (
    cfg.results_dir / cfg.get("paths.ml") / cfg.get("prediction.outputs.legacy_cutoffs")
)
SEED = int(cfg.get("prediction.legacy_cutoffs.seed"))
# Guards (paper values; 0 disables): rows of inputs.incidents_gt10 and regular incidents.
EXPECTED_SOURCE_N = int(cfg.get("prediction.guards.source_rows"))
EXPECTED_COHORT_N = int(cfg.get("clustering.guards.regular_incidents"))
PREDICTION_CUTOFFS = tuple(int(value) for value in cfg.get("probability.cutoff_alternatives_min"))
PREDICTION_LONG_COUNTS = {
    int(key): int(value) for key, value in cfg.get("probability.long_counts_by_cutoff").items()
}
REGULAR_TRIP_PATTERN = str(cfg.get("prediction.cohort.regular_trip_pattern"))
PEAK_HOURS = [int(value) for value in cfg.get("prediction.cohort.peak_hours")]
OUTPUT_PREFIX = "probability_tuned_" + "_".join(str(value) for value in PREDICTION_CUTOFFS)

FEATURE_SETS: dict[str, dict[str, list[str]]] = {
    "full": {
        "categorical": [
            "Station", "ModelCorridor", "ItemTaxonomyRoot", "Hour",
            "WeekdayCategory", "MonthCategory",
        ],
        "numeric": ["RainMm", "TemperatureC"],
    },
    "no_station": {
        "categorical": [
            "ModelCorridor", "ItemTaxonomyRoot", "Hour",
            "WeekdayCategory", "MonthCategory",
        ],
        "numeric": ["RainMm", "TemperatureC"],
    },
    "compact": {
        "categorical": [
            "Station", "ModelCorridor", "ItemTaxonomyRoot", "MonthCategory",
        ],
        "numeric": ["IsPeak", "IsWeekend", "RainMm", "TemperatureC"],
    },
    "geo_tax_time": {
        "categorical": [
            "Station", "ModelCorridor", "ItemTaxonomyRoot", "Hour",
        ],
        "numeric": ["IsPeak", "IsWeekend", "RainMm", "TemperatureC"],
    },
    "taxonomy_interactions": {
        "categorical": [
            "Station", "ModelCorridor", "ItemTaxonomyRoot", "Hour",
            "MonthCategory", "TaxonomyCorridor", "TaxonomyHour",
        ],
        "numeric": ["IsWeekend", "RainMm", "TemperatureC"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Nested-CV Brier tuning for Long30/Long35/Long40 probability "
            "models."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--outer-folds", type=int, default=int(cfg.get("prediction.legacy_cutoffs.outer_folds")))
    parser.add_argument("--inner-folds", type=int, default=int(cfg.get("prediction.legacy_cutoffs.inner_folds")))
    parser.add_argument("--trees", type=int, default=int(cfg.get("prediction.legacy_cutoffs.trees")))
    parser.add_argument("--threads", type=int, default=int(cfg.get("parallel.ml_threads")))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--rf-final-only",
        action="store_true",
        help=(
            "Skip nested model-family evaluation and fit only the final "
            "expanded Random Forest for each cutoff."
        ),
    )
    return parser.parse_args()


def normalize_category(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    result = str(value).strip()
    return result if result and result.lower() not in {"nan", "none", "null"} else "Unknown"


def normalize_taxonomy(values: pd.Series) -> pd.Series:
    result = values.astype("string").str.strip()
    invalid = result.isna() | result.str.lower().isin(
        ["", "nan", "none", "null", "unknown"]
    )
    return result.mask(invalid).str.replace(r"\s*;\s*", " ; ", regex=True)


def required_paths() -> dict[str, Path]:
    return {
        "incident": cfg.path("inputs.incidents_gt10"),
        "raw": cfg.path("inputs.incidents_raw"),
        "weather": cfg.path("inputs.weather"),
    }


def validate_inputs() -> dict[str, Path]:
    paths = required_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prediction input(s):\n  " + "\n  ".join(missing))
    with paths["incident"].open(encoding="utf-8-sig") as handle:
        source_rows = sum(1 for _ in handle) - 1
    if EXPECTED_SOURCE_N and EXPECTED_SOURCE_N != source_rows:
        raise ValueError(f"Expected {EXPECTED_SOURCE_N} incident rows in {paths['incident'].name}")
    return paths


def select_regular_passenger_incidents(
    incidents: pd.DataFrame, raw_path: Path
) -> pd.DataFrame:
    raw = pd.read_csv(
        raw_path,
        usecols=["ItemId", "TripId", "PrimaryServiceForItem"],
        low_memory=False,
    )
    raw["IncidentID"] = pd.to_numeric(raw["ItemId"], errors="coerce").astype("Int64")
    primary = raw.loc[
        raw["IncidentID"].notna()
        & pd.to_numeric(raw["PrimaryServiceForItem"], errors="coerce").eq(1)
    ].copy()
    primary["IncidentID"] = primary["IncidentID"].astype(int)
    primary["IsRegular8xxx"] = (
        primary["TripId"].astype(str).str.strip().str.fullmatch(REGULAR_TRIP_PATTERN)
    )
    eligible = primary.groupby("IncidentID")["IsRegular8xxx"].all()
    selected = incidents.loc[incidents["IncidentID"].isin(eligible[eligible].index)].copy()
    if EXPECTED_COHORT_N and len(selected) != EXPECTED_COHORT_N:
        raise ValueError(
            f"Expected {EXPECTED_COHORT_N} regular 8xxx incidents, found {len(selected)}"
        )
    return selected


def attach_first_context(data: pd.DataFrame, raw_path: Path, weather_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        raw_path,
        usecols=[
            "ItemId", "ItemCreated", "ItemTypeTaxonomy"
        ],
        low_memory=False,
    )
    raw["SourceRow"] = np.arange(len(raw), dtype=int)
    raw["IncidentID"] = pd.to_numeric(raw["ItemId"], errors="coerce").astype("Int64")
    raw["ItemTaxonomy"] = normalize_taxonomy(raw["ItemTypeTaxonomy"])

    first_context = (
        raw.dropna(subset=["IncidentID"])
        .sort_values("SourceRow", kind="mergesort")
        .drop_duplicates("IncidentID", keep="first")
        [["IncidentID", "ItemCreated"]]
    )
    first_taxonomy = (
        raw.dropna(subset=["IncidentID", "ItemTaxonomy"])
        .sort_values("SourceRow", kind="mergesort")
        .drop_duplicates("IncidentID", keep="first")
        [["IncidentID", "ItemTaxonomy"]]
    )
    first_taxonomy["ItemTaxonomyRoot"] = (
        first_taxonomy["ItemTaxonomy"].str.split(" ; ", n=1).str[0].str.strip()
    )
    first = first_context.merge(
        first_taxonomy[["IncidentID", "ItemTaxonomyRoot"]],
        on="IncidentID",
        how="left",
        validate="one_to_one",
    )
    first["IncidentID"] = first["IncidentID"].astype(int)
    first["ItemTaxonomyRoot"] = first["ItemTaxonomyRoot"].map(normalize_category)
    first["CreatedHour"] = pd.to_datetime(
        first["ItemCreated"], errors="coerce", format="mixed"
    ).dt.floor("h")

    result = data.merge(first, on="IncidentID", how="left", validate="one_to_one")
    fallback = (
        pd.to_datetime(result["EventDate"], errors="raise")
        + pd.to_timedelta(pd.to_numeric(result["Hour"], errors="raise"), unit="h")
    )
    result["CreatedHour"] = result["CreatedHour"].fillna(fallback)
    result["HourNumber"] = result["CreatedHour"].dt.hour.astype(int)
    result["Hour"] = result["HourNumber"].astype(str)
    result["WeekdayCategory"] = result["CreatedHour"].dt.dayofweek.astype(str)
    result["MonthCategory"] = result["CreatedHour"].dt.month.astype(str)
    result["IsWeekend"] = result["CreatedHour"].dt.dayofweek.ge(5).astype(int)
    result["IsPeak"] = result["HourNumber"].isin(PEAK_HOURS).astype(int)

    weather = pd.read_csv(
        weather_path,
        usecols=["WeatherHour", "TemperatureC", "RainMm"],
        low_memory=False,
    )
    weather["CreatedHour"] = pd.to_datetime(
        weather["WeatherHour"], errors="raise"
    ).dt.floor("h")
    weather = weather[["CreatedHour", "TemperatureC", "RainMm"]]
    result = result.merge(weather, on="CreatedHour", how="left", validate="many_to_one")
    if result[["TemperatureC", "RainMm"]].isna().any().any():
        raise ValueError("Missing weather at the incident's first-created hour")
    if result["ItemTaxonomyRoot"].eq("Unknown").any():
        missing = result.loc[result["ItemTaxonomyRoot"].eq("Unknown"), "IncidentID"].tolist()
        raise ValueError(f"Missing first-observed taxonomy for incident(s): {missing[:10]}")

    result["RainMm"] = pd.to_numeric(result["RainMm"], errors="raise")
    result["ModelCorridor"] = result["StationCorridor"].map(normalize_category)
    result["Station"] = result["Station"].map(normalize_category)
    result["TaxonomyCorridor"] = (
        result["ItemTaxonomyRoot"] + "|" + result["ModelCorridor"]
    )
    result["TaxonomyHour"] = result["ItemTaxonomyRoot"] + "|" + result["Hour"]
    return result


def load_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, object]]:
    """Regular-incident population of inputs.incidents_gt10 with first-report context."""
    paths = validate_inputs()
    data = pd.read_csv(paths["incident"], low_memory=False)
    if data["IncidentID"].duplicated().any():
        raise ValueError("IncidentID must be unique in the augmented incident table")
    data = select_regular_passenger_incidents(data, paths["raw"])
    identifiers = data[
        ["IncidentID", "Station", "StationCorridor", "CorridorSource", "CorridorIsImputed"]
    ].copy()
    duration = pd.to_numeric(data.pop("MaxDelayMinutes"), errors="raise")
    duration_array = duration.to_numpy(dtype=float)
    data = data.drop(columns=["TripIDs"], errors="ignore")
    data = attach_first_context(data, paths["raw"], paths["weather"])

    trip_features = [column for column in data.columns if "trip" in column.lower()]
    if trip_features:
        raise ValueError(f"Trip-derived columns remain before fitting: {trip_features}")
    used_features = sorted(
        {
            feature
            for specification in FEATURE_SETS.values()
            for kind in ("categorical", "numeric")
            for feature in specification[kind]
        }
    )
    missing_features = sorted(set(used_features) - set(data.columns))
    if missing_features:
        raise ValueError(f"Missing modeling features: {missing_features}")
    observed_counts = {
        cutoff: int((duration_array >= cutoff).sum())
        for cutoff in PREDICTION_CUTOFFS
    }
    audit = {
        "source_rows": EXPECTED_SOURCE_N,
        "cohort_rows": int(len(data)),
        "targets": {
            f"Long{cutoff}": {
                "long": observed_counts[cutoff],
                "short": int(len(data) - observed_counts[cutoff]),
            }
            for cutoff in PREDICTION_CUTOFFS
        },
        "cohort": "2019 incidents with MaxDelayMinutes > 10 and only primary 8xxx services",
        "target": "Long-c = 1 iff MaxDelayMinutes >= c, c in {30,35,40}",
        "trip_id_policy": "used only for cohort eligibility; excluded from every model feature",
        "taxonomy_policy": "first non-missing ItemTypeTaxonomy in raw source order; root only",
        "line_direction_policy": "Line and Direction are excluded from every model feature",
        "corridor_policy": "preprocessed StationCorridor retained, including documented inference",
    }
    return data, duration_array, identifiers, audit


RF_CANDIDATE_COUNT = int(cfg.get("prediction.legacy_cutoffs.rf_candidates"))


def expanded_random_forest_candidates(
    n_trees: int, threads: int
) -> list[tuple[str, object]]:
    """Return six legacy anchors plus 42 seeded expanded-grid samples."""
    anchors = [
        {
            "n_estimators": n_trees,
            "criterion": "gini",
            "max_features": features,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": leaf,
            "bootstrap": True,
            "class_weight": None,
        }
        for leaf in (1, 3, 7)
        for features in ("sqrt", 0.5)
    ]
    space = {
        "n_estimators": [n_trees, 2 * n_trees],
        "criterion": ["gini", "log_loss"],
        "max_features": ["sqrt", 0.3, 0.5, 0.7],
        "max_depth": [None, 6, 10, 16],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 3, 5, 7, 10],
        "bootstrap": [True, False],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }

    def key(parameters: dict[str, object]) -> tuple[object, ...]:
        return tuple(
            parameters[name]
            for name in (
                "n_estimators", "criterion", "max_features", "max_depth",
                "min_samples_split", "min_samples_leaf", "bootstrap",
                "class_weight",
            )
        )

    selected = list(anchors)
    seen = {key(parameters) for parameters in selected}
    for parameters in ParameterSampler(
        space, n_iter=100, random_state=SEED
    ):
        if key(parameters) in seen:
            continue
        selected.append(parameters)
        seen.add(key(parameters))
        if len(selected) == RF_CANDIDATE_COUNT:
            break
    if len(selected) != RF_CANDIDATE_COUNT:
        raise RuntimeError(
            f"Expected {RF_CANDIDATE_COUNT} RF candidates, got {len(selected)}"
        )

    result = []
    for parameters in selected:
        features = parameters["max_features"]
        feature_label = str(features).replace(".", "p")
        depth = parameters["max_depth"]
        weight = parameters["class_weight"]
        name = (
            f"trees{parameters['n_estimators']}_"
            f"{parameters['criterion']}_feat{feature_label}_"
            f"depth{depth if depth is not None else 'none'}_"
            f"split{parameters['min_samples_split']}_"
            f"leaf{parameters['min_samples_leaf']}_"
            f"boot{int(parameters['bootstrap'])}_"
            f"weight{weight if weight is not None else 'none'}"
        )
        result.append(
            (
                name,
                RandomForestClassifier(
                    **parameters,
                    random_state=SEED,
                    n_jobs=threads,
                ),
            )
        )
    return result


def candidate_models(
    n_trees: int, threads: int
) -> dict[str, list[tuple[str, object]]]:
    """Probability-oriented grids for the eight reported model families."""
    return {
        "Logistic Regression": [
            (
                f"C{c:g}",
                LogisticRegression(C=c, max_iter=4000, random_state=SEED),
            )
            for c in (0.03, 0.1, 0.3, 1.0, 3.0)
        ],
        "KNN": [
            (
                f"k{k}_{weight}",
                KNeighborsClassifier(
                    n_neighbors=k, weights=weight, p=2, n_jobs=threads
                ),
            )
            for k in (7, 15, 25, 35)
            for weight in ("uniform", "distance")
        ],
        "Random Forest": expanded_random_forest_candidates(
            n_trees, threads
        ),
        "GBDT": [
            (
                name,
                GradientBoostingClassifier(
                    n_estimators=trees,
                    learning_rate=rate,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    random_state=SEED,
                ),
            )
            for name, trees, rate, depth, leaf in (
                ("depth1_lr03_leaf5", 200, 0.03, 1, 5),
                ("depth1_lr05_leaf10", 150, 0.05, 1, 10),
                ("depth2_lr03_leaf5", 200, 0.03, 2, 5),
                ("depth2_lr05_leaf10", 150, 0.05, 2, 10),
                ("depth3_lr03_leaf10", 200, 0.03, 3, 10),
            )
        ],
        "Extra Trees": [
            (
                f"leaf{leaf}_{features}",
                ExtraTreesClassifier(
                    n_estimators=n_trees,
                    min_samples_leaf=leaf,
                    max_features=features,
                    random_state=SEED,
                    n_jobs=threads,
                ),
            )
            for leaf in (1, 3, 7)
            for features in ("sqrt", 0.5)
        ],
        "XGBoost": [
            (
                name,
                XGBClassifier(
                    n_estimators=trees,
                    max_depth=depth,
                    learning_rate=rate,
                    min_child_weight=child,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=regularization,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=SEED,
                    n_jobs=threads,
                    verbosity=0,
                ),
            )
            for name, trees, depth, rate, child, regularization in (
                ("depth1_lr03_child3", 240, 1, 0.03, 3, 2.0),
                ("depth2_lr03_child3", 240, 2, 0.03, 3, 2.0),
                ("depth2_lr05_child8", 180, 2, 0.05, 8, 4.0),
                ("depth3_lr03_child3", 240, 3, 0.03, 3, 2.0),
                ("depth3_lr03_child8", 240, 3, 0.03, 8, 4.0),
            )
        ],
        "LightGBM": [
            (
                name,
                LGBMClassifier(
                    n_estimators=trees,
                    learning_rate=rate,
                    num_leaves=leaves,
                    max_depth=depth,
                    min_child_samples=child,
                    reg_lambda=regularization,
                    random_state=SEED,
                    n_jobs=threads,
                    verbosity=-1,
                ),
            )
            for name, trees, rate, leaves, depth, child, regularization in (
                ("leaves5_lr03_child10", 240, 0.03, 5, 3, 10, 2.0),
                ("leaves7_lr03_child20", 240, 0.03, 7, 4, 20, 2.0),
                ("leaves7_lr05_child10", 180, 0.05, 7, 4, 10, 4.0),
                ("leaves15_lr03_child20", 240, 0.03, 15, 5, 20, 4.0),
            )
        ],
        "CatBoost": [
            (
                name,
                CatBoostClassifier(
                    iterations=trees,
                    depth=depth,
                    learning_rate=rate,
                    l2_leaf_reg=regularization,
                    loss_function="Logloss",
                    random_seed=SEED,
                    thread_count=threads,
                    verbose=False,
                    allow_writing_files=False,
                ),
            )
            for name, trees, depth, rate, regularization in (
                ("depth3_lr03_l2_3", 240, 3, 0.03, 3.0),
                ("depth4_lr03_l2_10", 240, 4, 0.03, 10.0),
                ("depth4_lr05_l2_3", 180, 4, 0.05, 3.0),
                ("depth5_lr03_l2_10", 240, 5, 0.03, 10.0),
            )
        ],
    }


def candidate_grid(n_trees: int, threads: int) -> list[dict[str, object]]:
    specification = FEATURE_SETS["full"]
    return [
        {
            "Name": f"{family}__{configuration}",
            "Family": family,
            "Configuration": configuration,
            "FeatureSet": "full",
            "Specification": specification,
            "Estimator": estimator,
        }
        for family, configurations in candidate_models(
            n_trees, threads
        ).items()
        for configuration, estimator in configurations
    ]


def build_pipeline(candidate: dict[str, object]) -> Pipeline:
    specification = candidate["Specification"]
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("categorical", categorical, specification["categorical"]),
            ("numeric", numeric, specification["numeric"]),
        ],
        sparse_threshold=0.0,
    )
    return Pipeline(
        [("preprocessor", preprocessor), ("classifier", clone(candidate["Estimator"]))]
    )


def calibration_bins(
    target: np.ndarray, probability: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    """Return equal-frequency calibration bins used for ECE."""
    order = np.argsort(probability, kind="mergesort")
    groups = np.array_split(order, min(bins, len(order)))
    rows = []
    for number, indices in enumerate(groups, 1):
        if len(indices):
            rows.append(
                {
                    "Bin": number,
                    "Count": len(indices),
                    "MeanProbability": float(probability[indices].mean()),
                    "ObservedRate": float(target[indices].mean()),
                    "AbsoluteGap": float(
                        abs(
                            probability[indices].mean()
                            - target[indices].mean()
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def expected_calibration_error(
    target: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    table = calibration_bins(target, probability, bins)
    return float(
        (table["Count"] / table["Count"].sum() * table["AbsoluteGap"]).sum()
    )


def calibration_coefficients(
    target: np.ndarray, probability: np.ndarray
) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    log_odds = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    if np.ptp(log_odds) < 1e-12:
        return float("nan"), float("nan")
    calibration = LogisticRegression(
        penalty=None, solver="lbfgs", max_iter=4000
    ).fit(log_odds, target)
    return (
        float(calibration.intercept_[0]),
        float(calibration.coef_[0, 0]),
    )


def probability_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    reference_probability: np.ndarray | None = None,
) -> dict[str, float]:
    probability = np.clip(
        np.asarray(probability, dtype=float), 1e-15, 1.0 - 1e-15
    )
    brier = float(brier_score_loss(target, probability))
    intercept, slope = calibration_coefficients(target, probability)
    result = {
        "BrierScore": brier,
        "LogLoss": float(log_loss(target, probability, labels=[0, 1])),
        "ROC_AUC": float(roc_auc_score(target, probability)),
        "PR_AUC": float(average_precision_score(target, probability)),
        "ObservedRate": float(np.mean(target)),
        "MeanPredictedProbability": float(np.mean(probability)),
        "CalibrationBias": float(np.mean(probability) - np.mean(target)),
        "CalibrationIntercept": intercept,
        "CalibrationSlope": slope,
        "ECE10": expected_calibration_error(target, probability),
    }
    if reference_probability is not None:
        reference_brier = float(
            brier_score_loss(target, reference_probability)
        )
        result["ReferenceBrierScore"] = reference_brier
        result["BrierSkillScore"] = float(1.0 - brier / reference_brier)
    return result


def oof_probabilities(
    data: pd.DataFrame,
    target: np.ndarray,
    indices: np.ndarray,
    candidate: dict[str, object],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    probability = np.full(len(indices), np.nan)
    for train_pos, valid_pos in splits:
        train_index = indices[train_pos]
        valid_index = indices[valid_pos]
        model = build_pipeline(candidate)
        model.fit(data.iloc[train_index], target[train_index])
        probability[valid_pos] = model.predict_proba(data.iloc[valid_index])[:, 1]
    if np.isnan(probability).any():
        raise RuntimeError("Incomplete OOF probabilities")
    return probability


def select_configuration(
    data: pd.DataFrame,
    target: np.ndarray,
    outer_train: np.ndarray,
    candidates: list[dict[str, object]],
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, object], dict[str, float], list[dict[str, object]]]:
    rows = []
    best: tuple[tuple[float, float, float, float, str], dict[str, object], dict[str, float]] | None = None
    for candidate in candidates:
        probability = oof_probabilities(
            data, target, outer_train, candidate, inner_splits
        )
        result = probability_metrics(target[outer_train], probability)
        name = str(candidate["Name"])
        rows.append({"Candidate": name, **result})
        key = (
            result["BrierScore"],
            result["LogLoss"],
            -result["ROC_AUC"],
            -result["PR_AUC"],
            name,
        )
        if best is None or key < best[0]:
            best = (key, candidate, result)
    if best is None:
        raise RuntimeError("No model candidate was evaluated")
    return best[1], best[2], rows


def nested_evaluation(
    data: pd.DataFrame,
    duration: np.ndarray,
    identifiers: pd.DataFrame,
    candidates: list[dict[str, object]],
    outer_folds: int,
    inner_folds: int,
) -> tuple[pd.DataFrame, ...]:
    by_family: dict[str, list[dict[str, object]]] = {}
    for candidate in candidates:
        by_family.setdefault(str(candidate["Family"]), []).append(candidate)
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    inner_rows: list[dict[str, object]] = []
    oof_rows: list[pd.DataFrame] = []
    bin_rows: list[pd.DataFrame] = []

    for cutoff in PREDICTION_CUTOFFS:
        target = (duration >= cutoff).astype(int)
        outer = StratifiedKFold(
            outer_folds, shuffle=True, random_state=SEED
        )
        outer_splits = list(outer.split(data, target))
        reference_probability = np.full(len(data), np.nan)
        for train, test in outer_splits:
            reference_probability[test] = target[train].mean()
        print(
            f"Long{cutoff}: N={len(target)}, Long={int(target.sum())}, "
            f"Short={int((target == 0).sum())}",
            flush=True,
        )

        for family, family_candidates in by_family.items():
            probability = np.full(len(data), np.nan)
            fold_assignment = np.zeros(len(data), dtype=int)
            family_fold_rows: list[dict[str, object]] = []
            for fold, (outer_train, outer_test) in enumerate(
                outer_splits, 1
            ):
                inner = StratifiedKFold(
                    inner_folds,
                    shuffle=True,
                    random_state=SEED + fold,
                )
                inner_splits = list(
                    inner.split(outer_train, target[outer_train])
                )
                selected, inner_result, screened = select_configuration(
                    data,
                    target,
                    outer_train,
                    family_candidates,
                    inner_splits,
                )
                model = build_pipeline(selected).fit(
                    data.iloc[outer_train], target[outer_train]
                )
                fold_probability = model.predict_proba(
                    data.iloc[outer_test]
                )[:, 1]
                probability[outer_test] = fold_probability
                fold_assignment[outer_test] = fold
                result = probability_metrics(
                    target[outer_test],
                    fold_probability,
                    reference_probability[outer_test],
                )
                row = {
                    "CutoffMinutes": cutoff,
                    "Model": family,
                    "Fold": fold,
                    "Configuration": selected["Configuration"],
                    **result,
                }
                fold_rows.append(row)
                family_fold_rows.append(row)
                selection_rows.append(
                    {
                        "CutoffMinutes": cutoff,
                        "Model": family,
                        "Fold": fold,
                        "SelectedConfiguration": selected["Configuration"],
                        **{
                            f"Inner{key}": value
                            for key, value in inner_result.items()
                        },
                    }
                )
                for screened_row in screened:
                    inner_rows.append(
                        {
                            "CutoffMinutes": cutoff,
                            "Model": family,
                            "Fold": fold,
                            **screened_row,
                        }
                    )

            if np.isnan(probability).any():
                raise RuntimeError(
                    f"Incomplete outer probability: Long{cutoff}/{family}"
                )
            pooled = probability_metrics(
                target, probability, reference_probability
            )
            family_folds = pd.DataFrame(family_fold_rows)
            summary = {
                "CutoffMinutes": cutoff,
                "Model": family,
                "N": len(target),
                "Long": int(target.sum()),
                "Short": int((target == 0).sum()),
                **pooled,
            }
            for metric_name in (
                "BrierScore", "LogLoss", "ROC_AUC", "PR_AUC", "ECE10"
            ):
                summary[f"FoldMean_{metric_name}"] = float(
                    family_folds[metric_name].mean()
                )
                summary[f"FoldSD_{metric_name}"] = float(
                    family_folds[metric_name].std(ddof=1)
                )
            summary_rows.append(summary)

            model_oof = identifiers.copy()
            model_oof.insert(0, "CutoffMinutes", cutoff)
            model_oof.insert(1, "Model", family)
            model_oof["OuterFold"] = fold_assignment
            model_oof["Target"] = target
            model_oof["Probability"] = probability
            model_oof["ReferenceProbability"] = reference_probability
            oof_rows.append(model_oof)
            bins = calibration_bins(target, probability)
            bins.insert(0, "CutoffMinutes", cutoff)
            bins.insert(1, "Model", family)
            bin_rows.append(bins)
            print(
                f"  {family}: Brier={pooled['BrierScore']:.4f}, "
                f"BSS={pooled['BrierSkillScore']:.3f}, "
                f"ECE={pooled['ECE10']:.3f}, "
                f"AUC={pooled['ROC_AUC']:.3f}",
                flush=True,
            )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["CutoffMinutes", "BrierScore", "LogLoss", "ROC_AUC"],
        ascending=[True, True, True, False],
    )
    summary["BrierRank"] = summary.groupby("CutoffMinutes")[
        "BrierScore"
    ].rank(method="min")
    summary["ECERank"] = summary.groupby("CutoffMinutes")["ECE10"].rank(
        method="min"
    )
    return (
        summary,
        pd.DataFrame(fold_rows),
        pd.DataFrame(selection_rows),
        pd.DataFrame(inner_rows),
        pd.concat(oof_rows, ignore_index=True),
        pd.concat(bin_rows, ignore_index=True),
    )


def fit_final_models(
    data: pd.DataFrame,
    duration: np.ndarray,
    candidates: list[dict[str, object]],
    summary: pd.DataFrame,
    inner_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict[str, object]]]:
    by_family: dict[str, list[dict[str, object]]] = {}
    for candidate in candidates:
        by_family.setdefault(str(candidate["Family"]), []).append(candidate)
    best_rows = []
    final_candidate_rows = []
    model_artifacts: dict[int, dict[str, object]] = {}
    for cutoff in PREDICTION_CUTOFFS:
        winner = summary.loc[
            summary["CutoffMinutes"].eq(cutoff)
        ].sort_values(
            ["BrierScore", "LogLoss", "ROC_AUC"],
            ascending=[True, True, False],
        ).iloc[0]
        family = str(winner["Model"])
        target = (duration >= cutoff).astype(int)
        indices = np.arange(len(data))
        cv = StratifiedKFold(
            inner_folds, shuffle=True, random_state=SEED + 100
        )
        splits = list(cv.split(indices, target))
        selected, result, screened = select_configuration(
            data,
            target,
            indices,
            by_family[family],
            splits,
        )
        model = build_pipeline(selected).fit(data, target)
        artifact = {
            "pipeline": model,
            "cutoff_minutes": cutoff,
            "model_family": family,
            "configuration": selected["Configuration"],
            "feature_set": selected["FeatureSet"],
            "features": selected["Specification"],
            "training_n": len(target),
            "training_long": int(target.sum()),
            "training_prevalence": float(target.mean()),
            "selection_objective": "minimum inner-OOF Brier score",
            "classification_threshold": None,
        }
        model_artifacts[cutoff] = artifact
        best_rows.append(
            {
                **winner.to_dict(),
                "FinalConfiguration": selected["Configuration"],
                "FullDataCV_BrierScore": result["BrierScore"],
                "FullDataCV_LogLoss": result["LogLoss"],
                "FinalModelFile": f"long{cutoff}_probability_model.joblib",
            }
        )
        for row in screened:
            final_candidate_rows.append(
                {
                    "CutoffMinutes": cutoff,
                    "Model": family,
                    **row,
                }
            )
    return (
        pd.DataFrame(best_rows),
        pd.DataFrame(final_candidate_rows),
        model_artifacts,
    )


def fit_expanded_random_forest_models(
    data: pd.DataFrame,
    duration: np.ndarray,
    candidates: list[dict[str, object]],
    inner_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict[str, object]]]:
    """Select and fit a full-data Random Forest for every cutoff."""
    rf_candidates = [
        candidate
        for candidate in candidates
        if candidate["Family"] == "Random Forest"
    ]
    if len(rf_candidates) != RF_CANDIDATE_COUNT:
        raise RuntimeError(
            f"Expected {RF_CANDIDATE_COUNT} Random Forest candidates, "
            f"got {len(rf_candidates)}"
        )

    best_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    artifacts: dict[int, dict[str, object]] = {}
    for cutoff in PREDICTION_CUTOFFS:
        target = (duration >= cutoff).astype(int)
        indices = np.arange(len(data))
        cv = StratifiedKFold(
            inner_folds, shuffle=True, random_state=SEED + 100
        )
        splits = list(cv.split(indices, target))
        selected, result, screened = select_configuration(
            data,
            target,
            indices,
            rf_candidates,
            splits,
        )
        model = build_pipeline(selected).fit(data, target)
        artifacts[cutoff] = {
            "pipeline": model,
            "cutoff_minutes": cutoff,
            "model_family": "Random Forest",
            "configuration": selected["Configuration"],
            "feature_set": selected["FeatureSet"],
            "features": selected["Specification"],
            "training_n": len(target),
            "training_long": int(target.sum()),
            "training_prevalence": float(target.mean()),
            "selection_objective": "minimum full-data CV Brier score",
            "classification_threshold": None,
        }
        best_rows.append(
            {
                "CutoffMinutes": cutoff,
                "Model": "Random Forest",
                "N": len(target),
                "Long": int(target.sum()),
                "Short": int((target == 0).sum()),
                "FinalConfiguration": selected["Configuration"],
                **{f"FullDataCV_{name}": value for name, value in result.items()},
                "FinalModelFile": (
                    f"long{cutoff}_expanded_random_forest_model.joblib"
                ),
            }
        )
        for row in screened:
            candidate_rows.append(
                {
                    "CutoffMinutes": cutoff,
                    "Model": "Random Forest",
                    **row,
                }
            )
        print(
            f"Long{cutoff} final Random Forest: "
            f"{selected['Configuration']} "
            f"(CV Brier={result['BrierScore']:.4f}, "
            f"ECE={result['ECE10']:.4f})",
            flush=True,
        )
    return pd.DataFrame(best_rows), pd.DataFrame(candidate_rows), artifacts


def save_expanded_random_forest_models(
    output_dir: Path,
    best: pd.DataFrame,
    candidates: pd.DataFrame,
    artifacts: dict[int, dict[str, object]],
) -> None:
    for cutoff, artifact in artifacts.items():
        joblib.dump(
            artifact,
            output_dir / f"long{cutoff}_expanded_random_forest_model.joblib",
        )
    best.to_csv(
        output_dir / "expanded_random_forest_best_by_cutoff.csv",
        index=False,
        encoding="utf-8-sig",
    )
    candidates.to_csv(
        output_dir / "expanded_random_forest_final_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    args = parse_args()
    if args.outer_folds < 2 or args.inner_folds < 2:
        raise ValueError("Both CV fold counts must be at least 2")
    if args.trees < 1:
        raise ValueError("--trees must be positive")
    if args.threads == 0 or args.threads < -1:
        raise ValueError("--threads must be -1 or a positive integer")
    data, duration, identifiers, audit = load_data()
    candidates = candidate_grid(args.trees, args.threads)
    counts = {
        cutoff: int((duration >= cutoff).sum())
        for cutoff in PREDICTION_CUTOFFS
    }
    if counts != PREDICTION_LONG_COUNTS:  # guard: probability.long_counts_by_cutoff
        raise ValueError(f"Unexpected long counts {counts} (probability.long_counts_by_cutoff)")
    print(
        f"Preflight OK: N={len(data)}, Long counts={counts}, "
        f"families={len(candidate_models(args.trees, args.threads))}, "
        f"configurations={len(candidates)}"
    )
    if args.check_only:
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.rf_final_only:
        rf_best, rf_candidates, rf_artifacts = (
            fit_expanded_random_forest_models(
                data, duration, candidates, args.inner_folds
            )
        )
        save_expanded_random_forest_models(
            output_dir, rf_best, rf_candidates, rf_artifacts
        )
        return

    summary, folds, selections, inner, oof, bins = nested_evaluation(
        data,
        duration,
        identifiers,
        candidates,
        args.outer_folds,
        args.inner_folds,
    )
    best, final_candidates, artifacts = fit_final_models(
        data, duration, candidates, summary, args.inner_folds
    )
    for cutoff, artifact in artifacts.items():
        joblib.dump(
            artifact,
            output_dir / f"long{cutoff}_probability_model.joblib",
        )
    rf_best, rf_candidates, rf_artifacts = fit_expanded_random_forest_models(
        data, duration, candidates, args.inner_folds
    )
    save_expanded_random_forest_models(
        output_dir, rf_best, rf_candidates, rf_artifacts
    )

    outputs = {
        "summary": output_dir / f"{OUTPUT_PREFIX}_summary.csv",
        "best": output_dir / f"{OUTPUT_PREFIX}_best_by_cutoff.csv",
        "folds": output_dir / f"{OUTPUT_PREFIX}_folds.csv",
        "selections": output_dir / f"{OUTPUT_PREFIX}_selections.csv",
        "inner": output_dir / f"{OUTPUT_PREFIX}_inner_candidates.csv",
        "final": output_dir / f"{OUTPUT_PREFIX}_final_candidates.csv",
        "oof": output_dir / f"{OUTPUT_PREFIX}_oof.csv",
        "bins": output_dir / f"{OUTPUT_PREFIX}_calibration_bins.csv",
    }
    for frame, key in (
        (summary, "summary"),
        (best, "best"),
        (folds, "folds"),
        (selections, "selections"),
        (inner, "inner"),
        (final_candidates, "final"),
        (oof, "oof"),
        (bins, "bins"),
    ):
        frame.to_csv(outputs[key], index=False, encoding="utf-8-sig")

    recommended = best.sort_values(
        ["BrierSkillScore", "ECE10", "ROC_AUC"],
        ascending=[False, True, False],
    ).iloc[0]
    recommendation = {
        "recommended_cutoff_minutes": int(recommended["CutoffMinutes"]),
        "model": str(recommended["Model"]),
        "configuration": str(recommended["FinalConfiguration"]),
        "basis": (
            "highest nested outer-OOF Brier skill score versus each "
            "cutoff's cross-fitted prevalence forecast; ECE10 and ROC-AUC "
            "are tie-breakers"
        ),
        "important_note": (
            "Raw Brier scores are not compared across 30/35/40 because the "
            "target prevalences differ. A rarer target has a mechanically "
            "lower climatology Brier score."
        ),
        "metrics": {
            "BrierScore": float(recommended["BrierScore"]),
            "ReferenceBrierScore": float(
                recommended["ReferenceBrierScore"]
            ),
            "BrierSkillScore": float(recommended["BrierSkillScore"]),
            "ECE10": float(recommended["ECE10"]),
            "LogLoss": float(recommended["LogLoss"]),
            "ROC_AUC": float(recommended["ROC_AUC"]),
            "PR_AUC": float(recommended["PR_AUC"]),
        },
    }
    (output_dir / f"{OUTPUT_PREFIX}_recommendation.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    audit.update(
        {
            "outer_cv": f"StratifiedKFold({args.outer_folds}, shuffle=True, random_state={SEED})",
            "inner_cv": f"StratifiedKFold({args.inner_folds}, shuffle=True, random_state={SEED}+fold)",
            "model_families": list(candidate_models(
                args.trees, args.threads
            )),
            "candidate_count": len(candidates),
            "random_forest_search": {
                "strategy": (
                    "six legacy anchors plus fixed-seed randomized search"
                ),
                "candidate_count": RF_CANDIDATE_COUNT,
                "n_estimators": [args.trees, 2 * args.trees],
                "criterion": ["gini", "log_loss"],
                "max_features": ["sqrt", 0.3, 0.5, 0.7],
                "max_depth": [None, 6, 10, 16],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 3, 5, 7, 10],
                "bootstrap": [True, False],
                "class_weight": [None, "balanced", "balanced_subsample"],
                "fixed_random_state": SEED,
            },
            "selection_metric": (
                "minimum inner-OOF Brier score; ties: log loss, ROC-AUC, "
                "PR-AUC"
            ),
            "threshold_policy": (
                "none; F1, precision, recall and accuracy do not influence "
                "selection"
            ),
            "reported_performance": "pooled outer-fold OOF only",
            "probability_metrics": {
                "BrierScore": "lower is better; proper scoring rule",
                "LogLoss": "lower is better; proper scoring rule",
                "BrierSkillScore": (
                    "higher is better versus outer-train prevalence forecast"
                ),
                "ECE10": "lower is better; 10 equal-frequency bins",
                "CalibrationBias": "ideal 0",
                "CalibrationIntercept": "ideal 0",
                "CalibrationSlope": "ideal 1",
                "ROC_AUC": "higher is better; ranking only",
                "PR_AUC": "higher is better; positive-class ranking",
            },
            "cross_cutoff_rule": recommendation["basis"],
            "cross_cutoff_warning": recommendation["important_note"],
            "recommended_cutoff_minutes": recommendation[
                "recommended_cutoff_minutes"
            ],
        }
    )
    (output_dir / f"{OUTPUT_PREFIX}_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nBest probability model by cutoff")
    print(
        best[
            [
                "CutoffMinutes", "Model", "N", "Long", "Short",
                "BrierScore", "BrierSkillScore", "ECE10", "LogLoss",
                "ROC_AUC", "PR_AUC", "FinalConfiguration",
            ]
        ].to_string(index=False)
    )
    print(
        f"\nRecommended cutoff: {recommendation['recommended_cutoff_minutes']} "
        f"minutes ({recommendation['model']})"
    )


if __name__ == "__main__":
    main()
