"""Stage 2 duration probabilities for every retained location-window case (Sec. 4.4, Table 3).

The outer-fold configurations selected by ``05_duration_model_comparison.py`` for
``prediction.full_grid.model_family`` are each refitted on the complete cohort and their
probabilities are averaged (refit ensemble).  A reconstruction check first verifies that the
stored configurations and outer splits reproduce the stored outer-fold OOF probabilities.
Queries: the retained (window, station) cases of the reduction (``paths.clustering``),
the corridor of the station, the fixed incident type, the hour of the window, and the
weekday, month and hourly weather of ``prediction.full_grid.reference_date``.

Profiles
--------
* ``pinned.use_pinned = true`` (paper): the reference-type file of ``[pinned.probabilities]``
  is SHA-256 verified and copied to ``cfg.results_dir / paths.probabilities``.
  ``--recompute`` refits the ensemble for every type in ``prediction.full_grid.incident_types``,
  writes the files to ``recompute_check/`` and diffs the reference type with the pinned file.
* ``pinned.use_pinned = false`` (synthetic): always recomputed; with
  ``prediction.outputs.save_models`` the ensemble is also written (LightGBM text models,
  hyperparameters, feature list) to ``prediction.outputs.model_dir``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
from bbconfig import PACKAGE_ROOT, cfg  # noqa: E402
import network_data  # noqa: E402

PREDICTION_FILE = ROOT / "05_duration_model_library.py"
COMPARISON_FILE = ROOT / "05_duration_model_comparison.py"
SCHEMA = "full-grid-fixed-type-lightgbm-303-refit-ensemble-v1"
OOF_TOLERANCE = 1e-12      # comparison tolerance of the OOF reconstruction check
PINNED_TOLERANCE = 1e-12   # comparison tolerance of the recompute-vs-pinned check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=cfg.results_dir / cfg.get("paths.probabilities"))
    parser.add_argument("--threads", type=int, default=int(cfg.get("parallel.ml_threads")))
    parser.add_argument("--recompute", action="store_true",
                        help="paper profile: refit the ensemble and diff it against the pinned file")
    parser.add_argument(
        "--comparison-dir", type=Path,
        default=cfg.results_dir / cfg.get("paths.ml") / cfg.get("prediction.outputs.comparison"),
        help="folder with outer_fold_selections.csv and oof_<family>.csv from step 05")
    parser.add_argument(
        "--clustering-dir", type=Path, default=cfg.results_dir / cfg.get("paths.clustering"),
        help="folder with the reduction outputs (retained location-window cases)")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.output_root = args.output_root.resolve()
    return args


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def probability_file(root: Path, slug: str) -> Path:
    return root / str(cfg.get("prediction.outputs.probability_file")).format(slug=slug)


def reduction_file(key: str) -> str:
    durations = "_".join(str(value) for value in cfg.get("time.durations_min"))
    return str(cfg.get(f"clustering.files.{key}")).format(k=int(cfg.get("clustering.k_per_window")),
                                                           durations=durations)


def validate_probability_file(path: Path, fixed_incident_type: str, expected_cases: int) -> None:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"window", "station", "p30", "p60", "model_family", "fixed_incident_type", "probability_schema"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if expected_cases and len(frame) != expected_cases:
        raise ValueError(f"{path}: expected {expected_cases} rows, got {len(frame)}")
    if frame[["window", "station"]].duplicated().any():
        raise ValueError(f"{path}: duplicate station-hour rows")
    if set(frame["fixed_incident_type"].astype(str)) != {fixed_incident_type}:
        raise ValueError(f"{path}: fixed incident type mismatch")
    probability = frame[["p30", "p60"]].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError(f"{path}: probability outside [0, 1]")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError(f"{path}: duration probabilities do not sum to one")


def corridor_label_map(data: pd.DataFrame) -> dict[str, str]:
    """Corridor id -> corridor label used by the classifier (ModelCorridor), learned from the
    training incidents whose station lies on the network (must be one-to-one)."""
    groups = {}
    for station in network_data.station_corridors():
        groups[str(station).strip().upper()] = network_data.station_group(station)
    stations = data["Station"].astype(str).str.strip().str.upper()
    known = stations.isin(groups)
    pairs = pd.DataFrame({"corridor": stations[known].map(groups),
                          "label": data.loc[known, "ModelCorridor"].astype(str)}).drop_duplicates()
    if pairs["corridor"].duplicated().any() or pairs["label"].duplicated().any():
        raise ValueError("ModelCorridor labels are not one-to-one with the network corridors:\n"
                         + pairs.to_string(index=False))
    return dict(zip(pairs["corridor"], pairs["label"]))


def retained_cases(clustering_dir: Path) -> pd.DataFrame:
    """Retained (window, station) cases of the reduction, sorted."""
    path = clustering_dir / reduction_file("assignments")
    if not path.is_file():
        raise FileNotFoundError(f"reduction assignments not found (run step 02 first): {path}")
    cases = pd.read_csv(path, usecols=["hour", "station"], encoding="utf-8-sig")
    cases["station"] = cases["station"].astype(str).str.strip().str.upper()
    return cases.drop_duplicates().sort_values(["hour", "station"]).reset_index(drop=True)


def scenario_queries(cases: pd.DataFrame, labels: dict[str, str], fixed_incident_type: str) -> pd.DataFrame:
    weather = pd.read_csv(cfg.path("inputs.weather"), usecols=["WeatherHour", "TemperatureC", "RainMm"],
                          low_memory=False)
    weather["CreatedHour"] = pd.to_datetime(weather["WeatherHour"], errors="raise").dt.floor("h")
    weather = weather.set_index("CreatedHour")
    reference_date = pd.Timestamp(str(cfg.get("prediction.full_grid.reference_date")))
    rows = []
    for hour, station in cases[["hour", "station"]].itertuples(index=False):
        group = network_data.station_group(station)
        if group not in labels:
            raise KeyError(f"no classifier corridor label for {station} ({group})")
        created = reference_date + pd.Timedelta(hours=int(hour))
        if created not in weather.index:
            raise KeyError(f"missing weather for {created}")
        weather_row = weather.loc[created]
        rows.append({
            "window": int(hour), "station": station, "Station": station,
            "ModelCorridor": labels[group],
            "ItemTaxonomyRoot": fixed_incident_type,
            "Hour": str(int(hour)),
            "WeekdayCategory": str(int(created.dayofweek)),
            "MonthCategory": str(int(created.month)),
            "RainMm": float(weather_row["RainMm"]),
            "TemperatureC": float(weather_row["TemperatureC"]),
        })
    return pd.DataFrame(rows)


def save_models(models: list, parameters: list[dict], prediction, family: str) -> Path:
    folder = PACKAGE_ROOT / str(cfg.get("prediction.outputs.model_dir"))
    folder.mkdir(parents=True, exist_ok=True)
    for member, model in enumerate(models, 1):
        # LightGBM's C writer rejects non-ASCII paths; write the native text model from Python.
        text = model.named_steps["classifier"].booster_.model_to_string()
        (folder / f"lightgbm_member{member}.txt").write_text(text, encoding="utf-8")
    preprocessor = models[0].named_steps["preprocessor"]
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scale"]
    write_json(folder / "hyperparameters.json", {
        "profile": cfg.name, "model_family": family,
        "ensemble": "outer-fold selected configurations refitted on the full cohort; arithmetic mean of p_long",
        "members": parameters,
        "fixed_params": cfg.get(f"prediction.fixed_params.{family}"),
        "seed": int(cfg.get("prediction.comparison.seed")),
    })
    write_json(folder / "features.json", {
        "raw_features": prediction.FEATURE_SETS["full"],
        "model_input_features": [str(name) for name in preprocessor.get_feature_names_out()],
        "numeric_scaling": {"mean": [float(v) for v in scaler.mean_], "scale": [float(v) for v in scaler.scale_]},
        "target": f"MaxDelayMinutes >= {int(cfg.get('probability.duration_cutoff_min'))} (long)",
    })
    return folder


def main() -> int:
    args = parse_args()
    use_pinned = bool(cfg.get("pinned.use_pinned"))
    types = {str(slug): str(label) for slug, label in cfg.get("prediction.full_grid.incident_types")}
    reference_slug = str(cfg.get("prediction.full_grid.reference_incident_type"))
    if reference_slug not in types:
        raise ValueError(f"reference_incident_type {reference_slug!r} not in prediction.full_grid.incident_types")
    if use_pinned:
        pinned = cfg.package_path(f"pinned.probabilities.{reference_slug}")
        expected = str(cfg.get(f"pinned.probabilities.{reference_slug}_sha256")).upper()
        observed = sha256(pinned)
        if observed != expected:
            raise RuntimeError(f"SHA-256 mismatch for pinned {pinned.name}: expected {expected}, observed {observed}")
        target = probability_file(args.output_root, reference_slug)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pinned, target)
        print(f"pinned {pinned.name} (SHA-256 verified) -> {target}", flush=True)
        if not args.recompute:
            print("other incident types (sensitivity) are produced with --recompute", flush=True)
            return 0

    family = str(cfg.get("prediction.full_grid.model_family"))
    seed = int(cfg.get("prediction.comparison.seed"))
    outer_folds = int(cfg.get("prediction.comparison.outer_folds"))
    expected_cases = int(cfg.get("clustering.guards.retained_cases"))
    output_root = args.output_root / "recompute_check" if use_pinned else args.output_root
    selection_file = args.comparison_dir / "outer_fold_selections.csv"
    stored_oof_file = args.comparison_dir / f"oof_{family.lower().replace(' ', '_')}.csv"
    for path in (selection_file, stored_oof_file):
        if not path.is_file():
            raise FileNotFoundError(f"step-05 output missing (run 05_duration_model_comparison first): {path}")

    prediction = load_module("full_grid_prediction", PREDICTION_FILE)
    comparison = load_module("full_grid_comparison", COMPARISON_FILE)
    data, target, identifiers, source_audit = comparison.load_operational_cohort(prediction)
    unknown_types = set(types.values()) - set(data["ItemTaxonomyRoot"].astype(str))
    if unknown_types:
        raise ValueError(f"incident types absent from the training cohort: {sorted(unknown_types)}")

    selections = pd.read_csv(selection_file, encoding="utf-8-sig")
    selections = selections[selections["Model"].eq(family)].copy()
    selections["Fold"] = pd.to_numeric(selections["Fold"], errors="raise").astype(int)
    selections = selections.sort_values("Fold")
    if selections["Fold"].tolist() != list(range(1, outer_folds + 1)):
        raise ValueError(f"expected {outer_folds} stored {family} selections in {selection_file}")
    parameters = [json.loads(value) for value in selections["BestParameters"]]

    stored_oof = pd.read_csv(stored_oof_file, encoding="utf-8-sig")
    stored_oof = stored_oof.set_index(pd.to_numeric(stored_oof["IncidentID"], errors="raise").astype(int))
    reconstructed_oof = np.full(len(data), np.nan)
    outer = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
    for (train, test), fold_parameters in zip(outer.split(data, target), parameters):
        estimator = comparison.make_estimator(family, fold_parameters, args.threads)
        model = prediction.build_pipeline(comparison.candidate(prediction, family, estimator)).fit(
            data.iloc[train], target[train])
        reconstructed_oof[test] = model.predict_proba(data.iloc[test])[:, 1]
    stored_probability = stored_oof.loc[identifiers["IncidentID"].astype(int), "Probability"].to_numpy(float)
    max_oof_difference = float(np.max(np.abs(reconstructed_oof - stored_probability)))
    if max_oof_difference > OOF_TOLERANCE:
        raise AssertionError(f"stored {family} OOF reconstruction failed: max_abs_diff={max_oof_difference:.3g}")
    print(f"OOF reconstruction max_abs_diff={max_oof_difference:.3g}", flush=True)

    deployment_models = []
    for fold_parameters in parameters:
        estimator = comparison.make_estimator(family, fold_parameters, args.threads)
        deployment_models.append(prediction.build_pipeline(
            comparison.candidate(prediction, family, estimator)).fit(data, target))

    cases = retained_cases(args.clustering_dir.resolve())
    if expected_cases and len(cases) != expected_cases:
        raise AssertionError(f"{len(cases)} retained cases != clustering.guards.retained_cases {expected_cases}")
    labels = corridor_label_map(data)
    reference_date = str(pd.Timestamp(str(cfg.get("prediction.full_grid.reference_date"))).date())
    file_audits = {}
    for slug, fixed_incident_type in types.items():
        queries = scenario_queries(cases, labels, fixed_incident_type)
        member_probability = np.vstack([model.predict_proba(queries)[:, 1] for model in deployment_models])
        p60 = member_probability.mean(axis=0)
        probabilities = queries[["window", "station", "ModelCorridor"]].copy()
        probabilities["p30"] = 1.0 - p60
        probabilities["p60"] = p60
        probabilities["probability_method"] = (
            f"{len(deployment_models)} selected {family} configurations refitted on all {len(data)} incidents; "
            "arithmetic ensemble mean")
        probabilities["model_family"] = family
        probabilities["training_cohort"] = str(cfg.get("probability.duration_source"))
        probabilities["fixed_incident_type"] = fixed_incident_type
        probabilities["operating_date"] = reference_date
        probabilities["ensemble_members"] = len(deployment_models)
        probabilities["probability_schema"] = SCHEMA
        output_path = probability_file(output_root, slug)
        write_csv(output_path, probabilities)
        validate_probability_file(output_path, fixed_incident_type, expected_cases)
        file_audits[slug] = {"fixed_incident_type": fixed_incident_type, "file": output_path.name,
                             "sha256": sha256(output_path), "p60_min": float(p60.min()), "p60_max": float(p60.max())}
        print(f"FULL_GRID type={fixed_incident_type!r} cases={len(probabilities)} "
              f"p60=[{p60.min():.6f},{p60.max():.6f}]", flush=True)

    audit = {
        "schema": SCHEMA, "profile": cfg.name, "model_family": family,
        "cohort_counts": {"total": int(len(data)), "long": int(target.sum()), "short": int((target == 0).sum())},
        "deployment_fit": f"{len(parameters)} outer-fold-selected configurations, each refitted on the complete "
                          "cohort, then equally averaged",
        "outer_oof_reconstruction_max_abs_diff": max_oof_difference,
        "operating_date": reference_date,
        "selection_file_sha256": sha256(selection_file),
        "stored_oof_file_sha256": sha256(stored_oof_file),
        "source_audit": source_audit,
        "files": file_audits,
    }
    if use_pinned:
        new = pd.read_csv(probability_file(output_root, reference_slug), encoding="utf-8-sig")
        old = pd.read_csv(cfg.package_path(f"pinned.probabilities.{reference_slug}"), encoding="utf-8-sig")
        merged = old.merge(new, on=["window", "station"], how="outer", suffixes=("_pinned", "_recomputed"),
                           indicator=True)
        both = merged[merged["_merge"] == "both"]
        difference = float(np.max(np.abs(both["p60_pinned"] - both["p60_recomputed"]))) if len(both) else float("nan")
        audit["pinned_comparison"] = {
            "rows_pinned": int(len(old)), "rows_recomputed": int(len(new)),
            "unmatched_rows": int((merged["_merge"] != "both").sum()),
            "max_abs_p60_difference": difference,
            "corridor_labels_identical": bool((both["ModelCorridor_pinned"] == both["ModelCorridor_recomputed"]).all()),
            "identical_within_tolerance": bool((merged["_merge"] == "both").all() and difference <= PINNED_TOLERANCE),
        }
        print(f"recompute vs pinned: {audit['pinned_comparison']}", flush=True)
    if bool(cfg.get("prediction.outputs.save_models")):
        folder = save_models(deployment_models, parameters, prediction, family)
        audit["saved_models"] = str(folder.relative_to(PACKAGE_ROOT)).replace("\\", "/")
        print(f"classifier written to {folder}", flush=True)
    write_json(output_root / "probability_bundle_audit.json", audit)
    print(f"PROBABILITY_BUNDLE_COMPLETE root={output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
