"""Aggregate the Stage 1 K and Stage 2 sensitivity summaries (Sec. 6.7).

Windows and cases come from ``[evaluation.sensitivity]`` of the active profile;
results are read from ``cfg.results_dir / paths.sensitivity``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
PACKAGE = SCRIPT.parents[3]
sys.path.insert(0, str(PACKAGE))

from bbconfig import cfg  # noqa: E402


STAGE1_WINDOWS = tuple(int(w) for w in cfg.get("evaluation.sensitivity.stage1_k_windows"))
STAGE2_WINDOWS = tuple(int(w) for w in cfg.get("evaluation.sensitivity.windows"))
BASELINE_CASE = str(cfg.get("evaluation.sensitivity.baseline_case"))
BASE_MIN_TRIP_BOARDING = float(cfg.get("operations.min_trip_boarding"))
CORRECTED_RESULTS = cfg.results_dir / cfg.get("paths.sensitivity")
SUMMARY_DIR = CORRECTED_RESULTS / "summary"
# Clustering-stage output (the paper profile copies the verified pinned file there).
STATION_PROBABILITY_FILE = (
    cfg.results_dir / cfg.get("paths.clustering")
    / str(cfg.get("clustering.files.station_probability"))
)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def stage1_rows(
    results_root: Path, baseline_stage1: Path
) -> pd.DataFrame:
    definitions = {
        BASELINE_CASE: baseline_stage1,
    }
    stage1_root = results_root / "stage1"
    if stage1_root.is_dir():
        definitions.update({
            path.name: path
            for path in stage1_root.iterdir()
            if path.is_dir()
        })
    rows = []
    for case_id, output in definitions.items():
        for window in STAGE1_WINDOWS:
            summary_path = (
                output / "Performance" / f"FullMILP_{window:02d}_summary.json"
            )
            summary = read_json(summary_path)
            if summary is None:
                rows.append(
                    {
                        "case_id": case_id,
                        "window": window,
                        "available": False,
                        "summary_file": str(summary_path.resolve()),
                    }
                )
                continue
            rows.append(
                {
                    "case_id": case_id,
                    "window": window,
                    "available": True,
                    "k": summary.get("k"),
                    "fleet_size": summary.get("depot_total"),
                    "status": summary.get("status"),
                    "has_incumbent": summary.get("has_incumbent"),
                    "certified_within_gap": summary.get("certified_within_gap"),
                    "objective": summary.get("objective"),
                    "bound": summary.get("bound"),
                    "relative_gap": summary.get("relative_gap"),
                    "candidate_macro_columns": summary.get(
                        "candidate_macro_columns"
                    ),
                    "selected_column_count": summary.get("selected_column_count"),
                    "elapsed_seconds": summary.get("total_elapsed_seconds"),
                    "allocation_file": summary.get("allocation_file"),
                    "summary_file": str(summary_path.resolve()),
                }
            )
    frame = pd.DataFrame(rows)
    baseline = frame[frame["case_id"].eq(BASELINE_CASE)][
        ["window", "objective"]
    ].rename(columns={"objective": "baseline_objective"})
    frame = frame.merge(baseline, on="window", how="left")
    frame["objective_pct_change"] = (
        (frame["objective"] - frame["baseline_objective"])
        / frame["baseline_objective"]
        * 100.0
    )
    return frame.sort_values(["case_id", "window"])


def matrix_comparability() -> dict[str, bool]:
    """Objective comparable to the baseline unless the case reweights the objective.

    Cases that change the out-of-vehicle time weight (kappa) change the objective
    coefficients, so their objective values are not compared with the baseline.
    """
    return {
        str(case["id"]).strip().upper(): "out_of_vehicle_weight" not in case
        for case in cfg.get("evaluation.sensitivity.cases")
    }


def stage2_rows(
    results_root: Path, baseline_stage2: Path,
    station_probability_file: Path = STATION_PROBABILITY_FILE,
) -> pd.DataFrame:
    station_probability = pd.read_csv(
        station_probability_file,
        usecols=["hour", "station", "StationProbability"],
    )
    station_probability["station"] = (
        station_probability["station"].astype(str).str.strip().str.upper()
    )
    probability_lookup = station_probability.set_index(
        ["hour", "station"]
    )["StationProbability"]
    roots = {BASELINE_CASE: baseline_stage2}
    stage2_root = results_root / "stage2"
    if stage2_root.is_dir():
        roots.update(
            {
                path.name: path
                for path in stage2_root.iterdir()
                if path.is_dir()
            }
        )
    comparable = matrix_comparability()
    rows = []
    for case_id, root in sorted(roots.items()):
        performance = root / "Performance"
        if not performance.is_dir():
            continue
        for path in sorted(performance.glob("FullMILP_*_summary.json")):
            summary = read_json(path)
            if summary is None or int(summary.get("window", -1)) not in STAGE2_WINDOWS:
                continue
            sensitivity = summary.get("sensitivity") or {}
            window = int(summary["window"])
            station = str(summary.get("station", "")).strip().upper()
            if (window, station) not in probability_lookup.index:
                raise ValueError(
                    f"missing historical station probability for H{window:02d}/{station}"
                )
            rows.append(
                {
                    "case_id": case_id,
                    "window": window,
                    "station": station,
                    "station_probability": float(
                        probability_lookup.loc[(window, station)]
                    ),
                    "status": summary.get("status"),
                    "has_incumbent": summary.get("has_incumbent"),
                    "certified_within_gap": summary.get("certified_within_gap"),
                    "objective": summary.get("objective"),
                    "Z30": summary.get("Z30"),
                    "Z60": summary.get("Z60"),
                    "fleet_size": sensitivity.get(
                        "fleet_size", summary.get("fleet_total")
                    ),
                    "min_trip_boarding": sensitivity.get(
                        "min_trip_boarding", BASE_MIN_TRIP_BOARDING
                    ),
                    "travel_time_multiplier": sensitivity.get(
                        "travel_time_multiplier", 1.0
                    ),
                    "queue_multiplier": sensitivity.get("queue_multiplier", 1.0),
                    "walking_multiplier": sensitivity.get(
                        "walking_multiplier", 1.0
                    ),
                    "candidate_routes": summary.get("candidate_routes"),
                    "candidate_trips": summary.get("candidate_trips"),
                    "selected_policy_rows": summary.get("selected_policy_rows"),
                    "elapsed_seconds": summary.get("elapsed_seconds"),
                    "allocation_file": summary.get("fixed_stage1_allocation"),
                    "summary_file": str(path.resolve()),
                    "objective_comparable": comparable.get(case_id, True),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    baseline = frame[frame["case_id"].eq(BASELINE_CASE)][
        ["window", "station", "objective"]
    ].rename(columns={"objective": "baseline_objective"})
    frame = frame.merge(baseline, on=["window", "station"], how="left")
    frame["objective_pct_change"] = (
        (frame["objective"] - frame["baseline_objective"])
        / frame["baseline_objective"]
        * 100.0
    )
    frame.loc[
        ~frame["objective_comparable"], "objective_pct_change"
    ] = float("nan")
    return frame.sort_values(["case_id", "window", "station"])


def stage2_aggregate(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return detail
    summary = (
        detail.groupby(["case_id", "window"], as_index=False)
        .agg(
            case_count=("station", "size"),
            incumbent_count=("has_incumbent", "sum"),
            certified_count=("certified_within_gap", "sum"),
            mean_objective=("objective", "mean"),
            median_objective=("objective", "median"),
            mean_objective_pct_change=("objective_pct_change", "mean"),
            median_objective_pct_change=("objective_pct_change", "median"),
            mean_selected_policy_rows=("selected_policy_rows", "mean"),
            mean_elapsed_seconds=("elapsed_seconds", "mean"),
        )
        .sort_values(["case_id", "window"])
    )
    weighted_rows = []
    for (case_id, window), group in detail.groupby(["case_id", "window"]):
        weights = group["station_probability"].to_numpy(float)
        objective = group["objective"].to_numpy(float)
        change = group["objective_pct_change"].to_numpy(float)
        valid_objective = np.isfinite(objective) & np.isfinite(weights)
        valid_change = np.isfinite(change) & np.isfinite(weights)
        weighted_rows.append({
            "case_id": case_id,
            "window": window,
            "historical_probability_mass": float(weights.sum()),
            "weighted_mean_objective": float(np.average(
                objective[valid_objective], weights=weights[valid_objective]
            )),
            "weighted_mean_objective_pct_change": (
                float(np.average(change[valid_change], weights=weights[valid_change]))
                if valid_change.any() else float("nan")
            ),
        })
    return summary.merge(
        pd.DataFrame(weighted_rows), on=["case_id", "window"], how="left"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=CORRECTED_RESULTS)
    parser.add_argument(
        "--baseline-stage1", type=Path,
        default=cfg.results_dir / cfg.get("paths.stage1"))
    parser.add_argument(
        "--baseline-stage2", type=Path,
        default=cfg.results_dir / cfg.get("paths.stage2_proposed"))
    parser.add_argument(
        "--station-probability-file", type=Path,
        default=STATION_PROBABILITY_FILE,
        help="hourly station probabilities used as historical weights")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    args.results_root = args.results_root.resolve()
    args.baseline_stage1 = args.baseline_stage1.resolve()
    args.baseline_stage2 = args.baseline_stage2.resolve()
    args.station_probability_file = args.station_probability_file.resolve()
    args.output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.results_root / "summary"
    )
    return args


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1 = stage1_rows(args.results_root, args.baseline_stage1)
    stage2 = stage2_rows(
        args.results_root, args.baseline_stage2, args.station_probability_file)
    aggregate = stage2_aggregate(stage2)
    stage1.to_csv(
        args.output_dir / "stage1_k_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stage2.to_csv(
        args.output_dir / "stage2_case_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate.to_csv(
        args.output_dir / "stage2_summary_by_variant_window.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"Wrote {len(stage1)} Stage1 rows and {len(stage2)} Stage2 rows to "
        f"{args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
