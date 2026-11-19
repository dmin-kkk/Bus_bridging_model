"""Evaluate operational value of Long30 out-of-fold probabilities.

Each eligible held-out incident is mapped to its observed station and incident
creation hour.  The fixed Stage-1 allocation for that hour is then used in
three Stage-2 policies:

* Historical: the Stage-1 duration prior (probability.duration_counts;
  paper profile P(D30)=208/303 and P(D60)=95/303).
* ML: the Optuna-tuned LightGBM Long30 OOF probability.
* Oracle: the realized D30/D60 state is known at incident onset.

Historical and ML use the selected Stage-2 information structure.  The
default remains bus-only sharing; ``passenger-and-bus`` additionally shares
passenger flows while preserving the original duration-specific disruption
and canceled-train state in each branch.

The realized state is D30 for durations below 30 minutes and D60 otherwise,
which is exactly the target definition used by the probability model.  Cases
with no affected OD in the realized D30 branch are retained as zero-impact
observations.  Stage-2 solutions are cached by station/hour whenever their
probabilities are identical.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


SCRIPT = Path(__file__).resolve()
PIPELINE = SCRIPT.parent
PROJECT = PIPELINE.parent
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PROJECT))

from bbconfig import cfg  # noqa: E402
import network_data  # noqa: E402


RESULTS = cfg.results_dir
# OOF probabilities written by 05_duration_model_comparison.py (ML stage).
OOF_FILE = (
    RESULTS / cfg.get("paths.ml") / cfg.get("prediction.outputs.comparison")
    / cfg.get("evaluation.oof.oof_file")
)
INCIDENT_FILE = cfg.path("inputs.incidents_gt10")
RAW_INCIDENT_FILE = cfg.path("inputs.incidents_raw")
STAGE2_RUNNER_FILE = PIPELINE / "06_stage2_core.py"
EVALUATOR_FILE = PIPELINE / "Sensitivity" / "source" / "evaluate_ri.py"
DEFAULT_OUTPUT = RESULTS / cfg.get("paths.oof")
DEFAULT_ALLOCATION_DIR = RESULTS / cfg.get("paths.stage1") / "Allocation"
DEFAULT_NORMAL_DIR = RESULTS / cfg.get("paths.normal")
DEFAULT_DISRUPTION_DIR = RESULTS / cfg.get("paths.disruptions")
RI_CACHE_DIR = RESULTS / cfg.get("paths.cache") / "ri"
# Stage 2 inputs (same files as 06_stage2_core.py reads).
ROUTE_FILE = (
    RESULTS / cfg.get("paths.routes") / cfg.get("routes.library_file")
)
SCHEDULE_FILE = cfg.path("inputs.timetable")
OD_FILE = cfg.path("inputs.od_matrix")
DISTANCE_FILE = cfg.path("inputs.road_times")
CORRIDOR_FILE = cfg.path("inputs.corridors")
COORDINATE_FILE = cfg.path("inputs.stations")
DEPOT_FILE = cfg.path("inputs.depots")
WINDOWS = tuple(int(hour) for hour in cfg.get("time.windows"))
DURATIONS = tuple(int(value) for value in cfg.get("time.durations_min"))
SHORT, LONG = DURATIONS            # Sec. 3.1: short and long duration states
HORIZON = int(cfg.get("time.horizon_min"))
BUFFER = int(cfg.get("time.clearance_buffer_min"))
FLEET_SIZE = int(cfg.get("operations.fleet_size"))
DEPOT_ACCESS_BOUND = cfg.get("operations.depot_access_bound_min")
MIN_TRIP_BOARDING = float(cfg.get("operations.min_trip_boarding"))
OD_HOURLY_MIN_TRIPS = cfg.get("operations.od_hourly_min_trips")
LOGIC_VERSION = cfg.get("scenarios.logic_version")
OUTER_FOLDS = int(cfg.get("prediction.comparison.outer_folds"))
FOLD_SEED = int(cfg.get("prediction.comparison.seed"))
# Guards (-1 = not checked): incidents inside the planning windows with D60 support,
# and eligible incidents whose realized short branch has no affected OD.
EXPECTED_ELIGIBLE = int(cfg.get("evaluation.oof.expected_eligible_incidents"))
EXPECTED_ZERO_IMPACT = int(cfg.get("evaluation.oof.expected_zero_impact_realized"))
NONANTICIPATIVITY_MODE = "bus-only"
FORMULATION_MODE = "BUS_ONLY_WAIT_AND_SEE"
PASSENGER_RECOURSE = "WAIT_AND_SEE"
SCHEMA = (
    "paper-oof-operational-value-lightgbm-operational303-"
    "bus-only-wait-and-see-v3"
)
INPUT_AUDIT_SCHEMA = "paper-oof-operational303-input-audit-v1"
BASE_STAGE2_SCHEMA = (
    "paper-stage2-full-pool-joint-30-60-bus-only-na-v5"
)
EVALUATION_SCHEMA = (
    "paper-oof-evaluation-physical-duration-support-ri-pt-"
    "t0-demand-denominator-v4"
)
MODEL = cfg.get("prediction.comparison.selected_family")
CUTOFF = int(cfg.get("probability.duration_cutoff_min"))
EVALUATION_START = int(cfg.get("time.evaluation_start_min"))
HISTORICAL_COUNTS = {
    int(duration): int(count)
    for duration, count in cfg.get("probability.duration_counts").items()
}
HISTORICAL_WEIGHTS = {
    duration: count / sum(HISTORICAL_COUNTS.values())
    for duration, count in HISTORICAL_COUNTS.items()
}
COHORT_SIZE = sum(HISTORICAL_COUNTS.values())
PROBABILITY_SOURCE = cfg.get("probability.duration_source")
BUS_ONLY_NONANTICIPATIVITY = "bus-only"
PASSENGER_AND_BUS_NONANTICIPATIVITY = "passenger-and-bus"
PASSENGER_AND_BUS_FORMULATION_MODE = (
    "PASSENGER_AND_BUS_SHARED_RAW_DURATION_RAIL"
)
PASSENGER_AND_BUS_STAGE2_SCHEMA = (
    "paper-stage2-full-pool-joint-30-60-passenger-bus-na-raw-rail-v5"
)
PASSENGER_AND_BUS_OOF_SCHEMA = (
    "paper-oof-operational-value-lightgbm-operational303-"
    "passenger-and-bus-shared-raw-rail-v5"
)
PASSENGER_AND_BUS_EVALUATION_SCHEMA = (
    "paper-oof-evaluation-common-support-ri-physical-pt-"
    "raw-rail-t0-demand-denominator-v6"
)
RAW_RAIL_WITHDRAWAL_POLICY = (
    "ZERO_FROM_RAW_BRANCH_PHYSICAL_CLEARANCE_EVENT"
)


def activate_nonanticipativity_mode(mode: str) -> None:
    """Select wrapper provenance without changing the Stage-2 core."""
    global NONANTICIPATIVITY_MODE, FORMULATION_MODE, PASSENGER_RECOURSE
    global SCHEMA, BASE_STAGE2_SCHEMA, EVALUATION_SCHEMA
    NONANTICIPATIVITY_MODE = mode
    if mode == PASSENGER_AND_BUS_NONANTICIPATIVITY:
        FORMULATION_MODE = PASSENGER_AND_BUS_FORMULATION_MODE
        PASSENGER_RECOURSE = "NONANTICIPATIVE"
        SCHEMA = PASSENGER_AND_BUS_OOF_SCHEMA
        BASE_STAGE2_SCHEMA = PASSENGER_AND_BUS_STAGE2_SCHEMA
        EVALUATION_SCHEMA = PASSENGER_AND_BUS_EVALUATION_SCHEMA
    elif mode == BUS_ONLY_NONANTICIPATIVITY:
        FORMULATION_MODE = "BUS_ONLY_WAIT_AND_SEE"
        PASSENGER_RECOURSE = "WAIT_AND_SEE"
    else:
        raise ValueError(f"unknown non-anticipativity mode: {mode!r}")


def rail_state_policy() -> str:
    return (
        "PRESERVE_EACH_DURATION_INPUT_BRANCH"
        if NONANTICIPATIVITY_MODE == PASSENGER_AND_BUS_NONANTICIPATIVITY
        else "DURATION_SPECIFIC_WAIT_AND_SEE"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--single-group", action="store_true")
    parser.add_argument(
        "--nonanticipativity-mode",
        choices=(BUS_ONLY_NONANTICIPATIVITY,
                 PASSENGER_AND_BUS_NONANTICIPATIVITY),
        default=cfg.get("evaluation.solver.nonanticipativity_mode"),
    )
    parser.add_argument("--window", type=int)
    parser.add_argument("--station")
    parser.add_argument(
        "--workers", type=int, default=cfg.get("parallel.oof.workers"))
    parser.add_argument(
        "--threads", type=int, default=cfg.get("parallel.oof.threads"))
    parser.add_argument(
        "--time-limit-seconds", type=float,
        default=cfg.get("evaluation.solver.stage2_time_limit_s"))
    parser.add_argument(
        "--mip-gap", type=float, default=cfg.get("evaluation.solver.mip_gap"))
    parser.add_argument(
        "--mip-focus", type=int, default=cfg.get("evaluation.solver.mip_focus"))
    parser.add_argument(
        "--heuristics", type=float,
        default=cfg.get("evaluation.solver.heuristics"))
    parser.add_argument(
        "--cuts", type=int, default=cfg.get("evaluation.solver.cuts"))
    parser.add_argument(
        "--accept-incumbent",
        action=argparse.BooleanOptionalAction,
        default=bool(cfg.get("evaluation.solver.accept_incumbent")),
        help="accept any complete feasible incumbent without a gap retry",
    )
    parser.add_argument(
        "--allocation-dir", type=Path, default=DEFAULT_ALLOCATION_DIR,
        help="frozen Stage-1 DepotAlloc_HH.csv directory",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--oof-file", type=Path, default=OOF_FILE,
        help="out-of-fold probability file of the ML stage",
    )
    parser.add_argument("--normal-dir", type=Path, default=DEFAULT_NORMAL_DIR)
    parser.add_argument(
        "--disruption-dir", type=Path, default=DEFAULT_DISRUPTION_DIR)
    parser.add_argument(
        "--solve-source-dir",
        type=Path,
        help=(
            "with --evaluate-only: read the completed solves of every policy "
            "from this directory (read-only) and write the evaluation to "
            "--output-dir"
        ),
    )
    parser.add_argument(
        "--no-condition-cache", action="store_true",
        help="read the CSV outputs directly instead of the RI condition cache",
    )
    parser.add_argument(
        "--reuse-non-ml-output-dir",
        type=Path,
        help=(
            "Read completed Historical and Oracle solves from this output "
            "directory while writing and re-solving ML only in --output-dir."
        ),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 0:
        parser.error("--workers must be positive and --threads nonnegative")
    if args.time_limit_seconds <= 0:
        parser.error("--time-limit-seconds must be positive")
    if not 0 <= args.mip_gap < 1:
        parser.error("--mip-gap must be in [0,1)")
    if args.mip_focus not in (0, 1, 2, 3):
        parser.error("--mip-focus must be one of 0, 1, 2, 3")
    if not 0 <= args.heuristics <= 1:
        parser.error("--heuristics must be in [0,1]")
    if args.cuts not in (-1, 0, 1, 2, 3):
        parser.error("--cuts must be one of -1, 0, 1, 2, 3")
    if args.single_group and (
        args.window not in WINDOWS or not str(args.station or "").strip()
    ):
        parser.error("--single-group requires a valid --window and --station")
    if args.solve_source_dir is not None and not args.evaluate_only:
        parser.error("--solve-source-dir requires --evaluate-only")
    args.output_dir = args.output_dir.resolve()
    args.allocation_dir = args.allocation_dir.resolve()
    args.oof_file = args.oof_file.resolve()
    args.normal_dir = args.normal_dir.resolve()
    args.disruption_dir = args.disruption_dir.resolve()
    if args.solve_source_dir is not None:
        args.solve_source_dir = args.solve_source_dir.resolve()
    if args.reuse_non_ml_output_dir is not None:
        args.reuse_non_ml_output_dir = args.reuse_non_ml_output_dir.resolve()
    if args.station:
        args.station = str(args.station).strip().upper()
    return args


def safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def current_scenario_support(
    disruption_dir: Path,
) -> set[tuple[int, str, int]]:
    stats_path = disruption_dir / "_scenario_stats.csv"
    stats = pd.read_csv(stats_path, low_memory=False)
    required = {"window", "target", "duration", "checkpoint", "logic_version"}
    missing = required - set(stats.columns)
    if missing:
        raise ValueError(f"{stats_path}: missing {sorted(missing)}")
    stats["Hour"] = stats["window"].astype(str).str.split("_").str[0].astype(int)
    valid = stats[
        stats["checkpoint"].eq("EXTRACTED")
        & stats["logic_version"].eq(LOGIC_VERSION)
        & stats["duration"].isin(DURATIONS)
    ]
    return {
        (int(row.Hour), str(row.target).strip().upper(), int(row.duration))
        for row in valid.itertuples(index=False)
    }


def build_case_table(oof_file: Path, disruption_dir: Path) -> pd.DataFrame:
    oof = pd.read_csv(oof_file)
    if len(oof) != COHORT_SIZE or oof["IncidentID"].duplicated().any():
        raise ValueError(f"unexpected Long30 {MODEL} OOF rows: {len(oof)}")
    if "Model" not in oof or set(oof["Model"].astype(str)) != {MODEL}:
        raise ValueError(f"{oof_file}: expected only nested-OOF {MODEL} rows")
    targets = oof["Target"].astype(int).value_counts().to_dict()
    if targets != {0: HISTORICAL_COUNTS[SHORT], 1: HISTORICAL_COUNTS[LONG]}:
        raise ValueError(
            f"{oof_file}: target counts {targets} do not match "
            "probability.duration_counts"
        )
    if "OuterFold" not in oof:
        outer_fold = np.zeros(len(oof), dtype=int)
        splitter = StratifiedKFold(
            OUTER_FOLDS, shuffle=True, random_state=FOLD_SEED)
        for fold, (_, test) in enumerate(
            splitter.split(np.arange(len(oof)), oof["Target"].astype(int)), 1
        ):
            outer_fold[test] = fold
        oof["OuterFold"] = outer_fold

    incident = pd.read_csv(INCIDENT_FILE, low_memory=False)
    # The OOF artifact already contains the exact corridor fields used by the
    # fitted fold.  Join only physical incident metadata here so pandas does
    # not suffix and obscure those model-audit columns.
    incident_columns = ["IncidentID", "EventDate", "MaxDelayMinutes"]
    incident = incident[incident_columns]
    raw = pd.read_csv(
        RAW_INCIDENT_FILE,
        usecols=["ItemId", "ItemCreated", "ItemTypeTaxonomy"],
        low_memory=False,
    ).drop_duplicates("ItemId", keep="first")
    raw["IncidentID"] = pd.to_numeric(raw["ItemId"], errors="coerce")
    raw = raw.dropna(subset=["IncidentID"])
    raw["IncidentID"] = raw["IncidentID"].astype(int)
    raw["CreatedAt"] = pd.to_datetime(
        raw["ItemCreated"], errors="coerce", format="mixed"
    )
    raw["Hour"] = raw["CreatedAt"].dt.hour
    raw["IncidentTaxonomy"] = raw["ItemTypeTaxonomy"].fillna("Unknown").astype(str)
    raw["IncidentType"] = (
        raw["IncidentTaxonomy"].str.split(";").str[0].str.strip()
        .replace("", "Unknown")
    )

    cases = oof.merge(incident, on="IncidentID", validate="one_to_one")
    cases = cases.merge(
        raw[["IncidentID", "CreatedAt", "Hour", "IncidentType", "IncidentTaxonomy"]],
        on="IncidentID",
        validate="one_to_one",
    )
    cases["Station"] = cases["Station"].astype(str).str.strip().str.upper()
    cases["Hour"] = pd.to_numeric(cases["Hour"], errors="raise").astype(int)
    cases["ActualDurationMinutes"] = pd.to_numeric(
        cases["MaxDelayMinutes"], errors="raise"
    )
    cases["RealizedState"] = np.where(
        cases["ActualDurationMinutes"].ge(CUTOFF), LONG, SHORT
    )
    expected_target = cases["RealizedState"].eq(LONG).astype(int)
    if not expected_target.equals(cases["Target"].astype(int)):
        raise ValueError("OOF target and realized-state mapping differ")
    # Column names keep the paper's state labels (P30 = short, P60 = long).
    cases["HistoricalP60"] = HISTORICAL_WEIGHTS[LONG]
    cases["HistoricalP30"] = HISTORICAL_WEIGHTS[SHORT]
    cases["MLP60"] = pd.to_numeric(cases["Probability"], errors="raise")
    cases["MLP30"] = 1.0 - cases["MLP60"]

    support = current_scenario_support(disruption_dir)
    cases["InStage1Window"] = cases["Hour"].isin(WINDOWS)
    cases["D60Support"] = [
        (hour, station, LONG) in support
        for hour, station in zip(cases["Hour"], cases["Station"])
    ]
    cases["RealizedImpactAvailable"] = [
        (hour, station, int(duration)) in support
        for hour, station, duration in zip(
            cases["Hour"], cases["Station"], cases["RealizedState"]
        )
    ]
    cases["Eligible"] = cases["InStage1Window"] & cases["D60Support"]
    cases["ZeroImpactRealized"] = (
        cases["Eligible"] & ~cases["RealizedImpactAvailable"]
    )
    cases["EligibilityReason"] = np.select(
        [
            ~cases["InStage1Window"],
            cases["InStage1Window"] & ~cases["D60Support"],
            cases["ZeroImpactRealized"],
        ],
        [
            f"OUTSIDE_H{min(WINDOWS):02d}_H{max(WINDOWS):02d}",
            "NO_CURRENT_D60_STATION_HOUR_SUPPORT",
            "ELIGIBLE_ZERO_IMPACT_REALIZED_D30",
        ],
        default="ELIGIBLE_WITH_REALIZED_IMPACT",
    )
    cases["HistoricalJobID"] = [
        f"HIST_H{hour:02d}_{station}"
        for hour, station in zip(cases["Hour"], cases["Station"])
    ]
    cases["MLJobID"] = [f"ML_I{int(value)}" for value in cases["IncidentID"]]
    cases["OracleJobID"] = [
        f"ORACLE_H{hour:02d}_{station}_D{int(duration)}"
        for hour, station, duration in zip(
            cases["Hour"], cases["Station"], cases["RealizedState"]
        )
    ]
    columns = [
        "IncidentID", "OuterFold", "EventDate", "CreatedAt", "Hour", "Station",
        "StationCorridor", "CorridorSource", "CorridorIsImputed",
        "IncidentType", "IncidentTaxonomy", "ActualDurationMinutes",
        "RealizedState", "Target", "HistoricalP30", "HistoricalP60",
        "MLP30", "MLP60", "ReferenceProbability", "InStage1Window",
        "D60Support", "RealizedImpactAvailable", "ZeroImpactRealized",
        "Eligible", "EligibilityReason", "HistoricalJobID", "MLJobID",
        "OracleJobID",
    ]
    cases = cases[columns].sort_values("IncidentID").reset_index(drop=True)
    if EXPECTED_ELIGIBLE >= 0 and int(cases["Eligible"].sum()) != EXPECTED_ELIGIBLE:
        raise ValueError(
            f"expected {EXPECTED_ELIGIBLE} executable OOF incidents, "
            f"found {cases['Eligible'].sum()}"
        )
    if (EXPECTED_ZERO_IMPACT >= 0
            and int(cases["ZeroImpactRealized"].sum()) != EXPECTED_ZERO_IMPACT):
        raise ValueError(
            f"expected {EXPECTED_ZERO_IMPACT} retained zero-impact realized "
            "short-state cases"
        )
    if not cases["Eligible"].any():
        raise ValueError("no OOF incident falls inside the planning windows")
    return cases


def prepare(args: argparse.Namespace) -> pd.DataFrame:
    output, allocation_dir = args.output_dir, args.allocation_dir
    cases = build_case_table(args.oof_file, args.disruption_dir)
    write_csv(cases, output / "inputs" / "oof_incident_cases.csv")
    eligible = cases[cases["Eligible"]]
    type_audit = (
        cases.groupby("IncidentType", dropna=False)
        .agg(
            OOF_Incidents=("IncidentID", "size"),
            H06_H20=("InStage1Window", "sum"),
            Eligible=("Eligible", "sum"),
            ZeroImpactRealized=("ZeroImpactRealized", "sum"),
        )
        .reset_index()
        .sort_values("OOF_Incidents", ascending=False)
    )
    write_csv(type_audit, output / "inputs" / "incident_type_coverage.csv")
    audit = {
        "schema": SCHEMA,
        "formulation_mode": FORMULATION_MODE,
        "nonanticipativity_mode": NONANTICIPATIVITY_MODE,
        "passenger_recourse": PASSENGER_RECOURSE,
        "rail_state_policy": rail_state_policy(),
        "oof_model": MODEL,
        "cutoff_minutes": CUTOFF,
        "state_mapping": {
            f"D{SHORT}": f"duration < {CUTOFF}",
            f"D{LONG}": f"duration >= {CUTOFF}",
        },
        "input_audit_schema": INPUT_AUDIT_SCHEMA,
        "probability_source": PROBABILITY_SOURCE,
        "historical_counts": HISTORICAL_COUNTS,
        "historical_weights": HISTORICAL_WEIGHTS,
        "oof_incidents": len(cases),
        "eligible_incidents": int(cases["Eligible"].sum()),
        "excluded_incidents": int((~cases["Eligible"]).sum()),
        "zero_impact_realized_d30_retained": int(
            cases["ZeroImpactRealized"].sum()
        ),
        "eligible_station_hour_groups": int(
            eligible[["Hour", "Station"]].drop_duplicates().shape[0]
        ),
        "eligible_incident_types": int(eligible["IncidentType"].nunique()),
        "exclusion_counts": cases["EligibilityReason"].value_counts().to_dict(),
        "probability_input": f"nested-OOF {MODEL} tuned by inner-CV Brier score",
        "oof_file": str(args.oof_file),
        "oof_file_sha256": sha256_file(args.oof_file),
        "oof_model_audit": str(
            (args.oof_file.parent / "audit.json").resolve()
        ),
        "stage1_allocation": (
            "fixed x*(hour) from operational303 reclustered K14 Stage 1"
        ),
        "stage1_allocation_dir": str(allocation_dir.resolve()),
        "evaluation": "realized branch, unweighted over held-out incidents",
    }
    write_json(output / "inputs" / "case_construction_audit.json", audit)
    return cases


def planned_job_counts(cases: pd.DataFrame) -> dict[str, int]:
    eligible = cases[cases["Eligible"]]
    oracle_blocks = list(eligible.groupby("OracleJobID", sort=False))
    zero_impact_oracle_jobs = sum(
        bool(block["ZeroImpactRealized"].all())
        for _, block in oracle_blocks
    )
    historical_jobs = int(eligible["HistoricalJobID"].nunique())
    ml_jobs = int(eligible["MLJobID"].nunique())
    oracle_jobs = len(oracle_blocks)
    oracle_solver_jobs = oracle_jobs - zero_impact_oracle_jobs
    return {
        "historical_solver_jobs": historical_jobs,
        "ml_solver_jobs": ml_jobs,
        "oracle_jobs_total": oracle_jobs,
        "oracle_zero_impact_no_solve_jobs": zero_impact_oracle_jobs,
        "oracle_solver_jobs": oracle_solver_jobs,
        "total_solver_jobs": historical_jobs + ml_jobs + oracle_solver_jobs,
    }


def validate_preflight(
    args: argparse.Namespace, cases: pd.DataFrame
) -> dict:
    model_audit_path = args.oof_file.parent / "audit.json"
    model_audit = json.loads(model_audit_path.read_text(encoding="utf-8"))
    if model_audit.get("schema") != "long30-303-optuna-nested-comparison-v1":
        raise ValueError(f"unexpected nested-OOF audit schema: {model_audit_path}")
    if (
        int(model_audit.get("N", -1)) != COHORT_SIZE
        or int(model_audit.get("Long", -1)) != HISTORICAL_COUNTS[LONG]
        or int(model_audit.get("Short", -1)) != HISTORICAL_COUNTS[SHORT]
        or model_audit.get("reported_performance")
        != "pooled outer-fold OOF only"
    ):
        raise ValueError(f"{model_audit_path}: not the final nested-OOF cohort")

    input_audit_path = (
        args.output_dir / "inputs" / "case_construction_audit.json"
    )
    if not input_audit_path.is_file():
        raise FileNotFoundError(input_audit_path)
    input_audit = json.loads(input_audit_path.read_text(encoding="utf-8"))
    expected_input_values = {
        "schema": SCHEMA,
        "formulation_mode": FORMULATION_MODE,
        "nonanticipativity_mode": NONANTICIPATIVITY_MODE,
        "input_audit_schema": INPUT_AUDIT_SCHEMA,
        "oof_model": MODEL,
        "probability_source": PROBABILITY_SOURCE,
        "oof_file": str(args.oof_file),
        "oof_file_sha256": sha256_file(args.oof_file),
        "stage1_allocation_dir": str(args.allocation_dir.resolve()),
    }
    if NONANTICIPATIVITY_MODE == PASSENGER_AND_BUS_NONANTICIPATIVITY:
        expected_input_values["rail_state_policy"] = rail_state_policy()
    mismatches = {
        key: {"expected": expected, "actual": input_audit.get(key)}
        for key, expected in expected_input_values.items()
        if input_audit.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"prepared OOF input provenance mismatch: {mismatches}"
        )
    audit_counts = {
        int(duration): int(count)
        for duration, count in (input_audit.get("historical_counts") or {}).items()
    }
    if audit_counts != HISTORICAL_COUNTS:
        raise ValueError(
            f"prepared historical counts {audit_counts} differ from "
            "probability.duration_counts"
        )

    runner = load_module("oof_stage2_preflight", STAGE2_RUNNER_FILE)
    actual_stage2_schema = runner.schema_for_mode(NONANTICIPATIVITY_MODE)
    if actual_stage2_schema != BASE_STAGE2_SCHEMA:
        raise ValueError(
            f"Stage-2 runner schema {actual_stage2_schema!r} does not match "
            f"{BASE_STAGE2_SCHEMA!r}"
        )
    adaptive_source = runner.ADAPTIVE_FILE.read_text(encoding="utf-8")
    required_markers = {
        "NonanticipativityMode",
        "FormulationMode",
        "BusDecisionInformationSetNonanticipative",
        "PassengerFlowVariableSharing",
        "PassengerRecourse",
        "TrainAvailabilityInformationSetNonanticipative",
        "DurationSpecificPhysicalOmega",
        "DurationSpecificRailState",
        "RawBranchCanceledTrainsPreserved",
        "RawBranchCancellationStartsPreserved",
        "RawBranchWithdrawalThetaPreserved",
        "RawBranchTerminationEventsPreserved",
        "RawBranchPassengerCapacityThetaPreserved",
        "RawBranchRailStatePreserved",
        "TerminationActionSupportInformationSetNonanticipative",
        "ZeroImpactRailReconstructionsVerified",
    }
    missing_markers = sorted(
        marker for marker in required_markers if marker not in adaptive_source
    )
    if missing_markers:
        raise ValueError(
            f"Stage-2 implementation lacks markers: {missing_markers}"
        )

    allocation_rows = []
    for hour in WINDOWS:
        path = args.allocation_dir / f"DepotAlloc_{hour:02d}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if not {"Depot", "Count"}.issubset(frame.columns):
            raise ValueError(f"{path}: expected Depot and Count columns")
        count = pd.to_numeric(frame["Count"], errors="raise")
        if frame["Depot"].astype(str).duplicated().any():
            raise ValueError(f"{path}: duplicate depot rows")
        if (count < 0).any() or not np.allclose(count, np.rint(count)):
            raise ValueError(f"{path}: allocation counts must be nonnegative integers")
        total = int(count.sum())
        if total != FLEET_SIZE:
            raise ValueError(
                f"{path}: allocation totals {total}, expected {FLEET_SIZE}"
            )
        allocation_rows.append({"hour": hour, "fleet": total, "file": str(path)})

    if len(cases) != COHORT_SIZE or (
        EXPECTED_ELIGIBLE >= 0 and int(cases["Eligible"].sum()) != EXPECTED_ELIGIBLE
    ):
        raise ValueError(
            f"prepared OOF table must contain {COHORT_SIZE} incidents "
            f"({EXPECTED_ELIGIBLE} eligible)"
        )
    if cases["IncidentID"].duplicated().any():
        raise ValueError("prepared OOF table has duplicate incidents")
    if not np.allclose(
        cases["HistoricalP30"].astype(float), HISTORICAL_WEIGHTS[SHORT]
    ) or not np.allclose(
        cases["HistoricalP60"].astype(float), HISTORICAL_WEIGHTS[LONG]
    ):
        raise ValueError(
            "prepared Historical probabilities differ from "
            "probability.duration_counts"
        )
    if (
        (~cases["MLP30"].astype(float).between(0.0, 1.0)).any()
        or (~cases["MLP60"].astype(float).between(0.0, 1.0)).any()
        or not np.allclose(
            cases["MLP30"].astype(float) + cases["MLP60"].astype(float), 1.0
        )
    ):
        raise ValueError("prepared ML probabilities are invalid")

    return {
        "input_audit_schema": INPUT_AUDIT_SCHEMA,
        "base_stage2_schema": actual_stage2_schema,
        "probability_source": PROBABILITY_SOURCE,
        "historical_counts": HISTORICAL_COUNTS,
        "historical_weights": HISTORICAL_WEIGHTS,
        "oof_file_sha256": expected_input_values["oof_file_sha256"],
        "allocation_dir": str(args.allocation_dir),
        "allocation_files": len(allocation_rows),
        "allocation_fleet_total_each": FLEET_SIZE,
        "solver_profile": {
            "mip_focus": args.mip_focus,
            "heuristics": args.heuristics,
            "cuts": args.cuts,
        },
        "completion_policy": (
            "any_complete_feasible_incumbent_no_gap_retry"
            if args.accept_incumbent
            else "certified_within_target_gap"
        ),
        "job_counts": planned_job_counts(cases),
    }


def solve_root(output: Path, policy: str, job_id: str) -> Path:
    return output / "solves" / policy / job_id


def result_summary_path(output: Path, policy: str, job_id: str) -> Path:
    return solve_root(output, policy, job_id) / "summary.json"


def policy_output(args: argparse.Namespace, policy: str) -> Path:
    if policy != "ML" and args.reuse_non_ml_output_dir is not None:
        return args.reuse_non_ml_output_dir
    if getattr(args, "solve_source_dir", None) is not None:
        return args.solve_source_dir
    return args.output_dir


def decision_path(
    output: Path, policy: str, job_id: str, hour: int, station: str, duration: int
) -> Path:
    return solve_root(output, policy, job_id) / "DecisionVars" / (
        f"Disrupt_{hour:02d}_{station}_{duration}.csv"
    )


def oracle_mip_start(
    output: Path, selected: pd.DataFrame, state: int
) -> dict | None:
    candidates = [
        ("Historical", str(selected["HistoricalJobID"].iloc[0]))
    ]
    candidates.extend(
        ("ML", str(job_id))
        for job_id in selected["MLJobID"].drop_duplicates()
    )
    feasible = []
    for policy, job_id in candidates:
        root = solve_root(output, policy, job_id)
        summary_path = root / "summary.json"
        policy_path = root / "policy.csv"
        if not summary_path.is_file() or not policy_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("has_incumbent"):
            continue
        z = summary.get("Z") or {}
        objective = z.get(str(state), z.get(state))
        if objective is None or not math.isfinite(float(objective)):
            continue
        policy_frame = pd.read_csv(policy_path, low_memory=False)
        if policy_frame.empty:
            selected_trip_ids = []
        else:
            duration_support = policy_frame["SharedDurations"].astype(str).map(
                lambda value: state in {
                    int(float(item)) for item in value.split("|") if item.strip()
                }
            )
            selected_trip_ids = policy_frame.loc[
                duration_support, "TripID"
            ].astype(str).tolist()
        feasible.append({
            "selected_trip_ids": selected_trip_ids,
            "source_policy": policy,
            "source_job_id": job_id,
            "source_realized_objective": float(objective),
        })
    return min(
        feasible, key=lambda item: item["source_realized_objective"]
    ) if feasible else None


def best_policy_mip_start(
    historical_output: Path,
    ml_output: Path,
    selected: pd.DataFrame,
    weights: dict[int, float],
) -> dict:
    candidates = [
        (
            historical_output,
            "Historical",
            str(selected["HistoricalJobID"].iloc[0]),
        )
    ]
    candidates.extend(
        (ml_output, "ML", str(job_id))
        for job_id in selected["MLJobID"].drop_duplicates()
    )
    feasible = []
    for output, policy_name, job_id in candidates:
        root = solve_root(output, policy_name, job_id)
        summary_path = root / "summary.json"
        policy_path = root / "policy.csv"
        if not summary_path.is_file() or not policy_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("has_incumbent"):
            continue
        z = summary.get("Z") or {}
        if any(z.get(str(duration), z.get(duration)) is None
               for duration in DURATIONS):
            continue
        expected_objective = sum(
            float(weights[duration])
            * float(z.get(str(duration), z.get(duration)))
            for duration in DURATIONS
        )
        policy = pd.read_csv(policy_path, low_memory=False)
        selected_trip_cells = []
        for row in policy.itertuples(index=False):
            durations = [
                int(float(item))
                for item in str(row.SharedDurations).split("|")
                if item.strip()
            ]
            selected_trip_cells.append({
                "trip_id": str(row.TripID),
                "durations": durations,
            })
        feasible.append({
            "selected_trip_cells": selected_trip_cells,
            "source_policy": policy_name,
            "source_job_id": job_id,
            "source_expected_objective": float(expected_objective),
        })
    if not feasible:
        raise FileNotFoundError("no feasible Historical/ML MIP start found")
    return min(
        feasible, key=lambda item: item["source_expected_objective"]
    )


def strict_raw_rail_information_valid(
    stats: dict, durations: tuple[int, ...]
) -> bool:
    """Validate passenger NA without rewriting either raw rail branch."""
    branch_count = len(tuple(durations))
    passenger_sharing_is_valid = bool(
        branch_count == 1
        and stats.get("PassengerFlowVariableSharing") is False
        or branch_count > 1
        and stats.get("PassengerFlowVariableSharing") is True
        and stats.get("PassengerStateNonanticipativeByConservation") is True
        and stats.get("PassengerTransferFlowProjectedExactly") is True
        and stats.get("PassengerExitFlowProjectedExactly") is True
    )
    return bool(
        stats.get("PassengerRecourse") == "NONANTICIPATIVE"
        and passenger_sharing_is_valid
        and stats.get("DurationSpecificPhysicalOmega") is False
        and stats.get("DurationSpecificRailState") is True
        and stats.get("RailCommonEventTimetable") is True
        and stats.get(
            "RailEventSupportInformationSetNonanticipative") is True
        and int(stats.get("RailEventSupportMismatchCount", -1)) == 0
        and stats.get("TerminationAlightEventsVerified") is True
        and stats.get(
            "TerminationActionSupportInformationSetNonanticipative") is True
        and int(stats.get(
            "TerminationActionSupportMismatchCount", -1)) == 0
        and stats.get("PassengerCapacityThetaPresent") is True
        and stats.get("PassengerCapacityThetaMatchesClearanceEvents") is True
        and stats.get("PassengerCapacityWithdrawalPolicy")
        == RAW_RAIL_WITHDRAWAL_POLICY
        and stats.get("RawBranchCanceledTrainsPreserved") is True
        and stats.get("RawBranchCancellationStartsPreserved") is True
        and stats.get("RawBranchWithdrawalThetaPreserved") is True
        and stats.get("RawBranchTerminationEventsPreserved") is True
        and stats.get("RawBranchPassengerCapacityThetaPreserved") is True
        and stats.get("RawBranchRailStatePreserved") is True
        and stats.get("ZeroImpactRailReconstructionsVerified") is True
        and stats.get("TrainAvailabilityInformationSetNonanticipative") is False
    )


def completed_result(
    output: Path,
    policy: str,
    job_id: str,
    hour: int,
    station: str,
    durations: tuple[int, ...],
    args: argparse.Namespace,
) -> bool:
    path = result_summary_path(output, policy, job_id)
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        stats = value.get("stats") or {}
        raw_gap = stats.get("MIPGap")
        gap = (
            0.0
            if raw_gap is None and value.get("status") == "OPTIMAL"
            else float(raw_gap) if raw_gap is not None else math.nan
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    files = all(
        decision_path(output, policy, job_id, hour, station, duration).is_file()
        for duration in durations
    ) and all((solve_root(output, policy, job_id) / name).is_file()
              for name in ("policy.csv", "chain_policy.csv"))
    oracle_uses_common_pool = True
    if (
        policy == "Oracle"
        and tuple(durations) == (SHORT,)
        and not value.get("zero_impact_realized_branch")
    ):
        oracle_uses_common_pool = int(
            (value.get("stats") or {}).get("CandidateTripBasisDuration", -1)
        ) == max(DURATIONS) and bool(
            (value.get("stats") or {}).get(
                "CandidatePoolMatchesHistoricalML", False)
        ) and bool(
            (value.get("stats") or {}).get("MIPStartSourcePolicy")
        )
    formulation_is_valid = bool(
        value.get("schema") == SCHEMA
        and stats.get("ExactStateCompression") is True
    )
    zero_impact = value.get("zero_impact_realized_branch") is True
    common_information_audit = bool(
        stats.get("NonanticipativityMode") == NONANTICIPATIVITY_MODE
        and stats.get("FormulationMode") == FORMULATION_MODE
        and stats.get("BusDecisionInformationSetNonanticipative") is True
        and stats.get("VehicleChainInformationSetNonanticipative") is True
        and stats.get("PassengerSharedEqualityConstrs") == 0
    )
    if NONANTICIPATIVITY_MODE == PASSENGER_AND_BUS_NONANTICIPATIVITY:
        information_is_valid = bool(
            common_information_audit
            and strict_raw_rail_information_valid(stats, durations)
        )
    else:
        information_is_valid = bool(
            common_information_audit
            and stats.get("PassengerFlowVariableSharing") is False
            and stats.get("PassengerRecourse") == "WAIT_AND_SEE"
            and stats.get(
                "TrainAvailabilityInformationSetNonanticipative") is False
            and stats.get("DurationSpecificPhysicalOmega") is True
            and stats.get("DurationSpecificRailState") is True
        )
    information_is_valid = bool(
        information_is_valid
        or zero_impact and common_information_audit
        and (
            NONANTICIPATIVITY_MODE != PASSENGER_AND_BUS_NONANTICIPATIVITY
            or strict_raw_rail_information_valid(stats, durations)
        )
    )
    expected_allocation = str(
        (args.allocation_dir / f"DepotAlloc_{hour:02d}.csv").resolve()
    )
    profile = value.get("solver_profile") or {}
    provenance_is_valid = bool(
        value.get("probability_source") == PROBABILITY_SOURCE
        and (
            NONANTICIPATIVITY_MODE != PASSENGER_AND_BUS_NONANTICIPATIVITY
            or value.get("rail_state_policy") == rail_state_policy()
        )
        and value.get("stage1_allocation") == expected_allocation
        and value.get("target_mip_gap") == args.mip_gap
        and value.get("time_limit_seconds") == args.time_limit_seconds
        and profile.get("mip_focus") == args.mip_focus
        and math.isclose(
            float(profile.get("heuristics", math.nan)),
            args.heuristics,
            abs_tol=1e-12,
        )
        and profile.get("cuts") == args.cuts
    )
    quality_is_valid = bool(
        value.get("has_incumbent") is True
        and (
            args.accept_incumbent
            or (
                value.get("certified_within_gap") is True
                and math.isfinite(gap)
                and gap <= args.mip_gap + 1e-9
            )
        )
    )
    return bool(
        formulation_is_valid
        and information_is_valid
        and provenance_is_valid
        and quality_is_valid
        and files
        and oracle_uses_common_pool
    )


def uniform_walking_minutes() -> int:
    """Rail-bus transfer time sigma (Table 3); the Stage 2 core takes one value."""
    expected = int(cfg.get("operations.walking_transfer_min"))
    values = set(network_data.transfer_minutes().values())
    if values != {float(expected)}:
        raise ValueError(
            f"{cfg.get('inputs.walking_times')}: all WalkingMinutes must equal "
            f"operations.walking_transfer_min = {expected}; found {sorted(values)}"
        )
    return expected


def configure_stage2(core, adaptive, args) -> None:
    core.DISRUPT_BUFFER = BUFFER
    core.TIME_LIMIT = float(args.time_limit_seconds)
    core.MIP_GAP = float(args.mip_gap)
    core.DEPOT_ACCESS_BOUND_MIN = float(DEPOT_ACCESS_BOUND)
    core.MAX_TOTAL_BUSES = FLEET_SIZE
    core.MIN_TRIP_BOARDING = MIN_TRIP_BOARDING
    adaptive.WALKING_MINUTES = int(args.walking_minutes)
    adaptive.EXACT_STATE_COMPRESSION = True
    core.ROUTE_SOURCE = "predefined"
    core.BUS_BOARD_MODE = "general"
    core.FILE_PREDEFINED_ROUTES = str(ROUTE_FILE)
    core.FILE_SCHEDULE = str(SCHEDULE_FILE)
    core.FILE_OD = str(OD_FILE)
    core.FILE_DISTANCE = str(DISTANCE_FILE)
    core.FILE_CORRIDOR = str(CORRIDOR_FILE)
    core.FILE_EXCEL = str(COORDINATE_FILE)
    core.DIR_NORMAL = str(args.normal_dir)
    core.DIR_DISRUPT = str(args.disruption_dir)


def save_result(
    runner,
    output: Path,
    policy: str,
    job_id: str,
    incident_ids: list[int],
    hour: int,
    station: str,
    durations: tuple[int, ...],
    weights: dict[int, float],
    result: dict,
    elapsed: float,
    args: argparse.Namespace,
) -> None:
    root = solve_root(output, policy, job_id)
    stats = result.get("stats") or {}
    has_incumbent = int(stats.get("SolCount", 0) or 0) > 0
    gap = stats.get("MIPGap")
    if (
        (gap is None or not math.isfinite(float(gap)))
        and result.get("status") == "OPTIMAL"
    ):
        gap = 0.0
        stats["MIPGap"] = gap
    certified = bool(
        has_incumbent
        and gap is not None
        and math.isfinite(float(gap))
        and float(gap) <= args.mip_gap + 1e-9
    )
    if has_incumbent:
        runner.write_policy(root / "policy.csv", runner.policy_rows(result))
        runner.write_chain_policy(
            root / "chain_policy.csv", result.get("chain_decisions", [])
        )
        for duration in durations:
            runner.write_decision_variables(
                decision_path(output, policy, job_id, hour, station, duration),
                result.get("dv_rows", {}).get(duration, []),
            )
    write_json(
        root / "summary.json",
        {
            "schema": SCHEMA,
            "formulation_mode": FORMULATION_MODE,
            "nonanticipativity_mode": NONANTICIPATIVITY_MODE,
            "passenger_recourse": PASSENGER_RECOURSE,
            "rail_state_policy": rail_state_policy(),
            "policy": policy,
            "job_id": job_id,
            "incident_ids": incident_ids,
            "window": hour,
            "station": station,
            "durations": durations,
            "input_weights": weights,
            "objective": result.get("objective") if has_incumbent else None,
            "Z": result.get("Z") if has_incumbent else {},
            "status": result.get("status"),
            "stats": stats,
            "has_incumbent": has_incumbent,
            "certified_within_gap": certified,
            "exact_state_compression": True,
            "vehicle_chain_information_set_nonanticipative": True,
            "passenger_flow_variable_sharing": bool(
                stats.get("PassengerFlowVariableSharing")),
            "duration_specific_physical_omega": bool(
                stats.get("DurationSpecificPhysicalOmega")),
            "duration_specific_rail_state": bool(
                stats.get("DurationSpecificRailState")),
            "raw_branch_rail_state_preserved": bool(
                stats.get("RawBranchRailStatePreserved")),
            "raw_branch_canceled_trains_preserved": bool(
                stats.get("RawBranchCanceledTrainsPreserved")),
            "raw_branch_termination_events_preserved": bool(
                stats.get("RawBranchTerminationEventsPreserved")),
            "raw_rail_availability_equal_across_durations": bool(
                stats.get("RawRailAvailabilityEqualAcrossDurations")),
            "probability_source": PROBABILITY_SOURCE,
            "historical_duration_counts": HISTORICAL_COUNTS,
            "target_mip_gap": args.mip_gap,
            "time_limit_seconds": args.time_limit_seconds,
            "accept_incumbent": bool(args.accept_incumbent),
            "solver_profile": {
                "mip_focus": args.mip_focus,
                "heuristics": args.heuristics,
                "cuts": args.cuts,
            },
            "elapsed_seconds": elapsed,
            "stage1_allocation": str(
                runner.allocation_path(hour, args.allocation_dir).resolve()
            ),
        },
    )


def save_zero_impact_oracle(
    runner, output: Path, job_id: str, incident_ids: list[int],
    hour: int, station: str, args: argparse.Namespace,
) -> None:
    strict_raw_rail = bool(
        NONANTICIPATIVITY_MODE == PASSENGER_AND_BUS_NONANTICIPATIVITY
    )
    path = decision_path(output, "Oracle", job_id, hour, station, SHORT)
    runner.write_decision_variables(path, [])
    runner.write_policy(solve_root(output, "Oracle", job_id) / "policy.csv", [])
    runner.write_chain_policy(
        solve_root(output, "Oracle", job_id) / "chain_policy.csv", []
    )
    write_json(
        result_summary_path(output, "Oracle", job_id),
        {
            "schema": SCHEMA,
            "formulation_mode": FORMULATION_MODE,
            "nonanticipativity_mode": NONANTICIPATIVITY_MODE,
            "passenger_recourse": PASSENGER_RECOURSE,
            "rail_state_policy": rail_state_policy(),
            "policy": "Oracle",
            "job_id": job_id,
            "incident_ids": incident_ids,
            "window": hour,
            "station": station,
            "durations": [SHORT],
            "input_weights": {SHORT: 1.0},
            "objective": 0.0,
            "Z": {SHORT: 0.0},
            "status": "ZERO_IMPACT_NO_SOLVE",
            "stats": {
                "SolCount": 1,
                "MIPGap": 0.0,
                "Runtime": 0.0,
                "ExactStateCompression": True,
                "NonanticipativityMode": NONANTICIPATIVITY_MODE,
                "FormulationMode": FORMULATION_MODE,
                "BusDecisionInformationSetNonanticipative": True,
                "VehicleChainInformationSetNonanticipative": True,
                "PassengerFlowVariableSharing": False,
                "PassengerSharedEqualityConstrs": 0,
                "PassengerRecourse": PASSENGER_RECOURSE,
                "PassengerStateNonanticipativeByConservation": False,
                "PassengerTransferFlowProjectedExactly": True,
                "PassengerExitFlowProjectedExactly": True,
                "RailCommonEventTimetable": True,
                "RailEventSupportInformationSetNonanticipative": (
                    strict_raw_rail),
                "RailEventSupportMismatchCount": 0,
                "TerminationAlightEventsVerified": strict_raw_rail,
                "TerminationActionSupportInformationSetNonanticipative": (
                    strict_raw_rail),
                "TerminationActionSupportMismatchCount": 0,
                "PassengerCapacityThetaPresent": strict_raw_rail,
                "PassengerCapacityThetaMatchesClearanceEvents": (
                    strict_raw_rail),
                "PassengerCapacityWithdrawalPolicy": (
                    RAW_RAIL_WITHDRAWAL_POLICY if strict_raw_rail
                    else "DURATION_SPECIFIC_SERVICE_THETA"
                ),
                "RawBranchCanceledTrainsPreserved": strict_raw_rail,
                "RawBranchCancellationStartsPreserved": strict_raw_rail,
                "RawBranchWithdrawalThetaPreserved": strict_raw_rail,
                "RawBranchTerminationEventsPreserved": strict_raw_rail,
                "RawBranchPassengerCapacityThetaPreserved": strict_raw_rail,
                "RawBranchRailStatePreserved": strict_raw_rail,
                "ZeroImpactRailReconstructionCount": int(strict_raw_rail),
                "ZeroImpactRailReconstructionsVerified": strict_raw_rail,
                "TrainAvailabilityInformationSetNonanticipative": False,
                "DurationSpecificPhysicalOmega": bool(
                    NONANTICIPATIVITY_MODE
                    == BUS_ONLY_NONANTICIPATIVITY),
                "DurationSpecificRailState": True,
            },
            "has_incumbent": True,
            "certified_within_gap": True,
            "exact_state_compression": True,
            "vehicle_chain_information_set_nonanticipative": True,
            "passenger_flow_variable_sharing": False,
            "duration_specific_physical_omega": bool(
                NONANTICIPATIVITY_MODE == BUS_ONLY_NONANTICIPATIVITY),
            "duration_specific_rail_state": True,
            "raw_branch_rail_state_preserved": strict_raw_rail,
            "probability_source": PROBABILITY_SOURCE,
            "historical_duration_counts": HISTORICAL_COUNTS,
            "target_mip_gap": args.mip_gap,
            "time_limit_seconds": args.time_limit_seconds,
            "accept_incumbent": bool(args.accept_incumbent),
            "solver_profile": {
                "mip_focus": args.mip_focus,
                "heuristics": args.heuristics,
                "cuts": args.cuts,
            },
            "elapsed_seconds": 0.0,
            "stage1_allocation": str(
                runner.allocation_path(hour, args.allocation_dir).resolve()
            ),
            "zero_impact_realized_branch": True,
        },
    )


def run_single_group(args, cases: pd.DataFrame) -> int:
    import gurobipy as gp

    selected = cases[
        cases["Eligible"]
        & cases["Hour"].eq(args.window)
        & cases["Station"].eq(args.station)
    ].copy()
    if selected.empty:
        raise ValueError(f"no eligible incidents for H{args.window:02d}/{args.station}")

    runner = load_module("oof_stage2_runner", STAGE2_RUNNER_FILE)
    args.runner = runner
    args.walking_minutes = uniform_walking_minutes()
    os.environ["TSB_DEPOT_ACCESS_BOUND"] = str(DEPOT_ACCESS_BOUND)
    os.environ["TSB_MAX_BUSES"] = str(FLEET_SIZE)
    os.environ["TSB_ROUTE_SOURCE"] = "predefined"
    os.environ["TSB_BUS_BOARD"] = "general"
    os.environ["PAPER_OD_HOURLY_MIN_TRIPS"] = str(OD_HOURLY_MIN_TRIPS)
    core, adaptive = runner.load_local_stage2_modules()
    configure_stage2(core, adaptive, args)

    depots = runner.load_depots(DEPOT_FILE)
    allocation = runner.load_allocation(
        runner.allocation_path(args.window, args.allocation_dir), depots
    )
    schedule = core.load_schedule(core.FILE_SCHEDULE)
    od = core.load_od(core.FILE_OD)
    bus_tt = core.load_distance_matrix(core.FILE_DISTANCE)
    route_library = core.load_route_library(
        core.FILE_PREDEFINED_ROUTES,
        distance_path=core.FILE_DISTANCE,
        validate_forward_times=True,
    )
    window_data = core.build_window_data(args.window, schedule, od)
    core.DURATIONS = list(DURATIONS)
    core.DURATION_WEIGHT = dict(HISTORICAL_WEIGHTS)
    raw_scenarios = core.load_disruption_scenarios(args.window, window_data)

    environment = gp.Env(empty=True)
    environment.setParam("OutputFlag", 0)
    environment.setParam("Threads", args.threads)
    environment.start()
    try:
        full_group = adaptive.build_adaptive_group(
            core,
            args.station,
            args.window,
            window_data,
            raw_scenarios,
            depots,
            bus_tt,
            route_library,
            durations=DURATIONS,
            buffer=BUFFER,
            nonanticipativity_mode=NONANTICIPATIVITY_MODE,
        )
        common_candidate_trip_ids = set(full_group["trips"])
        common_available_trip_ids = {
            int(duration): set(full_group["avail"].get(duration, ()))
            for duration in DURATIONS
        }
        historical_job = str(selected["HistoricalJobID"].iloc[0])
        incident_ids = selected["IncidentID"].astype(int).tolist()
        historical_output = policy_output(args, "Historical")
        if not completed_result(
            historical_output, "Historical", historical_job, args.window,
            args.station, DURATIONS, args,
        ):
            if args.reuse_non_ml_output_dir is not None:
                raise FileNotFoundError(
                    f"Reusable Historical solve is incomplete: {historical_job}"
                )
            started = time.time()
            result = adaptive.solve_stage2_adaptive(
                core, environment, depots, allocation, full_group, window_data,
                duration_weight=HISTORICAL_WEIGHTS,
                mip_gap=args.mip_gap,
                time_limit=args.time_limit_seconds,
                output_flag=0,
                threads=args.threads,
                mip_focus=args.mip_focus,
                heuristics=args.heuristics,
                cuts=args.cuts,
                nonanticipativity_mode=NONANTICIPATIVITY_MODE,
            )
            save_result(
                runner, args.output_dir, "Historical", historical_job,
                incident_ids, args.window, args.station, DURATIONS,
                dict(HISTORICAL_WEIGHTS), result, time.time() - started, args,
            )
            del result
            gc.collect()

        for row in selected.sort_values("IncidentID").itertuples(index=False):
            job_id = str(row.MLJobID)
            if completed_result(
                args.output_dir, "ML", job_id, args.window, args.station,
                DURATIONS, args,
            ):
                continue
            weights = {SHORT: float(row.MLP30), LONG: float(row.MLP60)}
            mip_start = best_policy_mip_start(
                historical_output, args.output_dir, selected, weights
            )
            started = time.time()
            result = adaptive.solve_stage2_adaptive(
                core, environment, depots, allocation, full_group, window_data,
                duration_weight=weights,
                mip_gap=args.mip_gap,
                time_limit=args.time_limit_seconds,
                output_flag=0,
                threads=args.threads,
                mip_start=mip_start,
                mip_focus=args.mip_focus,
                heuristics=args.heuristics,
                cuts=args.cuts,
                nonanticipativity_mode=NONANTICIPATIVITY_MODE,
            )
            if result.get("Z"):
                source_objective = float(
                    mip_start["source_expected_objective"]
                )
                tolerance = 1e-6 * max(1.0, abs(source_objective))
                if float(result["objective"]) > source_objective + tolerance:
                    raise RuntimeError(
                        f"{job_id}: ML incumbent {result['objective']} is "
                        f"worse than Historical feasible incumbent "
                        f"{source_objective} under the ML probabilities"
                    )
            save_result(
                runner, args.output_dir, "ML", job_id, [int(row.IncidentID)],
                args.window, args.station, DURATIONS, weights, result,
                time.time() - started, args,
            )
            del result
            gc.collect()
        del full_group
        gc.collect()

        for state, block in selected.groupby("RealizedState", sort=True):
            state = int(state)
            job_id = str(block["OracleJobID"].iloc[0])
            ids = block["IncidentID"].astype(int).tolist()
            oracle_output = policy_output(args, "Oracle")
            if completed_result(
                oracle_output, "Oracle", job_id, args.window, args.station,
                (state,), args,
            ):
                continue
            if args.reuse_non_ml_output_dir is not None:
                raise FileNotFoundError(
                    f"Reusable Oracle solve is incomplete: {job_id}"
                )
            if state == SHORT and bool(block["ZeroImpactRealized"].all()):
                save_zero_impact_oracle(
                    runner, args.output_dir, job_id, ids, args.window,
                    args.station, args,
                )
                continue
            core.DURATIONS = [state]
            core.DURATION_WEIGHT = {state: 1.0}
            oracle_group = adaptive.build_adaptive_group(
                core,
                args.station,
                args.window,
                window_data,
                raw_scenarios,
                depots,
                bus_tt,
                route_library,
                durations=(state,),
                buffer=BUFFER,
                trip_basis_duration=max(DURATIONS),
                nonanticipativity_mode=NONANTICIPATIVITY_MODE,
            )
            if set(oracle_group["trips"]) != common_candidate_trip_ids:
                raise RuntimeError(
                    f"{job_id}: Oracle candidate trip pool differs from "
                    "Historical/ML"
                )
            if set(oracle_group["avail"].get(state, ())) != (
                common_available_trip_ids[state]
            ):
                raise RuntimeError(
                    f"{job_id}: Oracle duration-{state} availability differs "
                    "from Historical/ML"
                )
            mip_start = oracle_mip_start(args.output_dir, selected, state)
            started = time.time()
            result = adaptive.solve_stage2_adaptive(
                core, environment, depots, allocation, oracle_group, window_data,
                duration_weight={state: 1.0},
                mip_gap=args.mip_gap,
                time_limit=args.time_limit_seconds,
                output_flag=0,
                threads=args.threads,
                mip_start=mip_start,
                mip_focus=args.mip_focus,
                heuristics=args.heuristics,
                cuts=args.cuts,
                nonanticipativity_mode=NONANTICIPATIVITY_MODE,
            )
            result.setdefault("stats", {})[
                "CandidatePoolMatchesHistoricalML"
            ] = True
            if mip_start and result.get("Z"):
                oracle_objective = float(result["Z"][state])
                source_objective = float(
                    mip_start["source_realized_objective"])
                tolerance = 1e-6 * max(1.0, abs(source_objective))
                if oracle_objective > source_objective + tolerance:
                    raise RuntimeError(
                        f"{job_id}: Oracle incumbent {oracle_objective} is "
                        f"worse than feasible {mip_start['source_policy']} "
                        f"incumbent {source_objective}"
                    )
            save_result(
                runner, args.output_dir, "Oracle", job_id, ids, args.window,
                args.station, (state,), {state: 1.0}, result,
                time.time() - started, args,
            )
            del result, oracle_group
            gc.collect()
    finally:
        environment.dispose()
    return 0


def group_complete(args, block: pd.DataFrame) -> bool:
    hour = int(block["Hour"].iloc[0])
    station = str(block["Station"].iloc[0])
    historical = completed_result(
        policy_output(args, "Historical"), "Historical",
        str(block["HistoricalJobID"].iloc[0]),
        hour, station, DURATIONS, args,
    )
    ml = all(
        completed_result(
            args.output_dir, "ML", str(row.MLJobID), hour, station,
            DURATIONS, args,
        )
        for row in block.itertuples(index=False)
    )
    oracle = all(
        completed_result(
            policy_output(args, "Oracle"), "Oracle",
            str(state_block["OracleJobID"].iloc[0]),
            hour, station, (int(state),), args,
        )
        for state, state_block in block.groupby("RealizedState")
    )
    return historical and ml and oracle


def run_group_subprocess(args, hour: int, station: str) -> tuple[int, str]:
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"H{hour:02d}_{station}"
    command = [
        sys.executable, "-X", "utf8", "-u", str(SCRIPT),
        "--single-group", "--window", str(hour), "--station", station,
        "--nonanticipativity-mode", args.nonanticipativity_mode,
        "--threads", str(args.threads),
        "--time-limit-seconds", str(args.time_limit_seconds),
        "--mip-gap", str(args.mip_gap),
        "--mip-focus", str(args.mip_focus),
        "--heuristics", str(args.heuristics),
        "--cuts", str(args.cuts),
        "--allocation-dir", str(args.allocation_dir),
        "--output-dir", str(args.output_dir),
        "--oof-file", str(args.oof_file),
        "--normal-dir", str(args.normal_dir),
        "--disruption-dir", str(args.disruption_dir),
        "--accept-incumbent" if args.accept_incumbent
        else "--no-accept-incumbent",
    ]
    if args.reuse_non_ml_output_dir is not None:
        command.extend(
            ["--reuse-non-ml-output-dir", str(args.reuse_non_ml_output_dir)]
        )
    with (log_dir / f"{stem}.out.log").open("a", encoding="utf-8") as stdout, (
        log_dir / f"{stem}.err.log"
    ).open("a", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command, cwd=PROJECT, stdout=stdout, stderr=stderr, check=False
        )
    return completed.returncode, stem


def _validated_strict_output_support(
    *,
    job_id: str,
    support_key: tuple,
    output_ods: frozenset[str],
    required_physical_ods: frozenset[str],
    normal_ods: frozenset[str],
    declared_count: int | None,
    support_cache: dict,
) -> frozenset[str]:
    """Validate and retain the causal support declared by a strict solve.

    Raw disruption CSVs contain their duration-specific physical OD support.
    A strict solve can legitimately contain additional ODs when causal train
    withdrawal makes them part of the common modelling support.  The decision
    output is authoritative only after checking the certified model count,
    physical coverage, Normal universe, and policy-to-policy consistency.
    """
    output_ods = frozenset(output_ods)
    required_physical_ods = frozenset(required_physical_ods)
    normal_ods = frozenset(normal_ods)
    if declared_count is None:
        raise ValueError(f"{job_id}: strict result omits CommonAffectedODCount")
    if len(output_ods) != int(declared_count):
        raise ValueError(
            f"{job_id}: output support count {len(output_ods)} differs from "
            f"certified common count {int(declared_count)}"
        )
    missing_physical = required_physical_ods - output_ods
    if missing_physical:
        raise ValueError(
            f"{job_id}: output omits {len(missing_physical)} physical-support ODs"
        )
    outside_normal = output_ods - normal_ods
    if outside_normal:
        raise ValueError(
            f"{job_id}: output has {len(outside_normal)} ODs outside Normal"
        )
    previous = support_cache.get(support_key)
    if previous is not None and previous != output_ods:
        raise ValueError(
            f"{job_id}: strict output support differs across policies for "
            f"{support_key}"
        )
    support_cache[support_key] = output_ods
    return output_ods


def evaluate(args, cases: pd.DataFrame) -> int:
    evaluator = load_module("oof_evaluate_ri", EVALUATOR_FILE)
    # The RI/PT helpers read these module paths.
    evaluator.NORMAL_DIR = args.normal_dir
    evaluator.OD_FILE = OD_FILE
    evaluator.CONDITION_CACHE_DIR = RI_CACHE_DIR
    if args.no_condition_cache:
        evaluator.CONDITION_CACHE_ENABLED = False
    eligible = cases[cases["Eligible"]].copy()
    available_support = current_scenario_support(args.disruption_dir)
    normal_cache = {
        int(hour): evaluator.load_normal(int(hour))
        for hour in sorted(eligible["Hour"].unique())
    }
    condition_cache = {}
    scenario_cache = {}
    strict_output_support_cache = {}
    rows = []
    missing = []
    evaluation_index = pd.Index(
        range(EVALUATION_START, HORIZON + 1), name="Time")

    def physical_condition(hour: int, station: str, duration: int):
        key = (hour, station, duration)
        if key not in condition_cache:
            if key not in available_support:
                condition_cache[key] = None
            else:
                path = (
                    args.disruption_dir
                    / f"Disrupt_{hour}_{station}_{duration}.csv"
                )
                if not path.is_file():
                    raise FileNotFoundError(path)
                condition_cache[key] = evaluator.read_condition(path)
        return condition_cache[key]

    for case in eligible.itertuples(index=False):
        hour = int(case.Hour)
        station = str(case.Station)
        duration = int(case.RealizedState)
        key = (hour, station, duration)
        if key not in scenario_cache:
            normal = normal_cache[hour]
            branch_conditions = {
                branch: physical_condition(hour, station, int(branch))
                for branch in DURATIONS
            }
            duration_union_audit_ods = frozenset().union(*(
                condition.od_set
                for condition in branch_conditions.values()
                if condition is not None
            ))
            unknown_duration_union = duration_union_audit_ods - normal.od_set
            if unknown_duration_union:
                raise ValueError(
                    f"{hour, station}: duration-support union has "
                    f"{len(unknown_duration_union)} "
                    "ODs outside Normal"
                )
            disruption = branch_conditions[duration]
            if disruption is None:
                affected_ods = frozenset()
                unaffected_qlw = normal.total_qlw.copy()
                disrupted_qlw = normal.total_qlw.copy()
                disrupted_state = normal.total_state_by_variable_time.iloc[0:0].copy()
                source = None
            else:
                affected_ods = disruption.od_set
                unknown = affected_ods - normal.od_set
                if unknown:
                    raise ValueError(f"{key}: affected ODs outside Normal")
                unaffected_qlw = evaluator.normal_unaffected_qlw(
                    normal, affected_ods
                )
                disrupted_qlw = disruption.total_qlw + unaffected_qlw
                disrupted_state = disruption.state_by_variable_time
                source = (
                    args.disruption_dir
                    / f"Disrupt_{hour}_{station}_{duration}.csv"
                )
            affected_passenger_demand = float(normal.demand_by_od.reindex(
                sorted(affected_ods), fill_value=0.0
            ).sum())
            affected_initial_passengers = float(
                normal.initial_passengers_by_od.reindex(
                    sorted(affected_ods), fill_value=0.0
                ).sum()
            )
            affected_post_disruption_demand = float(
                normal.post_disruption_demand_by_od.reindex(
                    sorted(affected_ods), fill_value=0.0
                ).sum()
            )
            disrupted_metrics = evaluator.passenger_time_metrics(
                disrupted_state, affected_passenger_demand, evaluation_index
            )
            scenario_cache[key] = {
                "normal": normal,
                "affected_ods": affected_ods,
                "duration_union_audit_ods": duration_union_audit_ods,
                "disruption_source": source,
                "disruption_ri": evaluator.resilience_index(
                    normal.total_qlw, disrupted_qlw, EVALUATION_START
                ),
                "disruption_metrics": disrupted_metrics,
                "affected_passenger_demand": affected_passenger_demand,
                "affected_initial_passengers": affected_initial_passengers,
                "affected_post_disruption_demand": (
                    affected_post_disruption_demand
                ),
            }
        scenario = scenario_cache[key]

        policies = [
            ("Historical", str(case.HistoricalJobID), float(case.HistoricalP60)),
            ("ML", str(case.MLJobID), float(case.MLP60)),
            ("Oracle", str(case.OracleJobID), float(duration == LONG)),
        ]
        for policy, job_id, p60 in policies:
            source_output = policy_output(args, policy)
            summary_file = result_summary_path(source_output, policy, job_id)
            dv_file = decision_path(
                source_output, policy, job_id, hour, station, duration
            )
            if not summary_file.is_file() or not dv_file.is_file():
                missing.append((int(case.IncidentID), policy, job_id))
                continue
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            if not summary.get("has_incumbent"):
                missing.append((int(case.IncidentID), policy, job_id))
                continue
            stats = summary.get("stats") or {}
            solver_profile = summary.get("solver_profile") or {}
            expected_allocation = str(
                (args.allocation_dir / f"DepotAlloc_{hour:02d}.csv").resolve()
            )
            if (
                summary.get("schema") != SCHEMA
                or summary.get("probability_source") != PROBABILITY_SOURCE
                or (
                    NONANTICIPATIVITY_MODE
                    == PASSENGER_AND_BUS_NONANTICIPATIVITY
                    and summary.get("rail_state_policy") != rail_state_policy()
                )
                or summary.get("stage1_allocation") != expected_allocation
                or stats.get("ExactStateCompression") is not True
                or solver_profile.get("mip_focus") != args.mip_focus
                or not math.isclose(
                    float(solver_profile.get("heuristics", math.nan)),
                    args.heuristics,
                    abs_tol=1e-12,
                )
                or solver_profile.get("cuts") != args.cuts
            ):
                raise ValueError(
                    f"{job_id}: result provenance/formulation schema mismatch"
                )
            result_durations = tuple(int(item) for item in summary["durations"])
            common_audit = bool(
                stats.get("NonanticipativityMode")
                == NONANTICIPATIVITY_MODE
                and stats.get("FormulationMode") == FORMULATION_MODE
                and stats.get("BusDecisionInformationSetNonanticipative")
                is True
                and stats.get("VehicleChainInformationSetNonanticipative")
                is True
                and stats.get("PassengerSharedEqualityConstrs") == 0
            )
            if NONANTICIPATIVITY_MODE == PASSENGER_AND_BUS_NONANTICIPATIVITY:
                result_information_valid = bool(
                    common_audit
                    and strict_raw_rail_information_valid(
                        stats, result_durations)
                )
            else:
                result_information_valid = bool(
                    common_audit
                    and stats.get("PassengerFlowVariableSharing") is False
                    and stats.get("PassengerRecourse") == "WAIT_AND_SEE"
                    and stats.get(
                        "TrainAvailabilityInformationSetNonanticipative")
                    is False
                    and stats.get("DurationSpecificPhysicalOmega") is True
                    and stats.get("DurationSpecificRailState") is True
                )
            if not result_information_valid:
                raise ValueError(
                    f"{job_id}: result is not {FORMULATION_MODE}"
                )
            bridging = evaluator.read_condition(dv_file)
            uses_common_support = bool(
                NONANTICIPATIVITY_MODE
                == PASSENGER_AND_BUS_NONANTICIPATIVITY
                and stats.get("CommonAffectedODCount") is not None
            )
            if uses_common_support:
                required_physical_ods = (
                    scenario["duration_union_audit_ods"]
                    if len(result_durations) > 1
                    else scenario["affected_ods"]
                )
                allowed_ods = _validated_strict_output_support(
                    job_id=job_id,
                    support_key=(hour, station, tuple(result_durations)),
                    output_ods=bridging.od_set,
                    required_physical_ods=required_physical_ods,
                    normal_ods=scenario["normal"].od_set,
                    declared_count=stats.get("CommonAffectedODCount"),
                    support_cache=strict_output_support_cache,
                )
            else:
                allowed_ods = scenario["affected_ods"]
            outside = bridging.od_set - allowed_ods
            if outside:
                raise ValueError(
                    f"{job_id}: output has {len(outside)} ODs outside its "
                    "declared support"
                )
            missing_physical = scenario["affected_ods"] - bridging.od_set
            outside_physical_output = bridging.od_set - scenario["affected_ods"]
            declared_support_mode = (
                "common-support-raw-rail-v5"
                if uses_common_support else "physical-support-bus-only"
            )
            expected_output_ods = allowed_ods
            missing_expected = expected_output_ods - bridging.od_set
            if not uses_common_support and outside_physical_output:
                raise ValueError(
                    f"{job_id}: physical-support output contains ODs outside "
                    "the realized physical support"
                )
            if uses_common_support and missing_expected:
                raise ValueError(
                    f"{job_id}: common-support output omits "
                    f"{len(missing_expected)} ODs"
                )
            physical_unaffected_qlw = evaluator.normal_unaffected_qlw(
                scenario["normal"], scenario["affected_ods"]
            )
            common_unaffected_qlw = evaluator.normal_unaffected_qlw(
                scenario["normal"], allowed_ods
            )
            condition_qlw, physical_state, support_mode = (
                evaluator.bridging_condition_for_ri(
                    bridging,
                    scenario["affected_ods"],
                    physical_unaffected_qlw,
                    common_unaffected_qlw,
                    "full",
                    declared_support_mode,
                )
            )
            metrics = evaluator.passenger_time_metrics(
                physical_state,
                scenario["affected_passenger_demand"],
                evaluation_index,
            )
            ri = evaluator.resilience_index(
                scenario["normal"].total_qlw,
                condition_qlw,
                EVALUATION_START,
            )
            z = summary.get("Z") or {}
            realized_objective = z.get(str(duration), z.get(duration, 0.0))
            row = {
                "IncidentID": int(case.IncidentID),
                "OuterFold": int(case.OuterFold),
                "CreatedAt": case.CreatedAt,
                "Hour": hour,
                "Station": station,
                "StationCorridor": case.StationCorridor,
                "IncidentType": case.IncidentType,
                "IncidentTaxonomy": case.IncidentTaxonomy,
                "ActualDurationMinutes": float(case.ActualDurationMinutes),
                "RealizedState": duration,
                "ZeroImpactRealized": bool(case.ZeroImpactRealized),
                "Policy": policy,
                "InputP30": 1.0 - p60,
                "InputP60": p60,
                "JobID": job_id,
                "Status": summary.get("status"),
                "MIPGap": stats.get("MIPGap"),
                "RuntimeSeconds": stats.get("Runtime"),
                "CertifiedWithinGap": summary.get("certified_within_gap"),
                "RealizedObjective": realized_objective,
                "AffectedODCount": len(scenario["affected_ods"]),
                "DurationUnionAuditODCount": len(
                    scenario["duration_union_audit_ods"]
                ),
                "ExpectedOutputSupport": (
                    "raw-rail-common-model-support"
                    if uses_common_support else "realized-physical"
                ),
                "RawBranchRailStatePreserved": bool(
                    stats.get("RawBranchRailStatePreserved")),
                "ExpectedOutputSupportODCount": len(expected_output_ods),
                "OutputODCount": len(bridging.od_set),
                "OutputPhysicalAffectedODCount": len(
                    bridging.od_set & scenario["affected_ods"]
                ),
                "OutputOutsidePhysicalSupportODCount": len(
                    outside_physical_output
                ),
                "MissingExpectedSupportODCount": len(missing_expected),
                "MissingPhysicalAffectedODCount": len(missing_physical),
                "MissingDurationUnionAuditODCount": (
                    len(scenario["duration_union_audit_ods"] - bridging.od_set)
                    if uses_common_support else np.nan
                ),
                "RI_Output_Support_Mode": support_mode,
                "Passenger_Time_OD_Scope": "physical-realized-affected",
                "PT_Denominator_Initial_t0_minus_1": scenario[
                    "affected_initial_passengers"
                ],
                "PT_Denominator_Post_t0_Demand": scenario[
                    "affected_post_disruption_demand"
                ],
                "EvaluationSchema": EVALUATION_SCHEMA,
                "RI_Disruption": scenario["disruption_ri"],
                "RI": ri,
                "RI_Improvement": ri - scenario["disruption_ri"],
                "DecisionFile": str(dv_file),
                "DisruptionFile": (
                    str(scenario["disruption_source"])
                    if scenario["disruption_source"] else None
                ),
            }
            row.update(metrics)
            for component in (
                "Queue_Passenger_Minutes",
                "Onboard_Passenger_Minutes",
                "Walking_Passenger_Minutes",
                "Total_Travel_Passenger_Minutes",
            ):
                row[f"Disruption_{component}"] = (
                    scenario["disruption_metrics"][component]
                )
            row["PT_Reduction_Passenger_Minutes"] = (
                scenario["disruption_metrics"]["Total_Travel_Passenger_Minutes"]
                - metrics["Total_Travel_Passenger_Minutes"]
            )
            for metric in (
                "Average_Waiting_Time_min",
                "Average_InVehicle_Time_min",
                "Average_Walking_Time_min",
                "Average_Total_Travel_Time_min",
            ):
                row[f"Disruption_{metric}"] = scenario["disruption_metrics"][metric]
            rows.append(row)

    if missing and not args.allow_incomplete:
        write_json(
            args.output_dir / "evaluation" / "missing_results.json",
            {"missing_count": len(missing), "missing": missing},
        )
        raise RuntimeError(f"{len(missing)} incident-policy results are missing")
    if not rows:
        return 0
    result = pd.DataFrame(rows).sort_values(["IncidentID", "Policy"])
    evaluation_dir = args.output_dir / "evaluation"
    write_csv(result, evaluation_dir / "incident_policy_results.csv")

    metrics = [
        "RI", "RI_Improvement", "Average_Waiting_Time_min",
        "Average_InVehicle_Time_min", "Average_Walking_Time_min",
        "Average_Total_Travel_Time_min",
    ]
    comparison = result.pivot(index="IncidentID", columns="Policy", values=metrics)
    comparison.columns = [f"{metric}_{policy}" for metric, policy in comparison.columns]
    comparison = comparison.reset_index()
    required = {f"{metric}_{policy}" for metric in metrics for policy in ("Historical", "ML", "Oracle")}
    if required.issubset(comparison.columns):
        comparison["ML_minus_Historical_RI"] = (
            comparison["RI_ML"] - comparison["RI_Historical"]
        )
        comparison["Historical_minus_ML_Waiting_min"] = (
            comparison["Average_Waiting_Time_min_Historical"]
            - comparison["Average_Waiting_Time_min_ML"]
        )
        comparison["Historical_minus_ML_TotalTravel_min"] = (
            comparison["Average_Total_Travel_Time_min_Historical"]
            - comparison["Average_Total_Travel_Time_min_ML"]
        )
        comparison["ML_minus_Oracle_TotalTravel_Regret_min"] = (
            comparison["Average_Total_Travel_Time_min_ML"]
            - comparison["Average_Total_Travel_Time_min_Oracle"]
        )
        # Historical/ML optimize a probability-weighted two-state policy,
        # whereas Oracle optimizes one realized state. Their raw Z values are
        # therefore not reported as like-for-like regret.
    metadata = eligible[
        ["IncidentID", "IncidentType", "StationCorridor", "RealizedState",
         "ZeroImpactRealized"]
    ]
    comparison = metadata.merge(comparison, on="IncidentID", how="left")
    write_csv(comparison, evaluation_dir / "incident_paired_comparison.csv")

    summary = (
        result.groupby("Policy", sort=False)[metrics]
        .agg(["count", "mean", "std", "median"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    policy_aggregate = result.groupby("Policy", sort=False).agg(
        Incident_Count=("IncidentID", "size"),
        Passenger_Demand_sum=("Passenger_Demand", "sum"),
        Queue_Passenger_Minutes_sum=("Queue_Passenger_Minutes", "sum"),
        Onboard_Passenger_Minutes_sum=("Onboard_Passenger_Minutes", "sum"),
        Walking_Passenger_Minutes_sum=("Walking_Passenger_Minutes", "sum"),
        Total_Travel_Passenger_Minutes_sum=(
            "Total_Travel_Passenger_Minutes", "sum"
        ),
        Disruption_Queue_Passenger_Minutes_sum=(
            "Disruption_Queue_Passenger_Minutes", "sum"
        ),
        Disruption_Onboard_Passenger_Minutes_sum=(
            "Disruption_Onboard_Passenger_Minutes", "sum"
        ),
        Disruption_Walking_Passenger_Minutes_sum=(
            "Disruption_Walking_Passenger_Minutes", "sum"
        ),
        Disruption_Total_Travel_Passenger_Minutes_sum=(
            "Disruption_Total_Travel_Passenger_Minutes", "sum"
        ),
        PT_Reduction_Passenger_Minutes_sum=(
            "PT_Reduction_Passenger_Minutes", "sum"
        ),
        Passenger_Time_Defined_Incident_Count=(
            "Passenger_Time_Defined", "sum"
        ),
    ).reset_index()
    policy_aggregate["Passenger_Time_Defined_Incident_Count"] = (
        policy_aggregate["Passenger_Time_Defined_Incident_Count"].astype(int)
    )
    policy_aggregate["Passenger_Time_Undefined_Incident_Count"] = (
        policy_aggregate["Incident_Count"]
        - policy_aggregate["Passenger_Time_Defined_Incident_Count"]
    )
    zero_demand_counts = (
        result["Passenger_Demand"].le(0.0)
        .groupby(result["Policy"], sort=False)
        .sum()
        .astype(int)
        .rename("Zero_Demand_Incident_Count")
        .reset_index()
    )
    policy_aggregate = policy_aggregate.merge(
        zero_demand_counts, on="Policy", how="left", validate="one_to_one"
    )
    aggregate_denominator = policy_aggregate["Passenger_Demand_sum"].replace(
        0.0, np.nan
    )
    policy_aggregate["Aggregate_Average_Total_Travel_Time_min"] = (
        policy_aggregate["Total_Travel_Passenger_Minutes_sum"]
        / aggregate_denominator
    )
    policy_aggregate["Aggregate_Disruption_Average_Total_Travel_Time_min"] = (
        policy_aggregate["Disruption_Total_Travel_Passenger_Minutes_sum"]
        / aggregate_denominator
    )
    policy_aggregate["Aggregate_PT_Reduction_min"] = (
        policy_aggregate["PT_Reduction_Passenger_Minutes_sum"]
        / aggregate_denominator
    )
    summary = summary.merge(
        policy_aggregate, on="Policy", how="left", validate="one_to_one"
    )
    write_csv(summary, evaluation_dir / "policy_summary.csv")
    type_summary = (
        result.groupby(["IncidentType", "Policy"], sort=False)[metrics]
        .agg(["count", "mean", "std"])
    )
    type_summary.columns = [
        f"{metric}_{stat}" for metric, stat in type_summary.columns
    ]
    write_csv(type_summary.reset_index(), evaluation_dir / "policy_by_incident_type.csv")
    write_json(
        evaluation_dir / "evaluation_audit.json",
        {
            "schema": SCHEMA,
            "evaluation_schema": EVALUATION_SCHEMA,
            "rail_state_policy": rail_state_policy(),
            "eligible_incidents": len(eligible),
            "evaluated_incident_policy_rows": len(result),
            "complete_incidents": int(
                result.groupby("IncidentID")["Policy"].nunique().eq(3).sum()
            ),
            "zero_impact_realized_d30_retained": int(
                eligible["ZeroImpactRealized"].sum()
            ),
            "missing_results": len(missing),
            "incident_weighting": "uniform",
            "reported_metric_probability_weighting": "none",
            "duration_probabilities_role": (
                "policy optimization inputs only; reported outcomes use each "
                "incident's realized branch"
            ),
            "probability_source": PROBABILITY_SOURCE,
            "historical_duration_counts": HISTORICAL_COUNTS,
            "historical_duration_weights": HISTORICAL_WEIGHTS,
            "oof_probability_source": (
                "nested outer-fold held-out LightGBM probabilities"
            ),
            "stage1_allocation_dir": str(args.allocation_dir),
            "evaluation_branch": "realized D30/D60 state",
            "ri_od_scope": "full",
            "ri_bridging_support": {
                "Historical": (
                    "D30/D60 common support plus Normal outside support"
                    if NONANTICIPATIVITY_MODE
                    == PASSENGER_AND_BUS_NONANTICIPATIVITY
                    else "duration-specific physical support in the evaluated "
                    "branch plus Normal outside support"
                ),
                "ML": (
                    "D30/D60 common support plus Normal outside support"
                    if NONANTICIPATIVITY_MODE
                    == PASSENGER_AND_BUS_NONANTICIPATIVITY
                    else "duration-specific physical support in the evaluated "
                    "branch plus Normal outside support"
                ),
                "Oracle": "realized physical support plus Normal outside support",
            },
            "passenger_time_od_scope": "physical realized affected ODs",
            "passenger_time_denominator": (
                "affected Q+L+W inventory at t0-1 plus affected demand "
                "arriving at t>=t0"
            ),
            "zero_physical_demand_pt": "undefined (NaN averages)",
            "policy_summary_pt_aggregation": (
                "sum passenger-minutes divided by sum affected demand"
            ),
            "raw_z_oracle_regret_reported": False,
            "raw_z_policy_summary_reported": False,
            "raw_z_incident_audit_only": True,
            "raw_z_oracle_regret_reason": (
                "Historical/ML optimize a probability-weighted two-state "
                "objective; Oracle optimizes one realized duration state"
            ),
            "travel_time_components": ["Queue", "Load", "Walking"],
            "evaluation_start_minute": EVALUATION_START,
        },
    )
    print(
        f"OOF_EVALUATION rows={len(result)} incidents={result.IncidentID.nunique()} "
        f"missing={len(missing)} output={evaluation_dir}",
        flush=True,
    )
    return 0


def check(
    args, cases: pd.DataFrame, preflight: dict | None = None
) -> int:
    preflight = preflight or validate_preflight(args, cases)
    eligible = cases[cases["Eligible"]]
    groups = list(eligible.groupby(["Hour", "Station"], sort=True))
    complete = sum(group_complete(args, block) for _, block in groups)
    payload = {
        "schema": SCHEMA,
        "formulation_mode": FORMULATION_MODE,
        "nonanticipativity_mode": NONANTICIPATIVITY_MODE,
        "passenger_recourse": PASSENGER_RECOURSE,
        "rail_state_policy": rail_state_policy(),
        "oof_incidents": len(cases),
        "eligible_incidents": len(eligible),
        "zero_impact_realized_retained": int(
            eligible["ZeroImpactRealized"].sum()
        ),
        "station_hour_groups": len(groups),
        "completed_groups": complete,
        "pending_groups": len(groups) - complete,
        "target_mip_gap": args.mip_gap,
        "accept_incumbent": bool(args.accept_incumbent),
        "completion_policy": preflight["completion_policy"],
        "time_limit_seconds": args.time_limit_seconds,
        "workers": args.workers,
        "threads_per_worker": args.threads,
        "solver_profile": preflight["solver_profile"],
        "probability_source": PROBABILITY_SOURCE,
        "historical_counts": HISTORICAL_COUNTS,
        "historical_weights": HISTORICAL_WEIGHTS,
        "allocation_dir": str(args.allocation_dir),
        "base_stage2_schema": BASE_STAGE2_SCHEMA,
        "job_counts": preflight["job_counts"],
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_batch(args, cases: pd.DataFrame) -> int:
    eligible = cases[cases["Eligible"]]
    groups = [
        (int(hour), str(station), block)
        for (hour, station), block in eligible.groupby(
            ["Hour", "Station"], sort=True
        )
    ]
    pending = [
        (hour, station) for hour, station, block in groups
        if not group_complete(args, block)
    ]
    status_path = args.output_dir / "BatchStatus" / "batch_progress.json"
    write_json(
        status_path,
        {
            "state": "RUNNING",
            "schema": SCHEMA,
            "formulation_mode": FORMULATION_MODE,
            "rail_state_policy": rail_state_policy(),
            "total_groups": len(groups),
            "pending_groups": len(pending),
            "eligible_incidents": len(eligible),
            "workers": args.workers,
            "threads_per_worker": args.threads,
            "target_mip_gap": args.mip_gap,
            "accept_incumbent": bool(args.accept_incumbent),
            "time_limit_seconds": args.time_limit_seconds,
            "solver_profile": {
                "mip_focus": args.mip_focus,
                "heuristics": args.heuristics,
                "cuts": args.cuts,
            },
            "allocation_dir": str(args.allocation_dir),
            "probability_source": PROBABILITY_SOURCE,
            "job_counts": planned_job_counts(cases),
        },
    )
    failures = []
    completed_now = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_group_subprocess, args, hour, station):
            (hour, station)
            for hour, station in pending
        }
        for future in as_completed(futures):
            hour, station = futures[future]
            try:
                code, stem = future.result()
            except Exception as exc:
                code, stem = -1, f"H{hour:02d}_{station}"
                failures.append({"group": stem, "error": repr(exc)})
            if code != 0:
                failures.append({"group": stem, "exit_code": code})
            else:
                completed_now += 1
            write_json(
                status_path,
                {
                    "state": "RUNNING",
                    "schema": SCHEMA,
                    "formulation_mode": FORMULATION_MODE,
                    "rail_state_policy": rail_state_policy(),
                    "total_groups": len(groups),
                    "initial_pending_groups": len(pending),
                    "completed_this_run": completed_now,
                    "failures": failures,
                    "last_group": stem,
                    "workers": args.workers,
                    "threads_per_worker": args.threads,
                },
            )
    remaining = sum(not group_complete(args, block) for _, _, block in groups)
    state = "FINISHED" if remaining == 0 else "INCOMPLETE"
    write_json(
        status_path,
        {
            "state": state,
            "schema": SCHEMA,
            "formulation_mode": FORMULATION_MODE,
            "rail_state_policy": rail_state_policy(),
            "total_groups": len(groups),
            "completed_groups": len(groups) - remaining,
            "remaining_groups": remaining,
            "failures": failures,
            "workers": args.workers,
            "threads_per_worker": args.threads,
            "target_mip_gap": args.mip_gap,
            "accept_incumbent": bool(args.accept_incumbent),
            "time_limit_seconds": args.time_limit_seconds,
            "solver_profile": {
                "mip_focus": args.mip_focus,
                "heuristics": args.heuristics,
                "cuts": args.cuts,
            },
            "allocation_dir": str(args.allocation_dir),
            "probability_source": PROBABILITY_SOURCE,
        },
    )
    evaluate(args, cases)
    return 0 if remaining == 0 else 2


def main() -> int:
    args = parse_args()
    activate_nonanticipativity_mode(args.nonanticipativity_mode)
    cases_path = args.output_dir / "inputs" / "oof_incident_cases.csv"
    if args.prepare_only or not cases_path.is_file():
        cases = prepare(args)
    else:
        cases = pd.read_csv(cases_path, low_memory=False)
        for column in (
            "Eligible", "ZeroImpactRealized", "RealizedImpactAvailable",
            "D60Support", "InStage1Window",
        ):
            cases[column] = cases[column].astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )
    preflight = validate_preflight(args, cases)
    if args.prepare_only:
        return check(args, cases, preflight)
    if args.check_only:
        return check(args, cases, preflight)
    if args.evaluate_only:
        return evaluate(args, cases)
    if args.single_group:
        try:
            return run_single_group(args, cases)
        except Exception:
            traceback.print_exc()
            return 1
    return run_batch(args, cases)


if __name__ == "__main__":
    raise SystemExit(main())
