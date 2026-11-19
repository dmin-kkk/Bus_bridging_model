"""Run the paper's Stage-2 direct full-pool MILP.

The file is both the resumable batch launcher and the isolated single-case
worker.  It imports only the vendored modules under ``_core/stage2`` and reads
only inputs or upstream results inside ``Paper_Execution_Pipeline``.
"""
from __future__ import annotations

import argparse
import csv
import gc
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

import pandas as pd

from pipeline_config import (
    DISPATCH_BUFFER_MINUTES,
    DURATIONS,
    OPERATIONAL_303_DURATION_COUNTS,
    OPERATIONAL_303_DURATION_WEIGHTS,
    FLEET_SIZE as FLEET_TOTAL,
    DEPOT_ACCESS_BOUND_MIN as DEPOT_ACCESS_BOUND,
    MIN_TRIP_BOARDING,
    OD_HOURLY_MIN_TRIPS,
    ROUTE_FILENAME,
    SCENARIO_LOGIC_VERSION as LOGIC_VERSION,
    WALKING_MINUTES,
    WINDOWS,
)
from walking_times import load_uniform_walking_minutes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bbconfig import PACKAGE_ROOT, cfg  # noqa: E402


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parent
RESULTS = cfg.results_dir
CORE_DIR = ROOT / "_core" / "stage2"
NORMAL_DIR = RESULTS / cfg.get("paths.normal")
DISRUPTION_DIR = RESULTS / cfg.get("paths.disruptions")
ROUTE_DIR = RESULTS / cfg.get("paths.routes")
STAGE1_DIR = RESULTS / cfg.get("paths.stage1")
OUTPUT_DIR = RESULTS / cfg.get("paths.stage2_proposed")
STAGE2 = cfg.get("stage2")
SOLVER = cfg.get("stage2.solver")
PARALLEL = cfg.get("parallel.stage2")
ROUTE_SOURCE = str(cfg.get("routes.source_mode"))
BUS_BOARD_MODE = str(cfg.get("routes.bus_boarding_mode"))

ADAPTIVE_FILE = CORE_DIR / "stage2_adaptive.py"
MILP_CORE_FILE = CORE_DIR / "milp_core.py"
SCENARIO_STATS = DISRUPTION_DIR / "_scenario_stats.csv"
ROUTE_FILE = ROUTE_DIR / ROUTE_FILENAME
DEPOT_FILE = cfg.path("inputs.depots")
SCHEDULE_FILE = cfg.path("inputs.timetable")
OD_FILE = cfg.path("inputs.od_matrix")
DISTANCE_FILE = cfg.path("inputs.road_times")
CORRIDOR_FILE = cfg.path("inputs.corridors")
COORDINATE_FILE = cfg.path("inputs.stations")
WALKING_TIME_FILE = cfg.path("inputs.walking_times")

BUS_ONLY_NONANTICIPATIVITY = "bus-only"
PASSENGER_AND_BUS_NONANTICIPATIVITY = "passenger-and-bus"
BUS_ONLY_WAIT_AND_SEE = "BUS_ONLY_WAIT_AND_SEE"
PASSENGER_AND_BUS_SHARED_RAW_DURATION_RAIL = (
    "PASSENGER_AND_BUS_SHARED_RAW_DURATION_RAIL"
)
SCHEMA = "paper-stage2-full-pool-joint-30-60-bus-only-na-v5"
PASSENGER_AND_BUS_RAW_RAIL_SCHEMA = (
    "paper-stage2-full-pool-joint-30-60-passenger-bus-na-raw-rail-v5"
)


def schema_for_mode(mode: str) -> str:
    if mode == BUS_ONLY_NONANTICIPATIVITY:
        return SCHEMA
    if mode == PASSENGER_AND_BUS_NONANTICIPATIVITY:
        return PASSENGER_AND_BUS_RAW_RAIL_SCHEMA
    raise ValueError(f"unknown non-anticipativity mode: {mode!r}")


def formulation_mode_for(mode: str) -> str:
    if mode == BUS_ONLY_NONANTICIPATIVITY:
        return BUS_ONLY_WAIT_AND_SEE
    if mode == PASSENGER_AND_BUS_NONANTICIPATIVITY:
        return PASSENGER_AND_BUS_SHARED_RAW_DURATION_RAIL
    raise ValueError(f"unknown non-anticipativity mode: {mode!r}")


def rail_state_policy_for(mode: str) -> str:
    return (
        "PRESERVE_EACH_DURATION_INPUT_BRANCH"
        if mode == PASSENGER_AND_BUS_NONANTICIPATIVITY
        else "DURATION_SPECIFIC_WAIT_AND_SEE"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct full-pool Stage-2 MILP for all D30/D60 cases."
    )
    parser.add_argument("--single-case", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--nonanticipativity-mode",
        choices=(BUS_ONLY_NONANTICIPATIVITY,
                 PASSENGER_AND_BUS_NONANTICIPATIVITY),
        default=STAGE2["nonanticipativity_mode"],
        help=("bus-only shares y/u/psi while retaining duration-specific "
              "passenger recourse and rail state"),
    )
    parser.add_argument(
        "--exact-state-compression", action=argparse.BooleanOptionalAction,
        default=bool(STAGE2["exact_state_compression"]),
        help="use the lossless Stage-1 state projection in Stage 2",
    )
    parser.add_argument(
        "--exact-trip-cell-pruning", action=argparse.BooleanOptionalAction,
        default=bool(STAGE2["exact_trip_cell_pruning"]),
        help="pre-remove zero-fleet and minimum-service-infeasible trip cells",
    )
    parser.add_argument(
        "--local-routes-only", action="store_true",
        help="restrict the predefined route library to local routes",
    )
    parser.add_argument("--window", type=int)
    parser.add_argument("--station")
    parser.add_argument(
        "--oracle-duration", type=int, choices=DURATIONS,
        help=("solve only the realized duration branch; this removes "
              "cross-duration non-anticipativity for perfect information"),
    )
    parser.add_argument("--start-window", type=int, default=WINDOWS[0])
    parser.add_argument("--end-window", type=int, default=WINDOWS[-1])
    parser.add_argument(
        "--buffer", type=int, default=DISPATCH_BUFFER_MINUTES
    )
    parser.add_argument("--time-limit-seconds", type=float,
                        default=float(SOLVER["time_limit_s"]))
    parser.add_argument("--mip-gap", type=float, default=float(SOLVER["mip_gap"]))
    parser.add_argument("--threads", type=int, default=int(PARALLEL["threads"]))
    parser.add_argument("--mip-focus", type=int, default=int(SOLVER["mip_focus"]))
    parser.add_argument("--heuristics", type=float,
                        default=float(SOLVER["heuristics"]))
    parser.add_argument("--cuts", type=int, default=int(SOLVER["cuts"]))
    parser.add_argument("--mip-start-policy", type=Path)
    parser.add_argument(
        "--export-model-dir", type=Path, default=default_export_dir(),
        help="write .mps/.sol/.json of each solved case here "
             "(default: stage2.export_model_dir; empty = off)",
    )
    parser.add_argument(
        "--workers", type=int, default=int(PARALLEL["workers"]),
        help="number of independent station cases solved concurrently",
    )
    parser.add_argument(
        "--allocation-dir",
        type=Path,
        default=STAGE1_DIR / "Allocation",
        help="directory containing DepotAlloc_HH.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="isolated Stage-2 result directory",
    )
    parser.add_argument(
        "--case-label",
        default=case_label(),
        help="audit label stored in checkpoints and summaries",
    )
    parser.add_argument(
        "--duration-probability-source",
        choices=("operational303",),
        default=str(cfg.get("probability.duration_source")),
        help=("duration prior; operational303 uses D30=208/303 and "
              "D60=95/303"),
    )
    parser.add_argument(
        "--duration-p60", type=float,
        help=("override the joint D30/D60 objective weights with this "
              "D60 probability"),
    )
    args = parser.parse_args()
    if args.start_window > args.end_window:
        parser.error("--start-window must not exceed --end-window")
    if args.start_window not in WINDOWS or args.end_window not in WINDOWS:
        parser.error(
            f"planning windows must lie within "
            f"{WINDOWS[0]:02d}--{WINDOWS[-1]:02d}"
        )
    if args.buffer < 0:
        parser.error("--buffer must be nonnegative")
    if args.time_limit_seconds <= 0:
        parser.error("--time-limit-seconds must be positive")
    if args.duration_p60 is not None and not 0 <= args.duration_p60 <= 1:
        parser.error("--duration-p60 must be in [0, 1]")
    if args.duration_p60 is not None and args.oracle_duration is not None:
        parser.error("--duration-p60 cannot be combined with --oracle-duration")
    if not 0 <= args.mip_gap < 1:
        parser.error("--mip-gap must be in [0, 1)")
    if args.threads < 0:
        parser.error("--threads must be nonnegative")
    if args.mip_focus not in (0, 1, 2, 3):
        parser.error("--mip-focus must be one of 0, 1, 2, 3")
    if not 0 <= args.heuristics <= 1:
        parser.error("--heuristics must be in [0, 1]")
    if args.cuts not in (-1, 0, 1, 2, 3):
        parser.error("--cuts must be one of -1, 0, 1, 2, 3")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.single_case and (args.window is None or not args.station):
        parser.error("--single-case requires --window and --station")
    if args.oracle_duration is not None and not args.single_case:
        parser.error("--oracle-duration requires --single-case")
    if not args.single_case and args.mip_start_policy is not None:
        parser.error("--mip-start-policy is supported only with --single-case")
    args.allocation_dir = args.allocation_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.export_model_dir is not None and str(args.export_model_dir).strip():
        args.export_model_dir = args.export_model_dir.resolve()
    else:
        args.export_model_dir = None
    if args.mip_start_policy is not None:
        args.mip_start_policy = args.mip_start_policy.resolve()
        if not args.mip_start_policy.is_file():
            parser.error(f"MIP-start policy not found: {args.mip_start_policy}")
    args.case_label = str(args.case_label).strip().upper()
    if not args.case_label:
        parser.error("--case-label cannot be empty")
    return args


def default_export_dir() -> Path | None:
    """stage2.export_model_dir resolved against the package root (None = off)."""
    value = str(STAGE2.get("export_model_dir", "") or "").strip()
    return (PACKAGE_ROOT / value).resolve() if value else None


def case_label(benchmark: bool = False) -> str:
    """Case label built from the profile (stage2.*_case_label templates).

    Paper: PROPOSED_LIGHTGBM_BOARD1_BUF30 / FIXED_SPE30_LIGHTGBM_BOARD1_BUF30.
    """
    template = STAGE2["benchmark_case_label" if benchmark else "proposed_case_label"]
    return str(template).format(
        model=str(cfg.get("prediction.full_grid.model_family")).upper(),
        boarding=f"{float(MIN_TRIP_BOARDING):g}",
        buffer=int(DISPATCH_BUFFER_MINUTES),
        depot=str(cfg.get("evaluation.benchmark_depot")).upper(),
        fleet=int(cfg.get("evaluation.benchmark_fleet")),
    ).upper()


def load_mip_start_policy(path: Path | None) -> dict | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    required = {"TripID", "SharedDurations"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MIP-start policy missing columns: {sorted(missing)}")
    selected = []
    for row in frame.itertuples(index=False):
        durations = tuple(
            int(float(value))
            for value in str(row.SharedDurations).split("|")
            if str(value).strip()
        )
        selected.append({"trip_id": str(row.TripID), "durations": durations})
    return {
        "selected_trip_cells": selected,
        "source_policy": str(path),
    }


def safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def duration_prior(source: str) -> tuple[dict[int, int], dict[int, float]]:
    if source == "operational303":
        return (dict(OPERATIONAL_303_DURATION_COUNTS),
                dict(OPERATIONAL_303_DURATION_WEIGHTS))
    raise ValueError(f"unknown duration probability source: {source}")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot create module specification for {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def load_local_stage2_modules():
    """Load only the two source snapshots vendored in this pipeline."""
    core_root = CORE_DIR.resolve()
    for path in (ADAPTIVE_FILE, MILP_CORE_FILE):
        if path.resolve().parent != core_root:
            raise RuntimeError(f"Stage-2 module escaped local core: {path}")
    sys.path.insert(0, str(core_root))
    adaptive = load_module("stage2_adaptive", ADAPTIVE_FILE)
    core = load_module("paper_stage2_milp_core", MILP_CORE_FILE)
    for module in (adaptive, core):
        if Path(module.__file__).resolve().parent != core_root:
            raise RuntimeError(f"External Python module loaded: {module.__file__}")
    return core, adaptive


def load_depots(path: Path) -> tuple[str, ...]:
    frame = pd.read_csv(path)
    if "Code" not in frame.columns:
        raise ValueError(f"{path}: missing Code column")
    values = tuple(sorted({
        str(value).strip().upper()
        for value in frame["Code"]
        if str(value).strip()
    }))
    if not values:
        raise ValueError(f"{path}: no depot candidates")
    return values


def allocation_path(window: int, allocation_dir: Path | None = None) -> Path:
    root = allocation_dir or (STAGE1_DIR / "Allocation")
    return root / f"DepotAlloc_{window:02d}.csv"


def load_allocation(path: Path, depots: tuple[str, ...]) -> dict[str, int]:
    frame = pd.read_csv(path)
    if not {"Depot", "Count"}.issubset(frame.columns):
        raise ValueError(f"{path}: expected Depot,Count columns")
    counts = {depot: 0 for depot in depots}
    seen = set()
    for row in frame.itertuples(index=False):
        depot = str(row.Depot).strip().upper()
        raw = float(row.Count)
        count = int(round(raw))
        invalid = (
            depot in seen or depot not in counts or count < 0
            or abs(raw - count) > 1e-9
        )
        if invalid:
            raise ValueError(f"{path}: invalid allocation row for {depot}")
        seen.add(depot)
        counts[depot] = count
    if sum(counts.values()) != FLEET_TOTAL:
        raise ValueError(
            f"{path}: fleet total is {sum(counts.values())}, "
            f"expected {FLEET_TOTAL}"
        )
    return counts


def discover_jobs(path: Path, start: int, end: int) -> list[tuple[int, str]]:
    frame = pd.read_csv(path)
    required = {"window", "target", "duration", "checkpoint", "logic_version"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"scenario stats missing columns: {sorted(missing)}")
    frame["hour"] = frame["window"].astype(str).str.split("_").str[0].astype(int)
    valid = frame[
        frame["hour"].between(start, end)
        & frame["duration"].isin(DURATIONS)
        & frame["checkpoint"].eq("EXTRACTED")
        & frame["logic_version"].eq(LOGIC_VERSION)
    ].copy()
    if valid.duplicated(["hour", "target", "duration"], keep=False).any():
        raise ValueError("duplicate current scenario rows found")
    grouped = valid.groupby(["hour", "target"])["duration"].agg(set)
    jobs = []
    longest = max(DURATIONS)
    for (hour, station), durations in grouped.items():
        if longest not in durations:
            raise ValueError(f"{hour:02d}/{station}: D{longest} input is absent")
        jobs.append((int(hour), str(station).strip().upper()))
    jobs = sorted(jobs)
    if not jobs:
        raise ValueError(
            f"no current D{longest} jobs found for windows {start}-{end}")
    return jobs


def required_paths(
    start: int, end: int, allocation_dir: Path | None = None
) -> tuple[list[Path], list[Path]]:
    static = [
        ADAPTIVE_FILE,
        MILP_CORE_FILE,
        DEPOT_FILE,
        SCHEDULE_FILE,
        OD_FILE,
        DISTANCE_FILE,
        CORRIDOR_FILE,
        COORDINATE_FILE,
        WALKING_TIME_FILE,
    ]
    generated = [SCENARIO_STATS, ROUTE_FILE]
    # Profile windows inside [start, end] (time.windows need not be consecutive).
    for window in (w for w in WINDOWS if start <= w <= end):
        generated.extend([
            NORMAL_DIR / f"Normal_{window:02d}_{window + 4:02d}.csv",
            allocation_path(window, allocation_dir),
        ])
    return static, generated


def check_only(args: argparse.Namespace) -> int:
    start = args.window if args.single_case else args.start_window
    end = args.window if args.single_case else args.end_window
    static, generated = required_paths(start, end, args.allocation_dir)
    missing_static = [relative(path) for path in static if not path.is_file()]
    missing_generated = [relative(path) for path in generated if not path.is_file()]
    walking_input = None
    if WALKING_TIME_FILE.is_file() and DISTANCE_FILE.is_file():
        walking_input = load_uniform_walking_minutes(
            WALKING_TIME_FILE,
            DISTANCE_FILE,
            WALKING_MINUTES,
        )
    jobs = None
    stats_error = None
    if SCENARIO_STATS.is_file():
        try:
            jobs = len(discover_jobs(SCENARIO_STATS, start, end))
        except Exception as exc:  # diagnostic mode reports rather than raises
            stats_error = str(exc)
    prior_counts, prior_weights = duration_prior(
        args.duration_probability_source)
    payload = {
        "check": "stage2",
        "schema": schema_for_mode(args.nonanticipativity_mode),
        "formulation_mode": formulation_mode_for(
            args.nonanticipativity_mode),
        "pipeline_root": str(ROOT),
        "local_core": ADAPTIVE_FILE.is_file() and MILP_CORE_FILE.is_file(),
        "missing_static_inputs": missing_static,
        "missing_upstream_results": missing_generated,
        "scenario_stats_error": stats_error,
        "discovered_jobs": jobs,
        "ready_to_run": not missing_static and not missing_generated
        and stats_error is None,
        "solver": "FULL_POOL_MILP",
        "column_generation": False,
        "nonanticipativity_mode": args.nonanticipativity_mode,
        "bus_decision_nonanticipativity": True,
        "passenger_recourse": (
            "WAIT_AND_SEE"
            if args.nonanticipativity_mode == BUS_ONLY_NONANTICIPATIVITY
            else "NONANTICIPATIVE"),
        "duration_specific_physical_omega": (
            args.nonanticipativity_mode == BUS_ONLY_NONANTICIPATIVITY),
        "duration_specific_rail_state": True,
        "raw_branch_rail_state_policy": rail_state_policy_for(
            args.nonanticipativity_mode),
        "durations": list(DURATIONS),
        "duration_probability_source": args.duration_probability_source,
        "duration_counts": prior_counts,
        "duration_weights": prior_weights,
        "fleet": FLEET_TOTAL,
        "depot_access_bound_minutes": DEPOT_ACCESS_BOUND,
        "buffer_minutes": args.buffer,
        "min_trip_boarding": MIN_TRIP_BOARDING,
        "walking_minutes": WALKING_MINUTES,
        "walking_time_input": walking_input,
        "case_label": args.case_label,
        "allocation_dir": str(args.allocation_dir),
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    # Check-only is diagnostic and remains usable before upstream stages run.
    return 0 if not missing_static else 2


def validate_runtime_inputs(
    start: int, end: int, allocation_dir: Path | None = None
) -> None:
    static, generated = required_paths(start, end, allocation_dir)
    missing = [path for path in static + generated if not path.is_file()]
    if missing:
        listed = "\n".join(f"  - {relative(path)}" for path in missing)
        raise FileNotFoundError(f"Stage-2 prerequisites are missing:\n{listed}")


def summary_path(output: Path, window: int, station: str) -> Path:
    return output / "Performance" / f"FullMILP_{window:02d}_{station}_summary.json"


def effective_mip_gap(stats: dict) -> float:
    raw_gap = stats.get("MIPGap")
    if raw_gap is not None:
        return float(raw_gap)
    status = str(stats.get("Status", "")).strip().upper()
    integer_variables = int(stats.get("NumIntVars", -1))
    objective = float(stats.get("ObjVal", math.nan))
    bound = float(stats.get("ObjBound", math.nan))
    if (
        status == "OPTIMAL"
        and integer_variables == 0
        and math.isfinite(objective)
        and math.isfinite(bound)
        and math.isclose(objective, bound, rel_tol=1e-9, abs_tol=1e-8)
    ):
        return 0.0
    return math.inf


def _matches_nonanticipativity_audit(value: dict, mode: str) -> bool:
    stats = value.get("stats") or {}
    if mode == PASSENGER_AND_BUS_NONANTICIPATIVITY:
        tolerance = 1e-5

        def within_tolerance(name: str) -> bool:
            try:
                result = float(stats.get(name, math.inf))
            except (TypeError, ValueError):
                return False
            return math.isfinite(result) and result <= tolerance

        return bool(
            stats.get("NonanticipativityMode")
            == PASSENGER_AND_BUS_NONANTICIPATIVITY
            and stats.get("FormulationMode")
            == PASSENGER_AND_BUS_SHARED_RAW_DURATION_RAIL
            and stats.get("BusDecisionInformationSetNonanticipative") is True
            and stats.get("VehicleChainInformationSetNonanticipative") is True
            and stats.get("PassengerFlowVariableSharing")
            and stats.get("PassengerRecourse") == "NONANTICIPATIVE"
            and int(stats.get("PassengerSharedEqualityConstrs", -1)) == 0
            and stats.get("PassengerStateNonanticipativeByConservation") is True
            and stats.get("ExactStateCompression") is True
            and stats.get("WalkingEliminatedExactly") is True
            and stats.get("DestinationStatesEliminatedExactly") is True
            and stats.get("PassengerTransferFlowProjectedExactly") is True
            and stats.get("PassengerExitFlowProjectedExactly") is True
            and stats.get(
                "PassengerTransferFlowInformationSetNonanticipative") is True
            and stats.get(
                "PassengerExitFlowInformationSetNonanticipative") is True
            and within_tolerance("PassengerSharedQueueMaxAbsDiff")
            and within_tolerance("PassengerSharedLoadMaxAbsDiff")
            and within_tolerance("PassengerSharedTrainLoadMaxAbsDiff")
            and within_tolerance("PassengerSharedBusLoadMaxAbsDiff")
            and within_tolerance("PassengerSharedTransferMaxAbsDiff")
            and within_tolerance("PassengerSharedExitMaxAbsDiff")
            and stats.get("RailCommonEventTimetable") is True
            and stats.get(
                "RailEventSupportInformationSetNonanticipative") is True
            and int(stats.get("RailEventSupportMismatchCount", -1)) == 0
            and stats.get("TerminationAlightEventsVerified") is True
            and stats.get(
                "TerminationActionSupportInformationSetNonanticipative")
            is True
            and int(stats.get(
                "TerminationActionSupportMismatchCount", -1)) == 0
            and stats.get("PassengerCapacityThetaPresent") is True
            and stats.get(
                "PassengerCapacityThetaMatchesClearanceEvents") is True
            and stats.get("PassengerCapacityWithdrawalPolicy")
            == "ZERO_FROM_RAW_BRANCH_PHYSICAL_CLEARANCE_EVENT"
            and stats.get("RawBranchCanceledTrainsPreserved") is True
            and stats.get("RawBranchCancellationStartsPreserved") is True
            and stats.get("RawBranchWithdrawalThetaPreserved") is True
            and stats.get("RawBranchTerminationEventsPreserved") is True
            and stats.get(
                "RawBranchPassengerCapacityThetaPreserved") is True
            and stats.get("RawBranchRailStatePreserved") is True
            and stats.get("ZeroImpactRailReconstructionsVerified") is True
            and stats.get("TrainAvailabilityInformationSetNonanticipative")
            is False
            and stats.get("DurationSpecificPhysicalOmega") is False
            and stats.get("DurationSpecificRailState") is True
        )
    return bool(
        value.get("nonanticipativity_mode") == BUS_ONLY_NONANTICIPATIVITY
        and stats.get("NonanticipativityMode")
        == BUS_ONLY_NONANTICIPATIVITY
        and stats.get("FormulationMode") == BUS_ONLY_WAIT_AND_SEE
        and stats.get("BusDecisionInformationSetNonanticipative") is True
        and stats.get("VehicleChainInformationSetNonanticipative") is True
        and stats.get("PassengerFlowVariableSharing") is False
        and stats.get("PassengerRecourse") == "WAIT_AND_SEE"
        and stats.get("TrainAvailabilityInformationSetNonanticipative")
        is False
        and stats.get("DurationSpecificPhysicalOmega") is True
        and stats.get("DurationSpecificRailState") is True
    )


def is_certified(
        output: Path, window: int, station: str, mip_gap: float,
        probability_source: str | None = None,
        nonanticipativity_mode: str = BUS_ONLY_NONANTICIPATIVITY) -> bool:
    path = summary_path(output, window, station)
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        actual_gap = effective_mip_gap(value.get("stats") or {})
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    decision_complete = all(
        (output / "DecisionVars" / f"Disrupt_{window:02d}_{station}_{duration}.csv").is_file()
        for duration in DURATIONS
    )
    policy_complete = (
        output / "Policies" / f"FullMILP_{window:02d}_{station}_policy.csv"
    ).is_file()
    chain_complete = (
        output / "Policies" /
        f"FullMILP_{window:02d}_{station}_chain_policy.csv"
    ).is_file()
    return bool(
        value.get("schema") == schema_for_mode(nonanticipativity_mode)
        and (
            nonanticipativity_mode == BUS_ONLY_NONANTICIPATIVITY
            or value.get("rail_state_policy")
            == rail_state_policy_for(nonanticipativity_mode)
        )
        and (probability_source is None or value.get(
            "duration_probability_source") == probability_source)
        and float(value.get("od_hourly_min_trips", -1.0))
        == OD_HOURLY_MIN_TRIPS
        and value.get("column_generation") is False
        and value.get("has_incumbent") is True
        and math.isfinite(actual_gap)
        and actual_gap <= mip_gap + 1e-9
        and decision_complete
        and policy_complete
        and chain_complete
        and _matches_nonanticipativity_audit(
            value, nonanticipativity_mode)
    )


def has_usable_result(
        output: Path, window: int, station: str,
        probability_source: str | None = None,
        nonanticipativity_mode: str = BUS_ONLY_NONANTICIPATIVITY) -> bool:
    """Return true when the one-hour solve produced a complete incumbent output."""
    path = summary_path(output, window, station)
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    decision_complete = all(
        (output / "DecisionVars" / f"Disrupt_{window:02d}_{station}_{duration}.csv").is_file()
        for duration in DURATIONS
    )
    policy_complete = (
        output / "Policies" / f"FullMILP_{window:02d}_{station}_policy.csv"
    ).is_file()
    chain_complete = (
        output / "Policies" /
        f"FullMILP_{window:02d}_{station}_chain_policy.csv"
    ).is_file()
    return bool(
        value.get("schema") == schema_for_mode(nonanticipativity_mode)
        and (
            nonanticipativity_mode == BUS_ONLY_NONANTICIPATIVITY
            or value.get("rail_state_policy")
            == rail_state_policy_for(nonanticipativity_mode)
        )
        and (probability_source is None or value.get(
            "duration_probability_source") == probability_source)
        and float(value.get("od_hourly_min_trips", -1.0))
        == OD_HOURLY_MIN_TRIPS
        and value.get("column_generation") is False
        and value.get("has_incumbent") is True
        and decision_complete
        and policy_complete
        and chain_complete
        and _matches_nonanticipativity_audit(
            value, nonanticipativity_mode)
    )


def policy_rows(result: dict) -> list[dict]:
    rows = []
    for trip in result.get("policy", []):
        row = dict(trip)
        row["SharedDurations"] = "|".join(map(str, trip["SharedDurations"]))
        for duration in DURATIONS:
            row[f"Boarded{duration}"] = trip.get("Boarded", {}).get(duration, 0.0)
        row.pop("Boarded", None)
        rows.append(row)
    return rows


def write_policy(path: Path, rows: list[dict]) -> None:
    fields = [
        "TripID", "SharedDurations", "Depot", "Route", "RouteType",
        "StartStation", "EndStation", "DispatchStart", "RunStart", "RunEnd",
        *(f"Boarded{duration}" for duration in DURATIONS),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_chain_policy(path: Path, rows: list[dict]) -> None:
    fields = [
        "DecisionType", "PredecessorTripID", "SuccessorTripID",
        "DecisionTime", "InformationCell", "Value",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for raw in rows:
            row = dict(raw)
            row["InformationCell"] = "|".join(map(
                str, row.get("InformationCell", ())))
            writer.writerow(row)
    os.replace(temporary, path)


def write_decision_variables(path: Path, rows) -> None:
    fields = [
        "Variable", "Omega", "Origin", "Dest", "Time", "StationID",
        "TrainID", "Value",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)
    os.replace(temporary, path)


def run_single_case(args: argparse.Namespace) -> int:
    import gurobipy as gp

    window = int(args.window)
    station = str(args.station).strip().upper()
    formulation_schema = schema_for_mode(args.nonanticipativity_mode)
    model_durations = (
        (int(args.oracle_duration),)
        if args.oracle_duration is not None else tuple(DURATIONS)
    )
    prior_counts, prior_weights = duration_prior(
        args.duration_probability_source)
    if args.oracle_duration is not None:
        model_weights = {int(args.oracle_duration): 1.0}
    elif args.duration_p60 is not None:
        # Short / long state of the two-state duration set P (time.durations_min).
        model_weights = {
            min(DURATIONS): 1.0 - float(args.duration_p60),
            max(DURATIONS): float(args.duration_p60),
        }
    else:
        model_weights = prior_weights
    validate_runtime_inputs(window, window, args.allocation_dir)
    walking_input = load_uniform_walking_minutes(
        WALKING_TIME_FILE,
        DISTANCE_FILE,
        WALKING_MINUTES,
    )
    walking_minutes = int(walking_input["walking_minutes"])
    prefix = f"FullMILP_{window:02d}_{station}"
    output = args.output_dir
    allocation_file = allocation_path(window, args.allocation_dir)
    summary = summary_path(output, window, station)
    progress = output / "Performance" / f"{prefix}_progress.json"
    policy_file = output / "Policies" / f"{prefix}_policy.csv"
    chain_policy_file = output / "Policies" / f"{prefix}_chain_policy.csv"
    started = time.time()
    environment = None

    def checkpoint(phase: str, **extra) -> None:
        write_json(progress, {
            "schema": formulation_schema,
            "status": "FINISHED" if phase == "FINISHED" else "RUNNING",
            "phase": phase,
            "solver": "FULL_POOL_MILP",
            "column_generation": False,
            "nonanticipativity_mode": args.nonanticipativity_mode,
            "formulation_mode": formulation_mode_for(
                args.nonanticipativity_mode),
            "rail_state_policy": rail_state_policy_for(
                args.nonanticipativity_mode),
            "window": window,
            "station": station,
            "durations": model_durations,
            "duration_weights": model_weights,
            "duration_p60_override": args.duration_p60,
            "oracle_duration": args.oracle_duration,
            "case_label": args.case_label,
            "exact_state_compression": bool(args.exact_state_compression),
            "elapsed_seconds": time.time() - started,
            **extra,
        })

    try:
        checkpoint("LOADING_INPUTS")
        os.environ["TSB_DEPOT_ACCESS_BOUND"] = str(DEPOT_ACCESS_BOUND)
        os.environ["TSB_MAX_BUSES"] = str(FLEET_TOTAL)
        os.environ["TSB_ROUTE_SOURCE"] = ROUTE_SOURCE
        os.environ["TSB_BUS_BOARD"] = BUS_BOARD_MODE
        os.environ["PAPER_OD_HOURLY_MIN_TRIPS"] = str(
            OD_HOURLY_MIN_TRIPS)
        core, adaptive = load_local_stage2_modules()
        # Load both raw duration files so the common D60 candidate pool remains
        # available; ``build_adaptive_group`` below controls which branch is
        # actually instantiated in the oracle MILP.
        core.DURATIONS = list(DURATIONS)
        core.DURATION_WEIGHT = dict(model_weights)
        core.DISRUPT_BUFFER = int(args.buffer)
        core.TIME_LIMIT = float(args.time_limit_seconds)
        core.MIP_GAP = float(args.mip_gap)
        core.DEPOT_ACCESS_BOUND_MIN = float(DEPOT_ACCESS_BOUND)
        core.MAX_TOTAL_BUSES = FLEET_TOTAL
        core.MIN_TRIP_BOARDING = float(MIN_TRIP_BOARDING)
        adaptive.WALKING_MINUTES = walking_minutes
        adaptive.EXACT_STATE_COMPRESSION = bool(
            args.exact_state_compression)
        adaptive.EXACT_TRIP_CELL_PRUNING = bool(
            args.exact_trip_cell_pruning)
        core.ROUTE_SOURCE = ROUTE_SOURCE
        core.BUS_BOARD_MODE = BUS_BOARD_MODE
        core.FILE_PREDEFINED_ROUTES = str(ROUTE_FILE)
        core.FILE_SCHEDULE = str(SCHEDULE_FILE)
        core.FILE_OD = str(OD_FILE)
        core.FILE_DISTANCE = str(DISTANCE_FILE)
        core.FILE_CORRIDOR = str(CORRIDOR_FILE)
        core.FILE_EXCEL = str(COORDINATE_FILE)
        core.DIR_NORMAL = str(NORMAL_DIR)
        core.DIR_DISRUPT = str(DISRUPTION_DIR)

        depots = load_depots(DEPOT_FILE)
        depot_counts = load_allocation(allocation_file, depots)
        schedule = core.load_schedule(core.FILE_SCHEDULE)
        od = core.load_od(core.FILE_OD)
        bus_tt = core.load_distance_matrix(core.FILE_DISTANCE)
        route_library = core.load_route_library(
            core.FILE_PREDEFINED_ROUTES,
            distance_path=core.FILE_DISTANCE,
            validate_forward_times=True,
        )
        if args.local_routes_only:
            route_library = {
                route_id: route
                for route_id, route in route_library.items()
                if str(route.get("route_type", "")).upper() == "LOC"
            }
        if not route_library:
            raise ValueError(f"No routes loaded from {ROUTE_FILE}")
        window_data = core.build_window_data(window, schedule, od)
        raw_scenarios = core.load_disruption_scenarios(window, window_data)
        available = sorted({
            int(item["duration"])
            for item in raw_scenarios
            if item["station"] == station
        })
        if max(DURATIONS) not in available:
            raise ValueError(
                f"{window:02d}/{station}: no current D{max(DURATIONS)} scenario")

        group = adaptive.build_adaptive_group(
            core, station, window, window_data, raw_scenarios, depots,
            bus_tt, route_library, durations=model_durations,
            buffer=args.buffer, trip_basis_duration=max(DURATIONS),
            nonanticipativity_mode=args.nonanticipativity_mode,
        )
        active = tuple(map(int, group["active"]))
        raw_active = tuple(map(int, group.get("raw_active", active)))
        if args.oracle_duration is not None:
            if (args.nonanticipativity_mode ==
                    PASSENGER_AND_BUS_NONANTICIPATIVITY and
                    set(active) != set(model_durations)):
                raise AssertionError(f"unexpected active durations: {active}")
            if any(duration not in model_durations for duration in active):
                raise AssertionError(f"unexpected active durations: {active}")
        elif max(DURATIONS) not in active or any(
            duration not in model_durations for duration in active
        ):
            raise AssertionError(f"unexpected active durations: {active}")
        checkpoint(
            "FULL_POOL_READY",
            active_raw_durations=raw_active,
            zero_impact_durations=[
                d for d in model_durations if d not in raw_active],
            common_affected_od_count=len(group.get("common_omega", ())),
            physical_affected_od_count={
                str(d): len(group.get(
                    "physical_affected_omega", {}).get(d, ()))
                for d in model_durations},
            causal_affected_od_count={
                str(d): len(group.get(
                    "causal_affected_omega", {}).get(d, ()))
                for d in model_durations},
            candidate_routes=len({trip["Route"] for trip in group["trips"].values()}),
            candidate_trips=len(group["trips"]),
            information_cell_variables=sum(len(cells) for cells in group["cells"].values()),
            raw_trip_cell_count=group.get("raw_trip_cell_count"),
            minimum_service_infeasible_trip_cells_removed=group.get(
                "minimum_service_infeasible_trip_cells_removed"),
            fleet_total=sum(depot_counts.values()),
        )

        environment = gp.Env(empty=True)
        environment.setParam("OutputFlag", 1)
        environment.setParam("Threads", args.threads)
        environment.start()
        mip_start = load_mip_start_policy(args.mip_start_policy)
        # Optional model export (stage2.export_model_dir; synthetic profile only).
        # getattr: evaluation scripts build their own argparse Namespace.
        export_dir = getattr(args, "export_model_dir", None)
        export_kwargs = {}
        if export_dir is not None:
            export_kwargs["export_prefix"] = (
                Path(export_dir) / f"{args.case_label}_H{window:02d}_{station}")
        result = adaptive.solve_stage2_adaptive(
            core, environment, depots, depot_counts, group, window_data,
            duration_weight=model_weights,
            mip_gap=args.mip_gap,
            time_limit=args.time_limit_seconds,
            output_flag=1,
            threads=args.threads,
            mip_start=mip_start,
            mip_focus=args.mip_focus,
            heuristics=args.heuristics,
            cuts=args.cuts,
            nonanticipativity_mode=args.nonanticipativity_mode,
            **export_kwargs,
        )
        stats = result.get("stats", {})
        has_incumbent = int(stats.get("SolCount", 0) or 0) > 0
        gap = effective_mip_gap(stats)
        if math.isfinite(gap):
            stats["MIPGap"] = gap
        certified = bool(
            has_incumbent and math.isfinite(gap)
            and gap <= args.mip_gap + 1e-9
        )
        rows = policy_rows(result) if has_incumbent else []
        decision_files = {}
        decision_counts = {}
        if has_incumbent:
            write_policy(policy_file, rows)
            write_chain_policy(
                chain_policy_file, result.get("chain_decisions", []))
            for duration in model_durations:
                decision_file = output / "DecisionVars" / (
                    f"Disrupt_{window:02d}_{station}_{duration}.csv"
                )
                decision_rows = result.get("dv_rows", {}).get(duration, [])
                write_decision_variables(decision_file, decision_rows)
                decision_files[str(duration)] = relative(decision_file)
                decision_counts[str(duration)] = len(decision_rows)

        payload = {
            "schema": formulation_schema,
            "status": result.get("status"),
            "solver": "FULL_POOL_MILP",
            "scope": "FULL_FINITE_ROUTE_AND_TRIP_POOL",
            "column_generation": False,
            "window": window,
            "station": station,
            "durations": model_durations,
            "active_raw_durations": raw_active,
            "zero_impact_durations": [
                d for d in model_durations if d not in raw_active
            ],
            "duration_weights": model_weights,
            "duration_p60_override": args.duration_p60,
            "oracle_duration": args.oracle_duration,
            "cross_duration_nonanticipativity": len(model_durations) > 1,
            "nonanticipativity_mode": args.nonanticipativity_mode,
            "formulation_mode": formulation_mode_for(
                args.nonanticipativity_mode),
            "rail_state_policy": rail_state_policy_for(
                args.nonanticipativity_mode),
            "bus_decision_nonanticipativity": bool(
                stats.get("BusDecisionInformationSetNonanticipative")),
            "passenger_recourse": stats.get("PassengerRecourse"),
            "duration_specific_physical_omega": bool(
                stats.get("DurationSpecificPhysicalOmega")),
            "duration_specific_rail_state": bool(
                stats.get("DurationSpecificRailState")),
            "passenger_flow_variable_sharing": bool(
                stats.get("PassengerFlowVariableSharing")),
            "passenger_shared_interval": [
                stats.get("PassengerSharedIntervalStart"),
                stats.get("PassengerSharedIntervalEndExclusive"),
            ],
            "passenger_shared_variables": stats.get(
                "PassengerSharedVars"),
            "passenger_shared_equality_constraints": stats.get(
                "PassengerSharedEqualityConstrs"),
            "passenger_shared_load_max_abs_diff": stats.get(
                "PassengerSharedLoadMaxAbsDiff"),
            "passenger_shared_queue_max_abs_diff": stats.get(
                "PassengerSharedQueueMaxAbsDiff"),
            "passenger_shared_bus_load_max_abs_diff": stats.get(
                "PassengerSharedBusLoadMaxAbsDiff"),
            "passenger_shared_transfer_max_abs_diff": stats.get(
                "PassengerSharedTransferMaxAbsDiff"),
            "passenger_shared_exit_max_abs_diff": stats.get(
                "PassengerSharedExitMaxAbsDiff"),
            "passenger_transfer_flow_projected_exactly": bool(
                stats.get("PassengerTransferFlowProjectedExactly")),
            "passenger_exit_flow_projected_exactly": bool(
                stats.get("PassengerExitFlowProjectedExactly")),
            "rail_common_event_timetable": bool(
                stats.get("RailCommonEventTimetable")),
            "rail_event_support_information_set_nonanticipative": bool(
                stats.get(
                    "RailEventSupportInformationSetNonanticipative")),
            "termination_action_support_information_set_nonanticipative": bool(
                stats.get(
                    "TerminationActionSupportInformationSetNonanticipative")),
            "train_withdrawal_theta_information_set_nonanticipative": bool(
                stats.get(
                    "TrainWithdrawalThetaInformationSetNonanticipative")),
            "train_capacity_information_set_nonanticipative": bool(
                stats.get(
                    "TrainCapacityInformationSetNonanticipative")),
            "train_availability_information_set_nonanticipative": bool(
                stats.get(
                    "TrainAvailabilityInformationSetNonanticipative")),
            "raw_branch_canceled_trains_preserved": bool(
                stats.get("RawBranchCanceledTrainsPreserved")),
            "raw_branch_cancellation_starts_preserved": bool(
                stats.get("RawBranchCancellationStartsPreserved")),
            "raw_branch_withdrawal_theta_preserved": bool(
                stats.get("RawBranchWithdrawalThetaPreserved")),
            "raw_branch_termination_events_preserved": bool(
                stats.get("RawBranchTerminationEventsPreserved")),
            "raw_branch_passenger_capacity_theta_preserved": bool(
                stats.get("RawBranchPassengerCapacityThetaPreserved")),
            "raw_branch_rail_state_preserved": bool(
                stats.get("RawBranchRailStatePreserved")),
            "raw_rail_availability_equal_across_durations": bool(
                stats.get("RawRailAvailabilityEqualAcrossDurations")),
            "zero_impact_rail_reconstructions_verified": bool(
                stats.get("ZeroImpactRailReconstructionsVerified")),
            "common_affected_od_count": stats.get(
                "CommonAffectedODCount"),
            "duration_probability_source": args.duration_probability_source,
            "duration_prior_counts": {
                **{f"D{d}": prior_counts[d] for d in DURATIONS},
                "total": sum(prior_counts.values()),
            },
            "fixed_stage1_allocation": relative(allocation_file),
            "case_label": args.case_label,
            "exact_state_compression": bool(args.exact_state_compression),
            "exact_trip_cell_pruning": bool(args.exact_trip_cell_pruning),
            "fleet_total": sum(depot_counts.values()),
            "buffer_minutes": args.buffer,
            "depot_access_bound_minutes": DEPOT_ACCESS_BOUND,
            "min_trip_boarding": MIN_TRIP_BOARDING,
            "od_hourly_min_trips": OD_HOURLY_MIN_TRIPS,
            "walking_minutes": WALKING_MINUTES,
            "walking_time_input": walking_input,
            "route_source": (
                f"{ROUTE_SOURCE}-local-only"
                if args.local_routes_only else ROUTE_SOURCE
            ),
            "boarding_mode": BUS_BOARD_MODE,
            "candidate_routes": len({trip["Route"] for trip in group["trips"].values()}),
            "candidate_trips": len(group["trips"]),
            "information_cell_variables": sum(len(cells) for cells in group["cells"].values()),
            "retained_trip_cell_variables": stats.get("TripCellBinaryCount"),
            "raw_trip_cell_count": group.get("raw_trip_cell_count"),
            "tight_boarding_upper_applied": group.get(
                "tight_boarding_upper_applied"),
            "minimum_service_infeasible_trip_cells_removed": group.get(
                "minimum_service_infeasible_trip_cells_removed"),
            "zero_fleet_trip_cells_removed": stats.get(
                "ZeroFleetTripCellsRemoved"),
            "vehicle_chain_information_set_nonanticipative": bool(
                stats.get("VehicleChainInformationSetNonanticipative")),
            "vehicle_chain_decision_scope": stats.get(
                "VehicleChainDecisionScope"),
            "objective": result.get("objective") if has_incumbent else None,
            **{f"Z{duration}": (result.get("Z", {}).get(duration)
                                if has_incumbent else None)
               for duration in DURATIONS},
            "selected_policy_rows": len(rows),
            "policy_file": relative(policy_file) if has_incumbent else None,
            "chain_policy_file": (
                relative(chain_policy_file) if has_incumbent else None),
            "selected_chain_decisions": (
                len(result.get("chain_decisions", [])) if has_incumbent else 0),
            "decision_variable_files": decision_files,
            "decision_variable_rows": decision_counts,
            "stats": stats,
            "has_incumbent": has_incumbent,
            "certified_within_gap": certified,
            "target_mip_gap": args.mip_gap,
            "time_limit_seconds": args.time_limit_seconds,
            "solver_profile": {
                "mip_focus": args.mip_focus,
                "heuristics": args.heuristics,
                "cuts": args.cuts,
                "mip_start_policy": (
                    relative(args.mip_start_policy)
                    if args.mip_start_policy is not None else None
                ),
            },
            "elapsed_seconds": time.time() - started,
        }
        write_json(summary, payload)
        checkpoint(
            "FINISHED",
            final_status=payload["status"],
            certified_within_gap=certified,
        )
        print(
            f"STAGE2_CASE_COMPLETE {window:02d} {station} "
            f"status={payload['status']} certified={certified}",
            flush=True,
        )
        if not has_incumbent:
            return 2
        return 0 if certified else 3
    except Exception as exc:
        failure = {
            "schema": formulation_schema,
            "status": "ERROR",
            "solver": "FULL_POOL_MILP",
            "column_generation": False,
            "nonanticipativity_mode": args.nonanticipativity_mode,
            "formulation_mode": formulation_mode_for(
                args.nonanticipativity_mode),
            "rail_state_policy": rail_state_policy_for(
                args.nonanticipativity_mode),
            "window": window,
            "station": station,
            "durations": DURATIONS,
            "duration_probability_source": args.duration_probability_source,
            "duration_weights": model_weights,
            "case_label": args.case_label,
            "exact_state_compression": bool(args.exact_state_compression),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
        write_json(summary, failure)
        write_json(progress, failure)
        traceback.print_exc()
        return 1
    finally:
        if environment is not None:
            environment.dispose()
        gc.collect()


def run_batch_job(
    args: argparse.Namespace,
    index: int,
    total: int,
    window: int,
    station: str,
) -> dict[str, object]:
    command = [
        sys.executable, "-X", "utf8", "-u", str(SCRIPT),
        "--single-case", "--window", str(window), "--station", station,
        "--buffer", str(args.buffer),
        "--time-limit-seconds", str(args.time_limit_seconds),
        "--mip-gap", str(args.mip_gap),
        "--threads", str(args.threads),
        "--mip-focus", str(args.mip_focus),
        "--heuristics", str(args.heuristics),
        "--cuts", str(args.cuts),
        "--allocation-dir", str(args.allocation_dir),
        "--output-dir", str(args.output_dir),
        "--case-label", args.case_label,
        "--duration-probability-source",
        args.duration_probability_source,
        "--nonanticipativity-mode", args.nonanticipativity_mode,
    ]
    command.append("--exact-state-compression" if args.exact_state_compression
                   else "--no-exact-state-compression")
    command.append("--exact-trip-cell-pruning" if args.exact_trip_cell_pruning
                   else "--no-exact-trip-cell-pruning")
    if args.export_model_dir is not None:
        command.extend(["--export-model-dir", str(args.export_model_dir)])
    if args.mip_start_policy is not None:
        command.extend(["--mip-start-policy", str(args.mip_start_policy)])
    output = args.output_dir
    logs = output / "BatchLogs"
    log_path = logs / f"FullMILP_{window:02d}_{station}.log"
    print(f"RUN {index}/{total} {window:02d}/{station}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "window": window,
        "station": station,
        "exit_code": completed.returncode,
        "certified": is_certified(
            output, window, station, args.mip_gap,
            args.duration_probability_source, args.nonanticipativity_mode),
        "usable": has_usable_result(
            output, window, station, args.duration_probability_source,
            args.nonanticipativity_mode),
    }


def run_batch(args: argparse.Namespace) -> int:
    validate_runtime_inputs(
        args.start_window, args.end_window, args.allocation_dir
    )
    jobs = discover_jobs(
        SCENARIO_STATS, args.start_window, args.end_window
    )
    output = args.output_dir
    logs = output / "BatchLogs"
    logs.mkdir(parents=True, exist_ok=True)
    progress = output / "BatchStatus" / "batch_progress.json"
    certified_jobs = sum(
        is_certified(output, window, station, args.mip_gap,
                     args.duration_probability_source,
                     args.nonanticipativity_mode)
        for window, station in jobs
    )
    usable_jobs = sum(
        has_usable_result(
            output, window, station, args.duration_probability_source,
            args.nonanticipativity_mode)
        for window, station in jobs
    )
    hard_failures = 0
    write_json(progress, {
        "schema": schema_for_mode(args.nonanticipativity_mode),
        "state": "RUNNING",
        "total_jobs": len(jobs),
        "usable_jobs": usable_jobs,
        "certified_jobs": certified_jobs,
        "hard_failures": hard_failures,
        "workers": args.workers,
        "threads_per_case": args.threads,
        "column_generation": False,
        "nonanticipativity_mode": args.nonanticipativity_mode,
        "formulation_mode": formulation_mode_for(
            args.nonanticipativity_mode),
        "rail_state_policy": rail_state_policy_for(
            args.nonanticipativity_mode),
        "exact_state_compression": bool(args.exact_state_compression),
        "durations": DURATIONS,
        "completion_rule": (
            "any complete feasible incumbent; MIP gap is reported but is not "
            "an execution gate"
        ),
        "case_label": args.case_label,
        "duration_probability_source": args.duration_probability_source,
        "allocation_dir": str(args.allocation_dir),
        "output_dir": str(output),
    })
    print(
        f"STAGE2_BATCH jobs={len(jobs)} "
        f"windows={args.start_window}-{args.end_window} "
        f"workers={args.workers} threads_per_case={args.threads}",
        flush=True,
    )

    pending = []
    for index, (window, station) in enumerate(jobs, start=1):
        if has_usable_result(
                output, window, station,
                args.duration_probability_source,
                args.nonanticipativity_mode):
            print(f"SKIP-USABLE {index}/{len(jobs)} {window:02d}/{station}", flush=True)
            continue
        pending.append((index, window, station))

    if args.workers == 1:
        batch_results = (
            run_batch_job(args, index, len(jobs), window, station)
            for index, window, station in pending
        )
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=args.workers)
        futures = [
            executor.submit(
                run_batch_job,
                args,
                index,
                len(jobs),
                window,
                station,
            )
            for index, window, station in pending
        ]
        batch_results = (future.result() for future in as_completed(futures))

    try:
        for result in batch_results:
            window = int(result["window"])
            station = str(result["station"])
            exit_code = int(result["exit_code"])
            certified = bool(result["certified"])
            usable = bool(result["usable"])
            if not usable:
                hard_failures += 1
                print(
                    f"FAILED {window:02d}/{station} "
                    f"exit={exit_code} usable={usable} certified={certified}",
                    flush=True,
                )
            else:
                usable_jobs += 1
                certified_jobs += int(certified)
                label = "CERTIFIED" if certified else "INCUMBENT"
                print(f"DONE-{label} {window:02d}/{station}", flush=True)

            write_json(progress, {
                "schema": schema_for_mode(args.nonanticipativity_mode),
                "state": "RUNNING",
                "total_jobs": len(jobs),
                "usable_jobs": usable_jobs,
                "certified_jobs": certified_jobs,
                "hard_failures": hard_failures,
                "workers": args.workers,
                "threads_per_case": args.threads,
                "last_window": window,
                "last_station": station,
                "last_exit_code": exit_code,
                "last_usable": usable,
                "last_certified": certified,
                "column_generation": False,
                "nonanticipativity_mode": args.nonanticipativity_mode,
                "formulation_mode": formulation_mode_for(
                    args.nonanticipativity_mode),
                "rail_state_policy": rail_state_policy_for(
                    args.nonanticipativity_mode),
                "durations": DURATIONS,
                "completion_rule": (
                    "any complete feasible incumbent; MIP gap is reported but "
                    "is not an execution gate"
                ),
                "case_label": args.case_label,
                "duration_probability_source": (
                    args.duration_probability_source),
                "allocation_dir": str(args.allocation_dir),
                "output_dir": str(output),
            })
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    all_usable = usable_jobs == len(jobs) and hard_failures == 0
    all_certified = certified_jobs == len(jobs) and hard_failures == 0
    write_json(progress, {
        "schema": schema_for_mode(args.nonanticipativity_mode),
        "state": (
            "COMPLETE" if all_certified
            else "COMPLETE_WITH_UNCERTIFIED" if all_usable
            else "INCOMPLETE"
        ),
        "total_jobs": len(jobs),
        "usable_jobs": usable_jobs,
        "certified_jobs": certified_jobs,
        "hard_failures": hard_failures,
        "workers": args.workers,
        "threads_per_case": args.threads,
        "column_generation": False,
        "nonanticipativity_mode": args.nonanticipativity_mode,
        "formulation_mode": formulation_mode_for(
            args.nonanticipativity_mode),
        "rail_state_policy": rail_state_policy_for(
            args.nonanticipativity_mode),
        "durations": DURATIONS,
        "completion_rule": (
            "any complete feasible incumbent; MIP gap is reported but is not "
            "an execution gate"
        ),
        "case_label": args.case_label,
        "duration_probability_source": args.duration_probability_source,
        "allocation_dir": str(args.allocation_dir),
        "output_dir": str(output),
    })
    print(
        f"STAGE2_BATCH_FINISHED jobs={len(jobs)} "
        f"usable={usable_jobs} certified={certified_jobs} "
        f"hard_failures={hard_failures}",
        flush=True,
    )
    return 0 if all_usable else 1


def main() -> int:
    args = parse_args()
    if args.check_only:
        return check_only(args)
    if args.single_case:
        return run_single_case(args)
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
