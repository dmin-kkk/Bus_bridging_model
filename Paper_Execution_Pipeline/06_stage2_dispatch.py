"""Run full-grid LightGBM Stage 2 under the selected information structure.

Without arguments it runs the proposed case of the active profile: Stage 1
allocations from ``paths.stage1``/Allocation, output to ``paths.stage2_proposed``.
``--benchmark`` runs the fixed-benchmark case: allocations from
``paths.benchmark_allocations``, output to ``paths.stage2_benchmark``.
Probabilities: the pinned file of the profile (paper) or the ML stage output
``paths.probabilities``/``prediction.outputs.probability_file``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from pipeline_config import (
    DISPATCH_BUFFER_MINUTES,
    DURATIONS,
    DEPOT_ACCESS_BOUND_MIN,
    MIN_TRIP_BOARDING,
    WINDOWS,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bbconfig import cfg  # noqa: E402


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parent
BASE_RUNNER = ROOT / "06_stage2_core.py"
STAGE2 = cfg.get("stage2")
SOLVER = cfg.get("stage2.solver")
PARALLEL = cfg.get("parallel.stage2")
# Stage 1 writes Allocation/DepotAlloc_HH.csv below paths.stage1 (04_stage1_positioning.py).
ALLOCATION_SUBDIR = "Allocation"


def default_probability_file() -> Path:
    """Pinned probability file (paper) or the ML stage output (09)."""
    slug = str(cfg.get("prediction.full_grid.reference_incident_type"))
    pinned_key = f"pinned.probabilities.{slug}"
    if cfg.get("pinned.use_pinned") and str(cfg.get(pinned_key, default="")):
        return cfg.package_path(pinned_key)
    return (cfg.results_dir / cfg.get("paths.probabilities")
            / str(cfg.get("prediction.outputs.probability_file")).format(slug=slug))


def default_allocation_dir(benchmark: bool) -> Path:
    if benchmark:
        return cfg.results_dir / cfg.get("paths.benchmark_allocations")
    return cfg.results_dir / cfg.get("paths.stage1") / ALLOCATION_SUBDIR


def default_output_dir(benchmark: bool) -> Path:
    key = "paths.stage2_benchmark" if benchmark else "paths.stage2_proposed"
    return cfg.results_dir / cfg.get(key)


def default_case_label(benchmark: bool) -> str:
    """Paper: PROPOSED_LIGHTGBM_BOARD1_BUF30 / FIXED_SPE30_LIGHTGBM_BOARD1_BUF30."""
    template = STAGE2["benchmark_case_label" if benchmark else "proposed_case_label"]
    return str(template).format(
        model=str(cfg.get("prediction.full_grid.model_family")).upper(),
        boarding=f"{float(MIN_TRIP_BOARDING):g}",
        buffer=int(DISPATCH_BUFFER_MINUTES),
        depot=str(cfg.get("evaluation.benchmark_depot")).upper(),
        fleet=int(cfg.get("evaluation.benchmark_fleet")),
    ).upper()


def default_export_dir() -> Path | None:
    value = str(STAGE2.get("export_model_dir", "") or "").strip()
    if not value:
        return None
    return (Path(__file__).resolve().parents[1] / value).resolve()


NONANTICIPATIVITY_MODE = "bus-only"
FORMULATION_MODE = "BUS_ONLY_WAIT_AND_SEE"
PASSENGER_AND_BUS_NONANTICIPATIVITY = "passenger-and-bus"
PASSENGER_AND_BUS_FORMULATION_MODE = (
    "PASSENGER_AND_BUS_SHARED_RAW_DURATION_RAIL"
)
# Format tag written by 05_duration_probabilities.py (SCHEMA).
PROBABILITY_SCHEMA = "full-grid-fixed-type-lightgbm-refit-ensemble-v1"
TRAINING_COHORT = str(cfg.get("probability.duration_source"))
MODEL_FAMILY = str(cfg.get("prediction.full_grid.model_family"))
# prediction.full_grid.incident_types: [slug, label] pairs (or a slug -> label table).
FIXED_INCIDENT_TYPE = str(dict(cfg.get("prediction.full_grid.incident_types"))[
    str(cfg.get("prediction.full_grid.reference_incident_type"))])
# Sec. 4.2: 1,058 retained location-window cases (paper); 0 disables the guard.
EXPECTED_PROBABILITY_ROWS = int(cfg.get("clustering.guards.retained_cases"))
PROBABILITY_COLUMNS = tuple(f"p{duration}" for duration in DURATIONS)
SCHEMA = "full-grid-fixed-infrastructure-stage2-v3-bus-only-wait-and-see"
BASE_SCHEMA = (
    "paper-stage2-full-pool-joint-30-60-bus-only-na-v5"
)
PASSENGER_AND_BUS_BASE_SCHEMA = (
    "paper-stage2-full-pool-joint-30-60-passenger-bus-na-raw-rail-v5"
)


def formulation_mode_for(mode: str) -> str:
    return (
        PASSENGER_AND_BUS_FORMULATION_MODE
        if mode == PASSENGER_AND_BUS_NONANTICIPATIVITY
        else FORMULATION_MODE
    )


def base_schema_for(mode: str) -> str:
    return (
        PASSENGER_AND_BUS_BASE_SCHEMA
        if mode == PASSENGER_AND_BUS_NONANTICIPATIVITY
        else BASE_SCHEMA
    )


def wrapper_schema_for(mode: str) -> str:
    if mode == PASSENGER_AND_BUS_NONANTICIPATIVITY:
        return (
            "full-grid-fixed-infrastructure-stage2-v5-"
            "passenger-and-bus-shared-raw-rail"
        )
    return SCHEMA


def passenger_recourse_for(mode: str) -> str:
    return (
        "NONANTICIPATIVE"
        if mode == PASSENGER_AND_BUS_NONANTICIPATIVITY
        else "WAIT_AND_SEE"
    )


def rail_state_policy_for(mode: str) -> str:
    return (
        "PRESERVE_EACH_DURATION_INPUT_BRANCH"
        if mode == PASSENGER_AND_BUS_NONANTICIPATIVITY
        else "DURATION_SPECIFIC_WAIT_AND_SEE"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-case", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--benchmark", action="store_true",
        help="fixed-benchmark case (paths.benchmark_allocations -> "
             "paths.stage2_benchmark, stage2.benchmark_case_label)",
    )
    parser.add_argument(
        "--nonanticipativity-mode",
        choices=(NONANTICIPATIVITY_MODE,
                 PASSENGER_AND_BUS_NONANTICIPATIVITY),
        default=STAGE2["nonanticipativity_mode"],
    )
    parser.add_argument("--window", type=int)
    parser.add_argument("--station")
    parser.add_argument("--start-window", type=int, default=WINDOWS[0])
    parser.add_argument("--end-window", type=int, default=WINDOWS[-1])
    parser.add_argument(
        "--probability-file", type=Path, default=default_probability_file())
    parser.add_argument(
        "--require-probability-method",
        default=STAGE2["require_probability_method"],
        help="case-insensitive text required in every probability_method row",
    )
    parser.add_argument("--allocation-dir", type=Path,
                        help="default: profile allocation folder of the case")
    parser.add_argument("--output-dir", type=Path,
                        help="default: profile output folder of the case")
    parser.add_argument("--normal-dir", type=Path)
    parser.add_argument("--disruption-dir", type=Path)
    parser.add_argument("--case-label", help="default: built from the profile")
    parser.add_argument("--workers", type=int, default=int(PARALLEL["workers"]))
    parser.add_argument("--threads", type=int, default=int(PARALLEL["threads"]))
    parser.add_argument("--buffer", type=int, default=DISPATCH_BUFFER_MINUTES)
    parser.add_argument(
        "--min-trip-boarding", type=float, default=MIN_TRIP_BOARDING
    )
    parser.add_argument("--time-limit-seconds", type=float,
                        default=float(SOLVER["time_limit_s"]))
    parser.add_argument("--mip-gap", type=float, default=float(SOLVER["mip_gap"]))
    parser.add_argument("--exact-trip-cell-pruning",
                        action=argparse.BooleanOptionalAction,
                        default=bool(STAGE2["exact_trip_cell_pruning"]))
    parser.add_argument(
        "--accept-incumbent", action=argparse.BooleanOptionalAction,
        default=bool(STAGE2["accept_incumbent"]),
        help="accept a feasible incumbent even when the target gap is missed",
    )
    parser.add_argument("--mip-focus", type=int, default=int(SOLVER["mip_focus"]))
    parser.add_argument("--heuristics", type=float,
                        default=float(SOLVER["heuristics"]))
    parser.add_argument("--cuts", type=int, default=int(SOLVER["cuts"]))
    parser.add_argument("--mip-start-policy", type=Path)
    parser.add_argument("--mip-start-dir", type=Path)
    parser.add_argument(
        "--export-model-dir", type=Path, default=default_export_dir(),
        help="write .mps/.sol/.json per solved case (stage2.export_model_dir)")
    args = parser.parse_args()
    if args.allocation_dir is None:
        args.allocation_dir = default_allocation_dir(args.benchmark)
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.benchmark)
    if args.case_label is None:
        args.case_label = default_case_label(args.benchmark)
    if args.export_model_dir is not None and not str(args.export_model_dir).strip():
        args.export_model_dir = None
    if args.single_case and (args.window is None or not args.station):
        parser.error("--single-case requires --window and --station")
    if args.start_window > args.end_window:
        parser.error("--start-window must not exceed --end-window")
    if args.workers < 1 or args.threads < 0:
        parser.error("invalid worker or thread count")
    if args.buffer < 0 or args.min_trip_boarding < 0:
        parser.error("buffer and minimum trip boarding must be nonnegative")
    if args.time_limit_seconds <= 0 or not 0 <= args.mip_gap < 1:
        parser.error("invalid time limit or MIP gap")
    if args.mip_focus not in (0, 1, 2, 3):
        parser.error("--mip-focus must be one of 0, 1, 2, 3")
    if not 0 <= args.heuristics <= 1:
        parser.error("--heuristics must be in [0, 1]")
    if args.cuts not in (-1, 0, 1, 2, 3):
        parser.error("--cuts must be one of -1, 0, 1, 2, 3")
    if not args.single_case and args.mip_start_policy is not None:
        parser.error("--mip-start-policy is supported only with --single-case")
    if args.single_case and args.mip_start_dir is not None:
        parser.error("--mip-start-dir is supported only in batch mode")
    args.probability_file = args.probability_file.resolve()
    if args.export_model_dir is not None:
        args.export_model_dir = args.export_model_dir.resolve()
    args.allocation_dir = args.allocation_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.normal_dir is not None:
        args.normal_dir = args.normal_dir.resolve()
    if args.disruption_dir is not None:
        args.disruption_dir = args.disruption_dir.resolve()
    if args.mip_start_policy is not None:
        args.mip_start_policy = args.mip_start_policy.resolve()
        if not args.mip_start_policy.is_file():
            parser.error(f"MIP-start policy not found: {args.mip_start_policy}")
    if args.mip_start_dir is not None:
        args.mip_start_dir = args.mip_start_dir.resolve()
        if not args.mip_start_dir.is_dir():
            parser.error(f"MIP-start directory not found: {args.mip_start_dir}")
    return args


def load_base():
    sys.path.insert(0, str(ROOT))
    specification = importlib.util.spec_from_file_location(
        "full_grid_ml_stage2_base", BASE_RUNNER)
    if specification is None or specification.loader is None:
        raise ImportError(BASE_RUNNER)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def configure_data_directories(base, args: argparse.Namespace) -> None:
    if args.normal_dir is not None:
        base.NORMAL_DIR = args.normal_dir
    if args.disruption_dir is not None:
        base.DISRUPTION_DIR = args.disruption_dir
        base.SCENARIO_STATS = args.disruption_dir / "_scenario_stats.csv"


def load_probabilities(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "window", "station", *PROBABILITY_COLUMNS, "probability_schema",
        "training_cohort", "model_family", "fixed_incident_type",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"probability file missing columns: {missing}")
    frame = frame.copy()
    frame["window"] = pd.to_numeric(
        frame["window"], errors="raise").astype(int)
    frame["station"] = frame["station"].astype(str).str.strip().str.upper()
    for column in PROBABILITY_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame[["window", "station"]].duplicated().any():
        raise ValueError("duplicate station-hour probability rows")
    if EXPECTED_PROBABILITY_ROWS and len(frame) != EXPECTED_PROBABILITY_ROWS:
        raise ValueError(
            f"expected {EXPECTED_PROBABILITY_ROWS} full-grid probabilities, "
            f"found {len(frame)}")
    if set(frame["probability_schema"].astype(str)) != {PROBABILITY_SCHEMA}:
        raise ValueError("probability schema is not the final refit ensemble")
    if set(frame["training_cohort"].astype(str)) != {TRAINING_COHORT}:
        raise ValueError(f"probability training cohort is not {TRAINING_COHORT}")
    if set(frame["model_family"].astype(str)) != {MODEL_FAMILY}:
        raise ValueError(f"probability model family is not {MODEL_FAMILY}")
    if set(frame["fixed_incident_type"].astype(str)) != {FIXED_INCIDENT_TYPE}:
        raise ValueError(
            f"probability table is not the {FIXED_INCIDENT_TYPE} grid")
    columns = list(PROBABILITY_COLUMNS)
    if not (
        frame[columns].ge(0).all().all()
        and frame[columns].le(1).all().all()
        and ((frame[columns].sum(axis=1) - 1.0).abs() <= 1e-10).all()
    ):
        raise ValueError("invalid duration probabilities")
    return frame


def probability_row(frame: pd.DataFrame, window: int, station: str) -> pd.Series:
    selected = frame[
        frame["window"].eq(int(window))
        & frame["station"].eq(str(station).strip().upper())
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one probability row for H{window:02d}/{station}, "
            f"found {len(selected)}")
    return selected.iloc[0]


def validate_probability_method(
    frame: pd.DataFrame,
    required_method: str | None,
) -> None:
    if required_method is None:
        return
    if "probability_method" not in frame:
        raise ValueError(
            "probability_method column is required by "
            "--require-probability-method"
        )
    required = str(required_method).strip().casefold()
    methods = frame["probability_method"].fillna("").astype(str)
    invalid = ~methods.str.casefold().str.contains(required, regex=False)
    if invalid.any():
        observed = sorted(methods[invalid].unique())
        raise ValueError(
            f"probability method must contain {required_method!r}; "
            f"observed incompatible methods: {observed}"
        )


def summary_path(output: Path, window: int, station: str) -> Path:
    return (
        output / "Performance" /
        f"FullMILP_{window:02d}_{station}_summary.json"
    )


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


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


def is_complete(
    output: Path,
    probabilities: pd.DataFrame,
    window: int,
    station: str,
    args: argparse.Namespace,
    accept_incumbent: bool | None = None,
) -> bool:
    path = summary_path(output, window, station)
    if not path.is_file():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        expected = probability_row(probabilities, window, station)
        weights = summary.get("duration_weights") or {}
        stats = summary.get("stats") or {}
        gap = effective_mip_gap(stats)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    if accept_incumbent is None:
        accept_incumbent = bool(args.accept_incumbent)
    mode = args.nonanticipativity_mode
    formulation_mode = formulation_mode_for(mode)
    full_grid = summary.get("full_grid_ml") or {}
    profile = summary.get("solver_profile") or {}
    artifacts_complete = all((
        output / "DecisionVars" /
        f"Disrupt_{window:02d}_{station}_{duration}.csv"
    ).is_file() for duration in DURATIONS) and all((
        output / folder / f"FullMILP_{window:02d}_{station}_{suffix}.csv"
    ).is_file() for folder, suffix in (
        ("Policies", "policy"), ("Policies", "chain_policy")
    ))
    common_information_audit = bool(
        stats.get("NonanticipativityMode") == mode
        and stats.get("FormulationMode") == formulation_mode
        and stats.get("BusDecisionInformationSetNonanticipative") is True
        and stats.get("VehicleChainInformationSetNonanticipative") is True
        and stats.get("PassengerSharedEqualityConstrs") == 0
    )
    if mode == PASSENGER_AND_BUS_NONANTICIPATIVITY:
        information_audit = bool(
            common_information_audit
            and stats.get("PassengerFlowVariableSharing") is True
            and stats.get("PassengerRecourse") == "NONANTICIPATIVE"
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
    else:
        information_audit = bool(
            common_information_audit
            and stats.get("PassengerFlowVariableSharing") is False
            and stats.get("PassengerRecourse") == "WAIT_AND_SEE"
            and stats.get("TrainAvailabilityInformationSetNonanticipative")
            is False
            and stats.get("DurationSpecificPhysicalOmega") is True
            and stats.get("DurationSpecificRailState") is True
        )
    return bool(
        summary.get("schema") == base_schema_for(mode)
        and summary.get("has_incumbent") is True
        and summary.get("exact_state_compression") is True
        and stats.get("ExactStateCompression") is True
        and stats.get("ExactTripCellPruning")
        is bool(args.exact_trip_cell_pruning)
        and information_audit
        and (accept_incumbent or gap <= args.mip_gap + 1e-9)
        and all(
            abs(float(weights.get(str(duration), weights.get(duration)))
                - float(expected[f"p{duration}"])) <= 1e-10
            for duration in DURATIONS)
        and summary.get("target_mip_gap") == args.mip_gap
        and summary.get("time_limit_seconds") == args.time_limit_seconds
        and summary.get("buffer_minutes") == args.buffer
        and math.isclose(
            float(summary.get("min_trip_boarding", math.nan)),
            float(args.min_trip_boarding),
            abs_tol=1e-12,
        )
        and summary.get("depot_access_bound_minutes") == DEPOT_ACCESS_BOUND_MIN
        and profile.get("mip_focus") == args.mip_focus
        and math.isclose(
            float(profile.get("heuristics", math.nan)), args.heuristics,
            abs_tol=1e-12,
        )
        and profile.get("cuts") == args.cuts
        and full_grid.get("schema") == wrapper_schema_for(mode)
        and full_grid.get("formulation_mode") == formulation_mode
        and full_grid.get("nonanticipativity_mode") == mode
        and full_grid.get("passenger_recourse") == passenger_recourse_for(mode)
        and (
            mode != PASSENGER_AND_BUS_NONANTICIPATIVITY
            or full_grid.get("rail_state_policy") == rail_state_policy_for(mode)
        )
        and full_grid.get("probability_schema") == PROBABILITY_SCHEMA
        and full_grid.get("training_cohort") == TRAINING_COHORT
        and full_grid.get("probability_file") == str(args.probability_file)
        and full_grid.get("allocation_dir") == str(args.allocation_dir)
        and full_grid.get("normal_dir") == (
            str(args.normal_dir) if args.normal_dir is not None else None
        )
        and full_grid.get("disruption_dir") == (
            str(args.disruption_dir) if args.disruption_dir is not None else None
        )
        and artifacts_complete
    )


def run_single(args: argparse.Namespace) -> int:
    probabilities = load_probabilities(args.probability_file)
    validate_probability_method(
        probabilities, args.require_probability_method
    )
    station = str(args.station).strip().upper()
    row = probability_row(probabilities, int(args.window), station)
    base = load_base()
    configure_data_directories(base, args)
    mode = args.nonanticipativity_mode
    duration_weights = {
        duration: float(row[f"p{duration}"]) for duration in DURATIONS}
    base.duration_prior = lambda _source: (
        {duration: 0 for duration in DURATIONS}, duration_weights)
    base.MIN_TRIP_BOARDING = float(args.min_trip_boarding)
    base.parse_args = lambda: argparse.Namespace(
        single_case=True,
        check_only=False,
        nonanticipativity_mode=mode,
        exact_state_compression=bool(STAGE2["exact_state_compression"]),
        exact_trip_cell_pruning=bool(args.exact_trip_cell_pruning),
        local_routes_only=False,
        mip_focus=int(args.mip_focus),
        heuristics=float(args.heuristics),
        cuts=int(args.cuts),
        mip_start_policy=args.mip_start_policy,
        window=int(args.window),
        station=station,
        oracle_duration=None,
        duration_p60=None,
        start_window=int(args.window),
        end_window=int(args.window),
        buffer=int(args.buffer),
        time_limit_seconds=float(args.time_limit_seconds),
        mip_gap=float(args.mip_gap),
        threads=int(args.threads),
        workers=1,
        allocation_dir=args.allocation_dir,
        output_dir=args.output_dir,
        case_label=str(args.case_label),
        duration_probability_source="full_grid_ml",
        export_model_dir=args.export_model_dir,
    )
    return_code = int(base.main())
    path = summary_path(args.output_dir, int(args.window), station)
    if path.is_file():
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["full_grid_ml"] = {
            "schema": wrapper_schema_for(mode),
            "formulation_mode": formulation_mode_for(mode),
            "nonanticipativity_mode": mode,
            "passenger_recourse": passenger_recourse_for(mode),
            "rail_state_policy": rail_state_policy_for(mode),
            "probability_file": str(args.probability_file),
            "probability_schema": PROBABILITY_SCHEMA,
            "training_cohort": TRAINING_COHORT,
            "probability_method": row.get(
                "probability_method",
                "undeclared"),
            **{f"p{duration}": float(row[f"p{duration}"])
               for duration in DURATIONS},
            "fixed_incident_type": str(row.get(
                "fixed_incident_type", FIXED_INCIDENT_TYPE)),
            "allocation_dir": str(args.allocation_dir),
            "normal_dir": (
                str(args.normal_dir) if args.normal_dir is not None else None
            ),
            "disruption_dir": (
                str(args.disruption_dir)
                if args.disruption_dir is not None else None
            ),
            "case_label": str(args.case_label),
            "buffer_minutes": int(args.buffer),
            "min_trip_boarding": float(args.min_trip_boarding),
            "depot_access_bound_minutes": DEPOT_ACCESS_BOUND_MIN,
            "comparison_scope": (
                "same Stage-1 allocation and Stage-2 case; only duration "
                "probability differs from Historical"
            ),
        }
        write_json(path, summary)
    return return_code


def run_one(
    args: argparse.Namespace,
    index: int,
    total: int,
    window: int,
    station: str,
) -> dict[str, object]:
    command = [
        sys.executable, "-X", "utf8", "-u", str(SCRIPT),
        "--single-case", "--window", str(window), "--station", station,
        "--probability-file", str(args.probability_file),
        "--allocation-dir", str(args.allocation_dir),
        "--output-dir", str(args.output_dir),
        "--case-label", str(args.case_label),
        "--threads", str(args.threads),
        "--buffer", str(args.buffer),
        "--min-trip-boarding", str(args.min_trip_boarding),
        "--time-limit-seconds", str(args.time_limit_seconds),
        "--mip-gap", str(args.mip_gap),
        "--mip-focus", str(args.mip_focus),
        "--heuristics", str(args.heuristics),
        "--cuts", str(args.cuts),
        "--nonanticipativity-mode", args.nonanticipativity_mode,
    ]
    if args.normal_dir is not None:
        command.extend(["--normal-dir", str(args.normal_dir)])
    if args.disruption_dir is not None:
        command.extend(["--disruption-dir", str(args.disruption_dir)])
    command.append("--exact-trip-cell-pruning" if args.exact_trip_cell_pruning
                   else "--no-exact-trip-cell-pruning")
    command.append("--accept-incumbent" if args.accept_incumbent
                   else "--no-accept-incumbent")
    if args.export_model_dir is not None:
        command.extend(["--export-model-dir", str(args.export_model_dir)])
    if args.mip_start_dir is not None:
        policy = (
            args.mip_start_dir
            / f"FullMILP_{window:02d}_{station}_policy.csv"
        )
        if policy.is_file():
            command.extend(["--mip-start-policy", str(policy)])
    log = args.output_dir / "BatchLogs" / f"FullMILP_{window:02d}_{station}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"RUN {index}/{total} H{window:02d}/{station}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=handle,
            stderr=subprocess.STDOUT, check=False)
    return {
        "window": window,
        "station": station,
        "return_code": completed.returncode,
    }


def run_batch(args: argparse.Namespace) -> int:
    base = load_base()
    configure_data_directories(base, args)
    probabilities = load_probabilities(args.probability_file)
    validate_probability_method(
        probabilities, args.require_probability_method
    )
    jobs = base.discover_jobs(
        base.SCENARIO_STATS, args.start_window, args.end_window)
    support = set(zip(probabilities["window"], probabilities["station"]))
    missing = sorted(set(jobs) - support)
    if missing:
        raise ValueError(f"probability support misses {len(missing)} jobs")
    pending = [
        (index, window, station)
        for index, (window, station) in enumerate(jobs, start=1)
        if not is_complete(
            args.output_dir, probabilities, window, station, args)
    ]
    progress = args.output_dir / "BatchStatus" / "batch_progress.json"
    completed_count = len(jobs) - len(pending)
    failures = 0
    write_json(progress, {
        "state": "RUNNING", "schema": wrapper_schema_for(
            args.nonanticipativity_mode),
        "formulation_mode": formulation_mode_for(
            args.nonanticipativity_mode),
        "nonanticipativity_mode": args.nonanticipativity_mode,
        "rail_state_policy": rail_state_policy_for(
            args.nonanticipativity_mode),
        "total_jobs": len(jobs), "completed_jobs": completed_count,
        "pending_jobs": len(pending), "failures": failures,
        "workers": args.workers, "threads_per_case": args.threads,
        "exact_state_compression": True,
        "exact_trip_cell_pruning": bool(args.exact_trip_cell_pruning),
        "accept_incumbent": bool(args.accept_incumbent),
        "completion_policy": (
            "any_complete_feasible_incumbent_no_gap_retry"
            if args.accept_incumbent else "certified_within_target_gap"
        ),
        "time_limit_seconds": args.time_limit_seconds,
        "target_mip_gap": args.mip_gap,
        "solver_profile": {
            "mip_focus": args.mip_focus,
            "heuristics": args.heuristics,
            "cuts": args.cuts,
        },
        "probability_file": str(args.probability_file),
        "probability_schema": PROBABILITY_SCHEMA,
        "allocation_dir": str(args.allocation_dir),
        "mip_start_dir": (
            str(args.mip_start_dir) if args.mip_start_dir is not None else None
        ),
    })
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = [
        executor.submit(run_one, args, index, len(jobs), window, station)
        for index, window, station in pending
    ]
    try:
        for future in as_completed(futures):
            result = future.result()
            window = int(result["window"])
            station = str(result["station"])
            complete = is_complete(
                args.output_dir, probabilities, window, station, args)
            completed_count += int(complete)
            failures += int(not complete)
            write_json(progress, {
                "state": "RUNNING", "schema": wrapper_schema_for(
                    args.nonanticipativity_mode),
                "formulation_mode": formulation_mode_for(
                    args.nonanticipativity_mode),
                "nonanticipativity_mode": args.nonanticipativity_mode,
                "rail_state_policy": rail_state_policy_for(
                    args.nonanticipativity_mode),
                "total_jobs": len(jobs), "completed_jobs": completed_count,
                "failures": failures, "last_window": window,
                "last_station": station,
                "last_return_code": int(result["return_code"]),
                "workers": args.workers, "threads_per_case": args.threads,
                "exact_state_compression": True,
                "exact_trip_cell_pruning": bool(args.exact_trip_cell_pruning),
                "accept_incumbent": bool(args.accept_incumbent),
                "completion_policy": (
                    "any_complete_feasible_incumbent_no_gap_retry"
                    if args.accept_incumbent
                    else "certified_within_target_gap"
                ),
                "time_limit_seconds": args.time_limit_seconds,
                "target_mip_gap": args.mip_gap,
                "solver_profile": {
                    "mip_focus": args.mip_focus,
                    "heuristics": args.heuristics,
                    "cuts": args.cuts,
                },
                "probability_file": str(args.probability_file),
                "probability_schema": PROBABILITY_SCHEMA,
                "allocation_dir": str(args.allocation_dir),
                "mip_start_dir": (
                    str(args.mip_start_dir)
                    if args.mip_start_dir is not None else None
                ),
            })
    finally:
        executor.shutdown(wait=True)
    state = "COMPLETE" if completed_count == len(jobs) else "INCOMPLETE"
    write_json(progress, {
        "state": state, "schema": wrapper_schema_for(
            args.nonanticipativity_mode),
        "formulation_mode": formulation_mode_for(
            args.nonanticipativity_mode),
        "nonanticipativity_mode": args.nonanticipativity_mode,
        "rail_state_policy": rail_state_policy_for(
            args.nonanticipativity_mode),
        "total_jobs": len(jobs),
        "completed_jobs": completed_count, "failures": failures,
        "workers": args.workers, "threads_per_case": args.threads,
        "exact_state_compression": True,
        "exact_trip_cell_pruning": bool(args.exact_trip_cell_pruning),
        "accept_incumbent": bool(args.accept_incumbent),
        "completion_policy": (
            "any_complete_feasible_incumbent_no_gap_retry"
            if args.accept_incumbent else "certified_within_target_gap"
        ),
        "time_limit_seconds": args.time_limit_seconds,
        "target_mip_gap": args.mip_gap,
        "solver_profile": {
            "mip_focus": args.mip_focus,
            "heuristics": args.heuristics,
            "cuts": args.cuts,
        },
        "probability_file": str(args.probability_file),
        "probability_schema": PROBABILITY_SCHEMA,
        "allocation_dir": str(args.allocation_dir),
        "mip_start_dir": (
            str(args.mip_start_dir) if args.mip_start_dir is not None else None
        ),
    })
    print(
        f"FULL_GRID_STAGE2_{state} completed={completed_count}/{len(jobs)} "
        f"failures={failures}", flush=True)
    return 0 if state == "COMPLETE" else 1


def check_only(args: argparse.Namespace) -> int:
    base = load_base()
    configure_data_directories(base, args)
    expected_base_schema = base_schema_for(args.nonanticipativity_mode)
    actual_base_schema = base.schema_for_mode(args.nonanticipativity_mode)
    if actual_base_schema != expected_base_schema:
        raise ValueError(
            f"Stage-2 schema mismatch: {actual_base_schema!r} != "
            f"{expected_base_schema!r}"
        )
    probabilities = load_probabilities(args.probability_file)
    validate_probability_method(
        probabilities, args.require_probability_method
    )
    start = int(args.window) if args.single_case else int(args.start_window)
    end = int(args.window) if args.single_case else int(args.end_window)
    discovered = base.discover_jobs(base.SCENARIO_STATS, start, end)
    if args.single_case:
        target = (int(args.window), str(args.station).strip().upper())
        if target not in set(discovered):
            raise ValueError(f"unknown Stage-2 job: H{target[0]:02d}/{target[1]}")
        jobs = [target]
    else:
        jobs = discovered
    base.validate_runtime_inputs(
        start, end, args.allocation_dir
    )
    job_support = set(jobs)
    probability_support = set(zip(
        probabilities["window"], probabilities["station"]
    ))
    missing = sorted(job_support - probability_support)
    extra = sorted(probability_support - job_support)
    if missing or (extra and not args.single_case):
        raise ValueError(
            "probability/job support mismatch: "
            f"missing={len(missing)} extra={len(extra)}"
        )
    methods = sorted(
        probabilities.get(
            "probability_method", pd.Series(["UNDECLARED"])
        ).dropna().astype(str).unique()
    )
    completed_strict = sum(
        is_complete(
            args.output_dir, probabilities, window, station,
            args, False,
        )
        for window, station in jobs
    )
    completed_accepted = sum(
        is_complete(
            args.output_dir, probabilities, window, station,
            args, True,
        )
        for window, station in jobs
    )
    print(json.dumps({
        "stage2_schema": expected_base_schema,
        "wrapper_schema": wrapper_schema_for(args.nonanticipativity_mode),
        "formulation_mode": formulation_mode_for(
            args.nonanticipativity_mode),
        "nonanticipativity_mode": args.nonanticipativity_mode,
        "passenger_recourse": passenger_recourse_for(
            args.nonanticipativity_mode),
        "rail_state_policy": rail_state_policy_for(
            args.nonanticipativity_mode),
        "jobs": len(jobs),
        "probability_rows": len(probabilities),
        "probability_schema": PROBABILITY_SCHEMA,
        "training_cohort": TRAINING_COHORT,
        "probability_methods": methods,
        "required_probability_method": args.require_probability_method,
        "p60_range": [
            float(probabilities["p60"].min()),
            float(probabilities["p60"].max()),
        ],
        "allocation_dir": str(args.allocation_dir),
        "output_dir": str(args.output_dir),
        "output_exists": args.output_dir.exists(),
        "workers": args.workers,
        "threads_per_case": args.threads,
        "time_limit_seconds": args.time_limit_seconds,
        "mip_gap": args.mip_gap,
        "accept_incumbent": args.accept_incumbent,
        "completed_strict": completed_strict,
        "completed_when_accepting_incumbent": completed_accepted,
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.check_only:
        return check_only(args)
    return run_single(args) if args.single_case else run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
