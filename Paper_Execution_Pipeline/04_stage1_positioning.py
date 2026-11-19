"""Run the paper Stage-1 model as a full finite-pool MILP.

This runner deliberately performs no column generation, pricing, reduced-cost
search, or restricted-master iterations.  It constructs every feasible macro
column in the finite candidate pool and passes all of them to one integer
Gurobi model.  Every imported project module is bundled below
``_core/stage1``; no Python code is imported from the parent workspace.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import gurobipy as gp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # package root (bbconfig)
from bbconfig import cfg  # noqa: E402

from pipeline_config import (  # noqa: E402  (legacy names, read from the profile)
    DISPATCH_BUFFER_MINUTES,
    DURATIONS as MODELED_DURATIONS,
    FLEET_SIZE,
    DEPOT_ACCESS_BOUND_MIN,
    MIN_TRIP_BOARDING,
    OD_HOURLY_MIN_TRIPS,
    OPERATIONAL_303_DURATION_WEIGHTS,
    REPRESENTATIVE_BRANCH_FILENAME,
    REPRESENTATIVES_PER_DURATION,
    ROUTE_FILENAME,
    WALKING_MINUTES,
    WALKING_TIME_FILENAME,
    WINDOWS,
)
from walking_times import load_uniform_walking_minutes  # noqa: E402


SCRIPT = Path(__file__).resolve()
PIPELINE_ROOT = SCRIPT.parent
# Inputs: $BB_DATA_DIR (paper) or data/synthetic; outputs: cfg.results_dir (BB_RESULTS_DIR).
INPUT_DIR = cfg.data_dir
RESULTS_DIR = cfg.results_dir
CORE_DIR = PIPELINE_ROOT / "_core" / "stage1"
ENGINE_DIR = CORE_DIR / "CG_Stage1_Stage2"
ENGINE_RUNNER = (
    CORE_DIR / "stage1_exact_cg.py"
)
NORMAL_DIR = RESULTS_DIR / cfg.get("paths.normal")
DISRUPTION_DIR = RESULTS_DIR / cfg.get("paths.disruptions")
# Representative branches (Sec. 4.3 / 5.1, K per window): the paper profile reads the pinned,
# hash-checked file shipped in the package; other profiles read the 02 clustering output.
PINNED_BRANCH_FILE = (
    cfg.package_path("pinned.clustering.branches")
    if cfg.get("pinned.use_pinned") else None)
BRANCH_FILE = (
    PINNED_BRANCH_FILE if PINNED_BRANCH_FILE is not None else
    RESULTS_DIR / cfg.get("paths.clustering") / REPRESENTATIVE_BRANCH_FILENAME
)
ROUTE_FILE = (
    RESULTS_DIR / cfg.get("paths.routes") / ROUTE_FILENAME
)
WALKING_TIME_FILE = INPUT_DIR / WALKING_TIME_FILENAME
OUTPUT_DIR = RESULTS_DIR / cfg.get("paths.stage1")
WARM_START_DIRS = (OUTPUT_DIR,)
WARM_START_DEPOT_ONLY = bool(cfg.get("stage1.warm_start_depot_only"))
WARM_START_ENABLED = bool(cfg.get("stage1.warm_start_enabled"))
WARM_START_HINT_ONLY = bool(cfg.get("stage1.warm_start_hint_only"))
# Model files (.mps/.sol/.json); "" = off.  Relative paths are below the package root.
EXPORT_MODEL_DIR = (
    str(cfg.package_path("stage1.export_model_dir"))
    if str(cfg.get("stage1.export_model_dir")).strip() else "")
SCHEMA = (
    "paper-stage1-full-finite-pool-bus-only-na-"
    "passenger-wait-and-see-v1"
)
NONANTICIPATIVITY_SCOPE = "BUS_Y_U_PSI_ONLY"
PASSENGER_RECOURSE_MODE = "DURATION_SPECIFIC_WAIT_AND_SEE"
BUS_DECISION_NONANTICIPATIVITY = {
    "trip_activation_y": True,
    "duty_initiation_u": True,
    "vehicle_chaining_psi": True,
}
PASSENGER_SHARING_METADATA_KEYS = frozenset({
    "passenger_na_version",
    "passenger_information_block",
    "passenger_shared_start",
    "passenger_shared_end",
    "common_omega",
    "normal_fixed_omega",
    "normal_trajectory",
})

VEHICLE_CHAIN_SCHEDULE_FIELDS = (
    "Window", "ScenarioID", "Duration", "Block", "Depot", "Bus",
    "ChainPosition", "ChainLength", "TripID", "Route", "RouteKey", "Type",
    "Direction", "Path", "DepMin", "ArrMin", "BoardedPassengers",
    "PreviousTripID", "NextTripID", "DeadheadBeforeFrom",
    "DeadheadBeforeTo", "DeadheadBeforeStartMin", "DeadheadBeforeEndMin",
    "DeadheadBeforeMin", "WaitBeforeRunMin", "DeadheadAfterPurpose",
    "DeadheadAfterFrom", "DeadheadAfterTo", "DeadheadAfterStartMin",
    "DeadheadAfterEndMin", "DeadheadAfterMin", "WaitAfterDeadheadMin",
    "NextDepartureMin",
)

STATIC_INPUTS = (
    INPUT_DIR / cfg.get("inputs.timetable"),
    INPUT_DIR / cfg.get("inputs.od_matrix"),
    INPUT_DIR / cfg.get("inputs.road_times"),
    INPUT_DIR / cfg.get("inputs.corridors"),
    INPUT_DIR / cfg.get("inputs.stations"),
    INPUT_DIR / cfg.get("inputs.depots"),
    WALKING_TIME_FILE,
)

LOCAL_MODULE_FILES = (
    ENGINE_RUNNER,
    CORE_DIR / "stage1_kmeans_runner.py",
    CORE_DIR / "stage1_network_model.py",
    CORE_DIR / "stage1_recourse.py",
    ENGINE_DIR / "cg_stage1.py",
    ENGINE_DIR / "cg_problem_builder.py",
    ENGINE_DIR / "state_compression.py",
    ENGINE_DIR / "queue_compression.py",
    ENGINE_DIR / "walking_elimination.py",
)


def configure_environment(output_dir: Path = OUTPUT_DIR) -> None:
    """Remove stale experiment switches and install the paper configuration."""
    for name in tuple(os.environ):
        if name.startswith(("TSB_", "RD_", "RG_", "LR_")):
            os.environ.pop(name, None)
    os.environ.update({
        "PAPER_PIPELINE_INPUT_DIR": str(INPUT_DIR.resolve()),
        "PAPER_PIPELINE_RESULTS_DIR": str(RESULTS_DIR.resolve()),
        "PAPER_OD_HOURLY_MIN_TRIPS": str(OD_HOURLY_MIN_TRIPS),
        "TSB_ROUTE_SOURCE": str(cfg.get("routes.source_mode")),
        "TSB_BUS_BOARD": str(cfg.get("routes.bus_boarding_mode")),
        "TSB_MAX_BUSES": str(FLEET_SIZE),
        "TSB_DEPOT_ACCESS_BOUND": str(DEPOT_ACCESS_BOUND_MIN),
        "TSB_MAX_DEPOTS_PER_ROUTE": str(int(cfg.get("stage1.max_depots_per_route"))),
        "TSB_CG_OPT_TOL": repr(float(cfg.get("solver.stage1.cg_opt_tol"))),
        "TSB_CG_FEAS_TOL": repr(float(cfg.get("solver.stage1.cg_feas_tol"))),
        "TSB_VERBOSE": "1",
        "TSB_STAGE1_CORE_OUTPUT_ROOT": str(
            (Path(output_dir) / "_core_runtime").resolve()),
        "TSB_NA_OUTPUT_ROOT": str(
            (Path(output_dir) / "_nonanticipative_runtime").resolve()),
        "OMP_NUM_THREADS": "1",
    })

    # Local paths take precedence even if a caller has a workspace PYTHONPATH.
    for path in (CORE_DIR, ENGINE_DIR):
        value = str(path.resolve())
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


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


def verify_pinned_branch_file(path: Path) -> None:
    """SHA-256 check of the pinned representative-branch file (pinned.toml)."""
    if PINNED_BRANCH_FILE is None or Path(path).resolve() != PINNED_BRANCH_FILE.resolve():
        return
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()
    expected = str(cfg.get("pinned.clustering.branches_sha256")).upper()
    if digest != expected:
        raise ValueError(
            f"pinned representative-branch file differs from pinned.toml: {digest}")


def duration_weights(source: str) -> dict[int, float]:
    if source == "operational303":
        return dict(OPERATIONAL_303_DURATION_WEIGHTS)
    raise ValueError(f"unknown duration probability source: {source}")


def reweight_duration_prior(scenarios, weights) -> None:
    """Replace only the duration prior; retain conditional cluster masses."""
    for scenario in scenarios:
        duration = int(scenario["duration"])
        scenario["duration_weight"] = float(weights[duration])
        scenario["probability"] = (
            float(scenario.get("prob_station", 0.0)) * float(weights[duration])
            if scenario.get("weighted_terminal", False) else 0.0
        )
    total = math.fsum(float(scenario["probability"])
                      for scenario in scenarios)
    if not math.isclose(total, 1.0, abs_tol=1e-12):
        raise AssertionError(
            f"reweighted Stage 1 scenario probability sums to {total}")


def audit_bus_only_passenger_recourse(scenarios) -> dict:
    """Prove that Stage 1 retains branch-wise passenger recourse inputs."""
    metadata_by_scenario = {}
    omega_counts = {}
    theta_nonzero_counts = {}
    for scenario in scenarios:
        scenario_id = str(scenario.get("id", ""))
        if not scenario_id:
            raise ValueError("Stage 1 scenario is missing id")
        missing = sorted({"duration", "Omega_m", "theta"} - set(scenario))
        if missing:
            raise ValueError(
                f"{scenario_id}: missing branch-wise passenger inputs: "
                f"{missing}")
        present = sorted(PASSENGER_SHARING_METADATA_KEYS & set(scenario))
        if present:
            metadata_by_scenario[scenario_id] = present
        omega_counts[scenario_id] = len(scenario["Omega_m"])
        theta_nonzero_counts[scenario_id] = sum(
            abs(float(value)) > 1e-12
            for value in scenario["theta"].values()
        )
    if metadata_by_scenario:
        raise ValueError(
            "bus-only Stage 1 received passenger-sharing metadata: "
            f"{metadata_by_scenario}")
    return {
        "nonanticipativity_scope": NONANTICIPATIVITY_SCOPE,
        "passenger_recourse_mode": PASSENGER_RECOURSE_MODE,
        "strict_passenger_na_preprocessor_invoked": False,
        "passenger_variable_sharing": False,
        "passenger_sharing_metadata_absent": True,
        "duration_specific_omega_theta_preserved": True,
        "scenario_count": len(scenarios),
        "omega_count_by_scenario": omega_counts,
        "theta_nonzero_count_by_scenario": theta_nonzero_counts,
    }


def load_warm_start(
        window: int, runner, depots, active_columns,
        output_dir: Path = OUTPUT_DIR) -> tuple[dict, set, str | None]:
    """Load the newest available x/y incumbent as a nonbinding MIP start."""
    if not WARM_START_ENABLED:
        return {}, set(), None
    source = next((
        directory for directory in (Path(output_dir), *WARM_START_DIRS)
        if (directory / "Allocation" / f"DepotAlloc_{window:02d}.csv").is_file()
        or (directory / f"SelectedColumns_{window:02d}.csv").is_file()
    ), None)
    if source is None:
        return {}, set(), None
    allocation = source / "Allocation" / f"DepotAlloc_{window:02d}.csv"
    selected = source / f"SelectedColumns_{window:02d}.csv"
    depot_start = runner._load_depot_start(allocation, depots, FLEET_SIZE)
    column_start = set()
    if selected.is_file() and not WARM_START_DEPOT_ONLY:
        with selected.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                raw_cell = row.get("InformationCell", "")
                try:
                    parsed = json.loads(raw_cell)
                    cell = tuple(int(value) for value in parsed)
                except (TypeError, ValueError, json.JSONDecodeError):
                    cell = tuple(int(token) for token in raw_cell.replace(
                        "[", "").replace("]", "").replace("|", ",").split(",")
                                 if token.strip())
                key = (str(row["Block"]), str(row["TripID"]), cell)
                if key in active_columns:
                    column_start.add(key)
    return depot_start, column_start, str(source.resolve())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(safe(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _vehicle_chain_schedule_rows(bus_rows) -> list[dict]:
    """Flatten selected one-way services into publication-friendly duties."""
    selected = [dict(row) for row in bus_rows if bool(row.get("Selected"))]
    by_trip = {
        (str(row.get("ScenarioID")), str(row.get("TripID"))): row
        for row in selected
    }
    schedule = []
    for row in selected:
        departure = int(row["RunStart"])
        if departure % 10:
            raise AssertionError(
                "passenger-service departure is not on the 10-minute grid: "
                f"{row.get('TripID')} at {departure}")

        sid = str(row.get("ScenarioID"))
        next_id = row.get("NextTripID")
        if next_id:
            successor = by_trip.get((sid, str(next_id)))
            if successor is None:
                raise AssertionError(
                    f"chain successor {next_id} is absent for "
                    f"{row.get('TripID')}")
            after_purpose = "NEXT_TRIP"
            after_from = successor.get("DeadheadBeforeFrom")
            after_to = successor.get("DeadheadBeforeTo")
            after_start = successor.get("DeadheadBeforeStart")
            after_end = successor.get("DeadheadBeforeEnd")
            after_minutes = successor.get("DeadheadBeforeMinutes")
            wait_after = successor.get("WaitBeforeRun")
            next_departure = successor.get("RunStart")
        else:
            after_purpose = "RETURN_TO_DEPOT"
            after_from = row.get("EndStation")
            after_to = row.get("Depot")
            after_start = row.get("ReturnStart")
            after_end = row.get("ReturnEnd")
            after_minutes = row.get("ReturnDeadhead")
            wait_after = None
            next_departure = None

        schedule.append({
            "Window": row.get("Window"),
            "ScenarioID": sid,
            "Duration": row.get("Duration"),
            "Block": row.get("Block"),
            "Depot": row.get("Depot"),
            "Bus": row.get("FleetBusID"),
            "ChainPosition": row.get("ChainPosition"),
            "ChainLength": row.get("ChainLength"),
            "TripID": row.get("TripID"),
            "Route": row.get("Route"),
            "RouteKey": row.get("RouteKey"),
            "Type": row.get("RouteType"),
            "Direction": row.get("Direction"),
            "Path": f"{row.get('StartStation')}--{row.get('EndStation')}",
            "DepMin": departure,
            "ArrMin": row.get("ServiceEnd", row.get("RunEnd")),
            "BoardedPassengers": row.get("BoardedPassengers"),
            "PreviousTripID": row.get("PreviousTripID"),
            "NextTripID": next_id,
            "DeadheadBeforeFrom": row.get("DeadheadBeforeFrom"),
            "DeadheadBeforeTo": row.get("DeadheadBeforeTo"),
            "DeadheadBeforeStartMin": row.get("DeadheadBeforeStart"),
            "DeadheadBeforeEndMin": row.get("DeadheadBeforeEnd"),
            "DeadheadBeforeMin": row.get("DeadheadBeforeMinutes"),
            "WaitBeforeRunMin": row.get("WaitBeforeRun"),
            "DeadheadAfterPurpose": after_purpose,
            "DeadheadAfterFrom": after_from,
            "DeadheadAfterTo": after_to,
            "DeadheadAfterStartMin": after_start,
            "DeadheadAfterEndMin": after_end,
            "DeadheadAfterMin": after_minutes,
            "WaitAfterDeadheadMin": wait_after,
            "NextDepartureMin": next_departure,
        })
    return sorted(schedule, key=lambda row: (
        str(row["ScenarioID"]), str(row["Depot"]), str(row["Bus"]),
        int(row["ChainPosition"]), str(row["TripID"])))


def _write_vehicle_chain_schedule_outputs(
        runner, output: Path, rows, scenarios, window: int) -> dict[str, str]:
    """Write one ordered one-way/deadhead duty CSV for every scenario."""
    metadata = {str(scenario["id"]): scenario for scenario in scenarios}
    by_scenario = {}
    for row in rows:
        by_scenario.setdefault(str(row["ScenarioID"]), []).append(row)
    files = {}
    for sid, scenario in sorted(metadata.items()):
        location = scenario.get(
            "block_station", scenario.get("station", sid))
        duration = scenario.get("duration", "NA")
        token = (
            f"{int(window):02d}_{runner._safe_filename_token(location)}_"
            f"{runner._safe_filename_token(duration)}")
        path = (
            Path(output) / "VehicleChainSchedules" /
            f"VehicleChainSchedule_{token}.csv")
        runner._write_csv_atomic(
            path, by_scenario.get(sid, ()), VEHICLE_CHAIN_SCHEDULE_FIELDS)
        files[sid] = str(path.resolve())
    return files


def load_input_runner():
    specification = importlib.util.spec_from_file_location(
        "full_pool_stage1_input_runner", ENGINE_RUNNER
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def parse_window_tokens(tokens: list[str]) -> tuple[int, ...]:
    windows: set[int] = set()
    for raw_token in tokens:
        for token in raw_token.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                left, right = token.split("-", 1)
                start, end = int(left), int(right)
                if end < start:
                    raise ValueError(f"descending window range: {token}")
                windows.update(range(start, end + 1))
            else:
                windows.add(int(token))
    if not windows:
        raise ValueError("at least one planning window is required")
    invalid = sorted(window for window in windows if window not in WINDOWS)
    if invalid:
        raise ValueError(
            f"windows must be planning windows of the profile {list(WINDOWS)}: "
            f"{invalid}"
        )
    return tuple(sorted(windows))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows", nargs="+",
        default=[",".join(str(window) for window in WINDOWS)],
        metavar="HOUR_OR_RANGE",
        help="start hours, e.g. --windows 6-20 or --windows 6 8 17")
    parser.add_argument(
        "--workers", type=int, default=int(cfg.get("parallel.stage1.workers")),
        help="number of independent planning windows solved concurrently")
    parser.add_argument(
        "--time-limit", type=float,
        default=float(cfg.get("solver.stage1.time_limit_s")),
        help="Gurobi time limit in seconds for each planning window")
    parser.add_argument(
        "--mip-gap", type=float, default=float(cfg.get("solver.stage1.mip_gap")))
    parser.add_argument(
        "--threads", type=int, default=int(cfg.get("parallel.stage1.threads")),
        help="Gurobi threads per planning window; zero lets Gurobi decide")
    parser.add_argument(
        "--bound-focus", action="store_true",
        help=("use the strong-bound profile for sequential "
              "retries of difficult windows"))
    parser.add_argument(
        "--root-escape", action="store_true",
        help=("cap root cut passes at 3 and use the optimality-proof "
              "profile so difficult windows enter branch-and-bound"))
    parser.add_argument(
        "--concurrent-retry", action="store_true",
        help=("run two concurrent MIP strategies for a difficult-window "
              "retry; Gurobi divides the configured threads between them"))
    parser.add_argument(
        "--incumbent-target", type=float,
        help=("use an incumbent-improvement profile and stop when this "
              "objective value is reached"))
    parser.add_argument(
        "--auto-parameters", action=argparse.BooleanOptionalAction,
        default=bool(cfg.get("solver.stage1.auto_parameters")),
        help="leave Stage 1 strategy parameters at Gurobi automatic defaults")
    parser.add_argument(
        "--tune-parameters", action="store_true",
        help=("run Gurobi Automatic Parameter Tuning before each selected "
              "window, save the best .prm file, then run a fresh solve"))
    parser.add_argument(
        "--tune-time-limit", type=float,
        default=float(cfg.get("stage1.tuning.time_limit_s")),
        help="total Gurobi tuning budget in seconds per planning window")
    parser.add_argument(
        "--tune-trials", type=int, default=int(cfg.get("stage1.tuning.trials")),
        help="Gurobi TuneTrials value; zero lets Gurobi choose automatically")
    parser.add_argument(
        "--tune-candidate-time-limit", type=float,
        default=float(cfg.get("stage1.tuning.candidate_time_limit_s")),
        help="maximum seconds for each parameter set tested by the tuner")
    parser.add_argument(
        "--tune-only", action="store_true",
        help="write the tuned .prm file without a subsequent production solve")
    parser.add_argument(
        "--tuning-dir", type=Path,
        default=OUTPUT_DIR / "Performance" / "Tuning",
        help="directory for per-window tuned .prm files")
    parser.add_argument(
        "--parameter-dir", type=Path,
        help=("reuse Stage1_Hxx.prm files from this directory instead of "
              "running the tuner"))
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="isolated Stage 1 result directory")
    parser.add_argument(
        "--representative-branches", type=Path, default=BRANCH_FILE,
        help="representative branch CSV used by every selected window")
    parser.add_argument(
        "--duration-probability-source",
        choices=("operational303",),
        default=str(cfg.get("stage1.duration_probability_source")),
        help=("duration prior used in the Stage 1 objective; operational303 "
              "uses probability.duration_counts (Table 3: pi_30, pi_60)"))
    parser.add_argument(
        "--accept-incumbent", action=argparse.BooleanOptionalAction,
        default=bool(cfg.get("solver.stage1.accept_incumbent")),
        help=("treat any feasible incumbent as usable even when the target "
              "MIP gap is not certified; the achieved gap is still stored"))
    parser.add_argument(
        "--allocation-only", action="store_true",
        help=("write the depot allocation and solver audit only; skip detailed "
              "scenario, bus-operation, and decision-variable extraction"))
    parser.add_argument(
        "--export-model-dir", default=EXPORT_MODEL_DIR,
        help=("write .mps/.sol/.json of every solved window here "
              "(default: stage1.export_model_dir; empty = off)"))
    parser.add_argument(
        "--check-only", action="store_true",
        help="validate the local bundle and report upstream readiness")
    args = parser.parse_args(argv)
    try:
        args.windows = parse_window_tokens(args.windows)
    except ValueError as exception:
        parser.error(str(exception))
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.time_limit <= 0:
        parser.error("--time-limit must be positive")
    if not 0 <= args.mip_gap < 1:
        parser.error("--mip-gap must be in [0, 1)")
    if args.threads < 0:
        parser.error("--threads must be nonnegative")
    if args.tune_time_limit <= 0:
        parser.error("--tune-time-limit must be positive")
    if args.tune_trials < 0:
        parser.error("--tune-trials must be nonnegative")
    if args.tune_candidate_time_limit <= 0:
        parser.error("--tune-candidate-time-limit must be positive")
    args.output_dir = args.output_dir.resolve()
    args.representative_branches = args.representative_branches.resolve()
    try:
        verify_pinned_branch_file(args.representative_branches)
    except (OSError, ValueError) as exception:
        parser.error(str(exception))
    if args.tune_parameters and args.parameter_dir is not None:
        parser.error("--tune-parameters and --parameter-dir are exclusive")
    if args.tune_only and not args.tune_parameters:
        parser.error("--tune-only requires --tune-parameters")
    if args.root_escape and args.concurrent_retry:
        parser.error("--root-escape and --concurrent-retry are exclusive")
    if (args.incumbent_target is not None and
            not math.isfinite(args.incumbent_target)):
        parser.error("--incumbent-target must be finite")
    return args


def make_window_args(window: int, options):
    parameter_file = (
        options.parameter_dir.resolve() / f"Stage1_H{int(window):02d}.prm"
        if options.parameter_dir is not None else None
    )
    tune_output_path = (
        options.tuning_dir.resolve() / f"Stage1_H{int(window):02d}.prm"
        if options.tune_parameters else None
    )
    return argparse.Namespace(
        window=int(window),
        route_file=ROUTE_FILE.resolve(),
        representative_branches=options.representative_branches.resolve(),
        output=options.output_dir.resolve(),
        duration_probability_source=options.duration_probability_source,
        duration_weights=duration_weights(
            options.duration_probability_source),
        k=REPRESENTATIVES_PER_DURATION,
        buffer=DISPATCH_BUFFER_MINUTES,
        time_limit_seconds=float(options.time_limit),
        mip_gap=float(options.mip_gap),
        threads=int(options.threads),
        bound_focus=bool(options.bound_focus),
        root_escape=bool(options.root_escape),
        concurrent_retry=bool(options.concurrent_retry),
        incumbent_target=options.incumbent_target,
        auto_parameters=bool(options.auto_parameters),
        tune_parameters=bool(options.tune_parameters),
        tune_time_limit=float(options.tune_time_limit),
        tune_trials=int(options.tune_trials),
        tune_candidate_time_limit=float(options.tune_candidate_time_limit),
        tune_only=bool(options.tune_only),
        allocation_only=bool(getattr(options, "allocation_only", False)),
        export_model_dir=str(getattr(options, "export_model_dir", EXPORT_MODEL_DIR) or ""),
        parameter_file=parameter_file,
        tune_output_path=tune_output_path,
    )


def solve_window(args) -> int:
    configure_environment(args.output)
    started = time.time()
    output = args.output.resolve()
    performance = output / "Performance"
    performance.mkdir(parents=True, exist_ok=True)
    prefix = f"FullMILP_{args.window:02d}"
    progress_path = performance / f"{prefix}_progress.json"
    summary_path = performance / f"{prefix}_summary.json"
    environment = None
    tune_parameters = bool(getattr(args, "tune_parameters", False))
    parameter_file = getattr(args, "parameter_file", None)
    tune_output_path = getattr(args, "tune_output_path", None)
    tune_only = bool(getattr(args, "tune_only", False))
    auto_parameters = bool(getattr(args, "auto_parameters", False))
    allocation_only = bool(getattr(args, "allocation_only", False))
    automatic_parameter_strategy = bool(
        auto_parameters or tune_parameters or parameter_file is not None
    )
    heuristics_override = getattr(args, "heuristics", None)
    export_model_dir = getattr(args, "export_model_dir", EXPORT_MODEL_DIR)

    def checkpoint(event):
        row = dict(event) if isinstance(event, dict) else {"event": event}
        row["runner_elapsed_seconds"] = time.time() - started
        write_json(progress_path, {
            "schema": SCHEMA,
            "status": "RUNNING",
            "solver": "FULL_POOL_MILP",
            "scope": "FULL_FINITE_MACRO_POOL",
            "column_generation": False,
            "nonanticipativity_scope": NONANTICIPATIVITY_SCOPE,
            "passenger_recourse_mode": PASSENGER_RECOURSE_MODE,
            "window": args.window,
            "k": args.k,
            "target_gap": args.mip_gap,
            "solver_time_limit_seconds": args.time_limit_seconds,
            "last_event": row,
            "elapsed_seconds": time.time() - started,
        })

    try:
        checkpoint({"phase": "RUNNER_STARTED"})
        walking_input = load_uniform_walking_minutes(
            WALKING_TIME_FILE,
            STATIC_INPUTS[2],
            WALKING_MINUTES,
        )
        walking_minutes = int(walking_input["walking_minutes"])
        import cg_stage1 as model_engine

        runner = load_input_runner()
        nonanticipative = runner.load_na_module()
        core = nonanticipative.load_core()
        modeled_durations = tuple(runner._modeled_durations(args))
        if modeled_durations != MODELED_DURATIONS:
            raise ValueError(
                f"representative branches must contain durations "
                f"{MODELED_DURATIONS}, found {modeled_durations}")
        core.DURATIONS = list(MODELED_DURATIONS)
        core.TIME_LIMIT = args.time_limit_seconds
        core.MIP_GAP = args.mip_gap
        core.DISRUPT_BUFFER = args.buffer
        core.MAX_TOTAL_BUSES = FLEET_SIZE
        core.DEPOT_ACCESS_BOUND_MIN = float(DEPOT_ACCESS_BOUND_MIN)
        core.MIN_TRIP_BOARDING = float(MIN_TRIP_BOARDING)
        core.ROUTE_SOURCE = str(cfg.get("routes.source_mode"))
        core.BUS_BOARD_MODE = str(cfg.get("routes.bus_boarding_mode"))
        core.FILE_SCHEDULE = str(STATIC_INPUTS[0].resolve())
        core.FILE_OD = str(STATIC_INPUTS[1].resolve())
        core.FILE_DISTANCE = str(STATIC_INPUTS[2].resolve())
        core.FILE_EXCEL = str(STATIC_INPUTS[4].resolve())
        core.DIR_NORMAL = str(NORMAL_DIR.resolve())
        core.DIR_DISRUPT = str(DISRUPTION_DIR.resolve())
        core.FILE_PREDEFINED_ROUTES = str(args.route_file.resolve())

        checkpoint({"phase": "PREPARING_WINDOW"})
        depots, scenarios, window_data, travel_times = runner.prepare_window(
            core,
            nonanticipative,
            args.window,
            args.buffer,
            args.k,
            args.representative_branches.resolve(),
        )
        passenger_recourse_audit = audit_bus_only_passenger_recourse(scenarios)
        reweight_duration_prior(scenarios, args.duration_weights)
        checkpoint({
            "phase": "BUS_ONLY_PASSENGER_WAIT_AND_SEE_READY",
            "nonanticipativity_scope": NONANTICIPATIVITY_SCOPE,
            "passenger_recourse_mode": PASSENGER_RECOURSE_MODE,
            "scenario_count": passenger_recourse_audit["scenario_count"],
            "omega_count_by_scenario": passenger_recourse_audit[
                "omega_count_by_scenario"],
            "theta_nonzero_count_by_scenario": passenger_recourse_audit[
                "theta_nonzero_count_by_scenario"],
        })
        incumbent_focus = args.incumbent_target is not None
        # Strategy parameters: Gurobi automatic defaults in every paper run
        # (solver.stage1.auto_parameters); the manual profile below is legacy.
        manual = cfg.get("solver.stage1.manual_profile")
        config = runner.build_cg_config(
            core,
            final_method=(None if automatic_parameter_strategy
                          else int(manual["method"])),
            final_mip_focus=(None if automatic_parameter_strategy else (
                int(manual["incumbent_mip_focus"]) if incumbent_focus else (
                    int(manual["root_escape_mip_focus"]) if args.root_escape
                    else int(manual["mip_focus"])))),
            final_cuts=(None if automatic_parameter_strategy else (
                int(manual["incumbent_cuts"]) if incumbent_focus
                else int(manual["cuts"]))),
            final_cut_passes=(None if automatic_parameter_strategy else (
                int(manual["incumbent_cut_passes"]) if incumbent_focus else
                (int(manual["root_escape_cut_passes"]) if args.root_escape
                 else None))),
            final_concurrent_mip=(None if automatic_parameter_strategy else (
                int(manual["concurrent_mip"]) if args.concurrent_retry
                else None)),
            final_symmetry=(None if automatic_parameter_strategy
                            else int(manual["symmetry"])),
            final_heuristics=(None if automatic_parameter_strategy else (
                float(heuristics_override)
                if heuristics_override is not None else
                (float(manual["incumbent_heuristics"]) if incumbent_focus else
                 (float(manual["heuristics"]) if args.root_escape else
                  (float(manual["bound_focus_heuristics"]) if args.bound_focus
                   else float(manual["heuristics"])))))),
            final_no_rel_heur_time=(None if automatic_parameter_strategy else (
                float(manual["incumbent_no_rel_heur_time_s"])
                if incumbent_focus else None)),
            final_improve_start_time=(
                None if automatic_parameter_strategy else
                (float(manual["incumbent_improve_start_time_s"])
                 if incumbent_focus else None)),
            final_best_obj_stop=(
                None if automatic_parameter_strategy else
                args.incumbent_target),
            independent_trips=False,
        )
        config = replace(
            config,
            time_limit=args.time_limit_seconds,
            mip_gap=args.mip_gap,
            output_flag=1,
            enforce_vehicle_chains=True,
            min_trip_boarding=float(MIN_TRIP_BOARDING),
            sigma=walking_minutes,
        )
        if int(config.max_total_buses) != FLEET_SIZE:
            raise AssertionError(
                f"fleet configuration drift: {config.max_total_buses}")
        if int(config.disrupt_buffer) != DISPATCH_BUFFER_MINUTES:
            raise AssertionError(
                f"dispatch-buffer drift: {config.disrupt_buffer}")
        if float(config.min_trip_boarding) != MIN_TRIP_BOARDING:
            raise AssertionError(
                f"minimum-boarding drift: {config.min_trip_boarding}")
        if int(config.sigma) != walking_minutes:
            raise AssertionError(f"walking-time drift: {config.sigma}")
        if float(runner.EXACT_DEPOT_ACCESS_BOUND_MIN) != DEPOT_ACCESS_BOUND_MIN:
            raise AssertionError(
                "bundled depot access bound differs from operations.depot_access_bound_min")
        metadata = model_engine.CGProblem(
            tuple(depots), tuple(scenarios), {}, window_data,
            travel_times=travel_times,
        )
        raw_columns = model_engine.count_macro_columns(metadata, config)
        checkpoint({
            "phase": "WINDOW_PREPARED",
            "scenario_count": len(scenarios),
            "depot_count": len(depots),
            "raw_candidate_macro_columns": raw_columns,
        })

        caches = runner.build_full_caches(
            core,
            nonanticipative,
            scenarios,
            window_data,
            progress_hook=checkpoint,
            config=config,
        )
        checkpoint({
            "phase": "EXACT_CACHES_READY",
            "cache_scenarios": len(caches),
        })
        environment = gp.Env(empty=True)
        environment.setParam("OutputFlag", 1)
        environment.setParam("Threads", args.threads)
        environment.start()
        problem = model_engine.CGProblem(
            tuple(depots), tuple(scenarios), caches, window_data,
            environment, travel_times=travel_times,
        )
        pool = tuple(model_engine.build_column_pool(problem, config))
        active_columns = {column.key for column in pool}
        checkpoint({
            "phase": "EXACT_COLUMN_SUPPORT_PRUNED",
            "raw_candidate_macro_columns": raw_columns,
            "candidate_macro_columns": len(pool),
            "minimum_service_infeasible_columns_removed": (
                raw_columns - len(pool)),
        })

        exact_flags = {
            "walking_eliminated_exactly": bool(config.eliminate_w),
            "destination_states_eliminated_exactly": bool(
                config.eliminate_destination_states
            ),
            "load_states_event_compressed": bool(config.compress_load_states),
            "queue_states_event_compressed": bool(config.compress_queue_states),
            "hybrid_bus_load_projection": bool(config.hybrid_bus_load_states),
            "trip_chain_enforced": bool(config.enforce_vehicle_chains),
            "trip_activation_information_set_nonanticipative": True,
            "duty_initiation_information_set_nonanticipative": True,
            "vehicle_chain_information_set_nonanticipative": True,
            "duration_specific_passenger_omega_theta_preserved": bool(
                passenger_recourse_audit.get(
                    "duration_specific_omega_theta_preserved")),
            "passenger_sharing_metadata_absent": bool(
                passenger_recourse_audit.get(
                    "passenger_sharing_metadata_absent")),
        }
        if not all(exact_flags.values()):
            raise AssertionError(f"exact flags disabled: {exact_flags}")
        checkpoint({
            "phase": "FULL_POOL_READY",
            "candidate_macro_columns": len(pool),
            "raw_candidate_macro_columns": raw_columns,
            "minimum_service_infeasible_columns_removed": (
                raw_columns - len(pool)),
            "exact_reformulation": exact_flags,
        })

        depot_start, column_start, warm_start_source = load_warm_start(
            args.window, runner, depots, active_columns, output)
        primal_hint = None
        if depot_start or column_start:
            primal_hint = {
                "x": depot_start,
                "y": {
                    key: float(key in column_start) for key in active_columns
                },
            }
            checkpoint({
                "phase": "CAUSAL_MIP_START_READY",
                "depot_start_values": len(depot_start),
                "selected_y_start_values": len(column_start),
                "chain_start_policy": "solver-repaired causal path cover",
            })

        solve_started = time.time()
        if tune_output_path is not None:
            Path(tune_output_path).parent.mkdir(parents=True, exist_ok=True)
        depot_counts, status, scenario_results, stats = model_engine.solve_master(
            problem,
            config,
            lp_relax=False,
            active_columns=active_columns,
            primal_hint=primal_hint,
            depot_start=(
                None if WARM_START_HINT_ONLY else (depot_start or None)),
            column_start=(
                None if WARM_START_HINT_ONLY else (column_start or None)),
            extract_solution=not allocation_only,
            progress_hook=checkpoint,
            parameter_file=(
                str(Path(parameter_file).resolve())
                if parameter_file is not None else None),
            tune_output_path=(
                str(Path(tune_output_path).resolve())
                if tune_output_path is not None else None),
            tune_time_limit=(
                float(args.tune_time_limit) if tune_parameters else None),
            tune_trials=(
                int(args.tune_trials) if tune_parameters else 0),
            tune_candidate_time_limit=(
                float(args.tune_candidate_time_limit)
                if tune_parameters else None),
            tune_only=tune_only,
            export_model_dir=(export_model_dir or None),
            export_name=f"stage1_H{int(args.window):02d}",
            export_metadata={
                "stage": 1, "window": int(args.window), "k": int(args.k),
                "fleet_size": FLEET_SIZE,
                "durations": list(MODELED_DURATIONS),
                "duration_weights": {
                    str(key): float(value)
                    for key, value in args.duration_weights.items()},
                "warm_start_source": (
                    None if warm_start_source is None
                    else Path(warm_start_source).name),
            },
        )
        solve_seconds = time.time() - solve_started
        has_incumbent = bool(stats and int(stats.get("SolCount", 0)) > 0)
        passenger_variable_sharing = bool(
            stats and stats.get("PassengerFlowVariableSharing", False))
        if passenger_variable_sharing:
            raise AssertionError(
                "bus-only Stage 1 unexpectedly enabled passenger variable "
                "sharing")
        objective = float(stats["ObjVal"]) if has_incumbent else None
        bound = (
            float(stats["ObjBound"])
            if stats and stats.get("ObjBound") is not None else None
        )
        relative_gap = (
            max(0.0, objective - bound) / max(abs(objective), 1e-12)
            if objective is not None and bound is not None
            and bound <= objective + 1e-6 else None
        )
        certified = bool(
            relative_gap is not None
            and relative_gap <= args.mip_gap + 1e-9
        )

        allocation_file = None
        selected_file = None
        scenario_files = {}
        bus_files = {}
        vehicle_chain_schedule_files = {}
        decision_files = {}
        combined_decisions = None
        decision_count = 0
        selected_columns = []
        if has_incumbent:
            allocation_path = (
                output / "Allocation" / f"DepotAlloc_{args.window:02d}.csv"
            )
            integer_counts = runner._write_depot_allocation(
                allocation_path, depots, depot_counts
            )
            if sum(integer_counts.values()) != FLEET_SIZE:
                raise AssertionError(
                    "fleet equality violated in extracted allocation: "
                    f"{sum(integer_counts.values())} != {FLEET_SIZE}")
            allocation_file = str(allocation_path.resolve())
            if not allocation_only:
                selected_columns = [
                    column.key for column in pool
                    if any(
                        float(
                            scenario_results.get(scenario_id, {})
                            .get("y", {}).get(column.trip_id, 0.0)
                        ) > 0.5
                        for scenario_id in column.scenario_ids
                    )
                ]
                selected_path = output / f"SelectedColumns_{args.window:02d}.csv"
                runner._write_csv_atomic(
                    selected_path,
                    ({
                        "Block": key[0],
                        "TripID": key[1],
                        "InformationCell": json.dumps(
                            list(key[2]), ensure_ascii=False
                        ),
                    } for key in sorted(selected_columns, key=repr)),
                    ("Block", "TripID", "InformationCell"),
                )
                selected_file = str(selected_path.resolve())
                reporting_results, reporting_scenarios = (
                    runner._terminal_reporting_view(scenario_results, scenarios)
                )
                scenario_rows = runner._scenario_rows(
                    reporting_results, reporting_scenarios, args.window
                )
                bus_rows, bus_warnings = runner._bus_rows(
                    reporting_results, integer_counts, args.window
                )
                scenario_fields = [
                    "Window", "ScenarioID", "Block", "Station", "Duration",
                    "Probability", "WeightedTerminal", "Status", "Objective",
                    "BoardedAllVehicles", "QueueArea", "OnboardArea",
                    "WalkingArea", "SelectedTrips",
                ]
                bus_fields = [
                    "Window", "ScenarioID", "Duration", "Block",
                    "InformationCell", "FleetBusID", "FleetBusIDScope", "BusID",
                    "TripID", "Y", "Selected", "AssignedDepot", "Depot",
                    "DispatchStart", "RunStart", "RunEnd", "ServiceEnd",
                    "ResourceEnd", "RouteKey", "Route", "Direction", "RouteDir",
                    "RouteType", "StartStation", "EndStation", "ChainPosition",
                    "ChainLength", "PreviousTripID", "NextTripID", "IsFirstTrip",
                    "IsLastTrip", "ChainDecisionTime", "ChainInformationCell",
                    "DeadheadBeforeFrom", "DeadheadBeforeTo",
                    "DeadheadBeforeStart", "DeadheadBeforeEnd",
                    "DeadheadBeforeMinutes", "WaitBeforeRun", "ReturnDeadhead",
                    "ReturnStart", "ReturnEnd", "VehicleDutyStart",
                    "VehicleDutyEnd", "RouteStops", "StopTimeList", "Legs",
                    "BoardedPassengers", "MaxOnboard", "PeakLoadMinute",
                    "ActualLastPassengerMinute",
                ]
                scenario_files, bus_files = runner._write_scenario_csv_outputs(
                    output,
                    scenario_rows,
                    bus_rows,
                    reporting_scenarios,
                    args.window,
                    scenario_fields,
                    bus_fields,
                )
                vehicle_chain_schedule_rows = _vehicle_chain_schedule_rows(
                    bus_rows)
                vehicle_chain_schedule_files = (
                    _write_vehicle_chain_schedule_outputs(
                        runner,
                        output,
                        vehicle_chain_schedule_rows,
                        reporting_scenarios,
                        args.window,
                    )
                )
                decision_files, combined_decisions, decision_count = (
                    runner._write_decision_variable_outputs(
                        output,
                        reporting_results,
                        reporting_scenarios,
                        args.window,
                    )
                )

        summary = {
            "schema": SCHEMA,
            "solver": "FULL_POOL_MILP",
            "scope": "FULL_FINITE_MACRO_POOL",
            "column_generation": False,
            "extraction_mode": (
                "ALLOCATION_ONLY" if allocation_only else "FULL_SOLUTION"),
            "allocation_only": allocation_only,
            "cg_iterations": 0,
            "window": args.window,
            "k": args.k,
            "duration_set": list(MODELED_DURATIONS),
            "duration_probability_source": args.duration_probability_source,
            "duration_weights": dict(args.duration_weights),
            "representative_branches": str(
                args.representative_branches.resolve()),
            "disrupt_buffer_minutes": args.buffer,
            "fleet_size": FLEET_SIZE,
            "fleet_constraint": "equality",
            "depot_access_bound_minutes": DEPOT_ACCESS_BOUND_MIN,
            "min_trip_boarding": MIN_TRIP_BOARDING,
            "od_hourly_min_trips": OD_HOURLY_MIN_TRIPS,
            "walking_minutes": WALKING_MINUTES,
            "status": status,
            "termination_reason": (
                stats.get("TerminationReason") if stats else status
            ),
            "raw_candidate_macro_columns": raw_columns,
            "candidate_macro_columns": len(pool),
            "minimum_service_infeasible_columns_removed": (
                raw_columns - len(pool)),
            "exact_event_reformulation": True,
            "exact_reformulation": exact_flags,
            "nonanticipativity_scope": NONANTICIPATIVITY_SCOPE,
            "bus_decision_nonanticipativity": dict(
                BUS_DECISION_NONANTICIPATIVITY),
            "passenger_recourse_mode": PASSENGER_RECOURSE_MODE,
            "passenger_variable_sharing": passenger_variable_sharing,
            "strict_causal_passenger_nonanticipativity": False,
            "passenger_information_structure_audit": (
                passenger_recourse_audit),
            "trip_chain_enforced": True,
            "vehicle_chain_information_set_nonanticipative": bool(
                stats and stats.get(
                    "VehicleChainInformationSetNonanticipative")),
            "vehicle_chain_decision_scope": (
                stats.get("VehicleChainDecisionScope") if stats else None),
            "return_to_home_depot": True,
            "return_within_service_horizon": False,
            "solver_time_limit_seconds": args.time_limit_seconds,
            "target_gap": args.mip_gap,
            "bound_focus_profile": bool(args.bound_focus),
            "root_escape_profile": bool(args.root_escape),
            "concurrent_retry_profile": bool(args.concurrent_retry),
            "incumbent_target": args.incumbent_target,
            "solver_parameter_strategy": (
                "GUROBI_AUTOMATIC_DEFAULTS" if auto_parameters else
                (stats.get("SolverParameterStrategy") if stats else None)),
            "tuned_parameter_file": (
                stats.get("TunedParameterFile") if stats else None),
            "gurobi_tuning_seconds": (
                stats.get("TuningSeconds") if stats else None),
            "gurobi_tune_result_count": (
                stats.get("TuneResultCount") if stats else None),
            "gurobi_tune_time_limit_seconds": (
                float(args.tune_time_limit) if tune_parameters else None),
            "gurobi_tune_trials": (
                int(args.tune_trials) if tune_parameters else None),
            "gurobi_tune_candidate_time_limit_seconds": (
                float(args.tune_candidate_time_limit)
                if tune_parameters else None),
            "tune_only": tune_only,
            "objective": objective,
            "bound": bound,
            "relative_gap": relative_gap,
            "has_incumbent": has_incumbent,
            "bound_certified": bound is not None,
            "certified_within_gap": certified,
            "selected_column_count": len(selected_columns),
            "warm_start_depot_values": len(depot_start),
            "warm_start_selected_y": len(column_start),
            "warm_start_source": warm_start_source,
            "warm_start_application": (
                "VARIABLE_HINT_ONLY" if WARM_START_HINT_ONLY
                and (depot_start or column_start) else
                ("MIP_START_AND_VARIABLE_HINT"
                 if depot_start or column_start else None)
            ),
            "warm_start_chain_policy": (
                ("depot allocation variable hint only"
                 if WARM_START_HINT_ONLY else
                 "solver-repaired causal path cover")
                if depot_start or column_start else None),
            "depot_total": (
                sum(int(round(value)) for value in depot_counts.values())
                if has_incumbent else None
            ),
            "allocation_file": allocation_file,
            "selected_columns_file": selected_file,
            "scenario_results_files": scenario_files,
            "bus_operations_files": bus_files,
            "vehicle_chain_schedule_files": vehicle_chain_schedule_files,
            "decision_variables_files": decision_files,
            "decision_variables_combined_file": combined_decisions,
            "decision_variable_rows": decision_count,
            "model_solve_and_extract_seconds": solve_seconds,
            "production_solver_runtime_seconds": (
                stats.get("Runtime") if stats else None),
            "total_elapsed_seconds": time.time() - started,
            "stats": stats,
        }
        write_json(summary_path, summary)
        write_json(progress_path, {
            "schema": SCHEMA,
            "status": "FINISHED",
            "solver": "FULL_POOL_MILP",
            "scope": "FULL_FINITE_MACRO_POOL",
            "column_generation": False,
            "nonanticipativity_scope": NONANTICIPATIVITY_SCOPE,
            "passenger_recourse_mode": PASSENGER_RECOURSE_MODE,
            "window": args.window,
            "k": args.k,
            "elapsed_seconds": time.time() - started,
            "last_event": {"phase": "RUNNER_FINISHED"},
        })
        if tune_only and status == "TUNED":
            return 0
        return 0 if certified else (3 if has_incumbent else 2)
    except BaseException as exception:
        failure = {
            "schema": SCHEMA,
            "status": "FAILED",
            "solver": "FULL_POOL_MILP",
            "scope": "FULL_FINITE_MACRO_POOL",
            "column_generation": False,
            "nonanticipativity_scope": NONANTICIPATIVITY_SCOPE,
            "passenger_recourse_mode": PASSENGER_RECOURSE_MODE,
            "window": args.window,
            "k": args.k,
            "elapsed_seconds": time.time() - started,
            "error": f"{type(exception).__name__}: {exception}",
            "traceback": traceback.format_exc(),
        }
        write_json(summary_path, failure)
        write_json(progress_path, failure)
        raise
    finally:
        if environment is not None:
            environment.dispose()


def _module_path(module) -> Path:
    raw = getattr(module, "__file__", None)
    if not raw:
        raise RuntimeError(f"module has no source path: {module!r}")
    return Path(raw).resolve()


def validate_local_bundle(output_dir: Path = OUTPUT_DIR) -> dict:
    """Import every project dependency and prove that it is locally bundled."""
    configure_environment(output_dir)
    missing_core = [str(path) for path in LOCAL_MODULE_FILES if not path.is_file()]
    missing_inputs = [str(path) for path in STATIC_INPUTS if not path.is_file()]
    if missing_core or missing_inputs:
        raise FileNotFoundError(json.dumps({
            "missing_core": missing_core,
            "missing_static_inputs": missing_inputs,
        }, ensure_ascii=False))
    walking_input = load_uniform_walking_minutes(
        WALKING_TIME_FILE,
        STATIC_INPUTS[2],
        WALKING_MINUTES,
    )

    import cg_stage1 as model_engine

    runner = load_input_runner()
    nonanticipative = runner.load_na_module()
    core = nonanticipative.load_core()
    modules = {
        "input_runner": runner,
        "nonanticipative_adapter": nonanticipative,
        "stage1_core": core,
        "cg_stage1": model_engine,
    }
    for name in (
        "cg_problem_builder", "state_compression", "queue_compression",
        "walking_elimination", "stage1_recourse",
    ):
        module = sys.modules.get(name)
        if module is None:
            raise RuntimeError(f"bundled dependency was not imported: {name}")
        modules[name] = module

    core_root = CORE_DIR.resolve()
    imported = {}
    for name, module in modules.items():
        source = _module_path(module)
        try:
            source.relative_to(core_root)
        except ValueError as exception:
            raise RuntimeError(
                f"external project import is forbidden: {name} -> {source}"
            ) from exception
        imported[name] = str(source)

    engine_text = (ENGINE_DIR / "cg_stage1.py").read_text(
        encoding="utf-8", errors="strict")
    equality = (
        "gp.quicksum(x.values()) == config.max_total_buses" in engine_text
    )
    if not equality:
        raise AssertionError("bundled fleet-budget row is not an equality")
    if Path(runner.DEPOT_CANDIDATE_FILE).resolve() != STATIC_INPUTS[5].resolve():
        raise AssertionError("bundled depot-candidate path is not pipeline-local")
    if Path(core.FILE_CORRIDOR).resolve() != STATIC_INPUTS[3].resolve():
        raise AssertionError("bundled corridor path is not pipeline-local")
    if Path(core.FILE_EXCEL).resolve() != STATIC_INPUTS[4].resolve():
        raise AssertionError("bundled coordinate path is not pipeline-local")
    if Path(core.FILE_PREDEFINED_ROUTES).resolve() != ROUTE_FILE.resolve():
        raise AssertionError("bundled route path is not pipeline-local")
    if Path(core.DIR_NORMAL).resolve() != NORMAL_DIR.resolve():
        raise AssertionError("bundled normal-results path is not pipeline-local")
    if Path(core.DIR_DISRUPT).resolve() != DISRUPTION_DIR.resolve():
        raise AssertionError(
            "bundled disruption-results path is not pipeline-local")
    problem_builder = modules["cg_problem_builder"]
    if Path(problem_builder.ROUTE_FILE).resolve() != ROUTE_FILE.resolve():
        raise AssertionError("problem-builder route path is not pipeline-local")
    if (Path(problem_builder.DEPOT_CANDIDATE_FILE).resolve()
            != STATIC_INPUTS[5].resolve()):
        raise AssertionError(
            "problem-builder depot path is not pipeline-local")

    return {
        "core_root": str(core_root),
        "imports": imported,
        "fleet_constraint": "equality",
        "walking_time_input": walking_input,
        "gurobi_version": ".".join(
            str(value) for value in gp.gurobi.version()),
    }


def _validate_route_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Route", "Type", "StopSeq", "StationCode", "CumTime_min"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"route file lacks columns: {sorted(missing)}")
        rows = list(reader)
    routes = {str(row["Route"]).strip() for row in rows}
    local = {
        str(row["Route"]).strip() for row in rows
        if str(row["Type"]).strip().upper() == "LOC"
    }
    express = {
        str(row["Route"]).strip() for row in rows
        if str(row["Type"]).strip().upper() == "EXP"
    }
    if not rows or routes != local | express or not local or not express:
        raise ValueError("route file must contain nonempty LOC and EXP sets")
    return {
        "path": str(path.resolve()),
        "rows": len(rows),
        "routes": len(routes),
        "local_routes": len(local),
        "express_routes": len(express),
    }


def _validate_branch_file(path: Path, windows: tuple[int, ...]) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "hour", "k", "cluster", "representative_station", "duration",
            "cluster_size", "conditional_cluster_probability",
            "duration_weight", "scenario_probability", "method",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"representative-branch file lacks columns: {sorted(missing)}")
        rows = list(reader)
    methods = {str(row["method"]).strip() for row in rows}
    if len(methods) != 1 or not next(iter(methods), ""):
        raise ValueError(
            f"representative-branch file must contain one method: {methods}")
    by_window = {}
    for window in windows:
        block = [row for row in rows if int(row["hour"]) == window]
        expected = REPRESENTATIVES_PER_DURATION * len(MODELED_DURATIONS)
        if len(block) != expected:
            raise ValueError(
                f"window {window:02d}: expected {expected} branches, "
                f"found {len(block)}")
        if {int(row["duration"]) for row in block} != set(MODELED_DURATIONS):
            raise ValueError(f"window {window:02d}: duration support drift")
        if {int(row["k"]) for row in block} != {REPRESENTATIVES_PER_DURATION}:
            raise ValueError(f"window {window:02d}: K drift")
        station_sets = {
            duration: {
                str(row["representative_station"]).strip().upper()
                for row in block if int(row["duration"]) == duration
            }
            for duration in MODELED_DURATIONS
        }
        if any(len(value) != REPRESENTATIVES_PER_DURATION
               for value in station_sets.values()):
            raise ValueError(
                f"window {window:02d}: representative stations are not unique")
        if len({frozenset(value) for value in station_sets.values()}) != 1:
            raise ValueError(
                f"window {window:02d}: station support differs by duration")
        by_window[f"{window:02d}"] = len(block)
    return {
        "path": str(path.resolve()),
        "method": next(iter(methods)),
        "rows_by_window": by_window,
    }


def inspect_upstream(
    windows: tuple[int, ...], branch_file: Path = BRANCH_FILE
) -> dict:
    pending = []
    ready = {}
    if ROUTE_FILE.is_file():
        ready["routes"] = _validate_route_file(ROUTE_FILE)
    else:
        pending.append(str(ROUTE_FILE))
    if branch_file.is_file():
        ready["branches"] = _validate_branch_file(branch_file, windows)
    else:
        pending.append(str(branch_file))

    normal = {}
    disruptions = {}
    for window in windows:
        normal_path = NORMAL_DIR / f"Normal_{window:02d}_{window + 4:02d}.csv"
        if normal_path.is_file():
            normal[f"{window:02d}"] = str(normal_path.resolve())
        else:
            pending.append(str(normal_path))

        files_by_duration = {
            duration: list(DISRUPTION_DIR.glob(
                f"Disrupt_{window}_*_{duration}.csv"))
            for duration in MODELED_DURATIONS
        }
        files = [
            path for duration_files in files_by_duration.values()
            for path in duration_files
        ]
        manifest_candidates = (
            DISRUPTION_DIR / f"_scenario_stats_w{window:02d}.csv",
            DISRUPTION_DIR / "_scenario_stats.csv",
        )
        manifests = [path for path in manifest_candidates if path.is_file()]
        if all(files_by_duration.values()) and manifests:
            disruptions[f"{window:02d}"] = {
                "scenario_files": len({path.resolve() for path in files}),
                "files_by_duration": {
                    str(duration): len(duration_files)
                    for duration, duration_files in files_by_duration.items()
                },
                "manifests": [str(path.resolve()) for path in manifests],
            }
        else:
            for duration, duration_files in files_by_duration.items():
                if duration_files:
                    continue
                pending.append(
                    str(DISRUPTION_DIR /
                        f"Disrupt_{window}_*_{duration}.csv"))
            if not manifests:
                pending.append(
                    str(DISRUPTION_DIR / f"_scenario_stats_w{window:02d}.csv"))
    ready["normal"] = normal
    ready["disruptions"] = disruptions
    return {"ready": ready, "pending": pending}


def _resume_is_valid(
        window: int, requested_gap: float, output_dir: Path,
        probability_source: str, representative_branches: Path,
        accept_incumbent: bool = False,
        allocation_only: bool = False) -> bool:
    path = (Path(output_dir) / "Performance" /
            f"FullMILP_{window:02d}_summary.json")
    if not path.is_file():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        relative_gap = float(summary["relative_gap"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    exact_flags = summary.get("exact_reformulation", {})
    passenger_audit = summary.get("passenger_information_structure_audit", {})
    if not isinstance(exact_flags, dict) or not isinstance(passenger_audit, dict):
        return False
    return bool(
        summary.get("schema") == SCHEMA
        and summary.get("solver") == "FULL_POOL_MILP"
        and summary.get("scope") == "FULL_FINITE_MACRO_POOL"
        and summary.get("column_generation") is False
        and bool(summary.get("allocation_only", False)) == bool(allocation_only)
        and summary.get("extraction_mode", "FULL_SOLUTION") == (
            "ALLOCATION_ONLY" if allocation_only else "FULL_SOLUTION")
        and summary.get("nonanticipativity_scope")
        == NONANTICIPATIVITY_SCOPE
        and summary.get("bus_decision_nonanticipativity")
        == BUS_DECISION_NONANTICIPATIVITY
        and summary.get("passenger_recourse_mode")
        == PASSENGER_RECOURSE_MODE
        and summary.get("passenger_variable_sharing") is False
        and summary.get("strict_causal_passenger_nonanticipativity") is False
        and exact_flags.get(
            "trip_activation_information_set_nonanticipative") is True
        and exact_flags.get(
            "duty_initiation_information_set_nonanticipative") is True
        and exact_flags.get(
            "vehicle_chain_information_set_nonanticipative") is True
        and exact_flags.get(
            "duration_specific_passenger_omega_theta_preserved") is True
        and exact_flags.get("passenger_sharing_metadata_absent") is True
        and passenger_audit.get(
            "strict_passenger_na_preprocessor_invoked") is False
        and passenger_audit.get("passenger_variable_sharing") is False
        and passenger_audit.get("passenger_sharing_metadata_absent") is True
        and passenger_audit.get(
            "duration_specific_omega_theta_preserved") is True
        and int(summary.get("window", -1)) == int(window)
        and int(summary.get("k", -1)) == REPRESENTATIVES_PER_DURATION
        and tuple(summary.get("duration_set", ())) == MODELED_DURATIONS
        and summary.get("duration_probability_source") == probability_source
        and Path(summary.get("representative_branches", "")).resolve()
        == representative_branches.resolve()
        and all(math.isclose(
            float(summary.get("duration_weights", {}).get(str(duration), -1)),
            float(duration_weights(probability_source)[duration]),
            abs_tol=1e-12)
            for duration in MODELED_DURATIONS)
        and int(summary.get("disrupt_buffer_minutes", -1))
        == DISPATCH_BUFFER_MINUTES
        and int(summary.get("fleet_size", -1)) == FLEET_SIZE
        and float(summary.get("od_hourly_min_trips", -1.0))
        == OD_HOURLY_MIN_TRIPS
        and summary.get("fleet_constraint") == "equality"
        and int(summary.get("depot_total", -1)) == FLEET_SIZE
        and summary.get("has_incumbent") is True
        and summary.get(
            "vehicle_chain_information_set_nonanticipative") is True
        and math.isfinite(relative_gap)
        and (accept_incumbent
             or (summary.get("certified_within_gap") is True
                 and relative_gap <= float(requested_gap) + 1e-12))
    )


def _solve_task(payload) -> dict:
    (window, time_limit, mip_gap, threads, bound_focus, root_escape,
     concurrent_retry, incumbent_target, auto_parameters, tune_parameters,
     tune_time_limit, tune_trials, tune_candidate_time_limit, tune_only,
     tuning_dir, parameter_dir, output_dir,
     duration_probability_source, representative_branches,
     allocation_only, *rest) = payload
    export_model_dir = rest[0] if rest else EXPORT_MODEL_DIR
    options = argparse.Namespace(
        time_limit=float(time_limit), mip_gap=float(mip_gap),
        threads=int(threads), bound_focus=bool(bound_focus),
        root_escape=bool(root_escape),
        concurrent_retry=bool(concurrent_retry),
        incumbent_target=incumbent_target,
        auto_parameters=bool(auto_parameters),
        tune_parameters=bool(tune_parameters),
        tune_time_limit=float(tune_time_limit),
        tune_trials=int(tune_trials),
        tune_candidate_time_limit=float(tune_candidate_time_limit),
        tune_only=bool(tune_only),
        tuning_dir=Path(tuning_dir),
        parameter_dir=(
            Path(parameter_dir) if parameter_dir is not None else None),
        output_dir=Path(output_dir),
        duration_probability_source=str(duration_probability_source),
        representative_branches=Path(representative_branches),
        allocation_only=bool(allocation_only),
        export_model_dir=str(export_model_dir or ""),
    )
    code = solve_window(make_window_args(int(window), options))
    return {"window": int(window), "status": "SOLVED", "exit_code": int(code)}


def run_batch(options) -> int:
    local = validate_local_bundle(options.output_dir)
    upstream = inspect_upstream(
        options.windows, options.representative_branches)
    if upstream["pending"]:
        raise FileNotFoundError(
            "Stage 1 upstream outputs are incomplete:\n  "
            + "\n  ".join(upstream["pending"]))

    results = []
    pending_windows = []
    for window in options.windows:
        if _resume_is_valid(
                window, options.mip_gap, options.output_dir,
                options.duration_probability_source,
                options.representative_branches,
                options.accept_incumbent,
                bool(getattr(options, "allocation_only", False))):
            results.append({
                "window": window, "status": "RESUMED", "exit_code": 0})
        else:
            pending_windows.append(window)

    payloads = [
        (window, options.time_limit, options.mip_gap, options.threads,
         options.bound_focus, options.root_escape, options.concurrent_retry,
         options.incumbent_target, options.auto_parameters,
         options.tune_parameters, options.tune_time_limit,
         options.tune_trials, options.tune_candidate_time_limit,
         options.tune_only, options.tuning_dir, options.parameter_dir,
         options.output_dir, options.duration_probability_source)
         + (options.representative_branches,
            bool(getattr(options, "allocation_only", False)),
            str(getattr(options, "export_model_dir", EXPORT_MODEL_DIR) or ""))
        for window in pending_windows
    ]
    if options.workers == 1:
        for payload in payloads:
            try:
                results.append(_solve_task(payload))
            except BaseException as exception:
                results.append({
                    "window": payload[0], "status": "FAILED", "exit_code": 1,
                    "error": f"{type(exception).__name__}: {exception}",
                })
                break
    elif payloads:
        with ProcessPoolExecutor(
                max_workers=min(options.workers, len(payloads))) as executor:
            future_map = {
                executor.submit(_solve_task, payload): payload[0]
                for payload in payloads
            }
            for future in as_completed(future_map):
                window = future_map[future]
                try:
                    results.append(future.result())
                except BaseException as exception:
                    results.append({
                        "window": window, "status": "FAILED", "exit_code": 1,
                        "error": f"{type(exception).__name__}: {exception}",
                    })

    results.sort(key=lambda row: int(row["window"]))
    report = {
        "schema": SCHEMA,
        "status": (
            "COMPLETE" if all(
                row["exit_code"] == 0
                or (options.accept_incumbent and row["exit_code"] == 3)
                for row in results)
            else "INCOMPLETE"),
        "windows": list(options.windows),
        "workers": int(options.workers),
        "time_limit_seconds": float(options.time_limit),
        "mip_gap": float(options.mip_gap),
        "threads_per_window": int(options.threads),
        "output_dir": str(options.output_dir),
        "duration_probability_source": options.duration_probability_source,
        "duration_weights": duration_weights(
            options.duration_probability_source),
        "nonanticipativity_scope": NONANTICIPATIVITY_SCOPE,
        "bus_decision_nonanticipativity": dict(
            BUS_DECISION_NONANTICIPATIVITY),
        "passenger_recourse_mode": PASSENGER_RECOURSE_MODE,
        "passenger_variable_sharing": False,
        "strict_causal_passenger_nonanticipativity": False,
        "extraction_mode": (
            "ALLOCATION_ONLY"
            if bool(getattr(options, "allocation_only", False))
            else "FULL_SOLUTION"),
        "allocation_only": bool(getattr(options, "allocation_only", False)),
        "representative_branches": str(
            options.representative_branches),
        "accept_incumbent": bool(options.accept_incumbent),
        "bound_focus_profile": bool(options.bound_focus),
        "root_escape_profile": bool(options.root_escape),
        "concurrent_retry_profile": bool(options.concurrent_retry),
        "incumbent_target": options.incumbent_target,
        "local_bundle": local,
        "results": results,
    }
    write_json(
        options.output_dir / "Performance" / "batch_status.json", report)
    print(json.dumps(safe(report), ensure_ascii=False, indent=2))
    return 0 if report["status"] == "COMPLETE" else 1


def main(argv=None) -> int:
    options = parse_args(argv)
    if options.check_only:
        report = {
            "schema": SCHEMA,
            "status": "CHECKED",
            "configuration": {
                "windows": list(options.windows),
                "durations": list(MODELED_DURATIONS),
                "duration_probability_source": (
                    options.duration_probability_source),
                "duration_weights": duration_weights(
                    options.duration_probability_source),
                "output_dir": str(options.output_dir),
                "representative_branches": str(
                    options.representative_branches),
                "accept_incumbent": bool(options.accept_incumbent),
                "k": REPRESENTATIVES_PER_DURATION,
                "fleet": FLEET_SIZE,
                "fleet_constraint": "equality",
                "depot_access_bound_minutes": DEPOT_ACCESS_BOUND_MIN,
                "dispatch_buffer_minutes": DISPATCH_BUFFER_MINUTES,
                "min_trip_boarding": MIN_TRIP_BOARDING,
                "walking_minutes": WALKING_MINUTES,
                "lp_relax": False,
                "column_generation": False,
                "scope": "FULL_FINITE_MACRO_POOL",
                "nonanticipativity_scope": NONANTICIPATIVITY_SCOPE,
                "bus_decision_nonanticipativity": dict(
                    BUS_DECISION_NONANTICIPATIVITY),
                "passenger_recourse_mode": PASSENGER_RECOURSE_MODE,
                "passenger_variable_sharing": False,
                "strict_causal_passenger_nonanticipativity": False,
                "extraction_mode": (
                    "ALLOCATION_ONLY" if options.allocation_only
                    else "FULL_SOLUTION"),
                "allocation_only": bool(options.allocation_only),
                "root_escape_profile": bool(options.root_escape),
                "concurrent_retry_profile": bool(options.concurrent_retry),
            },
            "local_bundle": validate_local_bundle(options.output_dir),
            "upstream": inspect_upstream(
                options.windows, options.representative_branches),
        }
        report["upstream_status"] = (
            "READY" if not report["upstream"]["pending"]
            else "PENDING_PREVIOUS_STAGES")
        print(json.dumps(safe(report), ensure_ascii=False, indent=2))
        return 0
    return run_batch(options)


if __name__ == "__main__":
    raise SystemExit(main())
