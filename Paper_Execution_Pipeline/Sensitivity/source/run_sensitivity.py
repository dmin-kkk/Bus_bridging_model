"""Resumable Stage 2 sensitivity runner (Sec. 6.7) with its Stage 1 prerequisites.

Cases, windows, solver settings and file names come from the active profile
(``[evaluation.sensitivity]``, ``[evaluation.solver]``, ``[parallel]``).  Inputs
prepared by ``prepare_inputs.py`` and all results live below
``cfg.results_dir / paths.sensitivity``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve()
SOURCE = SCRIPT.parent
PIPELINE = SCRIPT.parents[2]
PACKAGE = SCRIPT.parents[3]
sys.path.insert(0, str(PACKAGE))

from bbconfig import cfg  # noqa: E402


PYTHON = Path(sys.executable).resolve()
EPS = 1e-9          # tolerance for gap comparisons
PROB_EPS = 1e-12    # tolerance for probability identities
WEIGHT_EPS = 1e-10  # tolerance for recorded Stage 2 branch weights
OBJ_REL_EPS = 1e-9  # objective/bound equality of LP-only (no integer) solves
OBJ_ABS_EPS = 1e-8
POLL_SECONDS = 30   # --wait-for-pid polling interval (wall clock only)
DURATIONS = tuple(int(d) for d in cfg.get("time.durations_min"))
LONG_DURATION = max(DURATIONS)
DURATION_COUNTS = {
    int(key): int(value) for key, value in cfg.get("probability.duration_counts").items()
}
STAGE1_DURATION_WEIGHTS = {
    duration: DURATION_COUNTS[duration] / sum(DURATION_COUNTS.values())
    for duration in DURATIONS
}
BUFFER_MINUTES = int(cfg.get("time.clearance_buffer_min"))
BASE_FLEET_SIZE = int(cfg.get("operations.fleet_size"))
BASE_MIN_TRIP_BOARDING = float(cfg.get("operations.min_trip_boarding"))
DEPOT_ACCESS_BOUND_MIN = int(cfg.get("operations.depot_access_bound_min"))
OD_HOURLY_MIN_TRIPS = float(cfg.get("operations.od_hourly_min_trips"))
BASE_K = int(cfg.get("clustering.k_per_window"))
WINDOWS = tuple(int(w) for w in cfg.get("evaluation.sensitivity.windows"))
STAGE1_K_WINDOWS = tuple(int(w) for w in cfg.get("evaluation.sensitivity.stage1_k_windows"))
LOGIC_VERSION = str(cfg.get("scenarios.logic_version"))
SENSITIVITY_ROOT = cfg.results_dir / cfg.get("paths.sensitivity")
INPUTS = SENSITIVITY_ROOT / "inputs"
REPRESENTATIVE_ROOT = INPUTS / "representatives"
CORRECTED_RESULTS = SENSITIVITY_ROOT
STAGE1_RESULT_ROOT = CORRECTED_RESULTS / "stage1"
STAGE1_PREREQUISITE_ROOT = CORRECTED_RESULTS / "stage1_prerequisites"
STAGE2_RESULT_ROOT = CORRECTED_RESULTS / "stage2"
BASELINE_ALLOCATION_DIR = cfg.results_dir / cfg.get("paths.stage1") / "Allocation"
STAGE1_DURATION_PROBABILITY_SOURCE = str(cfg.get("probability.duration_source"))
STAGE1_SCHEMA = (
    "paper-stage1-full-finite-pool-bus-only-na-"
    "passenger-wait-and-see-v1"
)
STAGE1_SENSITIVITY_SCHEMA = (
    "production-core-stage1-sensitivity-v3-"
    "bus-only-wait-and-see-cohort"
)
STAGE1_NONANTICIPATIVITY_SCOPE = "BUS_Y_U_PSI_ONLY"
STAGE1_PASSENGER_RECOURSE_MODE = "DURATION_SPECIFIC_WAIT_AND_SEE"
STAGE1_BUS_DECISION_NONANTICIPATIVITY = {
    "trip_activation_y": True,
    "duty_initiation_u": True,
    "vehicle_chaining_psi": True,
}
ACCEPT_INCUMBENT = bool(cfg.get("evaluation.solver.accept_incumbent"))
STAGE1_RUNNER = SOURCE / "run_stage1_full_pool_milp.py"
STAGE2_RUNNER = SOURCE / "run_stage2_sensitivity_case_bus_only.py"
STAGE2_SCHEMA = "paper-stage2-full-pool-joint-30-60-bus-only-na-v5"
STAGE2_SENSITIVITY_SCHEMA = (
    "production-core-stage2-sensitivity-v7-bus-only-wait-and-see"
)
STAGE2_FORMULATION_MODE = "BUS_ONLY_WAIT_AND_SEE"
STAGE2_NONANTICIPATIVITY_MODE = "bus-only"   # mode implemented by STAGE2_RUNNER
PREPARE_INPUTS = SOURCE / "prepare_inputs.py"
SCENARIO_STATS = cfg.results_dir / cfg.get("paths.disruptions") / "_scenario_stats.csv"
FINAL_LIGHTGBM_PROBABILITY_ROOT = cfg.results_dir / cfg.get("paths.probabilities")
SAVED_FLEET_ALLOCATION_ROOT = STAGE1_PREREQUISITE_ROOT
LIGHTGBM_PROBABILITY_SCHEMA = (
    "full-grid-fixed-type-lightgbm-refit-ensemble-v1"
)
PROBABILITY_MODEL = str(cfg.get("evaluation.oof.model"))
# Incident-type label -> slug of the ML stage's fixed-type probability files.
_PROBABILITY_SLUGS = {
    str(label): str(slug)
    for slug, label in cfg.get("prediction.full_grid.incident_types")
}
BASELINE_INCIDENT_TYPE = str(cfg.get("evaluation.sensitivity.baseline_incident_type"))
ROUTE_LIBRARY_FILE = str(cfg.get("routes.library_file"))
DISTANCE_FILE_STEM = Path(str(cfg.get("inputs.road_times"))).stem
DEPOT_FILE_NAME = Path(str(cfg.get("inputs.depots"))).name


def travel_time_label(multiplier: float) -> str:
    """File label of a road-time multiplier (0.8 -> TT080); same rule as prepare_inputs.py."""
    return f"TT{round(100 * float(multiplier)):03d}"


def probability_table(root: Path, incident_type: str) -> Path:
    """Fixed-type duration-probability table written by the ML stage below ``root``."""
    if incident_type not in _PROBABILITY_SLUGS:
        raise ValueError(
            f"incident type {incident_type!r} is not in prediction.full_grid.incident_types")
    return root / str(cfg.get("prediction.outputs.probability_file")).format(
        slug=_PROBABILITY_SLUGS[incident_type])


def representative_branch_name(k: int, window: int | None = None) -> str:
    """Representative-branch file name (``clustering.files.branches``), optionally per window."""
    name = str(cfg.get("clustering.files.branches")).format(
        k=k, durations="_".join(str(d) for d in DURATIONS))
    if window is None:
        return name
    path = Path(name)
    return f"{path.stem}_H{window:02d}{path.suffix}"


@dataclass(frozen=True)
class Stage2Case:
    case_id: str
    fleet_size: int = BASE_FLEET_SIZE
    allocation_source: str = "baseline"
    min_trip_boarding: float = BASE_MIN_TRIP_BOARDING
    travel_time_multiplier: float = 1.0
    route_name: str = ROUTE_LIBRARY_FILE
    distance_name: str = f"{DISTANCE_FILE_STEM}_{travel_time_label(1)}.csv"
    queue_multiplier: float = 1.0
    walking_multiplier: float = 1.0
    allocation_k: int | None = None
    nested_route_pool: str = "none"
    probability_file: Path = probability_table(
        FINAL_LIGHTGBM_PROBABILITY_ROOT, BASELINE_INCIDENT_TYPE)
    fixed_incident_type: str = BASELINE_INCIDENT_TYPE
    objective_comparable: bool = True


_CASE_KEYS = {
    "id", "fleet_size", "min_trip_boarding", "travel_time_multiplier",
    "route_pool", "out_of_vehicle_weight", "incident_type",
}


def build_stage2_case(spec: dict) -> Stage2Case:
    """Baseline setting plus the one-factor overrides of ``evaluation.sensitivity.cases``."""
    unknown = set(spec) - _CASE_KEYS
    if unknown:
        raise ValueError(f"sensitivity case {spec.get('id')}: unknown keys {sorted(unknown)}")
    fields: dict[str, object] = {"case_id": str(spec["id"]).strip().upper()}
    if "fleet_size" in spec:
        fields["fleet_size"] = int(spec["fleet_size"])
        fields["allocation_source"] = "fleet"
    if "min_trip_boarding" in spec:
        fields["min_trip_boarding"] = spec["min_trip_boarding"]
    if "travel_time_multiplier" in spec:
        multiplier = float(spec["travel_time_multiplier"])
        label = travel_time_label(multiplier)
        fields["travel_time_multiplier"] = multiplier
        fields["route_name"] = f"{Path(ROUTE_LIBRARY_FILE).stem}_{label}.csv"
        fields["distance_name"] = f"{DISTANCE_FILE_STEM}_{label}.csv"
    if "route_pool" in spec:
        pool = str(spec["route_pool"])
        if pool not in ("lower", "upper"):
            raise ValueError(f"{spec['id']}: route_pool must be 'lower' or 'upper'")
        fields["route_name"] = str(cfg.get(f"evaluation.sensitivity.{pool}_route_pool_file"))
        fields["nested_route_pool"] = pool
    if "out_of_vehicle_weight" in spec:
        weight = float(spec["out_of_vehicle_weight"])
        fields["queue_multiplier"] = weight
        fields["walking_multiplier"] = weight
        fields["objective_comparable"] = False
    if "incident_type" in spec:
        incident_type = str(spec["incident_type"])
        fields["fixed_incident_type"] = incident_type
        fields["probability_file"] = probability_table(
            FINAL_LIGHTGBM_PROBABILITY_ROOT, incident_type)
    return Stage2Case(**fields)


STAGE1_CASES = {
    f"S1_K{int(k)}": int(k)
    for k in cfg.get("evaluation.sensitivity.stage1_k_values")
}

STAGE2_CASES = {
    case.case_id: case
    for case in map(build_stage2_case, cfg.get("evaluation.sensitivity.cases"))
}


def configure_probability_root(root: Path) -> None:
    """Point every Stage-2 case at one explicit probability bundle."""
    global STAGE2_CASES
    configured = {}
    for case_id, case in STAGE2_CASES.items():
        configured[case_id] = replace(
            case,
            probability_file=probability_table(
                root, case.fixed_incident_type).resolve(),
        )
    STAGE2_CASES = configured


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    solver = "evaluation.solver"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        help=("case IDs; default: the Stage 2 cases of evaluation.sensitivity.cases "
              "(Stage 1 S1_K<k> cases run only when named here)"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--accept-incumbent", action=argparse.BooleanOptionalAction,
        default=bool(cfg.get(f"{solver}.accept_incumbent")),
        help="accept feasible one-shot solutions even when the gap is not certified",
    )
    parser.add_argument(
        "--output-root", type=Path, default=CORRECTED_RESULTS,
        help="isolated sensitivity result root",
    )
    parser.add_argument(
        "--baseline-allocation-dir", type=Path,
        default=BASELINE_ALLOCATION_DIR,
        help="Stage-1 allocations (main K and fleet) used by baseline-fleet cases",
    )
    parser.add_argument(
        "--existing-fleet-allocation-root", type=Path,
        default=SAVED_FLEET_ALLOCATION_ROOT,
        help=("saved N<fleet> Stage-1 prerequisite roots; used only "
              "with --use-existing-fleet-allocations"),
    )
    parser.add_argument(
        "--probability-root", type=Path,
        default=FINAL_LIGHTGBM_PROBABILITY_ROOT,
        help="root of the fixed-type Stage-2 probability tables",
    )
    parser.add_argument(
        "--require-lightgbm-probabilities", action=argparse.BooleanOptionalAction,
        default=bool(cfg.get("evaluation.sensitivity.require_lightgbm_probabilities")),
        help=("require the final classifier probability schema; "
              "reject preliminary tables"),
    )
    parser.add_argument(
        "--representative-root", type=Path, default=REPRESENTATIVE_ROOT,
        help="isolated directory containing Stage-1 sensitivity branches",
    )
    parser.add_argument(
        "--stage1-duration-probability-source",
        choices=("cohort",),
        default=STAGE1_DURATION_PROBABILITY_SOURCE,
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prerequisites-only", action="store_true")
    parser.add_argument(
        "--use-existing-fleet-allocations", action="store_true",
        help=("use saved feasible fleet allocations without rerunning "
              "uncertified Stage-1 cases"),
    )
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--threads", type=int,
                        default=int(cfg.get("parallel.sensitivity.threads")))
    parser.add_argument("--workers", type=int,
                        default=int(cfg.get("parallel.sensitivity.workers")))
    parser.add_argument("--stage1-threads", type=int,
                        default=int(cfg.get("parallel.fleet_stage1.threads")))
    parser.add_argument("--stage1-workers", type=int,
                        default=int(cfg.get("parallel.fleet_stage1.workers")))
    parser.add_argument(
        "--stage1-parameter-dir", type=Path,
        help="directory containing Gurobi-tuned Stage1_H<window>.prm files",
    )
    parser.add_argument(
        "--stage1-auto-parameters", action=argparse.BooleanOptionalAction,
        default=None,
        help=("use Gurobi automatic defaults for Stage 1 (default: "
              "evaluation.solver.stage1_auto_parameters unless --stage1-parameter-dir)"),
    )
    parser.add_argument(
        "--windows", nargs="+", type=int,
        choices=[int(w) for w in cfg.get("time.windows")],
        default=list(WINDOWS),
    )
    parser.add_argument("--time-limit-seconds", type=float,
                        default=float(cfg.get(f"{solver}.stage2_time_limit_s")))
    parser.add_argument(
        "--stage1-time-limit-seconds", type=float,
        default=float(cfg.get(f"{solver}.stage1_time_limit_s")),
        help="Stage 1 time limit of the fleet/K prerequisites",
    )
    parser.add_argument("--mip-gap", type=float, default=float(cfg.get(f"{solver}.mip_gap")))
    parser.add_argument("--mip-focus", type=int, default=int(cfg.get(f"{solver}.mip_focus")))
    parser.add_argument("--heuristics", type=float,
                        default=float(cfg.get(f"{solver}.heuristics")))
    parser.add_argument("--cuts", type=int, default=int(cfg.get(f"{solver}.cuts")))
    parser.add_argument(
        "--wait-for-pid", type=int,
        help="wait for this process to finish before starting any solve",
    )
    args = parser.parse_args()
    if args.stage1_auto_parameters is None:
        args.stage1_auto_parameters = (
            args.stage1_parameter_dir is None
            and bool(cfg.get(f"{solver}.stage1_auto_parameters"))
        )
    if args.time_limit_seconds <= 0:
        parser.error("--time-limit-seconds must be positive")
    if (args.stage1_time_limit_seconds is not None and
            args.stage1_time_limit_seconds <= 0):
        parser.error("--stage1-time-limit-seconds must be positive")
    if not 0 <= args.mip_gap < 1:
        parser.error("--mip-gap must be in [0, 1)")
    if args.mip_focus not in (0, 1, 2, 3):
        parser.error("--mip-focus must be one of 0, 1, 2, 3")
    if not 0 <= args.heuristics <= 1:
        parser.error("--heuristics must be in [0, 1]")
    if args.cuts not in (-1, 0, 1, 2, 3):
        parser.error("--cuts must be one of -1, 0, 1, 2, 3")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.stage1_workers is not None and args.stage1_workers < 1:
        parser.error("--stage1-workers must be positive")
    if args.stage1_threads is not None and args.stage1_threads < 0:
        parser.error("--stage1-threads must be nonnegative")
    if args.stage1_parameter_dir is not None and args.stage1_auto_parameters:
        parser.error(
            "--stage1-parameter-dir and --stage1-auto-parameters are exclusive"
        )
    args.output_root = args.output_root.resolve()
    args.baseline_allocation_dir = args.baseline_allocation_dir.resolve()
    args.existing_fleet_allocation_root = (
        args.existing_fleet_allocation_root.resolve()
    )
    args.probability_root = args.probability_root.resolve()
    args.representative_root = args.representative_root.resolve()
    return args


def selected_case_ids(raw: list[str] | None) -> list[str]:
    available = [*STAGE1_CASES, *STAGE2_CASES]
    if not raw:
        return list(STAGE2_CASES)
    selected = []
    for token in raw:
        for case_id in token.split(","):
            case_id = case_id.strip().upper()
            if case_id and case_id not in selected:
                selected.append(case_id)
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"unknown case IDs: {unknown}")
    return selected


def wait_for_process(process_id: int) -> None:
    if process_id <= 0:
        raise ValueError("--wait-for-pid must be positive")
    print(f"WAIT process={process_id}", flush=True)
    while True:
        try:
            os.kill(process_id, 0)
        except OSError:
            break
        time.sleep(POLL_SECONDS)
    print(f"WAIT_COMPLETE process={process_id}", flush=True)


def run_process(
    command: list[str],
    *,
    log_path: Path | None = None,
    environment: dict[str, str] | None = None,
    dry_run: bool = False,
) -> int:
    print("COMMAND " + subprocess.list2cmdline(command), flush=True)
    if dry_run:
        return 0
    if log_path is None:
        return subprocess.run(
            command, cwd=PIPELINE, env=environment, check=False
        ).returncode
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.run(
            command,
            cwd=PIPELINE,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def stage1_is_complete(
    output: Path,
    window: int,
    k: int,
    fleet_size: int,
    mip_gap: float,
    parameter_file: Path | None = None,
    auto_parameters: bool = False,
) -> bool:
    summary_path = output / "Performance" / f"FullMILP_{window:02d}_summary.json"
    allocation_path = output / "Allocation" / f"DepotAlloc_{window:02d}.csv"
    if not summary_path.is_file() or not allocation_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    stats = summary.get("stats") or {}
    exact = summary.get("exact_reformulation") or {}
    passenger_audit = (
        summary.get("passenger_information_structure_audit") or {})
    sensitivity = summary.get("sensitivity") or {}
    gap = float(stats.get("MIPGap", summary.get("relative_gap", math.inf)))
    tuned_parameter_matches = True
    if parameter_file is not None:
        recorded = stats.get("TunedParameterFile")
        tuned_parameter_matches = bool(
            stats.get("SolverParameterStrategy")
            == "GUROBI_TUNED_PARAMETER_FILE"
            and recorded
            and Path(str(recorded)).resolve() == parameter_file.resolve()
        )
    automatic_parameters_match = bool(
        not auto_parameters
        or summary.get("solver_parameter_strategy")
        == "GUROBI_AUTOMATIC_DEFAULTS"
    )
    return bool(
        summary.get("schema") == STAGE1_SCHEMA
        and summary.get("column_generation") is False
        and summary.get("nonanticipativity_scope")
        == STAGE1_NONANTICIPATIVITY_SCOPE
        and summary.get("bus_decision_nonanticipativity")
        == STAGE1_BUS_DECISION_NONANTICIPATIVITY
        and summary.get("passenger_recourse_mode")
        == STAGE1_PASSENGER_RECOURSE_MODE
        and summary.get("passenger_variable_sharing") is False
        and summary.get("strict_causal_passenger_nonanticipativity") is False
        and exact.get(
            "trip_activation_information_set_nonanticipative") is True
        and exact.get(
            "duty_initiation_information_set_nonanticipative") is True
        and exact.get(
            "vehicle_chain_information_set_nonanticipative") is True
        and exact.get(
            "duration_specific_passenger_omega_theta_preserved") is True
        and exact.get("passenger_sharing_metadata_absent") is True
        and passenger_audit.get(
            "strict_passenger_na_preprocessor_invoked") is False
        and passenger_audit.get("passenger_variable_sharing") is False
        and passenger_audit.get(
            "passenger_sharing_metadata_absent") is True
        and passenger_audit.get(
            "duration_specific_omega_theta_preserved") is True
        and stats.get("PassengerFlowVariableSharing") is False
        and summary.get("has_incumbent") is True
        and (ACCEPT_INCUMBENT or (
            summary.get("certified_within_gap") is True
            and gap <= mip_gap + EPS))
        and int(summary.get("k", -1)) == k
        and int(summary.get("fleet_size", -1)) == fleet_size
        and int(summary.get("depot_total", -1)) == fleet_size
        and float(summary.get("min_trip_boarding", math.nan))
        == BASE_MIN_TRIP_BOARDING
        and int(summary.get("disrupt_buffer_minutes", -1)) == BUFFER_MINUTES
        and summary.get("warm_start_source") is None
        and int(summary.get("warm_start_depot_values", -1)) == 0
        and int(summary.get("warm_start_selected_y", -1)) == 0
        and float(summary.get("od_hourly_min_trips", -1.0))
        == OD_HOURLY_MIN_TRIPS
        and sensitivity.get("schema") == STAGE1_SENSITIVITY_SCHEMA
        and sensitivity.get("production_stage1_schema") == STAGE1_SCHEMA
        and sensitivity.get("nonanticipativity_scope")
        == STAGE1_NONANTICIPATIVITY_SCOPE
        and sensitivity.get("bus_decision_nonanticipativity")
        == STAGE1_BUS_DECISION_NONANTICIPATIVITY
        and sensitivity.get("passenger_recourse_mode")
        == STAGE1_PASSENGER_RECOURSE_MODE
        and sensitivity.get("passenger_variable_sharing") is False
        and float(sensitivity.get("min_trip_boarding", math.nan))
        == BASE_MIN_TRIP_BOARDING
        and int(sensitivity.get("disrupt_buffer_minutes", -1))
        == BUFFER_MINUTES
        and sensitivity.get("warm_start_enabled") is False
        and summary.get("duration_probability_source")
        == STAGE1_DURATION_PROBABILITY_SOURCE
        and all(
            abs(float((summary.get("duration_weights") or {}).get(
                str(duration), math.nan)) - weight) <= PROB_EPS
            for duration, weight in STAGE1_DURATION_WEIGHTS.items()
        )
        and bool((summary.get("stats") or {}).get(
            "VehicleChainInformationSetNonanticipative"))
        and tuned_parameter_matches
        and automatic_parameters_match
    )


def run_stage1(
    *,
    label: str,
    output: Path,
    k: int,
    fleet_size: int,
    representatives: Path,
    threads: int,
    time_limit_seconds: float,
    mip_gap: float,
    dry_run: bool,
    workers: int = 1,
    parameter_dir: Path | None = None,
    auto_parameters: bool = False,
    windows: tuple[int, ...] | None = None,
) -> None:
    route_file = INPUTS / "routes" / ROUTE_LIBRARY_FILE
    solve_windows = tuple(WINDOWS if windows is None else windows)

    def solve(window: int) -> None:
        parameter_file = (
            parameter_dir.resolve() / f"Stage1_H{window:02d}.prm"
            if parameter_dir is not None else None
        )
        command = [
            str(PYTHON),
            "-X",
            "utf8",
            "-u",
            str(STAGE1_RUNNER),
            "--case-id",
            label,
            "--window",
            str(window),
            "--route-file",
            str(route_file),
            "--representative-branches",
            str(representatives),
            "--output",
            str(output),
            "--k",
            str(k),
            "--fleet-size",
            str(fleet_size),
            "--buffer",
            str(BUFFER_MINUTES),
            "--min-trip-boarding",
            f"{BASE_MIN_TRIP_BOARDING:g}",
            "--no-warm-start",
            "--time-limit-seconds",
            str(time_limit_seconds),
            "--mip-gap",
            str(mip_gap),
            "--threads",
            str(threads),
            "--duration-probability-source",
            STAGE1_DURATION_PROBABILITY_SOURCE,
        ]
        if parameter_file is not None:
            command.extend(["--parameter-file", str(parameter_file)])
        if auto_parameters:
            command.append("--auto-parameters")
        environment = os.environ.copy()
        environment.update(
            {
                "TSB_MAX_BUSES": str(fleet_size),
                "TSB_DEPOT_ACCESS_BOUND": str(DEPOT_ACCESS_BOUND_MIN),
                "TSB_ROUTE_SOURCE": "predefined",
                "TSB_BUS_BOARD": "general",
            }
        )
        log = output / "BatchLogs" / f"FullMILP_{window:02d}.log"
        print(
            f"RUN {label} H{window:02d} K={k} fleet={fleet_size}", flush=True
        )
        return_code = run_process(
            command,
            log_path=log,
            environment=environment,
            dry_run=dry_run,
        )
        if not dry_run and not stage1_is_complete(
            output, window, k, fleet_size, mip_gap, parameter_file,
            auto_parameters,
        ):
            raise RuntimeError(
                f"{label} H{window:02d} did not produce a usable allocation; "
                f"exit={return_code}, log={log}"
            )

    pending = []
    for window in solve_windows:
        parameter_file = (
            parameter_dir.resolve() / f"Stage1_H{window:02d}.prm"
            if parameter_dir is not None else None
        )
        if stage1_is_complete(
            output, window, k, fleet_size, mip_gap, parameter_file,
            auto_parameters,
        ):
            print(f"SKIP {label} H{window:02d}: usable result exists", flush=True)
        else:
            pending.append(window)
    if workers == 1:
        for window in pending:
            solve(window)
    elif pending:
        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = [executor.submit(solve, window) for window in pending]
            for future in as_completed(futures):
                future.result()


def ensure_fleet_allocations(
    fleet_size: int,
    *,
    threads: int,
    time_limit_seconds: float,
    mip_gap: float,
    dry_run: bool,
    workers: int,
    parameter_dir: Path | None,
    auto_parameters: bool,
) -> Path:
    output = STAGE1_PREREQUISITE_ROOT / f"N{fleet_size:02d}"
    representatives = REPRESENTATIVE_ROOT / representative_branch_name(BASE_K)
    run_stage1(
        label=f"PREREQ_N{fleet_size:02d}",
        output=output,
        k=BASE_K,
        fleet_size=fleet_size,
        representatives=representatives,
        threads=threads,
        time_limit_seconds=time_limit_seconds,
        mip_gap=mip_gap,
        dry_run=dry_run,
        workers=workers,
        parameter_dir=parameter_dir,
        auto_parameters=auto_parameters,
    )
    return output / "Allocation"


def discover_stage2_jobs() -> list[tuple[int, str]]:
    frame = pd.read_csv(SCENARIO_STATS)
    required = {"window", "target", "duration", "checkpoint", "logic_version"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"scenario stats missing columns: {sorted(missing)}")
    frame["hour"] = frame["window"].astype(str).str.split("_").str[0].astype(int)
    valid = frame[
        frame["hour"].isin(WINDOWS)
        & frame["duration"].isin(DURATIONS)
        & frame["checkpoint"].eq("EXTRACTED")
        & frame["logic_version"].eq(LOGIC_VERSION)
    ].copy()
    if valid.duplicated(["hour", "target", "duration"], keep=False).any():
        raise ValueError("duplicate current scenario rows found")
    grouped = valid.groupby(["hour", "target"])["duration"].agg(set)
    jobs = []
    for (hour, station), durations in grouped.items():
        if LONG_DURATION not in durations:
            raise ValueError(
                f"H{hour:02d}/{station}: D{LONG_DURATION} input is absent")
        jobs.append((int(hour), str(station).strip().upper()))
    jobs = sorted(jobs)
    if not jobs:
        raise ValueError(f"no current D60 jobs found for windows {WINDOWS}")
    observed_windows = {window for window, _ in jobs}
    missing_windows = sorted(set(WINDOWS).difference(observed_windows))
    if missing_windows:
        raise ValueError(f"current D60 jobs absent for windows {missing_windows}")
    return jobs


def allocation_total(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return sum(int(round(float(row["Count"]))) for row in rows)


def result_file_exists(raw_path: object) -> bool:
    if not raw_path:
        return False
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = PIPELINE / path
    return path.is_file()


_PROBABILITY_CACHE: dict[Path, dict[tuple[int, str], tuple[float, float]]] = {}


def probability_weights(path: Path) -> dict[tuple[int, str], tuple[float, float]]:
    path = path.resolve()
    if path not in _PROBABILITY_CACHE:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        required = {"window", "station", "p30", "p60"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"probability table missing columns {sorted(missing)}: {path}"
            )
        frame["window"] = pd.to_numeric(frame["window"], errors="raise").astype(int)
        frame["station"] = frame["station"].astype(str).str.strip().str.upper()
        frame["p30"] = pd.to_numeric(frame["p30"], errors="raise")
        frame["p60"] = pd.to_numeric(frame["p60"], errors="raise")
        if frame[["window", "station"]].duplicated().any():
            raise ValueError(f"duplicate probability rows: {path}")
        if not all(
            math.isfinite(float(value))
            for value in frame[["p30", "p60"]].to_numpy().ravel()
        ):
            raise ValueError(f"non-finite duration probability: {path}")
        if (
            (frame[["p30", "p60"]] < 0).any().any()
            or (frame[["p30", "p60"]] > 1).any().any()
        ):
            raise ValueError(f"duration probability outside [0, 1]: {path}")
        if not all(
            math.isclose(float(row.p30 + row.p60), 1.0, abs_tol=PROB_EPS)
            for row in frame.itertuples(index=False)
        ):
            raise ValueError(f"duration probabilities do not sum to one: {path}")
        _PROBABILITY_CACHE[path] = {
            (int(row.window), str(row.station)): (float(row.p30), float(row.p60))
            for row in frame.itertuples(index=False)
        }
    return _PROBABILITY_CACHE[path]


def validate_stage2_probability_inputs(
    cases: list[Stage2Case],
    jobs: list[tuple[int, str]],
    *,
    require_lightgbm: bool,
) -> None:
    """Validate model provenance, fixed type, and complete job coverage."""
    expected_jobs = {
        (int(window), str(station).strip().upper())
        for window, station in jobs
    }
    checked: set[Path] = set()
    for case in cases:
        path = case.probability_file.resolve()
        weights = probability_weights(path)
        missing_jobs = sorted(expected_jobs - set(weights))
        if missing_jobs:
            sample = ", ".join(
                f"H{window:02d}/{station}"
                for window, station in missing_jobs[:5]
            )
            raise ValueError(
                f"{path}: missing {len(missing_jobs)} Stage-2 jobs ({sample})"
            )
        if path in checked:
            continue
        checked.add(path)
        if not require_lightgbm:
            continue
        frame = pd.read_csv(path, encoding="utf-8-sig")
        required = {
            "model_family", "training_cohort", "fixed_incident_type",
            "probability_schema",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"final {PROBABILITY_MODEL} probability metadata absent "
                f"{sorted(missing)}: {path}"
            )
        if set(frame["model_family"].astype(str)) != {PROBABILITY_MODEL}:
            raise ValueError(
                f"preliminary/non-{PROBABILITY_MODEL} probabilities rejected: {path}")
        if set(frame["training_cohort"].astype(str)) != {
            STAGE1_DURATION_PROBABILITY_SOURCE
        }:
            raise ValueError(
                f"non-{STAGE1_DURATION_PROBABILITY_SOURCE} probabilities rejected: {path}")
        if set(frame["fixed_incident_type"].astype(str)) != {
            case.fixed_incident_type
        }:
            raise ValueError(f"fixed incident type mismatch: {path}")
        if set(frame["probability_schema"].astype(str)) != {
            LIGHTGBM_PROBABILITY_SCHEMA
        }:
            raise ValueError(
                f"final {PROBABILITY_MODEL} probability schema mismatch: {path}")


def stage2_is_complete(
    output: Path,
    case_id: str,
    window: int,
    station: str,
    mip_gap: float,
    mip_focus: int,
    heuristics: float,
    cuts: int,
) -> bool:
    summary_path = (
        output / "Performance" / f"FullMILP_{window:02d}_{station}_summary.json"
    )
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    sensitivity = summary.get("sensitivity") or {}
    stats = summary.get("stats") or {}
    decision_files = summary.get("decision_variable_files") or {}
    route_audit = sensitivity.get("route_nesting_audit") or []
    nesting_mode = sensitivity.get("nested_route_pool", "none")
    case = STAGE2_CASES[case_id]
    expected = probability_weights(case.probability_file)[
        (int(window), str(station).strip().upper())
    ]
    actual = sensitivity.get("duration_weights") or {}
    raw_gap = stats.get("MIPGap")
    if raw_gap is None:
        status = str(stats.get("Status", "")).strip().upper()
        integer_variables = int(stats.get("NumIntVars", -1))
        objective = float(stats.get("ObjVal", math.nan))
        bound = float(stats.get("ObjBound", math.nan))
        gap = (
            0.0
            if status == "OPTIMAL"
            and integer_variables == 0
            and math.isfinite(objective)
            and math.isfinite(bound)
            and math.isclose(objective, bound, rel_tol=OBJ_REL_EPS, abs_tol=OBJ_ABS_EPS)
            else math.inf
        )
    else:
        gap = float(raw_gap)
    return bool(
        summary.get("schema") == STAGE2_SCHEMA
        and summary.get("column_generation") is False
        and summary.get("exact_state_compression") is True
        and summary.get("has_incumbent") is True
        and (ACCEPT_INCUMBENT or gap <= mip_gap + EPS)
        and float(summary.get("od_hourly_min_trips", -1.0)) == OD_HOURLY_MIN_TRIPS
        and sensitivity.get("schema") == STAGE2_SENSITIVITY_SCHEMA
        and sensitivity.get("formulation_mode") == STAGE2_FORMULATION_MODE
        and sensitivity.get("nonanticipativity_mode")
        == STAGE2_NONANTICIPATIVITY_MODE
        and sensitivity.get("exact_state_compression") is True
        and sensitivity.get("case_id") == case_id
        and int(sensitivity.get("fleet_size", -1)) == int(case.fleet_size)
        and math.isclose(
            float(sensitivity.get("min_trip_boarding", math.nan)),
            float(case.min_trip_boarding), abs_tol=PROB_EPS)
        and math.isclose(
            float(sensitivity.get("travel_time_multiplier", math.nan)),
            float(case.travel_time_multiplier), abs_tol=PROB_EPS)
        and math.isclose(
            float(sensitivity.get("queue_multiplier", math.nan)),
            float(case.queue_multiplier), abs_tol=PROB_EPS)
        and math.isclose(
            float(sensitivity.get("walking_multiplier", math.nan)),
            float(case.walking_multiplier), abs_tol=PROB_EPS)
        and sensitivity.get("fixed_incident_type")
        == case.fixed_incident_type
        and Path(sensitivity.get("duration_probability_file", "")).resolve()
        == case.probability_file.resolve()
        and all(
            abs(float(actual.get(str(duration), actual.get(duration, math.nan)))
                - expected[index]) <= WEIGHT_EPS
            for index, duration in enumerate(DURATIONS)
        )
        and sensitivity.get("directed_route_timing")
        == "raw-directed-cumulative-rounding-validated"
        and int((sensitivity.get("solver_profile") or {}).get(
            "mip_focus", -1)) == mip_focus
        and abs(float((sensitivity.get("solver_profile") or {}).get(
            "heuristics", math.nan)) - heuristics) <= PROB_EPS
        and int((sensitivity.get("solver_profile") or {}).get(
            "cuts", -99)) == cuts
        and (
            nesting_mode == "none"
            or route_audit
            and all(bool(row.get("nesting_valid")) for row in route_audit)
        )
        and bool((summary.get("stats") or {}).get(
            "VehicleChainInformationSetNonanticipative"))
        and bool((summary.get("stats") or {}).get(
            "ExactStateCompression"))
        and stats.get("NonanticipativityMode") == STAGE2_NONANTICIPATIVITY_MODE
        and stats.get("FormulationMode") == STAGE2_FORMULATION_MODE
        and stats.get("BusDecisionInformationSetNonanticipative") is True
        and stats.get("PassengerFlowVariableSharing") is False
        and int(stats.get("PassengerSharedEqualityConstrs", -1)) == 0
        and stats.get("PassengerRecourse") == "WAIT_AND_SEE"
        and stats.get(
            "TrainAvailabilityInformationSetNonanticipative") is False
        and stats.get("DurationSpecificPhysicalOmega") is True
        and stats.get("DurationSpecificRailState") is True
        and result_file_exists(summary.get("chain_policy_file"))
        and all(
            str(duration) in decision_files
            and result_file_exists(decision_files[str(duration)])
            for duration in DURATIONS
        )
    )


def allocation_for_case(
    case: Stage2Case,
    allocation_dirs: dict[str, Path],
    window: int,
    *,
    validate: bool = True,
) -> Path:
    if case.allocation_source == "baseline":
        directory = BASELINE_ALLOCATION_DIR
    elif case.allocation_source == "k":
        if case.allocation_k is None:
            raise ValueError(f"{case.case_id}: missing allocation_k")
        directory = allocation_dirs[f"K{case.allocation_k}"]
    else:
        directory = allocation_dirs[f"N{case.fleet_size}"]
    path = directory / f"DepotAlloc_{window:02d}.csv"
    if not validate:
        return path
    if not path.is_file():
        raise FileNotFoundError(path)
    total = allocation_total(path)
    if total != case.fleet_size:
        raise ValueError(
            f"{case.case_id} H{window:02d}: allocation total {total}, "
            f"expected {case.fleet_size}"
        )
    return path


def run_stage2_case(
    case: Stage2Case,
    *,
    jobs: list[tuple[int, str]],
    allocation_dirs: dict[str, Path],
    threads: int,
    time_limit_seconds: float,
    mip_gap: float,
    mip_focus: int,
    heuristics: float,
    cuts: int,
    dry_run: bool,
    workers: int = 1,
) -> None:
    output = STAGE2_RESULT_ROOT / case.case_id
    progress_path = output / "BatchStatus" / "batch_progress.json"
    completed_jobs = sum(
        stage2_is_complete(
            output, case.case_id, window, station, mip_gap,
            mip_focus, heuristics, cuts,
        )
        for window, station in jobs
    )
    failures = 0
    if not dry_run:
        write_json(
            progress_path,
            {
                "state": "RUNNING",
                "formulation_mode": STAGE2_FORMULATION_MODE,
                "nonanticipativity_mode": STAGE2_NONANTICIPATIVITY_MODE,
                "case_id": case.case_id,
                "windows": list(WINDOWS),
                "total_jobs": len(jobs),
                "completed_jobs": completed_jobs,
                "failures": failures,
                "workers": workers,
                "threads_per_case": threads,
                "mip_focus": mip_focus,
                "heuristics": heuristics,
                "cuts": cuts,
                "accept_incumbent": ACCEPT_INCUMBENT,
            },
        )

    def solve(index: int, window: int, station: str) -> dict[str, object]:
        allocation = allocation_for_case(
            case, allocation_dirs, window, validate=not dry_run
        )
        route_file = INPUTS / "routes" / case.route_name
        distance_file = INPUTS / "travel_time" / case.distance_name
        command = [
            str(PYTHON),
            "-X",
            "utf8",
            "-u",
            str(STAGE2_RUNNER),
            "--case-id",
            case.case_id,
            "--window",
            str(window),
            "--station",
            station,
            "--allocation",
            str(allocation),
            "--fleet-size",
            str(case.fleet_size),
            "--route-file",
            str(route_file),
            "--distance-file",
            str(distance_file),
            "--depot-file",
            str(INPUTS / "static" / DEPOT_FILE_NAME),
            "--output",
            str(output),
            "--min-trip-boarding",
            f"{float(case.min_trip_boarding):g}",
            "--travel-time-multiplier",
            str(case.travel_time_multiplier),
            "--queue-multiplier",
            str(case.queue_multiplier),
            "--walking-multiplier",
            str(case.walking_multiplier),
            "--nested-route-pool",
            case.nested_route_pool,
            "--probability-file",
            str(case.probability_file),
            "--fixed-incident-type",
            case.fixed_incident_type,
            "--buffer",
            str(BUFFER_MINUTES),
            "--time-limit-seconds",
            str(time_limit_seconds),
            "--mip-gap",
            str(mip_gap),
            "--threads",
            str(threads),
            "--mip-focus",
            str(mip_focus),
            "--heuristics",
            str(heuristics),
            "--cuts",
            str(cuts),
        ]
        mip_start_policy = (
            output / "Policies" /
            f"FullMILP_{window:02d}_{station}_policy.csv"
        )
        if mip_start_policy.is_file():
            command.extend(["--mip-start-policy", str(mip_start_policy)])
        log = (
            output
            / "BatchLogs"
            / f"FullMILP_{window:02d}_{station}.log"
        )
        print(
            f"RUN {case.case_id} {index}/{len(jobs)} "
            f"H{window:02d}/{station}",
            flush=True,
        )
        return_code = run_process(command, log_path=log, dry_run=dry_run)
        return {
            "window": window,
            "station": station,
            "return_code": return_code,
            "log_parent": str(log.parent),
        }

    pending = []
    for index, (window, station) in enumerate(jobs, start=1):
        if stage2_is_complete(
            output, case.case_id, window, station, mip_gap,
            mip_focus, heuristics, cuts,
        ):
            print(
                f"SKIP {case.case_id} {index}/{len(jobs)} "
                f"H{window:02d}/{station}",
                flush=True,
            )
        else:
            pending.append((index, window, station))

    if workers == 1:
        batch_results = (solve(*payload) for payload in pending)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [executor.submit(solve, *payload) for payload in pending]
        batch_results = (future.result() for future in as_completed(futures))
    try:
        for result in batch_results:
            if dry_run:
                continue
            window = int(result["window"])
            station = str(result["station"])
            return_code = int(result["return_code"])
            if return_code in (0, 3) and stage2_is_complete(
                output, case.case_id, window, station, mip_gap,
                mip_focus, heuristics, cuts,
            ):
                completed_jobs += 1
            else:
                failures += 1
            write_json(
                progress_path,
                {
                        "state": "RUNNING",
                        "formulation_mode": STAGE2_FORMULATION_MODE,
                        "nonanticipativity_mode": STAGE2_NONANTICIPATIVITY_MODE,
                    "case_id": case.case_id,
                    "windows": list(WINDOWS),
                    "total_jobs": len(jobs),
                    "completed_jobs": completed_jobs,
                    "failures": failures,
                    "workers": workers,
                    "threads_per_case": threads,
                    "mip_focus": mip_focus,
                    "heuristics": heuristics,
                    "cuts": cuts,
                    "accept_incumbent": ACCEPT_INCUMBENT,
                    "last_window": window,
                    "last_station": station,
                    "last_exit_code": return_code,
                },
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    if not dry_run:
        state = "COMPLETE" if failures == 0 else "COMPLETE_WITH_FAILURES"
        write_json(
            progress_path,
            {
                "state": state,
                "formulation_mode": STAGE2_FORMULATION_MODE,
                "nonanticipativity_mode": STAGE2_NONANTICIPATIVITY_MODE,
                "case_id": case.case_id,
                "windows": list(WINDOWS),
                "total_jobs": len(jobs),
                "completed_jobs": completed_jobs,
                "failures": failures,
                "workers": workers,
                "threads_per_case": threads,
                "mip_focus": mip_focus,
                "heuristics": heuristics,
                "cuts": cuts,
                "accept_incumbent": ACCEPT_INCUMBENT,
            },
        )
        if failures:
            message = f"{case.case_id}: completed with {failures} failures"
            if ACCEPT_INCUMBENT:
                print(f"CONTINUE_WITH_RECORDED_FAILURES {message}", flush=True)
            else:
                raise RuntimeError(message)


def main() -> int:
    global WINDOWS, CORRECTED_RESULTS, STAGE1_RESULT_ROOT
    global STAGE1_PREREQUISITE_ROOT, STAGE2_RESULT_ROOT
    global BASELINE_ALLOCATION_DIR, STAGE1_DURATION_PROBABILITY_SOURCE
    global ACCEPT_INCUMBENT, REPRESENTATIVE_ROOT
    args = parse_args()
    WINDOWS = tuple(dict.fromkeys(args.windows))
    CORRECTED_RESULTS = args.output_root
    STAGE1_RESULT_ROOT = CORRECTED_RESULTS / "stage1"
    STAGE1_PREREQUISITE_ROOT = CORRECTED_RESULTS / "stage1_prerequisites"
    STAGE2_RESULT_ROOT = CORRECTED_RESULTS / "stage2"
    BASELINE_ALLOCATION_DIR = args.baseline_allocation_dir
    REPRESENTATIVE_ROOT = args.representative_root
    configure_probability_root(args.probability_root)
    STAGE1_DURATION_PROBABILITY_SOURCE = (
        args.stage1_duration_probability_source
    )
    ACCEPT_INCUMBENT = bool(args.accept_incumbent)
    configured_mode = str(cfg.get("evaluation.solver.nonanticipativity_mode"))
    if configured_mode != STAGE2_NONANTICIPATIVITY_MODE:
        raise ValueError(
            f"{STAGE2_RUNNER.name} implements the {STAGE2_NONANTICIPATIVITY_MODE} "
            f"structure; evaluation.solver.nonanticipativity_mode is {configured_mode!r}"
        )
    if STAGE1_DURATION_PROBABILITY_SOURCE != str(cfg.get("probability.duration_source")):
        raise ValueError(
            "this isolated rigorous run requires the profile's "
            "Stage-1 prior"
        )
    if args.stage1_parameter_dir is not None and not args.dry_run:
        missing_parameter_files = [
            args.stage1_parameter_dir / f"Stage1_H{window:02d}.prm"
            for window in WINDOWS
            if not (args.stage1_parameter_dir /
                    f"Stage1_H{window:02d}.prm").is_file()
        ]
        if missing_parameter_files:
            raise FileNotFoundError(
                "missing tuned Stage 1 parameter files: "
                + ", ".join(map(str, missing_parameter_files))
            )
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    case_ids = selected_case_ids(args.only)
    stage1_workers = (
        args.stage1_workers
        if args.stage1_workers is not None else args.workers
    )
    stage1_threads = (
        args.stage1_threads
        if args.stage1_threads is not None else args.threads
    )
    stage1_time_limit_seconds = (
        args.stage1_time_limit_seconds
        if args.stage1_time_limit_seconds is not None
        else args.time_limit_seconds
    )

    if args.wait_for_pid is not None and not args.dry_run:
        wait_for_process(args.wait_for_pid)

    if not args.skip_prepare:
        prepare_command = [str(PYTHON), "-X", "utf8", "-u", str(PREPARE_INPUTS)]
        return_code = run_process(prepare_command, dry_run=args.dry_run)
        if return_code != 0:
            return return_code
    if args.prepare_only:
        return 0
    if args.dry_run and not (INPUTS / "input_manifest.json").is_file():
        print("DRY-RUN note: inputs will be created by prepare_inputs.py", flush=True)

    standalone_k_values = {
        STAGE1_CASES[case_id]
        for case_id in case_ids
        if case_id in STAGE1_CASES
    }
    stage2_k_values = {
        int(STAGE2_CASES[case_id].allocation_k)
        for case_id in case_ids
        if case_id in STAGE2_CASES
        and STAGE2_CASES[case_id].allocation_source == "k"
        and STAGE2_CASES[case_id].allocation_k is not None
    }
    allocation_dirs: dict[str, Path] = {}
    if BASE_K in standalone_k_values or BASE_K in stage2_k_values:
        allocation_dirs[f"K{BASE_K}"] = BASELINE_ALLOCATION_DIR
    standalone_k_values.discard(BASE_K)
    stage2_k_values.discard(BASE_K)

    def solve_required_k(k: int, within_case_workers: int) -> tuple[int, Path]:
        case_id = f"S1_K{k}"
        representative_candidates = [
            REPRESENTATIVE_ROOT / representative_branch_name(k, window)
            for window in STAGE1_K_WINDOWS
        ]
        representatives = next(
            (path for path in representative_candidates if path.is_file()),
            representative_candidates[0],
        )
        stage1_output = STAGE1_RESULT_ROOT / case_id
        run_stage1(
            label=case_id,
            output=stage1_output,
            k=k,
            fleet_size=BASE_FLEET_SIZE,
            representatives=representatives,
            threads=stage1_threads,
            time_limit_seconds=stage1_time_limit_seconds,
            mip_gap=args.mip_gap,
            dry_run=args.dry_run,
            workers=within_case_workers,
            parameter_dir=args.stage1_parameter_dir,
            auto_parameters=args.stage1_auto_parameters,
            windows=STAGE1_K_WINDOWS,
        )
        return k, stage1_output / "Allocation"

    def solve_k_values(k_values: list[int]) -> None:
        if len(k_values) > 1 and stage1_workers > 1:
            with ThreadPoolExecutor(
                max_workers=min(stage1_workers, len(k_values))
            ) as executor:
                futures = [
                    executor.submit(solve_required_k, k, 1)
                    for k in k_values
                ]
                for future in as_completed(futures):
                    k, allocation_dir = future.result()
                    allocation_dirs[f"K{k}"] = allocation_dir
        else:
            for k in k_values:
                k, allocation_dir = solve_required_k(k, stage1_workers)
                allocation_dirs[f"K{k}"] = allocation_dir

    # Only allocations consumed by a Stage-2 variant are prerequisites.
    # Standalone K sensitivity is deliberately deferred until all Stage-2
    # cases finish.
    solve_k_values(sorted(stage2_k_values))

    selected_stage2 = [
        STAGE2_CASES[case_id]
        for case_id in case_ids
        if case_id in STAGE2_CASES
    ]
    for fleet_size in sorted(
        {case.fleet_size for case in selected_stage2 if case.allocation_source == "fleet"}
    ):
        if args.use_existing_fleet_allocations:
            allocation_dir = (
                args.existing_fleet_allocation_root
                / f"N{fleet_size:02d}" / "Allocation"
            )
            for window in WINDOWS:
                allocation = allocation_dir / f"DepotAlloc_{window:02d}.csv"
                if not allocation.is_file():
                    raise FileNotFoundError(allocation)
                if allocation_total(allocation) != fleet_size:
                    raise ValueError(
                        f"N{fleet_size:02d} H{window:02d}: invalid saved fleet total"
                    )
            allocation_dirs[f"N{fleet_size}"] = allocation_dir
        else:
            allocation_dirs[f"N{fleet_size}"] = ensure_fleet_allocations(
                fleet_size,
                threads=stage1_threads,
                time_limit_seconds=stage1_time_limit_seconds,
                mip_gap=args.mip_gap,
                dry_run=args.dry_run,
                workers=stage1_workers,
                parameter_dir=args.stage1_parameter_dir,
                auto_parameters=args.stage1_auto_parameters,
            )

    if args.prerequisites_only:
        solve_k_values(sorted(standalone_k_values - stage2_k_values))
        print("Completed requested Stage 1 allocation prerequisites", flush=True)
        return 0

    jobs = discover_stage2_jobs() if selected_stage2 and not (
        args.dry_run and not SCENARIO_STATS.is_file()
    ) else []
    if args.dry_run and selected_stage2 and not jobs:
        print(
            "DRY-RUN: Stage2 jobs will be discovered for "
            + "/".join(f"H{window:02d}" for window in WINDOWS),
            flush=True,
        )
    if selected_stage2 and jobs:
        validate_stage2_probability_inputs(
            selected_stage2,
            jobs,
            require_lightgbm=bool(args.require_lightgbm_probabilities),
        )
    for case in selected_stage2:
        run_stage2_case(
            case,
            jobs=jobs,
            allocation_dirs=allocation_dirs,
            threads=args.threads,
            time_limit_seconds=args.time_limit_seconds,
            mip_gap=args.mip_gap,
            mip_focus=args.mip_focus,
            heuristics=args.heuristics,
            cuts=args.cuts,
            dry_run=args.dry_run,
            workers=args.workers,
        )
    solve_k_values(sorted(standalone_k_values - stage2_k_values))
    print(f"Completed requested cases: {case_ids}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
