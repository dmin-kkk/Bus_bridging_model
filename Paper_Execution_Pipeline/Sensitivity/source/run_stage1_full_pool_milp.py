"""Run one sensitivity Stage-1 window through the production Stage-1 runner."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve()
PIPELINE = SCRIPT.parents[2]
sys.path.insert(0, str(PIPELINE.parent))
from bbconfig import cfg  # noqa: E402

BASE_RUNNER = PIPELINE / "04_run_stage1.py"
PRODUCTION_SCHEMA = (
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
SENSITIVITY_SCHEMA = (
    "production-core-stage1-sensitivity-v3-"
    "bus-only-wait-and-see-operational303"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument(
        "--window", type=int, required=True,
        choices=[int(value) for value in cfg.get("time.windows")],
    )
    parser.add_argument("--route-file", type=Path, required=True)
    parser.add_argument("--representative-branches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--fleet-size", type=int, required=True)
    parser.add_argument(
        "--buffer", type=int, default=int(cfg.get("time.clearance_buffer_min")))
    parser.add_argument(
        "--min-trip-boarding", type=float,
        default=float(cfg.get("operations.min_trip_boarding")))
    parser.add_argument(
        "--time-limit-seconds", type=float,
        default=float(cfg.get("evaluation.solver.stage1_time_limit_s")))
    parser.add_argument(
        "--mip-gap", type=float, default=float(cfg.get("evaluation.solver.mip_gap")))
    parser.add_argument(
        "--threads", type=int, default=int(cfg.get("parallel.scaling.threads")))
    parser.add_argument("--tune-parameters", action="store_true")
    # Gurobi tuning is not used by any paper run; its limits must be given
    # explicitly together with --tune-parameters.
    parser.add_argument("--tune-time-limit", type=float)
    parser.add_argument("--tune-trials", type=int)
    parser.add_argument("--tune-candidate-time-limit", type=float)
    parser.add_argument("--tune-only", action="store_true")
    parser.add_argument("--parameter-file", type=Path)
    parser.add_argument("--tune-output", type=Path)
    parser.add_argument("--root-escape", action="store_true")
    parser.add_argument("--bound-focus", action="store_true")
    parser.add_argument("--warm-start-dir", type=Path, action="append")
    parser.add_argument("--warm-start-depot-only", action="store_true")
    parser.add_argument("--warm-start-hint-only", action="store_true")
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--auto-parameters", action="store_true")
    parser.add_argument("--heuristics", type=float)
    parser.add_argument(
        "--duration-probability-source",
        choices=("operational303",),
        default=str(cfg.get("probability.duration_source")),
    )
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")
    if args.fleet_size <= 0:
        parser.error("--fleet-size must be positive")
    if args.min_trip_boarding <= 0:
        parser.error("--min-trip-boarding must be positive")
    if args.tune_parameters and None in (
            args.tune_time_limit, args.tune_trials,
            args.tune_candidate_time_limit):
        parser.error(
            "--tune-parameters requires --tune-time-limit, --tune-trials "
            "and --tune-candidate-time-limit")
    if args.tune_time_limit is not None and args.tune_time_limit <= 0:
        parser.error("--tune-time-limit must be positive")
    if args.tune_trials is not None and args.tune_trials < 0:
        parser.error("--tune-trials must be nonnegative")
    if (args.tune_candidate_time_limit is not None
            and args.tune_candidate_time_limit <= 0):
        parser.error("--tune-candidate-time-limit must be positive")
    if args.tune_parameters and args.parameter_file is not None:
        parser.error("--tune-parameters and --parameter-file are exclusive")
    if args.tune_parameters and args.tune_output is None:
        parser.error("--tune-output is required with --tune-parameters")
    if args.tune_only and not args.tune_parameters:
        parser.error("--tune-only requires --tune-parameters")
    if args.heuristics is not None and not 0 <= args.heuristics <= 1:
        parser.error("--heuristics must be in [0, 1]")
    if args.no_warm_start and (
            args.warm_start_dir or args.warm_start_depot_only
            or args.warm_start_hint_only):
        parser.error(
            "--no-warm-start cannot be combined with warm-start options")
    return args


def load_production_runner():
    sys.path.insert(0, str(PIPELINE))
    specification = importlib.util.spec_from_file_location(
        "sensitivity_production_stage1", BASE_RUNNER
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {BASE_RUNNER}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def augment_summary(args: argparse.Namespace) -> None:
    summary_path = (
        args.output.resolve() / "Performance" /
        f"FullMILP_{args.window:02d}_summary.json"
    )
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") != PRODUCTION_SCHEMA:
        raise AssertionError(
            f"production Stage-1 schema drift: {summary.get('schema')}"
        )
    exact = summary.get("exact_reformulation") or {}
    audit = summary.get("passenger_information_structure_audit") or {}
    valid_information_structure = bool(
        summary.get("nonanticipativity_scope") == NONANTICIPATIVITY_SCOPE
        and summary.get("bus_decision_nonanticipativity")
        == BUS_DECISION_NONANTICIPATIVITY
        and summary.get("passenger_recourse_mode")
        == PASSENGER_RECOURSE_MODE
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
        and audit.get("strict_passenger_na_preprocessor_invoked") is False
        and audit.get("passenger_variable_sharing") is False
        and audit.get("passenger_sharing_metadata_absent") is True
        and audit.get("duration_specific_omega_theta_preserved") is True
    )
    if not valid_information_structure:
        raise AssertionError(
            "production Stage-1 passenger information structure drift"
        )
    summary["sensitivity"] = {
        "schema": SENSITIVITY_SCHEMA,
        "case_id": args.case_id,
        "production_runner": str(BASE_RUNNER.resolve()),
        "production_stage1_schema": PRODUCTION_SCHEMA,
        "route_file": str(args.route_file.resolve()),
        "representative_branches": str(
            args.representative_branches.resolve()
        ),
        "k": args.k,
        "fleet_size": args.fleet_size,
        "disrupt_buffer_minutes": args.buffer,
        "min_trip_boarding": args.min_trip_boarding,
        "trip_service_unit": "ONE_WAY",
        "dispatch_interval_minutes": int(cfg.get("time.dispatch_interval_min")),
        "warm_start_depot_only": bool(args.warm_start_depot_only),
        "warm_start_hint_only": bool(args.warm_start_hint_only),
        "warm_start_enabled": not bool(args.no_warm_start),
        "duration_probability_source": args.duration_probability_source,
        "duration_weights": base_duration_weights(args),
        "nonanticipativity_scope": NONANTICIPATIVITY_SCOPE,
        "bus_decision_nonanticipativity": dict(
            BUS_DECISION_NONANTICIPATIVITY),
        "passenger_recourse_mode": PASSENGER_RECOURSE_MODE,
        "passenger_variable_sharing": False,
        "directed_route_timing": "raw-directed-cumulative-rounding-validated",
    }
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)


def base_duration_weights(args: argparse.Namespace) -> dict[str, float]:
    # Table 3 (pi_30, pi_60): cohort counts of probability.duration_counts.
    counts = {str(state): int(count) for state, count
              in cfg.get("probability.duration_counts").items()}
    total = sum(counts.values())
    return {state: count / total for state, count in counts.items()}


def main() -> int:
    args = parse_args()
    for path in (
        BASE_RUNNER,
        args.route_file,
        args.representative_branches,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    base = load_production_runner()
    base.FLEET_SIZE = int(args.fleet_size)
    base.DISPATCH_BUFFER_MINUTES = int(args.buffer)
    base.MIN_TRIP_BOARDING = float(args.min_trip_boarding)
    base.WARM_START_DEPOT_ONLY = bool(args.warm_start_depot_only)
    base.WARM_START_ENABLED = not bool(args.no_warm_start)
    base.WARM_START_HINT_ONLY = bool(args.warm_start_hint_only)
    base.REPRESENTATIVES_PER_DURATION = int(args.k)
    base.ROUTE_FILE = args.route_file.resolve()
    base.BRANCH_FILE = args.representative_branches.resolve()
    base.OUTPUT_DIR = args.output.resolve()
    if args.no_warm_start:
        base.WARM_START_DIRS = ()
    elif args.warm_start_dir:
        base.WARM_START_DIRS = tuple(
            [args.output.resolve()]
            + [path.resolve() for path in args.warm_start_dir]
        )
    elif args.fleet_size == int(cfg.get("operations.fleet_size")):
        base.WARM_START_DIRS = (
            args.output.resolve(),
            (cfg.results_dir / cfg.get("paths.stage1")).resolve(),
        )
    else:
        base.WARM_START_DIRS = (args.output.resolve(),)

    solve_args = argparse.Namespace(
        window=args.window,
        route_file=args.route_file.resolve(),
        representative_branches=args.representative_branches.resolve(),
        output=args.output.resolve(),
        duration_probability_source=args.duration_probability_source,
        duration_weights=base.duration_weights(
            args.duration_probability_source),
        k=args.k,
        buffer=args.buffer,
        time_limit_seconds=args.time_limit_seconds,
        mip_gap=args.mip_gap,
        threads=args.threads,
        bound_focus=bool(args.bound_focus),
        root_escape=bool(args.root_escape),
        concurrent_retry=False,
        incumbent_target=None,
        auto_parameters=bool(args.auto_parameters),
        heuristics=args.heuristics,
        tune_parameters=bool(args.tune_parameters),
        tune_time_limit=args.tune_time_limit,
        tune_trials=args.tune_trials,
        tune_candidate_time_limit=args.tune_candidate_time_limit,
        tune_only=bool(args.tune_only),
        parameter_file=(
            args.parameter_file.resolve()
            if args.parameter_file is not None else None),
        tune_output_path=(
            args.tune_output.resolve()
            if args.tune_output is not None else None),
    )
    return_code = int(base.solve_window(solve_args))
    augment_summary(args)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
