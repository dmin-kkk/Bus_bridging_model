"""Run one Stage-2 sensitivity instance (window, station) through the production Stage-2 runner.

Defaults come from the active profile; ``run_sensitivity.py`` passes every case
parameter explicitly.  ``--window`` and ``--station`` identify the instance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve()
PIPELINE = SCRIPT.parents[2]
PACKAGE = SCRIPT.parents[3]
sys.path.insert(0, str(PACKAGE))

from bbconfig import cfg  # noqa: E402


BASE_RUNNER = PIPELINE / "06_stage2_core.py"
BASELINE_ROUTE_FILE = (
    cfg.results_dir / cfg.get("paths.routes") / str(cfg.get("routes.library_file"))
)
DURATIONS = tuple(int(d) for d in cfg.get("time.durations_min"))
PROB_EPS = 1e-10    # tolerance of the duration-probability sum
BASELINE_CASE = str(cfg.get("evaluation.sensitivity.baseline_case"))
BASELINE_INCIDENT_TYPE = str(cfg.get("evaluation.sensitivity.baseline_incident_type"))
PROBABILITY_MODEL = str(cfg.get("evaluation.oof.model"))
PRODUCTION_SCHEMA = (
    "paper-stage2-full-pool-joint-30-60-passenger-bus-na-raw-rail-v5"
)
SENSITIVITY_SCHEMA = (
    "production-core-stage2-sensitivity-v7-passenger-na-raw-rail"
)
NONANTICIPATIVITY_MODE = "passenger-and-bus"
FORMULATION_MODE = "PASSENGER_AND_BUS_SHARED_RAW_DURATION_RAIL"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    solver = "evaluation.solver"
    slugs = {
        str(label): str(slug)
        for slug, label in cfg.get("prediction.full_grid.incident_types")
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default=BASELINE_CASE)
    parser.add_argument("--window", type=int, required=True,
                        choices=[int(w) for w in cfg.get("time.windows")])
    parser.add_argument("--station", required=True)
    parser.add_argument(
        "--allocation", type=Path,
        help="Stage-1 DepotAlloc file (default: <results>/paths.stage1/Allocation/DepotAlloc_<window>.csv)")
    parser.add_argument("--fleet-size", type=int, default=int(cfg.get("operations.fleet_size")))
    parser.add_argument("--route-file", type=Path, default=BASELINE_ROUTE_FILE)
    parser.add_argument("--distance-file", type=Path, default=cfg.path("inputs.road_times"))
    parser.add_argument("--depot-file", type=Path, default=cfg.path("inputs.depots"))
    parser.add_argument(
        "--output", type=Path,
        help="case output folder (default: <results>/paths.sensitivity/stage2/<case-id>)")
    parser.add_argument(
        "--probability-file", type=Path,
        help="fixed-type probability table (default: ML stage file of --fixed-incident-type)")
    parser.add_argument("--fixed-incident-type", default=BASELINE_INCIDENT_TYPE,
                        choices=sorted(slugs))
    parser.add_argument("--min-trip-boarding", type=float,
                        default=float(cfg.get("operations.min_trip_boarding")))
    parser.add_argument("--queue-multiplier", type=float, default=1.0)
    parser.add_argument("--walking-multiplier", type=float, default=1.0)
    parser.add_argument("--travel-time-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--nested-route-pool",
        choices=("none", "lower", "upper"),
        default="none",
    )
    parser.add_argument("--buffer", type=int, default=int(cfg.get("time.clearance_buffer_min")))
    parser.add_argument("--time-limit-seconds", type=float,
                        default=float(cfg.get(f"{solver}.stage2_time_limit_s")))
    parser.add_argument("--mip-gap", type=float, default=float(cfg.get(f"{solver}.mip_gap")))
    parser.add_argument("--threads", type=int, default=int(cfg.get("parallel.sensitivity.threads")))
    parser.add_argument("--mip-focus", type=int, default=int(cfg.get(f"{solver}.mip_focus")))
    parser.add_argument("--heuristics", type=float, default=float(cfg.get(f"{solver}.heuristics")))
    parser.add_argument("--cuts", type=int, default=int(cfg.get(f"{solver}.cuts")))
    parser.add_argument("--mip-start-policy", type=Path)
    args = parser.parse_args()
    if args.allocation is None:
        args.allocation = (
            cfg.results_dir / cfg.get("paths.stage1") / "Allocation"
            / f"DepotAlloc_{args.window:02d}.csv")
    if args.output is None:
        args.output = (
            cfg.results_dir / cfg.get("paths.sensitivity") / "stage2" / args.case_id)
    if args.probability_file is None:
        args.probability_file = (
            cfg.results_dir / cfg.get("paths.probabilities")
            / str(cfg.get("prediction.outputs.probability_file")).format(
                slug=slugs[args.fixed_incident_type]))
    if args.fleet_size <= 0:
        parser.error("--fleet-size must be positive")
    if args.min_trip_boarding < 0:
        parser.error("--min-trip-boarding cannot be negative")
    for name in ("queue_multiplier", "walking_multiplier", "travel_time_multiplier"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.mip_focus not in (0, 1, 2, 3):
        parser.error("--mip-focus must be one of 0, 1, 2, 3")
    if not 0 <= args.heuristics <= 1:
        parser.error("--heuristics must be in [0, 1]")
    if args.cuts not in (-1, 0, 1, 2, 3):
        parser.error("--cuts must be one of -1, 0, 1, 2, 3")
    return args


def load_production_runner():
    sys.path.insert(0, str(PIPELINE))
    specification = importlib.util.spec_from_file_location(
        "sensitivity_production_stage2", BASE_RUNNER
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {BASE_RUNNER}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def write_augmented_summary(
    args: argparse.Namespace,
    route_audit: list[dict[str, object]],
    duration_weights: dict[int, float],
    probability_method: str,
    probability_metadata: dict[str, object],
) -> None:
    station = args.station.strip().upper()
    summary_path = (
        args.output.resolve() / "Performance" /
        f"FullMILP_{args.window:02d}_{station}_summary.json"
    )
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") != PRODUCTION_SCHEMA:
        raise AssertionError(
            f"production Stage-2 schema drift: {summary.get('schema')}"
        )
    summary["sensitivity"] = {
        "schema": SENSITIVITY_SCHEMA,
        "case_id": args.case_id,
        "window": args.window,
        "production_runner": str(BASE_RUNNER.resolve()),
        "production_stage2_schema": PRODUCTION_SCHEMA,
        "formulation_mode": FORMULATION_MODE,
        "nonanticipativity_mode": NONANTICIPATIVITY_MODE,
        "rail_state_policy": "PRESERVE_EACH_DURATION_INPUT_BRANCH",
        "allocation_rule": "matching time-specific Stage1 allocation",
        "allocation_file": str(args.allocation.resolve()),
        "allocation_sha256": sha256(args.allocation.resolve()),
        "fleet_size": args.fleet_size,
        "min_trip_boarding": args.min_trip_boarding,
        "travel_time_multiplier": args.travel_time_multiplier,
        "queue_multiplier": args.queue_multiplier,
        "walking_multiplier": args.walking_multiplier,
        "duration_probability_method": probability_method,
        "duration_probability_file": str(args.probability_file.resolve()),
        "duration_probability_sha256": sha256(args.probability_file.resolve()),
        "probability_schema": probability_metadata["probability_schema"],
        "probability_model_family": probability_metadata["model_family"],
        "probability_training_cohort": probability_metadata["training_cohort"],
        "probability_operating_date": probability_metadata["operating_date"],
        "probability_ensemble_members": probability_metadata["ensemble_members"],
        "fixed_incident_type": args.fixed_incident_type,
        "duration_weights": {
            str(duration): probability
            for duration, probability in duration_weights.items()
        },
        "exact_state_compression": True,
        "route_file": str(args.route_file.resolve()),
        "distance_file": str(args.distance_file.resolve()),
        "directed_route_timing": "raw-directed-cumulative-rounding-validated",
        "nested_route_pool": args.nested_route_pool,
        "route_nesting_audit": route_audit,
        "solver_profile": {
            "mip_focus": args.mip_focus,
            "heuristics": args.heuristics,
            "cuts": args.cuts,
            "mip_start_policy": (
                str(args.mip_start_policy.resolve())
                if args.mip_start_policy is not None else None
            ),
        },
    }
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)


def main() -> int:
    args = parse_args()
    for path in (
        args.allocation,
        args.route_file,
        args.distance_file,
        args.depot_file,
        args.probability_file,
        BASE_RUNNER,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.nested_route_pool != "none" and not BASELINE_ROUTE_FILE.is_file():
        raise FileNotFoundError(BASELINE_ROUTE_FILE)

    base = load_production_runner()
    probabilities = pd.read_csv(args.probability_file)
    probabilities["window"] = pd.to_numeric(
        probabilities["window"], errors="raise").astype(int)
    probabilities["station"] = (
        probabilities["station"].astype(str).str.strip().str.upper()
    )
    selected_probability = probabilities[
        probabilities["window"].eq(args.window)
        & probabilities["station"].eq(args.station.strip().upper())
    ]
    if len(selected_probability) != 1:
        raise ValueError(
            f"expected one probability row for H{args.window:02d}/"
            f"{args.station}, found {len(selected_probability)}"
        )
    probability_row = selected_probability.iloc[0]
    probability_method = str(probability_row.get(
        "probability_method", "")).strip()
    if PROBABILITY_MODEL.lower() not in probability_method.lower():
        raise ValueError(
            f"the final sensitivity run requires {PROBABILITY_MODEL} probabilities; "
            f"found {probability_method!r}"
        )
    if str(probability_row["fixed_incident_type"]) != args.fixed_incident_type:
        raise ValueError("fixed incident type and probability file disagree")
    probability_metadata = {
        "probability_schema": str(probability_row["probability_schema"]),
        "model_family": str(probability_row["model_family"]),
        "training_cohort": str(probability_row["training_cohort"]),
        "operating_date": str(probability_row["operating_date"]),
        "ensemble_members": int(probability_row["ensemble_members"]),
    }
    duration_weights = {
        duration: float(probability_row[f"p{duration}"]) for duration in DURATIONS
    }
    if abs(sum(duration_weights.values()) - 1.0) > PROB_EPS:
        raise ValueError("duration probabilities do not sum to one")
    original_load_module = base.load_module
    route_audit: list[dict[str, object]] = []

    def patched_load_module(name: str, path: Path):
        module = original_load_module(name, path)
        resolved = path.resolve()
        if resolved == base.ADAPTIVE_FILE.resolve():
            module.EXACT_STATE_COMPRESSION = True
            original_add_flow_block = module._add_flow_block

            def weighted_add_flow_block(*flow_args, **flow_kwargs):
                return module.reweight_flow_block(
                    original_add_flow_block(*flow_args, **flow_kwargs),
                    flow_args[4],
                    queue_multiplier=args.queue_multiplier,
                    walking_multiplier=args.walking_multiplier,
                )

            module._add_flow_block = weighted_add_flow_block
        elif (
            resolved == base.MILP_CORE_FILE.resolve()
            and args.nested_route_pool != "none"
        ):
            original_select_routes = module.select_routes
            baseline_library = None

            def nested_select_routes(scenario, depots, route_library, bus_tt):
                nonlocal baseline_library
                if baseline_library is None:
                    baseline_library = module.load_route_library(
                        BASELINE_ROUTE_FILE,
                        distance_path=args.distance_file.resolve(),
                        validate_forward_times=True,
                    )
                baseline_selected = original_select_routes(
                    scenario, depots, baseline_library, bus_tt
                )
                current_selected = original_select_routes(
                    scenario, depots, route_library, bus_tt
                )
                baseline_keys = set(baseline_selected)
                if args.nested_route_pool == "lower":
                    returned = {
                        key: baseline_selected[key]
                        for key in sorted(baseline_keys & set(route_library))
                    }
                    valid = set(returned).issubset(baseline_keys)
                else:
                    returned = dict(baseline_selected)
                    returned.update(current_selected)
                    valid = baseline_keys.issubset(returned)
                audit = {
                    "mode": args.nested_route_pool,
                    "baseline_selected_routes": len(baseline_keys),
                    "independent_current_selected_routes": len(current_selected),
                    "returned_selected_routes": len(returned),
                    "nesting_valid": valid,
                }
                route_audit.append(audit)
                if not valid:
                    raise AssertionError(f"route-pool nesting failed: {audit}")
                return returned

            module.select_routes = nested_select_routes
        return module

    base.load_module = patched_load_module
    base.FLEET_TOTAL = int(args.fleet_size)
    base.MIN_TRIP_BOARDING = float(args.min_trip_boarding)
    base.ROUTE_FILE = args.route_file.resolve()
    base.DISTANCE_FILE = args.distance_file.resolve()
    base.DEPOT_FILE = args.depot_file.resolve()
    base.duration_prior = lambda _source: (
        {duration: 0 for duration in DURATIONS}, dict(duration_weights)
    )
    base.parse_args = lambda: argparse.Namespace(
        single_case=True,
        check_only=False,
        nonanticipativity_mode=NONANTICIPATIVITY_MODE,
        exact_state_compression=True,
        exact_trip_cell_pruning=False,
        local_routes_only=False,
        window=args.window,
        station=args.station,
        oracle_duration=None,
        duration_p60=None,
        start_window=args.window,
        end_window=args.window,
        buffer=args.buffer,
        time_limit_seconds=args.time_limit_seconds,
        mip_gap=args.mip_gap,
        threads=args.threads,
        mip_focus=args.mip_focus,
        heuristics=args.heuristics,
        cuts=args.cuts,
        mip_start_policy=(
            args.mip_start_policy.resolve()
            if args.mip_start_policy is not None else None
        ),
        workers=1,
        allocation_dir=args.allocation.resolve().parent,
        output_dir=args.output.resolve(),
        case_label=args.case_id,
        duration_probability_source="sensitivity_ml",
    )
    return_code = int(base.main())
    write_augmented_summary(
        args,
        route_audit,
        duration_weights,
        probability_method,
        probability_metadata,
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
