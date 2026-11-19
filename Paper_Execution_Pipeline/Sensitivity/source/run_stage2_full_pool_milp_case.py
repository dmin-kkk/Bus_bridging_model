"""Solve one fixed-depot Stage-2 station case as a direct joint MILP (legacy check).

There is no column generation, pricing, or restricted master.  All candidate
trips produced from the finite route library (``routes.library_file``) are
present in one MILP.  The duration branches share dispatch decisions until their
duration is observed, and use the Table 3 duration prior
(``probability.duration_counts``).  Not used by any reported result.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import gurobipy as gp
import pandas as pd


SCRIPT = Path(__file__).resolve()
PIPELINE = SCRIPT.parents[2]
PACKAGE = SCRIPT.parents[3]
sys.path.insert(0, str(PACKAGE))

from bbconfig import cfg  # noqa: E402


STAGE2_SOURCE = PIPELINE / "_core" / "stage2"
CORE_FILE = STAGE2_SOURCE / "milp_core.py"
SCHEMA = "stage2-full-finite-pool-joint-30-60-causal-chain-v2"
EPS = 1e-9  # integrality / gap tolerance
DURATIONS = tuple(int(d) for d in cfg.get("time.durations_min"))
LONG_DURATION = max(DURATIONS)
PRIOR_COUNTS = {   # Table 3: pi_30, pi_60
    int(key): int(value)
    for key, value in cfg.get("probability.duration_counts").items()
}
DURATION_WEIGHTS = {
    duration: float(PRIOR_COUNTS[duration]) / float(sum(PRIOR_COUNTS.values()))
    for duration in DURATIONS
}
FLEET_SIZE = int(cfg.get("operations.fleet_size"))


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


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(safe(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def load_depots(path: Path) -> tuple[str, ...]:
    frame = pd.read_csv(path)
    if "Code" not in frame.columns:
        raise ValueError(f"{path}: missing Code column")
    values = tuple(sorted({str(value).strip().upper() for value in frame["Code"]
                           if str(value).strip()}))
    if not values:
        raise ValueError(f"{path}: no depot candidates")
    return values


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
        if depot in seen or depot not in counts or count < 0 or abs(raw - count) > EPS:
            raise ValueError(f"{path}: invalid allocation row for {depot}")
        seen.add(depot)
        counts[depot] = count
    if sum(counts.values()) != FLEET_SIZE:
        raise ValueError(
            f"{path}: fleet total is {sum(counts.values())}, expected {FLEET_SIZE}")
    return counts


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


def parse_args():
    solver = "evaluation.solver"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, required=True,
                        choices=[int(w) for w in cfg.get("time.windows")])
    parser.add_argument("--station", required=True)
    parser.add_argument(
        "--allocation", type=Path,
        help="Stage-1 DepotAlloc file (default: <results>/paths.stage1/Allocation/DepotAlloc_<window>.csv)")
    parser.add_argument(
        "--route-file", type=Path,
        default=cfg.results_dir / cfg.get("paths.routes") / str(cfg.get("routes.library_file")))
    parser.add_argument("--depot-file", type=Path, default=cfg.path("inputs.depots"))
    parser.add_argument(
        "--output", type=Path,
        default=cfg.results_dir / cfg.get("paths.sensitivity") / "stage2_full_pool_check")
    parser.add_argument("--buffer", type=int, default=int(cfg.get("time.clearance_buffer_min")))
    parser.add_argument("--time-limit-seconds", type=float,
                        default=float(cfg.get(f"{solver}.stage2_time_limit_s")))
    parser.add_argument("--mip-gap", type=float, default=float(cfg.get(f"{solver}.mip_gap")))
    parser.add_argument("--threads", type=int, default=int(cfg.get("parallel.sensitivity.threads")))
    args = parser.parse_args()
    if args.allocation is None:
        args.allocation = (
            cfg.results_dir / cfg.get("paths.stage1") / "Allocation"
            / f"DepotAlloc_{args.window:02d}.csv")
    return args


def main() -> int:
    args = parse_args()
    station = args.station.strip().upper()
    prefix = f"FullMILP_{args.window:02d}_{station}"
    output = args.output.resolve()
    summary_path = output / "Performance" / f"{prefix}_summary.json"
    progress_path = output / "Performance" / f"{prefix}_progress.json"
    policy_path = output / "Policies" / f"{prefix}_policy.csv"
    chain_policy_path = output / "Policies" / f"{prefix}_chain_policy.csv"
    started = time.time()
    environment = None

    def checkpoint(phase: str, **extra) -> None:
        write_json(progress_path, {
            "schema": SCHEMA,
            "status": "RUNNING",
            "phase": phase,
            "solver": "FULL_POOL_MILP",
            "scope": "FULL_FINITE_ROUTE_AND_TRIP_POOL",
            "column_generation": False,
            "window": args.window,
            "station": station,
            "durations": list(DURATIONS),
            "duration_weights": DURATION_WEIGHTS,
            "elapsed_seconds": time.time() - started,
            **extra,
        })

    try:
        checkpoint("LOADING_INPUTS")
        os.chdir(PIPELINE)
        sys.path.insert(0, str(STAGE2_SOURCE))
        os.environ["PAPER_PIPELINE_INPUT_DIR"] = str(cfg.data_dir)
        os.environ["PAPER_PIPELINE_RESULTS_DIR"] = str(cfg.results_dir)
        # Match the exact Stage-1 candidate-feasibility rules.
        os.environ["TSB_DEPOT_ACCESS_BOUND"] = str(int(cfg.get("operations.depot_access_bound_min")))
        os.environ["TSB_MAX_BUSES"] = str(FLEET_SIZE)
        os.environ["TSB_ROUTE_SOURCE"] = "predefined"
        os.environ["TSB_BUS_BOARD"] = "general"
        stage2 = load_module("stage2_adaptive_direct", STAGE2_SOURCE / "stage2_adaptive.py")
        # The core imports stage2_adaptive by its canonical name.
        sys.modules["stage2_adaptive"] = stage2
        core = load_module("stage2_direct_core", CORE_FILE)
        core.DURATIONS = list(DURATIONS)
        core.DURATION_WEIGHT = dict(DURATION_WEIGHTS)
        core.DISRUPT_BUFFER = int(args.buffer)
        core.TIME_LIMIT = float(args.time_limit_seconds)
        core.MIP_GAP = float(args.mip_gap)
        core.FILE_PREDEFINED_ROUTES = str(args.route_file.resolve())
        core.FILE_SCHEDULE = str(cfg.path("inputs.timetable"))
        core.FILE_OD = str(cfg.path("inputs.od_matrix"))
        core.FILE_DISTANCE = str(cfg.path("inputs.road_times"))
        core.DIR_NORMAL = str(cfg.results_dir / cfg.get("paths.normal"))
        core.DIR_DISRUPT = str(cfg.results_dir / cfg.get("paths.disruptions"))

        depots = load_depots(args.depot_file.resolve())
        depot_counts = load_allocation(args.allocation.resolve(), depots)
        schedule = core.load_schedule(core.FILE_SCHEDULE)
        od = core.load_od(core.FILE_OD)
        bus_tt = core.load_distance_matrix(core.FILE_DISTANCE)
        route_library = core.load_route_library(core.FILE_PREDEFINED_ROUTES)
        window_data = core.build_window_data(args.window, schedule, od)
        raw_scenarios = core.load_disruption_scenarios(args.window, window_data)
        available = sorted({int(item["duration"]) for item in raw_scenarios
                            if item["station"] == station})
        if LONG_DURATION not in available:
            raise ValueError(
                f"{args.window:02d}/{station}: no real D{LONG_DURATION} scenario")

        group = stage2.build_adaptive_group(
            core, station, args.window, window_data, raw_scenarios, depots,
            bus_tt, route_library, durations=DURATIONS, buffer=args.buffer,
        )
        active = tuple(map(int, group["active"]))
        if any(duration not in DURATIONS for duration in active) or LONG_DURATION not in active:
            raise AssertionError(f"unexpected active durations: {active}")
        checkpoint(
            "FULL_POOL_READY",
            active_raw_durations=list(active),
            zero_impact_durations=[d for d in DURATIONS if d not in active],
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
        result = stage2.solve_stage2_adaptive(
            core, environment, depots, depot_counts, group, window_data,
            duration_weight=DURATION_WEIGHTS,
            mip_gap=args.mip_gap,
            time_limit=args.time_limit_seconds,
            output_flag=1,
            threads=args.threads,
        )
        stats = result.get("stats", {})
        has_incumbent = int(stats.get("SolCount", 0) or 0) > 0
        gap = stats.get("MIPGap")
        certified = bool(
            has_incumbent and gap is not None and math.isfinite(float(gap))
            and float(gap) <= args.mip_gap + EPS
        )
        rows = policy_rows(result) if has_incumbent else []
        decision_files = {}
        decision_row_counts = {}
        if has_incumbent:
            write_policy(policy_path, rows)
            write_chain_policy(
                chain_policy_path, result.get("chain_decisions", []))
            for duration in DURATIONS:
                decision_path = output / "DecisionVars" / (
                    f"Disrupt_{args.window:02d}_{station}_{duration}.csv"
                )
                decision_rows = result.get("dv_rows", {}).get(duration, [])
                write_decision_variables(decision_path, decision_rows)
                decision_files[str(duration)] = str(decision_path.resolve())
                decision_row_counts[str(duration)] = len(decision_rows)

        summary = {
            "schema": SCHEMA,
            "status": result.get("status"),
            "solver": "FULL_POOL_MILP",
            "scope": "FULL_FINITE_ROUTE_AND_TRIP_POOL",
            "column_generation": False,
            "cg_iterations": 0,
            "window": args.window,
            "station": station,
            "durations": list(DURATIONS),
            "active_raw_durations": list(active),
            "zero_impact_durations": [d for d in DURATIONS if d not in active],
            "duration_weights": DURATION_WEIGHTS,
            "historical_counts": {
                **{f"D{duration}": LEGACY_COUNTS[duration] for duration in DURATIONS},
                "total": sum(LEGACY_COUNTS.values()),
            },
            "fixed_stage1_allocation": str(args.allocation.resolve()),
            "fleet_total": sum(depot_counts.values()),
            "buffer_minutes": args.buffer,
            "candidate_routes": len({trip["Route"] for trip in group["trips"].values()}),
            "candidate_trips": len(group["trips"]),
            "information_cell_variables": sum(len(cells) for cells in group["cells"].values()),
            "raw_trip_cell_count": group.get("raw_trip_cell_count"),
            "minimum_service_infeasible_trip_cells_removed": group.get(
                "minimum_service_infeasible_trip_cells_removed"),
            "vehicle_chain_information_set_nonanticipative": bool(
                stats.get("VehicleChainInformationSetNonanticipative")),
            "vehicle_chain_decision_scope": stats.get(
                "VehicleChainDecisionScope"),
            "objective": result.get("objective") if has_incumbent else None,
            **{
                f"Z{duration}": result.get("Z", {}).get(duration) if has_incumbent else None
                for duration in DURATIONS
            },
            "selected_policy_rows": len(rows),
            "policy_file": str(policy_path) if has_incumbent else None,
            "chain_policy_file": (
                str(chain_policy_path) if has_incumbent else None),
            "selected_chain_decisions": (
                len(result.get("chain_decisions", [])) if has_incumbent else 0),
            "decision_variable_files": decision_files,
            "decision_variable_rows": decision_row_counts,
            "stats": stats,
            "has_incumbent": has_incumbent,
            "certified_within_gap": certified,
            "target_mip_gap": args.mip_gap,
            "time_limit_seconds": args.time_limit_seconds,
            "elapsed_seconds": time.time() - started,
        }
        write_json(summary_path, summary)
        checkpoint("FINISHED", final_status=summary["status"],
                   certified_within_gap=certified)
        print(f"STAGE2_CASE_COMPLETE {args.window:02d} {station} "
              f"status={summary['status']} certified={certified}", flush=True)
        return 0 if has_incumbent else 2
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "status": "ERROR",
            "solver": "FULL_POOL_MILP",
            "scope": "FULL_FINITE_ROUTE_AND_TRIP_POOL",
            "column_generation": False,
            "window": args.window,
            "station": station,
            "durations": list(DURATIONS),
            "duration_weights": DURATION_WEIGHTS,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
        write_json(summary_path, failure)
        write_json(progress_path, failure)
        traceback.print_exc()
        return 1
    finally:
        if environment is not None:
            environment.dispose()
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
