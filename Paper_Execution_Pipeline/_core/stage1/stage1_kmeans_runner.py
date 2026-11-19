"""Nonanticipative Stage 1 with fixed-K common-station K-means support.

The original ``stage1_network_model.py`` is imported read-only and
left untouched.  This runner replaces only two extension points:

1. scenario reduction: the precomputed common-station K-means K=10 support;
2. cache preparation: provably fixed-zero variables and pre-event state prefixes
   are removed before the original flow constraints are built.

Dispatch variables use the original model's information-cell sharing, but with
``DISRUPT_BUFFER=0`` so it matches ``stage1_recourse.py`` exactly:

    tau <= start+30  -> (30,60,90)
    tau <= start+60  -> (60,90)
    otherwise        -> (90,)

Run the full model:
    python stage1_kmeans_runner.py

Audit cache-size reduction without solving:
    python stage1_kmeans_runner.py --audit-window 20
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
CORE_FILE = BASE / "stage1_network_model.py"
SUPPORT_FILE = (BASE / "Common_Station_KMeans_K10"
                / "common_kmeans_k10_representatives.csv")
OUTPUT_ROOT = Path(os.getenv("TSB_NA_OUTPUT_ROOT")
                   or str(BASE / "MILP_Nonanticipative_runtime"))
PRUNE_EPS = float(os.getenv("TSB_NA_PRUNE_EPS", "1e-9"))
PRUNE_STATE_PREFIX = os.getenv("TSB_NA_PRUNE_STATE_PREFIX", "1") != "0"


def load_core():
    spec = importlib.util.spec_from_file_location("stage1_core_readonly", CORE_FILE)
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    return core


def configure_output(core):
    core.OUTPUT_ROOT = str(OUTPUT_ROOT)
    core.DIR_OUT_ALL = str(OUTPUT_ROOT / "All")
    core.DIR_OUT_ALLOC = str(OUTPUT_ROOT / "Allocation")
    core.DIR_OUT_LOG = str(OUTPUT_ROOT / "Logs")
    core.DIR_OUT_TIMETABLE = str(OUTPUT_ROOT / "Generated_Timetables")
    core.DIR_OUT_DETAIL = str(OUTPUT_ROOT / "Decision_Variables")
    core.DIR_OUT_PERF = str(OUTPUT_ROOT / "Performance")
    for path in (core.OUTPUT_ROOT, core.DIR_OUT_ALL, core.DIR_OUT_ALLOC,
                 core.DIR_OUT_LOG, core.DIR_OUT_TIMETABLE,
                 core.DIR_OUT_DETAIL, core.DIR_OUT_PERF):
        os.makedirs(path, exist_ok=True)


def load_support():
    if not SUPPORT_FILE.exists():
        raise FileNotFoundError(
            f"Common K-means support not found: {SUPPORT_FILE}\n"
            "Provide the common K-means representative table first."
        )
    support = pd.read_csv(SUPPORT_FILE)
    required = {"hour", "representative_station", "duration",
                "scenario_probability"}
    missing = required - set(support.columns)
    if missing:
        raise ValueError(f"Support file missing columns: {sorted(missing)}")
    return support


SUPPORT = None


def reduce_common_kmeans(scenarios, k_per_duration=10, wdata=None,
                         include_zero_resource_branches=False):
    """Return the requested common-station K x 3 support.

    A missing raw branch is a zero-impact duration child.  It is intentionally
    omitted from the legacy flow model.  Exact nonanticipative CG sets
    ``include_zero_resource_branches=True``: such a child has no passenger-flow
    variables or objective, but retains the shared y and depot occupancy.
    """
    if not scenarios:
        return []
    global SUPPORT
    if SUPPORT is None:
        SUPPORT = load_support()

    k_per_duration = int(k_per_duration)
    if "k" in SUPPORT.columns:
        available_k = {int(value) for value in SUPPORT["k"].unique()}
        if available_k != {k_per_duration}:
            raise ValueError(
                f"support file contains K={sorted(available_k)}, "
                f"but K={k_per_duration} was requested")

    hour = int(str(scenarios[0]["id"]).split("_", 1)[0])
    table = SUPPORT[SUPPORT["hour"] == hour]
    if table.empty:
        raise ValueError(f"No common K-means support for hour {hour}")

    raw = {(str(s["station"]), int(s["duration"])): s for s in scenarios}
    reduced = []
    zero_branches = 0
    for row in table.itertuples(index=False):
        key = (str(row.representative_station), int(row.duration))
        scenario = raw.get(key)
        if scenario is None:
            zero_branches += 1
            if not include_zero_resource_branches:
                continue
            siblings = [s for (station, _), s in raw.items()
                        if station == key[0]]
            if not siblings:
                raise ValueError(
                    f"zero-impact branch {key} has no station sibling template")
            template = max(siblings, key=lambda s: int(s["duration"]))
            disrupt_start = int(template["DISRUPT_START_T"])
            branch_id = f"{hour}_{key[0]}_{key[1]}"
            rep = template.copy()
            rep.update({
                "id": branch_id,
                "station": key[0],
                "duration": key[1],
                "DISRUPT_START_T": disrupt_start,
                "DISRUPT_END_T": disrupt_start + key[1],
                "DISPATCH_END_MIN": disrupt_start + key[1],
                "canceled_trains": set(),
                "theta": {},
                "Omega_m": [],
                "spare_capacity": {},
                "pair_score": {},
                "strand_score": {},
                "impact_by_station": {},
                "probability": float(row.scenario_probability),
                "prob_station": float(row.cluster_station_probability),
                "block_station": key[0],
                "representative_id": branch_id,
                "cluster_id": int(row.cluster),
                "zero_impact_branch": True,
            })
            reduced.append(rep)
            continue
        rep = scenario.copy()
        rep["probability"] = float(row.scenario_probability)
        rep["prob_station"] = float(row.cluster_station_probability)
        rep["block_station"] = str(row.representative_station)
        rep["representative_id"] = rep["id"]
        rep["cluster_id"] = int(row.cluster)
        rep["zero_impact_branch"] = False
        reduced.append(rep)

    probability = sum(float(s["probability"]) for s in reduced)
    zero_mode = "resource-only" if include_zero_resource_branches else "omitted"
    expected = k_per_duration * 3
    print(f"    common K-means K={k_per_duration}: {k_per_duration} stations, "
          f"{len(reduced)}/{expected} modeled blocks, "
          f"zero-impact={zero_branches} ({zero_mode}), "
          f"modeled probability={probability:.6f}")
    return reduced


def _copy_cache(cache):
    """Shallow-copy the cache and copy only lists that will be pruned."""
    out = dict(cache)
    for key in ("trip_ids", "b_var_keys", "a_var_keys", "l_var_keys",
                "q_var_keys", "w_var_keys", "od_var_keys"):
        out[key] = list(cache.get(key, []))
    return out


def compact_cache(cache, scenario, wdata, eps=PRUNE_EPS,
                  prune_state_prefix=PRUNE_STATE_PREFIX,
                  preserve_trip_ids=False):
    """Remove variables whose value is provably zero.

    Safe eliminations:
    - b with cumulative-demand UB=0 or boarding at the OD destination;
    - a with cumulative-demand UB=0 or alighting at the OD origin;
    - od with cumulative-demand UB=0;
    - l for an (OD, vehicle) pair having no remaining boarding/alighting event;
    - leading q/w states before their first possible demand or flow event.

    ``preserve_trip_ids=True`` is for exact nonanticipative block CG: a trip
    with no passenger event in one duration must still keep the shared y and
    depot occupancy forced by another duration in its information cell.

    All lookup dictionaries may retain removed keys because the original model
    already guards every access with ``if key in variable_dict``.
    """
    compact = _copy_cache(cache)
    omega_map = wdata["Omega_Map"]
    cumulative = wdata["CumD"]
    demand = wdata["D"]
    sigma = 2

    b_keys = []
    for key in compact["b_var_keys"]:
        om, _, t, station = key
        destination = omega_map.get(om, (None, None))[1]
        if station == destination:
            continue
        if cumulative.get((om, t), 0.0) <= eps:
            continue
        b_keys.append(key)

    a_keys = []
    for key in compact["a_var_keys"]:
        om, _, t, station = key
        origin = omega_map.get(om, (None, None))[0]
        if station == origin:
            continue
        if cumulative.get((om, t), 0.0) <= eps:
            continue
        a_keys.append(key)

    od_keys = [key for key in compact["od_var_keys"]
               if cumulative.get((key[0], key[1]), 0.0) > eps]

    active_vehicle_pairs = {(om, vehicle) for om, vehicle, _, _ in b_keys}
    active_vehicle_pairs.update((om, vehicle) for om, vehicle, _, _ in a_keys)
    l_keys = [key for key in compact["l_var_keys"]
              if (key[0], key[1]) in active_vehicle_pairs
              and cumulative.get((key[0], key[2]), 0.0) > eps]

    if prune_state_prefix:
        first_q_event = {}
        w_support = {}
        demand_times = defaultdict(list)
        board_times = defaultdict(list)
        alight_times_by_node = defaultdict(list)
        for (om, station, t), value in demand.items():
            if value > eps:
                demand_times[(om, station)].append(t)
        for om, _, t, station in b_keys:
            board_times[(om, station)].append(t)
        for om, _, t, station in a_keys:
            alight_times_by_node[(om, station)].append(t)
        for om in scenario["Omega_m"]:
            if om not in omega_map:
                continue
            for station in cache["N_omega"].get(om, set()):
                node = (om, station)
                event_times = list(demand_times.get(node, ()))
                event_times.extend(board_times.get(node, ()))
                event_times.extend(t + sigma
                                   for t in alight_times_by_node.get(node, ()))
                if event_times:
                    first_q_event[node] = min(event_times)

                alight_times = alight_times_by_node.get(node, ())
                if alight_times:
                    # w_t is the rolling sigma-minute inventory generated by
                    # alighting.  After the last alight plus sigma-1 it is
                    # identically zero, so the suffix needs no state variable.
                    support = set()
                    for event_time in alight_times:
                        support.update(range(event_time, event_time + sigma))
                    w_support[node] = support

        q_keys = [key for key in compact["q_var_keys"]
                  if key[:2] in first_q_event and key[2] >= first_q_event[key[:2]]]
        w_keys = [key for key in compact["w_var_keys"]
                  if key[:2] in w_support and key[2] in w_support[key[:2]]]
    else:
        q_keys = compact["q_var_keys"]
        w_keys = compact["w_var_keys"]

    if not preserve_trip_ids:
        active_bus_trips = {vehicle for _, vehicle, _, _ in b_keys
                            if str(vehicle).startswith("B_")}
        compact["trip_ids"] = [tid for tid in compact["trip_ids"]
                               if tid in active_bus_trips]
    compact["b_var_keys"] = b_keys
    compact["a_var_keys"] = a_keys
    compact["l_var_keys"] = l_keys
    compact["q_var_keys"] = q_keys
    compact["w_var_keys"] = w_keys
    compact["od_var_keys"] = od_keys
    return compact


def cache_counts(caches):
    keys = ("trip_ids", "b_var_keys", "a_var_keys", "l_var_keys",
            "q_var_keys", "w_var_keys", "od_var_keys")
    return {key: sum(len(cache.get(key, [])) for cache in caches.values())
            for key in keys}


ORIGINAL_SOLVE = None


def solve_milp_compact(env, depots, scenarios, all_caches, wdata):
    before = cache_counts(all_caches)
    compact = {}
    for scenario in scenarios:
        sid = scenario["id"]
        cache = compact_cache(all_caches[sid], scenario, wdata)
        compact[sid] = cache
        active = set(cache["trip_ids"])
        scenario["bus_trips"] = {
            tid: trip for tid, trip in scenario.get("bus_trips", {}).items()
            if tid in active
        }
    after = cache_counts(compact)
    old_total, new_total = sum(before.values()), sum(after.values())
    print("    [compact cache] "
          + ", ".join(f"{key} {before[key]:,}->{after[key]:,}"
                      for key in before))
    print(f"    [compact cache] indexed variables {old_total:,}->{new_total:,} "
          f"({1 - new_total / max(old_total, 1):.1%} reduction)")
    return ORIGINAL_SOLVE(env, depots, scenarios, compact, wdata)


def install_extensions(core):
    global ORIGINAL_SOLVE
    configure_output(core)
    core.REDUCTION = "common_kmeans_k10"
    core.K_PER_DURATION = 10
    core.K_STATIONS = 10
    core.DISRUPT_BUFFER = 0
    core.reduce_scenarios = reduce_common_kmeans

    # The original function's default buffer was bound to 30 at definition
    # time, so wrap it explicitly to match Stage 2's buffer=0 information tree.
    original_shared_y = core._build_shared_y

    def shared_y_buffer_zero(model, scenarios, buffer=0, **kwargs):
        return original_shared_y(model, scenarios, buffer=0, **kwargs)

    core._build_shared_y = shared_y_buffer_zero
    ORIGINAL_SOLVE = core.solve_milp
    core.solve_milp = solve_milp_compact


def audit_window(core, start_h):
    """Build real caches for one window and report compaction without solving."""
    schedule = core.load_schedule(core.FILE_SCHEDULE)
    od = core.load_od(core.FILE_OD)
    bus_tt = core.load_distance_matrix(core.FILE_DISTANCE)
    routes = core.load_route_library(core.FILE_PREDEFINED_ROUTES)
    wdata = core.build_window_data(start_h, schedule, od)
    raw = core.load_disruption_scenarios(start_h, wdata)
    reduced = reduce_common_kmeans(raw, 10, wdata)
    depots = sorted({station for route in routes.values()
                     for station in (route["stops"][0], route["stops"][-1])})
    block_trips = core.build_block_trips(reduced, start_h, depots, bus_tt, routes)

    original, compacted = {}, {}
    for scenario in reduced:
        selected_routes, trips = block_trips[scenario["id"]]
        scenario["bus_trips"] = trips
        cache = core.build_stage2_cache(scenario, wdata, trips)
        original[scenario["id"]] = cache
        compacted[scenario["id"]] = compact_cache(cache, scenario, wdata)

    before, after = cache_counts(original), cache_counts(compacted)
    rows = []
    for key in before:
        reduction = 1 - after[key] / max(before[key], 1)
        rows.append({"family": key, "before": before[key], "after": after[key],
                     "reduction": reduction})
    report = pd.DataFrame(rows)
    out = OUTPUT_ROOT / "Performance"
    out.mkdir(parents=True, exist_ok=True)
    report.to_csv(out / f"Cache_Compaction_Audit_{start_h:02d}.csv", index=False)
    print(report.to_string(index=False, formatters={"reduction": "{:.1%}".format}))
    print(f"TOTAL {sum(before.values()):,}->{sum(after.values()):,} "
          f"({1 - sum(after.values()) / max(sum(before.values()), 1):.1%} reduction)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-window", type=int)
    args = parser.parse_args()
    core = load_core()
    install_extensions(core)
    if args.audit_window is not None:
        audit_window(core, args.audit_window)
    else:
        core.main()


if __name__ == "__main__":
    main()
