"""Standalone exact finite-pool column generation for Stage 1.

The module consumes already prepared scenarios, full-support caches, and
window data through :class:`CGProblem`.  It intentionally imports no project
module: the information filtration, load aggregation, full/RMP master, pricing
models, and final restricted MILP are all defined here.

A macro-column is ``(block_station, trip_id, information_cell)``.  Its shared
dispatch variable is present in every duration in the cell, while each
duration owns its passenger-flow variables.  Coupling rows are always built
from the fixed full cache; only bus-local variables and rows are restricted.
"""
from __future__ import annotations

import hashlib
import math
import shutil
import tempfile
import threading
import time
from bisect import bisect_left
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, Mapping, Sequence

import gurobipy as gp
from gurobipy import GRB
from state_compression import (
    ExpandedStateView,
    StateSegment,
    build_bus_load_segments,
    build_train_load_segments,
)
from queue_compression import build_queue_compression
from walking_elimination import build_walking_elimination_plan

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # package root (bbconfig)
from bbconfig import cfg as _bbcfg  # noqa: E402


SIGMA = int(_bbcfg.get("operations.walking_transfer_min"))   # Table 3: sigma_ij = 2 min
RC_EPS = 1e-7
BOX_BOUND_REL_SAFETY = 1e-12
# Exact support reduction may remove only variables whose mathematical upper
# bound is zero.  A positive tolerance would silently delete tiny, but valid,
# demand and change the optimization problem.
SUPPORT_EPS = 0.0
DEFAULT_DISRUPT_BUFFER = int(_bbcfg.get("time.clearance_buffer_min"))   # Table 3: delta (callers pass it explicitly)
DECISION_ROW_EPS = 1e-4
DECISION_TIME_MIN = 0
DECISION_TIME_MAX = int(_bbcfg.get("time.horizon_min"))   # Table 3: T
# Eq. (13) [eq:obj_db]: Z_m = 1/(T - t0) sum_{omega} sum_{t in Gamma, t >= t0} z_t, and
# Eq. (2) [eq:inventory]: z_t = kappa [queue + walking] + onboard load (Table 3: T, t0, kappa).
OBJECTIVE_ONSET_MIN = int(_bbcfg.get("time.disruption_onset_min"))       # t0 (window minutes)
OBJECTIVE_NORMALISER = float(
    int(_bbcfg.get("time.horizon_min")) - OBJECTIVE_ONSET_MIN)          # T - t0
OUT_OF_VEHICLE_WEIGHT = float(_bbcfg.get("operations.out_of_vehicle_weight"))   # kappa
# Gurobi can return an accepted incumbent with bound noise a few multiples of
# FeasibilityTol.  Extraction clamps only this numerical noise; larger
# negative passenger flows still fail loudly.
EXTRACTION_SOLUTION_TOLERANCE = 1e-5
ColumnKey = tuple[str, str, tuple[int, ...]]


@dataclass(frozen=True)
class CGConfig:
    """All numerical/model choices needed by the standalone engine."""

    sigma: int = SIGMA
    disrupt_buffer: int = DEFAULT_DISRUPT_BUFFER
    bus_capacity: float = float(_bbcfg.get("operations.bus_capacity"))      # Table 3: C_v, v in V_rho
    train_capacity: float = float(_bbcfg.get("operations.rail_capacity"))   # Table 3: C_v, v in V_r
    min_trip_boarding: float = float(_bbcfg.get("operations.min_trip_boarding"))   # Table 3: epsilon
    max_total_buses: int = int(_bbcfg.get("operations.fleet_size"))         # Table 3: B
    mip_gap: float = float(_bbcfg.get("solver.stage1.mip_gap"))
    # ``None`` retains Gurobi's unlimited default.  Tests or experiments that
    # need a finite limit pass one explicitly.
    time_limit: float | None = None
    output_flag: int = 0
    optimality_tolerance: float = 1e-9
    feasibility_tolerance: float = 1e-9
    pricing_output_flag: int = 0
    rc_epsilon: float = RC_EPS
    # Production models contain millions of generated names that are never
    # queried.  Discarding them saves build time and memory without changing
    # any variable, row, coefficient, bound, or objective term.  Keep the
    # standalone API default off so callers that inspect names retain them.
    ignore_names: bool = False
    # Root CG retains a separable service-overlap relaxation.  Integer masters
    # replace it with an exact depot-based path cover with deadheading.
    enforce_vehicle_chains: bool = True
    turnaround_minutes: int = 0
    # Exact computational reformulations.  Defaults preserve the explicit
    # minute-indexed reference model for standalone audits; the production
    # runner enables them after parity tests.
    eliminate_w: bool = False
    eliminate_destination_states: bool = False
    compress_load_states: bool = False
    compress_queue_states: bool = False
    # Exact projection of bus loads.  The aggregate load state is retained at
    # union b/a events, while OD feasibility is enforced by cumulative
    # nonnegativity rows after alighting events.  OD-minute load values needed
    # by result CSVs are reconstructed exactly from the solved b/a ledger.
    hybrid_bus_load_states: bool = False
    # Optional RMP-only algorithm controls.  ``None`` leaves the corresponding
    # Gurobi parameter untouched, preserving the installed solver's default.
    # These controls deliberately do not apply to pricing LPs or the final
    # restricted MILP.
    rmp_method: int | None = None
    rmp_lp_warm_start: int | None = None
    final_method: int | None = None
    final_mip_focus: int | None = None
    final_cuts: int | None = None
    final_cut_passes: int | None = None
    final_concurrent_mip: int | None = None
    final_symmetry: int | None = None
    final_heuristics: float | None = None
    final_no_rel_heur_time: float | None = None
    final_improve_start_time: float | None = None
    final_best_obj_stop: float | None = None

    def __post_init__(self):
        if self.sigma < 0:
            raise ValueError("sigma must be nonnegative")
        if self.disrupt_buffer < 0:
            raise ValueError("disrupt_buffer must be nonnegative")
        if self.bus_capacity <= 0 or self.train_capacity <= 0:
            raise ValueError("vehicle capacities must be positive")
        if self.max_total_buses < 0:
            raise ValueError("max_total_buses must be nonnegative")
        if self.time_limit is not None and self.time_limit <= 0:
            raise ValueError("time_limit must be positive or None")
        if self.rc_epsilon < 0:
            raise ValueError("rc_epsilon must be nonnegative")
        if not isinstance(self.ignore_names, bool):
            raise ValueError("ignore_names must be a bool")
        if not isinstance(self.enforce_vehicle_chains, bool):
            raise ValueError("enforce_vehicle_chains must be a bool")
        for name in (
                "eliminate_w", "eliminate_destination_states",
                "compress_load_states",
                "compress_queue_states", "hybrid_bus_load_states"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        if (self.rmp_method is not None and
                (isinstance(self.rmp_method, bool) or
                 int(self.rmp_method) != self.rmp_method or
                 not -1 <= int(self.rmp_method) <= 6)):
            raise ValueError("rmp_method must be an integer in [-1, 6] or None")
        if (self.rmp_lp_warm_start is not None and
                (isinstance(self.rmp_lp_warm_start, bool) or
                 int(self.rmp_lp_warm_start) != self.rmp_lp_warm_start or
                 int(self.rmp_lp_warm_start) not in (0, 1, 2))):
            raise ValueError(
                "rmp_lp_warm_start must be one of 0, 1, 2, or None")
        if (self.final_method is not None and
                (isinstance(self.final_method, bool) or
                 int(self.final_method) != self.final_method or
                 not -1 <= int(self.final_method) <= 5)):
            raise ValueError(
                "final_method must be an integer in [-1, 5] or None")
        if (self.final_mip_focus is not None and
                int(self.final_mip_focus) not in (0, 1, 2, 3)):
            raise ValueError("final_mip_focus must be in [0, 3] or None")
        if (self.final_cuts is not None and
                int(self.final_cuts) not in (-1, 0, 1, 2, 3)):
            raise ValueError("final_cuts must be in [-1, 3] or None")
        if (self.final_cut_passes is not None and
                int(self.final_cut_passes) < -1):
            raise ValueError("final_cut_passes must be >= -1 or None")
        if (self.final_concurrent_mip is not None and
                int(self.final_concurrent_mip) < 1):
            raise ValueError("final_concurrent_mip must be >= 1 or None")
        if (self.final_symmetry is not None and
                int(self.final_symmetry) not in (-1, 0, 1, 2)):
            raise ValueError("final_symmetry must be in [-1, 2] or None")
        if (self.final_heuristics is not None and
                not 0.0 <= float(self.final_heuristics) <= 1.0):
            raise ValueError("final_heuristics must be in [0, 1] or None")
        if (self.final_no_rel_heur_time is not None and
                float(self.final_no_rel_heur_time) < 0.0):
            raise ValueError("final_no_rel_heur_time must be nonnegative or None")
        if (self.final_improve_start_time is not None and
                float(self.final_improve_start_time) < 0.0):
            raise ValueError(
                "final_improve_start_time must be nonnegative or None")
        if (self.final_best_obj_stop is not None and
                not math.isfinite(float(self.final_best_obj_stop))):
            raise ValueError("final_best_obj_stop must be finite or None")
        if self.turnaround_minutes < 0:
            raise ValueError("turnaround_minutes must be nonnegative")


@dataclass(frozen=True)
class CGProblem:
    """Prepared, fixed-support input for exact column generation.

    ``scenarios`` must already contain ``id``, ``duration``,
    ``DISRUPT_START_T``, ``Omega_m``, ``spare_capacity``, and ``bus_trips``.
    ``full_caches`` must be built once from the complete physical trip pool.
    ``wdata`` is the prepared window dictionary used by the original model.
    ``env`` is optional; when omitted, Gurobi creates its default environment.
    """

    depots: Sequence[str]
    scenarios: Sequence[Mapping]
    full_caches: Mapping[str, Mapping] | None
    wdata: Mapping
    env: object | None = None
    # Complete station-to-station deadhead times.  If a directed entry is
    # absent, the reverse entry is used as in the preprocessing loader.
    travel_times: Mapping[tuple[str, str], float] | None = None
    # ``None`` is the Stage-1 design problem.  A mapping fixes the same depot
    # variables to a realized Stage-1 allocation and turns the engine into the
    # structurally identical Stage-2 recovery problem.
    fixed_depot_counts: Mapping[str, float] | None = None

    def __post_init__(self):
        if not self.depots:
            raise ValueError("CGProblem.depots must not be empty")
        depots = {str(depot) for depot in self.depots}
        if self.fixed_depot_counts is not None:
            counts = {str(depot): float(value)
                      for depot, value in self.fixed_depot_counts.items()}
            unknown = set(counts) - depots
            if unknown:
                raise ValueError(
                    f"fixed_depot_counts contains unknown depots: {sorted(unknown)}")
            for depot in depots:
                value = counts.get(depot, 0.0)
                if (not math.isfinite(value) or value < 0 or
                        abs(value - round(value)) > 1e-9):
                    raise ValueError(
                        f"invalid fixed depot count for {depot}: {value}")
        ids = [str(s["id"]) for s in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("scenario ids must be unique")
        # ``None``/an empty mapping is permitted for the metadata-only
        # ``count_macro_columns`` audit.  Every optimization entry point calls
        # ``_require_full_caches`` before touching model support.
        if self.full_caches:
            missing = [sid for sid in ids if sid not in self.full_caches]
            if missing:
                raise ValueError(f"missing full caches for scenarios: {missing}")
        required = {
            "Omega_Map", "trains_at", "station_of", "train_active_times",
            "train_dest_map", "train_dwells", "Vm", "Gamma", "START_MIN",
            "CumD", "D",
        }
        absent = sorted(required - set(self.wdata))
        if absent:
            raise ValueError(f"wdata is missing keys: {absent}")

    @property
    def scenarios_by_id(self) -> dict[str, Mapping]:
        return {str(s["id"]): s for s in self.scenarios}


@dataclass(frozen=True)
class ColumnSpec:
    """Metadata for one nonanticipative trip-cell macro-column."""

    key: ColumnKey
    scenario_ids: tuple[str, ...]
    score: int

    @property
    def block(self) -> str:
        return self.key[0]

    @property
    def trip_id(self) -> str:
        return self.key[1]

    @property
    def cell(self) -> tuple[int, ...]:
        return self.key[2]


@dataclass
class PricingResult:
    """Optimal local price and an optional sparse coefficient profile."""

    key: ColumnKey
    status: str
    reduced_cost: float
    y: float
    direct_cost: float
    q_coefficients: dict[tuple, float]
    w_coefficients: dict[tuple, float]
    depot_coefficients: dict[tuple, float]
    b_values: dict[tuple, float]
    a_values: dict[tuple, float]
    l_values: dict[tuple, float]
    exact: bool = True
    box_lower_bound: float | None = None
    solver_lower_bound: float | None = None

    @property
    def certificate_value(self) -> float:
        if self.exact:
            return float(self.solver_lower_bound
                         if self.solver_lower_bound is not None
                         else self.reduced_cost)
        if self.box_lower_bound is None:
            raise ValueError("screened pricing result has no certified lower bound")
        return float(self.box_lower_bound)

    def dot_product_reduced_cost(self, duals: Mapping[str, Mapping]) -> float:
        if not self.exact:
            raise ValueError("a box-screened result has no primal profile")
        value = self.direct_cost
        value -= sum(duals.get("q", {}).get(k, 0.0) * coefficient
                     for k, coefficient in self.q_coefficients.items())
        value -= sum(duals.get("w", {}).get(k, 0.0) * coefficient
                     for k, coefficient in self.w_coefficients.items())
        value -= sum(duals.get("depot", {}).get(k, 0.0) * coefficient
                     for k, coefficient in self.depot_coefficients.items())
        return float(value)


def _cfg(config: CGConfig | None) -> CGConfig:
    return config if config is not None else CGConfig()


def _new_model(name: str, problem: CGProblem) -> gp.Model:
    return (gp.Model(name, env=problem.env) if problem.env is not None
            else gp.Model(name))


def _apply_rmp_lp_parameters(model: gp.Model, config: CGConfig) -> None:
    """Apply opt-in algorithm controls to an LP master, and nowhere else."""
    if config.rmp_method is not None:
        model.Params.Method = int(config.rmp_method)
    if config.rmp_lp_warm_start is not None:
        model.Params.LPWarmStart = int(config.rmp_lp_warm_start)


def _require_full_caches(problem: CGProblem) -> Mapping[str, Mapping]:
    if not problem.full_caches:
        raise ValueError(
            "full_caches are required for optimization; only "
            "count_macro_columns supports a metadata-only CGProblem")
    missing = [str(s["id"]) for s in problem.scenarios
               if str(s["id"]) not in problem.full_caches]
    if missing:
        raise ValueError(f"missing full caches for scenarios: {missing}")
    return problem.full_caches


def _normalise_key(key) -> ColumnKey:
    block, tid, cell = key
    return str(block), str(tid), tuple(int(duration) for duration in cell)


def _pool_hash(keys) -> str:
    payload = "\n".join(repr(key) for key in sorted(keys)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _status_name(code: int) -> str:
    return {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }.get(code, str(code))


def _travel_minutes(travel_times: Mapping | None, origin, destination):
    """Return a finite nonnegative deadhead time, or ``None`` if unavailable."""
    origin, destination = str(origin), str(destination)
    if origin == destination:
        return 0
    if travel_times is None:
        return None
    value = travel_times.get((origin, destination))
    if value is None:
        value = travel_times.get((destination, origin))
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        return None
    return int(round(value))


def _trip_run_start(trip: Mapping) -> int:
    return int(trip.get("RunStart", trip["DispatchStart"]))


def _trip_service_end(trip: Mapping) -> int:
    return int(trip.get(
        "ServiceEnd", trip.get("RunEnd", trip.get(
            "ResourceEnd", trip["DispatchStart"]))))


def _trip_start_station(trip: Mapping) -> str:
    station = trip.get("RouteOrigin")
    if station is None:
        stops = tuple(trip.get("RouteStops", ()))
        station = stops[0] if stops else None
    if station is None:
        raise ValueError(f"trip {trip.get('TripID')} has no start station")
    return str(station)


def _trip_end_station(trip: Mapping) -> str:
    station = trip.get("ServiceEndStation", trip.get("RouteDest"))
    if station is None:
        legs = tuple(trip.get("Legs", ()))
        if legs and tuple(legs[-1].get("stops", ())):
            station = tuple(legs[-1]["stops"])[-1]
    if station is None:
        stops = tuple(trip.get("RouteStops", ()))
        station = stops[-1] if stops else None
    if station is None:
        raise ValueError(f"trip {trip.get('TripID')} has no end station")
    return str(station)


def _depot_interval(trip: Mapping, config: CGConfig) -> tuple[int, int]:
    """Resource interval in the separable root-CG lower-bound relaxation."""
    if config.enforce_vehicle_chains:
        # Any physical chain uses a bus during passenger service.  Pull-out,
        # inter-trip, and pull-in deadheads are relaxed only in the root LP.
        return _trip_run_start(trip), _trip_service_end(trip)
    return int(trip["DispatchStart"]), int(trip["ResourceEnd"])


def _compatible_trip_arcs(trips: Mapping[str, Mapping], travel_times,
                          turnaround_minutes: int = 0) -> dict[str, tuple[str, ...]]:
    """Return every feasible direct ``trip -> deadhead -> trip`` arc."""
    normal = {str(tid): trip for tid, trip in trips.items()}
    by_depot = defaultdict(list)
    for tid, trip in normal.items():
        by_depot[str(trip["Depot"])].append(tid)
    successors = {tid: [] for tid in normal}
    for tids in by_depot.values():
        ordered = sorted(tids, key=lambda tid: (
            _trip_run_start(normal[tid]), _trip_service_end(normal[tid]), tid))
        for tid in ordered:
            first = normal[tid]
            ready = _trip_service_end(first) + int(turnaround_minutes)
            end_station = _trip_end_station(first)
            for nxt in ordered:
                if nxt == tid:
                    continue
                second = normal[nxt]
                if _trip_run_start(second) < ready:
                    continue
                deadhead = _travel_minutes(
                    travel_times, end_station, _trip_start_station(second))
                if (deadhead is not None and
                        ready + deadhead <= _trip_run_start(second)):
                    successors[tid].append(nxt)
    return {tid: tuple(values) for tid, values in successors.items()}


def _chain_arc_decision_time(predecessor: Mapping, successor: Mapping,
                             travel_times: Mapping) -> int:
    """Latest causal commitment time for ``predecessor -> successor``."""
    deadhead = _travel_minutes(
        travel_times, _trip_end_station(predecessor),
        _trip_start_station(successor))
    if deadhead is None:
        raise ValueError("chain arc has no finite inter-trip deadhead")
    return _trip_run_start(successor) - int(deadhead)


def _add_nonanticipative_vehicle_chain_constraints(
        model: gp.Model, problem: CGProblem, config: CGConfig,
        y_by_scenario: Mapping[str, Mapping[str, gp.Var]],
        y_by_column: Mapping[ColumnKey, gp.Var],
        fleet_capacity: Mapping[str, object]) -> tuple[dict, dict]:
    """Build causal vehicle duties on the duration information tree.

    ``first[v]`` uses trip ``v``'s depot-mobilisation information cell.  A
    predecessor choice ``link[u,v]`` uses its own arc information time,
    ``run_start[v] - deadhead[end(u), start(v)]``: the latest time at which
    the bus must leave ``u``'s end location to start ``v`` on time.  Duration
    branches that remain indistinguishable then reference the exact same
    variable; observed branches receive separate variables.

    Only incoming path-cover flow is required explicitly.  Temporal arc
    acyclicity and the outgoing inequality produce disjoint depot-rooted
    paths, while allowing a selected trip to end a duty without a separate
    anticipative ``last`` decision.
    """
    by_scenario = {
        str(scenario["id"]): {
            "first": {}, "link": {}, "first_meta": {}, "link_meta": {},
            "successors": {},
            "decision_scope": "arc_latest_departure_information_cell",
        }
        for scenario in problem.scenarios
    }
    scenarios_by_block = _group_scenarios(problem.scenarios)
    columns_by_block = defaultdict(list)
    for raw_key, variable in y_by_column.items():
        block, tid, cell = _normalise_key(raw_key)
        columns_by_block[block].append((tid, cell, variable))

    first_vars = {}
    first_decisions = {}
    link_vars = {}
    constraint_count = 0
    zero_fixed_links = 0
    first_substituted = 0
    physical_arc_count = 0

    for block, per_duration in scenarios_by_block.items():
        disruption_starts = {int(scenario["DISRUPT_START_T"])
                             for scenario in per_duration.values()}
        if len(disruption_starts) != 1:
            raise ValueError(f"block={block}: inconsistent disruption starts")
        disruption_start = disruption_starts.pop()
        durations = tuple(sorted(int(value) for value in per_duration))

        trips = {}
        for scenario in per_duration.values():
            for raw_tid, trip in scenario.get("bus_trips", {}).items():
                tid = str(raw_tid)
                previous = trips.get(tid)
                if previous is not None and (
                        str(previous.get("Depot")) != str(trip.get("Depot")) or
                        _trip_run_start(previous) != _trip_run_start(trip) or
                        _trip_service_end(previous) != _trip_service_end(trip)):
                    raise ValueError(
                        f"block={block}, trip={tid}: inconsistent trip metadata")
                trips[tid] = trip

        active_trip_ids = {
            tid for tid, _cell, _variable in columns_by_block.get(block, ())
        }
        # Screen trips with no usable information-cell support before the
        # quadratic physical arc pass.  The surviving timetable/road arcs are
        # computed once for this location block and reused by all durations.
        successors = _compatible_trip_arcs(
            {tid: trips[tid] for tid in active_trip_ids},
            problem.travel_times, config.turnaround_minutes)
        physical_arc_count += sum(len(values) for values in successors.values())
        for tid in active_trip_ids:
            trip = trips[tid]
            depot = str(trip["Depot"])
            if depot not in fleet_capacity:
                raise ValueError(f"trip {tid} uses unknown depot {depot}")
            outbound = _travel_minutes(
                problem.travel_times, depot, _trip_start_station(trip))
            returned = _travel_minutes(
                problem.travel_times, _trip_end_station(trip), depot)
            if outbound is None or returned is None:
                raise ValueError(
                    f"trip {tid} cannot be connected to home depot {depot}")
            if int(trip["DispatchStart"]) + outbound > _trip_run_start(trip):
                raise ValueError(f"trip {tid} has infeasible recorded pull-out time")

        # A link decision has its own information partition because its bus
        # leaves the predecessor's end location, not the home depot.  If a
        # shared cell contains a duration in which either endpoint activation
        # is unavailable, the common link is fixed to zero.  This prevents an
        # early relocation from using knowledge of a later duration branch.
        for predecessor, next_ids in successors.items():
            if predecessor not in active_trip_ids:
                continue
            for successor in next_ids:
                if successor not in active_trip_ids:
                    continue
                decision_time = _chain_arc_decision_time(
                    trips[predecessor], trips[successor],
                    problem.travel_times)
                for cell in info_cells(
                        decision_time, disruption_start, durations,
                        config.disrupt_buffer):
                    supported = []
                    unsupported = []
                    for duration in cell:
                        scenario = per_duration[int(duration)]
                        sid = str(scenario["id"])
                        if (predecessor in y_by_scenario[sid] and
                                successor in y_by_scenario[sid]):
                            supported.append(sid)
                        else:
                            unsupported.append(sid)
                    if not supported:
                        continue
                    if unsupported:
                        # This shared pre-observation action is unavailable in
                        # at least one indistinguishable branch, so its exact
                        # value is zero.  Substitute that value instead of
                        # creating a binary variable and an equality row.
                        zero_fixed_links += 1
                        continue
                    link_key = (block, predecessor, successor, tuple(cell))
                    # Eq. (24) [eq:chain_set] (compatible pair, same depot, deadhead-feasible),
                    # Eq. (33) [eq:na_chain] with Eq. (26): one psi_{v,v'} per information cell of
                    # the arc's latest departure; Eq. (40) [eq:domain_psi].
                    link = model.addVar(
                        lb=0.0, ub=1.0, vtype=GRB.BINARY,
                        name=(f"chain_link[{block},{predecessor},{successor}|"
                              f"{'-'.join(map(str, cell))}]"))
                    link_vars[link_key] = link
                    for sid in supported:
                        arc = (predecessor, successor)
                        if arc in by_scenario[sid]["link"]:
                            raise ValueError(
                                f"scenario {sid}, arc {arc} belongs to two cells")
                        by_scenario[sid]["link"][arc] = link
                        by_scenario[sid]["link_meta"][arc] = {
                            "decision_time": int(decision_time),
                            "information_cell": tuple(cell),
                        }

        # Create u only where a potentially nonzero incoming psi exists.
        # Otherwise chain_pred is exactly u_v = y_v, so direct substitution
        # removes one binary and one redundant row without changing the model.
        for tid, cell, y_variable in columns_by_block.get(block, ()):
            first_key = (block, tid, cell)
            scenario_ids = []
            has_incoming = False
            for duration in cell:
                scenario = per_duration.get(int(duration))
                if scenario is None:
                    raise ValueError(
                        f"column {(block, tid, cell)} has no duration {duration}")
                sid = str(scenario["id"])
                if tid not in y_by_scenario[sid]:
                    raise ValueError(
                        f"scenario {sid} has no activation for trip {tid}")
                scenario_ids.append(sid)
                has_incoming = has_incoming or any(
                    successor == tid
                    for _predecessor, successor
                    in by_scenario[sid]["link"])
            if has_incoming:
                # Eq. (32) [eq:na_duty]: u_v shares y_v's information cell; Eq. (39) binary.
                first = model.addVar(
                    lb=0.0, ub=1.0, vtype=GRB.BINARY,
                    name=(f"chain_first[{block},{tid}|"
                          f"{'-'.join(map(str, cell))}]"))
                first_vars[first_key] = first
            else:
                first = y_variable
                first_substituted += 1
            first_decisions[first_key] = first
            for sid in scenario_ids:
                if tid in by_scenario[sid]["first"]:
                    raise ValueError(
                        f"scenario {sid}, trip {tid} belongs to two chain cells")
                by_scenario[sid]["first"][tid] = first
                by_scenario[sid]["first_meta"][tid] = {
                    "decision_time": int(trips[tid]["DispatchStart"]),
                    "information_cell": tuple(cell),
                }

        for duration, scenario in per_duration.items():
            sid = str(scenario["id"])
            y = y_by_scenario[sid]
            first = by_scenario[sid]["first"]
            link = by_scenario[sid]["link"]
            incoming = defaultdict(list)
            outgoing = defaultdict(list)
            for (predecessor, successor), variable in link.items():
                incoming[successor].append(variable)
                outgoing[predecessor].append(variable)
            for tid, y_variable in y.items():
                if tid not in first:
                    raise ValueError(
                        f"scenario {sid} has no first-duty variable for trip {tid}")
                incoming_vars = incoming.get(tid, ())
                outgoing_vars = outgoing.get(tid, ())
                # Eq. (34) [eq:chain]: sum_{v'} psi_{v',v} + u_v = y_v (u_v substituted by y_v
                # when trip v has no feasible predecessor: Sec. 3.4.4).
                if first[tid] is not y_variable or incoming_vars:
                    model.addConstr(
                        first[tid] + gp.quicksum(incoming_vars) == y_variable,
                        name=f"chain_pred[{sid},{tid}]")
                    constraint_count += 1
                # Eq. (35) [eq:chain_successor]: sum_{v'} psi_{v,v'} <= y_v.
                if outgoing_vars:
                    model.addConstr(
                        gp.quicksum(outgoing_vars) <= y_variable,
                        name=f"chain_succ[{sid},{tid}]")
                    constraint_count += 1

            # Eq. (36) [eq:depot_vehicles]: sum_{v: g_v = g} u_v <= x_g.
            by_depot = defaultdict(list)
            for tid, variable in first.items():
                by_depot[str(trips[tid]["Depot"])].append(variable)
            for depot, variables in by_depot.items():
                model.addConstr(
                    gp.quicksum(variables) <= fleet_capacity[depot],
                    name=f"chain_fleet[{sid},{depot}]")
                constraint_count += 1
            by_scenario[sid]["successors"] = successors

    totals = {
        "num_vars": len(first_vars) + len(link_vars),
        "num_constraints": constraint_count,
        "physical_arc_count": physical_arc_count,
        "num_links": len(link_vars),
        "num_first": len(first_vars),
        "num_first_decisions": len(first_decisions),
        "first_substituted": first_substituted,
        "zero_fixed_links": zero_fixed_links,
        "decision_scope": "arc_latest_departure_information_cell",
        "binary_auxiliaries": True,
    }
    return by_scenario, totals


def _maximum_trip_matching(successors: Mapping[str, Sequence[str]]) -> dict:
    """Maximum bipartite matching for stable vehicle-duty extraction."""
    match_right = {}

    def augment(left: str, seen: set[str]) -> bool:
        for right in successors.get(left, ()):
            if right in seen:
                continue
            seen.add(right)
            owner = match_right.get(right)
            if owner is None or augment(owner, seen):
                match_right[right] = left
                return True
        return False

    for left in sorted(successors):
        augment(left, set())
    return {left: right for right, left in match_right.items()}


def _vehicle_chain_assignments(
        trips: Mapping[str, Mapping], selected_ids: Iterable[str],
        travel_times: Mapping | None, fleet_counts: Mapping[str, float],
        turnaround_minutes: int = 0, *,
        successor_links: Mapping[str, str] | None = None,
        first_trip_ids: Iterable[str] | None = None) -> dict[str, dict]:
    """Decompose selected trips into closed-home-depot vehicle duties."""
    selected = {str(tid) for tid in selected_ids}
    chosen = {str(tid): trip for tid, trip in trips.items()
              if str(tid) in selected}
    if not chosen:
        return {}
    if travel_times is None:
        raise ValueError("vehicle-chain extraction requires travel_times")
    assignments = {}
    assignment_scope = (
        "information_set_nonanticipative_trip_chain"
        if successor_links is not None else "scenario_endogenous_trip_chain"
    )
    by_depot = defaultdict(dict)
    for tid, trip in chosen.items():
        by_depot[str(trip["Depot"])][tid] = trip

    for depot, depot_trips in sorted(by_depot.items()):
        successors = _compatible_trip_arcs(
            depot_trips, travel_times, turnaround_minutes)
        if successor_links is None:
            successor = _maximum_trip_matching(successors)
        else:
            successor = {
                str(left): str(right)
                for left, right in successor_links.items()
                if str(left) in depot_trips or str(right) in depot_trips
            }
            if any(left not in depot_trips or right not in depot_trips
                   for left, right in successor.items()):
                raise AssertionError(
                    f"chain link crosses home depots for {depot}")
            invalid = [
                (left, right) for left, right in successor.items()
                if right not in successors.get(left, ())
            ]
            if invalid:
                raise AssertionError(f"infeasible selected chain links: {invalid[:3]}")
            if len(set(successor.values())) != len(successor):
                raise AssertionError("two selected predecessors feed one trip")
        predecessor = {right: left for left, right in successor.items()}
        derived_starts = set(depot_trips) - set(predecessor)
        if first_trip_ids is not None:
            declared_starts = {
                str(tid) for tid in first_trip_ids if str(tid) in depot_trips
            }
            if declared_starts != derived_starts:
                raise AssertionError(
                    f"chain first-trip mismatch at {depot}: "
                    f"declared={sorted(declared_starts)} "
                    f"derived={sorted(derived_starts)}")
        starts = sorted(
            derived_starts,
            key=lambda tid: (_trip_run_start(depot_trips[tid]), tid))
        capacity = int(round(float(fleet_counts.get(depot, 0.0))))
        if len(starts) > capacity:
            raise AssertionError(
                f"chain extraction needs {len(starts)} buses at {depot}, "
                f"but x={capacity}")
        visited = set()
        for slot, first_tid in enumerate(starts, start=1):
            chain, tid = [], first_tid
            while tid is not None:
                if tid in visited:
                    raise AssertionError(f"cycle in trip chain at {tid}")
                visited.add(tid)
                chain.append(tid)
                tid = successor.get(tid)
            fleet_id = f"{depot}_BUS_{slot:03d}"
            duty_start = int(depot_trips[chain[0]]["DispatchStart"])
            last_trip = depot_trips[chain[-1]]
            return_deadhead = _travel_minutes(
                travel_times, _trip_end_station(last_trip), depot)
            if return_deadhead is None:
                raise AssertionError(
                    f"last trip {chain[-1]} cannot return to {depot}")
            return_start = _trip_service_end(last_trip)
            duty_end = return_start + return_deadhead
            for position, current in enumerate(chain, start=1):
                trip = depot_trips[current]
                previous = chain[position - 2] if position > 1 else None
                nxt = chain[position] if position < len(chain) else None
                if previous is None:
                    before_from = depot
                    before_start = int(trip["DispatchStart"])
                    before_deadhead = _travel_minutes(
                        travel_times, depot, _trip_start_station(trip))
                else:
                    previous_trip = depot_trips[previous]
                    before_from = _trip_end_station(previous_trip)
                    before_start = (_trip_service_end(previous_trip) +
                                    int(turnaround_minutes))
                    before_deadhead = _travel_minutes(
                        travel_times, before_from, _trip_start_station(trip))
                if before_deadhead is None:
                    raise AssertionError(
                        f"extracted chain has no deadhead {previous}->{current}")
                before_end = before_start + before_deadhead
                is_last = nxt is None
                assignments[current] = {
                    "FleetBusID": fleet_id,
                    "FleetBusIDScope": assignment_scope,
                    "ChainPosition": position,
                    "ChainLength": len(chain),
                    "PreviousTripID": previous,
                    "NextTripID": nxt,
                    "IsFirstTrip": previous is None,
                    "IsLastTrip": is_last,
                    "DeadheadBeforeFrom": before_from,
                    "DeadheadBeforeTo": _trip_start_station(trip),
                    "DeadheadBeforeStart": before_start,
                    "DeadheadBeforeEnd": before_end,
                    "DeadheadBeforeMinutes": before_deadhead,
                    "WaitBeforeRun": max(0, _trip_run_start(trip) - before_end),
                    "ReturnDeadhead": (return_deadhead if is_last else None),
                    "ReturnStart": (return_start if is_last else None),
                    "ReturnEnd": (duty_end if is_last else None),
                    "VehicleDutyStart": duty_start,
                    "VehicleDutyEnd": duty_end,
                }
        if visited != set(depot_trips):
            raise AssertionError(
                f"trip-chain decomposition missed {set(depot_trips) - visited}")
    return assignments


def info_cells(tau: int, disrupt_start: int,
               durations: Sequence[int] = tuple(_bbcfg.get("time.durations_min")),
               buffer: int = DEFAULT_DISRUPT_BUFFER) -> list[tuple[int, ...]]:
    """Return the nonanticipative duration cells visible at dispatch time."""
    durations = tuple(sorted(int(duration) for duration in durations))
    available = tuple(
        duration for duration in durations
        if int(tau) <= int(disrupt_start) + duration + int(buffer)
    )
    if not available:
        return []
    ongoing = tuple(
        duration for duration in available
        if int(tau) <= int(disrupt_start) + duration
    )
    ended = tuple(
        duration for duration in available
        if int(tau) > int(disrupt_start) + duration
    )
    cells = [ongoing] if ongoing else []
    cells.extend((duration,) for duration in ended)
    return cells


def _group_scenarios(scenarios: Sequence[Mapping]) -> dict[str, dict[int, Mapping]]:
    groups: dict[str, dict[int, Mapping]] = defaultdict(dict)
    for scenario in scenarios:
        block = str(scenario.get("block_station", scenario["id"]))
        duration = int(scenario["duration"])
        if duration in groups[block]:
            raise ValueError(
                f"duplicate scenario for block={block}, duration={duration}")
        groups[block][duration] = scenario
    return dict(groups)


def aggregate_bus_load_groups(cache: Mapping,
                              trip_ids: Iterable[str] | None = None) -> dict:
    """Build one continuous, unique-event ledger per ``(trip, omega)``.

    The timeline is always ``min(all board/alight times)..max(...)``.  It is
    not the union of coverage intervals, so disjoint coverages such as
    ``30..60`` and ``80..100`` retain every load minute from 30 through 100.
    Duplicate b/a events are stored as sets and therefore counted once.
    """
    allowed = None if trip_ids is None else {str(tid) for tid in trip_ids}
    # A compacted full-pool cache can remove provably-zero b/a keys while
    # retaining the original ``trip_od_list`` lookup.  In that case the load
    # ledger must be built from the retained event support, otherwise removed
    # coverages recreate millions of zero-only load minutes.  Tiny synthetic
    # audit caches that contain only ``trip_od_list`` keep the historical
    # behaviour.
    has_event_support = ("b_var_keys" in cache or "a_var_keys" in cache)
    compacted = bool(cache.get("support_compacted"))
    retained_b = {
        (key if compacted else
         (key[0], str(key[1]), int(key[2]), key[3]))
        for key in cache.get("b_var_keys", ())
        if allowed is None or str(key[1]) in allowed
    }
    retained_a = {
        (key if compacted else
         (key[0], str(key[1]), int(key[2]), key[3]))
        for key in cache.get("a_var_keys", ())
        if allowed is None or str(key[1]) in allowed
    }

    raw: dict[tuple[str, str], dict] = {}
    for raw_tid, od_list in cache.get("trip_od_list", {}).items():
        tid = str(raw_tid)
        if allowed is not None and tid not in allowed:
            continue
        for od_info in od_list:
            om = od_info["omega"]
            board_t = int(od_info["board_t"])
            alight_t = int(od_info["alight_t"])
            board_j = od_info["board_j"]
            alight_j = od_info["alight_j"]
            board_key = (om, tid, board_t, board_j)
            alight_key = (om, tid, alight_t, alight_j)
            keep_board = not has_event_support or board_key in retained_b
            keep_alight = not has_event_support or alight_key in retained_a
            if not keep_board and not keep_alight:
                continue
            event_times = ([board_t] if keep_board else []) + (
                [alight_t] if keep_alight else [])
            group = raw.setdefault((tid, om), {
                "first": min(event_times),
                "last": max(event_times),
                "board": defaultdict(set),
                "alight": defaultdict(set),
            })
            group["first"] = min(group["first"], *event_times)
            group["last"] = max(group["last"], *event_times)
            if keep_board:
                group["board"][board_t].add(board_key)
            if keep_alight:
                group["alight"][alight_t].add(alight_key)

    groups: dict[str, dict] = defaultdict(dict)
    for (tid, om), group in raw.items():
        groups[tid][om] = {
            # ``range`` preserves the exact inclusive minute ledger and all
            # sequence operations used by model construction, while retaining
            # only its two bounds instead of one Python integer reference per
            # minute and per (trip, OD) stream.
            "times": range(group["first"], group["last"] + 1),
            "board": {
                minute: tuple(sorted(keys, key=repr))
                for minute, keys in group["board"].items()
            },
            "alight": {
                minute: tuple(sorted(keys, key=repr))
                for minute, keys in group["alight"].items()
            },
        }
    return {tid: dict(by_omega) for tid, by_omega in groups.items()}


def _termination_alight_keys(scenario: Mapping) -> frozenset[tuple[str, int, str]]:
    """Return normalized forced-detrain events without an OD component."""
    result = set()
    for raw_vehicle, raw_event in scenario.get(
            "termination_alight_event_by_train", {}).items():
        if isinstance(raw_event, Mapping):
            minute = int(raw_event["minute"])
            station = str(raw_event["station"])
        else:
            minute, station = raw_event
            minute, station = int(minute), str(station)
        result.add((str(raw_vehicle), minute, station))
    return frozenset(result)


def _passenger_action_support_termination_keys(
        scenario: Mapping) -> frozenset[tuple[str, int, str]]:
    """Return union support events without changing physical termination."""
    result = set(_termination_alight_keys(scenario))
    for raw_event in scenario.get(
            "passenger_action_support_termination_events", ()):
        if isinstance(raw_event, Mapping):
            vehicle = str(raw_event["vehicle"])
            minute = int(raw_event["minute"])
            station = str(raw_event["station"])
        else:
            vehicle, minute, station = raw_event
            vehicle, minute, station = (
                str(vehicle), int(minute), str(station))
        result.add((vehicle, minute, station))
    return frozenset(result)


def _physical_train_boarding_forbidden(
        scenario: Mapping, vehicle: str, minute: int) -> bool:
    """Whether this raw branch has withdrawn a train at the event minute."""
    return bool(
        not str(vehicle).startswith("B_")
        and int(scenario.get("theta", {}).get(
            (str(vehicle), int(minute)), 0))
    )


def _forbidden_origin_alight(
        key: tuple, omega_map: Mapping,
        termination_event_keys: frozenset[tuple[str, int, str]]) -> bool:
    """Whether an alighting action is an ordinary prohibited origin exit."""
    om, raw_vehicle, raw_minute, station = key
    return bool(
        om in omega_map and station == omega_map[om][0] and
        (str(raw_vehicle), int(raw_minute), str(station)) not in
        termination_event_keys
    )


def compact_full_pool_cache(cache: Mapping, scenario: Mapping,
                            wdata: Mapping, *, sigma: int = SIGMA,
                            eps: float = SUPPORT_EPS,
                            preserve_trip_ids: bool = True) -> dict:
    """Return a conservative exact support reduction for a *full* trip pool.

    This routine is intentionally independent of the active CG columns.  The
    caller must pass a cache constructed from the complete candidate-trip
    union for the scenario.  It removes only variables with a proved zero
    upper bound or state prefixes before any possible event.  Queue states are
    kept continuously from their first possible event through the horizon;
    event-free minutes in the middle are never removed.

    ``w`` has a finite ``sigma``-minute impulse response in this formulation,
    so its exact support is the union of ``[alight, alight + sigma)``.  Bus load
    keys may be omitted by the upstream cache builder and are reconstructed
    lazily by :func:`aggregate_bus_load_groups` on a continuous time line.
    """
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    compact = dict(cache)
    # Do not clone every large key family before filtering it.  Each retained
    # family below is built directly from the input, so the old implementation
    # briefly held two complete Python lists for b/a/l/q/w/od for no semantic
    # benefit.  ``trip_ids`` is the only family not unconditionally rebuilt.
    compact["trip_ids"] = list(cache.get("trip_ids", ()))

    omega_map = wdata["Omega_Map"]
    cumulative = wdata["CumD"]
    demand = wdata["D"]
    bus_ids = {str(tid) for tid in scenario.get("bus_trips", {})}
    termination_event_keys = (
        _passenger_action_support_termination_keys(scenario))

    b_keys = []
    for raw_key in cache.get("b_var_keys", ()):
        om, raw_vehicle, raw_minute, station = raw_key
        vehicle, minute = str(raw_vehicle), int(raw_minute)
        destination = omega_map.get(om, (None, None))[1]
        if station == destination:
            continue
        if float(cumulative.get((om, minute), 0.0)) <= eps:
            continue
        b_keys.append((om, vehicle, minute, station))

    a_keys = []
    for raw_key in cache.get("a_var_keys", ()):
        om, raw_vehicle, raw_minute, station = raw_key
        vehicle, minute = str(raw_vehicle), int(raw_minute)
        # Ordinary alighting at an OD's origin is dominated.  A forced
        # detrain at a withdrawn train's last safe stop is different: riders
        # may have boarded there before disruption onset and must leave the
        # train before its passenger capacity becomes zero.
        if _forbidden_origin_alight(
                (om, vehicle, minute, station), omega_map,
                termination_event_keys):
            continue
        if float(cumulative.get((om, minute), 0.0)) <= eps:
            continue
        a_keys.append((om, vehicle, minute, station))

    od_keys = [
        (om, int(minute))
        for om, minute in cache.get("od_var_keys", ())
        if float(cumulative.get((om, int(minute)), 0.0)) > eps
    ]

    active_vehicle_pairs = {(om, vehicle) for om, vehicle, _, _ in b_keys}
    active_vehicle_pairs.update((om, vehicle) for om, vehicle, _, _ in a_keys)
    l_keys = []
    for raw_key in cache.get("l_var_keys", ()):
        om, raw_vehicle, raw_minute = raw_key
        vehicle, minute = str(raw_vehicle), int(raw_minute)
        # Bus inventories are rebuilt below from trip_od_list as one lazy,
        # continuous ledger.  Keeping a partial materialized bus timeline is
        # unnecessary and makes it easier to reintroduce a gap.
        if vehicle in bus_ids:
            continue
        if (om, vehicle) not in active_vehicle_pairs:
            continue
        # A train load is a state, not an event variable.  Once an (OD,train)
        # pair can board/alight, retain its original complete active timeline.
        # Cutting the zero-demand prefix would remove the initial recurrence
        # and leave the first retained load disconnected from b/a.
        l_keys.append((om, vehicle, minute))

    demand_times = defaultdict(list)
    board_times = defaultdict(list)
    alight_times = defaultdict(list)
    for (om, station, raw_minute), value in demand.items():
        if abs(float(value)) > eps:
            demand_times[(om, station)].append(int(raw_minute))
    for om, _vehicle, minute, station in b_keys:
        board_times[(om, station)].append(minute)
    for om, _vehicle, minute, station in a_keys:
        alight_times[(om, station)].append(minute)

    first_q_event = {}
    w_support = defaultdict(set)
    for om in scenario.get("Omega_m", ()):
        if om not in omega_map:
            continue
        for station in cache.get("N_omega", {}).get(om, set()):
            node = (om, station)
            events = list(demand_times.get(node, ()))
            events.extend(board_times.get(node, ()))
            events.extend(
                minute + int(sigma) for minute in alight_times.get(node, ()))
            if events:
                first_q_event[node] = min(events)
            for minute in alight_times.get(node, ()):
                w_support[node].update(
                    range(minute, minute + int(sigma)))

    q_keys = [
        (om, station, int(minute))
        for om, station, minute in cache.get("q_var_keys", ())
        if (om, station) in first_q_event
        and int(minute) >= first_q_event[(om, station)]
    ]
    w_keys = [
        (om, station, int(minute))
        for om, station, minute in cache.get("w_var_keys", ())
        if int(minute) in w_support.get((om, station), set())
    ]

    if not preserve_trip_ids:
        active_bus_ids = {
            vehicle for _om, vehicle, _minute, _station in b_keys
            if vehicle in bus_ids
        }
        compact["trip_ids"] = [
            tid for tid in compact["trip_ids"] if str(tid) in active_bus_ids]
    compact["b_var_keys"] = b_keys
    compact["a_var_keys"] = a_keys
    compact["l_var_keys"] = l_keys
    compact["q_var_keys"] = q_keys
    compact["w_var_keys"] = w_keys
    compact["od_var_keys"] = od_keys
    compact["support_basis"] = "full_candidate_pool_union"
    compact["support_compacted"] = True
    return compact


def prepare_exact_state_support(
        full_caches: Mapping[str, Mapping], scenarios: Sequence[Mapping],
        wdata: Mapping, config: CGConfig | None = None, *,
        release_source_keys: bool = False) -> dict[str, dict]:
    """Replace explicit l/q/w key grids by exact immutable metadata.

    The returned caches shallow-share all unaffected support families.  With
    ``release_source_keys=True`` each mutable source cache drops its large
    l/q/w lists immediately after that scenario's immutable metadata is ready;
    model builders consume ``_queue_compression``/``_walking_plan`` directly.
    """
    config = _cfg(config)
    scenario_by_id = {str(scenario["id"]): scenario for scenario in scenarios}
    prepared = {}
    for raw_sid, raw_cache in full_caches.items():
        sid = str(raw_sid)
        if sid not in scenario_by_id:
            raise ValueError(f"state support has no scenario metadata for {sid}")
        cache = dict(raw_cache)
        if config.eliminate_destination_states:
            q_reference_keys = tuple(cache.get("q_var_keys", ()))
            retained_q_keys = tuple(
                key for key in q_reference_keys
                if key[1] != wdata["Omega_Map"].get(key[0], (None, None))[1])
            od_reference_keys = tuple(cache.get("od_var_keys", ()))
            cache["q_reference_key_count"] = len(q_reference_keys)
            cache["q_retained_reference_key_count"] = len(retained_q_keys)
            cache["destination_q_eliminated_count"] = (
                len(q_reference_keys) - len(retained_q_keys))
            cache["od_reference_key_count"] = len(od_reference_keys)
            cache["od_eliminated_count"] = len(od_reference_keys)
            cache["q_var_keys"] = retained_q_keys
            cache["od_var_keys"] = ()
            cache.pop("_queue_compression", None)
            if release_source_keys:
                try:
                    raw_cache["od_var_keys"] = ()
                except TypeError as error:
                    raise TypeError(
                        "release_source_keys requires mutable scenario caches"
                    ) from error
        if config.compress_load_states:
            train_segments = cache.get("_train_load_segments")
            if train_segments is None:
                train_segments, _board, _alight = _compressed_load_plan(
                    cache.get("l_var_keys", ()),
                    cache.get("b_var_keys", ()), cache.get("a_var_keys", ()),
                    {}, wdata["station_of"])
            cache["_train_load_segments"] = tuple(train_segments)
            cache["l_reference_key_count"] = len(
                cache.get("l_var_keys", ()))
            cache["l_var_keys"] = ()
            if release_source_keys:
                try:
                    raw_cache["l_var_keys"] = ()
                except TypeError as error:
                    raise TypeError(
                        "release_source_keys requires mutable scenario caches"
                    ) from error
        if config.compress_queue_states:
            queue = cache.get("_queue_compression")
            if queue is None:
                queue = build_queue_compression(
                    cache.get("q_var_keys", ()), cache.get("b_var_keys", ()),
                    cache.get("a_var_keys", ()), wdata["D"],
                    wdata["Omega_Map"], config.sigma)
            cache["_queue_compression"] = queue
            cache.setdefault(
                "q_reference_key_count", int(queue.original_key_count))
            cache["q_retained_reference_key_count"] = int(
                queue.original_key_count)
            cache["q_var_keys"] = ()
            if release_source_keys:
                try:
                    raw_cache["q_var_keys"] = ()
                except TypeError as error:
                    raise TypeError(
                        "release_source_keys requires mutable scenario caches"
                    ) from error
        if config.eliminate_w:
            walking = cache.get("_walking_plan")
            if walking is None:
                walking = build_walking_elimination_plan(
                    w_rows=cache.get("w_var_keys", ()),
                    alighting_keys=cache.get("a_var_keys", ()),
                    omega_map=wdata["Omega_Map"], gamma=wdata["Gamma"],
                    sigma=config.sigma)
            cache["_walking_plan"] = walking
            cache["w_reference_key_count"] = sum(
                stop - start
                for runs in walking.row_runs.values()
                for start, stop in runs)
            cache["w_var_keys"] = ()
            if release_source_keys:
                try:
                    raw_cache["w_var_keys"] = ()
                except TypeError as error:
                    raise TypeError(
                        "release_source_keys requires mutable scenario caches"
                    ) from error
        prepared[sid] = cache
    return prepared


# Backward-friendly name for audits, while remaining entirely local.
_aggregate_bus_load_groups = aggregate_bus_load_groups


def build_column_pool(problem: CGProblem,
                      config: CGConfig | None = None) -> tuple[ColumnSpec, ...]:
    """Enumerate canonical ``(block, trip, information-cell)`` metadata."""
    config = _cfg(config)
    full_caches = _require_full_caches(problem)
    b_count_by_sid = {}
    b_upper_by_sid = {}
    cumulative = problem.wdata["CumD"]
    for sid, cache in full_caches.items():
        counts = defaultdict(int)
        upper = defaultdict(float)
        for om, vehicle, minute, _station in cache.get("b_var_keys", ()):
            if str(vehicle).startswith("B_"):
                counts[str(vehicle)] += 1
                # This is the sum of the exact individual variable upper
                # bounds used by solve_master.  It may overestimate feasible
                # total boarding, but can never underestimate it, so
                # B_v < epsilon is a safe exact fixing of y_v to zero.
                upper[str(vehicle)] += max(float(cumulative.get(
                    (om, int(minute)), 0.0)), 0.0)
        b_count_by_sid[str(sid)] = counts
        b_upper_by_sid[str(sid)] = upper

    pool = []
    for block, per_duration in sorted(_group_scenarios(problem.scenarios).items()):
        causal_passenger_na = any(
            "passenger_shared_start" in scenario or
            "passenger_shared_end" in scenario
            for scenario in per_duration.values())
        durations = tuple(sorted(per_duration))
        starts = {int(s["DISRUPT_START_T"]) for s in per_duration.values()}
        if len(starts) != 1:
            raise ValueError(f"block={block}: inconsistent disruption starts {starts}")
        disrupt_start = starts.pop()
        trips = {}
        for scenario in per_duration.values():
            trips.update({str(tid): trip
                          for tid, trip in scenario.get("bus_trips", {}).items()})

        for tid, trip in sorted(trips.items()):
            for cell in info_cells(
                    int(trip["DispatchStart"]), disrupt_start, durations,
                    config.disrupt_buffer):
                scenario_ids = []
                score = 0
                minimum_boarding_upper = (
                    0.0 if causal_passenger_na else math.inf)
                for duration in cell:
                    scenario = per_duration.get(duration)
                    if scenario is None or tid not in {
                            str(key) for key in scenario.get("bus_trips", {})}:
                        raise ValueError(
                            f"column {(block, tid, cell)} is unavailable in "
                            f"duration {duration}")
                    sid = str(scenario["id"])
                    cache = full_caches[sid]
                    if tid not in {str(key) for key in cache.get("trip_ids", ())}:
                        raise ValueError(
                            f"full cache {sid} does not contain trip {tid}")
                    scenario_ids.append(sid)
                    score += b_count_by_sid[sid].get(tid, 0)
                    branch_upper = b_upper_by_sid[sid].get(tid, 0.0)
                    if causal_passenger_na:
                        # The causal filter is one row per information cell.
                        # Summing branch upper bounds may double-count a shared
                        # action, which is a safe overestimate for pruning: it
                        # can retain an unnecessary column but never remove a
                        # feasible one.
                        minimum_boarding_upper += branch_upper
                    else:
                        minimum_boarding_upper = min(
                            minimum_boarding_upper, branch_upper)
                tolerance = 1e-9 * max(1.0, config.min_trip_boarding)
                if (config.min_trip_boarding <= 0.0 or
                        minimum_boarding_upper + tolerance >=
                        config.min_trip_boarding):
                    pool.append(ColumnSpec(
                        key=(block, tid, tuple(cell)),
                        scenario_ids=tuple(scenario_ids), score=score))
    keys = [column.key for column in pool]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate trip-cell keys in canonical pool")
    return tuple(pool)


def count_macro_columns(problem: CGProblem,
                        config: CGConfig | None = None) -> int:
    """Count canonical macro-columns without inspecting flow keys."""
    config = _cfg(config)
    count = 0
    for block, per_duration in _group_scenarios(problem.scenarios).items():
        durations = tuple(sorted(per_duration))
        starts = {int(s["DISRUPT_START_T"]) for s in per_duration.values()}
        if len(starts) != 1:
            raise ValueError(f"block={block}: inconsistent starts {starts}")
        disrupt_start = starts.pop()
        trips = {}
        for scenario in per_duration.values():
            trips.update({str(tid): trip
                          for tid, trip in scenario.get("bus_trips", {}).items()})
        for trip in trips.values():
            count += len(info_cells(
                int(trip["DispatchStart"]), disrupt_start, durations,
                config.disrupt_buffer))
    return count


def build_depot_row_keys(
        problem: CGProblem,
        config: CGConfig | None = None) -> frozenset[tuple[str, str, int]]:
    """Return the exact interval-start depot-row universe.

    For a fixed ``(scenario, depot)``, every trip contributes a nonnegative
    ``y`` to each minute in one clipped half-open interval and every overlap
    row has the same right-hand side ``x[depot]``.  At an arbitrary minute
    ``t``, choose the latest clipped interval start among trips active at
    ``t``.  Every trip active at ``t`` was already active at that start, so the
    active set at ``t`` is a subset of the start-row active set.  Nonnegative
    ``y`` and the common RHS therefore make that start row dominate the minute
    row.  Empty rows are dominated by ``x >= 0``.  Retaining every distinct
    clipped interval start is consequently exact, not a relaxation.
    """
    config = _cfg(config)
    gamma = problem.wdata["Gamma"]
    if not gamma:
        return frozenset()
    first, after_last = int(gamma[0]), int(gamma[-1]) + 1
    rows = set()
    for scenario in problem.scenarios:
        sid = str(scenario["id"])
        for trip in scenario.get("bus_trips", {}).values():
            raw_start, raw_end = _depot_interval(trip, config)
            start = max(raw_start, first)
            end = min(raw_end, after_last)
            if start < end:
                rows.add((sid, str(trip["Depot"]), start))
    return frozenset(rows)


def _index_depot_row_minutes(
        row_keys: Iterable[tuple[str, str, int]]) -> dict[tuple[str, str], tuple[int, ...]]:
    """Index immutable depot-row minutes for logarithmic interval slicing."""
    grouped = defaultdict(set)
    for sid, depot, minute in row_keys:
        grouped[(str(sid), str(depot))].add(int(minute))
    return {
        key: tuple(sorted(minutes)) for key, minutes in grouped.items()
    }


def _depot_minutes_in_interval(
        row_index: Mapping[tuple[str, str], Sequence[int]], sid: str,
        depot: str, begin: int, end: int) -> tuple[int, ...]:
    """Return retained rows in ``[begin, end)`` without scanning its minutes."""
    minutes = row_index.get((str(sid), str(depot)), ())
    left = bisect_left(minutes, int(begin))
    right = bisect_left(minutes, int(end), lo=left)
    return tuple(minutes[left:right])


def seed_columns(pool: Iterable[ColumnSpec], per_block: int) -> set[ColumnKey]:
    """Select high-support columns without breaking information cells."""
    if per_block <= 0:
        return set()
    grouped = defaultdict(list)
    for column in pool:
        grouped[column.block].append(column)
    selected = set()
    for columns in grouped.values():
        ranked = sorted(columns, key=lambda column: (-column.score, column.key))
        selected.update(column.key for column in ranked[:per_block])
    return selected


def _trip_lookup(scenario: Mapping, tid: str) -> Mapping | None:
    trips = scenario.get("bus_trips", {})
    if tid in trips:
        return trips[tid]
    for raw_tid, trip in trips.items():
        if str(raw_tid) == tid:
            return trip
    return None


def _build_shared_y(model: gp.Model, problem: CGProblem, config: CGConfig,
                    vtype, active_columns: set[ColumnKey] | None,
                    column_ub: Mapping[ColumnKey, float]) -> tuple[dict, dict]:
    """Create each nonanticipative y exactly once and map it to scenarios."""
    by_scenario: dict[str, dict[str, gp.Var]] = {
        str(s["id"]): {} for s in problem.scenarios
    }
    by_column: dict[ColumnKey, gp.Var] = {}
    for block, per_duration in _group_scenarios(problem.scenarios).items():
        durations = tuple(sorted(per_duration))
        starts = {int(s["DISRUPT_START_T"]) for s in per_duration.values()}
        if len(starts) != 1:
            raise ValueError(f"block={block}: inconsistent disruption starts")
        disrupt_start = starts.pop()
        trips = {}
        for scenario in per_duration.values():
            trips.update({str(tid): trip
                          for tid, trip in scenario.get("bus_trips", {}).items()})
        for tid, trip in sorted(trips.items()):
            for cell in info_cells(
                    int(trip["DispatchStart"]), disrupt_start, durations,
                    config.disrupt_buffer):
                key = (str(block), tid, tuple(cell))
                if active_columns is not None and key not in active_columns:
                    continue
                variable = model.addVar(
                    lb=0.0, ub=float(column_ub.get(key, 1.0)), vtype=vtype,
                    name=f"y[{block},{tid}|{'-'.join(map(str, cell))}]")
                by_column[key] = variable
                for duration in cell:
                    scenario = per_duration.get(duration)
                    if scenario is None or _trip_lookup(scenario, tid) is None:
                        raise ValueError(
                            f"column {key} is unavailable in duration {duration}")
                    sid = str(scenario["id"])
                    if tid in by_scenario[sid]:
                        raise ValueError(
                            f"scenario {sid}, trip {tid} belongs to two cells")
                    by_scenario[sid][tid] = variable
    return by_scenario, by_column


def _safe_model_attr(model: gp.Model, name: str, default=math.nan):
    try:
        return getattr(model, name)
    except (AttributeError, gp.GurobiError):
        return default


def _model_stats(model: gp.Model, build_seconds: float,
                 aggregate: bool = True) -> dict:
    has_solution = int(model.SolCount) > 0
    status_code = int(_safe_model_attr(model, "Status", 0))
    is_mip = bool(_safe_model_attr(model, "IsMIP", False))
    stats = {
        "ObjVal": float(model.ObjVal) if has_solution else math.nan,
        # ObjBound remains useful when a time limit/interruption occurs before
        # an incumbent is found, so it is intentionally independent of
        # SolCount.
        "ObjBound": float(_safe_model_attr(model, "ObjBound")),
        "MIPGap": (float(model.MIPGap)
                   if has_solution and is_mip else 0.0),
        "SolCount": int(model.SolCount),
        "StatusCode": status_code,
        "Status": _status_name(status_code),
        "Runtime": float(_safe_model_attr(model, "Runtime", 0.0)),
        "BuildSec": float(build_seconds),
        "NumVars": int(model.NumVars),
        "NumConstrs": int(model.NumConstrs),
        "NumNZs": int(_safe_model_attr(model, "NumNZs", 0)),
        "NumIntVars": int(model.NumIntVars),
        "NumBinVars": int(_safe_model_attr(model, "NumBinVars", 0)),
        "NodeCount": float(_safe_model_attr(model, "NodeCount", 0.0)),
        "IterCount": float(_safe_model_attr(model, "IterCount", 0.0)),
        "BarIterCount": int(_safe_model_attr(model, "BarIterCount", 0)),
        "Work": float(_safe_model_attr(model, "Work", 0.0)),
        "IgnoreNames": int(model.Params.IgnoreNames),
        "Method": int(model.Params.Method),
        "LPWarmStart": int(model.Params.LPWarmStart),
        "OptimalityTol": float(model.Params.OptimalityTol),
        "FeasibilityTol": float(model.Params.FeasibilityTol),
        "AggregateBusLoadBalance": bool(aggregate),
        "BusTerminalEmpty": True,
    }
    return stats


_PASSENGER_ACTION_FAMILIES = ("b", "a", "w", "od")


def _passenger_key_minute(family: str, key: tuple) -> int:
    """Return the decision/state minute encoded by a passenger-flow key."""
    if family in ("b", "a", "l", "q", "w"):
        return int(key[2])
    if family == "od":
        return int(key[1])
    raise KeyError(family)


def _has_passenger_na_metadata(scenarios: Sequence[Mapping]) -> bool:
    """Whether strict passenger-sharing metadata was explicitly supplied."""
    return any(
        "passenger_shared_start" in scenario or
        "passenger_shared_end" in scenario
        for scenario in scenarios)


def _build_passenger_sharing_contexts(
        scenarios: Sequence[Mapping]) -> tuple[dict[str, dict], dict[str, dict]]:
    """Validate and index one passenger-variable registry per location block.

    Legacy inputs contain none of these fields and retain their former
    scenario-wise recourse. Once either shared-interval field is supplied,
        every duration in that location block must supply the complete strict
    metadata; a partly upgraded block is rejected rather than silently mixed.
    """
    by_sid: dict[str, dict] = {}
    by_block: dict[str, dict] = {}
    required = {
        "common_omega", "normal_fixed_omega", "normal_trajectory",
        "passenger_shared_start", "passenger_shared_end",
    }
    for block, per_duration in _group_scenarios(scenarios).items():
        enabled = any(
            "passenger_shared_start" in scenario or
            "passenger_shared_end" in scenario
            for scenario in per_duration.values())
        if not enabled:
            continue
        for duration, scenario in per_duration.items():
            missing = sorted(required - set(scenario))
            if missing:
                raise ValueError(
                    f"block={block}, duration={duration}: incomplete strict "
                    f"passenger metadata: {missing}")
        starts = {int(s["passenger_shared_start"])
                  for s in per_duration.values()}
        ends = {int(s["passenger_shared_end"])
                for s in per_duration.values()}
        common_sets = {
            frozenset(s.get("common_omega", ()))
            for s in per_duration.values()
        }
        if len(starts) != 1 or len(ends) != 1:
            raise ValueError(
                f"block={block}: inconsistent passenger shared interval")
        start, end = starts.pop(), ends.pop()
        if end < start:
            raise ValueError(
                f"block={block}: passenger shared interval is reversed")
        if len(common_sets) != 1:
            raise ValueError(
                f"block={block}: common passenger OD support differs by "
                "duration")
        context = {
            "block": str(block),
            "start": int(start),
            "end": int(end),
            "common": common_sets.pop(),
            "durations": tuple(sorted(map(int, per_duration))),
            "expected_sids": frozenset(
                str(scenario["id"]) for scenario in per_duration.values()),
            "registry": {},
            "fixed_registry": {},
            "normal_fixed_count": 0,
        }
        by_block[str(block)] = context
        for scenario in per_duration.values():
            by_sid[str(scenario["id"])] = context
    return by_sid, by_block


def _is_shared_passenger_key(
        family: str, key: tuple, context: Mapping | None) -> bool:
    if context is None or len(context["durations"]) < 2:
        return False
    return (
        key[0] in context["common"] and
        int(context["start"]) <= _passenger_key_minute(family, key) <
        int(context["end"])
    )


def validate_shared_passenger_action_cache_support(
        full_caches: Mapping[str, Mapping],
        scenarios: Sequence[Mapping]) -> dict:
    """Validate literal pre-revelation b/a support before model creation."""
    by_sid, by_block = _build_passenger_sharing_contexts(scenarios)
    scenarios_by_id = {
        str(scenario["id"]): scenario for scenario in scenarios}
    audit = {}
    for block, context in by_block.items():
        expected_sids = tuple(sorted(context["expected_sids"]))
        if any(sid not in full_caches for sid in expected_sids):
            missing = [sid for sid in expected_sids if sid not in full_caches]
            raise ValueError(
                f"block={block}: passenger cache missing scenarios {missing}")
        counts = {}
        for family, cache_field in (("b", "b_var_keys"),
                                    ("a", "a_var_keys")):
            support_by_sid = {}
            for sid in expected_sids:
                keys = {
                    (key[0], str(key[1]), int(key[2]), key[3])
                    for key in full_caches[sid].get(cache_field, ())
                    if _is_shared_passenger_key(
                        family,
                        (key[0], str(key[1]), int(key[2]), key[3]),
                        by_sid[sid],
                    )
                }
                support_by_sid[sid] = keys
            reference_sid = expected_sids[0]
            reference = support_by_sid[reference_sid]
            for sid in expected_sids[1:]:
                other = support_by_sid[sid]
                if other != reference:
                    only_reference = sorted(
                        reference - other, key=repr)[:5]
                    only_other = sorted(
                        other - reference, key=repr)[:5]
                    raise ValueError(
                        "pre-revelation passenger cache support differs; "
                        f"block={block}, family={family}, "
                        f"only {reference_sid}={only_reference}, "
                        f"only {sid}={only_other}"
                    )
            counts[family] = len(reference)
        audit[str(block)] = counts
    return {
        "passed": True,
        "information_block_count": len(audit),
        "shared_action_keys_by_block": audit,
    }


def _value_for_key(value, key, default):
    return value.get(key, default) if isinstance(value, Mapping) else value


def _batch_add_passenger_vars(
        model: gp.Model, keys: Iterable[tuple], *, family: str, sid: str,
        context: dict | None, lb=0.0, ub=GRB.INFINITY, obj=0.0,
        name: str, ignore_names: bool = False) -> dict:
    """Create branch-local actions and reuse actual Vars before revelation."""
    keys = tuple(keys)
    if context is None:
        return _batch_add_vars(
            model, keys, lb=lb, ub=ub, obj=obj, name=name,
            ignore_names=ignore_names)

    result = {}
    local_keys = []
    new_shared_keys = []
    for key in keys:
        if not _is_shared_passenger_key(family, key, context):
            local_keys.append(key)
            continue
        registry_key = (family, key)
        key_lb = float(_value_for_key(lb, key, 0.0))
        key_ub = float(_value_for_key(ub, key, GRB.INFINITY))
        entry = context["registry"].get(registry_key)
        if entry is None:
            new_shared_keys.append(key)
            continue
        finite_mismatch = (
            (math.isfinite(key_ub) or math.isfinite(entry["ub"])) and
            abs(float(entry["ub"]) - key_ub) > 1e-9)
        if abs(float(entry["lb"]) - key_lb) > 1e-9 or finite_mismatch:
            raise ValueError(
                "shared passenger variable has inconsistent bounds: "
                f"block={context['block']}, key={registry_key}")
        entry["sids"].add(str(sid))
        result[key] = entry["variable"]

    if new_shared_keys:
        shared_lb = {
            key: float(_value_for_key(lb, key, 0.0))
            for key in new_shared_keys}
        shared_ub = {
            key: float(_value_for_key(ub, key, GRB.INFINITY))
            for key in new_shared_keys}
        # _batch_add_vars accepts only a scalar LB. Passenger variables use
        # the same zero lower bound in the executable formulation.
        if len(set(shared_lb.values())) != 1:
            raise ValueError("shared passenger variables require uniform LB")
        created = _batch_add_vars(
            model, tuple(new_shared_keys), lb=next(iter(shared_lb.values())),
            ub=shared_ub, obj=0.0, name=f"{family}_shared",
            ignore_names=ignore_names)
        for key, variable in created.items():
            context["registry"][(family, key)] = {
                "variable": variable, "sids": {str(sid)},
                "lb": shared_lb[key], "ub": shared_ub[key],
            }
        result.update(created)

    if local_keys:
        local_lb_values = {
            float(_value_for_key(lb, key, 0.0)) for key in local_keys}
        if len(local_lb_values) != 1:
            raise ValueError("passenger variables require uniform local LB")
        local_ub = ({key: _value_for_key(ub, key, GRB.INFINITY)
                     for key in local_keys}
                    if isinstance(ub, Mapping) else ub)
        local_obj = ({key: _value_for_key(obj, key, 0.0)
                      for key in local_keys}
                     if isinstance(obj, Mapping) else obj)
        result.update(_batch_add_vars(
            model, tuple(local_keys), lb=next(iter(local_lb_values)),
            ub=local_ub, obj=local_obj, name=name,
            ignore_names=ignore_names))

    # A shared variable can carry a different probability-weighted objective
    # coefficient in each duration. Record every term; callers that use Var
    # objective attributes add the accumulated expression after all branches
    # have been constructed. solve_master and pricing rebuild objectives from
    # expressions and intentionally ignore this list.
    objective_terms = context.setdefault("objective_terms", [])
    for key, variable in result.items():
        if _is_shared_passenger_key(family, key, context):
            coefficient = float(_value_for_key(obj, key, 0.0))
            if coefficient:
                objective_terms.append(coefficient * variable)
    return result


def _normalise_normal_family(family: str, values: Mapping) -> dict:
    result = {}
    for raw_key, value in values.items():
        key = tuple(raw_key)
        if family in ("b", "a"):
            key = (key[0], str(key[1]), int(key[2]), key[3])
        elif family in ("q", "w"):
            key = (key[0], key[1], int(key[2]))
        elif family == "l":
            key = (key[0], str(key[1]), int(key[2]))
        elif family == "od":
            key = (key[0], int(key[1]))
        result[key] = float(value)
    return result


def _fix_normal_passenger_decisions(
        scenario: Mapping, families: Mapping[str, Mapping],
        context: dict | None) -> int:
    """Fix pre-disruption and branch-unaffected passenger actions."""
    if context is None:
        return 0
    normal = scenario.get("normal_trajectory")
    if not normal:
        raise ValueError(
            f"{scenario['id']}: strict passenger metadata has no normal "
            "trajectory")
    common = set(context["common"])
    fixed_omegas = set(scenario.get("normal_fixed_omega", ()))
    disruption_start = int(scenario["DISRUPT_START_T"])
    count = 0
    for family in _PASSENGER_ACTION_FAMILIES:
        variables = families.get(family, {})
        normal_values = _normalise_normal_family(
            family, normal.get(family, {}))
        for key, variable in variables.items():
            omega = key[0]
            minute = _passenger_key_minute(family, key)
            if not (omega in fixed_omegas or
                    (omega in common and minute < disruption_start)):
                continue
            value = float(normal_values.get(key, 0.0))
            marker = id(variable)
            previous = context["fixed_registry"].get(marker)
            if previous is not None and abs(previous - value) > 1e-8:
                raise ValueError(
                    f"inconsistent normal fix for shared {family} key "
                    f"{key}: {previous} vs {value}")
            variable.LB = value
            variable.UB = value
            if previous is None:
                context["fixed_registry"][marker] = value
                count += 1

        # An explicit family must contain every positive action needed by the
        # claimed normal trajectory. Eliminated w/od families are empty and
        # are therefore checked through their exact projection instead.
        if variables:
            for key, value in normal_values.items():
                omega = key[0]
                minute = _passenger_key_minute(family, key)
                required = (omega in fixed_omegas or
                            (omega in common and minute < disruption_start))
                if required and value > 1e-9 and key not in variables:
                    raise ValueError(
                        f"normal {family} key missing from {scenario['id']} "
                        f"support: {key}")
    context["normal_fixed_count"] += int(count)
    return count


def _validate_passenger_variable_sharing(contexts: Mapping[str, Mapping]) -> dict:
    by_family = defaultdict(int)
    total = 0
    for context in contexts.values():
        expected = set(context["expected_sids"])
        bad = [
            (key, tuple(sorted(entry["sids"])))
            for key, entry in context["registry"].items()
            if set(entry["sids"]) != expected
        ]
        if bad:
            raise ValueError(
                "pre-revelation passenger support differs across duration "
                f"branches in block={context['block']}; first mismatches: "
                f"{bad[:5]}")
        for family, _key in context["registry"]:
            by_family[family] += 1
            total += 1
    return {"total": total, "by_family": dict(by_family)}


def _add_decision_tree_minimum_service_rows(
        model: gp.Model, columns: Mapping[ColumnKey, gp.Var],
        scenario_variables: Mapping[str, Mapping], problem: CGProblem,
        config: CGConfig) -> int:
    """Add one causal minimum-service filter per macro-column.

    Boarding Vars are deduplicated by identity. Thus a pre-revelation action
    shared by two duration branches is counted once, while distinct
    post-revelation actions are each eligible to justify the pre-revelation
    activation. Singleton cells reduce exactly to the historical row.
    """
    grouped = _group_scenarios(problem.scenarios)
    rows = 0
    for (block, tid, cell), y_variable in columns.items():
        boarding = OrderedDict()
        for duration in cell:
            scenario = grouped[str(block)][int(duration)]
            sid = str(scenario["id"])
            for (_om, vehicle, _minute, _station), variable in (
                    scenario_variables[sid]["b"].items()):
                if str(vehicle) == str(tid):
                    boarding.setdefault(id(variable), variable)
        model.addConstr(
            gp.quicksum(boarding.values()) >=
            config.min_trip_boarding * y_variable,
            name=(f"min_svc_cell[{block},{tid}|"
                  f"{'-'.join(map(str, cell))}]"))
        rows += 1
    return rows


def _passenger_state_sharing_audit(
        scenario_variables: Mapping[str, Mapping],
        contexts: Mapping[str, Mapping]) -> dict:
    """Verify shared actions and their conservation-implied q/l states."""
    maxima = {family: 0.0 for family in ("b", "a", "q", "l")}
    by_block = _group_scenarios([
        variables["scenario"] for variables in scenario_variables.values()])
    for block, context in contexts.items():
        durations = tuple(context["durations"])
        if len(durations) < 2:
            continue
        reference_sid = str(by_block[block][durations[0]]["id"])
        reference = scenario_variables[reference_sid]
        for duration in durations[1:]:
            other_sid = str(by_block[block][duration]["id"])
            other = scenario_variables[other_sid]
            for family in maxima:
                left = reference[family]
                right = other[family]
                left_keys = {
                    key for key in left
                    if key[0] in context["common"] and
                    int(context["start"]) <=
                    _passenger_key_minute(family, key) < int(context["end"])}
                right_keys = {
                    key for key in right
                    if key[0] in context["common"] and
                    int(context["start"]) <=
                    _passenger_key_minute(family, key) < int(context["end"])}
                if left_keys != right_keys:
                    missing_left = sorted(right_keys - left_keys, key=repr)[:3]
                    missing_right = sorted(left_keys - right_keys, key=repr)[:3]
                    raise ValueError(
                        f"shared-interval {family} support differs in "
                        f"block={block}: missing D{durations[0]}="
                        f"{missing_left}, missing D{duration}={missing_right}")
                for key in left_keys:
                    maxima[family] = max(
                        maxima[family],
                        abs(float(left[key].X) - float(right[key].X)))
    return maxima


def _passenger_na_stats(
        scenarios: Sequence[Mapping], contexts: Mapping[str, Mapping],
        sharing_stats: Mapping, normal_fixed_count: int,
        minimum_service_rows: int) -> dict:
    enabled = bool(contexts)
    intervals = {
        block: (int(context["start"]), int(context["end"]))
        for block, context in contexts.items()}
    starts = {interval[0] for interval in intervals.values()}
    ends = {interval[1] for interval in intervals.values()}
    result = {
        "PassengerFlowVariableSharing": enabled,
        "PassengerSharedIntervalStart": (
            next(iter(starts)) if len(starts) == 1 else None),
        "PassengerSharedIntervalEndExclusive": (
            next(iter(ends)) if len(ends) == 1 else None),
        "PassengerSharedIntervalsByBlock": intervals,
        "PassengerSharedVars": int(sharing_stats.get("total", 0)),
        "PassengerSharedEqualityConstrs": 0,
        "PassengerStateNonanticipativeByConservation": enabled,
        "NormalFixedPassengerDecisionVars": int(normal_fixed_count),
        "CommonAffectedODCount": int(sum(
            len(context["common"]) for context in contexts.values())),
        # Passenger actions/states and bus decisions are nonanticipative.
        # Rail availability is intentionally the unmodified raw state of each
        # duration branch and therefore must not be advertised as shared.
        "TrainAvailabilityInformationSetNonanticipative": False,
        "TrainAvailabilityBranchSpecificRaw": enabled,
        "MinimumServiceDecisionTreeRows": int(minimum_service_rows),
        "MinimumServiceSemantics": (
            "UNIQUE_BOARDING_ACTIONS_PER_INFORMATION_CELL"
            if enabled else "PER_SCENARIO_TRIP"),
    }
    for family in _PASSENGER_ACTION_FAMILIES:
        result[f"PassengerShared_{family}"] = int(
            sharing_stats.get("by_family", {}).get(family, 0))
    for duration in sorted({int(s["duration"]) for s in scenarios}):
        branch = [s for s in scenarios if int(s["duration"]) == duration]
        result[f"PhysicalAffectedOD_D{duration}"] = int(sum(
            len(s.get("physical_affected_omega", ())) for s in branch))
        result[f"BranchAffectedOD_D{duration}"] = int(sum(
            len(s.get("branch_affected_omega", ())) for s in branch))
        result[f"NormalFixedOD_D{duration}"] = int(sum(
            len(s.get("normal_fixed_omega", ())) for s in branch))
        result[f"CrossDurationTrainWithdrawals_D{duration}"] = 0
    return result


def _trip_record(tid: str, trip: Mapping) -> dict:
    """Stable, export-friendly metadata for one candidate bus run."""
    return {
        "BusID": tid,
        "TripID": tid,
        "AssignedDepot": str(trip.get("Depot", "")),
        "Depot": str(trip.get("Depot", "")),
        "RouteKey": trip.get("RouteKey"),
        "Route": trip.get("Route", trip.get("RouteKey")),
        "Direction": trip.get("Direction"),
        "RouteDir": trip.get("Direction"),
        "RouteType": trip.get("RouteType"),
        "StartStation": trip.get("RouteOrigin"),
        "EndStation": trip.get("ServiceEndStation", trip.get("RouteDest")),
        "DispatchStart": int(trip.get("DispatchStart", 0)),
        "RunStart": trip.get("RunStart"),
        "RunEnd": trip.get("RunEnd"),
        "ServiceEnd": int(trip.get(
            "ServiceEnd", trip.get(
                "RunEnd", trip.get("DispatchStart", 0)))),
        "ResourceEnd": int(trip.get(
            "ResourceEnd", trip.get("RunEnd", trip.get("DispatchStart", 0)))),
        "DeadheadOut": trip.get("DeadheadOut"),
        "FleetBusID": None,
        "FleetBusIDScope": None,
        "ChainPosition": None,
        "ChainLength": None,
        "PreviousTripID": None,
        "NextTripID": None,
        "DeadheadBeforeMinutes": None,
        "WaitBeforeRun": None,
        "ReturnDeadhead": None,
        "ReturnStart": None,
        "ReturnEnd": None,
        "VehicleDutyStart": None,
        "VehicleDutyEnd": None,
        "RouteStops": tuple(trip.get("RouteStops", ())),
        "StopTimeList": tuple(trip.get("StopTimeList", ())),
        "Legs": tuple(trip.get("Legs", ())),
    }


def _positive_decision_rows(
        b: Mapping, a: Mapping, l: Mapping, q: Mapping, w: Mapping,
        od: Mapping, omega_map: Mapping, *,
        epsilon: float = DECISION_ROW_EPS,
        time_min: int = DECISION_TIME_MIN,
        time_max: int = DECISION_TIME_MAX) -> list[tuple]:
    """Extract the legacy disruption-CSV positive decision-variable rows.

    The returned tuple schema is exactly
    ``Variable, Omega, Origin, Dest, Time, StationID, TrainID, Value``.
    Keeping this conversion inside the standalone engine ensures that both
    :func:`solve_master` and :func:`solve_restricted_milp` expose the same
    result payload while the Gurobi model is still alive.
    """
    if float(epsilon) < 0:
        raise ValueError("decision-row epsilon must be nonnegative")
    lower, upper = int(time_min), int(time_max)
    if lower > upper:
        raise ValueError("decision-row time_min must not exceed time_max")

    rows: list[tuple] = []

    def endpoints(om):
        if om in omega_map:
            return omega_map[om]
        parts = str(om).split("_")
        return parts[0], parts[-1]

    def retained(variable, minute):
        minute = int(minute)
        value = float(variable.X)
        return minute, value, lower <= minute <= upper and value > epsilon

    for (om, vehicle, minute, station), variable in b.items():
        minute, value, keep = retained(variable, minute)
        if keep:
            origin, destination = endpoints(om)
            rows.append(("Boarding", om, origin, destination, minute,
                         station, vehicle, value))
    for (om, vehicle, minute, station), variable in a.items():
        minute, value, keep = retained(variable, minute)
        if keep:
            origin, destination = endpoints(om)
            rows.append(("Alighting", om, origin, destination, minute,
                         station, vehicle, value))
    for (om, vehicle, minute), variable in l.items():
        minute, value, keep = retained(variable, minute)
        if keep:
            origin, destination = endpoints(om)
            rows.append(("Load", om, origin, destination, minute,
                         None, vehicle, value))
    for (om, station, minute), variable in q.items():
        minute, value, keep = retained(variable, minute)
        if keep:
            origin, destination = endpoints(om)
            rows.append(("Queue", om, origin, destination, minute,
                         station, None, value))
    for (om, station, minute), variable in w.items():
        minute, value, keep = retained(variable, minute)
        if keep:
            origin, destination = endpoints(om)
            rows.append(("Walking", om, origin, destination, minute,
                         station, None, value))
    for (om, minute), variable in od.items():
        minute, value, keep = retained(variable, minute)
        if keep:
            origin, destination = endpoints(om)
            rows.append(("Exit", om, origin, destination, minute,
                         destination, None, value))
    return rows


def _absorbed_destination_exit_rows(
        a: Mapping, demand: Mapping, omega_map: Mapping, gamma: Iterable[int],
        sigma: int, *, epsilon: float = DECISION_ROW_EPS,
        time_min: int = DECISION_TIME_MIN,
        time_max: int = DECISION_TIME_MAX) -> list[tuple]:
    """Reconstruct immediate destination exits after q/od elimination."""
    modeled_minutes = {int(minute) for minute in gamma}
    exits = defaultdict(float)
    for (om, station, raw_minute), value in demand.items():
        if (om in omega_map and station == omega_map[om][1] and
                int(raw_minute) in modeled_minutes):
            exits[(om, int(raw_minute))] += float(value)
    for (om, _vehicle, raw_minute, station), variable in a.items():
        if om not in omega_map or station != omega_map[om][1]:
            continue
        minute = int(raw_minute) + int(sigma)
        if minute in modeled_minutes:
            exits[(om, minute)] += float(variable.X)

    rows = []
    for (om, minute), value in sorted(exits.items(), key=repr):
        if not (int(time_min) <= minute <= int(time_max) and value > epsilon):
            continue
        origin, destination = omega_map[om]
        rows.append(("Exit", om, origin, destination, minute,
                     destination, None, value))
    return rows


def _make_mip_progress_callback(*, stall_seconds: float | None,
                                progress_hook,
                                absolute_tolerance: float = 1e-7,
                                relative_tolerance: float = 1e-9,
                                progress_interval: float = 30.0):
    state = {
        "best_incumbent": math.inf,
        "last_material_improvement_runtime": None,
        "last_progress_runtime": -math.inf,
        "stall_triggered": False,
        "callback_error": None,
    }

    def callback(model, where):
        try:
            # POLLING explicitly permits no cbGet query.  More generally this
            # monitor needs information only at MIP and MIPSOL callbacks.
            if where not in (GRB.Callback.MIP, GRB.Callback.MIPSOL):
                return
            runtime = float(model.cbGet(GRB.Callback.RUNTIME))
            if where == GRB.Callback.MIPSOL:
                incumbent = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))
                threshold = max(
                    absolute_tolerance,
                    relative_tolerance * max(
                        1.0, abs(state["best_incumbent"])
                        if math.isfinite(state["best_incumbent"]) else 1.0))
                if incumbent < state["best_incumbent"] - threshold:
                    state["best_incumbent"] = incumbent
                    state["last_material_improvement_runtime"] = runtime
            if where == GRB.Callback.MIP:
                incumbent = float(model.cbGet(GRB.Callback.MIP_OBJBST))
                if (progress_hook is not None and
                        runtime - state["last_progress_runtime"] >=
                        progress_interval):
                    state["last_progress_runtime"] = runtime
                    bound = float(model.cbGet(GRB.Callback.MIP_OBJBND))
                    progress_hook({
                        "phase": "SOLVING_STAGE1_FINAL_MILP",
                        "runtime": runtime,
                        "incumbent": incumbent if math.isfinite(incumbent) else None,
                        "objective_bound": bound if math.isfinite(bound) else None,
                        "solution_count": int(model.cbGet(
                            GRB.Callback.MIP_SOLCNT)),
                        "node_count": float(model.cbGet(
                            GRB.Callback.MIP_NODCNT)),
                    })
            if (stall_seconds is not None and
                    state["last_material_improvement_runtime"] is not None and
                    runtime - state["last_material_improvement_runtime"] >=
                    float(stall_seconds)):
                state["stall_triggered"] = True
                model.terminate()
        except BaseException as exc:
            state["callback_error"] = f"{type(exc).__name__}: {exc}"
            model.terminate()

    return callback, state


_EXPORT_SKIPPED_PARAMETERS = ("License", "UserName", "Username", "Token", "Server", "WLS",
                               "Cloud", "LogFile", "ResultFile", "JobID", "Password",
                               "AccessID", "SecretKey", "CSAPI", "CSManager", "CSGroup",
                               "CSRouter", "CSAuth", "CSApp")


def _export_model_files(model: gp.Model, directory, name: str,
                        metadata: Mapping | None = None) -> dict:
    """Write ``<name>.mps.gz``, ``<name>.sol.gz`` (when an incumbent exists) and a JSON sidecar.

    Code-only (stage1.export_model_dir): used by the synthetic profile; the paper profile keeps
    it off because real-instance models are restricted data.
    """
    import json
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    def write(suffix: str) -> None:
        # Gurobi cannot open non-ASCII paths on Windows: write to a temporary
        # ASCII folder, then move the file into place.
        with tempfile.TemporaryDirectory() as scratch:
            temporary = Path(scratch) / f"model{suffix}"
            model.write(str(temporary))
            shutil.move(str(temporary), str(out / f"{name}{suffix}"))

    write(".mps.gz")   # gzip: Gurobi compresses by file extension
    parameters = {}
    for parameter in sorted(n for n in dir(GRB.Param) if not n.startswith("_")):
        if any(token in parameter for token in _EXPORT_SKIPPED_PARAMETERS):
            continue
        try:
            info = model.getParamInfo(parameter)
        except gp.GurobiError:
            continue
        if info is not None and info[2] != info[5]:
            parameters[info[0]] = info[2]
    sidecar = {
        "model": name,
        "status": _status_name(model.Status),
        "runtime_seconds": float(model.Runtime),
        "num_vars": int(model.NumVars),
        "num_constrs": int(model.NumConstrs),
        "num_nonzeros": int(model.NumNZs),
        "fingerprint": int(model.Fingerprint),
        "objective": None, "bound": None, "gap": None,
        "parameters": parameters,
        "gurobi_version": ".".join(str(v) for v in gp.gurobi.version()),
    }
    if int(model.SolCount) > 0:
        write(".sol.gz")
        sidecar["objective"] = float(model.ObjVal)
        if model.IsMIP:
            sidecar["bound"] = float(model.ObjBound)
            sidecar["gap"] = float(model.MIPGap)
    if metadata:
        sidecar["metadata"] = dict(metadata)
    (out / f"{name}.json").write_text(
        json.dumps(sidecar, indent=2, default=str) + "\n", encoding="utf-8")
    return sidecar


def solve_master(
        problem: CGProblem, config: CGConfig | None = None, *,
        lp_relax: bool = False, duals_out: dict | None = None,
        active_columns: Iterable[ColumnKey] | None = None,
        column_ub: Mapping[ColumnKey, float] | None = None,
        primal_hint: Mapping | None = None,
        depot_start: Mapping[str, float] | None = None,
        column_start: Iterable[ColumnKey] | None = None,
        extract_solution: bool = True,
        stall_seconds: float | None = None,
        progress_hook: Callable[[Mapping], None] | None = None,
        parameter_file: str | None = None,
        tune_output_path: str | None = None,
        tune_time_limit: float | None = None,
        tune_trials: int = 0,
        tune_candidate_time_limit: float | None = None,
        tune_only: bool = False,
        export_model_dir: str | None = None,
        export_name: str | None = None,
        export_metadata: Mapping | None = None):
    """Solve the standalone full/RMP master.

    ``export_model_dir`` (optional) writes the solved model as .mps/.sol/.json
    (see ``_export_model_files``); it does not change the model or the solve.

    Returns ``(depot_counts, status, scenario_results, stats)``.  When
    ``duals_out`` is supplied, optimal LP duals for ``q``, ``w``, and
    ``depot`` rows are copied into it before model disposal.
    """
    config = _cfg(config)
    if stall_seconds is not None and float(stall_seconds) <= 0:
        raise ValueError("stall_seconds must be positive or None")
    if parameter_file is not None and tune_output_path is not None:
        raise ValueError(
            "parameter_file and tune_output_path are mutually exclusive")
    if tune_output_path is not None:
        if tune_time_limit is None or float(tune_time_limit) <= 0:
            raise ValueError(
                "tune_time_limit must be positive when tuning is enabled")
        if int(tune_trials) < 0:
            raise ValueError("tune_trials must be nonnegative")
        if (tune_candidate_time_limit is not None and
                float(tune_candidate_time_limit) <= 0):
            raise ValueError("tune_candidate_time_limit must be positive")
    if tune_only and tune_output_path is None:
        raise ValueError("tune_only requires tune_output_path")
    if parameter_file is not None and not Path(parameter_file).is_file():
        raise FileNotFoundError(parameter_file)
    full_caches = _require_full_caches(problem)
    if duals_out is not None:
        duals_out.clear()
        duals_out.update({"q": {}, "w": {}, "depot": {}})
    active = (None if active_columns is None else
              {_normalise_key(key) for key in active_columns})
    upper = {_normalise_key(key): float(value)
             for key, value in (column_ub or {}).items()}
    started = time.time()
    model = _new_model("StandaloneExactCGMaster", problem)
    controlled_threads = int(model.Params.Threads)
    model.Params.IgnoreNames = int(config.ignore_names)
    model.Params.OutputFlag = int(config.output_flag)
    if config.time_limit is not None:
        model.Params.TimeLimit = float(config.time_limit)
    model.Params.MIPGap = float(config.mip_gap)
    if lp_relax:
        model.Params.OptimalityTol = float(config.optimality_tolerance)
        model.Params.FeasibilityTol = float(config.feasibility_tolerance)
        _apply_rmp_lp_parameters(model, config)
    else:
        if config.final_method is not None:
            model.Params.Method = int(config.final_method)
        if config.final_mip_focus is not None:
            model.Params.MIPFocus = int(config.final_mip_focus)
        if config.final_cuts is not None:
            model.Params.Cuts = int(config.final_cuts)
        if config.final_cut_passes is not None:
            model.Params.CutPasses = int(config.final_cut_passes)
        if config.final_concurrent_mip is not None:
            model.Params.ConcurrentMIP = int(config.final_concurrent_mip)
        if config.final_symmetry is not None:
            model.Params.Symmetry = int(config.final_symmetry)
        if config.final_heuristics is not None:
            model.Params.Heuristics = float(config.final_heuristics)
        if config.final_no_rel_heur_time is not None:
            model.Params.NoRelHeurTime = float(config.final_no_rel_heur_time)
        if config.final_improve_start_time is not None:
            model.Params.ImproveStartTime = float(
                config.final_improve_start_time)
        if config.final_best_obj_stop is not None:
            model.Params.BestObjStop = float(config.final_best_obj_stop)

    depots = tuple(str(depot) for depot in problem.depots)
    x_type = GRB.CONTINUOUS if lp_relax else GRB.INTEGER
    y_type = GRB.CONTINUOUS if lp_relax else GRB.BINARY
    fixed_counts = (
        None if problem.fixed_depot_counts is None else
        {str(depot): float(value)
         for depot, value in problem.fixed_depot_counts.items()})
    # Eq. (29) [eq:fleet_domain]: x_g in Z>=0, reserve buses positioned at depot g
    # (ub = B; Stage 2 / benchmark runs fix x_g through problem.fixed_depot_counts).
    x = {}
    for depot in depots:
        fixed_value = (None if fixed_counts is None else
                       float(fixed_counts.get(depot, 0.0)))
        x[depot] = model.addVar(
            lb=(0.0 if fixed_value is None else fixed_value),
            ub=(float(config.max_total_buses)
                if fixed_value is None else fixed_value),
            vtype=x_type, name=f"x[{depot}]")
    # Eq. (28) [eq:fleet]: sum_g x_g = B (fleet budget as an equality).
    model.addConstr(gp.quicksum(x.values()) == config.max_total_buses,
                    name="fleet_budget")
    # Eq. (31) [eq:na_dispatch] with Eq. (25) [eq:shared_trip_branches]: one trip-activation
    # variable y_v per (representative location, trip, information cell); the cell lists the
    # duration branches tau' in P_v(xi) that are indistinguishable at the mobilization time, so
    # y_v(xi) = y_v((j, tau')) holds by construction (shared variable, no equality rows).
    # Eq. (30) [eq:avail_cons] with Eq. (23) [eq:availability]: y_v is created only for trips
    # whose mobilization time lies inside the cell (alpha_{v,tau} = 1). Eq. (39): y binary.
    shared_y, columns = _build_shared_y(
        model, problem, config, y_type, active, upper)
    if config.enforce_vehicle_chains and not lp_relax:
        shared_chain, chain_totals = (
            _add_nonanticipative_vehicle_chain_constraints(
                model, problem, config, shared_y, columns, x)
        )
    else:
        shared_chain = {}
        chain_totals = {
            "num_vars": 0, "num_constraints": 0, "num_links": 0,
            "num_first": 0, "num_first_decisions": 0,
            "first_substituted": 0, "physical_arc_count": 0,
            "zero_fixed_links": 0,
            "decision_scope": None,
            "binary_auxiliaries": False,
        }
    hint_x = dict((primal_hint or {}).get("x", {}))
    hint_y = dict((primal_hint or {}).get("y", {}))
    if not lp_relax:
        for depot, value in hint_x.items():
            if str(depot) in x and math.isfinite(float(value)):
                x[str(depot)].VarHintVal = float(value)
                x[str(depot)].VarHintPri = 1
        for raw_key, value in hint_y.items():
            key = _normalise_key(raw_key)
            if key in columns and math.isfinite(float(value)):
                columns[key].VarHintVal = float(value)
                columns[key].VarHintPri = 1
        for depot, value in (depot_start or {}).items():
            if str(depot) in x and math.isfinite(float(value)):
                x[str(depot)].Start = float(value)
    column_by_variable = {id(variable): key
                          for key, variable in columns.items()}
    start_columns = (
        {_normalise_key(key) for key in column_start}
        if column_start is not None else set())
    unknown_start_columns = start_columns - set(columns)
    if unknown_start_columns:
        model.dispose()
        raise ValueError(
            "column start contains variables outside the restricted master: "
            f"{sorted(unknown_start_columns)[:3]}")
    if not lp_relax and column_start is not None:
        for key, variable in columns.items():
            variable.Start = float(key in start_columns)

    wdata = problem.wdata
    omega_map = wdata["Omega_Map"]
    station_of = wdata["station_of"]
    train_active_times = wdata["train_active_times"]
    train_dest_map = wdata["train_dest_map"]
    train_dwells = wdata["train_dwells"]
    trains = tuple(wdata["Vm"])
    gamma = tuple(wdata["Gamma"])
    start_min = int(wdata["START_MIN"])
    cumulative = wdata["CumD"]
    demand = wdata["D"]

    objective_terms = []
    scenario_variables = {}
    dual_handles = {"q": {}, "w": {}, "depot": {}}
    retain_dual_handles = duals_out is not None
    passenger_context_by_sid, passenger_contexts = (
        _build_passenger_sharing_contexts(problem.scenarios))
    normal_fixed_variable_count = 0
    minimum_service_rows = 0

    for scenario in problem.scenarios:
        sid = str(scenario["id"])
        cache = full_caches[sid]
        termination_alight_keys = (
            _passenger_action_support_termination_keys(scenario))
        probability = float(scenario.get(
            "probability", 1.0 / max(len(problem.scenarios), 1)))
        if probability < 0:
            model.dispose()
            raise ValueError(f"{sid}: invalid probability")
        onset = start_min + OBJECTIVE_ONSET_MIN          # t0 of Eq. (13)
        walking_plan = cache.get("_walking_plan") if config.eliminate_w else None
        if config.eliminate_w and walking_plan is None:
            walking_plan = build_walking_elimination_plan(
                w_rows=cache.get("w_var_keys", ()),
                alighting_keys=cache.get("a_var_keys", ()),
                omega_map=omega_map, gamma=gamma, sigma=config.sigma)
        omega_set = tuple(scenario.get("Omega_m", ()))
        spare_capacity = scenario.get("spare_capacity", {})
        y = shared_y[sid]
        active_trip_ids = set(y)

        b_keys = _unique_normalised_keys((
            (key[0], str(key[1]), int(key[2]), key[3])
            for key in cache.get("b_var_keys", ())
            if not (str(key[1]).startswith("B_") and
                    str(key[1]) not in active_trip_ids)
        ), f"{sid} b")
        a_keys = _unique_normalised_keys((
            (key[0], str(key[1]), int(key[2]), key[3])
            for key in cache.get("a_var_keys", ())
            if not (str(key[1]).startswith("B_") and
                    str(key[1]) not in active_trip_ids)
        ), f"{sid} a")
        l_reference_keys = _unique_normalised_keys((
            (key[0], str(key[1]), int(key[2]))
            for key in cache.get("l_var_keys", ())
            if not (str(key[1]).startswith("B_") and
                    (str(key[1]) not in active_trip_ids or
                     config.hybrid_bus_load_states))
        ), f"{sid} l")
        b_ub = {key: max(float(cumulative.get(
            (key[0], key[2]), 0.0)), 0.0) for key in b_keys}
        a_ub = {key: max(float(cumulative.get(
            (key[0], key[2]), 0.0)), 0.0) for key in a_keys}
        passenger_context = passenger_context_by_sid.get(sid)
        # Passenger flows of every affected OD pair omega in Omega_m(xi) (Eq. (11)):
        # b_{v,t,j} boarding, a_{v,t,j} alighting, l_{v,t} load, q_{j,t} queue, w = walking
        # transfer flow d_{i,j,t}, od = destination exit o_{j,t}. Eq. (10) [eq:nn_n]: all >= 0.
        # Eq. (8) [eq:dem_ab_n]: a, b <= cumulative OD demand sum_{t'<=t} D_{omega,s(omega),t'};
        # Eq. (9) [eq:dem_o_n]: od <= cumulative demand.
        b = _batch_add_passenger_vars(
            model, b_keys, family="b", sid=sid,
            context=passenger_context, ub=b_ub, name=f"b_{sid}")
        a = _batch_add_passenger_vars(
            model, a_keys, family="a", sid=sid,
            context=passenger_context, ub=a_ub, name=f"a_{sid}")

        load_groups = aggregate_bus_load_groups(cache, active_trip_ids)
        load_segments = ()
        hybrid_bus_states = {}
        if config.compress_load_states:
            load_segments, load_board, load_alight = _cache_load_plan(
                cache, l_reference_keys, b_keys, a_keys,
                ({} if config.hybrid_bus_load_states else load_groups),
                station_of)
            load_ub = {
                segment.stream_key[1]: (
                    config.bus_capacity
                    if str(segment.stream_key[1]).startswith("B_") else
                    config.train_capacity)
                for segment in load_segments
            }
            l_states, l = _add_compressed_load_states(
                model, b, a, load_segments, load_board, load_alight,
                ub_by_vehicle=load_ub, name=f"l_{sid}")
        else:
            l_ub = {
                key: float(config.bus_capacity if key[1].startswith("B_")
                           else config.train_capacity)
                for key in l_reference_keys
            }
            l = _batch_add_vars(
                model, l_reference_keys, ub=l_ub, name=f"l_{sid}")
            lazy_l_keys = []
            if not config.hybrid_bus_load_states:
                for tid, by_omega in load_groups.items():
                    for om, group in by_omega.items():
                        for minute in group["times"]:
                            load_key = (om, tid, minute)
                            if load_key not in l:
                                lazy_l_keys.append(load_key)
            lazy_l_keys = tuple(dict.fromkeys(lazy_l_keys))
            l.update(_batch_add_vars(
                model, lazy_l_keys, ub=float(config.bus_capacity),
                name=f"l_bus_{sid}"))
            l_states = l

        if config.hybrid_bus_load_states:
            bus_b_by_trip = defaultdict(dict)
            bus_a_by_trip = defaultdict(dict)
            for key, variable in b.items():
                if str(key[1]).startswith("B_"):
                    bus_b_by_trip[str(key[1])][key] = variable
            for key, variable in a.items():
                if str(key[1]).startswith("B_"):
                    bus_a_by_trip[str(key[1])][key] = variable
            for tid, by_omega in load_groups.items():
                plan = _build_bus_load_plan(tid, by_omega)
                hybrid_bus_states[tid] = _add_hybrid_bus_load_states(
                    model, bus_b_by_trip.get(tid, {}),
                    bus_a_by_trip.get(tid, {}),
                    plan.board, plan.alight, tid=tid, y=y[tid],
                    capacity=config.bus_capacity, objective_scale=0.0,
                    name=f"L_{sid}_{tid}",
                    ignore_names=config.ignore_names)

        q_reference_keys = _unique_normalised_keys((
            (om, station, int(minute)) for om, station, minute
            in cache.get("q_var_keys", ())), f"{sid} q")
        queue_compression = (
            cache.get("_queue_compression")
            if config.compress_queue_states else None)
        if config.compress_queue_states and queue_compression is None:
            queue_compression = build_queue_compression(
                q_reference_keys,
                cache.get("b_var_keys", ()), cache.get("a_var_keys", ()),
                demand, omega_map, config.sigma)
        q_keys = (queue_compression.model_keys
                  if queue_compression is not None else q_reference_keys)
        w_keys = (() if config.eliminate_w else _unique_normalised_keys((
            (om, station, int(minute)) for om, station, minute
            in cache.get("w_var_keys", ())), f"{sid} w"))
        od_keys = _unique_normalised_keys((
            (om, int(minute)) for om, minute
            in cache.get("od_var_keys", ())), f"{sid} od")
        od_ub = {key: max(float(cumulative.get(key, 0.0)), 0.0)
                 for key in od_keys}
        q = _batch_add_vars(model, q_keys, name=f"q_{sid}")
        w = _batch_add_passenger_vars(
            model, w_keys, family="w", sid=sid,
            context=passenger_context, name=f"w_{sid}")
        od = _batch_add_passenger_vars(
            model, od_keys, family="od", sid=sid,
            context=passenger_context, ub=od_ub, name=f"od_{sid}")
        normal_fixed_variable_count += _fix_normal_passenger_decisions(
            scenario, {"b": b, "a": a, "w": w, "od": od},
            passenger_context)
        if passenger_context is None:
            # Sec. 3.2.2: states immediately before onset are fixed to their
            # normal values (pre-onset b/a/od fixed; prepare_window attaches them).
            if "normal_pre_onset" not in scenario:
                model.dispose()
                raise ValueError(
                    f"{sid}: missing normal pre-onset actions (normal_pre_onset)")
            normal_fixed_variable_count += fix_pre_onset_to_normal(
                {"b": b, "a": a, "od": od}, scenario["normal_pre_onset"],
                onset, omega_set, label=sid)
        chain_block = shared_chain.get(sid)

        # Depot capacity enters only through Eq. (28) [eq:fleet] and Eqs. (34)-(36)
        # (duty start, successor, depot vehicles); no per-minute depot rows.

        terminal_loads = defaultdict(list)
        if not config.compress_load_states:
            train_board_events = defaultdict(list)
            train_alight_events = defaultdict(list)
            for key, variable in b.items():
                if not str(key[1]).startswith("B_"):
                    train_board_events[(key[0], str(key[1]),
                                        int(key[2]))].append(variable)
            for key, variable in a.items():
                if not str(key[1]).startswith("B_"):
                    train_alight_events[(key[0], str(key[1]),
                                         int(key[2]))].append(variable)
            explicit_bus_groups = (
                {} if config.hybrid_bus_load_states else load_groups)
            for tid, by_omega in explicit_bus_groups.items():
                for om, group in by_omega.items():
                    previous = 0
                    for index, minute in enumerate(group["times"]):
                        board = gp.quicksum(
                            b[key] for key in group["board"].get(minute, ())
                            if key in b)
                        alight = gp.quicksum(
                            a[key] for key in group["alight"].get(minute, ())
                            if key in a)
                        model.addConstr(
                            l[(om, tid, minute)] == previous + board - alight,
                            name=(f"lb0[{sid},{om},{tid},{minute}]"
                                  if index == 0 else
                                  f"lb[{sid},{om},{tid},{minute}]"))
                        previous = l[(om, tid, minute)]
                    if group["times"]:
                        last = int(group["times"][-1])
                        terminal_loads[tid].append(l[(om, tid, last)])

            if not config.compress_load_states:
                for om in omega_set:
                    if om not in omega_map:
                        continue
                    for train in trains:
                        if train not in train_active_times:
                            continue
                        active_times = train_active_times[train]
                        for minute in active_times:
                            load_key = (om, str(train), int(minute))
                            if load_key not in l:
                                continue
                            event_key = (om, str(train), int(minute))
                            board = gp.quicksum(
                                train_board_events.get(event_key, ()))
                            alight = gp.quicksum(
                                train_alight_events.get(event_key, ()))
                            if minute == active_times.start:
                                model.addConstr(
                                    l[load_key] == board - alight,
                                    name=(f"trlb0[{sid},{om},{train},"
                                          f"{minute}]"))
                            else:
                                previous = l.get(
                                    (om, str(train), int(minute) - 1))
                                if previous is not None:
                                    model.addConstr(
                                        l[load_key] == previous + board - alight,
                                        name=(f"trlb[{sid},{om},{train},"
                                              f"{minute}]"))

        if (config.compress_load_states and
                not config.hybrid_bus_load_states):
            for tid, by_omega in load_groups.items():
                for om, group in by_omega.items():
                    if group["times"]:
                        last = int(group["times"][-1])
                        terminal_loads[tid].append(l[(om, tid, last)])

        # Eq. (6) [eq:l_n] load balance and capacities on compressed load states:
        # rail v in V_r: Eq. (7) [eq:cap_n] with the residual capacity S_{v,t}(xi) of
        # Eq. (12) [eq:spare] (withdrawn services theta_{v,t} = 1 get capacity 0);
        # bus v in V_rho: Eq. (37) [eq:cap_bus] sum_omega l_{v,t} <= C_v y_v.
        if config.compress_load_states:
            _add_interval_load_capacities(
                model, l_states, load_segments, y=y,
                spare_capacity=spare_capacity, config=config,
                name=f"load_cap_{sid}")
        else:
            bus_loads = defaultdict(list)
            train_loads = defaultdict(list)
            for (om, vehicle, minute), variable in l.items():
                target = bus_loads if vehicle.startswith("B_") else train_loads
                target[(vehicle, minute)].append(variable)
            # Eq. (37) [eq:cap_bus] / Eq. (7) [eq:cap_n] on uncompressed load states.
            for (tid, minute), variables in bus_loads.items():
                model.addConstr(gp.quicksum(variables) <=
                                config.bus_capacity * y[tid],
                                name=f"bus_cap[{sid},{tid},{minute}]")
            for (train, minute), variables in train_loads.items():
                model.addConstr(
                    gp.quicksum(variables) <=
                    float(spare_capacity.get((train, minute), 0.0)),
                    name=f"tr_cap[{sid},{train},{minute}]")

        n_omega = cache.get("N_omega", {})
        boarding_index = cache.get("boarding_by_od_t_j", {})
        alighting_index = cache.get("alighting_by_od_t_j", {})
        if queue_compression is not None:
            q_flow_rows = queue_compression.iter_segments()
        else:
            q_flow_rows = (
                SimpleNamespace(
                    omega=om, station=station, start=int(minute),
                    key=(om, station, int(minute)),
                    predecessor=((om, station, int(minute) - 1)
                                 if position > 0 else None))
                for om in omega_set if om in omega_map
                for station in n_omega.get(om, set())
                for position, minute in enumerate(gamma)
                if (om, station, int(minute)) in q)
        for segment in q_flow_rows:
            om, station, minute = (
                segment.omega, segment.station, int(segment.start))
            destination = omega_map[om][1]
            board = gp.quicksum(
                b[(om, str(vehicle), minute, station)]
                for vehicle in boarding_index.get((om, minute, station), ())
                if (om, str(vehicle), minute, station) in b)
            alight = gp.LinExpr()
            if minute >= start_min + config.sigma:
                event_minute = minute - config.sigma
                for vehicle in alighting_index.get(
                        (om, event_minute, station), ()):
                    key = (om, str(vehicle), event_minute, station)
                    if key in a:
                        alight += a[key]
            previous = q.get(segment.predecessor, 0)
            rhs = (previous + float(demand.get(
                (om, station, minute), 0.0)) + alight - board)
            if station == destination:
                rhs -= od.get((om, minute), 0)
            # Eq. (4) [eq:q_dest_n] (destination, with exit o) and Eq. (5) [eq:q_nd_n]
            # (other platforms): q_{j,t} = q_{j,t-1} + D + walking arrivals + a - b,
            # written once per compressed queue segment (exact queue compression).
            constraint = model.addConstr(
                q[segment.key] == rhs,
                name=f"q_flow[{sid},{om},{station},{minute}]")
            if retain_dual_handles:
                dual_handles["q"][(sid, om, station, minute)] = constraint

        if not config.eliminate_w:
            for om in omega_set:
                if om not in omega_map:
                    continue
                destination = omega_map[om][1]
                for station in n_omega.get(om, set()):
                    if station == destination:
                        continue
                    for position, minute in enumerate(gamma):
                        w_key = (om, station, int(minute))
                        if w_key not in w:
                            continue
                        alight_in = gp.quicksum(
                            a[(om, str(vehicle), int(minute), station)]
                            for vehicle in alighting_index.get(
                                (om, minute, station), ())
                            if (om, str(vehicle), int(minute), station) in a)
                        alight_out = gp.LinExpr()
                        if minute >= start_min + config.sigma:
                            event_minute = int(minute) - config.sigma
                            for vehicle in alighting_index.get(
                                    (om, event_minute, station), ()):
                                key = (om, str(vehicle), event_minute, station)
                                if key in a:
                                    alight_out += a[key]
                        previous = (w.get(
                            (om, station, int(minute) - 1), 0)
                            if position > 0 else 0)
                        constraint = model.addConstr(
                            w[w_key] == previous + alight_in - alight_out,
                            name=f"w_flow[{sid},{om},{station},{minute}]")
                        if retain_dual_handles:
                            dual_handles["w"][
                                (sid, om, station, int(minute))] = constraint

        # Code-only exact fixings: no boarding at the OD destination, no alighting at the
        # OD origin, no boarding onto a physically blocked train (theta = 1, Eq. (12)).
        for (om, vehicle, minute, station), variable in b.items():
            if om in omega_map and station == omega_map[om][1]:
                model.addConstr(variable == 0.0,
                                name=f"no_board_dest[{sid},{om},{vehicle},{minute}]")
            if _physical_train_boarding_forbidden(
                    scenario, vehicle, minute):
                model.addConstr(
                    variable == 0.0,
                    name=(f"no_board_withdrawn["
                          f"{sid},{om},{vehicle},{minute}]"))
        for (om, vehicle, minute, station), variable in a.items():
            if _forbidden_origin_alight(
                    (om, vehicle, minute, station), omega_map,
                    termination_alight_keys):
                model.addConstr(variable == 0.0,
                                name=f"no_alight_orig[{sid},{om},{vehicle},{minute}]")

        for om in omega_set:
            if om not in omega_map:
                continue
            for train in trains:
                if train not in train_active_times:
                    continue
                for dwell in train_dwells.get(train, ()):
                    station = dwell["station"]
                    times = tuple(dwell["times"])
                    incoming = l.get((om, str(train), int(times[0]) - 1), 0)
                    alight = [a[(om, str(train), int(minute), station)]
                              for minute in times
                              if (om, str(train), int(minute), station) in a]
                    if alight:
                        model.addConstr(gp.quicksum(alight) <= incoming,
                                        name=f"dwell[{sid},{om},{train},{station}]")
                if train in train_dest_map:
                    last = train_active_times[train].stop - 1
                    key = (om, str(train), int(last))
                    if key in l:
                        terminal_loads[str(train)].append(l[key])

        # Code-only: every vehicle is empty after its last scheduled stop (follows from the
        # arrival indicator I_{v,t,j} of Eq. (20); written explicitly).
        _add_vehicle_terminal_empty_rows(
            model, terminal_loads, name=f"vehicle_empty_{sid}",
            ignore_names=config.ignore_names)

        b_by_trip = defaultdict(list)
        for (_om, vehicle, _minute, _station), variable in b.items():
            if vehicle.startswith("B_"):
                b_by_trip[vehicle].append(variable)
        # Eq. (38) [eq:min_svc]: sum_{omega,j,t} b_{v,t,j} >= epsilon y_v for every bus trip.
        if not passenger_contexts:
            for tid in active_trip_ids:
                model.addConstr(
                    gp.quicksum(b_by_trip.get(tid, ())) >=
                    config.min_trip_boarding * y[tid],
                    name=f"min_svc[{sid},{tid}]")
                minimum_service_rows += 1

        # Eq. (2) [eq:inventory] and Eq. (13) [eq:obj_db]: z_t = kappa [queue + walking]
        # + onboard load (Table 3: kappa), summed over Omega_m(xi) and t in Gamma with
        # t >= t0 and divided by T - t0; compressed states carry their number of
        # represented minutes t >= t0 (eq13_inventories).
        queue_inventory, load_inventory, walking_inventory = eq13_inventories(
            onset, q=q, queue_compression=queue_compression, l=l,
            l_states=l_states, load_segments=load_segments,
            compressed_loads=config.compress_load_states,
            hybrid_bus_states=hybrid_bus_states, a=a,
            walking_plan=walking_plan, w=w)
        objective_terms.append(probability * eq13_objective(
            queue_inventory, load_inventory, walking_inventory))
        q_output = (_expanded_queue_view(q, queue_compression)
                    if queue_compression is not None else q)
        scenario_variables[sid] = {
            "scenario": scenario, "cache": cache, "y": y, "b": b, "a": a,
            "l": l, "q": q_output, "w": w, "od": od,
            "l_states": l_states, "load_segments": load_segments,
            "hybrid_bus_states": hybrid_bus_states,
            "load_groups": load_groups,
            "q_states": q, "queue_compression": queue_compression,
            "walking_plan": walking_plan,
            "b_by_trip": b_by_trip, "chain": chain_block,
            "eq13_inventories": (
                queue_inventory, load_inventory, walking_inventory),
        }

    if not objective_terms:
        model.dispose()
        return ({depot: 0 for depot in depots}, "NO_SCENARIO", {}, None)

    passenger_sharing_stats = _validate_passenger_variable_sharing(
        passenger_contexts)
    if passenger_contexts:
        minimum_service_rows += _add_decision_tree_minimum_service_rows(
            model, columns, scenario_variables, problem, config)

    # Eq. (27) [eq:obj_milp]: min sum_{xi in Xi^R} p_xi Z_m^B(xi); p_xi = representative
    # cluster mass x duration prior pi_tau (Table 3), set by 04_run_stage1.reweight_duration_prior.
    model.setObjective(gp.quicksum(objective_terms), GRB.MINIMIZE)
    model.update()
    build_seconds = time.time() - started
    if progress_hook is not None:
        progress_hook({
            "phase": "STAGE1_MASTER_BUILT",
            "lp_relax": bool(lp_relax),
            "build_seconds": build_seconds,
            "num_vars": int(model.NumVars),
            "num_constraints": int(model.NumConstrs),
            "num_binary": int(model.NumBinVars),
        })
    tuning_seconds = 0.0
    tune_result_count = 0
    tune_result_index = None
    applied_tune_candidate_time_limit = None
    applied_parameter_file = None
    if parameter_file is not None:
        parameter_path = Path(parameter_file).resolve()
        if str(parameter_path).isascii():
            model.read(str(parameter_path))
        else:
            # Gurobi 12 on Windows cannot reliably read parameter files from
            # paths containing non-ASCII characters.  Stage through the
            # ASCII-only system temp directory without changing the artifact.
            with tempfile.TemporaryDirectory(prefix="gurobi_prm_read_") as temp:
                staged_parameter_path = Path(temp) / "parameters.prm"
                shutil.copy2(parameter_path, staged_parameter_path)
                model.read(str(staged_parameter_path))
        applied_parameter_file = str(parameter_path)
    elif tune_output_path is not None:
        applied_tune_candidate_time_limit = min(
            (float(tune_candidate_time_limit)
             if tune_candidate_time_limit is not None
             else float(tune_time_limit)),
            (float(config.time_limit)
             if config.time_limit is not None else float(tune_time_limit)),
        )
        if progress_hook is not None:
            progress_hook({
                "phase": "STAGE1_GUROBI_TUNING_STARTED",
                "tune_time_limit_seconds": float(tune_time_limit),
                "tune_trials": int(tune_trials),
                "target_mip_gap": float(config.mip_gap),
                "candidate_time_limit_seconds": (
                    applied_tune_candidate_time_limit),
            })
        model.Params.TuneTimeLimit = float(tune_time_limit)
        model.Params.TuneTrials = int(tune_trials)
        model.Params.TuneCriterion = 1
        model.Params.TuneTargetMIPGap = float(config.mip_gap)
        # TuneTimeLimit is checked between tuning runs.  Bound each candidate
        # as well, otherwise the first baseline run can exceed the requested
        # total tuning budget when the production TimeLimit is much larger.
        model.Params.TimeLimit = applied_tune_candidate_time_limit
        tune_started = time.time()
        model.tune()
        tuning_seconds = time.time() - tune_started
        tune_result_count = int(model.TuneResultCount)
        if tune_result_count <= 0:
            model.dispose()
            raise RuntimeError("Gurobi tuner returned no parameter result")
        # Gurobi stores tuning results in decreasing quality; result 0 is
        # always the best available parameter set.
        tune_result_index = 0
        model.getTuneResult(tune_result_index)
        tune_output = Path(tune_output_path).resolve()
        applied_parameter_file = str(tune_output)
        if str(tune_output).isascii():
            model.write(str(tune_output))
        else:
            # Use an ASCII staging path for the same Gurobi-on-Windows path
            # limitation, then copy the generated artifact to its real path.
            with tempfile.TemporaryDirectory(prefix="gurobi_prm_write_") as temp:
                staged_tune_output = Path(temp) / "tuned.prm"
                model.write(str(staged_tune_output))
                shutil.copy2(staged_tune_output, tune_output)
        # Do not let solutions produced during tuning leak into the timed
        # production solve.  reset() keeps starts and the selected parameters.
        model.reset()
        if progress_hook is not None:
            progress_hook({
                "phase": "STAGE1_GUROBI_TUNING_FINISHED",
                "tuning_seconds": tuning_seconds,
                "tune_result_count": tune_result_count,
                "tune_result_index": tune_result_index,
                "parameter_file": applied_parameter_file,
            })
        if tune_only:
            stats = _model_stats(model, build_seconds)
            stats.update({
                "SolverParameterStrategy": "GUROBI_AUTOMATIC_TUNING",
                "TunedParameterFile": applied_parameter_file,
                "TuningSeconds": float(tuning_seconds),
                "TuneResultCount": int(tune_result_count),
                "TuneResultIndex": int(tune_result_index),
                "TunedImprovementFound": bool(tune_result_count > 0),
                "TuneTimeLimit": float(tune_time_limit),
                "TuneTrials": int(tune_trials),
                "TuneCriterion": 1,
                "TuneTargetMIPGap": float(config.mip_gap),
                "TuneCandidateTimeLimit": (
                    applied_tune_candidate_time_limit),
            })
            model.dispose()
            return ({}, "TUNED", {}, stats)

    # These are experiment controls, not solver-strategy choices.  Reapply
    # them after reading/getting a tuning result so every reported run keeps
    # the requested resource limit, stopping rule, and log setting.
    model.Params.IgnoreNames = int(config.ignore_names)
    model.Params.OutputFlag = int(config.output_flag)
    model.Params.Threads = controlled_threads
    if config.time_limit is not None:
        model.Params.TimeLimit = float(config.time_limit)
    model.Params.MIPGap = float(config.mip_gap)
    callback_state = {
        "stall_triggered": False, "callback_error": None,
    }
    callback = None
    if not lp_relax and (stall_seconds is not None or progress_hook is not None):
        callback, callback_state = _make_mip_progress_callback(
            stall_seconds=stall_seconds,
            progress_hook=progress_hook)
    model.optimize(callback)
    if export_model_dir:
        _export_model_files(model, export_model_dir,
                            export_name or model.ModelName, export_metadata)
    status = _status_name(model.Status)
    stats = _model_stats(model, build_seconds)
    stats.update({
        "EarlyStopTriggered": bool(callback_state["stall_triggered"]),
        "StallSeconds": stall_seconds,
        "TerminationReason": (
            "OBJECTIVE_STALL" if callback_state["stall_triggered"] else status),
        "CertifiedOptimal": bool(model.Status == GRB.OPTIMAL),
        "CallbackError": callback_state["callback_error"],
        "VehicleChainEnforced": bool(
            config.enforce_vehicle_chains and not lp_relax),
        "VehicleChainRelaxation": (
            "SERVICE_OVERLAP" if config.enforce_vehicle_chains and lp_relax
            else "LEGACY_RESOURCE_INTERVAL"),
        "DepotOverlapStrengthening": bool(
            config.enforce_vehicle_chains and not lp_relax),
        "VehicleChainVars": int(chain_totals["num_vars"]),
        "VehicleChainConstrs": int(chain_totals["num_constraints"]),
        "VehicleChainPhysicalArcs": int(
            chain_totals["physical_arc_count"]),
        "VehicleChainLinks": int(chain_totals["num_links"]),
        "VehicleChainFirstVars": int(chain_totals["num_first"]),
        "VehicleChainFirstDecisions": int(
            chain_totals["num_first_decisions"]),
        "VehicleChainFirstSubstituted": int(
            chain_totals["first_substituted"]),
        "VehicleChainZeroFixedLinks": int(
            chain_totals["zero_fixed_links"]),
        "VehicleChainDecisionScope": chain_totals["decision_scope"],
        "VehicleChainInformationSetNonanticipative": bool(
            config.enforce_vehicle_chains and not lp_relax),
        "VehicleChainBinaryAuxiliaries": bool(
            chain_totals["binary_auxiliaries"]),
        "HybridFinalBusLoadProjection": bool(
            config.hybrid_bus_load_states),
        "HybridFinalBusLoadStates": int(sum(
            len(states)
            for variables in scenario_variables.values()
            for states in variables.get("hybrid_bus_states", {}).values())),
        "FinalMIPFocus": (
            None if config.final_mip_focus is None
            else int(config.final_mip_focus)),
        "FinalCuts": (
            None if config.final_cuts is None else int(config.final_cuts)),
        "FinalCutPasses": (
            None if config.final_cut_passes is None
            else int(config.final_cut_passes)),
        "FinalConcurrentMIP": (
            None if config.final_concurrent_mip is None
            else int(config.final_concurrent_mip)),
        "FinalSymmetry": (
            None if config.final_symmetry is None
            else int(config.final_symmetry)),
        "FinalHeuristics": (
            None if config.final_heuristics is None
            else float(config.final_heuristics)),
        "FinalNoRelHeurTime": (
            None if config.final_no_rel_heur_time is None
            else float(config.final_no_rel_heur_time)),
        "FinalImproveStartTime": (
            None if config.final_improve_start_time is None
            else float(config.final_improve_start_time)),
        "FinalBestObjStop": (
            None if config.final_best_obj_stop is None
            else float(config.final_best_obj_stop)),
        "FinalMethod": (
            None if config.final_method is None
            else int(config.final_method)),
        "PrimalHintX": int(len(hint_x) if not lp_relax else 0),
        "PrimalHintY": int(len(hint_y) if not lp_relax else 0),
        "DepotStartX": int(len(depot_start or {}) if not lp_relax else 0),
        "ColumnStartY": int(
            len(start_columns) if not lp_relax and column_start is not None
            else 0),
        "ReturnToHomeDepot": bool(
            config.enforce_vehicle_chains and not lp_relax),
        "ReturnWithinServiceHorizon": False,
        "SolverParameterStrategy": (
            "GUROBI_AUTOMATIC_TUNING" if tune_output_path is not None else
            ("GUROBI_TUNED_PARAMETER_FILE"
             if parameter_file is not None else "CALLER_CONFIGURATION")),
        "TunedParameterFile": applied_parameter_file,
        "TuningSeconds": float(tuning_seconds),
        "TuneResultCount": int(tune_result_count),
        "TuneResultIndex": tune_result_index,
        "TunedImprovementFound": bool(
            tune_result_index is not None and tune_result_count > 0),
        "TuneTimeLimit": (
            float(tune_time_limit) if tune_output_path is not None else None),
        "TuneTrials": (
            int(tune_trials) if tune_output_path is not None else None),
        "TuneCriterion": 1 if tune_output_path is not None else None,
        "TuneTargetMIPGap": (
            float(config.mip_gap) if tune_output_path is not None else None),
        "TuneCandidateTimeLimit": applied_tune_candidate_time_limit,
    })
    stats.update(_passenger_na_stats(
        problem.scenarios, passenger_contexts, passenger_sharing_stats,
        normal_fixed_variable_count, minimum_service_rows))

    if duals_out is not None:
        for family in ("q", "w", "depot"):
            duals_out[family] = (
                {key: float(constraint.Pi)
                 for key, constraint in dual_handles[family].items()}
                if lp_relax and model.Status == GRB.OPTIMAL else {})

    acceptable = model.Status in (
        GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED,
        GRB.USER_OBJ_LIMIT)
    if not acceptable or model.SolCount == 0:
        if progress_hook is not None:
            progress_hook({
                "phase": "STAGE1_MASTER_COMPLETE",
                "status": status, "stats": stats,
            })
        model.dispose()
        return ({depot: 0 for depot in depots}, status, {}, stats)

    passenger_state_audit = _passenger_state_sharing_audit(
        scenario_variables, passenger_contexts)
    stats["PassengerSharedBoardingMaxAbsDiff"] = float(
        passenger_state_audit["b"])
    stats["PassengerSharedAlightingMaxAbsDiff"] = float(
        passenger_state_audit["a"])
    stats["PassengerSharedQueueMaxAbsDiff"] = float(
        passenger_state_audit["q"])
    stats["PassengerSharedLoadMaxAbsDiff"] = float(
        passenger_state_audit["l"])
    if max(passenger_state_audit.values(), default=0.0) > 1e-5:
        model.dispose()
        raise RuntimeError(
            "shared passenger actions did not produce non-anticipative "
            f"states: {passenger_state_audit}")

    depot_counts = {
        depot: (float(x[depot].X) if lp_relax else int(round(x[depot].X)))
        for depot in depots
    }
    scenario_results = {}
    extract_started = time.perf_counter()
    extraction_tolerance = max(
        EXTRACTION_SOLUTION_TOLERANCE,
        float(config.feasibility_tolerance),
        10.0 * float(model.Params.FeasibilityTol),
    )
    if extract_solution:
        for sid, variables in scenario_variables.items():
            cache = variables["cache"]
            hybrid_bus_loads = (
                _reconstruct_hybrid_bus_loads(
                    variables.get("load_groups", {}),
                    variables["b"], variables["a"],
                    tolerance=extraction_tolerance)
                if variables.get("hybrid_bus_states") else {})
            # Eq. (13) [eq:obj_db]: areas over t >= t0, evaluated from the objective's
            # own expressions (eq13_inventories).
            queue_expr, load_expr, walking_expr = variables["eq13_inventories"]
            queue_area = float(queue_expr.getValue())
            onboard_area = float(load_expr.getValue())
            walking_plan = variables.get("walking_plan")
            if walking_plan is not None:
                # Validates that every positive alighting lies in the eliminated support.
                walking_plan.walking_area(
                    variables["a"],
                    feasibility_tolerance=extraction_tolerance,
                )
            walking_area = float(walking_expr.getValue())
            boarded = math.fsum(float(var.X) for var in variables["b"].values())
            y_values = {tid: float(var.X) for tid, var in variables["y"].items()}
            scenario = variables["scenario"]
            chain_block = variables.get("chain") or {}
            selected_links = {
                str(predecessor): str(successor)
                for (predecessor, successor), variable
                in chain_block.get("link", {}).items()
                if float(variable.X) > 0.5
            }
            selected_first = {
                str(tid) for tid, variable
                in chain_block.get("first", {}).items()
                if float(variable.X) > 0.5
            }
            chain_assignments = (
                _vehicle_chain_assignments(
                    scenario.get("bus_trips", {}),
                    (tid for tid, value in y_values.items() if value > 0.5),
                    problem.travel_times, depot_counts,
                    config.turnaround_minutes,
                    successor_links=selected_links,
                    first_trip_ids=selected_first)
                if config.enforce_vehicle_chains and not lp_relax else {})
            bus_operations = []
            for tid, y_variable in variables["y"].items():
                y_value = float(y_variable.X)
                if y_value <= 1e-7:
                    continue
                trip = _trip_lookup(scenario, tid)
                if trip is None:
                    continue
                loads_by_minute = defaultdict(float)
                event_minutes = []
                for load_source in (variables["l"], hybrid_bus_loads):
                    for (_om, vehicle, minute), variable in load_source.items():
                        if vehicle != tid:
                            continue
                        value = float(variable.X)
                        loads_by_minute[int(minute)] += value
                        if value > 1e-7:
                            event_minutes.append(int(minute))
                boarded_trip = 0.0
                for (_om, vehicle, minute, _station), variable in variables["b"].items():
                    if vehicle != tid:
                        continue
                    value = float(variable.X)
                    boarded_trip += value
                    if value > 1e-7:
                        event_minutes.append(int(minute))
                for (_om, vehicle, minute, _station), variable in variables["a"].items():
                    if vehicle == tid and float(variable.X) > 1e-7:
                        event_minutes.append(int(minute))
                if loads_by_minute:
                    peak_minute, max_onboard = min(
                        loads_by_minute.items(), key=lambda item: (-item[1], item[0]))
                else:
                    peak_minute, max_onboard = None, 0.0
                column_key = column_by_variable.get(id(y_variable))
                information_cell = (
                    tuple(column_key[2]) if column_key is not None else ())
                record = _trip_record(tid, trip)
                record.update({
                    "ScenarioID": sid,
                    "Duration": int(scenario.get("duration", 0)),
                    "Block": str(scenario.get(
                        "block_station", scenario.get("station", sid))),
                    "InformationCell": information_cell,
                    "DecisionState": information_cell,
                    "Y": y_value,
                    "Selected": bool(y_value > 0.5),
                    "BoardedPassengers": float(boarded_trip),
                    "MaxOnboard": float(max_onboard),
                    "PeakLoadMinute": peak_minute,
                    "ActualLastPassengerMinute": (
                        max(event_minutes) if event_minutes else None),
                })
                record.update(chain_assignments.get(tid, {}))
                predecessor = record.get("PreviousTripID")
                if predecessor is None:
                    chain_meta = chain_block.get("first_meta", {}).get(tid, {})
                else:
                    chain_meta = chain_block.get("link_meta", {}).get(
                        (str(predecessor), str(tid)), {})
                record["ChainDecisionTime"] = chain_meta.get("decision_time")
                record["ChainInformationCell"] = chain_meta.get(
                    "information_cell")
                bus_operations.append(record)
            decision_rows = _positive_decision_rows(
                variables["b"], variables["a"], variables["l"],
                variables["q"], variables["w"], variables["od"],
                problem.wdata["Omega_Map"])
            if hybrid_bus_loads:
                decision_rows.extend(_positive_decision_rows(
                    {}, {}, hybrid_bus_loads, {}, {}, {},
                    problem.wdata["Omega_Map"]))
            if config.eliminate_destination_states:
                decision_rows.extend(_absorbed_destination_exit_rows(
                    variables["a"], problem.wdata["D"],
                    problem.wdata["Omega_Map"], problem.wdata["Gamma"],
                    config.sigma))
            if walking_plan is not None:
                decision_rows.extend(walking_plan.iter_positive_decision_rows(
                    variables["a"], epsilon=DECISION_ROW_EPS,
                    time_min=DECISION_TIME_MIN, time_max=DECISION_TIME_MAX,
                    feasibility_tolerance=extraction_tolerance))
            scenario_results[sid] = {
                "status": status,
                "objective": ((OUT_OF_VEHICLE_WEIGHT * (queue_area + walking_area)
                               + onboard_area) / OBJECTIVE_NORMALISER),
                "boarded": boarded,
                "queue_area": queue_area,
                "onboard_area": onboard_area,
                "walking_area": walking_area,
                "y": y_values,
                "active_trip_ids": tuple(sorted(y_values)),
                "selected_trip_ids": tuple(sorted(
                    tid for tid, value in y_values.items() if value > 0.5)),
                "vehicle_chains": tuple(sorted(
                    (dict(value, TripID=tid)
                     for tid, value in chain_assignments.items()),
                    key=lambda row: (
                        str(row["FleetBusID"]), int(row["ChainPosition"])))),
                "bus_operations": tuple(bus_operations),
                "dv_rows": decision_rows,
            }
    stats["ExtractSec"] = float(time.perf_counter() - extract_started)
    if progress_hook is not None:
        progress_hook({
            "phase": "STAGE1_MASTER_COMPLETE",
            "status": status, "stats": stats,
        })
    model.dispose()
    return depot_counts, status, scenario_results, stats


def solve_rmp(problem: CGProblem, config: CGConfig | None,
              active_columns: Iterable[ColumnKey],
              depot_row_keys: Iterable[tuple[str, str, int]] | None = None) -> dict:
    """Solve one restricted LP over a fixed coupling-row universe."""
    config = _cfg(config)
    duals = {}
    _x, status, _results, stats = solve_master(
        problem, config, lp_relax=True, duals_out=duals,
        active_columns=active_columns,
        extract_solution=False)
    return {
        "status": status,
        "objective": float(stats["ObjVal"]) if stats else math.nan,
        "duals": duals,
        "stats": stats,
    }


def _unique_normalised_keys(keys: Iterable[tuple], family: str) -> tuple:
    normalised = tuple(keys)
    if len(normalised) != len(set(normalised)):
        raise ValueError(f"duplicate {family} variable keys")
    return normalised


def _batch_add_vars(model: gp.Model, keys: Iterable[tuple], *, lb=0.0,
                    ub=GRB.INFINITY, obj=0.0, vtype=GRB.CONTINUOUS,
                    name: str, ignore_names: bool = False) -> dict:
    """Create a homogeneous variable family with one Gurobi API call.

    Gurobi tupledict indices cannot contain a tuple as one member.  Such input
    is uncommon here, but the standalone API accepts arbitrary hashable OD
    identifiers, so that case deliberately falls back to scalar creation.
    """
    if not isinstance(keys, tuple):
        keys = tuple(keys)
    if not keys:
        return {}
    nested = any(
        isinstance(member, tuple) for key in keys
        for member in (key if isinstance(key, tuple) else (key,)))
    if nested:
        result = {}
        for key in keys:
            key_ub = ub.get(key, GRB.INFINITY) if isinstance(ub, Mapping) else ub
            key_obj = obj.get(key, 0.0) if isinstance(obj, Mapping) else obj
            if ignore_names:
                result[key] = model.addVar(
                    lb=lb, ub=key_ub, obj=key_obj, vtype=vtype)
            else:
                result[key] = model.addVar(
                    lb=lb, ub=key_ub, obj=key_obj, vtype=vtype,
                    name=f"{name}[{repr(key)}]")
        return result
    if ignore_names:
        variables = model.addVars(
            keys, lb=lb, ub=ub, obj=obj, vtype=vtype)
    else:
        variables = model.addVars(
            keys, lb=lb, ub=ub, obj=obj, vtype=vtype, name=name)
    # Gurobi's tupledict retains both its base dictionary and a positional
    # index.  Transfer only the Var references through a compact list, release
    # both tupledict tables, and then construct the one plain mapping callers
    # retain.  This avoids a three-hash-table transient peak without changing
    # the public return type.
    created = [variables[key] for key in keys]
    del variables
    return dict(zip(keys, created))


def _expanded_queue_view(states: Mapping, compression) -> ExpandedStateView:
    """Expose compressed q states on their original minute keys lazily."""
    segments = (
        StateSegment(
            stream_key=(segment.omega, segment.station),
            state_key=segment.key,
            start=segment.start,
            end=segment.end - 1,
            has_event=True)
        for segment in compression.iter_segments()
    )
    return ExpandedStateView(
        states, segments, validate_contiguous=False)


def _compressed_load_plan(
        train_l_keys: Iterable[tuple], b_keys: Iterable[tuple],
        a_keys: Iterable[tuple], load_groups: Mapping,
        station_of: Mapping) -> tuple[tuple[StateSegment, ...], dict, dict]:
    """Build exact event segments and their b/a incidence lists."""
    board_by_stream_minute = defaultdict(list)
    alight_by_stream_minute = defaultdict(list)
    train_minutes = defaultdict(list)
    for om, raw_vehicle, raw_minute in train_l_keys:
        vehicle, minute = str(raw_vehicle), int(raw_minute)
        if not vehicle.startswith("B_"):
            train_minutes[(om, vehicle)].append(minute)

    for family, keys, target in (
            ("b", b_keys, board_by_stream_minute),
            ("a", a_keys, alight_by_stream_minute)):
        for key in keys:
            om, raw_vehicle, raw_minute, station = key
            vehicle, minute = str(raw_vehicle), int(raw_minute)
            if vehicle.startswith("B_"):
                continue
            # Cache keys are the authoritative physical action support.  Most
            # train events coincide with the single arrival event retained in
            # ``station_of``.  A causal termination/detrain event can instead
            # occur at a scheduled departure or information-revelation minute
            # inside that dwell, where ``station_of`` deliberately has no
            # entry.  Retain that a-only event as long as its train load stream
            # exists; otherwise it would be silently omitted from the load
            # recurrence and zero capacity at termination would make the
            # correctly inherited onboard population infeasible.
            if (om, vehicle) in train_minutes:
                target[((om, vehicle), minute)].append(
                    (om, vehicle, minute, station))

    board_minutes_by_stream = defaultdict(list)
    alight_minutes_by_stream = defaultdict(list)
    for stream, minute in board_by_stream_minute:
        board_minutes_by_stream[stream].append(minute)
    for stream, minute in alight_by_stream_minute:
        alight_minutes_by_stream[stream].append(minute)

    segments = []
    for stream_key, raw_minutes in train_minutes.items():
        minutes = tuple(sorted(raw_minutes))
        segments.extend(build_train_load_segments(
            stream_key, minutes,
            boarding_minutes=board_minutes_by_stream.get(stream_key, ()),
            alighting_minutes=alight_minutes_by_stream.get(stream_key, ())))

    for raw_tid, by_omega in load_groups.items():
        tid = str(raw_tid)
        for om, group in by_omega.items():
            stream_key = (om, tid)
            stream_segments = build_bus_load_segments(stream_key, group)
            segments.extend(stream_segments)
            for minute, keys in group.get("board", {}).items():
                board_by_stream_minute[(stream_key, int(minute))].extend(keys)
            for minute, keys in group.get("alight", {}).items():
                alight_by_stream_minute[(stream_key, int(minute))].extend(keys)

    return (tuple(segments), dict(board_by_stream_minute),
            dict(alight_by_stream_minute))


@dataclass(frozen=True, slots=True)
class _BusLoadPlan:
    """Immutable-by-convention, horizon-compressed plan for one bus trip."""

    segments: tuple[StateSegment, ...]
    board: Mapping
    alight: Mapping


def _build_bus_load_plan(tid: str, load_groups: Mapping) -> _BusLoadPlan:
    """Build the bus-only part of ``_compressed_load_plan`` without b/a scans.

    Bus incidence is already stored exactly in ``load_groups``.  Scanning the
    full bus b/a key families is redundant because ``_compressed_load_plan``
    deliberately skips every key whose vehicle starts with ``B_`` and then
    reconstructs the same incidence from this ledger.
    """
    tid = str(tid)
    segments = []
    board_by_stream_minute = {}
    alight_by_stream_minute = {}
    for om, group in load_groups.items():
        stream_key = (om, tid)
        segments.extend(build_bus_load_segments(stream_key, group))
        for minute, keys in group.get("board", {}).items():
            board_by_stream_minute[(stream_key, int(minute))] = tuple(keys)
        for minute, keys in group.get("alight", {}).items():
            alight_by_stream_minute[(stream_key, int(minute))] = tuple(keys)
    return _BusLoadPlan(
        segments=tuple(segments),
        board=board_by_stream_minute,
        alight=alight_by_stream_minute,
    )


def _indexed_bus_load_plan(
        scenario_index: Mapping, tid: str,
        load_groups: Mapping) -> tuple[tuple[StateSegment, ...], Mapping, Mapping]:
    """Return a prebuilt bus plan, with an exact legacy-index fallback."""
    plan = scenario_index.get("bus_load_plans", {}).get(str(tid))
    if plan is None:
        plan = _build_bus_load_plan(str(tid), load_groups)
    return plan.segments, plan.board, plan.alight


def _cache_load_plan(
        cache: Mapping, l_reference_keys: Iterable[tuple],
        b_keys: Iterable[tuple], a_keys: Iterable[tuple],
        load_groups: Mapping, station_of: Mapping
        ) -> tuple[tuple[StateSegment, ...], dict, dict]:
    """Combine cached train intervals with current bus/event incidence."""
    cached_train_segments = cache.get("_train_load_segments")
    if cached_train_segments is None:
        return _compressed_load_plan(
            l_reference_keys, b_keys, a_keys, load_groups, station_of)

    # Exact-state preparation caches the train segments and then releases the
    # dense train ``l`` keys.  Reconstructing incidence with an empty load-key
    # family would therefore discard every train boarding/alighting action
    # from the cached load recurrences.  Use the cached segment streams as the
    # authoritative support and merge their events with the dynamic bus plan.
    train_streams = {
        tuple(segment.stream_key) for segment in cached_train_segments
    }
    train_board = defaultdict(list)
    train_alight = defaultdict(list)
    for keys, target in ((b_keys, train_board), (a_keys, train_alight)):
        for key in keys:
            om, raw_vehicle, raw_minute, _station = key
            vehicle = str(raw_vehicle)
            stream = (om, vehicle)
            if not vehicle.startswith("B_") and stream in train_streams:
                target[(stream, int(raw_minute))].append(key)

    dynamic_segments, bus_board, bus_alight = _compressed_load_plan(
        (), b_keys, a_keys, load_groups, station_of)
    board = defaultdict(list, bus_board)
    alight = defaultdict(list, bus_alight)
    for event, keys in train_board.items():
        board[event].extend(keys)
    for event, keys in train_alight.items():
        alight[event].extend(keys)
    return (tuple(cached_train_segments) + tuple(dynamic_segments),
            dict(board), dict(alight))


def _add_compressed_load_states(
        model: gp.Model, b: Mapping, a: Mapping,
        segments: Sequence[StateSegment], board_events: Mapping,
        alight_events: Mapping, *, ub_by_vehicle: Mapping[str, float],
        objective_scale: float = 0.0, name: str,
        ignore_names: bool = False) -> tuple[dict, ExpandedStateView]:
    keys = tuple(segment.state_key for segment in segments)
    upper = {
        segment.state_key: float(ub_by_vehicle[segment.stream_key[1]])
        for segment in segments
    }
    objective = {
        segment.state_key: float(objective_scale) * segment.length
        for segment in segments
    }
    states = _batch_add_vars(
        model, keys, ub=upper, obj=objective, name=name,
        ignore_names=ignore_names)
    previous_by_stream = {}
    for segment in segments:
        stream = segment.stream_key
        minute = segment.start
        board = gp.quicksum(
            b[key] for key in board_events.get((stream, minute), ())
            if key in b)
        alight = gp.quicksum(
            a[key] for key in alight_events.get((stream, minute), ())
            if key in a)
        previous = previous_by_stream.get(stream, 0)
        recurrence = states[segment.state_key] == previous + board - alight
        if ignore_names:
            model.addConstr(recurrence)
        else:
            model.addConstr(
                recurrence,
                name=f"{name}_flow[{repr(segment.state_key)}]")
        previous_by_stream[stream] = states[segment.state_key]
    return states, ExpandedStateView(states, segments)


def _add_interval_load_capacities(
        model: gp.Model, states: Mapping,
        segments: Sequence[StateSegment], *, y: Mapping,
        spare_capacity: Mapping, config: CGConfig, name: str,
        ignore_names: bool = False) -> int:
    """Replace identical minute capacity rows by exact vehicle intervals."""
    by_vehicle = defaultdict(list)
    for segment in segments:
        by_vehicle[str(segment.stream_key[1])].append(segment)
    row_count = 0
    for vehicle, vehicle_segments in by_vehicle.items():
        additions = defaultdict(list)
        removals = defaultdict(list)
        boundaries = set()
        for segment in vehicle_segments:
            stream = segment.stream_key
            additions[segment.start].append(
                (stream, states[segment.state_key]))
            removals[segment.end + 1].append(stream)
            boundaries.add(segment.start)
            boundaries.add(segment.end + 1)
        ordered = sorted(boundaries)
        active = {}
        for position, start in enumerate(ordered[:-1]):
            for stream in removals.get(start, ()):
                active.pop(stream, None)
            for stream, variable in additions.get(start, ()):
                active[stream] = variable
            stop = ordered[position + 1]
            if stop <= start or not active:
                continue
            lhs = gp.quicksum(active.values())
            if vehicle.startswith("B_"):
                if vehicle not in y:
                    raise KeyError(f"missing activation variable for {vehicle}")
                rhs = float(config.bus_capacity) * y[vehicle]
            else:
                rhs = min(
                    float(spare_capacity.get((vehicle, minute), 0.0))
                    for minute in range(start, stop))
            capacity = lhs <= rhs
            if ignore_names:
                model.addConstr(capacity)
            else:
                model.addConstr(
                    capacity,
                    name=f"{name}[{vehicle},{start},{stop - 1}]")
            row_count += 1
    return row_count


def _add_vehicle_terminal_empty_rows(
        model: gp.Model, terminal_loads: Mapping[str, Iterable[gp.Var]], *,
        name: str, ignore_names: bool = False) -> int:
    """Replace nonnegative singleton-empty rows by one row per vehicle.

    Every load variable has a zero lower bound.  Hence
    ``sum_omega last_l[omega, vehicle] == 0`` holds if and only if every
    singleton ``last_l[omega, vehicle] == 0`` holds.  The aggregation changes
    neither the passenger-flow projection nor fractional solutions; it only
    removes duplicate terminal rows.
    """
    row_count = 0
    for vehicle in sorted(terminal_loads, key=str):
        variables = tuple(terminal_loads[vehicle])
        if not variables:
            continue
        empty = gp.quicksum(variables) == 0.0
        if ignore_names:
            model.addConstr(empty)
        else:
            model.addConstr(empty, name=f"{name}[{vehicle}]")
        row_count += 1
    return row_count


def _add_hybrid_bus_load_states(
        model: gp.Model, b: Mapping, a: Mapping,
        board_events: Mapping, alight_events: Mapping, *, tid: str,
        y: gp.Var, capacity: float, objective_scale: float = 0.0,
        name: str, ignore_names: bool = False) -> dict:
    """Add the exact aggregate/prefix projection of one bus load ledger.

    The current bus-load family is used only for a common inventory objective,
    nonnegativity, a common vehicle capacity, and terminal emptiness.  Under
    those conditions its OD states can be projected out exactly:

    * one aggregate state follows all b/a events on their union timeline;
    * an OD cumulative-flow row is imposed just after each of its alighting
      events (between such events its load can only increase);
    * capacity is checked just after every event containing a boarding
      variable (between those events aggregate load can only decrease); and
    * the final aggregate state is zero.  Since every OD's final cumulative
      load is nonnegative, aggregate emptiness also implies OD-wise emptiness.

    The state at an event minute is the post-event state, matching
    :func:`_add_compressed_load_states`.  Therefore a nonfinal event represents
    exactly the minutes up to, but excluding, the next event; the final event
    represents one minute (and is constrained to zero).
    """
    tid = str(tid)
    capacity = float(capacity)
    if capacity <= 0.0:
        raise ValueError("hybrid bus-load capacity must be positive")

    board_by_minute = defaultdict(list)
    alight_by_minute = defaultdict(list)
    seen_board = set()
    seen_alight = set()

    for source, variables, target, seen, family in (
            (board_events, b, board_by_minute, seen_board, "boarding"),
            (alight_events, a, alight_by_minute, seen_alight, "alighting")):
        for (stream, raw_minute), event_keys in source.items():
            if len(stream) < 2 or str(stream[1]) != tid:
                raise ValueError(
                    f"{family} load incidence belongs to another trip: "
                    f"{stream!r} versus {tid!r}")
            minute = int(raw_minute)
            for event_key in event_keys:
                if event_key not in variables:
                    continue
                if event_key in seen:
                    raise ValueError(
                        f"duplicate {family} load incidence: {event_key!r}")
                seen.add(event_key)
                target[minute].append(event_key)

    missing_board = set(b) - seen_board
    missing_alight = set(a) - seen_alight
    if missing_board or missing_alight:
        raise ValueError(
            "hybrid bus-load incidence does not cover every flow variable: "
            f"missing b={sorted(missing_board, key=repr)[:3]}, "
            f"missing a={sorted(missing_alight, key=repr)[:3]}")

    event_minutes = tuple(sorted(set(board_by_minute) | set(alight_by_minute)))
    if not event_minutes:
        return {}

    state_keys = tuple((tid, minute) for minute in event_minutes)
    objective = {}
    for position, minute in enumerate(event_minutes):
        represented_minutes = (
            event_minutes[position + 1] - minute
            if position + 1 < len(event_minutes) else 1)
        if represented_minutes <= 0:
            raise AssertionError("hybrid bus-load events are not increasing")
        objective[(tid, minute)] = (
            float(objective_scale) * represented_minutes)
    states = _batch_add_vars(
        model, state_keys, ub=capacity, obj=objective, name=name,
        ignore_names=ignore_names)

    prefix_terms = defaultdict(list)
    previous = 0
    for minute in event_minutes:
        board_keys = board_by_minute.get(minute, ())
        alight_keys = alight_by_minute.get(minute, ())
        board = gp.quicksum(b[key] for key in board_keys)
        alight = gp.quicksum(a[key] for key in alight_keys)
        state = states[(tid, minute)]

        recurrence = state == previous + board - alight
        if ignore_names:
            model.addConstr(recurrence)
        else:
            model.addConstr(
                recurrence, name=f"{name}_flow[{tid},{minute}]")

        # Simultaneous b/a variables belong to the same post-event prefix.
        for event_key in board_keys:
            prefix_terms[event_key[0]].append(b[event_key])
        alighting_omegas = set()
        for event_key in alight_keys:
            prefix_terms[event_key[0]].append(-a[event_key])
            alighting_omegas.add(event_key[0])
        for om in sorted(alighting_omegas, key=repr):
            nonnegative = gp.quicksum(prefix_terms[om]) >= 0.0
            if ignore_names:
                model.addConstr(nonnegative)
            else:
                model.addConstr(
                    nonnegative,
                    name=f"{name}_od_prefix[{om},{tid},{minute}]")

        # An alight-only event cannot increase aggregate load.  Every event
        # with an actual b Var, including simultaneous b/a, must be checked.
        if board_keys:
            capacity_row = state <= capacity * y
            if ignore_names:
                model.addConstr(capacity_row)
            else:
                model.addConstr(
                    capacity_row, name=f"{name}_cap[{tid},{minute}]")
        previous = state

    empty = states[(tid, event_minutes[-1])] == 0.0
    if ignore_names:
        model.addConstr(empty)
    else:
        model.addConstr(
            empty, name=f"{name}_empty[{tid},{event_minutes[-1]}]")
    return states


def _hybrid_bus_load_inventory(
        states_by_trip: Mapping, onset: int | None = None) -> gp.LinExpr:
    """Return the exact minute-area represented by hybrid bus states.

    With ``onset`` only the represented minutes t >= onset are counted (Eq. (13)).
    """
    inventory = gp.LinExpr()
    for states in states_by_trip.values():
        ordered = sorted(
            ((int(key[1]), variable) for key, variable in states.items()),
            key=lambda item: item[0])
        for position, (minute, variable) in enumerate(ordered):
            length = (
                ordered[position + 1][0] - minute
                if position + 1 < len(ordered) else 1)
            if length <= 0:
                raise AssertionError("hybrid bus-load states are not increasing")
            if onset is not None:
                length = post_onset_minutes(minute, minute + length, onset)
            if length:
                inventory += length * variable
    return inventory


def post_onset_minutes(start: int, stop: int, onset: int) -> int:
    """Return |{t in [start, stop) : t >= onset}|; Eq. (13) sums z_t over t >= t0."""
    return max(0, int(stop) - max(int(start), int(onset)))


def eq13_inventories(
        onset: int, *, q: Mapping, queue_compression, l: Mapping,
        l_states: Mapping, load_segments: Sequence, compressed_loads: bool,
        hybrid_bus_states: Mapping, a: Mapping, walking_plan, w: Mapping):
    """Return the queue, onboard and walking parts of sum_{t >= t0} z_t.

    Eq. (2) [eq:inventory] and Eq. (13) [eq:obj_db]: every retained state is
    weighted by the number of its represented minutes t >= t0 (compressed queue
    segments [start, end), load segments [start, end], hybrid bus-load states);
    an alighting event at minute u contributes the minutes of its walking interval
    [u, u + sigma) with t >= t0 (walking elimination).
    """
    if queue_compression is not None:
        ends = queue_compression.segment_ends
        queue = gp.quicksum(
            post_onset_minutes(key[2], ends[key], onset) * variable
            for key, variable in q.items())
    else:
        queue = gp.quicksum(
            variable for key, variable in q.items() if int(key[2]) >= onset)
    if compressed_loads:
        load = gp.quicksum(
            post_onset_minutes(segment.start, segment.end + 1, onset)
            * l_states[segment.state_key]
            for segment in load_segments)
    else:
        load = gp.quicksum(
            variable for key, variable in l.items() if int(key[2]) >= onset)
    if hybrid_bus_states:
        load += _hybrid_bus_load_inventory(hybrid_bus_states, onset=onset)
    if walking_plan is not None:
        multipliers = walking_plan.multipliers
        walking = gp.quicksum(
            post_onset_minutes(
                minute, int(minute) + multipliers.get(
                    (om, station, int(minute)), 0), onset) * variable
            for (om, _vehicle, minute, station), variable in a.items())
    else:
        walking = gp.quicksum(
            variable for key, variable in w.items() if int(key[2]) >= onset)
    return queue, load, walking


def eq13_objective(queue, load, walking) -> gp.LinExpr:
    """Z_m of Eq. (13) with z_t of Eq. (2): (kappa [queue + walking] + load) / (T - t0)."""
    return (OUT_OF_VEHICLE_WEIGHT * (queue + walking) + load) / OBJECTIVE_NORMALISER


def pre_onset_normal_station_support(normal: Mapping, omegas, onset: int) -> dict:
    """Stations used by the positive normal actions of ``omegas`` before t0.

    The disrupted CFM fixes the states immediately before onset to their normal
    values (Sec. 3.2.2); the reduced platform subset N^omega must therefore
    contain every station of the normal pre-onset path.
    """
    wanted = set(omegas)
    support = defaultdict(set)
    for family in ("b", "a"):
        for (omega, _vehicle, minute, station), value in normal.get(
                family, {}).items():
            if (omega in wanted and station is not None and
                    int(minute) < onset and float(value) > 0):
                support[omega].add(station)
    for family in ("q", "w"):
        for (omega, station, minute), value in normal.get(family, {}).items():
            if (omega in wanted and station is not None and
                    int(minute) < onset and float(value) > 0):
                support[omega].add(station)
    return {omega: tuple(sorted(stations))
            for omega, stations in support.items()}


def fix_pre_onset_to_normal(
        families: Mapping[str, Mapping], normal: Mapping, onset: int,
        omegas, *, label: str) -> int:
    """Fix boarding, alighting and exit actions before t0 to the normal trajectory.

    Sec. 3.2.2: the disrupted CFM is restricted to t >= t0 and the states
    immediately before onset are fixed to their normal values.  With the same
    demand and the actions of every minute t < t0 fixed, the queue, onboard and
    walking states at t0 - 1 equal the normal states.  Every positive normal
    action of a modelled OD must exist in the model support.
    """
    wanted = set(omegas)
    count = 0
    for family in ("b", "a", "od"):
        variables = families.get(family) or {}
        if not variables:
            continue
        normal_values = normal.get(family, {})
        position = 1 if family == "od" else 2
        for key, variable in variables.items():
            if int(key[position]) >= onset:
                continue
            value = float(normal_values.get(key, 0.0))
            variable.LB = value
            variable.UB = value
            count += 1
        for key, value in normal_values.items():
            if (key[0] in wanted and int(key[position]) < onset and
                    float(value) > 0 and key not in variables):
                raise ValueError(
                    f"{label}: normal pre-onset {family} action {key} is "
                    "outside the model support")
    return count


def _reconstruct_hybrid_bus_loads(
        load_groups: Mapping, b: Mapping, a: Mapping, *,
        tolerance: float) -> dict:
    """Reconstruct positive OD-minute bus loads from the solved flow ledger."""
    result = {}
    zero_tolerance = max(float(tolerance), 1e-9)
    for raw_tid, by_omega in load_groups.items():
        tid = str(raw_tid)
        for om, group in by_omega.items():
            current = 0.0
            for raw_minute in group["times"]:
                minute = int(raw_minute)
                current += math.fsum(
                    float(b[key].X)
                    for key in group["board"].get(minute, ())
                    if key in b)
                current -= math.fsum(
                    float(a[key].X)
                    for key in group["alight"].get(minute, ())
                    if key in a)
                if abs(current) <= zero_tolerance:
                    current = 0.0
                if current > 0.0:
                    result[(om, tid, minute)] = SimpleNamespace(X=current)
    return result


def _bulk_dual_dict(
        model: gp.Model, handles: Mapping,
        constraints: Sequence[gp.Constr]) -> dict:
    """Read one constraint family's duals with a single Gurobi API call.

    ``handles`` and ``constraints`` must have the same insertion order.  The
    returned plain dict therefore preserves the historical tuple-key order and
    float-valued public API while avoiding one Python-to-Gurobi call per row.
    """
    if len(handles) != len(constraints):
        raise AssertionError("dual handle and constraint sequences differ")
    if not constraints:
        return {}
    values = model.getAttr("Pi", constraints)
    if len(values) != len(constraints):
        raise AssertionError("Gurobi returned an incomplete dual vector")
    return {
        key: float(value) for key, value in zip(handles, values)
    }


class IncrementalRMP:
    """Persistent LP master whose fixed support is built exactly once.

    Only complete macro-columns are appended.  A column adds its shared ``y``
    and, for every duration in its information cell, the bus-local ``b/a/l``
    variables and local constraints.  Queue, walking, and depot coefficients
    are attached to the fixed coupling rows with :class:`gurobipy.Column`.
    """

    def __init__(self, problem: CGProblem, config: CGConfig | None,
                 columns: Iterable[ColumnSpec],
                 depot_row_keys: Iterable[tuple[str, str, int]] | None = None,
                 pricing_index: Mapping | None = None):
        if _has_passenger_na_metadata(problem.scenarios):
            raise ValueError(
                "IncrementalRMP cannot attach a second duration row to an "
                "already-created shared passenger Var; use solve_rmp or let "
                "run_column_generation select its causal rebuild fallback")
        self.problem = problem
        self.config = _cfg(config)
        self.full_caches = _require_full_caches(problem)
        materialised = tuple(columns)
        self.columns = {column.key: column for column in materialised}
        if len(self.columns) != len(materialised):
            raise ValueError("duplicate persistent-RMP column keys")
        self.depot_row_keys = frozenset(
            build_depot_row_keys(problem, self.config)
            if depot_row_keys is None else
            ((str(sid), str(depot), int(minute))
             for sid, depot, minute in depot_row_keys))
        self.depot_row_index = _index_depot_row_minutes(
            self.depot_row_keys)
        # ``run_column_generation`` already needs this exact immutable lookup
        # for pricing.  Reuse it when supplied instead of materialising a
        # second multi-scenario copy inside the persistent master.
        self.pricing_index = (
            build_pricing_index(
                problem, include_profile_rows=False,
                eliminate_w=self.config.eliminate_w,
                compress_q=self.config.compress_queue_states,
                sigma=self.config.sigma,
                depot_row_keys=self.depot_row_keys)
            if pricing_index is None else
            pricing_index)
        self.scenarios_by_id = problem.scenarios_by_id
        self.active_columns: set[ColumnKey] = set()
        self.y_by_column: dict[ColumnKey, gp.Var] = {}
        self.scenario_column_vars: dict[str, dict[str, gp.Var]] = {
            str(scenario["id"]): {} for scenario in problem.scenarios}
        self.blocks = {}
        self.dual_handles = {"q": {}, "w": {}, "depot": {}}
        self.base_build_seconds = 0.0
        self.column_add_seconds = 0.0
        self.cumulative_solve_seconds = 0.0
        self.solve_count = 0
        self.closed = False

        started = time.time()
        self.model = _new_model("StandaloneExactCGPersistentRMP", problem)
        model = self.model
        model.Params.IgnoreNames = int(self.config.ignore_names)
        model.Params.OutputFlag = int(self.config.output_flag)
        # Presolve, aggregation, and sparsification stay automatic.  Method and
        # LPWarmStart also stay at their defaults unless an explicit RMP-only
        # pilot setting was requested.
        _apply_rmp_lp_parameters(model, self.config)
        if self.config.time_limit is not None:
            model.Params.TimeLimit = float(self.config.time_limit)
        model.Params.OptimalityTol = float(self.config.optimality_tolerance)
        model.Params.FeasibilityTol = float(self.config.feasibility_tolerance)

        self.depots = tuple(str(depot) for depot in problem.depots)
        fixed_counts = (
            None if problem.fixed_depot_counts is None else
            {str(depot): float(value)
             for depot, value in problem.fixed_depot_counts.items()})
        self.x = {}
        for depot in self.depots:
            fixed_value = (None if fixed_counts is None else
                           float(fixed_counts.get(depot, 0.0)))
            self.x[depot] = model.addVar(
                lb=(0.0 if fixed_value is None else fixed_value),
                ub=(float(self.config.max_total_buses)
                    if fixed_value is None else fixed_value),
                vtype=GRB.CONTINUOUS, name=f"x[{depot}]")
        model.addConstr(
            gp.quicksum(self.x.values()) == self.config.max_total_buses,
            name="fleet_budget")
        self._build_fixed_blocks()
        # Coupling-row handles never change after the fixed block is built.
        # Retain one compact ordered reference list per family so every RMP
        # solve can use Gurobi's bulk attribute API without rebuilding it.
        self._dual_constraint_lists = {
            family: list(handles.values())
            for family, handles in self.dual_handles.items()
        }
        model.update()
        self.base_build_seconds = time.time() - started

    def _build_fixed_blocks(self) -> None:
        model = self.model
        config = self.config
        wdata = self.problem.wdata
        omega_map = wdata["Omega_Map"]
        station_of = wdata["station_of"]
        train_active_times = wdata["train_active_times"]
        train_dest_map = wdata["train_dest_map"]
        train_dwells = wdata["train_dwells"]
        trains = tuple(wdata["Vm"])
        gamma = tuple(wdata["Gamma"])
        start_min = int(wdata["START_MIN"])
        cumulative = wdata["CumD"]
        demand = wdata["D"]

        # The depot row universe is fixed from the complete candidate pool,
        # including rows whose current left-hand side contains no y variable.
        for sid, depot, minute in sorted(self.depot_row_keys):
            if depot not in self.x:
                raise ValueError(
                    f"depot row {(sid, depot, minute)} uses unknown depot")
            self.dual_handles["depot"][(sid, depot, minute)] = model.addConstr(
                -self.x[depot] <= 0.0,
                name=f"depot_t[{sid},{depot},{minute}]")

        for scenario in self.problem.scenarios:
            sid = str(scenario["id"])
            cache = self.full_caches[sid]
            termination_alight_keys = (
                _passenger_action_support_termination_keys(scenario))
            # Pricing can use a q/w dual only where some candidate bus b/a
            # event has a nonzero coefficient.  Build that exact support one
            # scenario at a time.  Every retained projected primal row is
            # created, but Python keeps handles only for bus-touched rows.
            scenario_index = self.pricing_index[sid]
            q_dual_support = set()
            w_dual_support = set()
            for keys in scenario_index["b"].values():
                q_dual_support.update(
                    (om, station, int(minute))
                    for om, _vehicle, minute, station in keys)
            for keys in scenario_index["a"].values():
                for om, _vehicle, minute, station in keys:
                    minute = int(minute)
                    q_dual_support.add(
                        (om, station, minute + self.config.sigma))
                    if not config.eliminate_w:
                        w_dual_support.add((om, station, minute))
                        w_dual_support.add(
                            (om, station, minute + self.config.sigma))
            probability = float(scenario.get(
                "probability", 1.0 / max(len(self.problem.scenarios), 1)))
            horizon = float(cache["T"])
            if probability < 0 or horizon <= 0:
                raise ValueError(f"{sid}: invalid probability or horizon")
            coefficient = probability / horizon
            walking_plan = (
                scenario_index.get("walking_plan")
                if config.eliminate_w else None)
            if config.eliminate_w and walking_plan is None:
                walking_plan = cache.get("_walking_plan")
            if config.eliminate_w and walking_plan is None:
                walking_plan = build_walking_elimination_plan(
                    w_rows=cache.get("w_var_keys", ()),
                    alighting_keys=cache.get("a_var_keys", ()),
                    omega_map=omega_map, gamma=gamma, sigma=config.sigma)
            omega_set = tuple(scenario.get("Omega_m", ()))
            spare_capacity = scenario.get("spare_capacity", {})

            b_keys = _unique_normalised_keys((
                (key[0], str(key[1]), int(key[2]), key[3])
                for key in cache.get("b_var_keys", ())
                if not str(key[1]).startswith("B_")
            ), f"{sid} fixed b")
            a_keys = _unique_normalised_keys((
                (key[0], str(key[1]), int(key[2]), key[3])
                for key in cache.get("a_var_keys", ())
                if not str(key[1]).startswith("B_")
            ), f"{sid} fixed a")
            l_reference_keys = _unique_normalised_keys((
                (key[0], str(key[1]), int(key[2]))
                for key in cache.get("l_var_keys", ())
                if not str(key[1]).startswith("B_")
            ), f"{sid} fixed l")
            q_reference_keys = _unique_normalised_keys((
                (key[0], key[1], int(key[2]))
                for key in cache.get("q_var_keys", ())
            ), f"{sid} q")
            queue_compression = scenario_index.get("queue_compression")
            q_keys = (queue_compression.model_keys
                      if queue_compression is not None else q_reference_keys)
            w_keys = (() if config.eliminate_w else _unique_normalised_keys((
                (key[0], key[1], int(key[2]))
                for key in cache.get("w_var_keys", ())
            ), f"{sid} w"))
            od_keys = _unique_normalised_keys((
                (key[0], int(key[1]))
                for key in cache.get("od_var_keys", ())
            ), f"{sid} od")

            b_ub = {key: max(float(cumulative.get(
                (key[0], key[2]), 0.0)), 0.0) for key in b_keys}
            a_ub = {key: max(float(cumulative.get(
                (key[0], key[2]), 0.0)), 0.0) for key in a_keys}
            a_obj = ({
                key: coefficient * walking_plan.multipliers.get(
                    (key[0], key[3], key[2]), 0)
                for key in a_keys
            } if walking_plan is not None else 0.0)
            od_ub = {key: max(float(cumulative.get(key, 0.0)), 0.0)
                     for key in od_keys}
            b = _batch_add_vars(
                model, b_keys, ub=b_ub, name=f"b_fixed_{sid}")
            a = _batch_add_vars(
                model, a_keys, ub=a_ub, obj=a_obj, name=f"a_fixed_{sid}")
            if config.compress_load_states:
                load_segments, load_board, load_alight = _cache_load_plan(
                    cache, l_reference_keys, b_keys, a_keys, {}, station_of)
                load_ub = {
                    segment.stream_key[1]: config.train_capacity
                    for segment in load_segments
                }
                _l_states, l = _add_compressed_load_states(
                    model, b, a, load_segments, load_board, load_alight,
                    ub_by_vehicle=load_ub, objective_scale=coefficient,
                    name=f"l_fixed_{sid}")
            else:
                l = _batch_add_vars(
                    model, l_reference_keys, ub=float(config.train_capacity),
                    obj=coefficient, name=f"l_fixed_{sid}")
            q = _batch_add_vars(
                model, q_keys,
                obj=({key: coefficient * queue_compression.weights[key]
                      for key in q_keys}
                     if queue_compression is not None else coefficient),
                name=f"q_{sid}")
            w = _batch_add_vars(
                model, w_keys, obj=coefficient, name=f"w_{sid}")
            od = _batch_add_vars(
                model, od_keys, ub=od_ub, name=f"od_{sid}")
            if not config.compress_load_states:
                train_board_events = defaultdict(list)
                train_alight_events = defaultdict(list)
                for key, variable in b.items():
                    train_board_events[(key[0], str(key[1]),
                                        int(key[2]))].append(variable)
                for key, variable in a.items():
                    train_alight_events[(key[0], str(key[1]),
                                         int(key[2]))].append(variable)
                for om in omega_set:
                    if om not in omega_map:
                        continue
                    for train in trains:
                        if train not in train_active_times:
                            continue
                        active_times = train_active_times[train]
                        for minute in active_times:
                            load_key = (om, str(train), int(minute))
                            if load_key not in l:
                                continue
                            event_key = (om, str(train), int(minute))
                            board = gp.quicksum(
                                train_board_events.get(event_key, ()))
                            alight = gp.quicksum(
                                train_alight_events.get(event_key, ()))
                            if minute == active_times.start:
                                model.addConstr(
                                    l[load_key] == board - alight,
                                    name=(f"trlb0[{sid},{om},{train},"
                                          f"{minute}]"))
                            else:
                                previous = l.get(
                                    (om, str(train), int(minute) - 1))
                                if previous is not None:
                                    model.addConstr(
                                        l[load_key] == previous + board - alight,
                                        name=(f"trlb[{sid},{om},{train},"
                                              f"{minute}]"))

            if config.compress_load_states:
                _add_interval_load_capacities(
                    model, _l_states, load_segments, y={},
                    spare_capacity=spare_capacity, config=config,
                    name=f"tr_cap_{sid}")
            else:
                train_loads = defaultdict(list)
                for (_om, train, minute), variable in l.items():
                    train_loads[(train, minute)].append(variable)
                for (train, minute), variables in train_loads.items():
                    model.addConstr(
                        gp.quicksum(variables) <= float(
                            spare_capacity.get((train, minute), 0.0)),
                        name=f"tr_cap[{sid},{train},{minute}]")

            n_omega = cache.get("N_omega", {})
            boarding_index = cache.get("boarding_by_od_t_j", {})
            alighting_index = cache.get("alighting_by_od_t_j", {})
            if queue_compression is not None:
                q_flow_rows = queue_compression.iter_segments()
            else:
                q_flow_rows = (
                    SimpleNamespace(
                        omega=om, station=station, start=int(minute),
                        key=(om, station, int(minute)),
                        predecessor=((om, station, int(minute) - 1)
                                     if position > 0 else None))
                    for om in omega_set if om in omega_map
                    for station in n_omega.get(om, set())
                    for position, minute in enumerate(gamma)
                    if (om, station, int(minute)) in q)
            for segment in q_flow_rows:
                om, station, minute = (
                    segment.omega, segment.station, int(segment.start))
                destination = omega_map[om][1]
                board = gp.quicksum(
                    b[(om, str(vehicle), minute, station)]
                    for vehicle in boarding_index.get(
                        (om, minute, station), ())
                    if (om, str(vehicle), minute, station) in b)
                alight = gp.LinExpr()
                if minute >= start_min + config.sigma:
                    event_minute = minute - config.sigma
                    for vehicle in alighting_index.get(
                            (om, event_minute, station), ()):
                        key = (om, str(vehicle), event_minute, station)
                        if key in a:
                            alight += a[key]
                previous = q.get(segment.predecessor, 0)
                rhs = previous + float(demand.get(
                    (om, station, minute), 0.0)) + alight - board
                if station == destination:
                    rhs -= od.get((om, minute), 0)
                constraint = model.addConstr(
                    q[segment.key] == rhs,
                    name=f"q_flow[{sid},{om},{station},{minute}]")
                local_row_key = (om, station, minute)
                if local_row_key in q_dual_support:
                    self.dual_handles["q"][
                        (sid,) + local_row_key] = constraint

            if not config.eliminate_w:
                for om in omega_set:
                    if om not in omega_map:
                        continue
                    destination = omega_map[om][1]
                    for station in n_omega.get(om, set()):
                        if station == destination:
                            continue
                        for position, minute in enumerate(gamma):
                            w_key = (om, station, int(minute))
                            if w_key not in w:
                                continue
                            alight_in = gp.quicksum(
                                a[(om, str(vehicle), int(minute), station)]
                                for vehicle in alighting_index.get(
                                    (om, minute, station), ())
                                if (om, str(vehicle), int(minute), station) in a)
                            alight_out = gp.LinExpr()
                            if minute >= start_min + config.sigma:
                                event_minute = int(minute) - config.sigma
                                for vehicle in alighting_index.get(
                                        (om, event_minute, station), ()):
                                    key = (om, str(vehicle), event_minute, station)
                                    if key in a:
                                        alight_out += a[key]
                            previous = (w.get(
                                (om, station, int(minute) - 1), 0)
                                if position > 0 else 0)
                            constraint = model.addConstr(
                                w[w_key] == previous + alight_in - alight_out,
                                name=f"w_flow[{sid},{om},{station},{minute}]")
                            local_row_key = (om, station, int(minute))
                            if local_row_key in w_dual_support:
                                self.dual_handles["w"][
                                    (sid,) + local_row_key] = constraint

            q_dual_support.clear()
            w_dual_support.clear()
            for (om, vehicle, minute, station), variable in b.items():
                if om in omega_map and station == omega_map[om][1]:
                    model.addConstr(
                        variable == 0.0,
                        name=f"no_board_dest[{sid},{om},{vehicle},{minute}]")
                if _physical_train_boarding_forbidden(
                        scenario, vehicle, minute):
                    model.addConstr(
                        variable == 0.0,
                        name=(f"no_board_withdrawn["
                              f"{sid},{om},{vehicle},{minute}]"))
            for (om, vehicle, minute, station), variable in a.items():
                if _forbidden_origin_alight(
                        (om, vehicle, minute, station), omega_map,
                        termination_alight_keys):
                    model.addConstr(
                        variable == 0.0,
                        name=f"no_alight_orig[{sid},{om},{vehicle},{minute}]")

            train_terminal_loads = defaultdict(list)
            for om in omega_set:
                if om not in omega_map:
                    continue
                for train in trains:
                    if train not in train_active_times:
                        continue
                    for dwell in train_dwells.get(train, ()):
                        station = dwell["station"]
                        times = tuple(dwell["times"])
                        incoming = l.get(
                            (om, str(train), int(times[0]) - 1), 0)
                        alight = [
                            a[(om, str(train), int(minute), station)]
                            for minute in times
                            if (om, str(train), int(minute), station) in a]
                        if alight:
                            model.addConstr(
                                gp.quicksum(alight) <= incoming,
                                name=f"dwell[{sid},{om},{train},{station}]")
                    if train in train_dest_map:
                        last = train_active_times[train].stop - 1
                        key = (om, str(train), int(last))
                        if key in l:
                            train_terminal_loads[str(train)].append(l[key])

            _add_vehicle_terminal_empty_rows(
                model, train_terminal_loads,
                name=f"tr_empty_{sid}",
                ignore_names=config.ignore_names)

            # The variables and constraints above are owned by the Gurobi
            # model.  Keeping the Python key->Var dictionaries for every
            # scenario duplicates a very large fixed support and is not used
            # after construction.  Only these two scalars are required when
            # later macro-columns are appended.
            self.blocks[sid] = {
                "probability": probability,
                "horizon": horizon,
                "walking_plan": walking_plan,
            }

        # All fixed objective coefficients were attached in the batched
        # addVars calls; future bus-l variables do the same on insertion.
        model.ModelSense = GRB.MINIMIZE

    @staticmethod
    def _column_for_rows(coefficients, constraints) -> gp.Column | None:
        return gp.Column(coefficients, constraints) if constraints else None

    def add_columns(self, keys: Iterable[ColumnKey]) -> int:
        if self.closed:
            raise RuntimeError("persistent RMP is closed")
        requested = [_normalise_key(key) for key in keys]
        requested = [key for key in requested if key not in self.active_columns]
        unknown = [key for key in requested if key not in self.columns]
        if unknown:
            raise ValueError(f"columns outside persistent pool: {unknown[:3]}")
        if not requested:
            return 0

        started = time.time()
        model = self.model
        config = self.config
        cumulative = self.problem.wdata["CumD"]
        omega_map = self.problem.wdata["Omega_Map"]
        station_of = self.problem.wdata["station_of"]
        gamma = tuple(self.problem.wdata["Gamma"])
        first = int(gamma[0]) if gamma else 0
        after_last = int(gamma[-1]) + 1 if gamma else 0

        for key in requested:
            column = self.columns[key]
            block, tid, cell = key
            depot_constraints = []
            depot_coefficients = []
            per_duration = []
            for duration in cell:
                sid, scenario = _scenario_for_duration(
                    column, duration, self.scenarios_by_id)
                trip = _trip_lookup(scenario, tid)
                if trip is None:
                    raise ValueError(f"{key}: trip missing from {sid}")
                raw_begin, raw_end = _depot_interval(trip, config)
                begin = max(raw_begin, first)
                end = min(raw_end, after_last)
                depot = str(trip["Depot"])
                for minute in _depot_minutes_in_interval(
                        self.depot_row_index, sid, depot, begin, end):
                    row = self.dual_handles["depot"].get(
                        (sid, depot, minute))
                    if row is None:
                        raise AssertionError(
                            f"missing fixed depot row for {(sid, depot, minute)}")
                    depot_constraints.append(row)
                    depot_coefficients.append(1.0)
                per_duration.append((duration, sid, scenario, trip))
            y_column = self._column_for_rows(
                depot_coefficients, depot_constraints)
            y = model.addVar(
                lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                name=("" if config.ignore_names else
                      f"y[{block},{tid}|{'-'.join(map(str, cell))}]"),
                column=y_column)
            self.y_by_column[key] = y

            for duration, sid, scenario, trip in per_duration:
                if tid in self.scenario_column_vars[sid]:
                    raise ValueError(
                        f"scenario {sid}, trip {tid} belongs to two active cells")
                self.scenario_column_vars[sid][tid] = y
                index = self.pricing_index[sid]
                b_keys = tuple(index["b"].get(tid, ()))
                a_keys = tuple(index["a"].get(tid, ()))
                load_groups = index["load_groups"].get(tid, {})
                coefficient = float(self.blocks[sid]["probability"]) / float(
                    self.blocks[sid]["horizon"])
                walking_plan = self.blocks[sid].get("walking_plan")

                b = {}
                for b_key in b_keys:
                    om, _vehicle, minute, station = b_key
                    rows, coefficients = [], []
                    row = self.dual_handles["q"].get(
                        (sid, om, station, minute))
                    if row is not None:
                        rows.append(row)
                        coefficients.append(1.0)
                    b[b_key] = model.addVar(
                        lb=0.0,
                        ub=max(float(cumulative.get((om, minute), 0.0)), 0.0),
                        vtype=GRB.CONTINUOUS,
                        name=("" if config.ignore_names else
                              f"b[{sid},{om},{tid},{minute},{station}]"),
                        column=self._column_for_rows(coefficients, rows))

                a = {}
                for a_key in a_keys:
                    om, _vehicle, minute, station = a_key
                    rows, coefficients = [], []
                    targets = [
                        ("q", (sid, om, station, minute + config.sigma), -1.0)]
                    if not config.eliminate_w:
                        targets.extend((
                            ("w", (sid, om, station, minute), -1.0),
                            ("w", (sid, om, station,
                                   minute + config.sigma), 1.0)))
                    for family, row_key, row_coefficient in targets:
                        row = self.dual_handles[family].get(row_key)
                        if row is not None:
                            rows.append(row)
                            coefficients.append(row_coefficient)
                    a[a_key] = model.addVar(
                        lb=0.0,
                        ub=max(2.0 * float(
                            cumulative.get((om, minute), 0.0)), 0.0),
                        obj=(coefficient * walking_plan.multipliers.get(
                            (om, station, minute), 0)
                             if walking_plan is not None else 0.0),
                        vtype=GRB.CONTINUOUS,
                        name=("" if config.ignore_names else
                              f"a[{sid},{om},{tid},{minute},{station}]"),
                        column=self._column_for_rows(coefficients, rows))

                bus_terminal_loads = []
                if (config.hybrid_bus_load_states or
                        config.compress_load_states):
                    load_segments, load_board, load_alight = (
                        _indexed_bus_load_plan(index, tid, load_groups))

                if config.hybrid_bus_load_states:
                    _add_hybrid_bus_load_states(
                        model, b, a, load_board, load_alight, tid=tid, y=y,
                        capacity=config.bus_capacity,
                        objective_scale=coefficient,
                        name=("" if config.ignore_names else
                              f"L_{sid}_{tid}"),
                        ignore_names=config.ignore_names)
                elif config.compress_load_states:
                    load_ub = {tid: config.bus_capacity}
                    _l_states, l = _add_compressed_load_states(
                        model, b, a, load_segments, load_board, load_alight,
                        ub_by_vehicle=load_ub, objective_scale=coefficient,
                        name=("" if config.ignore_names else
                              f"l_{sid}_{tid}"),
                        ignore_names=config.ignore_names)
                    for om, group in load_groups.items():
                        if group["times"]:
                            last = int(group["times"][-1])
                            bus_terminal_loads.append(l[(om, tid, last)])
                else:
                    l_keys = set(index["l"].get(tid, ()))
                    for om, group in load_groups.items():
                        l_keys.update((om, tid, int(minute))
                                      for minute in group["times"])
                    ordered_l_keys = tuple(sorted(l_keys, key=repr))
                    l = _batch_add_vars(
                        model, ordered_l_keys, ub=float(config.bus_capacity),
                        obj=coefficient,
                        name=("" if config.ignore_names else
                              f"l_{sid}_{tid}"),
                        ignore_names=config.ignore_names)

                    for om, group in load_groups.items():
                        previous = 0
                        for index_position, minute in enumerate(group["times"]):
                            board = gp.quicksum(
                                b[event]
                                for event in group["board"].get(minute, ())
                                if event in b)
                            alight = gp.quicksum(
                                a[event]
                                for event in group["alight"].get(minute, ())
                                if event in a)
                            recurrence = (
                                l[(om, tid, minute)] ==
                                previous + board - alight)
                            if config.ignore_names:
                                model.addConstr(recurrence)
                            else:
                                model.addConstr(
                                    recurrence,
                                    name=(f"lb0[{sid},{om},{tid},{minute}]"
                                          if index_position == 0 else
                                          f"lb[{sid},{om},{tid},{minute}]"))
                            previous = l[(om, tid, minute)]
                        if group["times"]:
                            last = int(group["times"][-1])
                            bus_terminal_loads.append(l[(om, tid, last)])

                if not config.hybrid_bus_load_states:
                    _add_vehicle_terminal_empty_rows(
                        model, {tid: bus_terminal_loads},
                        name=("" if config.ignore_names else
                              f"bus_empty_{sid}"),
                        ignore_names=config.ignore_names)

                if (not config.hybrid_bus_load_states and
                        config.compress_load_states):
                    _add_interval_load_capacities(
                        model, _l_states, load_segments, y={tid: y},
                        spare_capacity={}, config=config,
                        name=("" if config.ignore_names else
                              f"bus_cap_{sid}_{tid}"),
                        ignore_names=config.ignore_names)
                elif not config.hybrid_bus_load_states:
                    loads_by_minute = defaultdict(list)
                    for (_om, _vehicle, minute), variable in l.items():
                        loads_by_minute[minute].append(variable)
                    for minute, variables in loads_by_minute.items():
                        capacity = (
                            gp.quicksum(variables) <= config.bus_capacity * y)
                        if config.ignore_names:
                            model.addConstr(capacity)
                        else:
                            model.addConstr(
                                capacity,
                                name=f"bus_cap[{sid},{tid},{minute}]")
                for (om, _vehicle, minute, station), variable in b.items():
                    if om in omega_map and station == omega_map[om][1]:
                        no_board = variable == 0.0
                        if config.ignore_names:
                            model.addConstr(no_board)
                        else:
                            model.addConstr(
                                no_board,
                                name=(f"no_board_dest["
                                      f"{sid},{om},{tid},{minute}]"))
                for (om, _vehicle, minute, station), variable in a.items():
                    if om in omega_map and station == omega_map[om][0]:
                        no_alight = variable == 0.0
                        if config.ignore_names:
                            model.addConstr(no_alight)
                        else:
                            model.addConstr(
                                no_alight,
                                name=(f"no_alight_orig["
                                      f"{sid},{om},{tid},{minute}]"))
                minimum_service = (
                    gp.quicksum(b.values()) >=
                    config.min_trip_boarding * y)
                if config.ignore_names:
                    model.addConstr(minimum_service)
                else:
                    model.addConstr(
                        minimum_service, name=f"min_svc[{sid},{tid}]")
                # The model owns the bus-local variables and constraints.
                # No later persistent-RMP operation reads their Python
                # dictionaries, so retaining them would make memory grow with
                # every active column.

            self.active_columns.add(key)

        # Gurobi applies pending model modifications when optimize() starts.
        # Callers that need exact dimensions before solving use
        # structural_stats(), which deliberately performs an explicit update.
        self.column_add_seconds += time.time() - started
        return len(requested)

    def solve(self) -> dict:
        if self.closed:
            raise RuntimeError("persistent RMP is closed")
        started = time.time()
        self.model.optimize()
        solve_seconds = time.time() - started
        self.cumulative_solve_seconds += solve_seconds
        self.solve_count += 1
        status = _status_name(self.model.Status)
        stats = _model_stats(
            self.model, self.base_build_seconds + self.column_add_seconds)
        stats.update({
            "PersistentRMP": True,
            "BaseBuildSec": float(self.base_build_seconds),
            "ColumnAddSec": float(self.column_add_seconds),
            "LastReoptimizeSec": float(solve_seconds),
            "CumulativeReoptimizeSec": float(self.cumulative_solve_seconds),
            "RMPSolveCount": int(self.solve_count),
            "ActiveColumns": int(len(self.active_columns)),
        })
        duals = {"q": {}, "w": {}, "depot": {}}
        if self.model.Status == GRB.OPTIMAL:
            for family in duals:
                duals[family] = _bulk_dual_dict(
                    self.model, self.dual_handles[family],
                    self._dual_constraint_lists[family])
        primal_hint = (
            {
                "x": {depot: float(variable.X)
                      for depot, variable in self.x.items()},
                "y": {key: float(variable.X)
                      for key, variable in self.y_by_column.items()},
            }
            if int(self.model.SolCount) > 0 else None)
        return {
            "status": status,
            "objective": (float(self.model.ObjVal)
                          if int(self.model.SolCount) > 0 else math.nan),
            "duals": duals,
            "primal_hint": primal_hint,
            "stats": stats,
        }

    def set_lp_tolerances(
            self, *, optimality: float, feasibility: float) -> None:
        """Change only numerical LP tolerances on the persistent master.

        This is used by the two-phase CG search: loose intermediate RMP
        solves may discover columns faster, but a tight re-solve and fresh
        omitted-pool pricing pass are mandatory before certification.
        """
        if self.closed:
            raise RuntimeError("persistent RMP is closed")
        self.model.Params.OptimalityTol = float(optimality)
        self.model.Params.FeasibilityTol = float(feasibility)

    def structural_stats(self) -> dict:
        """Return exact master dimensions without invoking the optimizer."""
        if self.closed:
            raise RuntimeError("persistent RMP is closed")
        self.model.update()
        return {
            "NumVars": int(self.model.NumVars),
            "NumConstrs": int(self.model.NumConstrs),
            "NumNZs": int(self.model.NumNZs),
            "ActiveColumns": int(len(self.active_columns)),
            "BaseBuildSec": float(self.base_build_seconds),
            "ColumnAddSec": float(self.column_add_seconds),
        }

    def close(self) -> None:
        if not self.closed:
            self.model.dispose()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def build_pricing_index(problem: CGProblem, *,
                        include_profile_rows: bool = True,
                        eliminate_w: bool = False,
                        compress_q: bool = False,
                        sigma: int = SIGMA,
                        depot_row_keys: Iterable[tuple[str, str, int]] | None = None
                        ) -> dict:
    """Index bus-local full-cache keys once for repeated pricing.

    ``q_rows`` and ``w_rows`` are needed only when a caller requests a full
    priced-column profile.  Root CG prices with ``keep_profiles=False`` and can
    therefore omit those two full copies of the minute-level row universe.
    The default remains ``True`` for backward compatibility with audit callers.
    """
    indexed = {}
    depot_row_index = (
        None if depot_row_keys is None else
        _index_depot_row_minutes(depot_row_keys))
    for raw_sid, cache in _require_full_caches(problem).items():
        sid = str(raw_sid)
        compacted = bool(cache.get("support_compacted"))
        by_family = {name: defaultdict(list) for name in ("b", "a", "l")}
        for family, cache_key in (("b", "b_var_keys"),
                                  ("a", "a_var_keys"),
                                  ("l", "l_var_keys")):
            for key in cache.get(cache_key, ()):
                if str(key[1]).startswith("B_"):
                    normal = (key if compacted else
                              ((key[0], str(key[1]), int(key[2]), key[3])
                               if family != "l" else
                               (key[0], str(key[1]), int(key[2]))))
                    by_family[family][str(key[1])].append(normal)
        bus_trip_ids = set().union(*(family.keys()
                                     for family in by_family.values()))
        load_groups = aggregate_bus_load_groups(cache, bus_trip_ids)
        scenario_index = {
            "b": {tid: tuple(keys) for tid, keys in by_family["b"].items()},
            "a": {tid: tuple(keys) for tid, keys in by_family["a"].items()},
            "l": {tid: tuple(keys) for tid, keys in by_family["l"].items()},
            "load_groups": load_groups,
            # These plans are dual-independent and shared read-only by every
            # pricing pass and every incremental RMP column insertion.
            "bus_load_plans": {
                str(tid): _build_bus_load_plan(str(tid), by_omega)
                for tid, by_omega in load_groups.items()
            },
        }
        if depot_row_index is not None:
            # Shared immutable-by-convention lookup.  Production pricing uses
            # it to visit only retained clique rows inside a trip interval.
            scenario_index["depot_base_row_index"] = depot_row_index
            scenario_index["depot_row_index"] = depot_row_index
        if eliminate_w:
            walking_plan = cache.get("_walking_plan")
            if walking_plan is None:
                walking_plan = build_walking_elimination_plan(
                    w_rows=cache.get("w_var_keys", ()),
                    alighting_keys=cache.get("a_var_keys", ()),
                    omega_map=problem.wdata["Omega_Map"],
                    gamma=problem.wdata["Gamma"], sigma=sigma)
            scenario_index["walking_plan"] = walking_plan
        if compress_q:
            queue_compression = cache.get("_queue_compression")
            if queue_compression is None:
                queue_compression = build_queue_compression(
                    cache.get("q_var_keys", ()), cache.get("b_var_keys", ()),
                    cache.get("a_var_keys", ()), problem.wdata["D"],
                    problem.wdata["Omega_Map"], sigma)
            scenario_index["queue_compression"] = queue_compression
        if include_profile_rows:
            queue_compression = scenario_index.get("queue_compression")
            scenario_index["q_rows"] = frozenset(
                queue_compression.iter_model_keys()
                if queue_compression is not None else
                ((om, station, int(minute))
                 for om, station, minute in cache.get("q_var_keys", ())))
            if not eliminate_w:
                scenario_index["w_rows"] = frozenset(
                    (om, station, int(minute))
                    for om, station, minute in cache.get("w_var_keys", ()))
        indexed[sid] = scenario_index
    return indexed


def _scenario_for_duration(column: ColumnSpec, duration: int,
                           scenarios_by_id: Mapping[str, Mapping]) -> tuple[str, Mapping]:
    matches = [sid for sid in column.scenario_ids
               if int(scenarios_by_id[sid]["duration"]) == int(duration)]
    if len(matches) != 1:
        raise ValueError(
            f"{column.key}: expected one scenario for D={duration}, got {matches}")
    sid = matches[0]
    return sid, scenarios_by_id[sid]


def _pricing_depot_minutes(
        pricing_index: Mapping, sid: str, depot: str,
        begin: int, end: int) -> Sequence[int]:
    """Use retained depot rows when indexed, else preserve the public fallback."""
    row_index = pricing_index.get(str(sid), {}).get("depot_row_index")
    if row_index is None:
        return range(int(begin), int(end))
    return _depot_minutes_in_interval(
        row_index, str(sid), str(depot), int(begin), int(end))


def _ensure_pricing_depot_index(
        pricing_index: Mapping, duals: Mapping[str, Mapping], *,
        problem: CGProblem, config: CGConfig) -> Mapping:
    """Attach and validate the actual depot-dual support once per pricing pass."""
    depot_duals = duals.get("depot", {})
    if all(
            scenario_index.get("depot_dual_support_mapping") is depot_duals
            for scenario_index in pricing_index.values()):
        return pricing_index
    base_row_index = next((
        scenario_index.get("depot_base_row_index")
        for scenario_index in pricing_index.values()
        if scenario_index.get("depot_base_row_index") is not None
    ), None)
    if base_row_index is None:
        base_row_index = _index_depot_row_minutes(
            build_depot_row_keys(problem, config))
    merged_rows = (
        (sid, depot, minute)
        for (sid, depot), minutes in base_row_index.items()
        for minute in minutes)
    row_index = _index_depot_row_minutes(
        tuple(merged_rows) + tuple(depot_duals.keys()))
    if isinstance(pricing_index, dict) and all(
            isinstance(value, dict) for value in pricing_index.values()):
        # The index is shared only read-only while worker tasks are running.
        # Refresh it here, before dispatch, so retained model caches keep the
        # same top-level object identity across CG iterations.
        for scenario_index in pricing_index.values():
            scenario_index["depot_base_row_index"] = base_row_index
            scenario_index["depot_row_index"] = row_index
            scenario_index["depot_dual_support_mapping"] = depot_duals
        return pricing_index
    return {
        sid: dict(
            scenario_index, depot_base_row_index=base_row_index,
            depot_row_index=row_index,
            depot_dual_support_mapping=depot_duals)
        for sid, scenario_index in pricing_index.items()
    }


def pricing_box_lower_bound(
        problem: CGProblem, config: CGConfig | None, column: ColumnSpec,
        duals: Mapping[str, Mapping], pricing_index: Mapping | None = None) -> float:
    """Certified lower bound obtained by dropping all local constraints."""
    config = _cfg(config)
    if pricing_index is None:
        pricing_index = build_pricing_index(
            problem, eliminate_w=config.eliminate_w,
            compress_q=config.compress_queue_states, sigma=config.sigma)
    pricing_index = _ensure_pricing_depot_index(
        pricing_index, duals, problem=problem, config=config)
    scenarios_by_id = problem.scenarios_by_id
    full_caches = _require_full_caches(problem)
    cumulative = problem.wdata["CumD"]
    omega_map = problem.wdata["Omega_Map"]
    gamma = tuple(problem.wdata["Gamma"])
    first = int(gamma[0]) if gamma else 0
    after_last = int(gamma[-1]) + 1 if gamma else 0
    _block, tid, cell = column.key
    terms = []
    shared_y_terms = []
    for duration in cell:
        sid, scenario = _scenario_for_duration(
            column, duration, scenarios_by_id)
        cache = full_caches[sid]
        trip = _trip_lookup(scenario, tid)
        if trip is None:
            raise ValueError(f"{column.key}: trip missing from {sid}")
        for om, _vehicle, minute, station in pricing_index[sid]["b"].get(tid, ()):
            upper = max(float(cumulative.get((om, minute), 0.0)), 0.0)
            if om in omega_map and station == omega_map[om][1]:
                upper = 0.0
            coefficient = -float(duals.get("q", {}).get(
                (sid, om, station, minute), 0.0))
            terms.append(min(0.0, coefficient * upper))
        for om, _vehicle, minute, station in pricing_index[sid]["a"].get(tid, ()):
            upper = max(float(cumulative.get((om, minute), 0.0)), 0.0)
            if om in omega_map and station == omega_map[om][0]:
                upper = 0.0
            coefficient = float(duals.get("q", {}).get(
                (sid, om, station, minute + config.sigma), 0.0))
            walking_plan = pricing_index[sid].get("walking_plan")
            if walking_plan is not None:
                probability = float(scenario.get(
                    "probability", 1.0 / max(len(problem.scenarios), 1)))
                coefficient += (
                    probability / float(cache["T"]) *
                    walking_plan.multipliers.get(
                        (om, station, int(minute)), 0))
            else:
                coefficient += (
                    float(duals.get("w", {}).get(
                        (sid, om, station, minute), 0.0))
                    - float(duals.get("w", {}).get(
                        (sid, om, station, minute + config.sigma), 0.0)))
            terms.append(min(0.0, coefficient * upper))
        raw_begin, raw_end = _depot_interval(trip, config)
        begin = max(raw_begin, first)
        end = min(raw_end, after_last)
        depot = str(trip["Depot"])
        shared_y_terms.extend(
            -float(duals.get("depot", {}).get(
                (sid, depot, minute), 0.0))
            for minute in _pricing_depot_minutes(
                pricing_index, sid, depot, begin, end))
    terms.append(min(0.0, math.fsum(shared_y_terms)))
    computed = math.fsum(terms)
    safety = BOX_BOUND_REL_SAFETY * max(
        1.0, math.fsum(abs(term) for term in terms))
    return float(computed - safety)


def _has_zero_boarding_support(
        column: ColumnSpec, pricing_index: Mapping) -> bool:
    """Return whether ``column`` has no retained bus-boarding variable.

    Canonical :class:`ColumnSpec` scores count exactly these keys.  The direct
    index check keeps the shortcut safe for callers that supply hand-built
    column metadata with a stale or otherwise inconsistent score.
    """
    if int(column.score) != 0:
        return False
    tid = column.trip_id
    return all(
        not pricing_index[str(sid)]["b"].get(tid, ())
        for sid in column.scenario_ids)


def _price_zero_boarding_support(
        problem: CGProblem, config: CGConfig, column: ColumnSpec,
        duals: Mapping[str, Mapping], pricing_index: Mapping) -> PricingResult:
    """Solve exactly a pricing LP that has no boarding support.

    With no nonnegative boarding variable, the nonnegative load recurrences
    force every alighting and load variable to zero.  The only remaining
    decision is the shared activation ``y`` in ``[0, 1]``.  Its objective
    coefficient is the sum of the depot-row reduced-cost terms over every
    scenario in the information cell, so the exact optimum is
    ``min(0, y_coefficient)``.

    This lean result is used only by the no-profile production pricing path.
    Full-profile audits continue through the ordinary Gurobi model so they
    retain explicit coefficient and primal dictionaries.
    """
    scenarios_by_id = problem.scenarios_by_id
    gamma = tuple(problem.wdata["Gamma"])
    first = int(gamma[0]) if gamma else 0
    after_last = int(gamma[-1]) + 1 if gamma else 0
    _block, tid, cell = column.key
    depot_duals = duals.get("depot", {})
    y_terms = []
    for duration in cell:
        sid, scenario = _scenario_for_duration(
            column, duration, scenarios_by_id)
        trip = _trip_lookup(scenario, tid)
        if trip is None:
            raise ValueError(f"{column.key}: trip missing from {sid}")
        raw_begin, raw_end = _depot_interval(trip, config)
        begin = max(raw_begin, first)
        end = min(raw_end, after_last)
        depot = str(trip["Depot"])
        y_terms.extend(
            -float(depot_duals.get(
                (sid, depot, minute), 0.0))
            for minute in _pricing_depot_minutes(
                pricing_index, sid, depot, begin, end))
    y_coefficient = float(math.fsum(y_terms))
    if not math.isfinite(y_coefficient):
        raise ValueError(
            f"{column.key}: non-finite zero-support depot coefficient")
    y_value = 1.0 if y_coefficient < 0.0 else 0.0
    optimum = min(0.0, y_coefficient)
    return PricingResult(
        key=column.key, status="OPTIMAL", reduced_cost=float(optimum),
        y=y_value, direct_cost=math.nan, q_coefficients={},
        w_coefficients={}, depot_coefficients={}, b_values={}, a_values={},
        l_values={}, solver_lower_bound=float(optimum))


def _pricing_objective(
        problem: CGProblem, config: CGConfig,
        duals: Mapping[str, Mapping], y: gp.Var, local: Mapping,
        first: int, after_last: int,
        pricing_index: Mapping) -> gp.LinExpr:
    """Rebuild only the dual-dependent objective of a retained pricing LP."""
    objective = gp.LinExpr()
    for data in local.values():
        sid = data["sid"]
        probability = float(data["scenario"].get(
            "probability", 1.0 / max(len(problem.scenarios), 1)))
        horizon = float(data["cache"]["T"])
        load_inventory = (
            gp.quicksum(
                segment.length * data["l_states"][segment.state_key]
                for segment in data["load_segments"])
            if config.compress_load_states else
            gp.quicksum(data["l"].values()))
        objective += probability / horizon * load_inventory
        for (om, _vehicle, minute, station), variable in data["b"].items():
            objective += -duals.get("q", {}).get(
                (sid, om, station, minute), 0.0) * variable
        for (om, _vehicle, minute, station), variable in data["a"].items():
            coefficient = duals.get("q", {}).get(
                (sid, om, station, minute + config.sigma), 0.0)
            walking_plan = data["walking_plan"]
            if walking_plan is not None:
                coefficient += (
                    probability / horizon * walking_plan.multipliers.get(
                        (om, station, int(minute)), 0))
            else:
                coefficient += (
                    duals.get("w", {}).get(
                        (sid, om, station, minute), 0.0)
                    - duals.get("w", {}).get(
                        (sid, om, station, minute + config.sigma), 0.0))
            objective += coefficient * variable
        trip = data["trip"]
        raw_begin, raw_end = _depot_interval(trip, config)
        begin = max(raw_begin, first)
        end = min(raw_end, after_last)
        for minute in _pricing_depot_minutes(
                pricing_index, sid, str(trip["Depot"]), begin, end):
            objective += -duals.get("depot", {}).get(
                (sid, str(trip["Depot"]), minute), 0.0) * y
    return objective


class _RetainedPricingModel:
    """One exact local LP whose constraints survive changing RMP duals."""

    def __init__(
            self, problem: CGProblem, config: CGConfig, column: ColumnSpec,
            pricing_index: Mapping, model: gp.Model, y: gp.Var,
            local: Mapping, first: int, after_last: int):
        self.problem = problem
        self.config = config
        self.column = column
        self.pricing_index = pricing_index
        self.model = model
        self.y = y
        self.local = local
        self.first = int(first)
        self.after_last = int(after_last)
        self.closed = False

    def solve(self, duals: Mapping[str, Mapping]) -> PricingResult:
        if self.closed:
            raise RuntimeError(
                f"retained pricing model is closed: {self.column.key}")
        objective = _pricing_objective(
            self.problem, self.config, duals, self.y, self.local,
            self.first, self.after_last, self.pricing_index)
        self.model.setObjective(objective, GRB.MINIMIZE)
        self.model.optimize()
        status = _status_name(self.model.Status)
        if self.model.Status != GRB.OPTIMAL:
            self.dispose()
            raise RuntimeError(
                f"pricing failed for {self.column.key}: {status}")
        return PricingResult(
            key=self.column.key, status=status,
            reduced_cost=float(self.model.ObjVal), y=float(self.y.X),
            direct_cost=math.nan, q_coefficients={}, w_coefficients={},
            depot_coefficients={}, b_values={}, a_values={}, l_values={},
            solver_lower_bound=float(self.model.ObjBound))

    def dispose(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.model.dispose()


def price_column(
        problem: CGProblem, config: CGConfig | None, column: ColumnSpec,
        duals: Mapping[str, Mapping], *, pricing_index: Mapping | None = None,
        keep_profile: bool = True,
        _parameters_preconfigured: bool = False,
        _retain_model: bool = False):
    """Solve one exact trip-information-cell local LP."""
    if keep_profile and _retain_model:
        raise ValueError("a retained pricing model cannot keep a full profile")
    config = _cfg(config)
    if pricing_index is None:
        pricing_index = build_pricing_index(
            problem, eliminate_w=config.eliminate_w,
            compress_q=config.compress_queue_states, sigma=config.sigma)
    pricing_index = _ensure_pricing_depot_index(
        pricing_index, duals, problem=problem, config=config)
    scenarios_by_id = problem.scenarios_by_id
    full_caches = _require_full_caches(problem)
    block, tid, cell = column.key
    model = _new_model(
        "" if config.ignore_names else f"price[{block},{tid}]", problem)
    if not _parameters_preconfigured:
        model.Params.IgnoreNames = int(config.ignore_names)
        model.Params.OutputFlag = int(config.pricing_output_flag)
        model.Params.OptimalityTol = float(config.optimality_tolerance)
        model.Params.FeasibilityTol = float(config.feasibility_tolerance)
    if config.ignore_names:
        y = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
    else:
        y = model.addVar(
            lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")
    local = {}
    cumulative = problem.wdata["CumD"]
    omega_map = problem.wdata["Omega_Map"]
    station_of = problem.wdata["station_of"]
    gamma = tuple(problem.wdata["Gamma"])
    first = int(gamma[0]) if gamma else 0
    after_last = int(gamma[-1]) + 1 if gamma else 0
    passenger_context_by_sid, passenger_contexts = (
        _build_passenger_sharing_contexts(problem.scenarios))
    normal_fixed_variable_count = 0

    for duration in cell:
        sid, scenario = _scenario_for_duration(
            column, duration, scenarios_by_id)
        cache = full_caches[sid]
        trip = _trip_lookup(scenario, tid)
        if trip is None:
            model.dispose()
            raise ValueError(f"{column.key}: trip missing from {sid}")
        scenario_index = pricing_index[sid]
        b_keys = tuple(scenario_index["b"].get(tid, ()))
        a_keys = tuple(scenario_index["a"].get(tid, ()))
        load_groups = scenario_index["load_groups"].get(tid, {})
        b_ub = {key: max(float(cumulative.get(
            (key[0], key[2]), 0.0)), 0.0) for key in b_keys}
        a_ub = {key: max(float(cumulative.get(
            (key[0], key[2]), 0.0)), 0.0) for key in a_keys}
        passenger_context = passenger_context_by_sid.get(sid)
        b = _batch_add_passenger_vars(
            model, b_keys, family="b", sid=sid,
            context=passenger_context, ub=b_ub,
            name=("" if config.ignore_names else f"b_price_{duration}"),
            ignore_names=config.ignore_names)
        a = _batch_add_passenger_vars(
            model, a_keys, family="a", sid=sid,
            context=passenger_context, ub=a_ub,
            name=("" if config.ignore_names else f"a_price_{duration}"),
            ignore_names=config.ignore_names)
        normal_fixed_variable_count += _fix_normal_passenger_decisions(
            scenario, {"b": b, "a": a}, passenger_context)
        terminal_loads = []
        if config.compress_load_states:
            load_segments, load_board, load_alight = _indexed_bus_load_plan(
                scenario_index, tid, load_groups)
            _l_states, l = _add_compressed_load_states(
                model, b, a, load_segments, load_board, load_alight,
                ub_by_vehicle={tid: config.bus_capacity},
                name=("" if config.ignore_names else f"l_price_{duration}"),
                ignore_names=config.ignore_names)
            for om, group in load_groups.items():
                if group["times"]:
                    terminal_loads.append(
                        l[(om, tid, int(group["times"][-1]))])
        else:
            l_keys = set(scenario_index["l"].get(tid, ()))
            for om, group in load_groups.items():
                l_keys.update((om, tid, minute) for minute in group["times"])
            # Preserve the former scalar-addVar insertion order.  Sorting would
            # leave the LP unchanged but could select a different degenerate
            # pricing optimum, which is undesirable for output reproducibility.
            ordered_l_keys = tuple(l_keys)
            l = _batch_add_vars(
                model, ordered_l_keys, ub=float(config.bus_capacity),
                name=("" if config.ignore_names else f"l_price_{duration}"),
                ignore_names=config.ignore_names)
            _l_states = l
            load_segments = ()

            for om, group in load_groups.items():
                previous = 0
                for minute in group["times"]:
                    board = gp.quicksum(
                        b[key] for key in group["board"].get(minute, ())
                        if key in b)
                    alight = gp.quicksum(
                        a[key] for key in group["alight"].get(minute, ())
                        if key in a)
                    model.addConstr(
                        l[(om, tid, minute)] == previous + board - alight)
                    previous = l[(om, tid, minute)]
                if group["times"]:
                    terminal_loads.append(
                        l[(om, tid, int(group["times"][-1]))])
        _add_vehicle_terminal_empty_rows(
            model, {tid: terminal_loads},
            name=("" if config.ignore_names else
                  f"bus_empty_price_{duration}"),
            ignore_names=config.ignore_names)
        if config.compress_load_states:
            _add_interval_load_capacities(
                model, _l_states, load_segments, y={tid: y},
                spare_capacity={}, config=config,
                name=("" if config.ignore_names else
                      f"bus_cap_price_{duration}_{tid}"),
                ignore_names=config.ignore_names)
        else:
            loads_by_minute = defaultdict(list)
            for (_om, _vehicle, minute), variable in l.items():
                loads_by_minute[minute].append(variable)
            for variables in loads_by_minute.values():
                model.addConstr(
                    gp.quicksum(variables) <= config.bus_capacity * y)
        for (om, _vehicle, _minute, station), variable in b.items():
            if om in omega_map and station == omega_map[om][1]:
                model.addConstr(variable == 0.0)
        for (om, _vehicle, _minute, station), variable in a.items():
            if om in omega_map and station == omega_map[om][0]:
                model.addConstr(variable == 0.0)
        if not passenger_contexts:
            model.addConstr(gp.quicksum(b.values()) >=
                            config.min_trip_boarding * y)

        probability = float(scenario.get(
            "probability", 1.0 / max(len(problem.scenarios), 1)))
        horizon = float(cache["T"])
        if probability < 0 or horizon <= 0:
            model.dispose()
            raise ValueError(f"{sid}: invalid probability or horizon")
        walking_plan = scenario_index.get("walking_plan")
        raw_begin, raw_end = _depot_interval(trip, config)
        begin = max(raw_begin, first)
        end = min(raw_end, after_last)
        depot_minutes = tuple(_pricing_depot_minutes(
            pricing_index, sid, str(trip["Depot"]), begin, end))
        local[duration] = {
            "sid": sid, "scenario": scenario, "cache": cache, "trip": trip,
            "b": b, "a": a, "l": l, "l_states": _l_states,
            "load_segments": load_segments, "walking_plan": walking_plan,
            "depot_minutes": depot_minutes,
        }

    _validate_passenger_variable_sharing(passenger_contexts)
    if passenger_contexts:
        boarding = OrderedDict()
        for data in local.values():
            for variable in data["b"].values():
                boarding.setdefault(id(variable), variable)
        model.addConstr(
            gp.quicksum(boarding.values()) >= config.min_trip_boarding * y)

    retained = _RetainedPricingModel(
        problem, config, column, pricing_index, model, y, local,
        first, after_last)
    result = retained.solve(duals)
    status = result.status

    if not keep_profile:
        if _retain_model:
            return result, retained
        retained.dispose()
        return result

    y_value = float(y.X)
    b_values, a_values, l_values = {}, {}, {}
    q_coefficients = defaultdict(float)
    w_coefficients = defaultdict(float)
    depot_coefficients = defaultdict(float)
    direct_cost = 0.0
    for duration, data in local.items():
        sid = data["sid"]
        probability = float(data["scenario"].get(
            "probability", 1.0 / max(len(problem.scenarios), 1)))
        horizon = float(data["cache"]["T"])
        # A lean production pricing index deliberately omits these enormous
        # row sets.  Full-profile audit calls reconstruct only the scenario(s)
        # they inspect, and release the transient sets with this local solve.
        q_rows = pricing_index[sid].get("q_rows")
        if q_rows is None:
            queue_compression = pricing_index[sid].get("queue_compression")
            q_rows = frozenset(
                queue_compression.iter_model_keys()
                if queue_compression is not None else
                ((om, station, int(minute))
                 for om, station, minute
                 in data["cache"].get("q_var_keys", ())))
        walking_plan = data.get("walking_plan")
        w_rows = pricing_index[sid].get("w_rows")
        if walking_plan is None and w_rows is None:
            w_rows = frozenset(
                (om, station, int(minute))
                for om, station, minute in data["cache"].get("w_var_keys", ()))
        for key, variable in data["b"].items():
            value = float(variable.X)
            b_values[(sid,) + key] = value
            om, _vehicle, minute, station = key
            if (om, station, minute) in q_rows:
                q_coefficients[(sid, om, station, minute)] += value
        for key, variable in data["a"].items():
            value = float(variable.X)
            a_values[(sid,) + key] = value
            om, _vehicle, minute, station = key
            if (om, station, minute + config.sigma) in q_rows:
                q_coefficients[(sid, om, station, minute + config.sigma)] -= value
            if walking_plan is not None:
                direct_cost += (
                    probability / horizon * value *
                    walking_plan.multipliers.get(
                        (om, station, int(minute)), 0))
            else:
                if (om, station, minute) in w_rows:
                    w_coefficients[(sid, om, station, minute)] -= value
                if (om, station, minute + config.sigma) in w_rows:
                    w_coefficients[(sid, om, station,
                                    minute + config.sigma)] += value
        for key, variable in data["l"].items():
            value = float(variable.X)
            l_values[(sid,) + key] = value
            direct_cost += probability * value / horizon
        trip = data["trip"]
        raw_begin, raw_end = _depot_interval(trip, config)
        begin = max(raw_begin, first)
        end = min(raw_end, after_last)
        for minute in data.get("depot_minutes", range(begin, end)):
            depot_coefficients[(sid, str(trip["Depot"]), minute)] += y_value

    result = PricingResult(
        key=column.key, status=status, reduced_cost=float(model.ObjVal),
        y=y_value, direct_cost=float(direct_cost),
        q_coefficients=dict(q_coefficients),
        w_coefficients=dict(w_coefficients),
        depot_coefficients=dict(depot_coefficients),
        b_values=b_values, a_values=a_values, l_values=l_values,
        solver_lower_bound=float(model.ObjBound))
    dot_value = result.dot_product_reduced_cost(duals)
    if abs(dot_value - result.reduced_cost) > 1e-6 * max(1.0, abs(dot_value)):
        retained.dispose()
        raise AssertionError(
            f"pricing coefficient audit failed for {column.key}: "
            f"model={result.reduced_cost}, dot={dot_value}")
    retained.dispose()
    return result


def _pricing_worker_count(value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError("pricing_workers must be a positive integer")
    return int(value)


def _pricing_model_cache_size(value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError("pricing_model_cache_size must be a nonnegative integer")
    return int(value)


def _pricing_model_cache_policy(value: str) -> str:
    policy = str(value).strip().lower()
    if policy not in {"stable", "lru"}:
        raise ValueError(
            "pricing_model_cache_policy must be 'stable' or 'lru'")
    return policy


def _pricing_parameter_signature(config: CGConfig) -> tuple:
    return (
        int(config.pricing_output_flag),
        int(config.ignore_names),
        float(config.optimality_tolerance),
        float(config.feasibility_tolerance),
    )


def _price_candidate(
        problem: CGProblem, config: CGConfig, column: ColumnSpec,
        duals: Mapping[str, Mapping], pricing_index: Mapping, *,
        keep_profile: bool, use_box_screening: bool,
        rc_epsilon: float,
        parameters_preconfigured: bool = False,
        model_cache=None) -> PricingResult:
    """Price one candidate, including its optional certified box screen."""
    if (not keep_profile and
            _has_zero_boarding_support(column, pricing_index)):
        return _price_zero_boarding_support(
            problem, config, column, duals, pricing_index)
    box_bound = None
    if use_box_screening:
        box_bound = pricing_box_lower_bound(
            problem, config, column, duals, pricing_index)
        if box_bound >= -rc_epsilon:
            return PricingResult(
                key=column.key, status="SCREENED_BOX_LB",
                reduced_cost=math.nan, y=math.nan, direct_cost=math.nan,
                q_coefficients={}, w_coefficients={}, depot_coefficients={},
                b_values={}, a_values={}, l_values={}, exact=False,
                box_lower_bound=box_bound)
    if model_cache is not None and not keep_profile:
        result = model_cache.solve(
            problem, config, column, duals, pricing_index,
            parameters_preconfigured=parameters_preconfigured)
    else:
        result = price_column(
            problem, config, column, duals, pricing_index=pricing_index,
            keep_profile=keep_profile,
            _parameters_preconfigured=parameters_preconfigured)
    result.box_lower_bound = box_bound
    return result


class _PricingModelCache:
    """Worker-owned bounded cache of exact local pricing LPs."""

    def __init__(self, capacity: int, policy: str):
        self.capacity = int(capacity)
        self.policy = str(policy)
        self.entries: OrderedDict[ColumnKey, _RetainedPricingModel] = (
            OrderedDict())
        self.hits = 0
        self.misses = 0
        self.admissions = 0
        self.rejections = 0
        self.evictions = 0
        self.selected_discards = 0
        self.close_discards = 0
        self.solve_failures = 0

    def solve(
            self, problem: CGProblem, config: CGConfig,
            column: ColumnSpec, duals: Mapping[str, Mapping],
            pricing_index: Mapping, *,
            parameters_preconfigured: bool) -> PricingResult:
        retained = self.entries.get(column.key)
        if retained is not None:
            if retained.column != column:
                raise ValueError(
                    f"cached pricing metadata changed for {column.key}")
            if retained.pricing_index is not pricing_index:
                raise ValueError(
                    "cached pricing model received a different pricing index")
            self.hits += 1
            if self.policy == "lru":
                self.entries.move_to_end(column.key)
            try:
                return retained.solve(duals)
            except BaseException:
                self.solve_failures += 1
                self.entries.pop(column.key, None)
                retained.dispose()
                raise

        self.misses += 1
        result, retained = price_column(
            problem, config, column, duals, pricing_index=pricing_index,
            keep_profile=False,
            _parameters_preconfigured=parameters_preconfigured,
            _retain_model=True)
        if self.capacity <= 0:
            retained.dispose()
            return result
        if len(self.entries) >= self.capacity:
            if self.policy == "stable":
                # Preserve the first deterministic hot set.  A sequential
                # full-pool scan would make a plain LRU cache thrash; selected
                # columns explicitly free slots for later admissions.
                self.rejections += 1
                retained.dispose()
                return result
            _evicted_key, evicted = self.entries.popitem(last=False)
            evicted.dispose()
            self.evictions += 1
        self.entries[column.key] = retained
        self.admissions += 1
        return result

    def discard_selected(self, keys: Iterable[ColumnKey]) -> int:
        discarded = 0
        for key in keys:
            retained = self.entries.pop(_normalise_key(key), None)
            if retained is None:
                continue
            retained.dispose()
            discarded += 1
        self.selected_discards += discarded
        return discarded

    def dispose_all(self) -> None:
        retained_models = tuple(self.entries.values())
        self.entries.clear()
        for retained in retained_models:
            retained.dispose()
        self.close_discards += len(retained_models)

    def stats(self) -> dict:
        return {
            "capacity": self.capacity,
            "policy": self.policy,
            "size": len(self.entries),
            "hits": self.hits,
            "misses": self.misses,
            "admissions": self.admissions,
            "rejections": self.rejections,
            "evictions": self.evictions,
            "selected_discards": self.selected_discards,
            "close_discards": self.close_discards,
            "solve_failures": self.solve_failures,
        }


class ParallelPricingPool:
    """Reusable thread pool for independent exact candidate LPs.

    Gurobi environments are not shared across concurrent solves.  They are
    created and started sequentially by the constructing thread because Env
    lifecycle operations are not thread-safe.  Each executor worker is then
    pinned to one pre-started environment, fixed to one solver thread, and
    reuses it for every candidate subsequently scheduled on that Python
    thread.  The input problem, duals, and pricing index remain shared,
    read-only Python objects; no large cache is copied.
    """

    def __init__(
            self, problem: CGProblem, workers: int,
            config: CGConfig | None = None, *,
            model_cache_size: int = 0,
            model_cache_policy: str = "stable"):
        self.problem = problem
        self.config = _cfg(config)
        self._parameter_signature = _pricing_parameter_signature(self.config)
        self.workers = _pricing_worker_count(workers)
        self.model_cache_size = _pricing_model_cache_size(model_cache_size)
        self.model_cache_policy = _pricing_model_cache_policy(
            model_cache_policy)
        self._local = threading.local()
        self._environments: list[gp.Env] = []
        self._executors: list[ThreadPoolExecutor] = []
        self._pricing_index = None
        self._owner_cache: dict[ColumnKey, int] = {}
        self._final_cache_stats = None
        self._closed = False
        base_capacity, extra_capacity = divmod(
            self.model_cache_size, self.workers)
        self._worker_cache_capacities = tuple(
            base_capacity + (1 if owner < extra_capacity else 0)
            for owner in range(self.workers))
        try:
            for _index in range(self.workers):
                environment = gp.Env(empty=True)
                try:
                    environment.setParam(
                        "OutputFlag", int(self.config.pricing_output_flag))
                    environment.setParam("Threads", 1)
                    environment.setParam(
                        "IgnoreNames", int(self.config.ignore_names))
                    environment.setParam(
                        "OptimalityTol",
                        float(self.config.optimality_tolerance))
                    environment.setParam(
                        "FeasibilityTol",
                        float(self.config.feasibility_tolerance))
                    environment.start()
                except BaseException:
                    environment.dispose()
                    raise
                self._environments.append(environment)
        except BaseException:
            for environment in reversed(self._environments):
                environment.dispose()
            self._environments.clear()
            raise
        try:
            for owner, environment in enumerate(self._environments):
                self._executors.append(ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix=f"cg-pricing-{owner}",
                    initializer=self._initialize_worker,
                    initargs=(
                        owner, environment,
                        self._worker_cache_capacities[owner]),
                ))
        except BaseException:
            for executor in self._executors:
                executor.shutdown(wait=True, cancel_futures=False)
            self._executors.clear()
            for environment in reversed(self._environments):
                environment.dispose()
            self._environments.clear()
            raise

    def _initialize_worker(
            self, owner: int, environment: gp.Env,
            cache_capacity: int) -> None:
        # Each owner has a dedicated single-thread executor.  Only assignment
        # occurs here; Env create/start remains serialized on the pool's
        # constructing thread.
        self._local.owner = int(owner)
        self._local.problem = replace(self.problem, env=environment)
        self._local.model_cache = _PricingModelCache(
            cache_capacity, self.model_cache_policy)

    def _owner_for_key(self, key: ColumnKey) -> int:
        key = _normalise_key(key)
        owner = self._owner_cache.get(key)
        if owner is None:
            digest = hashlib.blake2b(
                repr(key).encode("utf-8"), digest_size=8).digest()
            owner = int.from_bytes(digest, "little") % self.workers
            self._owner_cache[key] = owner
        return owner

    def _run_shard(self, task) -> list[tuple[int, ColumnKey, PricingResult]]:
        (owner, indexed_columns, config, duals, pricing_index, keep_profiles,
         use_box_screening, rc_epsilon) = task
        if int(self._local.owner) != int(owner):
            raise AssertionError("pricing shard ran on the wrong cache owner")
        results = []
        # A shard is one executor task, so it cannot migrate between Python
        # workers.  Every local LP in this loop therefore uses the same pinned
        # worker environment and is solved strictly serially within the shard.
        for index, column in indexed_columns:
            result = _price_candidate(
                self._local.problem, config, column, duals, pricing_index,
                keep_profile=keep_profiles,
                use_box_screening=use_box_screening,
                rc_epsilon=rc_epsilon,
                parameters_preconfigured=True,
                model_cache=(
                    self._local.model_cache
                    if self.model_cache_size > 0 and not keep_profiles else
                    None))
            results.append((index, column.key, result))
        return results

    def price(
            self, config: CGConfig, columns: Iterable[ColumnSpec],
            duals: Mapping[str, Mapping], *, pricing_index: Mapping,
            keep_profiles: bool, use_box_screening: bool,
            rc_epsilon: float) -> dict[ColumnKey, PricingResult]:
        if self._closed:
            raise RuntimeError("ParallelPricingPool is closed")
        pricing_index = _ensure_pricing_depot_index(
            pricing_index, duals, problem=self.problem, config=config)
        if _pricing_parameter_signature(config) != self._parameter_signature:
            raise ValueError(
                "pricing config does not match the parameters preconfigured "
                "on this ParallelPricingPool")
        if self.model_cache_size > 0:
            if config != self.config:
                raise ValueError(
                    "pricing config does not match retained model structure")
            if self._pricing_index is None:
                self._pricing_index = pricing_index
            elif self._pricing_index is not pricing_index:
                raise ValueError(
                    "a retained pricing-model cache requires one stable "
                    "pricing_index object")
        ordered_columns = tuple(columns)
        if not ordered_columns:
            return {}
        shards = [[] for _index in range(self.workers)]
        for index, column in enumerate(ordered_columns):
            shards[self._owner_for_key(column.key)].append((index, column))
        tasks = tuple(
            (owner, tuple(shard), config, duals, pricing_index, keep_profiles,
             use_box_screening, rc_epsilon)
            for owner, shard in enumerate(shards)
        )
        # Submit exactly one Future per configured worker, instead of one per
        # candidate.  This removes tens of thousands of Future allocations and
        # queue operations in a full-pool pricing pass.
        futures = tuple(
            self._executors[owner].submit(self._run_shard, task)
            for owner, task in enumerate(tasks))
        indexed_results: list[tuple[ColumnKey, PricingResult] | None] = [
            None] * len(ordered_columns)
        first_error = None
        for future in futures:
            try:
                shard_results = future.result()
            except BaseException as error:
                if first_error is None:
                    first_error = error
                continue
            for index, key, result in shard_results:
                indexed_results[index] = (key, result)
        # Wait for all shards above even if one failed, so no work from this
        # call remains in flight when control returns to the caller.
        if first_error is not None:
            raise first_error
        if any(item is None for item in indexed_results):
            raise AssertionError("parallel pricing shard omitted a candidate")
        # Reassemble by original index because owner shards finish in an
        # arbitrary order.  Dict order therefore matches single-worker mode.
        return dict(indexed_results)

    def _discard_owner(self, task) -> int:
        owner, keys = task
        if int(self._local.owner) != int(owner):
            raise AssertionError("pricing discard ran on the wrong cache owner")
        return self._local.model_cache.discard_selected(keys)

    def discard_selected(self, keys: Iterable[ColumnKey]) -> int:
        """Dispose cached models for columns inserted into the active RMP."""
        if self._closed:
            raise RuntimeError("ParallelPricingPool is closed")
        if self.model_cache_size <= 0:
            return 0
        by_owner = [[] for _index in range(self.workers)]
        for key in keys:
            normalised = _normalise_key(key)
            by_owner[self._owner_for_key(normalised)].append(normalised)
        futures = tuple(
            self._executors[owner].submit(
                self._discard_owner, (owner, tuple(owner_keys)))
            for owner, owner_keys in enumerate(by_owner))
        return sum(int(future.result()) for future in futures)

    def _owner_cache_stats(self, owner: int) -> dict:
        if int(self._local.owner) != int(owner):
            raise AssertionError("pricing stats ran on the wrong cache owner")
        stats = self._local.model_cache.stats()
        stats["owner"] = int(owner)
        return stats

    @staticmethod
    def _aggregate_cache_stats(per_worker: Sequence[Mapping]) -> dict:
        counter_names = (
            "capacity", "size", "hits", "misses", "admissions",
            "rejections", "evictions", "selected_discards",
            "close_discards", "solve_failures", "size_before_close")
        aggregate = {
            name: sum(int(stats.get(name, 0)) for stats in per_worker)
            for name in counter_names
        }
        aggregate["workers"] = len(per_worker)
        aggregate["policy"] = (
            str(per_worker[0]["policy"]) if per_worker else None)
        aggregate["hit_rate"] = (
            aggregate["hits"] /
            max(aggregate["hits"] + aggregate["misses"], 1))
        aggregate["per_worker"] = tuple(dict(stats) for stats in per_worker)
        return aggregate

    def cache_stats(self) -> dict:
        if self._closed:
            return dict(self._final_cache_stats or {})
        futures = tuple(
            self._executors[owner].submit(
                self._owner_cache_stats, owner)
            for owner in range(self.workers))
        return self._aggregate_cache_stats(
            tuple(future.result() for future in futures))

    def _dispose_owner_cache(self, owner: int) -> dict:
        if int(self._local.owner) != int(owner):
            raise AssertionError("pricing cleanup ran on the wrong cache owner")
        before = self._local.model_cache.stats()
        self._local.model_cache.dispose_all()
        after = self._local.model_cache.stats()
        after["size_before_close"] = int(before["size"])
        after["owner"] = int(owner)
        return after

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Cached models must be disposed by their owning worker before that
        # worker exits.  Only then is it safe for the constructing thread to
        # release the corresponding Gurobi environment.
        cleanup_futures = tuple(
            self._executors[owner].submit(
                self._dispose_owner_cache, owner)
            for owner in range(self.workers))
        cleanup_stats = []
        first_error = None
        for future in cleanup_futures:
            try:
                cleanup_stats.append(future.result())
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if len(cleanup_stats) == self.workers:
            self._final_cache_stats = self._aggregate_cache_stats(
                cleanup_stats)
        for executor in self._executors:
            executor.shutdown(wait=True, cancel_futures=False)
        self._executors.clear()
        environments = tuple(self._environments)
        self._environments.clear()
        for environment in environments:
            environment.dispose()
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "ParallelPricingPool":
        if self._closed:
            raise RuntimeError("ParallelPricingPool is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def price_columns(
        problem: CGProblem, config: CGConfig | None,
        columns: Iterable[ColumnSpec], duals: Mapping[str, Mapping], *,
        pricing_index: Mapping | None = None, keep_profiles: bool = False,
        use_box_screening: bool = False,
        rc_epsilon: float | None = None, pricing_workers: int = 1,
        pricing_pool: ParallelPricingPool | None = None,
        pricing_model_cache_size: int = 0,
        pricing_model_cache_policy: str = "stable",
        ) -> dict[ColumnKey, PricingResult]:
    config = _cfg(config)
    epsilon = config.rc_epsilon if rc_epsilon is None else float(rc_epsilon)
    workers = _pricing_worker_count(pricing_workers)
    cache_size = _pricing_model_cache_size(pricing_model_cache_size)
    cache_policy = _pricing_model_cache_policy(pricing_model_cache_policy)
    if pricing_index is None:
        pricing_index = build_pricing_index(
            problem, eliminate_w=config.eliminate_w,
            compress_q=config.compress_queue_states, sigma=config.sigma)
    pricing_index = _ensure_pricing_depot_index(
        pricing_index, duals, problem=problem, config=config)
    ordered_columns = tuple(columns)
    if pricing_pool is not None:
        if pricing_pool.problem is not problem:
            raise ValueError(
                "pricing_pool was created for a different CGProblem")
        return pricing_pool.price(
            config, ordered_columns, duals, pricing_index=pricing_index,
            keep_profiles=keep_profiles,
            use_box_screening=use_box_screening, rc_epsilon=epsilon)
    if workers > 1 or cache_size > 0:
        with ParallelPricingPool(
                problem, workers, config, model_cache_size=cache_size,
                model_cache_policy=cache_policy) as temporary_pool:
            return temporary_pool.price(
                config, ordered_columns, duals, pricing_index=pricing_index,
                keep_profiles=keep_profiles,
                use_box_screening=use_box_screening, rc_epsilon=epsilon)
    return {
        column.key: _price_candidate(
            problem, config, column, duals, pricing_index,
            keep_profile=keep_profiles,
            use_box_screening=use_box_screening, rc_epsilon=epsilon)
        for column in ordered_columns
    }


def _persistent_shadow_dimensions_match(
        persistent_stats: Mapping | None, shadow_stats: Mapping | None, *,
        allow_exact_reduction: bool) -> bool:
    """Validate persistent dimensions against the OD-explicit shadow oracle."""
    if not persistent_stats or not shadow_stats:
        return False
    fields = ("NumVars", "NumConstrs", "NumNZs")
    if any(field not in persistent_stats or field not in shadow_stats
           for field in fields):
        return False
    if allow_exact_reduction:
        return all(
            int(persistent_stats[field]) <= int(shadow_stats[field])
            for field in fields)
    # Legacy persistent and rebuilt masters use the identical formulation.
    return all(
        int(persistent_stats[field]) == int(shadow_stats[field])
        for field in ("NumVars", "NumConstrs"))


def run_column_generation(
        problem: CGProblem, config: CGConfig | None = None, *,
        pool: Iterable[ColumnSpec] | None = None,
        initial_columns: Iterable[ColumnKey] | None = None,
        seed_per_block: int = 1, add_per_iteration: int = 50,
        partial_pricing_batch_size: int | None = None,
        partial_pricing_reoptimize_after: int | None = None,
        pricing_workers: int = 1,
        pricing_model_cache_size: int = 0,
        pricing_model_cache_policy: str = "stable",
        max_iterations: int = 100, rc_epsilon: float | None = None,
        search_optimality_tolerance: float | None = None,
        search_rc_epsilon: float | None = None,
        rotate_partial_pricing: bool = False,
        monotonic_tolerance: float = 1e-7,
        require_complete_pool: bool = True,
        use_box_screening: bool = False,
        persistent_rmp: bool = True,
        validate_persistent_final: bool = False,
        stall_seconds: float | None = None,
        stall_absolute_tolerance: float = 1e-7,
        stall_relative_tolerance: float = 1e-9,
        dual_sign_tolerance: float = 1e-7,
        retain_final_diagnostics: bool = True,
        progress_hook: Callable[[Mapping], None] | None = None,
        log: Callable[[str], None] | None = print) -> dict:
    """Run exact finite-pool block generation to LP convergence."""
    # Sec. 3.4.4: both stages are solved as single MILPs over the fully
    # enumerated pool (solve_master).  The RMP/pricing coefficients of this
    # path were not converted to the Eq. (13) t >= t0 window and the
    # pre-onset normal fixing, so the path is disabled rather than
    # optimizing a different objective.
    raise NotImplementedError(
        "column generation does not implement Eq. (13) [eq:obj_db]; "
        "solve the full finite-pool MILP with solve_master")
    config = _cfg(config)
    persistent_rmp_requested = bool(persistent_rmp)
    persistent_rmp_fallback_reason = None
    if persistent_rmp and _has_passenger_na_metadata(problem.scenarios):
        # A persistent Gurobi Column can be supplied only when a Var is first
        # created. Causal b/a Vars must subsequently acquire coefficients in
        # another duration's fixed q/w rows. The rebuilt RMP is mathematically
        # exact and already supports those shared Var objects, so causal runs
        # select it automatically instead of constructing a subtly different
        # master.
        persistent_rmp = False
        persistent_rmp_fallback_reason = (
            "CAUSAL_PASSENGER_VARIABLE_SHARING_REQUIRES_REBUILT_RMP")
    epsilon = config.rc_epsilon if rc_epsilon is None else float(rc_epsilon)
    if max_iterations <= 0 or add_per_iteration <= 0:
        raise ValueError("iteration limits must be positive")
    if (partial_pricing_batch_size is not None and
            int(partial_pricing_batch_size) <= 0):
        raise ValueError("partial_pricing_batch_size must be positive or None")
    pricing_batch_size = (
        None if partial_pricing_batch_size is None else
        int(partial_pricing_batch_size))
    pricing_reoptimize_after = (
        int(add_per_iteration)
        if partial_pricing_reoptimize_after is None else
        int(partial_pricing_reoptimize_after))
    if pricing_reoptimize_after <= 0:
        raise ValueError(
            "partial_pricing_reoptimize_after must be positive or None")
    if pricing_reoptimize_after > int(add_per_iteration):
        raise ValueError(
            "partial_pricing_reoptimize_after cannot exceed "
            "add_per_iteration")
    pricing_worker_count = _pricing_worker_count(pricing_workers)
    cache_size = _pricing_model_cache_size(pricing_model_cache_size)
    cache_policy = _pricing_model_cache_policy(pricing_model_cache_policy)
    if not isinstance(rotate_partial_pricing, bool):
        raise ValueError("rotate_partial_pricing must be a bool")
    if cache_size > 0 and cache_policy == "stable" and rotate_partial_pricing:
        raise ValueError(
            "stable pricing-model caching requires static partial-pricing "
            "ordering; disable rotate_partial_pricing or select policy='lru'")
    if (search_optimality_tolerance is not None and
            float(search_optimality_tolerance) <= 0.0):
        raise ValueError(
            "search_optimality_tolerance must be positive or None")
    search_enabled = bool(
        search_optimality_tolerance is not None and
        float(search_optimality_tolerance) >
        float(config.optimality_tolerance))
    search_opt_tolerance = (
        float(config.optimality_tolerance) if not search_enabled else
        float(search_optimality_tolerance))
    if search_rc_epsilon is not None and float(search_rc_epsilon) < 0.0:
        raise ValueError("search_rc_epsilon must be nonnegative or None")
    search_epsilon = (
        epsilon if not search_enabled else
        (max(epsilon, 10.0 * search_opt_tolerance)
         if search_rc_epsilon is None else float(search_rc_epsilon)))
    if stall_seconds is not None and float(stall_seconds) <= 0:
        raise ValueError("stall_seconds must be positive or None")
    if stall_absolute_tolerance < 0 or stall_relative_tolerance < 0:
        raise ValueError("stall tolerances must be nonnegative")
    canonical_pool = build_column_pool(problem, config)
    canonical_rank = {
        column.key: index for index, column in enumerate(canonical_pool)}
    pool = canonical_pool if pool is None else tuple(pool)
    canonical_keys = {column.key for column in canonical_pool}
    supplied_keys = {column.key for column in pool}
    pool_complete = supplied_keys == canonical_keys
    if require_complete_pool and not pool_complete:
        raise ValueError(
            "pricing pool is not canonical: "
            f"missing={len(canonical_keys - supplied_keys)}, "
            f"extra={len(supplied_keys - canonical_keys)}")
    by_key = {column.key: column for column in pool}
    if len(by_key) != len(pool):
        raise ValueError("column pool contains duplicate keys")
    active = (seed_columns(pool, seed_per_block) if initial_columns is None else
              {_normalise_key(key) for key in initial_columns})
    unknown = active - supplied_keys
    if unknown:
        raise ValueError(f"initial columns outside pool: {sorted(unknown)}")

    depot_rows = build_depot_row_keys(problem, config)
    pricing_index = build_pricing_index(
        problem, include_profile_rows=False,
        eliminate_w=config.eliminate_w,
        compress_q=config.compress_queue_states, sigma=config.sigma,
        depot_row_keys=depot_rows)
    history = []
    last_duals, last_prices = {}, {}
    rmp = None
    last_objective = math.nan
    last_correction = math.nan
    last_rmp_stats = None
    last_primal_hint = None
    initial_rmp_objective = math.nan
    best_rmp_objective = math.inf
    last_material_improvement_at = time.monotonic()
    certified = False
    termination = "NOT_STARTED"
    interrupted = False
    persistent_validation_passed = None
    engine = None
    pricing_executor = None
    pricing_cache_stats = {
        "capacity": cache_size, "policy": cache_policy, "size": 0,
        "hits": 0, "misses": 0, "admissions": 0, "rejections": 0,
        "evictions": 0, "selected_discards": 0, "close_discards": 0,
        "solve_failures": 0, "hit_rate": 0.0, "workers": 0,
        "per_worker": (),
    }
    tight_certification_phase = not search_enabled
    tight_certification_started = not search_enabled
    # Exact reduced costs from the preceding pass are ranking hints only.  A
    # stale value is never used as a certificate under a new RMP dual vector.
    pricing_priority: dict[ColumnKey, float] = {}
    pricing_last_visited: dict[ColumnKey, int] = {}
    last_pricing_complete = False
    last_omitted_count = 0

    def emit(phase: str, **extra) -> None:
        if progress_hook is None:
            return
        progress_hook({
            "phase": phase,
            "termination_reason": termination,
            "certified": bool(certified),
            "active_columns": len(active),
            "pool_columns": len(pool),
            "last_rmp_objective": last_objective,
            "history": tuple(history),
            "last_rmp_stats": last_rmp_stats,
            **extra,
        })

    def begin_tight_certification(reason: str, iteration: int) -> None:
        """Switch subsequent RMP solves to the certified tight tolerance."""
        nonlocal tight_certification_phase, tight_certification_started
        tight_certification_phase = True
        tight_certification_started = True
        if engine is not None:
            engine.set_lp_tolerances(
                optimality=config.optimality_tolerance,
                feasibility=config.feasibility_tolerance)
        emit(
            "TIGHT_CERTIFICATION_STARTED", iteration=iteration,
            certification_trigger=reason,
            tight_optimality_tolerance=float(config.optimality_tolerance),
            tight_rc_epsilon=float(epsilon))

    try:
        if pricing_worker_count > 1 or cache_size > 0:
            pricing_executor = ParallelPricingPool(
                problem, pricing_worker_count, config,
                model_cache_size=cache_size,
                model_cache_policy=cache_policy)
        if persistent_rmp:
            engine = IncrementalRMP(
                problem, config, pool, depot_rows,
                pricing_index=pricing_index)
            engine.add_columns(sorted(active, key=repr))
            if search_enabled:
                engine.set_lp_tolerances(
                    optimality=search_opt_tolerance,
                    feasibility=config.feasibility_tolerance)
            emit("BASE_RMP_BUILT", base_rmp_stats=engine.structural_stats())

        for iteration in range(1, max_iterations + 1):
            # Never expose a preceding iteration's complete-pricing state if
            # this RMP solve is interrupted or fails before fresh pricing.
            last_pricing_complete = False
            last_omitted_count = len(pool) - len(active)
            last_correction = math.nan
            last_prices = {}
            started = time.time()
            current_rmp_tolerance = (
                float(config.optimality_tolerance)
                if tight_certification_phase else search_opt_tolerance)
            current_epsilon = (
                epsilon if tight_certification_phase else search_epsilon)
            current_phase = (
                "CERTIFY" if tight_certification_phase else "SEARCH")
            rmp_config = (
                config if engine is not None or tight_certification_phase else
                replace(
                    config,
                    optimality_tolerance=current_rmp_tolerance,
                    feasibility_tolerance=config.feasibility_tolerance))
            rmp = (engine.solve() if engine is not None else
                   solve_rmp(problem, rmp_config, active, depot_rows))
            last_objective = rmp["objective"]
            last_rmp_stats = rmp.get("stats")
            last_primal_hint = rmp.get("primal_hint")
            if math.isnan(initial_rmp_objective):
                initial_rmp_objective = last_objective
            material_threshold = max(
                float(stall_absolute_tolerance),
                float(stall_relative_tolerance) * max(
                    1.0, abs(best_rmp_objective)
                    if math.isfinite(best_rmp_objective) else 1.0))
            if last_objective < best_rmp_objective - material_threshold:
                best_rmp_objective = last_objective
                last_material_improvement_at = time.monotonic()

            stats_fields = {
                "rmp_num_vars": (last_rmp_stats or {}).get("NumVars"),
                "rmp_num_constraints": (last_rmp_stats or {}).get("NumConstrs"),
                "rmp_num_binary": (last_rmp_stats or {}).get("NumBinVars"),
                "rmp_build_seconds": (last_rmp_stats or {}).get("BuildSec"),
                "rmp_solve_seconds": (last_rmp_stats or {}).get(
                    "LastReoptimizeSec", (last_rmp_stats or {}).get("Runtime")),
                "rmp_objective_bound": (last_rmp_stats or {}).get("ObjBound"),
                "solve_phase": current_phase,
                "rmp_optimality_tolerance": current_rmp_tolerance,
                "pricing_rc_epsilon": current_epsilon,
            }
            if rmp["status"] != "OPTIMAL":
                termination = f"RMP_{rmp['status']}"
                history.append({
                    "iteration": iteration, "objective": last_objective,
                    "active_columns": len(active),
                    "omitted_columns": last_omitted_count,
                    "pricing_complete": False,
                    "priced_columns": 0,
                    "unpriced_columns": last_omitted_count,
                    "negative_columns": math.nan,
                    "screened_columns": 0,
                    "exact_pricing_columns": 0,
                    "minimum_priced_rc": math.nan,
                    "minimum_omitted_rc": math.nan,
                    "full_lp_bound_correction": math.nan,
                    "pricing_seconds": 0.0,
                    "seconds": time.time() - started,
                    "rmp_status": rmp["status"],
                    **stats_fields,
                })
                emit("ITERATION_COMPLETE", iteration=iteration)
                break
            # Move, rather than alias, the large map out of the solve result.
            last_duals = rmp.pop("duals")
            maximum_depot_dual = max(
                last_duals.get("depot", {}).values(), default=-math.inf)
            if maximum_depot_dual > dual_sign_tolerance:
                if not tight_certification_phase and iteration < max_iterations:
                    begin_tight_certification("LOOSE_DEPOT_DUAL_SIGN", iteration)
                    last_duals = {}
                    rmp = None
                    continue
                termination = "INVALID_DEPOT_DUAL_SIGN"
                emit("INVALID_DUAL", iteration=iteration)
                break
            crossing_from_search = bool(
                history and current_phase == "CERTIFY" and
                history[-1].get("solve_phase") == "SEARCH" and
                int(history[-1].get("active_columns", -1)) == len(active))
            if (history and not crossing_from_search and
                    last_objective >
                    history[-1]["objective"] + monotonic_tolerance):
                if not tight_certification_phase and iteration < max_iterations:
                    begin_tight_certification("LOOSE_NONMONOTONE_RMP", iteration)
                    last_duals = {}
                    rmp = None
                    continue
                termination = "NONMONOTONE_RMP"
                emit("NONMONOTONE_RMP", iteration=iteration)
                break

            omitted = [column for column in pool if column.key not in active]
            last_omitted_count = len(omitted)
            pricing_started = time.time()
            if pricing_batch_size is None:
                last_prices = price_columns(
                    problem, config, omitted, last_duals,
                    pricing_index=pricing_index, keep_profiles=False,
                    use_box_screening=use_box_screening,
                    rc_epsilon=current_epsilon,
                    pricing_pool=pricing_executor)
                for key, priced in last_prices.items():
                    pricing_last_visited[key] = iteration
                    if (priced.exact and
                            priced.reduced_cost < -current_epsilon):
                        pricing_priority[key] = float(priced.reduced_cost)
                    else:
                        pricing_priority.pop(key, None)
                last_pricing_complete = True
            else:
                # Dynamic partial pricing is exact because every chunk in this
                # loop uses the *same* RMP dual vector.  We stop early only
                # after finding enough exact negative columns.  If that never
                # happens, every omitted column is priced before certification.
                if rotate_partial_pricing:
                    ordered = sorted(
                        omitted,
                        key=lambda column: (
                            0 if column.key in pricing_priority else 1,
                            pricing_priority.get(column.key, 0.0),
                            0 if column.key not in pricing_last_visited else 1,
                            pricing_last_visited.get(column.key, -1),
                            -int(column.score),
                            canonical_rank[column.key],
                        ))
                else:
                    ordered = sorted(
                        omitted,
                        key=lambda column: (
                            0 if column.key in pricing_priority else 1,
                            pricing_priority.get(column.key, 0.0),
                            -int(column.score),
                            canonical_rank[column.key],
                        ))
                partial_results = {}
                priced_until = 0
                negative_found = 0
                while priced_until < len(ordered):
                    chunk = ordered[
                        priced_until:priced_until + pricing_batch_size]
                    # The loose box bound is most often pure overhead on the
                    # high-priority first chunk, where negative columns are
                    # dense.  Defer it until a second chunk is actually needed.
                    # A screened result can only be nonnegative, so this changes
                    # neither the selected negative columns nor certification;
                    # a final sweep still screens every later chunk.
                    screen_this_chunk = bool(
                        use_box_screening and priced_until > 0)
                    chunk_results = price_columns(
                        problem, config, chunk, last_duals,
                        pricing_index=pricing_index, keep_profiles=False,
                        use_box_screening=screen_this_chunk,
                        rc_epsilon=current_epsilon,
                        pricing_pool=pricing_executor)
                    partial_results.update(chunk_results)
                    priced_until += len(chunk)
                    for key, priced in chunk_results.items():
                        pricing_last_visited[key] = iteration
                        if (priced.exact and
                                priced.reduced_cost < -current_epsilon):
                            pricing_priority[key] = float(priced.reduced_cost)
                            negative_found += 1
                        else:
                            pricing_priority.pop(key, None)
                    if negative_found >= pricing_reoptimize_after:
                        break
                last_prices = partial_results
                last_pricing_complete = priced_until == len(ordered)
            pricing_seconds = time.time() - pricing_started
            iteration_cache_stats = (
                pricing_executor.cache_stats()
                if pricing_executor is not None and cache_size > 0 else
                pricing_cache_stats)
            negative = sorted(
                (result.reduced_cost, key)
                for key, result in last_prices.items()
                if (result.exact and
                    result.reduced_cost < -current_epsilon))
            minimum_priced_certificate = min(
                (result.certificate_value for result in last_prices.values()),
                default=math.inf)
            screened = sum(not result.exact for result in last_prices.values())
            minimum_certificate = (
                minimum_priced_certificate if last_pricing_complete else
                math.nan)
            last_correction = (
                math.fsum(
                    min(0.0, result.certificate_value)
                    for result in last_prices.values())
                if last_pricing_complete else math.nan)
            row = {
                "iteration": iteration,
                "objective": last_objective,
                "objective_improvement_from_initial": (
                    initial_rmp_objective - last_objective),
                "active_columns": len(active),
                "omitted_columns": len(omitted),
                "pricing_complete": bool(last_pricing_complete),
                "priced_columns": len(last_prices),
                "unpriced_columns": len(omitted) - len(last_prices),
                "negative_columns": len(negative),
                "screened_columns": screened,
                "exact_pricing_columns": len(last_prices) - screened,
                "minimum_priced_rc": minimum_priced_certificate,
                "minimum_omitted_rc": minimum_certificate,
                "full_lp_bound_correction": last_correction,
                "pricing_seconds": pricing_seconds,
                "pricing_cache_size": int(
                    iteration_cache_stats.get("size", 0)),
                "pricing_cache_hits": int(
                    iteration_cache_stats.get("hits", 0)),
                "pricing_cache_misses": int(
                    iteration_cache_stats.get("misses", 0)),
                "pricing_cache_hit_rate": float(
                    iteration_cache_stats.get("hit_rate", 0.0)),
                "pricing_cache_admissions": int(
                    iteration_cache_stats.get("admissions", 0)),
                "pricing_cache_rejections": int(
                    iteration_cache_stats.get("rejections", 0)),
                "seconds": time.time() - started,
                "rmp_status": rmp["status"],
                "pricing_reoptimize_after": pricing_reoptimize_after,
                **stats_fields,
            }
            history.append(row)
            if log is not None:
                minimum_text = ("inf" if math.isinf(minimum_certificate)
                                else f"{minimum_certificate:.6g}")
                log(
                    f"  [exact CG {iteration} {current_phase}] "
                    f"obj={last_objective:.8f} "
                    f"active={len(active):,}/{len(pool):,} "
                    f"priced={len(last_prices):,}/{len(omitted):,} "
                    f"negative={len(negative):,} screened={screened:,} "
                    f"minCertRC={minimum_text} {row['seconds']:.2f}s")
            emit("ITERATION_COMPLETE", iteration=iteration)
            if not negative:
                if not last_pricing_complete:
                    raise AssertionError(
                        "partial pricing stopped without a negative column or "
                        "complete omitted-pool coverage")
                if not tight_certification_phase:
                    if iteration == max_iterations:
                        termination = "MAX_ITERATIONS"
                        break
                    begin_tight_certification(
                        "LOOSE_NO_NEGATIVE_FULL_SWEEP", iteration)
                    rmp = None
                    last_duals = {}
                    last_prices = {}
                    continue
                certified = pool_complete
                termination = (
                    "NO_NEGATIVE_OMITTED_COLUMN" if pool_complete else
                    "NO_NEGATIVE_INCOMPLETE_POOL")
                break
            if (stall_seconds is not None and
                    time.monotonic() - last_material_improvement_at >=
                    float(stall_seconds)):
                # A flat RMP objective is not an optimality certificate.  Keep
                # the last complete pricing pass for diagnostics, but never
                # publish this result as the full-LP lower bound.
                certified = False
                termination = "OBJECTIVE_STALL"
                emit("OBJECTIVE_STALL", iteration=iteration)
                break
            if iteration == max_iterations:
                termination = "MAX_ITERATIONS"
                break
            additions = [key for _price, key in negative[:add_per_iteration]]
            if pricing_executor is not None:
                # These candidates will never be priced again.  Dispose their
                # retained local LPs on the owning worker before expanding the
                # master, freeing stable-cache slots deterministically.
                pricing_executor.discard_selected(additions)
            if engine is not None:
                engine.add_columns(additions)
            active.update(additions)
            emit("COLUMNS_ADDED", iteration=iteration,
                 added_columns=len(additions))
            # Pricing and column insertion have consumed this iteration's
            # dictionaries.  Drop every alias before the next solve constructs
            # another multi-million-entry dual map.
            rmp = None
            last_duals = {}
            last_prices = {}
            omitted = None
            negative = None
    except KeyboardInterrupt:
        interrupted = True
        certified = False
        termination = "INTERRUPTED"
        emit("INTERRUPTED")
    finally:
        if pricing_executor is not None:
            live_cache_stats = None
            try:
                live_cache_stats = pricing_executor.cache_stats()
            finally:
                pricing_executor.close()
            # Preserve close-disposal counts while retaining the live size
            # observed immediately before shutdown.
            closed_stats = pricing_executor.cache_stats()
            closed_stats["size_before_close"] = (
                live_cache_stats.get("size", 0)
                if live_cache_stats is not None else
                closed_stats.get("size_before_close", 0))
            pricing_cache_stats = closed_stats
        if engine is not None:
            engine.close()

    # Optional rollout gate.  It is intentionally off for large production
    # runs because it performs one legacy rebuild; synthetic/pilot validation
    # can enable it to compare the persistent model against the old oracle.
    if (validate_persistent_final and persistent_rmp and certified and
            termination == "NO_NEGATIVE_OMITTED_COLUMN"):
        shadow = solve_rmp(problem, config, active, depot_rows)
        objective_matches = (
            shadow["status"] == "OPTIMAL" and
            abs(float(shadow["objective"]) - float(last_objective)) <=
            1e-6 * max(1.0, abs(float(last_objective))))
        dimension_matches = _persistent_shadow_dimensions_match(
            last_rmp_stats, shadow.get("stats"),
            allow_exact_reduction=config.hybrid_bus_load_states)
        persistent_validation_passed = objective_matches and dimension_matches
        if not persistent_validation_passed:
            certified = False
            termination = "PERSISTENT_RMP_VALIDATION_FAILED"

    absolute_improvement = (
        initial_rmp_objective - last_objective
        if math.isfinite(initial_rmp_objective) and math.isfinite(last_objective)
        else math.nan)
    relative_improvement = (
        absolute_improvement / abs(initial_rmp_objective)
        if math.isfinite(absolute_improvement) and initial_rmp_objective != 0
        else math.nan)
    final_minimum_certificate = (
        min(
            (priced.certificate_value for priced in last_prices.values()),
            default=math.inf)
        if last_pricing_complete else math.nan)
    final_box_screened = sum(
        not priced.exact for priced in last_prices.values())
    final_priced_count = len(last_prices)
    if retain_final_diagnostics:
        final_duals = last_duals
        final_omitted_prices = last_prices
    else:
        # The production runner consumes only the scalar summaries below.  Do
        # not carry the final dual/result dictionaries into the integer solve.
        rmp = None
        last_duals = {}
        last_prices = {}
        final_duals = {}
        final_omitted_prices = {}
    return {
        "pool": pool,
        "pool_complete": pool_complete,
        "pool_hash": _pool_hash(supplied_keys),
        "canonical_pool_hash": _pool_hash(canonical_keys),
        "box_screening": bool(use_box_screening),
        "partial_pricing_batch_size": pricing_batch_size,
        "partial_pricing_reoptimize_after": pricing_reoptimize_after,
        "rotate_partial_pricing": bool(rotate_partial_pricing),
        "pricing_workers": pricing_worker_count,
        "pricing_model_cache_size": cache_size,
        "pricing_model_cache_policy": cache_policy,
        "pricing_model_cache_stats": pricing_cache_stats,
        "search_optimality_tolerance": (
            search_opt_tolerance if search_enabled else None),
        "search_rc_epsilon": search_epsilon if search_enabled else None,
        "tight_optimality_tolerance": float(config.optimality_tolerance),
        "tight_rc_epsilon": float(epsilon),
        "tight_certification_started": bool(tight_certification_started),
        "persistent_rmp_requested": persistent_rmp_requested,
        "persistent_rmp": bool(persistent_rmp),
        "persistent_rmp_fallback_reason": persistent_rmp_fallback_reason,
        "persistent_validation_passed": persistent_validation_passed,
        "active_columns": frozenset(active),
        "history": tuple(history),
        "certified": certified,
        "lb": (last_objective + last_correction if certified else None),
        "rmp_lp_value": last_objective if certified else None,
        "pricing_bound_correction": last_correction if certified else None,
        "maximum_unresolved_lp_improvement": (
            max(0.0, -last_correction) if certified else None),
        "last_rmp_objective": last_objective,
        "initial_rmp_objective": initial_rmp_objective,
        "master_objective_improvement_absolute": absolute_improvement,
        "master_objective_improvement_relative": relative_improvement,
        "master_objective_improved": bool(
            math.isfinite(absolute_improvement) and
            absolute_improvement > monotonic_tolerance),
        "last_rmp_stats": last_rmp_stats,
        "final_rmp_primal_hint": last_primal_hint,
        "termination_reason": termination,
        "interrupted": interrupted,
        "final_minimum_omitted_certificate_value": final_minimum_certificate,
        "final_pricing_complete": bool(last_pricing_complete),
        "final_box_screened_columns": final_box_screened,
        "final_priced_columns": final_priced_count,
        "final_omitted_columns": last_omitted_count,
        "final_duals": final_duals,
        "final_omitted_prices": final_omitted_prices,
        "depot_row_keys": depot_rows,
    }


def solve_full_lp(problem: CGProblem, config: CGConfig | None = None,
                  pool: Iterable[ColumnSpec] | None = None) -> dict:
    """Solve the direct monolithic LP over every canonical macro-column."""
    config = _cfg(config)
    canonical = build_column_pool(problem, config)
    pool = canonical if pool is None else tuple(pool)
    if ({column.key for column in pool} !=
            {column.key for column in canonical}):
        raise ValueError("solve_full_lp requires the canonical complete pool")
    return solve_rmp(
        problem, config, {column.key for column in pool},
        build_depot_row_keys(problem, config))


def solve_restricted_milp(
        problem: CGProblem, config: CGConfig | None,
        active_columns: Iterable[ColumnKey],
        depot_row_keys: Iterable[tuple[str, str, int]] | None = None, *,
        primal_hint: Mapping | None = None,
        depot_start: Mapping[str, float] | None = None,
        column_start: Iterable[ColumnKey] | None = None,
        stall_seconds: float | None = None,
        progress_hook: Callable[[Mapping], None] | None = None):
    """Solve the integer master on generated columns; its incumbent is a UB."""
    config = _cfg(config)
    return solve_master(
        problem, config, active_columns=active_columns,
        primal_hint=primal_hint,
        depot_start=depot_start, column_start=column_start,
        extract_solution=True,
        stall_seconds=stall_seconds, progress_hook=progress_hook)


__all__ = [
    "BOX_BOUND_REL_SAFETY", "CGConfig", "CGProblem", "ColumnKey",
    "ColumnSpec", "PricingResult", "ParallelPricingPool", "RC_EPS", "SIGMA",
    "SUPPORT_EPS", "DEFAULT_DISRUPT_BUFFER", "IncrementalRMP",
    "aggregate_bus_load_groups",
    "compact_full_pool_cache", "prepare_exact_state_support",
    "build_column_pool",
    "build_depot_row_keys", "build_pricing_index", "count_macro_columns",
    "info_cells", "price_column", "price_columns",
    "pricing_box_lower_bound", "run_column_generation", "seed_columns",
    "solve_full_lp", "solve_master", "solve_restricted_milp", "solve_rmp",
]
