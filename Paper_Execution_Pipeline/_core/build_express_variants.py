"""Build reviewer-defensible nested express variants from the manual LOC routes.

The original express-route file is deliberately *not* used.  For every parent
local route, this script keeps only its upstream anchor (the first stop of the
local route in inputs.local_routes) and constructs one nested chain

    S_3 subset S_4 subset ... subset S_7,

where ``S_k`` has exactly ``k`` stops.  The end station is demand driven; it is
not forced to be the end of the parent LOC.  Every generated route is a complete
spatial service to that declared end station, not a prefix cut by the
optimization horizon.

Candidate feasibility is evaluated with the same deterministic road-time
matrix read by the MILP:

* the parent LOC order is preserved and the parent anchor is retained;
* one-way raw road time is at most ``--max-route-min`` (routes.express_max_travel_min);
* against the parent LOC over the *same anchor-to-end span*, the EXP saves at
  least ``--min-saving-min`` raw minutes (routes.express_min_saving_min) and at
  least one integer model minute;
* the selected stops cover positive two-way OD demand.

Among all feasible nested chains, the algorithm first maximizes the deepest K
that can be reached, then lexicographically maximizes covered OD demand at
Kmin, ..., Kmax.  Thus a Kmax route is never assembled independently of its
smaller predecessors.  Exact topological duplicates (including reversals) are
pruned globally after generation, as required by the manuscript.

Default outputs (names built from the profile, see 03_bus_routes.py)
---------------
``BusRoutes_LOC{n}_EXP_K{kmin}-{kmax}_T{tmax}.csv``
    The unchanged manual LOC routes plus the selected unique EXP routes.
``EXP_K{kmin}-{kmax}_T{tmax}_audit.csv``
    One row per parent LOC and requested K, including infeasibility and
    duplicate reasons.

The production CG input is not overwritten by this script.  Switching the CG
to the new route universe is a separate, explicit experiment requiring a full
rerun.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bbconfig import cfg  # noqa: E402

ROUTE_DIR = Path(
    os.environ.get("PAPER_ROUTE_DIR") or cfg.results_dir / cfg.get("paths.routes")
).resolve()
KMIN = int(cfg.get("routes.express_min_stops"))
KMAX = int(cfg.get("routes.express_chain_max_stops"))
TMAX = int(cfg.get("routes.express_max_travel_min"))
DEFAULT_LOCAL = ROUTE_DIR / cfg.get("routes.local_detail_file")
DEFAULT_OD = cfg.path("inputs.od_matrix")
DEFAULT_DISTANCE = cfg.path("inputs.road_times")
DEFAULT_OUT = ROUTE_DIR / (
    f"BusRoutes_LOC{cfg.get('routes.local_routes_expected')}_EXP_K{KMIN}-{KMAX}_T{TMAX}.csv")
DEFAULT_AUDIT = ROUTE_DIR / f"EXP_K{KMIN}-{KMAX}_T{TMAX}_audit.csv"

ROUTE_COLUMNS = [
    "Route", "BaseLine", "Type", "StopSeq", "StationCode",
    "CumTime_min", "SegTime_min", "SkippedStations",
    "OrigLine_NumStops", "OrigLine_TotalTime", "Route_TotalTime",
    "TimeSaving_pct", "DisruptionDemand",
]

EPS = 1e-9


@dataclass(frozen=True)
class ParentRoute:
    route: str
    baseline: str
    stops: tuple[str, ...]
    raw_cum: tuple[float, ...]
    model_cum: tuple[int, ...]


@dataclass(frozen=True)
class Candidate:
    parent_route: str
    baseline: str
    k: int
    indices: frozenset[int]
    stops: tuple[str, ...]
    raw_cum: tuple[float, ...]
    model_cum: tuple[int, ...]
    local_raw_time: float
    local_model_time: int
    covered_demand: float

    @property
    def raw_time(self) -> float:
        return self.raw_cum[-1]

    @property
    def model_time(self) -> int:
        return self.model_cum[-1]

    @property
    def saving_minutes(self) -> float:
        return self.local_raw_time - self.raw_time

    @property
    def saving_percent(self) -> float:
        if self.local_raw_time <= 0:
            return 0.0
        return 100.0 * self.saving_minutes / self.local_raw_time


@dataclass(frozen=True)
class ChainState:
    keys: tuple[frozenset[int], ...]
    coverage_profile: tuple[float, ...]
    saving_profile: tuple[float, ...]
    time_profile: tuple[float, ...]


def _read_road_times(path: Path) -> dict[tuple[str, str], float]:
    frame = pd.read_csv(path, usecols=[
        "Origin_Code", "Destination_Code", "Duration_min"])
    times: dict[tuple[str, str], float] = {}
    for row in frame.dropna().itertuples(index=False):
        origin = str(row.Origin_Code).strip()
        destination = str(row.Destination_Code).strip()
        duration = float(row.Duration_min)
        if not origin or not destination or duration < 0:
            continue
        key = (origin, destination)
        if key not in times or duration < times[key]:
            times[key] = duration
    if not times:
        raise ValueError(f"No usable road times in {path}")
    return times


def _road_time(times: dict[tuple[str, str], float], origin: str,
               destination: str) -> float | None:
    if origin == destination:
        return 0.0
    value = times.get((origin, destination))
    return value if value is not None else times.get((destination, origin))


def _route_cum(stops: Iterable[str],
               times: dict[tuple[str, str], float]) -> tuple[float, ...] | None:
    stops = tuple(stops)
    cumulative = [0.0]
    for origin, destination in zip(stops, stops[1:]):
        duration = _road_time(times, origin, destination)
        if duration is None:
            return None
        cumulative.append(cumulative[-1] + duration)
    return tuple(cumulative)


def _model_cum(raw_cum: Iterable[float]) -> tuple[int, ...] | None:
    cumulative = tuple(int(round(value)) for value in raw_cum)
    if not cumulative or cumulative[0] != 0:
        return None
    if any(right <= left for left, right in zip(cumulative, cumulative[1:])):
        return None
    return cumulative


def _load_parents(path: Path,
                  times: dict[tuple[str, str], float]) -> tuple[pd.DataFrame,
                                                               list[ParentRoute]]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = set(ROUTE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing route columns: {sorted(missing)}")
    local = frame[frame["Type"].astype(str).str.upper().eq("LOC")].copy()
    if local.empty:
        raise ValueError(f"No LOC rows in {path}")

    parents: list[ParentRoute] = []
    seen_baselines: set[str] = set()
    for route_name, group in local.groupby("Route", sort=False):
        ordered = group.sort_values("StopSeq", kind="stable")
        baseline = str(ordered.iloc[0]["BaseLine"]).strip()
        if baseline in seen_baselines:
            raise ValueError(
                f"Multiple parent LOC routes use BaseLine={baseline!r}; "
                "route naming would be ambiguous")
        seen_baselines.add(baseline)
        stops = tuple(ordered["StationCode"].astype(str).str.strip())
        if len(stops) < 2 or len(set(stops)) != len(stops):
            raise ValueError(f"Invalid stop sequence for {route_name}: {stops}")
        raw_cum = _route_cum(stops, times)
        if raw_cum is None:
            raise ValueError(f"Missing road-time link in parent {route_name}")
        model_cum = tuple(
            int(round(float(value))) for value in ordered["CumTime_min"])
        model_cum = tuple(value - model_cum[0] for value in model_cum)
        if any(right <= left for left, right in zip(model_cum, model_cum[1:])):
            raise ValueError(f"Non-increasing model time in parent {route_name}")
        calculated = _model_cum(raw_cum)
        if calculated is None:
            raise ValueError(f"Road-time rounding collapses a leg in {route_name}")
        if calculated != model_cum:
            raise ValueError(
                f"{route_name} does not match Station_Distance_Matrix after "
                f"rounding: file={model_cum}, matrix={calculated}")
        parents.append(ParentRoute(
            route=str(route_name), baseline=baseline, stops=stops,
            raw_cum=raw_cum, model_cum=model_cum))
    return local[ROUTE_COLUMNS].copy(), parents


def _load_two_way_od(path: Path) -> dict[tuple[str, str], float]:
    frame = pd.read_csv(path)
    required = {"OriginID", "DestinationID", "Trips_Hourly"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing OD columns: {sorted(missing)}")
    frame = frame[list(required)].copy()
    frame["OriginID"] = frame["OriginID"].astype(str).str.strip()
    frame["DestinationID"] = frame["DestinationID"].astype(str).str.strip()
    frame["Trips_Hourly"] = pd.to_numeric(
        frame["Trips_Hourly"], errors="coerce").fillna(0.0)
    directed = frame.groupby(
        ["OriginID", "DestinationID"], sort=False)["Trips_Hourly"].sum()

    two_way: dict[tuple[str, str], float] = {}
    for (origin, destination), demand in directed.items():
        if origin == destination or demand <= 0:
            continue
        key = tuple(sorted((str(origin), str(destination))))
        two_way[key] = two_way.get(key, 0.0) + float(demand)
    if not two_way:
        raise ValueError(f"No positive non-diagonal demand in {path}")
    return two_way


def _covered_demand(stops: Iterable[str],
                    two_way_od: dict[tuple[str, str], float]) -> float:
    stops = tuple(stops)
    return sum(two_way_od.get(tuple(sorted((left, right))), 0.0)
               for left, right in combinations(stops, 2))


def _candidate_sets(parent: ParentRoute, k: int, endpoint_policy: str):
    n_stops = len(parent.stops)
    if k >= n_stops:
        return
    if endpoint_policy == "parent":
        for internal in combinations(range(1, n_stops - 1), k - 2):
            yield (0, *internal, n_stops - 1)
    else:
        for tail in combinations(range(1, n_stops), k - 1):
            yield (0, *tail)


def _enumerate_feasible(
        parent: ParentRoute, k: int,
        times: dict[tuple[str, str], float],
        two_way_od: dict[tuple[str, str], float], *,
        endpoint_policy: str, max_route_min: float,
        min_saving_min: float) -> tuple[dict[frozenset[int], Candidate], dict]:
    stats = {
        "NumEnumerated": 0,
        "NumRoadFeasible": 0,
        "NumWithinTime": 0,
        "NumFaster": 0,
        "NumPositiveDemand": 0,
    }
    feasible: dict[frozenset[int], Candidate] = {}
    for indices in _candidate_sets(parent, k, endpoint_policy):
        stats["NumEnumerated"] += 1
        stops = tuple(parent.stops[index] for index in indices)
        raw_cum = _route_cum(stops, times)
        if raw_cum is None:
            continue
        model_cum = _model_cum(raw_cum)
        if model_cum is None:
            continue
        stats["NumRoadFeasible"] += 1
        if raw_cum[-1] > max_route_min + EPS:
            continue
        stats["NumWithinTime"] += 1

        end_index = indices[-1]
        local_raw_time = parent.raw_cum[end_index]
        local_model_time = parent.model_cum[end_index]
        raw_saving = local_raw_time - raw_cum[-1]
        model_saving = local_model_time - model_cum[-1]
        if raw_saving + EPS < min_saving_min or model_saving < 1:
            continue
        stats["NumFaster"] += 1

        demand = _covered_demand(stops, two_way_od)
        if demand <= EPS:
            continue
        stats["NumPositiveDemand"] += 1
        key = frozenset(indices)
        feasible[key] = Candidate(
            parent_route=parent.route, baseline=parent.baseline, k=k,
            indices=key, stops=stops, raw_cum=raw_cum,
            model_cum=model_cum, local_raw_time=local_raw_time,
            local_model_time=local_model_time, covered_demand=demand)
    return feasible, stats


def _state_quality(state: ChainState):
    # Coverage at lower K is compared first after maximum depth is fixed.
    # Saving and shorter runtime are deterministic tie breakers.
    return (
        state.coverage_profile,
        state.saving_profile,
        tuple(-value for value in state.time_profile),
        tuple(tuple(sorted(key)) for key in state.keys),
    )


def _best_nested_chain(
        parent: ParentRoute,
        times: dict[tuple[str, str], float],
        two_way_od: dict[tuple[str, str], float], *,
        kmin: int, kmax: int, endpoint_policy: str,
        max_route_min: float, min_saving_min: float):
    feasible_by_k: dict[int, dict[frozenset[int], Candidate]] = {}
    enumeration_stats: dict[int, dict] = {}
    states_by_k: dict[int, dict[frozenset[int], ChainState]] = {}

    for k in range(kmin, kmax + 1):
        feasible, stats = _enumerate_feasible(
            parent, k, times, two_way_od,
            endpoint_policy=endpoint_policy,
            max_route_min=max_route_min,
            min_saving_min=min_saving_min)
        feasible_by_k[k] = feasible
        enumeration_stats[k] = stats
        if k == kmin:
            states_by_k[k] = {
                key: ChainState(
                    keys=(key,), coverage_profile=(candidate.covered_demand,),
                    saving_profile=(candidate.saving_minutes,),
                    time_profile=(candidate.raw_time,))
                for key, candidate in feasible.items()
            }
            continue

        previous = states_by_k.get(k - 1, {})
        current: dict[frozenset[int], ChainState] = {}
        for key, candidate in feasible.items():
            predecessor_states = []
            for removed in key:
                if removed == 0:
                    continue
                predecessor = frozenset(set(key) - {removed})
                state = previous.get(predecessor)
                if state is not None:
                    predecessor_states.append(state)
            if not predecessor_states:
                continue
            best_predecessor = max(predecessor_states, key=_state_quality)
            current[key] = ChainState(
                keys=best_predecessor.keys + (key,),
                coverage_profile=(best_predecessor.coverage_profile
                                  + (candidate.covered_demand,)),
                saving_profile=(best_predecessor.saving_profile
                                + (candidate.saving_minutes,)),
                time_profile=(best_predecessor.time_profile
                              + (candidate.raw_time,)),
            )
        states_by_k[k] = current

    reachable = [k for k, states in states_by_k.items() if states]
    if not reachable:
        return [], feasible_by_k, enumeration_stats
    depth = max(reachable)
    best_state = max(states_by_k[depth].values(), key=_state_quality)
    chain = [feasible_by_k[k][key]
             for k, key in zip(range(kmin, depth + 1), best_state.keys)]
    return chain, feasible_by_k, enumeration_stats


def _canonical(stops: Iterable[str]) -> tuple[str, ...]:
    forward = tuple(stops)
    reverse = tuple(reversed(forward))
    return min(forward, reverse)


def _deduplicate(chains: dict[str, list[Candidate]],
                 parents: list[ParentRoute]):
    local_signatures = {_canonical(parent.stops): parent.route
                        for parent in parents}
    grouped: dict[tuple[str, ...], list[Candidate]] = {}
    for chain in chains.values():
        for candidate in chain:
            grouped.setdefault(_canonical(candidate.stops), []).append(candidate)

    winners: list[Candidate] = []
    duplicate_to: dict[tuple[str, int], str] = {}
    for signature, candidates in grouped.items():
        if signature in local_signatures:
            owner = local_signatures[signature]
            for candidate in candidates:
                duplicate_to[(candidate.parent_route, candidate.k)] = owner
            continue
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                -candidate.covered_demand, candidate.raw_time,
                candidate.parent_route, candidate.k))
        winner = ordered[0]
        winner_name = f"{winner.baseline}_EXP_K{winner.k}"
        winners.append(winner)
        for duplicate in ordered[1:]:
            duplicate_to[(duplicate.parent_route, duplicate.k)] = winner_name
    winners.sort(key=lambda candidate: (
        candidate.baseline, candidate.k, candidate.stops))
    return winners, duplicate_to


def _express_rows(winners: list[Candidate],
                  parent_by_route: dict[str, ParentRoute]) -> pd.DataFrame:
    rows = []
    for candidate in winners:
        parent = parent_by_route[candidate.parent_route]
        route_name = f"{candidate.baseline}_EXP_K{candidate.k}"
        selected = set(candidate.stops)
        skipped = [station for station in parent.stops
                   if station not in selected]
        for index, (station, cumulative) in enumerate(
                zip(candidate.stops, candidate.model_cum)):
            rows.append({
                "Route": route_name,
                "BaseLine": candidate.baseline,
                "Type": "EXP",
                "StopSeq": index + 1,
                "StationCode": station,
                "CumTime_min": cumulative,
                "SegTime_min": (0 if index == 0 else
                                cumulative - candidate.model_cum[index - 1]),
                "SkippedStations": (
                    "|".join(skipped) if index == 0 else ""),
                "OrigLine_NumStops": len(parent.stops),
                "OrigLine_TotalTime": parent.model_cum[-1],
                "Route_TotalTime": candidate.model_time,
                "TimeSaving_pct": round(candidate.saving_percent, 1),
                "DisruptionDemand": round(candidate.covered_demand, 4),
            })
    return pd.DataFrame(rows, columns=ROUTE_COLUMNS)


def _audit_rows(
        parents: list[ParentRoute], chains: dict[str, list[Candidate]],
        feasible_all: dict[str, dict[int, dict[frozenset[int], Candidate]]],
        stats_all: dict[str, dict[int, dict]], duplicate_to: dict,
        *, kmin: int, kmax: int, endpoint_policy: str,
        max_route_min: float, min_saving_min: float):
    rows = []
    for parent in parents:
        selected = {candidate.k: candidate
                    for candidate in chains.get(parent.route, [])}
        deepest = max(selected, default=None)
        for k in range(kmin, kmax + 1):
            candidate = selected.get(k)
            duplicate = duplicate_to.get((parent.route, k), "")
            stats = stats_all[parent.route][k]
            if k >= len(parent.stops):
                status = "NOT_PROPER_EXPRESS"
                reason = "K is not smaller than the parent LOC stop count"
            elif candidate is not None and duplicate:
                status = "DUPLICATE_PRUNED"
                reason = f"same topology already retained as {duplicate}"
            elif candidate is not None:
                status = "SELECTED"
                reason = ""
            elif not feasible_all[parent.route][k]:
                status = "NO_FEASIBLE_ROUTE"
                if stats["NumWithinTime"] == 0:
                    reason = f"no candidate has raw road time <= {max_route_min:g}"
                elif stats["NumFaster"] == 0:
                    reason = (
                        "no time-feasible candidate saves both "
                        f">= {min_saving_min:g} raw minute(s) and one model minute")
                elif stats["NumPositiveDemand"] == 0:
                    reason = "no feasible candidate covers positive OD demand"
                else:
                    reason = "no feasible candidate"
            else:
                status = "NO_NESTED_EXTENSION"
                reason = (
                    f"feasible K={k} sets exist but none extends the selected "
                    f"nested chain beyond K={deepest}")

            rows.append({
                "ParentRoute": parent.route,
                "BaseLine": parent.baseline,
                "Anchor": parent.stops[0],
                "K": k,
                "EndpointPolicy": endpoint_policy,
                "Status": status,
                "Route": (f"{candidate.baseline}_EXP_K{k}"
                          if candidate is not None else ""),
                "Stops": ("|".join(candidate.stops)
                          if candidate is not None else ""),
                "EndStation": (candidate.stops[-1]
                               if candidate is not None else ""),
                "RawRoadTime": (candidate.raw_time
                                if candidate is not None else None),
                "ModelRoadTime": (candidate.model_time
                                  if candidate is not None else None),
                "ComparableLocalRawTime": (
                    candidate.local_raw_time if candidate is not None else None),
                "ComparableLocalModelTime": (
                    candidate.local_model_time if candidate is not None else None),
                "SavingMinutes": (candidate.saving_minutes
                                  if candidate is not None else None),
                "SavingPercent": (candidate.saving_percent
                                  if candidate is not None else None),
                "CoveredDemand": (candidate.covered_demand
                                  if candidate is not None else None),
                **stats,
                "DuplicateOf": duplicate,
                "Reason": reason,
            })
    return rows


def _validate(
        local_rows: pd.DataFrame, parents: list[ParentRoute],
        chains: dict[str, list[Candidate]], winners: list[Candidate],
        output: pd.DataFrame, *, kmin: int, max_route_min: float,
        min_saving_min: float):
    parent_by_route = {parent.route: parent for parent in parents}
    if set(output["Type"].astype(str).str.upper()) - {"LOC", "EXP"}:
        raise AssertionError("Unexpected route type in output")
    output_local = output[output["Type"].astype(str).str.upper().eq("LOC")]
    pd.testing.assert_frame_equal(
        output_local[ROUTE_COLUMNS].reset_index(drop=True),
        local_rows[ROUTE_COLUMNS].reset_index(drop=True),
        check_dtype=False)

    for parent in parents:
        chain = chains.get(parent.route, [])
        previous: frozenset[int] | None = None
        expected_k = kmin
        for candidate in chain:
            if candidate.k != expected_k:
                raise AssertionError(
                    f"Nonconsecutive K chain for {parent.route}")
            if 0 not in candidate.indices:
                raise AssertionError(f"Anchor missing in {candidate}")
            if previous is not None and not previous < candidate.indices:
                raise AssertionError(f"Nesting violation for {parent.route}")
            if candidate.raw_time > max_route_min + EPS:
                raise AssertionError(f"EXP time cap violated by {candidate}")
            if candidate.saving_minutes + EPS < min_saving_min:
                raise AssertionError(f"Raw speed condition violated by {candidate}")
            if candidate.model_time >= candidate.local_model_time:
                raise AssertionError(f"Model speed condition violated by {candidate}")
            previous = candidate.indices
            expected_k += 1

    route_signatures: dict[tuple[str, ...], str] = {}
    for route_name, group in output.groupby("Route", sort=False):
        ordered = group.sort_values("StopSeq", kind="stable")
        stops = tuple(ordered["StationCode"].astype(str))
        signature = _canonical(stops)
        if signature in route_signatures:
            raise AssertionError(
                f"Duplicate topology: {route_name} and "
                f"{route_signatures[signature]}")
        route_signatures[signature] = str(route_name)
        cumulative = tuple(int(round(value)) for value in
                           ordered["CumTime_min"])
        if cumulative[0] != 0 or any(
                right <= left for left, right in zip(cumulative,
                                                     cumulative[1:])):
            raise AssertionError(f"Invalid cumulative times for {route_name}")
        if int(ordered.iloc[0]["Route_TotalTime"]) != cumulative[-1]:
            raise AssertionError(f"Total-time mismatch for {route_name}")

    if len(winners) != output[output["Type"].eq("EXP")]["Route"].nunique():
        raise AssertionError("Winner/output EXP count mismatch")
    horizon = int(cfg.get("time.horizon_min"))
    earliest_start = int(cfg.get("routes.local_earliest_start_min"))
    if max(parent.model_cum[-1] for parent in parents) + earliest_start > horizon:
        raise AssertionError(
            f"At least one manual LOC cannot finish by minute {horizon} even with "
            f"zero outbound deadhead and earliest run start at minute {earliest_start}")


def build(args):
    times = _read_road_times(args.distance)
    local_rows, parents = _load_parents(args.local, times)
    two_way_od = _load_two_way_od(args.od)

    chains: dict[str, list[Candidate]] = {}
    feasible_all = {}
    stats_all = {}
    for parent in parents:
        chain, feasible_by_k, enumeration_stats = _best_nested_chain(
            parent, times, two_way_od,
            kmin=args.kmin, kmax=args.kmax,
            endpoint_policy=args.endpoint_policy,
            max_route_min=args.max_route_min,
            min_saving_min=args.min_saving_min)
        chains[parent.route] = chain
        feasible_all[parent.route] = feasible_by_k
        stats_all[parent.route] = enumeration_stats

    winners, duplicate_to = _deduplicate(chains, parents)
    parent_by_route = {parent.route: parent for parent in parents}
    express = _express_rows(winners, parent_by_route)
    output = pd.concat([local_rows, express], ignore_index=True)

    _validate(
        local_rows, parents, chains, winners, output,
        kmin=args.kmin, max_route_min=args.max_route_min,
        min_saving_min=args.min_saving_min)
    audit = pd.DataFrame(_audit_rows(
        parents, chains, feasible_all, stats_all, duplicate_to,
        kmin=args.kmin, kmax=args.kmax,
        endpoint_policy=args.endpoint_policy,
        max_route_min=args.max_route_min,
        min_saving_min=args.min_saving_min))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False, encoding="utf-8-sig")
    audit.to_csv(args.audit, index=False, encoding="utf-8-sig")

    route_summary = output.groupby("Route", sort=False).agg(
        Type=("Type", "first"),
        BaseLine=("BaseLine", "first"),
        Stops=("StationCode", "size"),
        Time=("Route_TotalTime", "first"))
    exp_summary = route_summary[route_summary["Type"].eq("EXP")]
    print(f"Saved route file: {args.out}")
    print(f"Saved audit file: {args.audit}")
    print(f"LOC routes: {sum(route_summary['Type'].eq('LOC'))}")
    print(f"EXP routes before duplicate pruning: "
          f"{sum(len(chain) for chain in chains.values())}")
    print(f"EXP routes after duplicate pruning: {len(exp_summary)}")
    print(f"EXP road time: min={int(exp_summary['Time'].min())}, "
          f"max={int(exp_summary['Time'].max())}")
    print("Selected EXP by K:",
          exp_summary["Stops"].value_counts().sort_index().to_dict())
    print("Audit status:", audit["Status"].value_counts().to_dict())
    return output, audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL,
                        help="Manual LOC detail file (default: %(default)s)")
    parser.add_argument("--od", type=Path, default=DEFAULT_OD,
                        help="Hourly OD matrix used for demand coverage")
    parser.add_argument("--distance", type=Path, default=DEFAULT_DISTANCE,
                        help="Deterministic station road-time matrix")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--kmin", type=int, default=KMIN)
    parser.add_argument("--kmax", type=int, default=KMAX)
    parser.add_argument("--max-route-min", type=float, default=float(TMAX),
                        help="One-way raw road-time cap")
    parser.add_argument("--min-saving-min", type=float,
                        default=float(cfg.get("routes.express_min_saving_min")),
                        help="Minimum raw minutes saved against the same LOC span")
    parser.add_argument(
        "--endpoint-policy", choices=("demand", "parent"),
        default=cfg.get("routes.express_endpoint_policy"),
        help=("demand: retain only the upstream anchor and let demand choose "
              "the end; parent: force both parent LOC endpoints"))
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()
    if args.kmin < 2 or args.kmax < args.kmin:
        parser.error("require 2 <= kmin <= kmax")
    if args.max_route_min <= 0:
        parser.error("--max-route-min must be positive")
    if args.min_saving_min < 0:
        parser.error("--min-saving-min must be nonnegative")
    build(args)


if __name__ == "__main__":
    main()
