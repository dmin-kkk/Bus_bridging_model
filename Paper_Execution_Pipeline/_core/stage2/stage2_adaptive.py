"""
stage2_adaptive.py — 지속시간 불확실성을 고려하는 적응형 Stage 2

기존 구조
    p30·solve_stage2_fixed(D=30) + p60·solve_stage2_fixed(D=60) + p90·solve_stage2_fixed(D=90)
    → 세 문제를 독립으로 풀고 사후에 가중합. 각 문제는 D 를 '알고' 배차표를 짠다.
      Reviewer #1 (2) 가 지적한 wait-and-see / perfect-information 구조.

새 구조
    solve_stage2_adaptive()  하나의 문제
    → 배차변수 y 는 비예견성 정보집합을 공유하고,
      승객 흐름변수 b/a/l/q/w 는 지속시간별로 복제되며,
      목적은  min Σ_D P(D)·Z_D.

비예견성 (요구사항 5)
    장애 시작 후 τ 분에 배차하는 trip 은, 그 시점에 구분 불가능한 지속시간
    시나리오들에서 '같은 결정' 이어야 한다.

        τ ≤ disrupt_start + 30   →  (30, 60, 90)   아직 아무것도 모름
        τ ≤ disrupt_start + 60   →  (60, 90)       30분 종료를 관측함
        그 외                    →  (90,)          60분 종료까지 관측함

    등식 제약을 걸지 않고, 같은 정보집합의 시나리오가 **하나의 변수를 공유**하도록
    변수를 생성한다 (요구사항 5 단서).

Trip 가용성 (요구사항 6)
    후보 trip 은 90분 배차창으로 한 번만 생성한다.
    지속시간 D 에서 사용 가능한 조건:

        DispatchStart ≤ disrupt_start + D + DISPATCH_BUFFER

    DISPATCH_BUFFER = 0 이면 '장애가 종료된 시나리오에서는 이후 배차 없음' 이
    되고, 이때 가용성 분할과 정보집합 분할이 정확히 일치한다(기본값).
    양수로 두면 종료 직후 잔여 큐 처리를 허용하며, 그 구간의 배차는 이미
    종료를 관측한 뒤이므로 시나리오별 독립 변수가 된다.
"""

import csv
import importlib
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bbconfig import cfg  # noqa: E402

__all__ = ["info_cells", "trip_availability", "build_adaptive_group",
           "solve_stage2_adaptive", "DISPATCH_BUFFER", "WALKING_MINUTES",
           "EXACT_STATE_COMPRESSION", "EXACT_TRIP_CELL_PRUNING",
           "BUS_ONLY_NONANTICIPATIVITY",
           "PASSENGER_AND_BUS_NONANTICIPATIVITY", "reweight_flow_block"]

# 장애 종료 후 추가 배차 허용 시간(분). 0 이면 종료 즉시 배차 중단.
DISPATCH_BUFFER = 0
WALKING_MINUTES = int(cfg.get("operations.walking_transfer_min"))   # Table 3: sigma_ij
EXACT_STATE_COMPRESSION = False
EXACT_TRIP_CELL_PRUNING = False
_EXACT_ENGINE = None
# Gurobi can return an accepted incumbent with bound noise a few multiples of
# FeasibilityTol.  Extraction clamps only this numerical noise; larger
# negative passenger flows still fail loudly.
EXTRACTION_SOLUTION_TOLERANCE = 1e-5

BUS_ONLY_NONANTICIPATIVITY = "bus-only"
PASSENGER_AND_BUS_NONANTICIPATIVITY = "passenger-and-bus"
BUS_ONLY_WAIT_AND_SEE = "BUS_ONLY_WAIT_AND_SEE"
PASSENGER_AND_BUS_SHARED = "PASSENGER_AND_BUS_SHARED_RAW_DURATION_RAIL"
_NONANTICIPATIVITY_MODES = {
    BUS_ONLY_NONANTICIPATIVITY,
    PASSENGER_AND_BUS_NONANTICIPATIVITY,
}


def _normalize_nonanticipativity_mode(value):
    mode = str(value).strip().lower()
    if mode not in _NONANTICIPATIVITY_MODES:
        raise ValueError(
            f"unknown non-anticipativity mode {value!r}; expected one of "
            f"{sorted(_NONANTICIPATIVITY_MODES)}")
    return mode


def _load_exact_engine():
    """Load the already-audited Stage-1 exact state-reduction helpers."""
    global _EXACT_ENGINE
    if _EXACT_ENGINE is not None:
        return _EXACT_ENGINE
    source = (
        Path(__file__).resolve().parents[1]
        / "stage1" / "CG_Stage1_Stage2"
    )
    expected = (source / "cg_stage1.py").resolve()
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = importlib.import_module("cg_stage1")
    if Path(module.__file__).resolve() != expected:
        raise RuntimeError(
            f"unexpected exact-state engine: {module.__file__}; "
            f"expected {expected}"
        )
    _EXACT_ENGINE = module
    return module


def _aggregate_bus_load_groups(cache):
    """Return one deduplicated, continuous event ledger per (trip, OD).

    ``trip_od_list`` can contain several coverage records for the same
    passenger stream and trip.  A load state is nevertheless unique.  Board
    and alight variable keys are therefore treated as sets, and the state
    timeline spans every minute from the first retained event through the
    last.  This prevents duplicate or conflicting load-balance rows.
    """
    retained_board = set(cache.get("b_var_keys", ()))
    retained_alight = set(cache.get("a_var_keys", ()))
    has_event_support = "b_var_keys" in cache or "a_var_keys" in cache
    raw = {}

    for tid, od_list in cache.get("trip_od_list", {}).items():
        for od_info in od_list:
            omega = od_info["omega"]
            board_t = int(od_info["board_t"])
            alight_t = int(od_info["alight_t"])
            board_key = (omega, tid, board_t, od_info["board_j"])
            alight_key = (omega, tid, alight_t, od_info["alight_j"])
            keep_board = not has_event_support or board_key in retained_board
            keep_alight = not has_event_support or alight_key in retained_alight
            event_times = ([board_t] if keep_board else []) + (
                [alight_t] if keep_alight else [])
            if not event_times:
                continue

            group = raw.setdefault((tid, omega), {
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

    return raw


# =============================================================================
# 1. 비예견성 정보집합
# =============================================================================
def info_cells(tau, disrupt_start, durations=(30, 60, 90), buffer=DISPATCH_BUFFER):
    """
    배차시각 tau 에서 만들어야 할 배차변수의 '정보집합' 목록.

    각 원소는 하나의 y 변수를 공유하는 duration 튜플이다.

      - 아직 진행 중인(구분 불가) 시나리오들 → 하나로 묶인다
      - 이미 종료를 관측한 시나리오들       → 각각 독립 (buffer > 0 일 때만 등장)

    buffer=0 이면 결과는 요구사항 5 의 shared_group 과 정확히 같다:
        tau <= start+30 -> [(30,60,90)]
        tau <= start+60 -> [(60,90)]
        그 외           -> [(90,)]
    """
    durations = tuple(sorted(durations))
    avail = tuple(D for D in durations if tau <= disrupt_start + D + buffer)
    if not avail:
        return []
    ongoing = tuple(D for D in avail if tau <= disrupt_start + D)
    ended = tuple(D for D in avail if tau > disrupt_start + D)
    cells = []
    if ongoing:
        cells.append(ongoing)
    cells.extend((D,) for D in ended)      # 종료를 관측한 세계는 서로 구분된다
    return cells


def trip_availability(trip, disrupt_start, durations=(30, 60, 90),
                      buffer=DISPATCH_BUFFER):
    """지속시간 D 에서 이 trip 을 배차할 수 있는가 → {D: bool}."""
    tau = trip["DispatchStart"]
    return {D: (tau <= disrupt_start + D + buffer) for D in durations}


def _travel_minutes(travel_times, origin, destination):
    origin, destination = str(origin), str(destination)
    if origin == destination:
        return 0
    value = travel_times.get((origin, destination))
    if value is None:
        value = travel_times.get((destination, origin))
    if value is None:
        return None
    value = float(value)
    if value < 0:
        return None
    return int(round(value))


def _trip_run_start(trip):
    return int(trip.get("RunStart", trip["DispatchStart"]))


def _trip_service_end(trip):
    return int(trip.get(
        "ServiceEnd", trip.get("RunEnd", trip.get(
            "ResourceEnd", trip["DispatchStart"]))))


def _trip_start_station(trip):
    station = trip.get("RouteOrigin")
    if station is None:
        stops = tuple(trip.get("RouteStops", ()))
        station = stops[0] if stops else None
    if station is None:
        raise ValueError(f"trip {trip.get('TripID')} has no start station")
    return str(station)


def _trip_end_station(trip):
    station = trip.get("ServiceEndStation", trip.get("RouteDest"))
    if station is None:
        stops = tuple(trip.get("RouteStops", ()))
        station = stops[-1] if stops else None
    if station is None:
        raise ValueError(f"trip {trip.get('TripID')} has no end station")
    return str(station)


def _compatible_trip_arcs(trips, travel_times, turnaround_minutes=0):
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
            for nxt in ordered:
                if nxt == tid or _trip_run_start(normal[nxt]) < ready:
                    continue
                deadhead = _travel_minutes(
                    travel_times, _trip_end_station(first),
                    _trip_start_station(normal[nxt]))
                if (deadhead is not None and
                        ready + deadhead <= _trip_run_start(normal[nxt])):
                    successors[tid].append(nxt)
    return {tid: tuple(values) for tid, values in successors.items()}


def _chain_arc_decision_time(predecessor, successor, travel_times):
    deadhead = _travel_minutes(
        travel_times, _trip_end_station(predecessor),
        _trip_start_station(successor))
    if deadhead is None:
        raise ValueError("chain arc has no finite inter-trip deadhead")
    return _trip_run_start(successor) - int(deadhead)


# =============================================================================
# 2. 그룹 구성 — 공통 trip 집합 + 지속시간별 cache
# =============================================================================
def _normal_affected_ods(normal, theta, tolerance=1e-9):
    """Return normal ODs using a train after its withdrawal time."""
    affected = set()
    for family in ("l", "b", "a"):
        for key, value in normal.get(family, {}).items():
            if float(value) <= tolerance:
                continue
            omega, vehicle, minute = key[:3]
            if vehicle is not None and theta.get((str(vehicle), int(minute)), 0):
                affected.add(omega)
    return affected


def _normal_station_support(normal, omegas):
    """Return stations used by each common OD's positive normal trajectory.

    The reduced ``N_omega`` construction cannot infer every transfer station
    used by a normal passenger path.  Those stations must nevertheless exist
    in every duration cache when an OD is fixed to normal in one branch.
    Train-load and exit states have no station component and are already
    generated over the full time horizon.
    """
    common = set(omegas)
    support = defaultdict(set)
    for family in ("b", "a"):
        for (omega, _vehicle, _minute, station), value in normal.get(
                family, {}).items():
            if omega in common and station is not None and float(value) > 0:
                support[omega].add(station)
    for family in ("q", "w"):
        for (omega, station, _minute), value in normal.get(
                family, {}).items():
            if omega in common and station is not None and float(value) > 0:
                support[omega].add(station)
    return {omega: tuple(sorted(support.get(omega, ())))
            for omega in common}


def _normalized_text(value):
    if value is None:
        return None
    result = str(value).strip()
    if not result:
        return None
    return result[:-2] if result.endswith(".0") else result


def _schedule_minute(value):
    """Return an integral schedule minute, or ``None`` when unavailable."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return int(numeric) if math.isfinite(numeric) else None


def _termination_alight_events(scenario, wdata):
    """Locate the safe physical clearance event of each withdrawn train.

    Service withdrawal and passenger clearance are deliberately distinct.  A
    train whose next departure is suppressed clears at that dwell station; a
    train withdrawn on exposure clears at the block stop when it calls there,
    or at its first scheduled stop after exposure otherwise.
    """
    block_station = str(scenario.get(
        "block_station", scenario.get("station", ""))).strip()
    exposure = {
        str(vehicle): int(minute)
        for vehicle, minute in scenario.get(
            "exposure_time_by_train", {}).items()
    }
    trip_stops = wdata.get("trip_stops_seq", {})
    active_times = {
        str(vehicle): tuple(int(minute) for minute in minutes)
        for vehicle, minutes in wdata["train_active_times"].items()
    }
    events = {}

    for vehicle, raw_start in scenario[
            "physical_cancellation_start_by_train"].items():
        vehicle, start = str(vehicle), int(raw_start)
        if vehicle not in exposure:
            raise ValueError(
                f"{scenario['id']}: exposure minute missing for withdrawn "
                f"train {vehicle}")

        event_minute = start
        if start < exposure[vehicle]:
            departure_matches, arrival_matches = [], []
            for _sequence, raw_station, arrival, departure in trip_stops.get(
                    vehicle, ()):
                station = _normalized_text(raw_station)
                if station is None:
                    continue
                if _schedule_minute(departure) == start:
                    departure_matches.append(station)
                if _schedule_minute(arrival) == start:
                    arrival_matches.append(station)
            matches = tuple(dict.fromkeys(
                departure_matches or arrival_matches))
            if len(matches) != 1:
                raise ValueError(
                    f"{scenario['id']}: cannot identify one evacuation "
                    f"station for train {vehicle} at t={start}; scheduled "
                    f"matches={matches}")
            station = matches[0]
        elif start == exposure[vehicle]:
            target_matches, later_stops = [], []
            for sequence, raw_station, arrival, departure in trip_stops.get(
                    vehicle, ()):
                station = _normalized_text(raw_station)
                if station is None:
                    continue
                arrival_minute = _schedule_minute(arrival)
                departure_minute = _schedule_minute(departure)
                if station == block_station:
                    target_minute = (arrival_minute if arrival_minute is not None
                                     else departure_minute)
                    if target_minute is not None:
                        target_matches.append((target_minute, station))
                next_minute = (arrival_minute if arrival_minute is not None
                               else departure_minute)
                if next_minute is not None and next_minute > exposure[vehicle]:
                    later_stops.append((next_minute, int(sequence), station))
            calling = [item for item in target_matches
                       if item[0] == exposure[vehicle]]
            if calling:
                event_minute, station = calling[0]
            elif later_stops:
                event_minute, _sequence, station = min(later_stops)
            else:
                raise ValueError(
                    f"{scenario['id']}: noncalling train {vehicle} has no "
                    f"scheduled clearance stop after exposure "
                    f"t={exposure[vehicle]}")
        else:
            raise ValueError(
                f"{scenario['id']}: withdrawal of train {vehicle} starts "
                f"after its exposure ({start}>{exposure[vehicle]})")

        if event_minute not in active_times.get(vehicle, ()):
            raise ValueError(
                f"{scenario['id']}: evacuation event for train {vehicle} "
                f"at t={event_minute} lies outside its load stream")
        events[vehicle] = (int(event_minute), str(station))
    return events


def _passenger_capacity_theta(events, wdata):
    """Return the capacity mask beginning at physical passenger clearance."""
    theta = {}
    active_times = wdata["train_active_times"]
    for vehicle, (clearance_minute, _station) in events.items():
        for minute in active_times.get(str(vehicle), ()):
            if int(minute) >= int(clearance_minute):
                theta[(str(vehicle), int(minute))] = 1
    return theta


def _termination_alight_keys(scenario):
    """Return normalized forced-detrain events without an OD component."""
    result = set()
    for raw_vehicle, raw_event in scenario.get(
            "termination_alight_event_by_train", {}).items():
        if isinstance(raw_event, dict):
            minute, station = int(raw_event["minute"]), str(raw_event["station"])
        else:
            minute, station = raw_event
            minute, station = int(minute), str(station)
        result.add((str(raw_vehicle), minute, station))
    return frozenset(result)


def _termination_alight_support_keys(scenario):
    """Return the common action support without changing physical events.

    Duration branches retain their own forced-clearance events.  Passenger
    actions still need identical pre-revelation indices, so the union of those
    events is carried separately as support metadata.  Falling back to the
    physical events preserves the historical single-branch behavior.
    """
    raw_support = scenario.get("required_termination_alight_support")
    if raw_support is None:
        return _termination_alight_keys(scenario)
    return frozenset(
        (str(vehicle), int(minute), str(station))
        for vehicle, minute, station in raw_support
    )


def _forbidden_origin_alight(key, omega_map, termination_event_keys):
    """Whether an alighting action is an ordinary prohibited origin exit."""
    omega, vehicle, minute, station = key
    return bool(
        omega in omega_map and station == omega_map[omega][0]
        and (str(vehicle), int(minute), str(station))
        not in termination_event_keys
    )


def _dense_train_action_keys(
        omega, vehicle, minute, ordinary_station, termination_by_train):
    """Return the b key and all a keys used by one dense train recurrence."""
    vehicle, minute = str(vehicle), int(minute)
    ordinary_key = (
        (omega, vehicle, minute, str(ordinary_station))
        if ordinary_station is not None else None
    )
    alight_keys = [] if ordinary_key is None else [ordinary_key]
    termination = termination_by_train.get(vehicle)
    if termination is not None and minute == int(termination[0]):
        termination_key = (
            omega, vehicle, minute, str(termination[1]))
        if termination_key != ordinary_key:
            alight_keys.append(termination_key)
    return ordinary_key, tuple(alight_keys)


def _pair_score_for_ods(omegas, wdata):
    score = {}
    for omega in omegas:
        if omega not in wdata["Omega_Map"]:
            continue
        origin, destination = wdata["Omega_Map"][omega]
        value = sum(float(wdata["D"].get((omega, origin, minute), 0.0))
                    for minute in wdata["Gamma"])
        score[(origin, destination)] = (
            score.get((origin, destination), 0.0) + value)
    return score


def _audit_zero_impact_reconstruction(
        scenario_id, manifest_path, canceled, starts, exposure, cases):
    """Validate a missing result CSV against its NO_AFFECTED stats row."""
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"{scenario_id}: disruption stats manifest missing: {path}")
    row = None
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for candidate in csv.DictReader(stream):
            if str(candidate.get("scenario", "")).strip() == scenario_id:
                row = candidate
                break
    if row is None:
        raise ValueError(
            f"{scenario_id}: absent from disruption stats manifest {path}")

    leads = [
        float(int(exposure[vehicle]) - int(starts[vehicle]))
        for vehicle in canceled
    ]
    actual = {
        "canceled_trains": len(canceled),
        "case_a_canceled_trains": sum(
            str(cases.get(vehicle, "")).upper() == "A"
            for vehicle in canceled),
        "case_b_canceled_trains": sum(
            str(cases.get(vehicle, "")).upper() == "B"
            for vehicle in canceled),
        "mean_cancel_lead_min": (
            math.fsum(leads) / len(leads) if leads else 0.0),
        "max_cancel_lead_min": max(leads, default=0.0),
    }
    expected = {}
    for key in actual:
        raw = row.get(key)
        if raw is None or str(raw).strip() == "":
            raise ValueError(
                f"{scenario_id}: {key} missing from disruption stats {path}")
        expected[key] = float(raw)
    mismatches = {
        key: {"expected": expected[key], "actual": float(actual[key])}
        for key in actual
        if not math.isclose(
            float(actual[key]), expected[key], rel_tol=1e-9, abs_tol=1e-9)
    }
    if str(row.get("checkpoint", "")).strip() != "NO_AFFECTED":
        mismatches["checkpoint"] = {
            "expected": "NO_AFFECTED",
            "actual": str(row.get("checkpoint", "")).strip(),
        }
    if mismatches:
        raise ValueError(
            f"{scenario_id}: reconstructed raw rail branch disagrees with "
            f"{path}: {mismatches}")
    return {
        "passed": True,
        "manifest": str(path),
        "checkpoint": "NO_AFFECTED",
        "expected": expected,
        "actual": actual,
    }


def _physical_duration_scenario(
        station, start_h, duration, base, basis, wdata, buffer):
    """Materialize a duration branch even when its affected-OD CSV is absent.

    Scenario extraction historically omitted a duration when no normal OD used
    its cancelled trains.  A proper scenario tree still needs that physical
    branch, because it shares decisions with the other duration before the
    recovery observation.  The longest-duration timing map contains every
    train exposed in a shorter duration and is therefore a sufficient source
    for the missing cancellation mask.
    """
    duration = int(duration)
    scenario = dict(base if base is not None else basis)
    disruption_start = int(basis["DISRUPT_START_T"])
    disruption_end = disruption_start + duration
    scenario.update({
        "id": f"{int(start_h)}_{station}_{duration}",
        "station": station,
        "duration": duration,
        "DISRUPT_START_T": disruption_start,
        "DISRUPT_END_T": disruption_end,
        "DISPATCH_END_MIN": disruption_end + int(buffer),
    })
    scenario_id = scenario["id"]

    exposure = basis.get("exposure_time_by_train", {})
    basis_starts = basis.get("cancellation_start_by_train", {})
    if base is not None:
        canceled = {str(vehicle) for vehicle in base.get(
            "canceled_trains", ())}
        starts = {
            str(vehicle): int(minute)
            for vehicle, minute in base.get(
                "cancellation_start_by_train", {}).items()
            if str(vehicle) in canceled
        }
    else:
        canceled = {
            str(vehicle) for vehicle, minute in exposure.items()
            if disruption_start <= int(minute) <= disruption_end
        }
        starts = {
            vehicle: int(basis_starts[vehicle])
            for vehicle in canceled if vehicle in basis_starts
        }
    missing_starts = canceled - set(starts)
    if missing_starts:
        raise ValueError(
            f"{station}/D{duration}: cancellation starts missing for "
            f"{sorted(missing_starts)[:5]}")

    if base is not None and "theta" in base:
        # Preserve the disruption generator's branch verbatim (including its
        # explicit zero entries).  Reconstruct only a physically missing
        # branch, for which no raw snapshot exists.
        theta = {
            (str(vehicle), int(minute)): int(value)
            for (vehicle, minute), value in base["theta"].items()
        }
    else:
        theta = {}
        for vehicle, start in starts.items():
            for minute in wdata["train_active_times"].get(vehicle, ()):
                if int(minute) >= start:
                    theta[(vehicle, int(minute))] = 1
    scenario["canceled_trains"] = set(canceled)
    scenario["cancellation_start_by_train"] = dict(starts)
    scenario["theta"] = dict(theta)
    scenario["physical_canceled_trains"] = set(canceled)
    scenario["physical_cancellation_start_by_train"] = dict(starts)
    scenario["physical_theta"] = dict(theta)
    scenario["physical_termination_alight_event_by_train"] = (
        _termination_alight_events(scenario, wdata)
    )
    scenario["physical_passenger_capacity_theta"] = (
        _passenger_capacity_theta(
            scenario["physical_termination_alight_event_by_train"], wdata)
    )
    if base is None:
        scenario["zero_impact_reconstruction_audit"] = (
            _audit_zero_impact_reconstruction(
                scenario_id,
                basis.get("logic_manifest", ""),
                canceled,
                starts,
                exposure,
                basis.get("cancellation_case_by_train", {}),
            )
        )
        scenario["raw_rail_snapshot_source"] = "NO_AFFECTED_STATS_RECONSTRUCTION"
    else:
        scenario["zero_impact_reconstruction_audit"] = None
        scenario["raw_rail_snapshot_source"] = "DISRUPTION_SCENARIO"
    scenario["reported_affected_omega"] = set(
        base.get("Omega_m", ()) if base is not None else ())
    return scenario


def _initialize_branch_local_train_withdrawals(scenarios, durations):
    """Attach passenger-clearance state without changing either raw branch."""
    support = set()
    for duration in map(int, durations):
        scenario = scenarios[duration]
        physical_events = dict(
            scenario["physical_termination_alight_event_by_train"])
        scenario["termination_alight_event_by_train"] = physical_events
        scenario["passenger_capacity_theta"] = dict(
            scenario["physical_passenger_capacity_theta"])
        scenario["raw_duration_rail_state_preserved"] = True
        support.update(
            (str(vehicle), int(event[0]), str(event[1]))
            for vehicle, event in physical_events.items()
        )

    common_support = tuple(sorted(support))
    support_by_vehicle = defaultdict(set)
    for vehicle, minute, station in common_support:
        support_by_vehicle[vehicle].add((minute, station))
    conflicting = {
        vehicle: tuple(sorted(events))
        for vehicle, events in support_by_vehicle.items()
        if len(events) > 1
    }
    if conflicting:
        raise ValueError(
            "duration branches give one train conflicting passenger-clearance "
            f"events: {conflicting}")
    for duration in map(int, durations):
        scenarios[duration]["required_termination_alight_support"] = (
            common_support)
        scenarios[duration]["preserve_common_rail_event_support"] = True


def _rebuild_common_spare_capacity(
        scenario, common_omegas, normal, wdata, train_capacity):
    """Remove the common modelled population from normal occupancy once."""
    total = defaultdict(float)
    modeled = defaultdict(float)
    common = set(common_omegas)
    for (omega, vehicle, minute), value in normal.get("l", {}).items():
        key = (str(vehicle), int(minute))
        total[key] += float(value)
        if omega in common:
            modeled[key] += float(value)
    # Service withdrawal prohibits new boarding in the flow block.  An
    # already-onboard passenger remains physically present until the explicit
    # branch-local clearance event, so capacity is removed only from then on.
    clearance_theta = scenario.get(
        "passenger_capacity_theta", scenario["theta"])
    capacity = {}
    for vehicle in wdata["Vm"]:
        vehicle = str(vehicle)
        for minute in wdata["train_active_times"].get(vehicle, ()):
            key = (vehicle, int(minute))
            residual = float(train_capacity) - total.get(
                key, 0.0) + modeled.get(key, 0.0)
            capacity[key] = max(0.0, residual) * (
                1 - int(clearance_theta.get(key, 0)))
    scenario["spare_capacity"] = capacity


def _rail_information_set_audit(
        scenarios, caches, durations, wdata, shared_start, shared_end):
    """Audit common action support and duration-specific physical rail state.

    Timetable events are constructed once in ``wdata`` and then supplied to
    both duration caches.  Cancellations, withdrawal masks, forced-clearance
    events, and residual capacities deliberately remain equal to each raw
    duration branch; they are audited for preservation, not cross-duration
    equality.  Passenger action indices remain common on the sharing interval.
    """
    durations = tuple(sorted(map(int, durations)))
    shared_start, shared_end = int(shared_start), int(shared_end)
    i_vts = {
        (str(vehicle), int(minute), station)
        for vehicle, minute, station in wdata.get("I_vts", ())
    }
    valid_keys = {
        (str(vehicle), int(minute), station)
        for vehicle, minute, station in wdata.get("valid_keys", ())
    }
    station_of = wdata.get("station_of", {})
    active_times = wdata.get("train_active_times", {})
    timetable_consistent = i_vts == valid_keys and all(
        station_of.get(vehicle, {}).get(minute) == station
        and minute in active_times.get(vehicle, ())
        for vehicle, minute, station in i_vts
    )

    def event_support(cache):
        support = set()
        for family in ("b", "a"):
            for omega, vehicle, minute, station in cache.get(
                    f"{family}_var_keys", ()):
                vehicle, minute = str(vehicle), int(minute)
                if (not vehicle.startswith("B_") and
                        shared_start <= minute < shared_end):
                    support.add((family, omega, vehicle, minute, station))
        for omega, vehicle, minute in cache.get("l_var_keys", ()):
            vehicle, minute = str(vehicle), int(minute)
            if (not vehicle.startswith("B_") and
                    shared_start <= minute < shared_end):
                support.add(("l", omega, vehicle, minute, None))
        return support

    support_by_duration = {
        duration: event_support(caches[duration]) for duration in durations
    }
    first_support = support_by_duration[durations[0]] if durations else set()
    event_support_mismatch_count = sum(
        len(first_support.symmetric_difference(support_by_duration[duration]))
        for duration in durations[1:]
    )

    termination_events_verified = True
    termination_event_count = 0
    passenger_capacity_theta_matches_events = True
    raw_canceled_trains_preserved = True
    raw_cancellation_starts_preserved = True
    raw_theta_preserved = True
    raw_termination_events_preserved = True
    raw_passenger_capacity_theta_preserved = True
    common_termination_action_support_verified = True
    zero_impact_reconstruction_count = 0
    zero_impact_reconstructions_verified = True
    shared_withdrawal_events_by_duration = {}
    required_termination_support_by_duration = {}
    for duration in durations:
        scenario = scenarios[duration]
        reconstruction = scenario.get("zero_impact_reconstruction_audit")
        if reconstruction is not None:
            zero_impact_reconstruction_count += 1
            zero_impact_reconstructions_verified &= bool(
                reconstruction.get("passed"))
        scenario_events = {
            (vehicle, minute, station)
            for vehicle, minute, station in _termination_alight_keys(
                scenario)
        }
        required_support = set(_termination_alight_support_keys(scenario))
        required_termination_support_by_duration[duration] = required_support
        cache_events = {
            (str(vehicle), int(event[0]), str(event[1]))
            for vehicle, event in caches[duration].get(
                "termination_alight_event_by_train", {}).items()
        }
        starts = {
            str(vehicle): int(minute)
            for vehicle, minute in scenario.get(
                "cancellation_start_by_train", {}).items()
        }
        physical_canceled = {
            str(vehicle) for vehicle in scenario.get(
                "physical_canceled_trains", ())
        }
        actual_canceled = {
            str(vehicle) for vehicle in scenario.get("canceled_trains", ())
        }
        physical_starts = {
            str(vehicle): int(minute)
            for vehicle, minute in scenario.get(
                "physical_cancellation_start_by_train", {}).items()
        }
        physical_theta = {
            (str(vehicle), int(minute)): int(value)
            for (vehicle, minute), value in scenario.get(
                "physical_theta", {}).items()
        }
        actual_theta = {
            (str(vehicle), int(minute)): int(value)
            for (vehicle, minute), value in scenario.get("theta", {}).items()
        }
        physical_events = {
            (str(vehicle), int(event[0]), str(event[1]))
            for vehicle, event in scenario.get(
                "physical_termination_alight_event_by_train", {}).items()
        }
        physical_clearance_theta = {
            (str(vehicle), int(minute)): int(value)
            for (vehicle, minute), value in scenario.get(
                "physical_passenger_capacity_theta", {}).items()
        }
        actual_clearance_theta = {
            (str(vehicle), int(minute)): int(value)
            for (vehicle, minute), value in scenario.get(
                "passenger_capacity_theta", {}).items()
        }
        raw_canceled_trains_preserved &= actual_canceled == physical_canceled
        raw_cancellation_starts_preserved &= starts == physical_starts
        raw_theta_preserved &= actual_theta == physical_theta
        raw_termination_events_preserved &= scenario_events == physical_events
        raw_passenger_capacity_theta_preserved &= (
            actual_clearance_theta == physical_clearance_theta)
        common_termination_action_support_verified &= (
            required_support == {
                (str(vehicle), int(minute), str(station))
                for vehicle, minute, station in caches[duration].get(
                    "required_termination_alight_support", ())
            }
        )
        shared_withdrawal_events = set()
        for vehicle, minute, station in scenario_events:
            withdrawal = starts.get(vehicle)
            if withdrawal is None:
                termination_events_verified = False
            elif withdrawal < shared_end:
                # Retain this raw physical event only in its duration branch.
                # The separate action-support union is audited below.
                shared_withdrawal_events.add((vehicle, minute, station))
        shared_withdrawal_events_by_duration[duration] = (
            shared_withdrawal_events)
        termination_event_count += len(scenario_events)
        if scenario_events != cache_events:
            termination_events_verified = False
        raw_actual_clearance_theta = scenario.get("passenger_capacity_theta")
        expected_clearance_theta = _passenger_capacity_theta({
            vehicle: (minute, station)
            for vehicle, minute, station in scenario_events
        }, wdata)
        if raw_actual_clearance_theta is None:
            passenger_capacity_theta_matches_events = False
        else:
            for vehicle in map(str, wdata.get("Vm", ())):
                for minute in active_times.get(vehicle, ()):
                    key = (vehicle, int(minute))
                    if int(raw_actual_clearance_theta.get(key, 0)) != int(
                            expected_clearance_theta.get(key, 0)):
                        passenger_capacity_theta_matches_events = False
        a_keys = {
            (omega, str(vehicle), int(minute), str(station))
            for omega, vehicle, minute, station in caches[duration].get(
                "a_var_keys", ())
        }
        for vehicle, minute, station in scenario_events:
            for omega in scenario.get("Omega_m", ()):
                key = (omega, vehicle, minute, station)
                if key not in a_keys:
                    termination_events_verified = False

        # Every support-only event must be materialized in the action cache.
        for vehicle, minute, station in required_support:
            for omega in scenario.get("Omega_m", ()):
                if (omega, vehicle, minute, station) not in a_keys:
                    common_termination_action_support_verified = False

    termination_event_support_mismatch_count = 0
    termination_action_support_mismatch_count = 0
    if durations:
        reference = shared_withdrawal_events_by_duration[durations[0]]
        termination_event_support_mismatch_count = sum(
            len(reference.symmetric_difference(
                shared_withdrawal_events_by_duration[duration]))
            for duration in durations[1:]
        )
        support_reference = required_termination_support_by_duration[
            durations[0]]
        termination_action_support_mismatch_count = sum(
            len(support_reference.symmetric_difference(
                required_termination_support_by_duration[duration]))
            for duration in durations[1:]
        )

    theta_mismatch_count = 0
    passenger_capacity_theta_mismatch_count = 0
    passenger_capacity_theta_present = all(
        "passenger_capacity_theta" in scenarios[duration]
        for duration in durations
    )
    capacity_max_abs_diff = 0.0
    if len(durations) > 1:
        reference = durations[0]
        vehicles = tuple(map(str, wdata.get("Vm", ())))
        for duration in durations[1:]:
            for vehicle in vehicles:
                for minute in range(shared_start, shared_end):
                    left_theta = int(scenarios[reference].get(
                        "theta", {}).get((vehicle, minute), 0))
                    right_theta = int(scenarios[duration].get(
                        "theta", {}).get((vehicle, minute), 0))
                    theta_mismatch_count += int(left_theta != right_theta)
                    left_clearance = int(scenarios[reference].get(
                        "passenger_capacity_theta", {}).get(
                            (vehicle, minute), 0))
                    right_clearance = int(scenarios[duration].get(
                        "passenger_capacity_theta", {}).get(
                            (vehicle, minute), 0))
                    passenger_capacity_theta_mismatch_count += int(
                        left_clearance != right_clearance)
                    left_capacity = float(scenarios[reference].get(
                        "spare_capacity", {}).get((vehicle, minute), 0.0))
                    right_capacity = float(scenarios[duration].get(
                        "spare_capacity", {}).get((vehicle, minute), 0.0))
                    capacity_max_abs_diff = max(
                        capacity_max_abs_diff,
                        abs(left_capacity - right_capacity),
                    )

    return {
        "single_common_event_timetable": bool(timetable_consistent),
        "event_support_nonanticipative": (
            event_support_mismatch_count == 0),
        "event_support_mismatch_count": int(
            event_support_mismatch_count),
        "termination_alight_events_verified": bool(
            termination_events_verified),
        "termination_alight_event_count": int(termination_event_count),
        "termination_event_support_nonanticipative": (
            termination_event_support_mismatch_count == 0),
        "termination_event_support_mismatch_count": int(
            termination_event_support_mismatch_count),
        "termination_action_support_nonanticipative": bool(
            common_termination_action_support_verified
            and termination_action_support_mismatch_count == 0),
        "termination_action_support_mismatch_count": int(
            termination_action_support_mismatch_count),
        "theta_nonanticipative": theta_mismatch_count == 0,
        "theta_mismatch_count": int(theta_mismatch_count),
        "passenger_capacity_theta_present": bool(
            passenger_capacity_theta_present),
        "passenger_capacity_theta_matches_events": bool(
            passenger_capacity_theta_matches_events),
        "passenger_capacity_theta_nonanticipative": (
            passenger_capacity_theta_present
            and passenger_capacity_theta_matches_events
            and passenger_capacity_theta_mismatch_count == 0),
        "passenger_capacity_theta_mismatch_count": int(
            passenger_capacity_theta_mismatch_count),
        "capacity_nonanticipative": capacity_max_abs_diff <= 1e-9,
        "capacity_max_abs_diff": float(capacity_max_abs_diff),
        "raw_canceled_trains_preserved": bool(
            raw_canceled_trains_preserved),
        "raw_cancellation_starts_preserved": bool(
            raw_cancellation_starts_preserved),
        "raw_theta_preserved": bool(raw_theta_preserved),
        "raw_termination_events_preserved": bool(
            raw_termination_events_preserved),
        "raw_passenger_capacity_theta_preserved": bool(
            raw_passenger_capacity_theta_preserved),
        "raw_branch_rail_state_preserved": bool(
            raw_canceled_trains_preserved
            and raw_cancellation_starts_preserved
            and raw_theta_preserved
            and raw_termination_events_preserved
            and raw_passenger_capacity_theta_preserved
            and zero_impact_reconstructions_verified),
        "zero_impact_reconstruction_count": int(
            zero_impact_reconstruction_count),
        "zero_impact_reconstructions_verified": bool(
            zero_impact_reconstructions_verified),
        "timetable_event_count": int(len(i_vts)),
    }


def _raw_duration_rail_stats(rail_audit):
    """Separate raw-branch equality diagnostics from the policy contract.

    Raw-duration-rail mode never claims rail-state non-anticipativity: each
    duration keeps its own realized input even when two particular raw inputs
    happen to be numerically equal.  Equality remains useful diagnostic
    evidence and is therefore reported under explicit non-policy names.
    """
    theta_equal = bool(rail_audit.get("theta_nonanticipative"))
    clearance_equal = bool(
        rail_audit.get("passenger_capacity_theta_nonanticipative"))
    capacity_equal = bool(rail_audit.get("capacity_nonanticipative"))
    termination_equal = bool(
        rail_audit.get("termination_event_support_nonanticipative"))
    return {
        "RawRailThetaEqualAcrossDurations": theta_equal,
        "RawPassengerCapacityThetaEqualAcrossDurations": clearance_equal,
        "RawTrainCapacityEqualAcrossDurations": capacity_equal,
        "RawTerminationEventsEqualAcrossDurations": termination_equal,
        "RawRailAvailabilityEqualAcrossDurations": bool(
            theta_equal and clearance_equal and capacity_equal
            and termination_equal),
        "TrainAvailabilityInformationSetNonanticipative": False,
    }


def build_adaptive_group(m3, station, start_h, wdata, raw_scenarios,
                         depots, bus_tt, route_lib, durations=(30, 60, 90),
                         buffer=DISPATCH_BUFFER, trip_basis_duration=None,
                         nonanticipativity_mode=
                         BUS_ONLY_NONANTICIPATIVITY):
    """
    한 사고 위치(station)에 대해 적응형 Stage 2 입력을 만든다.

    m3 : 3MILP 모듈 객체 (select_routes / generate_bus_trips /
         build_stage2_cache / build_scenario_route_library 사용)

    반환 dict
      station, disrupt_start, durations
      active         피해가 있는 지속시간 (나머지는 Z_D ≡ 0)
      scen[D]        지속시간별 시나리오 dict (피해 없으면 None)
      trips          공통 trip 집합 (가장 긴 지속시간 기준으로 1회 생성)
      avail[D]       그 지속시간에서 쓸 수 있는 tid 집합
      cache[D]       지속시간별 Stage 2 인덱스 캐시 (피해 없으면 None)
      cells          {tid: [정보집합, ...]}

    짧은 지속시간에 피해 OD 가 없는 역이 윈도우당 12~18개 있다 (Omega_m 이 비어
    raw 에 아예 없다). 이는 결측이 아니라 '피해 없음' 이므로 Z_D = 0 인 지속시간
    으로 취급한다. 배차변수와 정보집합은 물리적 실행가능성으로만 정해지므로
    durations 전체를 그대로 쓴다 — 30분에 아무도 안 묶인다는 것은 배차 시점에
    알 수 없는 사실이고, 바로 그래서 비예견성이 구속된다.
    """
    mode = _normalize_nonanticipativity_mode(nonanticipativity_mode)
    passenger_nonanticipativity = (
        mode == PASSENGER_AND_BUS_NONANTICIPATIVITY)
    durations = tuple(sorted(int(D) for D in durations))
    by_dur = {
        int(s["duration"]): s for s in raw_scenarios
        if s["station"] == station
    }
    basis_D = int(trip_basis_duration) if trip_basis_duration is not None \
        else max(durations)
    if basis_D not in by_dur:
        # 임팩트는 지속시간에 단조이므로 최장에 피해가 없으면 어디에도 없다.
        raise ValueError(f"station={station}: duration {basis_D} 시나리오가 없음")

    basis = by_dur[basis_D]
    disrupt_start = int(basis["DISRUPT_START_T"])
    normal = m3.load_normal_trajectory(start_h)
    if not normal.get("l"):
        raise ValueError(
            f"station={station}: normal passenger trajectory is empty")

    # A zero-impact duration is still a physical branch of the scenario tree.
    scen = {}
    for D in durations:
        raw = by_dur.get(D)
        if not passenger_nonanticipativity and raw is not None:
            scenario = dict(raw)
            scenario["reported_affected_omega"] = set(raw.get("Omega_m", ()))
            scenario["physical_canceled_trains"] = set(
                raw.get("canceled_trains", ()))
            scenario["physical_cancellation_start_by_train"] = dict(
                raw.get("cancellation_start_by_train", {}))
            if "theta" not in scenario:
                theta = {}
                for vehicle, start in scenario[
                        "physical_cancellation_start_by_train"].items():
                    for minute in wdata["train_active_times"].get(
                            str(vehicle), ()):
                        if int(minute) >= int(start):
                            theta[(str(vehicle), int(minute))] = 1
                scenario["theta"] = theta
            scen[D] = scenario
        else:
            scen[D] = _physical_duration_scenario(
                station, start_h, D, raw, basis, wdata, buffer)
    if passenger_nonanticipativity:
        # Rail disruption inputs remain duration-specific and byte-for-byte
        # equivalent to their physical snapshots.  Only the passenger action
        # support is unioned; no cancellation, theta, or clearance event is
        # copied from one duration branch to another.
        _initialize_branch_local_train_withdrawals(scen, durations)

    physical_affected, causal_affected = {}, {}
    for D in durations:
        if passenger_nonanticipativity:
            physical_affected[D] = _normal_affected_ods(
                normal, scen[D]["physical_theta"])
            causal_affected[D] = set(physical_affected[D])
        else:
            # Bus-only non-anticipativity restores each raw duration branch:
            # neither its affected-OD support nor its rail mask is replaced by
            # information from the other duration.
            physical_affected[D] = set(
                scen[D].get("reported_affected_omega", ()))
            causal_affected[D] = set(physical_affected[D])

    physical_union = tuple(sorted(set().union(
        *(physical_affected[D] for D in durations))))
    if passenger_nonanticipativity and not physical_union:
        raise ValueError(
            f"station={station}: physical affected OD population is empty")

    if passenger_nonanticipativity:
        common_omegas = tuple(sorted(set().union(
            *(physical_affected[D] for D in durations))))
        common_pair_score = _pair_score_for_ods(common_omegas, wdata)
        common_station_support = _normal_station_support(
            normal, common_omegas)
        termination_stations = {
            str(event[1])
            for duration in durations
            for event in scen[duration][
                "termination_alight_event_by_train"].values()
        }
        common_station_support = {
            omega: tuple(sorted(
                set(stations).union(termination_stations)))
            for omega, stations in common_station_support.items()
        }
        for D in durations:
            scenario = scen[D]
            scenario["physical_affected_omega"] = tuple(sorted(
                physical_affected[D]))
            scenario["causal_affected_omega"] = tuple(sorted(
                causal_affected[D]))
            # A branch-unaffected common OD cannot be fixed to its complete
            # normal trajectory: its pre-revelation actions are shared with a
            # branch in which that OD may face a withdrawal.  Only the common
            # pre-onset prefix is fixed later by
            # ``_fix_normal_passenger_decisions``.
            scenario["normal_fixed_omega"] = ()
            scenario["Omega_m"] = list(common_omegas)
            scenario["pair_score"] = dict(common_pair_score)
            scenario["required_station_support"] = common_station_support
            _rebuild_common_spare_capacity(
                scenario, common_omegas, normal, wdata, m3.TRAIN_CAPACITY)
        trip_omegas = common_omegas
    else:
        common_omegas = ()
        for D in durations:
            scenario = scen[D]
            branch_omegas = tuple(sorted(physical_affected[D]))
            scenario["physical_affected_omega"] = branch_omegas
            scenario["causal_affected_omega"] = branch_omegas
            scenario["normal_fixed_omega"] = ()
            scenario["Omega_m"] = list(branch_omegas)
            scenario["pair_score"] = _pair_score_for_ods(
                branch_omegas, wdata)
            scenario.pop("required_station_support", None)
            _rebuild_common_spare_capacity(
                scenario, branch_omegas, normal, wdata, m3.TRAIN_CAPACITY)
        trip_omegas = tuple(sorted(
            physical_affected[basis_D]
            if basis_D in physical_affected
            else set(basis.get("Omega_m", ()))))
        if not trip_omegas:
            raise ValueError(
                f"station={station}: duration {basis_D} physical affected "
                "OD population is empty")

    trip_basis = dict(scen.get(basis_D, basis))
    trip_basis["Omega_m"] = list(trip_omegas)
    trip_basis["pair_score"] = _pair_score_for_ods(trip_omegas, wdata)
    if passenger_nonanticipativity:
        trip_basis["required_station_support"] = common_station_support
    else:
        trip_basis.pop("required_station_support", None)

    # (요구사항 6) 가장 넓은 배차창으로 후보 trip 을 한 번만 생성
    lib, _ = m3.build_scenario_route_library(
        trip_basis, start_h, depots, bus_tt, route_lib)
    sel = m3.select_routes(trip_basis, depots, lib, bus_tt)
    trips = m3.generate_bus_trips(trip_basis, sel, lib)

    common_support = None
    if passenger_nonanticipativity:
        support_scenario = dict(trip_basis)
        support_scenario["Omega_m"] = list(common_omegas)
        common_support = m3.build_stage2_cache(
            support_scenario, wdata, trips, support_only=True)["N_omega"]

    raw_active = tuple(D for D in durations if physical_affected[D])
    active = tuple(durations) if passenger_nonanticipativity else raw_active
    avail, cache = {}, {}
    for D in durations:
        ok = {tid: tr for tid, tr in trips.items()
              if trip_availability(tr, disrupt_start, durations, buffer)[D]}
        # Physical trip eligibility exists even when this duration has no
        # affected passenger-flow block.  Shared activations still consume
        # depot capacity in that completed branch.
        avail[D] = set(ok)
        s = scen[D]
        s["bus_trips"] = ok
        if passenger_nonanticipativity:
            cache[D] = m3.build_stage2_cache(
                s, wdata, ok, bus_stop_support=common_support)
        else:
            # Sec. 3.2.2: the pre-onset actions are fixed to the normal trajectory
            # (solve_stage2_adaptive), so N^omega keeps its normal pre-onset stations.
            support = defaultdict(set)
            for omega, stations in s.get(
                    "required_station_support", {}).items():
                support[omega].update(stations)
            for omega, stations in _load_exact_engine(
                    ).pre_onset_normal_station_support(
                        normal, s.get("Omega_m", ()), disrupt_start).items():
                support[omega].update(stations)
            s["required_station_support"] = {
                omega: tuple(sorted(stations))
                for omega, stations in support.items()}
            cache[D] = m3.build_stage2_cache(s, wdata, ok)

    raw_cells = {
        tid: info_cells(tr["DispatchStart"], disrupt_start, durations, buffer)
        for tid, tr in trips.items()
    }
    boarding_upper = defaultdict(float)
    boarding_upper_by_od = defaultdict(float)
    cumulative = wdata["CumD"]
    for D, duration_cache in cache.items():
        fixed_omegas = set(scen[D]["normal_fixed_omega"])
        for omega, vehicle, minute, _station in duration_cache.get(
                "b_var_keys", ()):
            if (str(vehicle).startswith("B_") and
                    omega not in fixed_omegas):
                upper = max(float(
                    cumulative.get((omega, int(minute)), 0.0)), 0.0)
                key = (int(D), str(vehicle))
                boarding_upper[key] += upper
                od_key = (int(D), str(vehicle), omega)
                boarding_upper_by_od[od_key] = max(
                    boarding_upper_by_od[od_key], upper)
    if EXACT_TRIP_CELL_PRUNING:
        # A passenger OD can board a particular physical trip at most once.
        # Count the largest reachable cumulative mass once per OD instead of
        # summing the same mass over every stop/minute boarding variable.
        boarding_upper = defaultdict(float)
        for (D, vehicle, _omega), upper in boarding_upper_by_od.items():
            boarding_upper[(D, vehicle)] += upper
    cells = {}
    minimum_service_infeasible_cells = 0
    tolerance = 1e-9 * max(1.0, float(m3.MIN_TRIP_BOARDING))
    for tid, trip_cells in raw_cells.items():
        retained = []
        for cell in trip_cells:
            modeled = [int(D) for D in cell if int(D) in active]
            if (modeled and all(
                    boarding_upper.get((D, str(tid)), 0.0) + tolerance >=
                    float(m3.MIN_TRIP_BOARDING) for D in modeled)):
                retained.append(tuple(cell))
            else:
                minimum_service_infeasible_cells += 1
        if retained:
            cells[str(tid)] = retained

    shared_end = (disrupt_start + min(durations)
                  if len(durations) > 1 else disrupt_start)
    rail_information_audit = {
        "single_common_event_timetable": True,
        "event_support_nonanticipative": False,
        "event_support_mismatch_count": 0,
        "termination_alight_events_verified": False,
        "termination_alight_event_count": 0,
        "termination_event_support_nonanticipative": False,
        "termination_event_support_mismatch_count": 0,
        "termination_action_support_nonanticipative": False,
        "termination_action_support_mismatch_count": 0,
        "theta_nonanticipative": False,
        "theta_mismatch_count": 0,
        "passenger_capacity_theta_present": False,
        "passenger_capacity_theta_matches_events": False,
        "passenger_capacity_theta_nonanticipative": False,
        "passenger_capacity_theta_mismatch_count": 0,
        "capacity_nonanticipative": False,
        "capacity_max_abs_diff": 0.0,
        "raw_canceled_trains_preserved": False,
        "raw_cancellation_starts_preserved": False,
        "raw_theta_preserved": False,
        "raw_termination_events_preserved": False,
        "raw_passenger_capacity_theta_preserved": False,
        "raw_branch_rail_state_preserved": False,
        "zero_impact_reconstruction_count": 0,
        "zero_impact_reconstructions_verified": False,
        "timetable_event_count": int(len(wdata.get("I_vts", ()))),
    }
    if passenger_nonanticipativity and durations:
        rail_information_audit = _rail_information_set_audit(
            scen, cache, durations, wdata, disrupt_start, shared_end)
        failed = [
            key for key in (
                "single_common_event_timetable",
                "event_support_nonanticipative",
                "termination_alight_events_verified",
                "termination_action_support_nonanticipative",
                "passenger_capacity_theta_present",
                "passenger_capacity_theta_matches_events",
                "raw_branch_rail_state_preserved",
            )
            if not rail_information_audit[key]
        ]
        if failed:
            raise ValueError(
                "raw rail preservation or common passenger action support "
                f"failed: failed={failed}, audit={rail_information_audit}")
    return {"station": station, "disrupt_start": disrupt_start,
            "durations": tuple(sorted(durations)), "active": tuple(active),
            "raw_active": raw_active,
            "buffer": buffer, "scen": scen, "trips": trips, "avail": avail,
            "cache": cache, "cells": cells, "travel_times": bus_tt,
            "normal_trajectory": normal,
            "nonanticipativity_mode": mode,
            "formulation_mode": (
                PASSENGER_AND_BUS_SHARED if passenger_nonanticipativity
                else BUS_ONLY_WAIT_AND_SEE),
            "passenger_recourse": (
                "NONANTICIPATIVE" if passenger_nonanticipativity
                else "WAIT_AND_SEE"),
            "passenger_flow_families": ("q", "l", "a", "b", "d", "o"),
            "duration_specific_physical_omega": bool(
                not passenger_nonanticipativity),
            "duration_specific_rail_state": True,
            "common_omega": common_omegas,
            "physical_affected_omega": {
                D: tuple(sorted(physical_affected[D])) for D in durations},
            "causal_affected_omega": {
                D: tuple(sorted(causal_affected[D])) for D in durations},
            "normal_fixed_omega": {
                D: tuple(scen[D].get("normal_fixed_omega", ()))
                for D in durations},
            "passenger_shared_start": (
                disrupt_start if passenger_nonanticipativity else None),
             "passenger_shared_end": (
                 shared_end if passenger_nonanticipativity else None),
            "rail_information_audit": rail_information_audit,
             "trip_basis_duration": basis_D,
            "raw_trip_cell_count": sum(len(value) for value in raw_cells.values()),
            "tight_boarding_upper_applied": bool(EXACT_TRIP_CELL_PRUNING),
            "minimum_service_infeasible_trip_cells_removed": int(
                minimum_service_infeasible_cells)}


# =============================================================================
# 3. 지속시간 하나의 승객 흐름 제약 (기존 solve_stage2_fixed 와 동일 구조)
# =============================================================================
def _passenger_key_minute(family, key):
    if family in ("b", "a"):
        return int(key[2])
    if family in ("l", "q", "w"):
        return int(key[2])
    if family == "od":
        return int(key[1])
    raise KeyError(family)


def _is_shared_passenger_key(family, key, sharing):
    if sharing is None:
        return False
    minute = _passenger_key_minute(family, key)
    return int(sharing["start"]) <= minute < int(sharing["end"])


def _add_or_share_continuous_var(
        model, key, *, family, duration, sharing, lb=0.0, ub=None,
        name=""):
    """Return one variable object for a pre-revelation decision index."""
    if not _is_shared_passenger_key(family, key, sharing):
        kwargs = {"lb": float(lb), "vtype": GRB.CONTINUOUS, "name": name}
        if ub is not None:
            kwargs["ub"] = float(ub)
        return model.addVar(**kwargs)

    registry_key = (family, key)
    entry = sharing["registry"].get(registry_key)
    normalized_ub = None if ub is None else float(ub)
    if entry is None:
        kwargs = {"lb": float(lb), "vtype": GRB.CONTINUOUS,
                  "name": f"{family}_shared"}
        if normalized_ub is not None:
            kwargs["ub"] = normalized_ub
        variable = model.addVar(**kwargs)
        entry = {
            "variable": variable, "durations": set(),
            "lb": float(lb), "ub": normalized_ub,
        }
        sharing["registry"][registry_key] = entry
    elif (abs(float(entry["lb"]) - float(lb)) > 1e-9 or
          entry["ub"] != normalized_ub):
        raise ValueError(
            f"shared passenger variable has inconsistent bounds: "
            f"{registry_key}")
    entry["durations"].add(int(duration))
    return entry["variable"]


def _batch_add_with_passenger_sharing(
        exact, model, keys, upper, *, family, duration, sharing, name):
    """Batch-create exact-model actions while reusing shared indices."""
    result, local_keys, new_shared_keys = {}, [], []
    for key in keys:
        if not _is_shared_passenger_key(family, key, sharing):
            local_keys.append(key)
            continue
        entry = sharing["registry"].get((family, key))
        if entry is None:
            new_shared_keys.append(key)
        else:
            expected = float(upper[key])
            if entry["ub"] != expected:
                raise ValueError(
                    f"shared passenger variable has inconsistent upper "
                    f"bound: {(family, key)}")
            entry["durations"].add(int(duration))
            result[key] = entry["variable"]

    if new_shared_keys:
        shared_upper = {key: float(upper[key]) for key in new_shared_keys}
        created = exact._batch_add_vars(
            model, tuple(new_shared_keys), ub=shared_upper,
            name=f"{family}_shared", ignore_names=True)
        for key, variable in created.items():
            sharing["registry"][(family, key)] = {
                "variable": variable, "durations": {int(duration)},
                "lb": 0.0, "ub": float(upper[key]),
            }
        result.update(created)
    if local_keys:
        local_upper = {key: float(upper[key]) for key in local_keys}
        result.update(exact._batch_add_vars(
            model, tuple(local_keys), ub=local_upper,
            name=name, ignore_names=True))
    return result


def _fix_normal_passenger_decisions(
        block, duration, group, *, fixed_registry):
    """Fix pre-disruption and branch-unaffected actions to normal values."""
    normal = group.get("normal_trajectory")
    if not normal:
        return 0
    fixed_omegas = set(group.get("normal_fixed_omega", {}).get(duration, ()))
    common = set(group.get("common_omega", ()))
    disruption_start = int(group["disrupt_start"])
    count = 0

    for family, variables in (("b", block[0]), ("a", block[1]),
                              ("od", block[5])):
        normal_values = normal.get(family, {})
        for key, variable in variables.items():
            omega = key[0]
            minute = _passenger_key_minute(family, key)
            if not (omega in fixed_omegas or
                    (omega in common and minute < disruption_start)):
                continue
            value = float(normal_values.get(key, 0.0))
            marker = id(variable)
            previous = fixed_registry.get(marker)
            if previous is not None and abs(previous - value) > 1e-8:
                raise ValueError(
                    f"inconsistent normal fix for shared {family} key {key}")
            variable.LB = value
            variable.UB = value
            if previous is None:
                fixed_registry[marker] = value
                count += 1

        # A positive normal action missing from the model support would make
        # "fixed to normal" only nominal.  Fail early instead of silently
        # changing that trajectory.
        if family in ("b", "a"):
            for key, value in normal_values.items():
                omega = key[0]
                minute = _passenger_key_minute(family, key)
                required = (omega in fixed_omegas or
                            (omega in common and minute < disruption_start))
                if required and float(value) > 1e-9 and key not in variables:
                    raise ValueError(
                        f"normal {family} key missing from D{duration} "
                        f"support: {key}")
    return count


def _validate_passenger_variable_sharing(sharing):
    if sharing is None:
        return {"total": 0, "by_family": {}}
    expected = set(map(int, sharing["durations"]))
    bad = [
        (key, tuple(sorted(entry["durations"])))
        for key, entry in sharing["registry"].items()
        if set(entry["durations"]) != expected
    ]
    if bad:
        raise ValueError(
            "pre-revelation passenger support differs across duration "
            f"branches; first mismatches: {bad[:5]}")
    by_family = defaultdict(int)
    for family, _key in sharing["registry"]:
        by_family[family] += 1
    return {"total": len(sharing["registry"]),
            "by_family": dict(by_family)}


def _solution_value(value):
    return float(value.X) if hasattr(value, "X") else float(value)


def _maximum_shared_difference(left, right, family, sharing, *, predicate=None):
    """Compare two minute-indexed solution maps on the sharing interval."""
    start, end = int(sharing["start"]), int(sharing["end"])
    left_keys = {
        key for key in left
        if start <= _passenger_key_minute(family, key) < end
        and (predicate is None or predicate(key))
    }
    right_keys = {
        key for key in right
        if start <= _passenger_key_minute(family, key) < end
        and (predicate is None or predicate(key))
    }
    if left_keys != right_keys:
        missing_left = sorted(right_keys - left_keys, key=repr)[:5]
        missing_right = sorted(left_keys - right_keys, key=repr)[:5]
        raise ValueError(
            f"shared-interval {family} support differs: "
            f"missing_Dfirst={missing_left}, missing_Dsecond={missing_right}")
    return max(
        (abs(_solution_value(left[key]) - _solution_value(right[key]))
         for key in left_keys),
        default=0.0,
    )


def _reconstructed_bus_loads(block):
    """Return the exact OD-minute bus loads, caching the solved projection."""
    metadata = block[8] if len(block) > 8 else None
    if not metadata or not metadata.get("exact_state_compression"):
        return {
            key: variable for key, variable in block[2].items()
            if str(key[1]).startswith("B_")
        }
    cached = metadata.get("_audited_bus_loads")
    if cached is None:
        cached = metadata["engine"]._reconstruct_hybrid_bus_loads(
            metadata["load_groups"], block[0], block[1], tolerance=1e-9)
        metadata["_audited_bus_loads"] = cached
    return cached


def _projected_walking_states(block, sharing):
    """Reconstruct code-``w`` (manuscript ``d``) from solved alightings."""
    metadata = block[8]
    plan = metadata["walking_plan"]
    start, end = int(sharing["start"]), int(sharing["end"])
    events = defaultdict(lambda: defaultdict(float))
    for (omega, _vehicle, minute, station), variable in block[1].items():
        if station == plan.omega_map.get(omega, (None, None))[1]:
            continue
        events[(omega, station)][int(minute)] += _solution_value(variable)

    states = {}
    for node, runs in plan.row_runs.items():
        by_minute = events.get(node, {})
        for run_start, run_stop in runs:
            first, stop = max(int(run_start), start), min(int(run_stop), end)
            for minute in range(first, stop):
                lower = max(int(plan.horizon_start), minute - int(plan.sigma) + 1)
                states[(node[0], node[1], minute)] = math.fsum(
                    by_minute.get(event_minute, 0.0)
                    for event_minute in range(lower, minute + 1)
                )
    return states


def _projected_exit_states(block, sharing, wdata):
    """Reconstruct code-``od`` (manuscript ``o``) from demand/alightings."""
    metadata = block[8]
    omega_map = wdata["Omega_Map"]
    demand = wdata["D"]
    sigma = int(WALKING_MINUTES)
    start, end = int(sharing["start"]), int(sharing["end"])
    destination_alight = defaultdict(float)
    for (omega, _vehicle, minute, station), variable in block[1].items():
        if omega in omega_map and station == omega_map[omega][1]:
            destination_alight[(omega, int(minute))] += _solution_value(variable)

    states = {}
    for omega in metadata["omega_set"]:
        if omega not in omega_map:
            continue
        destination = omega_map[omega][1]
        for minute in range(start, end):
            states[(omega, minute)] = (
                float(demand.get((omega, destination, minute), 0.0))
                + destination_alight.get((omega, minute - sigma), 0.0)
            )
    return states


def _passenger_state_sharing_audit(blocks, durations, sharing, wdata):
    """Verify literal actions and every reconstructed passenger-flow family."""
    if sharing is None or len(durations) != 2:
        return {"l": 0.0, "train_l": 0.0, "bus_l": 0.0,
                "q": 0.0, "d": 0.0, "o": 0.0}
    first, second = map(int, sorted(durations))
    left, right = blocks[first], blocks[second]
    result = {
        "q": _maximum_shared_difference(
            left[3], right[3], "q", sharing),
        "train_l": _maximum_shared_difference(
            left[2], right[2], "l", sharing,
            predicate=lambda key: not str(key[1]).startswith("B_")),
        "bus_l": _maximum_shared_difference(
            _reconstructed_bus_loads(left), _reconstructed_bus_loads(right),
            "l", sharing),
    }
    result["l"] = max(result["train_l"], result["bus_l"])

    exact = all(
        len(block) > 8 and block[8].get("exact_state_compression")
        for block in (left, right)
    )
    if exact:
        result["d"] = _maximum_shared_difference(
            _projected_walking_states(left, sharing),
            _projected_walking_states(right, sharing), "w", sharing)
        result["o"] = _maximum_shared_difference(
            _projected_exit_states(left, sharing, wdata),
            _projected_exit_states(right, sharing, wdata), "od", sharing)
    else:
        result["d"] = _maximum_shared_difference(
            left[4], right[4], "w", sharing)
        result["o"] = _maximum_shared_difference(
            left[5], right[5], "od", sharing)
    return result


def _add_flow_block_dense(
        m3, model, D, scenario, cache, wdata, y_of, tag,
        passenger_sharing=None):
    """
    지속시간 D 의 흐름변수와 제약을 모델에 추가한다.
    y_of(tid) : 그 지속시간에서 이 trip 의 배차변수 (없으면 None)
    반환 (b, a, l, q, w, od, b_by_trip, Z_expr)
    """
    Omega_m = scenario["Omega_m"]
    spare_cap = scenario["spare_capacity"]
    Omega_Map = wdata["Omega_Map"]
    Gamma = wdata["Gamma"]
    START_MIN = wdata["START_MIN"]
    station_of = wdata["station_of"]
    train_active_times = wdata["train_active_times"]
    train_dest_map = wdata["train_dest_map"]
    train_dwells = wdata["train_dwells"]
    Vm = wdata["Vm"]
    CumD = wdata["CumD"]
    Dem = wdata["D"]
    sigma = WALKING_MINUTES
    termination_event_keys = _termination_alight_support_keys(scenario)
    termination_by_train = {
        vehicle: (minute, station)
        for vehicle, minute, station in termination_event_keys
    }

    b, a, l, q, w, od = {}, {}, {}, {}, {}, {}
    for (om, vid, t, j) in cache["b_var_keys"]:
        key = (om, str(vid), int(t), j)
        ub = CumD.get((om, t), 0.0)
        b[key] = _add_or_share_continuous_var(
            model, key, family="b", duration=D,
            sharing=passenger_sharing, lb=0.0, ub=max(ub, 0.0),
            name=f"b{tag}")
    for (om, vid, t, j) in cache["a_var_keys"]:
        key = (om, str(vid), int(t), j)
        a[key] = _add_or_share_continuous_var(
            model, key, family="a", duration=D,
            sharing=passenger_sharing, lb=0.0,
            ub=CumD.get((om, t), 0.0), name=f"a{tag}")
    for (om, vid, t) in cache["l_var_keys"]:
        cap_v = m3.BUS_CAPACITY if vid.startswith("B_") else m3.TRAIN_CAPACITY
        l[(om, vid, t)] = model.addVar(lb=0, ub=cap_v, vtype=GRB.CONTINUOUS,
                                       name=f"l{tag}")
    bus_load_groups = _aggregate_bus_load_groups(cache)
    # Ensure a continuous state timeline even when coverage intervals are
    # disjoint and the legacy cache omits the intervening minutes.
    for (tid, om), group in bus_load_groups.items():
        for t in range(group["first"], group["last"] + 1):
            key = (om, tid, t)
            if key not in l:
                l[key] = model.addVar(
                    lb=0, ub=m3.BUS_CAPACITY, vtype=GRB.CONTINUOUS,
                    name=f"l{tag}")
    for (om, j, t) in cache["q_var_keys"]:
        q[(om, j, t)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"q{tag}")
    for (om, j, t) in cache["w_var_keys"]:
        key = (om, j, int(t))
        w[key] = _add_or_share_continuous_var(
            model, key, family="w", duration=D,
            sharing=passenger_sharing, lb=0.0, name=f"w{tag}")
    for (om, t) in cache["od_var_keys"]:
        key = (om, int(t))
        od[key] = _add_or_share_continuous_var(
            model, key, family="od", duration=D,
            sharing=passenger_sharing, lb=0.0,
            ub=CumD.get((om, t), 0.0), name=f"od{tag}")
    model.update()

    # --- Load balance (버스) ---
    for (tid, om), group in bus_load_groups.items():
        previous = 0
        for t in range(group["first"], group["last"] + 1):
            board = gp.quicksum(
                b[key] for key in group["board"].get(t, ()) if key in b)
            alight = gp.quicksum(
                a[key] for key in group["alight"].get(t, ()) if key in a)
            model.addConstr(l[(om, tid, t)] == previous + board - alight)
            previous = l[(om, tid, t)]

    # Keep the common rail boarding index in every duration cache, then impose
    # the raw branch's service-withdrawal mask as a branch row.  Removing the
    # variable in only one branch would break literal passenger sharing.
    service_theta = scenario.get("theta", {})
    for (omega, vehicle, minute, station), variable in b.items():
        if (not str(vehicle).startswith("B_") and int(service_theta.get(
                (str(vehicle), int(minute)), 0))):
            model.addConstr(variable == 0.0)

    # --- Load balance (열차) ---
    for om in Omega_m:
        if om not in Omega_Map:
            continue
        for v in Vm:
            if v not in train_active_times:
                continue
            for t in train_active_times[v]:
                lkey = (om, v, t)
                if lkey not in l:
                    continue
                s_t = station_of[v].get(t)
                board_key, alight_keys = _dense_train_action_keys(
                    om, v, t, s_t, termination_by_train)
                board = b.get(board_key, 0) if board_key is not None else 0
                alight = gp.quicksum(
                    a[key] for key in alight_keys if key in a)
                if t == train_active_times[v].start:
                    model.addConstr(l[lkey] == board - alight)
                else:
                    prev = l.get((om, v, t - 1))
                    if prev is not None:
                        model.addConstr(l[lkey] == prev + board - alight)

    # --- Bus capacity : 같은 지속시간의 배차변수를 참조 ---
    l_by_trip_t = defaultdict(list)
    for (om, tid, t), lvar in l.items():
        if tid.startswith("B_"):
            l_by_trip_t[(tid, t)].append(lvar)
    for (tid, t), lvars in l_by_trip_t.items():
        yv = y_of(tid)
        if yv is None:                       # 이 지속시간에서 못 쓰는 trip
            for lv in lvars:
                model.addConstr(lv == 0)
        else:
            model.addConstr(gp.quicksum(lvars) <= m3.BUS_CAPACITY * yv)

    # --- Train capacity ---
    l_by_train_t = defaultdict(list)
    for (om, vid, t), lvar in l.items():
        if not vid.startswith("B_"):
            l_by_train_t[(vid, t)].append(lvar)
    for (vid, t), lvars in l_by_train_t.items():
        model.addConstr(gp.quicksum(lvars) <= spare_cap.get((vid, t), 0.0))

    # --- Queue flow ---
    for om in Omega_m:
        if om not in Omega_Map:
            continue
        orig, dest = Omega_Map[om]
        for j in cache["N_omega"].get(om, set()):
            for ti, t in enumerate(Gamma):
                qvar = q.get((om, j, t))
                if qvar is None:
                    continue
                d_jt = Dem.get((om, j, t), 0.0)
                board_sum = gp.quicksum(
                    b[(om, vid, t, j)]
                    for vid in cache["boarding_by_od_t_j"].get((om, t, j), [])
                    if (om, vid, t, j) in b)
                alight_sum = gp.LinExpr()
                if t >= START_MIN + sigma:
                    for vid in cache["alighting_by_od_t_j"].get((om, t - sigma, j), []):
                        akey = (om, vid, t - sigma, j)
                        if akey in a:
                            alight_sum += a[akey]
                q_prev = q.get((om, j, t - 1), 0) if ti > 0 else 0
                if j == dest:
                    model.addConstr(qvar == q_prev + d_jt + alight_sum
                                    - board_sum - od.get((om, t), 0))
                else:
                    model.addConstr(qvar == q_prev + d_jt + alight_sum - board_sum)

    # --- Walking flow ---
    for om in Omega_m:
        if om not in Omega_Map:
            continue
        orig, dest = Omega_Map[om]
        for j in cache["N_omega"].get(om, set()):
            if j == dest:
                continue
            for ti, t in enumerate(Gamma):
                wvar = w.get((om, j, t))
                if wvar is None:
                    continue
                alight_in = gp.quicksum(
                    a[(om, vid, t, j)]
                    for vid in cache["alighting_by_od_t_j"].get((om, t, j), [])
                    if (om, vid, t, j) in a)
                alight_out = gp.LinExpr()
                if t >= START_MIN + sigma:
                    for vid in cache["alighting_by_od_t_j"].get((om, t - sigma, j), []):
                        akey = (om, vid, t - sigma, j)
                        if akey in a:
                            alight_out += a[akey]
                w_prev = w.get((om, j, t - 1), 0) if ti > 0 else 0
                model.addConstr(wvar == w_prev + alight_in - alight_out)

    # --- 탑승 논리 ---
    for om in Omega_m:
        if om not in Omega_Map:
            continue
        orig, dest = Omega_Map[om]
        for (om2, vid, t, j), bvar in b.items():
            if om2 == om and j == dest:
                model.addConstr(bvar == 0)
        for (om2, vid, t, j), avar in a.items():
            if om2 == om and _forbidden_origin_alight(
                    (om2, vid, t, j), Omega_Map,
                    termination_event_keys):
                model.addConstr(avar == 0)
        for v in Vm:
            if v not in train_active_times:
                continue
            for dwell in train_dwells[v]:
                s_d, times = dwell["station"], dwell["times"]
                inc_load = l.get((om, v, times[0] - 1), 0)
                alight_sum = gp.quicksum(
                    a[(om, v, t, s_d)] for t in times if (om, v, t, s_d) in a)
                if alight_sum.size() > 0:
                    model.addConstr(alight_sum <= inc_load)
            if v in train_dest_map:
                t_last = train_active_times[v].stop - 1
                if (om, v, t_last) in l:
                    model.addConstr(l[(om, v, t_last)] == 0)

    # Every bus passenger stream must be empty at the end of its retained
    # timeline; otherwise load could disappear just beyond the model support.
    for (tid, om), group in bus_load_groups.items():
        model.addConstr(l[(om, tid, group["last"])] == 0)

    # --- 최소 서비스 ---
    b_by_trip = defaultdict(list)
    for (om, tid, t, j), bvar in b.items():
        if tid.startswith("B_"):
            b_by_trip[tid].append(bvar)
    for tid in cache["trip_ids"]:
        yv = y_of(tid)
        bvars = b_by_trip.get(tid, [])
        if yv is not None:
            model.addConstr(gp.quicksum(bvars) >= m3.MIN_TRIP_BOARDING * yv)

    # Eq. (2) [eq:inventory] and Eq. (13) [eq:obj_db]: minute states t >= t0, 1/(T - t0).
    exact = _load_exact_engine()
    Z = exact.eq13_objective(*exact.eq13_inventories(
        int(START_MIN) + int(exact.OBJECTIVE_ONSET_MIN), q=q,
        queue_compression=None, l=l, l_states={}, load_segments=(),
        compressed_loads=False, hybrid_bus_states={}, a=a, walking_plan=None,
        w=w))
    return b, a, l, q, w, od, b_by_trip, Z


def _add_flow_block_exact(
        m3, model, D, scenario, cache, wdata, y_of, tag,
        passenger_sharing=None):
    """Add the lossless event-compressed equivalent of one duration block."""
    exact = _load_exact_engine()
    omega_set = tuple(scenario["Omega_m"])
    spare_capacity = scenario["spare_capacity"]
    omega_map = wdata["Omega_Map"]
    gamma = tuple(int(value) for value in wdata["Gamma"])
    start_min = int(wdata["START_MIN"])
    station_of = wdata["station_of"]
    train_active_times = wdata["train_active_times"]
    train_dest_map = wdata["train_dest_map"]
    train_dwells = wdata["train_dwells"]
    trains = tuple(wdata["Vm"])
    cumulative = wdata["CumD"]
    demand = wdata["D"]
    onset = start_min + int(exact.OBJECTIVE_ONSET_MIN)     # t0 of Eq. (13)
    sigma = int(WALKING_MINUTES)
    termination_event_keys = _termination_alight_support_keys(scenario)

    active_bus_ids = {
        str(tid) for tid in cache.get("trip_ids", ())
        if y_of(str(tid)) is not None
    }
    b_keys = tuple(dict.fromkeys(
        (om, str(vehicle), int(minute), station)
        for om, vehicle, minute, station in cache.get("b_var_keys", ())
        if not (str(vehicle).startswith("B_")
                and str(vehicle) not in active_bus_ids)
    ))
    a_keys = tuple(dict.fromkeys(
        (om, str(vehicle), int(minute), station)
        for om, vehicle, minute, station in cache.get("a_var_keys", ())
        if not (str(vehicle).startswith("B_")
                and str(vehicle) not in active_bus_ids)
    ))
    # Eq. (8) [eq:dem_ab_n]: b, a <= cumulative OD demand sum_{t'<=t} D_{omega,s(omega),t'}.
    # Eq. (10) [eq:nn_n]: b, a, l, q >= 0 (lb = 0 of every variable below).
    b_upper = {
        key: max(float(cumulative.get((key[0], key[2]), 0.0)), 0.0)
        for key in b_keys
    }
    a_upper = {
        key: max(float(cumulative.get((key[0], key[2]), 0.0)), 0.0)
        for key in a_keys
    }
    b = _batch_add_with_passenger_sharing(
        exact, model, b_keys, b_upper, family="b", duration=D,
        sharing=passenger_sharing, name=f"b{tag}")
    a = _batch_add_with_passenger_sharing(
        exact, model, a_keys, a_upper, family="a", duration=D,
        sharing=passenger_sharing, name=f"a{tag}")

    # Eq. (12) [eq:spare] (theta_(v,t) = 1): no boarding onto a withdrawn rail service.
    service_theta = scenario.get("theta", {})
    for (omega, vehicle, minute, station), variable in b.items():
        if (not str(vehicle).startswith("B_") and int(service_theta.get(
                (str(vehicle), int(minute)), 0))):
            model.addConstr(variable == 0.0)

    config = exact.CGConfig(
        sigma=sigma,
        bus_capacity=float(m3.BUS_CAPACITY),
        train_capacity=float(m3.TRAIN_CAPACITY),
        min_trip_boarding=float(m3.MIN_TRIP_BOARDING),
        ignore_names=True,
        eliminate_w=True,
        eliminate_destination_states=True,
        compress_load_states=True,
        compress_queue_states=True,
        hybrid_bus_load_states=True,
    )

    # Train inventories retain one variable per b/a event interval.  Bus OD
    # inventories are projected into one aggregate event ledger per trip plus
    # exact OD-prefix feasibility rows, matching the Stage-1 reformulation.
    state_support = cache.get("_exact_stage2_state_support")
    if state_support is None:
        train_l_keys = tuple(dict.fromkeys(
            (om, str(vehicle), int(minute))
            for om, vehicle, minute in cache.get("l_var_keys", ())
            if not str(vehicle).startswith("B_")
        ))
        original_train_l_states = len(train_l_keys)
        load_segments, load_board, load_alight = exact._compressed_load_plan(
            train_l_keys, b_keys, a_keys, {}, station_of)
        del train_l_keys
    else:
        original_train_l_states = state_support[
            "original_train_l_states"]
        load_segments = state_support["load_segments"]
        load_board = state_support["load_board"]
        load_alight = state_support["load_alight"]
    # Eq. (6) [eq:l_n] (rail v in V_r): l_(v,t) = l_(v,t-1) + b - a, compressed to
    # one state per board/alight event interval (Sec. 3.4.4, exact reduction).
    # Eq. (7) [eq:cap_n] with residual capacity S_(v,t) (Eq. (12)): load_cap rows.
    upper_by_vehicle = {
        segment.stream_key[1]: float(m3.TRAIN_CAPACITY)
        for segment in load_segments
    }
    l_states, l = exact._add_compressed_load_states(
        model, b, a, load_segments, load_board, load_alight,
        ub_by_vehicle=upper_by_vehicle,
        name=f"l{tag}", ignore_names=True)
    exact._add_interval_load_capacities(
        model, l_states, load_segments, y={},
        spare_capacity=spare_capacity, config=config,
        name=f"load_cap{tag}", ignore_names=True)

    # Eq. (6) for buses v in V_rho and Eq. (37) [eq:cap_bus]:
    # sum_omega l_(v,t) <= C_v y_v (hybrid aggregate bus load ledger).
    load_groups = exact.aggregate_bus_load_groups(cache, active_bus_ids)
    hybrid_bus_states = {}
    for tid, by_omega in load_groups.items():
        y_variable = y_of(tid)
        if y_variable is None:
            continue
        plan = exact._build_bus_load_plan(tid, by_omega)
        bus_b = {key: value for key, value in b.items() if key[1] == tid}
        bus_a = {key: value for key, value in a.items() if key[1] == tid}
        hybrid_bus_states[tid] = exact._add_hybrid_bus_load_states(
            model, bus_b, bus_a, plan.board, plan.alight,
            tid=tid, y=y_variable, capacity=float(m3.BUS_CAPACITY),
            name=f"L{tag}_{tid}", ignore_names=True)

    # Destination queue and exit states are definitional and vanish.  Every
    # other queue stream is constant between demand/boarding/walk-arrival
    # events, so one state and one balance row represent the full interval.
    if state_support is None:
        original_total_q_states = len(cache.get("q_var_keys", ()))
        q_reference_keys = tuple(dict.fromkeys(
            (om, station, int(minute))
            for om, station, minute in cache.get("q_var_keys", ())
            if station != omega_map.get(om, (None, None))[1]
        ))
        original_q_states = len(q_reference_keys)
        queue_compression = exact.build_queue_compression(
            q_reference_keys, b_keys, a_keys, demand, omega_map, sigma)
        del q_reference_keys
    else:
        original_total_q_states = state_support[
            "original_total_q_states"]
        original_q_states = state_support["original_q_states"]
        queue_compression = state_support["queue_compression"]
    q_states = exact._batch_add_vars(
        model, queue_compression.model_keys,
        name=f"q{tag}", ignore_names=True)
    q = exact._expanded_queue_view(q_states, queue_compression)
    if state_support is None:
        walking_plan = exact.build_walking_elimination_plan(
            w_rows=cache.get("w_var_keys", ()),
            alighting_keys=a_keys,
            omega_map=omega_map,
            gamma=gamma,
            sigma=sigma,
        )
        eliminated_w_states = len(cache.get("w_var_keys", ()))
        eliminated_od_states = len(cache.get("od_var_keys", ()))
        state_support = {
            "original_train_l_states": original_train_l_states,
            "load_segments": load_segments,
            "load_board": load_board,
            "load_alight": load_alight,
            "original_total_q_states": original_total_q_states,
            "original_q_states": original_q_states,
            "queue_compression": queue_compression,
            "walking_plan": walking_plan,
            "eliminated_w_states": eliminated_w_states,
            "eliminated_od_states": eliminated_od_states,
        }
        cache["_exact_stage2_state_support"] = state_support
        cache["l_var_keys"] = ()
        cache["q_var_keys"] = ()
        cache["w_var_keys"] = ()
        cache["od_var_keys"] = ()
    else:
        walking_plan = state_support["walking_plan"]
        eliminated_w_states = state_support["eliminated_w_states"]
        eliminated_od_states = state_support["eliminated_od_states"]
    w, od = {}, {}

    # Eq. (5) [eq:q_nd_n]: queue balance q_(j,t) = q_(j,t-1) + D + transfers
    # arriving after sigma + a - b at non-destination platforms; the walking
    # states d and the destination queue of Eq. (4) [eq:q_dest_n] are
    # eliminated by substitution (Sec. 3.4.4).
    boarding_index = cache["boarding_by_od_t_j"]
    alighting_index = cache["alighting_by_od_t_j"]
    for segment in queue_compression.iter_segments():
        om, station, minute = (
            segment.omega, segment.station, int(segment.start))
        board = gp.quicksum(
            b[(om, str(vehicle), minute, station)]
            for vehicle in boarding_index.get((om, minute, station), ())
            if (om, str(vehicle), minute, station) in b)
        alight = gp.LinExpr()
        if minute >= start_min + sigma:
            event_minute = minute - sigma
            for vehicle in alighting_index.get(
                    (om, event_minute, station), ()):
                key = (om, str(vehicle), event_minute, station)
                if key in a:
                    alight += a[key]
        previous = q_states.get(segment.predecessor, 0)
        model.addConstr(
            q_states[segment.key] == previous
            + float(demand.get((om, station, minute), 0.0))
            + alight - board)

    # Sec. 3.4.4 / Sec. 3.1: no boarding at e(omega), no alighting at s(omega).
    for (om, vehicle, minute, station), variable in b.items():
        if om in omega_map and station == omega_map[om][1]:
            model.addConstr(variable == 0.0)
    for (om, vehicle, minute, station), variable in a.items():
        if _forbidden_origin_alight(
                (om, vehicle, minute, station), omega_map,
                termination_event_keys):
            model.addConstr(variable == 0.0)

    # Train alighting feasibility and terminal emptiness remain unchanged;
    # the lazy expanded load view maps each minute to its interval state.
    # code (not in paper): alighting at a dwell <= incoming load, and every
    # train is empty at its terminal (implied boundary conditions of Eq. (6)).
    terminal_loads = defaultdict(list)
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
                alight_vars = [
                    a[(om, str(train), int(minute), station)]
                    for minute in times
                    if (om, str(train), int(minute), station) in a
                ]
                if alight_vars:
                    model.addConstr(gp.quicksum(alight_vars) <= incoming)
            if train in train_dest_map:
                last = int(train_active_times[train].stop - 1)
                variable = l.get((om, str(train), last))
                if variable is not None:
                    terminal_loads[str(train)].append(variable)
    exact._add_vehicle_terminal_empty_rows(
        model, terminal_loads,
        name=f"vehicle_empty{tag}", ignore_names=True)

    b_by_trip = defaultdict(list)
    for (_om, vehicle, _minute, _station), variable in b.items():
        if vehicle.startswith("B_"):
            b_by_trip[vehicle].append(variable)
    # Eq. (38) [eq:min_svc]: sum b_(v,t,j) >= epsilon y_v for every bus trip.
    for tid in active_bus_ids:
        model.addConstr(
            gp.quicksum(b_by_trip.get(tid, ()))
            >= float(m3.MIN_TRIP_BOARDING) * y_of(tid))

    # Eq. (2) [eq:inventory] and Eq. (13) [eq:obj_db]:
    # Z = 1/(T - t0) sum_{omega} sum_{t >= t0} z_t, z_t = kappa [queue + walking]
    # + onboard load (Table 3: kappa); the walking term is evaluated from the
    # alighting flows (walking_plan).  Every compressed state carries its number of
    # represented minutes t >= t0 (exact.eq13_inventories).
    queue_inventory, load_inventory, walking_inventory = exact.eq13_inventories(
        onset, q=q_states, queue_compression=queue_compression, l={},
        l_states=l_states, load_segments=load_segments, compressed_loads=True,
        hybrid_bus_states=hybrid_bus_states, a=a, walking_plan=walking_plan,
        w={})
    objective = exact.eq13_objective(
        queue_inventory, load_inventory, walking_inventory)
    metadata = {
        "exact_state_compression": True,
        "engine": exact,
        "omega_set": omega_set,
        "q_states": q_states,
        "queue_compression": queue_compression,
        "l_states": l_states,
        "load_segments": load_segments,
        "load_groups": load_groups,
        "hybrid_bus_states": hybrid_bus_states,
        "walking_plan": walking_plan,
        "queue_inventory": queue_inventory,
        "load_inventory": load_inventory,
        "walking_inventory": walking_inventory,
        "original_q_states": original_q_states,
        "retained_q_states": len(q_states),
        "original_l_states": original_train_l_states + sum(
            len(group["times"])
            for by_omega in load_groups.values()
            for group in by_omega.values()),
        "retained_l_states": len(l_states) + sum(
            len(states) for states in hybrid_bus_states.values()),
        "eliminated_w_states": eliminated_w_states,
        "eliminated_destination_q_states": (
            original_total_q_states - original_q_states),
        "eliminated_od_states": eliminated_od_states,
    }
    return b, a, l, q, w, od, b_by_trip, objective, metadata


def _add_flow_block(
        m3, model, D, scenario, cache, wdata, y_of, tag,
        passenger_sharing=None):
    if EXACT_STATE_COMPRESSION:
        return _add_flow_block_exact(
            m3, model, D, scenario, cache, wdata, y_of, tag,
            passenger_sharing=passenger_sharing)
    return _add_flow_block_dense(
        m3, model, D, scenario, cache, wdata, y_of, tag,
        passenger_sharing=passenger_sharing)


def reweight_flow_block(
        block, cache, queue_multiplier=1.0, walking_multiplier=1.0):
    """Return a block with its inventory objective reweighted exactly.

    The multipliers replace kappa of Eq. (2) for queue and walking; the
    inventories are the Eq. (13) sums over t >= t0, normalised by T - t0.
    """
    exact = _load_exact_engine()
    values = list(block)
    metadata = values[8] if len(values) > 8 else None
    if metadata and metadata.get("exact_state_compression"):
        queue_inventory = metadata["queue_inventory"]
        load_inventory = metadata["load_inventory"]
        walking_inventory = metadata["walking_inventory"]
    else:
        queue_inventory, load_inventory, walking_inventory = (
            exact.eq13_inventories(
                int(cache["Gamma"][0]) + int(exact.OBJECTIVE_ONSET_MIN),
                q=values[3], queue_compression=None, l=values[2],
                l_states={}, load_segments=(), compressed_loads=False,
                hybrid_bus_states={}, a=values[1], walking_plan=None,
                w=values[4]))
    values[7] = (
        float(queue_multiplier) * queue_inventory
        + load_inventory
        + float(walking_multiplier) * walking_inventory
    ) / exact.OBJECTIVE_NORMALISER
    return tuple(values)


def _add_nonanticipative_vehicle_chains(
        model, group, depots, depot_counts, yvar, y_index):
    """Add information-set-shared first-duty and inter-trip decisions."""
    durations = tuple(sorted(int(D) for D in group["durations"]))
    disruption_start = int(group["disrupt_start"])
    buffer = int(group.get("buffer", DISPATCH_BUFFER))
    travel_times = group["travel_times"]
    trips = {str(tid): trip for tid, trip in group["trips"].items()}
    active_tids = {str(tid) for tid, _cell in yvar}

    physical_successors = _compatible_trip_arcs(
        {tid: trips[tid] for tid in active_tids}, travel_times)
    for tid in active_tids:
        trip = trips[tid]
        depot = str(trip["Depot"])
        if depot not in depot_counts:
            raise ValueError(f"trip {tid} uses unknown depot {depot}")
        outbound = _travel_minutes(
            travel_times, depot, _trip_start_station(trip))
        returned = _travel_minutes(
            travel_times, _trip_end_station(trip), depot)
        if outbound is None or returned is None:
            raise ValueError(
                f"trip {tid} cannot be connected to home depot {depot}")
        if int(trip["DispatchStart"]) + outbound > _trip_run_start(trip):
            raise ValueError(f"trip {tid} has infeasible recorded pull-out time")

    by_duration = {
        D: {"first": {}, "link": {}, "first_meta": {}, "link_meta": {}}
        for D in durations
    }
    first_vars = {}
    first_decisions = {}
    link_vars = {}
    zero_fixed_links = 0
    first_substituted = 0
    constraint_count = 0

    # Eq. (24) [eq:chain_set]: Psi = physically compatible trip pairs (same depot,
    # successor starts after predecessor end + deadhead); built by _compatible_trip_arcs.
    # Eq. (33) [eq:na_chain] / Eq. (26) [eq:shared_chain_branches]: psi_(v,v') is one
    # variable per information cell P_(v,v') (inter-trip latest departure), i.e.
    # non-anticipativity by variable sharing (Sec. 3.4.4).
    # Eq. (40) [eq:domain_psi]: psi binary.
    for predecessor, next_ids in physical_successors.items():
        for successor in next_ids:
            decision_time = _chain_arc_decision_time(
                trips[predecessor], trips[successor], travel_times)
            for raw_cell in info_cells(
                    decision_time, disruption_start, durations, buffer):
                cell = tuple(int(D) for D in raw_cell)
                supported = [
                    D for D in cell
                    if (predecessor, D) in y_index and (successor, D) in y_index
                ]
                if not supported:
                    continue
                unsupported = set(cell) - set(supported)
                if unsupported:
                    # Exact fixed-zero arc: an indistinguishable branch lacks
                    # an endpoint, so the shared early action cannot occur.
                    zero_fixed_links += 1
                    continue
                psi = model.addVar(
                    lb=0.0, ub=1.0, vtype=GRB.BINARY,
                    name=(f"psi[{predecessor},{successor}|"
                          f"{'-'.join(map(str, cell))}]"))
                link = psi
                link_vars[(predecessor, successor, cell)] = link
                for D in supported:
                    arc = (predecessor, successor)
                    if arc in by_duration[D]["link"]:
                        raise ValueError(
                            f"duration {D}, arc {arc} belongs to two cells")
                    by_duration[D]["link"][arc] = link
                    by_duration[D]["link_meta"][arc] = {
                        "decision_time": int(decision_time),
                        "information_cell": cell,
                    }

    # Eq. (32) [eq:na_duty]: u_v uses y_v's information cell P_v (Eq. (25)).
    # Eq. (39) [eq:domain_yu]: u binary.  Sec. 3.4.4: when no potentially nonzero
    # incoming psi reaches the trip in any branch of that cell, Eq. (34) gives
    # u_v = y_v identically and u_v is substituted before model construction.
    for (raw_tid, raw_cell), y_variable in yvar.items():
        tid = str(raw_tid)
        cell = tuple(int(D) for D in raw_cell)
        has_incoming = False
        for D in cell:
            if (tid, D) not in y_index:
                raise ValueError(f"duration {D} has no activation for trip {tid}")
            has_incoming = has_incoming or any(
                successor == tid
                for _predecessor, successor in by_duration[D]["link"])
        if has_incoming:
            u = model.addVar(
                lb=0.0, ub=1.0, vtype=GRB.BINARY,
                name=f"u[{tid}|{'-'.join(map(str, cell))}]")
            first = u
            first_vars[(tid, cell)] = first
        else:
            first = y_variable
            first_substituted += 1
        first_decisions[(tid, cell)] = first
        for D in cell:
            if tid in by_duration[D]["first"]:
                raise ValueError(f"duration {D}, trip {tid} belongs to two cells")
            by_duration[D]["first"][tid] = first
            by_duration[D]["first_meta"][tid] = {
                "decision_time": int(trips[tid]["DispatchStart"]),
                "information_cell": cell,
            }

    for D in durations:
        y_D = {tid: variable for (tid, duration), variable in y_index.items()
               if int(duration) == D}
        first = by_duration[D]["first"]
        links = by_duration[D]["link"]
        incoming = defaultdict(list)
        outgoing = defaultdict(list)
        for (predecessor, successor), variable in links.items():
            incoming[successor].append(variable)
            outgoing[predecessor].append(variable)
        for tid, y_variable in y_D.items():
            if tid not in first:
                raise ValueError(
                    f"duration {D} has no first-duty variable for trip {tid}")
            incoming_vars = incoming.get(tid, ())
            outgoing_vars = outgoing.get(tid, ())
            # Eq. (34) [eq:chain]: sum_{v'} psi_(v',v) + u_v = y_v
            # (omitted when u_v = y_v was substituted and no psi enters v).
            if first[tid] is not y_variable or incoming_vars:
                model.addConstr(
                    first[tid] + gp.quicksum(incoming_vars) == y_variable,
                    name=f"eq34_chain[{tid},{D}]")
                constraint_count += 1
            # Eq. (35) [eq:chain_successor]: sum_{v'} psi_(v,v') <= y_v.
            if outgoing_vars:
                model.addConstr(
                    gp.quicksum(outgoing_vars) <= y_variable,
                    name=f"eq35_chain_successor[{tid},{D}]")
                constraint_count += 1
        by_depot = defaultdict(list)
        for tid, variable in first.items():
            by_depot[str(trips[tid]["Depot"])].append(variable)
        # Eq. (36) [eq:depot_vehicles]: sum_{v: g_v = g} u_v <= x*_g (Stage 1
        # positioning fixed in Stage 2, Sec. 3.4.3).
        for depot in depots:
            variables = by_depot.get(str(depot), ())
            if variables:
                model.addConstr(
                    gp.quicksum(variables) <= int(depot_counts.get(depot, 0)),
                    name=f"eq36_depot_vehicles[{depot},{D}]")
                constraint_count += 1

    return {
        "by_duration": by_duration,
        "first_vars": first_vars,
        "first_decisions": first_decisions,
        "link_vars": link_vars,
        "physical_arc_count": sum(len(value)
                                  for value in physical_successors.values()),
        "first_var_count": len(first_vars),
        "first_decision_count": len(first_decisions),
        "first_substituted_count": first_substituted,
        "link_var_count": len(link_vars),
        "constraint_count": constraint_count,
        "zero_fixed_link_count": zero_fixed_links,
        "decision_scope": "arc_latest_departure_information_cell",
        "binary_auxiliaries": True,
    }


def _export_model(model, prefix, status):
    """Write <prefix>.mps (with names), <prefix>.sol and a JSON sidecar.

    Called after optimize(); it reads attributes only and does not change the
    model or the solve.
    """
    import json

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    def attribute(name):
        try:
            value = model.getAttr(name)
        except gp.GurobiError:
            return None
        return None if isinstance(value, float) and not math.isfinite(value) else value

    solution_count = int(attribute("SolCount") or 0)
    # Gurobi cannot open non-ASCII paths on Windows: write into a temporary
    # folder, then store the file bzip2-compressed (Gurobi reads *.mps.bz2 and
    # *.sol.bz2 directly).
    import bz2
    import shutil
    import tempfile
    suffixes = (".mps", ".sol") if solution_count > 0 else (".mps",)
    names = {suffix: f"{prefix.name}{suffix}.bz2" for suffix in suffixes}
    with tempfile.TemporaryDirectory() as scratch:
        for suffix in suffixes:
            temporary = Path(scratch) / f"model{suffix}"
            model.write(str(temporary))
            with temporary.open("rb") as source, bz2.open(
                    prefix.parent / names[suffix], "wb") as target:
                shutil.copyfileobj(source, target)
    parameters = {}
    for name in ("TimeLimit", "MIPGap", "MIPFocus", "Heuristics", "Cuts",
                 "Threads", "Seed"):
        parameters[name] = model.getParamInfo(name)[2]
    sidecar = {
        "model": prefix.name,
        "status": status,
        "objective": attribute("ObjVal") if solution_count else None,
        "bound": attribute("ObjBound"),
        "gap": attribute("MIPGap") if solution_count else None,
        "runtime_seconds": attribute("Runtime"),
        "solution_count": solution_count,
        "num_vars": attribute("NumVars"),
        "num_constrs": attribute("NumConstrs"),
        "num_nonzeros": attribute("NumNZs"),
        "num_binary_vars": attribute("NumBinVars"),
        "fingerprint": attribute("Fingerprint"),
        "gurobi_version": ".".join(map(str, gp.gurobi.version())),
        "parameters": parameters,
        "files": {"mps": names[".mps"], "sol": names.get(".sol")},
    }
    prefix.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")


# =============================================================================
# 4. 적응형 Stage 2
# =============================================================================
def solve_stage2_adaptive(m3, env, depots, depot_counts, group, wdata,
                          duration_weight=None, mip_gap=None, time_limit=None,
                          fixed_tids=None, output_flag=0, threads=None,
                          mip_start=None, mip_focus=1, heuristics=0.5,
                          cuts=-1, nonanticipativity_mode=None,
                          export_prefix=None):
    """
    한 사고 위치에 대해 30/60/90 을 동시에 담은 적응형 배차 문제를 푼다.

    목적:  min  Σ_D  P(D) · Z_D
    배차:  y[(tid, cell)]  — cell 은 비예견성 정보집합
    흐름:  지속시간별 복제

    Paper (Sec. 3.4.3): objective Eq. (41) [eq:s2_eval] subject to
    Eqs. (4)-(10) (passenger flows, per duration branch) and Eqs. (30)-(40)
    with x_g = x*_g (``depot_counts``).  ``P`` = pi_hat_tau, ``Zs[D]`` =
    Z_m^B((j, tau)) of Eq. (13).

    export_prefix: optional path prefix; when given, the solved model is
    written as <prefix>.mps, <prefix>.sol and <prefix>.json (code, not in
    paper; stage2.export_model_dir).
    """
    group_mode = _normalize_nonanticipativity_mode(group.get(
        "nonanticipativity_mode", BUS_ONLY_NONANTICIPATIVITY))
    mode = (group_mode if nonanticipativity_mode is None else
            _normalize_nonanticipativity_mode(nonanticipativity_mode))
    if mode != group_mode:
        raise ValueError(
            "solver/group non-anticipativity mode mismatch: "
            f"solver={mode}, group={group_mode}")
    passenger_nonanticipativity = (
        mode == PASSENGER_AND_BUS_NONANTICIPATIVITY)

    t0 = time.time()
    durations = group["durations"]
    # 피해가 없는 지속시간은 Z_D = 0 이므로 흐름 블록을 만들지 않는다.
    # 확률은 durations 전체로 정규화한다 — 그 질량은 사라진 게 아니라 0 을 곱한다.
    active = group.get("active", durations)
    P = duration_weight or m3.DURATION_WEIGHT
    psum = sum(P.get(D, 0.0) for D in durations) or 1.0
    P = {D: P.get(D, 0.0) / psum for D in durations}

    sid = f"{group['station']}_adaptive"
    model = gp.Model(f"Stage2Adaptive_{sid}", env=env)
    model.Params.MIPGap = mip_gap if mip_gap is not None else m3.MIP_GAP
    model.Params.TimeLimit = time_limit if time_limit is not None else m3.TIME_LIMIT
    model.Params.OutputFlag = int(output_flag)
    if threads is not None:
        model.Params.Threads = int(threads)
    model.Params.MIPFocus = int(mip_focus)
    model.Params.Heuristics = float(heuristics)
    if int(cuts) >= 0:
        model.Params.Cuts = int(cuts)

    # ---- 배차변수: 정보집합 하나당 변수 하나 (등식 제약 없이 공유) ----
    # Eq. (31) [eq:na_dispatch] / Eq. (25) [eq:shared_trip_branches]: one y_v per
    # information cell P_v(xi) (variable sharing, Sec. 3.4.4).
    # Eq. (30) [eq:avail_cons] / Eq. (23) [eq:availability]: y_v exists only in
    # the branches where alpha_(v,tau) = 1 (group["cells"] from info_cells).
    # Eq. (39) [eq:domain_yu]: y binary.
    cells = group["cells"]
    zero_fleet_trip_cells_removed = 0
    if EXACT_TRIP_CELL_PRUNING:
        retained_cells = {}
        for tid, trip_cells in cells.items():
            depot = str(group["trips"][tid]["Depot"])
            if int(depot_counts.get(depot, 0)) <= 0:
                zero_fleet_trip_cells_removed += len(trip_cells)
            else:
                retained_cells[tid] = trip_cells
        cells = retained_cells

    yvar = {}
    for tid, trip_cells in cells.items():
        for cell in trip_cells:
            yvar[(tid, cell)] = model.addVar(
                vtype=GRB.BINARY, name=f"y[{tid}|{'-'.join(map(str, cell))}]")
    model.update()

    y_index = {}                       # (tid, D) -> 변수
    for (tid, cell), v in yvar.items():
        for D in cell:
            y_index[(tid, D)] = v

    # 순차 풀이용: 앞 단계에서 확정된 배차를 고정한다.
    # (단계별로 나눠 풀 때 이전 결정이 뒤 단계에서 번복되지 않게 한다)
    if fixed_tids:
        want = set(fixed_tids)
        for (tid, cell), v in yvar.items():
            if tid in want:
                v.LB = 1.0
        model.update()

    # Eqs. (24), (32)-(36), (39)-(40): duty start u, chaining psi, depot pools.
    chain = _add_nonanticipative_vehicle_chains(
        model, group, depots, depot_counts, yvar, y_index)

    def y_of_factory(D):
        return lambda tid: y_index.get((tid, D))

    # Passenger variables remain duration-specific in BUS_ONLY_WAIT_AND_SEE.
    # Passenger-and-bus mode reuses each pre-revelation Var object without
    # changing the raw duration-specific rail inputs.
    passenger_sharing = None
    if passenger_nonanticipativity and len(durations) == 2:
        passenger_sharing = {
            "start": int(group.get(
                "passenger_shared_start", group["disrupt_start"])),
            "end": int(group.get(
                "passenger_shared_end",
                group["disrupt_start"] + min(durations))),
            "durations": tuple(map(int, durations)),
            "registry": {},
        }

    # ---- 지속시간별 흐름 블록 ----
    # Eqs. (4)-(10) with V = V_r U V_rho, residual rail capacity S_(v,t) (Eq. (12)),
    # Eq. (37) [eq:cap_bus] and Eq. (38) [eq:min_svc]; one block per duration
    # branch tau (wait-and-see passenger recourse).  Zs[D] = Z_m^B((j, tau)).
    blocks, Zs = {}, {}
    normal_fixed_variable_count = 0
    normal_fixed_registry = {}
    for D in active:
        blocks[D] = _add_flow_block(m3, model, D, group["scen"][D],
                                    group["cache"][D], wdata,
                                    y_of_factory(D), tag=f"_{D}",
                                    passenger_sharing=passenger_sharing)
        if passenger_nonanticipativity:
            normal_fixed_variable_count += _fix_normal_passenger_decisions(
                blocks[D], D, group,
                fixed_registry=normal_fixed_registry)
        else:
            # Sec. 3.2.2: states immediately before onset are fixed to their
            # normal values (b/a/od of every minute t < t0).
            normal_fixed_variable_count += (
                _load_exact_engine().fix_pre_onset_to_normal(
                    {"b": blocks[D][0], "a": blocks[D][1],
                     "od": blocks[D][5]},
                    group["normal_trajectory"], int(group["disrupt_start"]),
                    group["scen"][D].get("Omega_m", ()),
                    label=f"{group['station']}_D{D}"))
        Zs[D] = blocks[D][7]
    passenger_sharing_stats = _validate_passenger_variable_sharing(
        passenger_sharing)
    model.update()

    if mip_start:
        selected_trip_cells = mip_start.get("selected_trip_cells")
        if selected_trip_cells is not None:
            selected = {
                (
                    str(item["trip_id"]),
                    tuple(int(duration) for duration in item["durations"]),
                )
                for item in selected_trip_cells
            }
            for (tid, cell), variable in yvar.items():
                variable.Start = (
                    1.0 if (str(tid), tuple(cell)) in selected else 0.0
                )
        else:
            selected_trip_ids = {
                str(tid) for tid in mip_start.get("selected_trip_ids", ())
            }
            for (tid, _cell), variable in yvar.items():
                variable.Start = 1.0 if str(tid) in selected_trip_ids else 0.0

    # Depot capacity: only the paper's vehicle-chain rows, Eqs. (34)-(36), limit the
    # buses leaving each depot (no per-minute occupancy rows; removed to match the paper).

    # ---- 목적함수 ----
    # Eq. (41) [eq:s2_eval]: min sum_tau pi_hat_tau Z_m^B((j, tau)); branches
    # without affected ODs have Z = 0 and carry no block.
    model.setObjective(gp.quicksum(P[D] * Zs[D] for D in active), GRB.MINIMIZE)
    model.update()
    build_sec = time.time() - t0

    t1 = time.time()
    model.optimize()
    solve_sec = time.time() - t1
    status = m3._status_name(model.Status)
    stats = m3.collect_gurobi_stats(model, solve_sec)
    if export_prefix is not None:
        _export_model(model, export_prefix, status)
    stats["BuildSec"] = float(build_sec)
    stats["NonanticipativityMode"] = mode
    stats["FormulationMode"] = (
        PASSENGER_AND_BUS_SHARED if passenger_nonanticipativity
        else BUS_ONLY_WAIT_AND_SEE)
    stats["BusDecisionInformationSetNonanticipative"] = True
    stats["BusTripActivationYInformationSetNonanticipative"] = True
    stats["BusDutyInitiationUInformationSetNonanticipative"] = True
    stats["BusVehicleChainingPsiInformationSetNonanticipative"] = True
    stats["PassengerRecourse"] = (
        "NONANTICIPATIVE" if passenger_nonanticipativity
        else "WAIT_AND_SEE")
    stats["PassengerFlowFamilies"] = ["q", "l", "a", "b", "d", "o"]
    stats["DurationSpecificPhysicalOmega"] = bool(
        not passenger_nonanticipativity)
    stats["DurationSpecificRailState"] = True
    stats["VehicleChainEnforced"] = True
    stats["VehicleChainInformationSetNonanticipative"] = True
    stats["VehicleChainDecisionScope"] = chain["decision_scope"]
    stats["VehicleChainBinaryAuxiliaries"] = chain["binary_auxiliaries"]
    stats["VehicleChainPhysicalArcs"] = chain["physical_arc_count"]
    stats["VehicleChainFirstVars"] = chain["first_var_count"]
    stats["VehicleChainFirstDecisions"] = chain["first_decision_count"]
    stats["VehicleChainFirstSubstituted"] = chain[
        "first_substituted_count"]
    stats["VehicleChainLinks"] = chain["link_var_count"]
    stats["VehicleChainConstrs"] = chain["constraint_count"]
    stats["VehicleChainZeroFixedLinks"] = chain["zero_fixed_link_count"]
    stats["PassengerFlowVariableSharing"] = bool(passenger_sharing)
    stats["PassengerSharedIntervalStart"] = (
        int(passenger_sharing["start"]) if passenger_sharing else None)
    stats["PassengerSharedIntervalEndExclusive"] = (
        int(passenger_sharing["end"]) if passenger_sharing else None)
    stats["PassengerSharedVars"] = int(
        passenger_sharing_stats["total"])
    for family in ("b", "a", "w", "od"):
        stats[f"PassengerShared_{family}"] = int(
            passenger_sharing_stats["by_family"].get(family, 0))
    stats["PassengerSharedEqualityConstrs"] = 0
    stats["PassengerStateNonanticipativeByConservation"] = bool(
        passenger_sharing)
    stats["NormalFixedPassengerDecisionVars"] = int(
        normal_fixed_variable_count)
    stats["CommonAffectedODCount"] = int(len(group.get(
        "common_omega", ())))
    for duration in durations:
        stats[f"PhysicalAffectedOD_D{duration}"] = int(len(
            group.get("physical_affected_omega", {}).get(duration, ())))
        stats[f"CausalAffectedOD_D{duration}"] = int(len(
            group.get("causal_affected_omega", {}).get(duration, ())))
        stats[f"NormalFixedOD_D{duration}"] = int(len(
            group.get("normal_fixed_omega", {}).get(duration, ())))
        stats[f"CausalSharedTrainWithdrawals_D{duration}"] = int(len(
            group["scen"][duration].get(
                "causal_shared_withdrawals", ())))
    rail_audit = group.get("rail_information_audit", {})
    stats["RailCommonEventTimetable"] = bool(
        rail_audit.get("single_common_event_timetable"))
    stats["RailTimetableEventCount"] = int(
        rail_audit.get("timetable_event_count", 0))
    stats["RailEventSupportInformationSetNonanticipative"] = bool(
        passenger_nonanticipativity
        and rail_audit.get("event_support_nonanticipative"))
    stats["RailEventSupportMismatchCount"] = int(
        rail_audit.get("event_support_mismatch_count", 0))
    stats["TerminationAlightEventsVerified"] = bool(
        passenger_nonanticipativity
        and rail_audit.get("termination_alight_events_verified"))
    stats["TerminationAlightEventCount"] = int(
        rail_audit.get("termination_alight_event_count", 0))
    stats["TerminationEventSupportInformationSetNonanticipative"] = bool(
        passenger_nonanticipativity
        and rail_audit.get("termination_event_support_nonanticipative"))
    stats["TerminationEventSupportMismatchCount"] = int(
        rail_audit.get("termination_event_support_mismatch_count", 0))
    stats["TerminationActionSupportInformationSetNonanticipative"] = bool(
        passenger_nonanticipativity
        and rail_audit.get("termination_action_support_nonanticipative"))
    stats["TerminationActionSupportMismatchCount"] = int(
        rail_audit.get("termination_action_support_mismatch_count", 0))
    stats["PassengerCapacityWithdrawalPolicy"] = (
        "ZERO_FROM_RAW_BRANCH_PHYSICAL_CLEARANCE_EVENT"
        if passenger_nonanticipativity else "DURATION_SPECIFIC_SERVICE_THETA")
    stats["TrainWithdrawalThetaInformationSetNonanticipative"] = bool(
        passenger_nonanticipativity
        and rail_audit.get("theta_nonanticipative"))
    stats["TrainWithdrawalThetaMismatchCount"] = int(
        rail_audit.get("theta_mismatch_count", 0))
    stats["PassengerCapacityThetaPresent"] = bool(
        passenger_nonanticipativity
        and rail_audit.get("passenger_capacity_theta_present"))
    stats["PassengerCapacityThetaMatchesClearanceEvents"] = bool(
        passenger_nonanticipativity
        and rail_audit.get("passenger_capacity_theta_matches_events"))
    stats["PassengerCapacityThetaInformationSetNonanticipative"] = bool(
        passenger_nonanticipativity
        and rail_audit.get("passenger_capacity_theta_nonanticipative"))
    stats["PassengerCapacityThetaMismatchCount"] = int(
        rail_audit.get("passenger_capacity_theta_mismatch_count", 0))
    stats["TrainCapacityInformationSetNonanticipative"] = bool(
        passenger_nonanticipativity
        and rail_audit.get("capacity_nonanticipative"))
    stats["TrainCapacitySharedMaxAbsDiff"] = float(
        rail_audit.get("capacity_max_abs_diff", 0.0))
    stats["RawBranchCanceledTrainsPreserved"] = bool(
        passenger_nonanticipativity
        and rail_audit.get("raw_canceled_trains_preserved"))
    stats["RawBranchCancellationStartsPreserved"] = bool(
        passenger_nonanticipativity
        and rail_audit.get("raw_cancellation_starts_preserved"))
    stats["RawBranchWithdrawalThetaPreserved"] = bool(
        passenger_nonanticipativity and rail_audit.get("raw_theta_preserved"))
    stats["RawBranchTerminationEventsPreserved"] = bool(
        passenger_nonanticipativity
        and rail_audit.get("raw_termination_events_preserved"))
    stats["RawBranchPassengerCapacityThetaPreserved"] = bool(
        passenger_nonanticipativity
        and rail_audit.get("raw_passenger_capacity_theta_preserved"))
    stats["RawBranchRailStatePreserved"] = bool(
        passenger_nonanticipativity
        and rail_audit.get("raw_branch_rail_state_preserved"))
    stats["ZeroImpactRailReconstructionCount"] = int(
        rail_audit.get("zero_impact_reconstruction_count", 0))
    stats["ZeroImpactRailReconstructionsVerified"] = bool(
        passenger_nonanticipativity
        and rail_audit.get("zero_impact_reconstructions_verified"))
    stats.update(_raw_duration_rail_stats(rail_audit))
    stats["CandidateTripBasisDuration"] = int(
        group.get("trip_basis_duration", max(durations)))
    stats["CandidateTripCount"] = int(len(group["trips"]))
    stats["ExactTripCellPruning"] = bool(EXACT_TRIP_CELL_PRUNING)
    stats["TripCellBinaryCount"] = int(len(yvar))
    stats["ZeroFleetTripCellsRemoved"] = int(
        zero_fleet_trip_cells_removed)
    stats["MIPStartSelectedTrips"] = int(
        len(mip_start.get("selected_trip_cells", ()))
        if mip_start and mip_start.get("selected_trip_cells") is not None
        else len(mip_start.get("selected_trip_ids", ())) if mip_start else 0)
    stats["MIPStartSourcePolicy"] = (
        mip_start.get("source_policy") if mip_start else None)
    stats["MIPStartSourceJobID"] = (
        mip_start.get("source_job_id") if mip_start else None)
    stats["MIPStartSourceRealizedObjective"] = (
        mip_start.get("source_realized_objective") if mip_start else None)
    stats["MIPStartSourceExpectedObjective"] = (
        mip_start.get("source_expected_objective") if mip_start else None)
    stats["MIPFocus"] = int(mip_focus)
    stats["Heuristics"] = float(heuristics)
    stats["Cuts"] = int(cuts)
    exact_metadata = [
        block[8] for block in blocks.values()
        if len(block) > 8 and block[8].get("exact_state_compression")
    ]
    stats["ExactStateCompression"] = bool(EXACT_STATE_COMPRESSION)
    stats["WalkingEliminatedExactly"] = bool(exact_metadata)
    stats["DestinationStatesEliminatedExactly"] = bool(exact_metadata)
    stats["PassengerLiteralSharedFamilies"] = (
        ["b", "a"] if passenger_sharing else [])
    stats["PassengerConservationSharedFamilies"] = (
        ["q", "l"] if passenger_sharing else [])
    stats["PassengerProjectedSharedFamilies"] = (
        ["d", "o"] if passenger_sharing and exact_metadata else [])
    stats["PassengerTransferFlowProjectedExactly"] = bool(
        passenger_sharing and exact_metadata
        and len(exact_metadata) == len(blocks))
    stats["PassengerExitFlowProjectedExactly"] = bool(
        passenger_sharing and exact_metadata
        and len(exact_metadata) == len(blocks))
    stats["PassengerTransferFlowNonanticipativityMechanism"] = (
        "EXACT_PROJECTION_FROM_SHARED_ALIGHTING"
        if stats["PassengerTransferFlowProjectedExactly"] else
        "LITERAL_VARIABLE_SHARING" if passenger_sharing else None)
    stats["PassengerExitFlowNonanticipativityMechanism"] = (
        "EXACT_PROJECTION_FROM_SHARED_ALIGHTING_AND_COMMON_DEMAND"
        if stats["PassengerExitFlowProjectedExactly"] else
        "LITERAL_VARIABLE_SHARING" if passenger_sharing else None)
    stats["QueueStatesOriginal"] = int(sum(
        metadata["original_q_states"] for metadata in exact_metadata))
    stats["QueueStatesRetained"] = int(sum(
        metadata["retained_q_states"] for metadata in exact_metadata))
    stats["LoadStatesOriginal"] = int(sum(
        metadata["original_l_states"] for metadata in exact_metadata))
    stats["LoadStatesRetained"] = int(sum(
        metadata["retained_l_states"] for metadata in exact_metadata))
    stats["WalkingStatesEliminated"] = int(sum(
        metadata["eliminated_w_states"] for metadata in exact_metadata))
    stats["DestinationQueueStatesEliminated"] = int(sum(
        metadata["eliminated_destination_q_states"]
        for metadata in exact_metadata))
    stats["ExitStatesEliminated"] = int(sum(
        metadata["eliminated_od_states"] for metadata in exact_metadata))

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) \
            or model.SolCount == 0:
        model.dispose()
        return {"status": status, "objective": 0.0, "Z": {}, "policy": [],
                "y_shared": {}, "stats": stats}

    state_audit = _passenger_state_sharing_audit(
        blocks, durations, passenger_sharing, wdata)
    stats["PassengerSharedLoadMaxAbsDiff"] = float(state_audit["l"])
    stats["PassengerSharedTrainLoadMaxAbsDiff"] = float(
        state_audit["train_l"])
    stats["PassengerSharedBusLoadMaxAbsDiff"] = float(
        state_audit["bus_l"])
    stats["PassengerSharedQueueMaxAbsDiff"] = float(state_audit["q"])
    stats["PassengerSharedTransferMaxAbsDiff"] = float(state_audit["d"])
    stats["PassengerSharedExitMaxAbsDiff"] = float(state_audit["o"])
    stats["PassengerTransferFlowInformationSetNonanticipative"] = bool(
        passenger_sharing and state_audit["d"] <= 1e-5)
    stats["PassengerExitFlowInformationSetNonanticipative"] = bool(
        passenger_sharing and state_audit["o"] <= 1e-5)
    if max(state_audit.values(), default=0.0) > 1e-5:
        model.dispose()
        raise RuntimeError(
            "shared passenger actions did not produce non-anticipative "
            f"states: {state_audit}")

    # ---- 결과 추출 ----
    horizon = int(wdata["Gamma"][-1])          # T (last minute of Gamma)
    Zval = {D: (float(Zs[D].getValue()) if D in Zs else 0.0) for D in durations}
    obj = float(model.ObjVal)
    extraction_tolerance = max(
        EXTRACTION_SOLUTION_TOLERANCE,
        10.0 * float(model.Params.FeasibilityTol),
    )

    policy = []
    for (tid, cell), v in sorted(yvar.items(),
                                 key=lambda kv: (group["trips"][kv[0][0]]["DispatchStart"],
                                                 kv[0][0])):
        if float(v.X) <= 0.5:
            continue
        tr = group["trips"][tid]
        boarded = {}
        for D in cell:
            bt = blocks[D][6].get(tid, []) if D in blocks else []
            boarded[D] = round(sum(float(x.X) for x in bt), 3)
        policy.append({
            "TripID": tid, "SharedDurations": cell,
            "Depot": tr["Depot"], "Route": tr["Route"],
            "RouteType": tr["RouteType"],
            "StartStation": tr["RouteOrigin"], "EndStation": tr["RouteDest"],
            "DispatchStart": tr["DispatchStart"], "RunStart": tr["RunStart"],
            "RunEnd": tr["RunEnd"], "Boarded": boarded,
        })

    chain_decisions = []
    for (tid, cell), variable in sorted(chain["first_decisions"].items()):
        if float(variable.X) <= 0.5:
            continue
        chain_decisions.append({
            "DecisionType": "FIRST",
            "PredecessorTripID": None,
            "SuccessorTripID": tid,
            "DecisionTime": int(group["trips"][tid]["DispatchStart"]),
            "InformationCell": tuple(cell),
            "Value": float(variable.X),
        })
    for (predecessor, successor, cell), variable in sorted(
            chain["link_vars"].items()):
        if float(variable.X) <= 0.5:
            continue
        chain_decisions.append({
            "DecisionType": "LINK",
            "PredecessorTripID": predecessor,
            "SuccessorTripID": successor,
            "DecisionTime": _chain_arc_decision_time(
                group["trips"][predecessor], group["trips"][successor],
                group["travel_times"]),
            "InformationCell": tuple(cell),
            "Value": float(variable.X),
        })

    # Positive passenger-flow decision variables in the established
    # disruption CSV schema.  Extraction must happen before model.dispose().
    omega_map = wdata["Omega_Map"]
    dv_rows = {D: [] for D in durations}

    def endpoints(omega):
        if omega in omega_map:
            return omega_map[omega]
        parts = str(omega).split("_")
        return parts[0], parts[-1]

    def append_row(rows, variable, omega, minute, station, vehicle, value):
        value = float(value.X)
        minute = int(minute)
        if value <= 1e-7 or not 0 <= minute <= horizon:
            return
        origin, destination = endpoints(omega)
        rows.append((variable, omega, origin, destination, minute,
                     station, vehicle, value))

    for D, block in blocks.items():
        rows = dv_rows[D]
        b_d, a_d, l_d, q_d, w_d, od_d = block[:6]
        metadata = block[8] if len(block) > 8 else None
        for (omega, vehicle, minute, station), variable in b_d.items():
            append_row(rows, "Boarding", omega, minute, station, vehicle, variable)
        for (omega, vehicle, minute, station), variable in a_d.items():
            append_row(rows, "Alighting", omega, minute, station, vehicle, variable)
        for (omega, vehicle, minute), variable in l_d.items():
            append_row(rows, "Load", omega, minute, None, vehicle, variable)
        if metadata and metadata.get("exact_state_compression"):
            reconstructed_bus_loads = metadata.get("_audited_bus_loads")
            if reconstructed_bus_loads is None:
                reconstructed_bus_loads = metadata["engine"].\
                    _reconstruct_hybrid_bus_loads(
                        metadata["load_groups"], b_d, a_d,
                        tolerance=extraction_tolerance)
            for (omega, vehicle, minute), variable in \
                    reconstructed_bus_loads.items():
                append_row(
                    rows, "Load", omega, minute, None, vehicle, variable)
        for (omega, station, minute), variable in q_d.items():
            append_row(rows, "Queue", omega, minute, station, None, variable)
        for (omega, station, minute), variable in w_d.items():
            append_row(rows, "Walking", omega, minute, station, None, variable)
        for (omega, minute), variable in od_d.items():
            append_row(rows, "Exit", omega, minute, endpoints(omega)[1], None, variable)
        if metadata and metadata.get("exact_state_compression"):
            rows.extend(metadata["walking_plan"].iter_positive_decision_rows(
                a_d, epsilon=1e-7, time_min=0, time_max=horizon,
                feasibility_tolerance=extraction_tolerance))
            rows.extend(metadata["engine"]._absorbed_destination_exit_rows(
                a_d, wdata["D"], omega_map, wdata["Gamma"],
                WALKING_MINUTES, epsilon=1e-7, time_min=0, time_max=horizon))

    y_shared = {(tid, cell): float(v.X) for (tid, cell), v in yvar.items()}
    model.dispose()
    return {"status": status, "objective": obj, "Z": Zval,
            "policy": policy, "y_shared": y_shared, "stats": stats,
            "weights": P, "dv_rows": dv_rows,
            "chain_decisions": chain_decisions}
