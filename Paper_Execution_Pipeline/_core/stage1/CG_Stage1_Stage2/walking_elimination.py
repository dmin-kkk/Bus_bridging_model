"""Exact elimination and reconstruction of the walking inventory.

The Stage-1 formulation defines, for every non-destination station,

    w[t] = w[t - 1] + sum_v a[v, t] - sum_v a[v, t - sigma].

On the production support this state is nonnegative by construction and is
therefore a definitional variable.  This module validates that support before
projecting it out.  It deliberately has no Gurobi dependency.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping


Node = tuple[object, object]
Event = tuple[object, object, int]
Run = tuple[int, int]  # half-open [start, stop)


def _normalise_alighting_key(key) -> tuple[object, object, int]:
    """Return ``(omega, station, minute)`` from an ``a`` variable key."""
    try:
        omega, _vehicle, minute, station = key
    except (TypeError, ValueError) as error:
        raise ValueError(
            "alighting keys must be (omega, vehicle, minute, station)"
        ) from error
    return omega, station, int(minute)


def _value(raw) -> float:
    """Read either a numeric solution value or a live Gurobi-like ``.X``."""
    value = float(raw.X) if hasattr(raw, "X") else float(raw)
    if not math.isfinite(value):
        raise ValueError(f"non-finite alighting solution value: {value!r}")
    return value


def _runs(minutes: list[int]) -> tuple[Run, ...]:
    if not minutes:
        return ()
    minutes.sort()
    output = []
    start = previous = minutes[0]
    for minute in minutes[1:]:
        if minute == previous:
            raise ValueError(f"duplicate walking row minute {minute}")
        if minute != previous + 1:
            output.append((start, previous + 1))
            start = minute
        previous = minute
    output.append((start, previous + 1))
    return tuple(output)


def _intersection_count(runs: tuple[Run, ...], start: int, stop: int) -> int:
    if stop <= start:
        return 0
    total = 0
    for run_start, run_stop in runs:
        if run_stop <= start:
            continue
        if run_start >= stop:
            break
        total += max(0, min(run_stop, stop) - max(run_start, start))
    return total


@dataclass(frozen=True, slots=True)
class WalkingEliminationPlan:
    """Validated, compressed information needed after eliminating ``w``.

    ``row_runs`` stores the actual modeled walking-row universe as half-open
    contiguous runs. ``multipliers[(omega, station, u)]`` is the exact number
    of modeled minute states to which an alighting event at minute ``u``
    contributes. Destination events and ``sigma == 0`` have multiplier zero.
    """

    sigma: int
    horizon_start: int
    horizon_stop: int
    omega_map: Mapping
    row_runs: Mapping[Node, tuple[Run, ...]]
    multipliers: Mapping[Event, int]

    @classmethod
    def build(
        cls,
        *,
        w_rows: Iterable[tuple],
        alighting_keys: Iterable[tuple],
        omega_map: Mapping,
        gamma: Iterable[int],
        sigma: int,
    ) -> "WalkingEliminationPlan":
        """Validate a production-safe row universe and build its projection.

        The minute horizon must be a contiguous integer grid.  Every possible
        non-destination alighting event inside that grid must own every walking
        row in its (horizon-truncated) interval ``[u, u + sigma)``.  This is the
        invariant produced by ``compact_full_pool_cache``.  It rules out sparse
        row patterns in which ``w >= 0`` would impose an otherwise hidden
        restriction on an alighting variable.
        """
        sigma = int(sigma)
        if sigma < 0:
            raise ValueError("sigma must be nonnegative")

        minutes = tuple(int(minute) for minute in gamma)
        if not minutes:
            raise ValueError("gamma must contain at least one minute")
        if len(set(minutes)) != len(minutes):
            raise ValueError("gamma contains duplicate minutes")
        if minutes != tuple(range(minutes[0], minutes[-1] + 1)):
            raise ValueError("gamma must be a sorted contiguous integer grid")
        horizon_start, horizon_stop = minutes[0], minutes[-1] + 1

        endpoints = {}
        for omega, pair in omega_map.items():
            try:
                origin, destination = pair
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"omega_map[{omega!r}] must contain (origin, destination)"
                ) from error
            endpoints[omega] = (origin, destination)

        by_node: dict[Node, list[int]] = defaultdict(list)
        for raw_row in w_rows:
            try:
                omega, station, raw_minute = raw_row
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "walking rows must be (omega, station, minute)"
                ) from error
            if omega not in endpoints:
                raise ValueError(f"walking row uses unknown omega {omega!r}")
            if station == endpoints[omega][1]:
                raise ValueError(
                    f"destination walking row is not valid: {raw_row!r}")
            minute = int(raw_minute)
            if not horizon_start <= minute < horizon_stop:
                raise ValueError(
                    f"walking row minute {minute} is outside gamma")
            by_node[(omega, station)].append(minute)

        compressed = {
            node: _runs(node_minutes)
            for node, node_minutes in by_node.items()
        }

        events = set()
        for raw_key in alighting_keys:
            omega, station, minute = _normalise_alighting_key(raw_key)
            if omega not in endpoints:
                raise ValueError(f"alighting key uses unknown omega {omega!r}")
            events.add((omega, station, minute))

        multipliers = {}
        for omega, station, minute in events:
            destination = endpoints[omega][1]
            if station == destination or sigma == 0:
                multipliers[(omega, station, minute)] = 0
                continue
            if not horizon_start <= minute < horizon_stop:
                raise ValueError(
                    "non-destination alighting minute outside gamma cannot be "
                    f"safely eliminated: {(omega, station, minute)!r}")
            stop = min(minute + sigma, horizon_stop)
            actual = _intersection_count(
                compressed.get((omega, station), ()), minute, stop)
            required = stop - minute
            if actual != required:
                raise ValueError(
                    "unsafe walking row universe for alighting event "
                    f"{(omega, station, minute)!r}: expected all {required} "
                    f"rows in [{minute}, {stop}), found {actual}")
            multipliers[(omega, station, minute)] = actual

        return cls(
            sigma=sigma,
            horizon_start=horizon_start,
            horizon_stop=horizon_stop,
            omega_map=MappingProxyType(endpoints),
            row_runs=MappingProxyType(compressed),
            multipliers=MappingProxyType(multipliers),
        )

    def objective_coefficients(
        self, *, probability: float, horizon: float
    ) -> dict[Event, float]:
        """Return exact walking-objective coefficients for aggregate ``a``."""
        probability, horizon = float(probability), float(horizon)
        if probability < 0 or not math.isfinite(probability):
            raise ValueError("probability must be finite and nonnegative")
        if horizon <= 0 or not math.isfinite(horizon):
            raise ValueError("horizon must be finite and positive")
        scale = probability / horizon
        return {key: scale * multiplier
                for key, multiplier in self.multipliers.items()}

    def _aggregate_solution(
        self, alighting_solution: Mapping, *, feasibility_tolerance: float
    ) -> dict[Node, dict[int, float]]:
        tolerance = float(feasibility_tolerance)
        if tolerance < 0:
            raise ValueError("feasibility_tolerance must be nonnegative")
        events: dict[Node, dict[int, float]] = defaultdict(
            lambda: defaultdict(float))
        for raw_key, raw_value in alighting_solution.items():
            omega, station, minute = _normalise_alighting_key(raw_key)
            if omega not in self.omega_map:
                raise ValueError(f"solution uses unknown omega {omega!r}")
            value = _value(raw_value)
            if value < -tolerance:
                raise ValueError(
                    f"negative alighting solution value {value} at {raw_key!r}")
            if value < 0:
                value = 0.0
            event = (omega, station, minute)
            destination = self.omega_map[omega][1]
            if event not in self.multipliers:
                if station == destination or abs(value) <= tolerance:
                    continue
                raise ValueError(
                    f"positive alighting value lies outside validated support: "
                    f"{raw_key!r}")
            if value:
                events[(omega, station)][minute] += value
        return {node: dict(values) for node, values in events.items()}

    def walking_area(
        self, alighting_solution: Mapping, *,
        feasibility_tolerance: float = 1e-9,
    ) -> float:
        """Calculate ``sum_t w[t]`` without constructing minute states."""
        events = self._aggregate_solution(
            alighting_solution,
            feasibility_tolerance=feasibility_tolerance)
        terms = []
        for (omega, station), by_minute in events.items():
            for minute, value in by_minute.items():
                terms.append(
                    value * self.multipliers[(omega, station, minute)])
        return float(math.fsum(terms))

    def iter_positive_decision_rows(
        self,
        alighting_solution: Mapping,
        *,
        epsilon: float = 1e-4,
        time_min: int | None = None,
        time_max: int | None = None,
        feasibility_tolerance: float = 1e-9,
    ) -> Iterator[tuple]:
        """Lazily yield legacy-format positive minute ``Walking`` rows.

        Each row has the existing schema
        ``Variable, Omega, Origin, Dest, Time, StationID, TrainID, Value``.
        """
        epsilon = float(epsilon)
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        lower = self.horizon_start if time_min is None else int(time_min)
        upper = self.horizon_stop - 1 if time_max is None else int(time_max)
        if lower > upper:
            raise ValueError("time_min must not exceed time_max")

        events = self._aggregate_solution(
            alighting_solution,
            feasibility_tolerance=feasibility_tolerance)
        for node in sorted(events, key=repr):
            omega, station = node
            origin, destination = self.omega_map[omega]
            by_minute = events[node]
            for run_start, run_stop in self.row_runs.get(node, ()):
                state = math.fsum(
                    by_minute.get(minute, 0.0)
                    for minute in range(
                        max(self.horizon_start,
                            run_start - self.sigma + 1),
                        run_start + 1))
                for minute in range(run_start, run_stop):
                    if minute > run_start:
                        state += by_minute.get(minute, 0.0)
                        state -= by_minute.get(minute - self.sigma, 0.0)
                    if lower <= minute <= upper and state > epsilon:
                        yield (
                            "Walking", omega, origin, destination, minute,
                            station, None, float(state))


def build_walking_elimination_plan(**kwargs) -> WalkingEliminationPlan:
    """Functional wrapper around :meth:`WalkingEliminationPlan.build`."""
    return WalkingEliminationPlan.build(**kwargs)


__all__ = ["WalkingEliminationPlan", "build_walking_elimination_plan"]
