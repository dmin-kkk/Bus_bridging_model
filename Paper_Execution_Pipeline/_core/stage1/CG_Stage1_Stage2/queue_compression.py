"""Exact event-interval contraction for non-destination queue states.

The helper is deliberately independent of Gurobi and of :mod:`cg_stage1`.
It contracts only a *fixed, complete* queue/coupling support.  For a
non-destination ``(omega, station)`` stream, a queue event is

* a nonzero exogenous-demand minute,
* a minute containing any retained boarding variable, or
* ``alight_minute + sigma`` for any retained alighting variable.

Between two such events the original minute balance has zero increment, so a
single state represents the whole interval.  Destination streams remain
minute-indexed because the time-indexed exit variable can change their queue
in every minute.  No numerical support tolerance is used.

Exactness requires ``b_keys`` and ``a_keys`` to come from the complete
candidate pool, not merely the currently active restricted-master columns.
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Hashable


QueueKey = tuple[Hashable, Hashable, int]
QueueNode = tuple[Hashable, Hashable]


def _minute(raw: Any, label: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} is not an integer minute: {raw!r}") from exc
    if isinstance(raw, float) and (not math.isfinite(raw) or raw != value):
        raise ValueError(f"{label} is not an integer minute: {raw!r}")
    return value


def _key(raw: Any, size: int, label: str) -> tuple:
    try:
        value = tuple(raw)
    except TypeError as exc:
        raise ValueError(f"{label} must be an iterable key") from exc
    if len(value) != size:
        raise ValueError(f"{label} must contain {size} fields: {raw!r}")
    return value


def _runs(minutes: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Return maximal ``[start, end)`` runs from sorted unique minutes."""
    if not minutes:
        return ()
    output = []
    start = previous = minutes[0]
    for minute in minutes[1:]:
        if minute != previous + 1:
            output.append((start, previous + 1))
            start = minute
        previous = minute
    output.append((start, previous + 1))
    return tuple(output)


@dataclass(frozen=True, slots=True)
class QueueSegment:
    """One retained state and the original minutes that it represents."""

    omega: Hashable
    station: Hashable
    start: int
    end: int
    predecessor_start: int | None
    destination: bool

    @property
    def key(self) -> QueueKey:
        return self.omega, self.station, self.start

    @property
    def predecessor(self) -> QueueKey | None:
        if self.predecessor_start is None:
            return None
        return self.omega, self.station, self.predecessor_start

    @property
    def weight(self) -> int:
        return self.end - self.start

    @property
    def minutes(self) -> range:
        return range(self.start, self.end)


@dataclass(frozen=True, slots=True)
class QueueNodeSegments:
    """Compact segment arrays for one ``(omega, station)`` stream."""

    omega: Hashable
    station: Hashable
    destination: bool
    support_runs: tuple[tuple[int, int], ...]
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    predecessor_starts: tuple[int | None, ...]

    def __post_init__(self) -> None:
        if not (len(self.starts) == len(self.ends) ==
                len(self.predecessor_starts)):
            raise ValueError("queue segment arrays have different lengths")

    def segment(self, start: int) -> QueueSegment:
        position = bisect_left(self.starts, int(start))
        if position >= len(self.starts) or self.starts[position] != int(start):
            raise KeyError((self.omega, self.station, int(start)))
        return QueueSegment(
            self.omega, self.station, self.starts[position],
            self.ends[position], self.predecessor_starts[position],
            self.destination)

    def segment_for_minute(self, minute: int) -> QueueSegment | None:
        minute = int(minute)
        position = bisect_right(self.starts, minute) - 1
        if position < 0 or minute >= self.ends[position]:
            return None
        return QueueSegment(
            self.omega, self.station, self.starts[position],
            self.ends[position], self.predecessor_starts[position],
            self.destination)

    def contains_original_minute(self, minute: int) -> bool:
        minute = int(minute)
        # A cache normally has one run; retaining the general check makes the
        # helper exactly match legacy sparse test/audit supports as well.
        return any(start <= minute < end for start, end in self.support_runs)

    def iter_segments(self) -> Iterator[QueueSegment]:
        for start in self.starts:
            yield self.segment(start)


class _SegmentFieldView(Mapping):
    """Read-only, non-materialising view of one segment metadata field."""

    def __init__(self, metadata: "QueueCompression", field: str):
        self._metadata = metadata
        self._field = field

    def __len__(self) -> int:
        return self._metadata.model_key_count

    def __iter__(self) -> Iterator[QueueKey]:
        return self._metadata.iter_model_keys()

    def __getitem__(self, key: QueueKey):
        return getattr(self._metadata.segment(key), self._field)


@dataclass(frozen=True, slots=True)
class QueueCompression:
    """Exact queue-contraction metadata with lazy minute reconstruction."""

    nodes: Mapping[QueueNode, QueueNodeSegments]
    original_key_count: int
    model_key_count: int
    represented_minute_count: int
    fixed_zero_minute_count: int

    def iter_segments(self) -> Iterator[QueueSegment]:
        for node in self.nodes.values():
            yield from node.iter_segments()

    def iter_model_keys(self) -> Iterator[QueueKey]:
        for segment in self.iter_segments():
            yield segment.key

    @property
    def model_keys(self) -> tuple[QueueKey, ...]:
        """Materialise model keys; prefer :meth:`iter_model_keys` for audits."""
        return tuple(self.iter_model_keys())

    @property
    def weights(self) -> Mapping[QueueKey, int]:
        return _SegmentFieldView(self, "weight")

    @property
    def predecessors(self) -> Mapping[QueueKey, QueueKey | None]:
        return _SegmentFieldView(self, "predecessor")

    @property
    def segment_ends(self) -> Mapping[QueueKey, int]:
        return _SegmentFieldView(self, "end")

    def segment(self, key: QueueKey) -> QueueSegment:
        omega, station, raw_minute = _key(key, 3, "queue key")
        node = self.nodes.get((omega, station))
        if node is None:
            raise KeyError((omega, station, int(raw_minute)))
        return node.segment(_minute(raw_minute, "queue minute"))

    def contains_original_key(self, key: QueueKey) -> bool:
        omega, station, raw_minute = _key(key, 3, "queue key")
        node = self.nodes.get((omega, station))
        return bool(node is not None and node.contains_original_minute(
            _minute(raw_minute, "queue minute")))

    def model_key_for_minute(self, key: QueueKey) -> QueueKey | None:
        """Map an original minute key to its state, or ``None`` if fixed zero.

        ``None`` also denotes a minute outside the supplied original support;
        callers can distinguish that case with :meth:`contains_original_key`.
        """
        omega, station, raw_minute = _key(key, 3, "queue key")
        node = self.nodes.get((omega, station))
        if node is None:
            return None
        segment = node.segment_for_minute(_minute(raw_minute, "queue minute"))
        return None if segment is None else segment.key

    def iter_original_mapping(
            self) -> Iterator[tuple[QueueKey, QueueKey | None]]:
        """Lazily map every supplied minute key to a retained model key."""
        for node in self.nodes.values():
            for run_start, run_end in node.support_runs:
                for minute in range(run_start, run_end):
                    segment = node.segment_for_minute(minute)
                    yield ((node.omega, node.station, minute),
                           None if segment is None else segment.key)

    def iter_minute_assignments(
            self, values: Mapping[QueueKey, Any], *,
            include_fixed_zero: bool = False, zero_value: Any = 0,
            strict: bool = True) -> Iterator[tuple[QueueKey, Any]]:
        """Lazily expand model-state values to the original minute support.

        Values may be numbers or live solver-variable objects.  The same object
        is yielded for every represented minute, so no full trajectory is
        allocated.  Fixed-zero prefixes/runs are omitted unless requested.
        """
        if include_fixed_zero:
            mappings = self.iter_original_mapping()
            for minute_key, model_key in mappings:
                if model_key is None:
                    yield minute_key, zero_value
                    continue
                try:
                    value = values[model_key]
                except KeyError:
                    if strict:
                        raise
                    continue
                yield minute_key, value
            return
        for segment in self.iter_segments():
            try:
                value = values[segment.key]
            except KeyError:
                if strict:
                    raise
                continue
            for minute in segment.minutes:
                yield (segment.omega, segment.station, minute), value


def build_queue_compression(
        q_keys: Iterable[tuple], b_keys: Iterable[tuple],
        a_keys: Iterable[tuple], demand: Mapping,
        omega_map: Mapping, sigma: int) -> QueueCompression:
    """Build exact queue event segments from a complete fixed support.

    ``q_keys`` are the original minute-level queue keys.  ``b_keys`` and
    ``a_keys`` must include every retained train and full-pool candidate-bus
    key that can ever enter the master.  Quiet prefixes and wholly quiet runs
    of a non-destination stream are identified as fixed zero.  Sparse gaps in
    the supplied queue support start a new recurrence, matching the legacy
    ``q[t-1] if present else 0`` semantics.
    """
    sigma = _minute(sigma, "sigma")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")

    minutes_by_node: dict[QueueNode, list[int]] = {}
    original_count = 0
    for raw in q_keys:
        omega, station, raw_minute = _key(raw, 3, "q key")
        try:
            hash((omega, station))
        except TypeError as exc:
            raise ValueError(f"unhashable queue node: {(omega, station)!r}") from exc
        minutes_by_node.setdefault((omega, station), []).append(
            _minute(raw_minute, "q minute"))
        original_count += 1

    for node, minutes in minutes_by_node.items():
        minutes.sort()
        if any(left == right for left, right in zip(minutes, minutes[1:])):
            raise ValueError(f"duplicate q variable key for {node!r}")

    events: dict[QueueNode, set[int]] = defaultdict(set)
    for raw, raw_value in demand.items():
        omega, station, raw_minute = _key(raw, 3, "demand key")
        node = (omega, station)
        # ``D`` can contain the union of many scenario/window streams.  Only
        # nodes represented by this scenario's q support can affect its event
        # partition; filtering here avoids rebuilding a full-demand event map
        # for every scenario.
        if node not in minutes_by_node:
            continue
        try:
            finite = math.isfinite(raw_value)
            nonzero = bool(raw_value != 0)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid demand value at {raw!r}: {raw_value!r}") from exc
        if not finite:
            raise ValueError(f"non-finite demand value at {raw!r}: {raw_value!r}")
        # Compare in the input numeric type.  Casting first would underflow a
        # valid Decimal/Fractional demand to float zero and cease to be exact.
        if nonzero:
            events[node].add(_minute(raw_minute, "demand minute"))
    for raw in b_keys:
        omega, _vehicle, raw_minute, station = _key(raw, 4, "b key")
        events[(omega, station)].add(_minute(raw_minute, "boarding minute"))
    for raw in a_keys:
        omega, _vehicle, raw_minute, station = _key(raw, 4, "a key")
        events[(omega, station)].add(
            _minute(raw_minute, "alighting minute") + sigma)

    node_metadata = {}
    model_count = represented_count = 0
    for (omega, station), raw_minutes in minutes_by_node.items():
        if omega not in omega_map:
            raise ValueError(f"omega_map has no destination for {omega!r}")
        endpoints = tuple(omega_map[omega])
        if len(endpoints) < 2:
            raise ValueError(f"omega_map entry has no destination: {omega!r}")
        destination = station == endpoints[1]
        minutes = tuple(raw_minutes)
        support_runs = _runs(minutes)
        starts: list[int] = []
        ends: list[int] = []
        predecessors: list[int | None] = []

        for run_start, run_end in support_runs:
            if destination:
                run_events = tuple(range(run_start, run_end))
            else:
                run_events = tuple(
                    minute for minute in range(run_start, run_end)
                    if minute in events.get((omega, station), ()))
            previous_event = None
            for position, event in enumerate(run_events):
                next_event = (run_events[position + 1]
                              if position + 1 < len(run_events) else run_end)
                starts.append(event)
                ends.append(next_event)
                predecessors.append(previous_event)
                previous_event = event

        metadata = QueueNodeSegments(
            omega=omega, station=station, destination=destination,
            support_runs=support_runs, starts=tuple(starts), ends=tuple(ends),
            predecessor_starts=tuple(predecessors))
        node_metadata[(omega, station)] = metadata
        model_count += len(starts)
        represented_count += sum(end - start for start, end in zip(starts, ends))

    return QueueCompression(
        nodes=MappingProxyType(node_metadata),
        original_key_count=original_count,
        model_key_count=model_count,
        represented_minute_count=represented_count,
        fixed_zero_minute_count=original_count - represented_count)


__all__ = [
    "QueueCompression", "QueueKey", "QueueNode", "QueueNodeSegments",
    "QueueSegment", "build_queue_compression",
]
