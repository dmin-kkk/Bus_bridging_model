"""Exact event-interval metadata for piecewise-constant state variables.

This module is deliberately independent of Gurobi and of the Stage 1/Stage 2
model builders.  It provides only the lossless bookkeeping needed to replace a
continuous minute ledger by one state variable per event interval and to expose
the original minute keys lazily after a solve.

For a fixed stream, the state is assumed to obey a recurrence whose right-hand
side can change only at the supplied event minutes.  The timeline itself is
still validated as a continuous, one-minute grid.  No temporal interpolation,
rounding, or aggregation is performed.
"""

from __future__ import annotations

import operator
from bisect import bisect_right
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    ItemsView,
    Mapping,
    ValuesView,
)
from dataclasses import dataclass
from itertools import chain
from typing import Generic, TypeVar


StateValue = TypeVar("StateValue")
StreamKey = tuple[Hashable, ...]
StateKeyFactory = Callable[[StreamKey, int], Hashable]


def _minute(value, *, label: str) -> int:
    """Return an integer minute without silently truncating a real value."""
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer minute, not bool")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{label} must be an integer minute: {value!r}") from exc


def _stream_key(value) -> StreamKey:
    if not isinstance(value, tuple):
        raise TypeError("stream_key must be a tuple")
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError("stream_key must contain only hashable members") from exc
    return value


def _timeline_bounds(timeline: Iterable[int]) -> tuple[int, int]:
    """Validate a nonempty increasing one-minute grid and return its bounds."""
    if isinstance(timeline, range):
        if not timeline:
            raise ValueError("state timeline must not be empty")
        if timeline.step != 1:
            raise ValueError("state timeline must increase in one-minute steps")
        return int(timeline.start), int(timeline.stop - 1)

    iterator = iter(timeline)
    try:
        first = _minute(next(iterator), label="timeline minute")
    except StopIteration as exc:
        raise ValueError("state timeline must not be empty") from exc
    previous = first
    for raw_minute in iterator:
        minute = _minute(raw_minute, label="timeline minute")
        if minute != previous + 1:
            raise ValueError(
                "state timeline has a gap or is not strictly increasing: "
                f"{previous!r} -> {minute!r}"
            )
        previous = minute
    return first, previous


def _default_state_key(stream_key: StreamKey, start: int) -> tuple:
    return stream_key + (start,)


@dataclass(frozen=True, slots=True)
class StateSegment:
    """Immutable metadata for one constant, inclusive minute interval.

    ``state_key`` identifies the single compressed solver value that represents
    every original minute state in ``[start, end]``.  ``has_event`` distinguishes
    a genuine boarding/alighting boundary from a timeline-start boundary added
    to preserve the initial-state recurrence.
    """

    stream_key: StreamKey
    state_key: Hashable
    start: int
    end: int
    has_event: bool = True

    def __post_init__(self) -> None:
        stream_key = _stream_key(self.stream_key)
        start = _minute(self.start, label="segment start")
        end = _minute(self.end, label="segment end")
        if end < start:
            raise ValueError(
                f"segment end precedes start for {stream_key!r}: {start}..{end}"
            )
        try:
            hash(self.state_key)
        except TypeError as exc:
            raise TypeError("state_key must be hashable") from exc
        object.__setattr__(self, "stream_key", stream_key)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "has_event", bool(self.has_event))

    @property
    def length(self) -> int:
        """Number of represented minutes, including both endpoints."""
        return self.end - self.start + 1

    def contains(self, minute: int) -> bool:
        minute = _minute(minute, label="minute")
        return self.start <= minute <= self.end

    def minute_keys(self):
        """Yield original minute-level keys without materializing them."""
        for minute in range(self.start, self.end + 1):
            yield self.stream_key + (minute,)


def build_load_segments(
    stream_key: StreamKey,
    timeline: Iterable[int],
    event_minutes: Iterable[int] = (),
    *,
    state_key_factory: StateKeyFactory | None = None,
) -> tuple[StateSegment, ...]:
    """Build an exact constant-state partition for one load stream.

    The timeline must be a nonempty, contiguous sequence of integer minutes.
    Duplicate event minutes are harmless and are deduplicated.  Every event
    must lie in the timeline.  The first timeline minute is always a segment
    boundary, even if no event occurs there, so an initial load (zero in the
    current corridor implementation) remains explicit.

    By default the compressed state key has the original load-key shape
    ``stream_key + (segment_start,)``.
    """
    stream_key = _stream_key(stream_key)
    first, last = _timeline_bounds(timeline)
    events = {
        _minute(raw_minute, label="event minute")
        for raw_minute in event_minutes
    }
    outside = sorted(minute for minute in events if minute < first or minute > last)
    if outside:
        raise ValueError(
            f"event minutes outside timeline {first}..{last}: {outside[:5]}"
        )

    factory = state_key_factory or _default_state_key
    boundaries = sorted(events | {first})
    segments = []
    for position, start in enumerate(boundaries):
        end = boundaries[position + 1] - 1 if position + 1 < len(boundaries) else last
        state_key = factory(stream_key, start)
        segments.append(StateSegment(
            stream_key=stream_key,
            state_key=state_key,
            start=start,
            end=end,
            has_event=start in events,
        ))
    return tuple(segments)


def build_bus_load_segments(
    stream_key: StreamKey,
    group: Mapping,
    *,
    state_key_factory: StateKeyFactory | None = None,
) -> tuple[StateSegment, ...]:
    """Build segments from an ``aggregate_bus_load_groups`` ledger.

    A valid bus ledger is continuous from its first retained boarding/alighting
    event through its last retained event.  Both endpoints are validated here;
    this catches a torn coverage interval or an accidentally padded ledger.
    """
    if not isinstance(group, Mapping):
        raise TypeError("bus load group must be a mapping")
    if "times" not in group:
        raise ValueError("bus load group is missing its continuous 'times' ledger")
    board = group.get("board", {})
    alight = group.get("alight", {})
    if not isinstance(board, Mapping) or not isinstance(alight, Mapping):
        raise TypeError("bus group 'board' and 'alight' entries must be mappings")

    event_minutes = {
        _minute(raw_minute, label="bus event minute")
        for raw_minute in chain(board, alight)
    }
    if not event_minutes:
        raise ValueError("bus load group must contain a boarding or alighting event")

    segments = build_load_segments(
        stream_key,
        group["times"],
        event_minutes,
        state_key_factory=state_key_factory,
    )
    first, last = segments[0].start, segments[-1].end
    if min(event_minutes) != first or max(event_minutes) != last:
        raise ValueError(
            "bus load timeline must run exactly from its first event through "
            f"its last event: timeline={first}..{last}, "
            f"events={min(event_minutes)}..{max(event_minutes)}"
        )
    return segments


def build_train_load_segments(
    stream_key: StreamKey,
    active_times: Iterable[int],
    *,
    boarding_minutes: Iterable[int] = (),
    alighting_minutes: Iterable[int] = (),
    state_key_factory: StateKeyFactory | None = None,
) -> tuple[StateSegment, ...]:
    """Build train-load segments on a continuous active-time range.

    Callers should supply all *possible* retained boarding and alighting
    minutes for the ``(OD, train)`` stream, not only events that happen to be
    positive in a solved profile.  The active-range start remains a boundary
    even when the first passenger-flow event occurs later.
    """
    return build_load_segments(
        stream_key,
        active_times,
        chain(boarding_minutes, alighting_minutes),
        state_key_factory=state_key_factory,
    )


class _ExpandedItemsView(ItemsView):
    def __iter__(self):
        return self._mapping.iter_expanded_items()


class _ExpandedValuesView(ValuesView):
    def __iter__(self):
        return self._mapping.iter_expanded_values()


class ExpandedStateView(Mapping, Generic[StateValue]):
    """Read-only lazy mapping from original minute keys to segment values.

    The view stores only the compressed state mapping and segment metadata.  It
    never materializes the minute-level dictionary.  Iteration is grouped by
    first-seen stream and ordered by minute within each stream.  Repeated values
    are the same underlying object, which is useful for Gurobi ``Var`` handles.

    By default segments for a stream must be adjacent.  Set
    ``validate_contiguous=False`` only for a state family whose original domain
    is intentionally disjoint; load ledgers should keep the default.
    """

    def __init__(
        self,
        states: Mapping[Hashable, StateValue],
        segments: Iterable[StateSegment],
        *,
        validate_contiguous: bool = True,
    ) -> None:
        if not isinstance(states, Mapping):
            raise TypeError("states must be a mapping")
        grouped: dict[StreamKey, list[StateSegment]] = {}
        state_keys = set()
        for segment in segments:
            if not isinstance(segment, StateSegment):
                raise TypeError("segments must contain StateSegment objects")
            if segment.state_key not in states:
                raise KeyError(f"missing compressed state value: {segment.state_key!r}")
            if segment.state_key in state_keys:
                raise ValueError(f"duplicate segment state_key: {segment.state_key!r}")
            state_keys.add(segment.state_key)
            grouped.setdefault(segment.stream_key, []).append(segment)

        ordered = []
        starts_by_stream = {}
        segments_by_stream = {}
        for stream_key, raw_segments in grouped.items():
            stream_segments = tuple(sorted(raw_segments, key=lambda item: item.start))
            previous = None
            for segment in stream_segments:
                if previous is not None:
                    if segment.start <= previous.end:
                        raise ValueError(
                            f"overlapping segments for {stream_key!r}: "
                            f"{previous.start}..{previous.end} and "
                            f"{segment.start}..{segment.end}"
                        )
                    if validate_contiguous and segment.start != previous.end + 1:
                        raise ValueError(
                            f"gap in segments for {stream_key!r}: "
                            f"{previous.end} -> {segment.start}"
                        )
                previous = segment
            ordered.extend(stream_segments)
            segments_by_stream[stream_key] = stream_segments
            starts_by_stream[stream_key] = tuple(
                segment.start for segment in stream_segments
            )

        self._states = states
        self._segments = tuple(ordered)
        self._segments_by_stream = segments_by_stream
        self._starts_by_stream = starts_by_stream
        self._length = sum(segment.length for segment in self._segments)

    @property
    def segments(self) -> tuple[StateSegment, ...]:
        return self._segments

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        for segment in self._segments:
            yield from segment.minute_keys()

    def segment_for(self, minute_key: tuple) -> StateSegment:
        """Return the segment representing an original minute-level key."""
        if not isinstance(minute_key, tuple) or not minute_key:
            raise KeyError(minute_key)
        try:
            minute = _minute(minute_key[-1], label="minute key")
        except TypeError as exc:
            raise KeyError(minute_key) from exc
        stream_key = minute_key[:-1]
        stream_segments = self._segments_by_stream.get(stream_key)
        if not stream_segments:
            raise KeyError(minute_key)
        starts = self._starts_by_stream[stream_key]
        position = bisect_right(starts, minute) - 1
        if position < 0:
            raise KeyError(minute_key)
        segment = stream_segments[position]
        if minute > segment.end:
            raise KeyError(minute_key)
        return segment

    def __getitem__(self, minute_key: tuple) -> StateValue:
        segment = self.segment_for(minute_key)
        return self._states[segment.state_key]

    def iter_expanded_items(self):
        """Yield ``(minute_key, value)`` pairs without lookup or allocation."""
        for segment in self._segments:
            value = self._states[segment.state_key]
            for minute_key in segment.minute_keys():
                yield minute_key, value

    def iter_expanded_values(self):
        """Yield each segment value once per represented minute."""
        for segment in self._segments:
            value = self._states[segment.state_key]
            for _minute_value in range(segment.start, segment.end + 1):
                yield value

    def items(self):
        return _ExpandedItemsView(self)

    def values(self):
        return _ExpandedValuesView(self)


__all__ = [
    "ExpandedStateView",
    "StateSegment",
    "build_bus_load_segments",
    "build_load_segments",
    "build_train_load_segments",
]
