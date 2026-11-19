"""Network topology read from the input data of the active profile.

No station, depot, corridor or line identifier is written in code. Everything below is
derived from three input files named in ``[inputs]`` of the profile:

* ``inputs.corridors``      track sequences, one column per physical track; the rows give the
                            station order along the track (only the order is used)
* ``inputs.corridor_tracks`` track column -> corridor id -> display label (columns Track, Corridor, CorridorLabel)
* ``inputs.walking_times``  rail-bus transfer (walking) time per station (columns StationCode, WalkingMinutes)

and the rules in ``[network]`` (central group id/label, minimum number of corridors that makes
a station part of the shared central group, expected central group used as a guard, maximum
number of transfers of a passenger path, uniform transfer-time guard).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bbconfig import cfg  # noqa: E402


@lru_cache(maxsize=None)
def corridor_tracks() -> pd.DataFrame:
    """Track -> corridor table in display order."""
    frame = pd.read_csv(cfg.path("inputs.corridor_tracks"), dtype=str).fillna("")
    missing = {"Track", "Corridor", "CorridorLabel"} - set(frame.columns)
    if missing:
        raise ValueError(f"{cfg.path('inputs.corridor_tracks')}: missing columns {sorted(missing)}")
    return frame


def track_to_corridor() -> dict[str, str]:
    return dict(zip(corridor_tracks()["Track"], corridor_tracks()["Corridor"]))


def corridor_order() -> list[str]:
    """Corridor ids in file order (each listed once)."""
    return list(dict.fromkeys(corridor_tracks()["Corridor"]))


def corridor_labels() -> dict[str, str]:
    frame = corridor_tracks().drop_duplicates("Corridor")
    labels = dict(zip(frame["Corridor"], frame["CorridorLabel"]))
    labels[central_group_id()] = cfg.get("network.central_group_label")
    return labels


def central_group_id() -> str:
    return str(cfg.get("network.central_group_id"))


@lru_cache(maxsize=None)
def track_stations() -> dict[str, list[str]]:
    """Ordered station list per track column of the corridor file."""
    frame = pd.read_csv(cfg.path("inputs.corridors"), dtype=str)
    return {
        column: [s.strip() for s in frame[column].dropna().tolist() if str(s).strip()]
        for column in frame.columns
    }


def track_positions() -> dict[str, dict[str, int]]:
    """Track -> {station: position along the track (0-based, file row order)}."""
    return {track: {s: i for i, s in enumerate(stations)}
            for track, stations in track_stations().items()}


def station_tracks() -> dict[str, list[str]]:
    """Station -> tracks (corridor-file columns) containing it, in column order."""
    result: dict[str, list[str]] = defaultdict(list)
    for track, stations in track_stations().items():
        for station in stations:
            result[station].append(track)
    return dict(result)


def station_corridors() -> dict[str, set[str]]:
    """Station -> set of corridor ids whose tracks contain it."""
    mapping = track_to_corridor()
    result: dict[str, set[str]] = defaultdict(set)
    for track, stations in track_stations().items():
        if track not in mapping:
            raise ValueError(f"track column '{track}' has no row in {cfg.get('inputs.corridor_tracks')}")
        for station in stations:
            result[station].add(mapping[track])
    return dict(result)


def central_shared_stations() -> set[str]:
    """Stations served by at least ``network.central_min_corridors`` corridors."""
    threshold = int(cfg.get("network.central_min_corridors"))
    shared = {s for s, corridors in station_corridors().items() if len(corridors) >= threshold}
    expected = cfg.get("network.central_shared_expected", default=None)
    if expected is not None and shared != set(expected):
        raise ValueError(
            f"central shared group derived from the corridor file {sorted(shared)} differs from "
            f"network.central_shared_expected {sorted(expected)}"
        )
    return shared


def station_group(station: str) -> str:
    """Corridor id of a station, or the central group id for shared stations."""
    corridors = station_corridors().get(station)
    if not corridors:
        raise KeyError(f"station '{station}' is not on any track of the corridor file")
    if station in central_shared_stations():
        return central_group_id()
    (only,) = corridors
    return only


def max_transfers() -> int:
    """Maximum number of rail-to-rail transfers of a passenger path in the flow model."""
    return int(cfg.get("network.max_transfers"))


@lru_cache(maxsize=None)
def transfer_minutes() -> dict[str, float]:
    """Rail-bus transfer (walking) time sigma_i per station [min].

    With ``network.uniform_transfer_time`` every value must equal
    ``operations.walking_transfer_min`` (Table 3 / Assumption 4: one constant sigma).
    """
    path = cfg.path("inputs.walking_times")
    frame = pd.read_csv(path)
    values = dict(zip(frame["StationCode"].astype(str).str.strip(),
                      frame["WalkingMinutes"].astype(float)))
    if cfg.get("network.uniform_transfer_time"):
        expected = float(cfg.get("operations.walking_transfer_min"))
        different = sorted(s for s, minutes in values.items() if minutes != expected)
        if different:
            raise ValueError(
                f"{path}: network.uniform_transfer_time is set but WalkingMinutes differs from "
                f"operations.walking_transfer_min={expected:g} at {different}")
    return values


def transfer_steps(stations: Iterable[str], time_step_min: int) -> dict[str, int]:
    """sigma per station in model time steps; every station must have a walking-time row."""
    minutes = transfer_minutes()
    missing = sorted(set(stations) - set(minutes))
    if missing:
        raise KeyError(f"{cfg.get('inputs.walking_times')}: no walking time for stations {missing}")
    result = {}
    for station in stations:
        steps, remainder = divmod(minutes[station], time_step_min)
        if remainder:
            raise ValueError(
                f"walking time {minutes[station]:g} min at {station} is not a multiple of the "
                f"{time_step_min}-min time step")
        result[station] = int(steps)
    return result
