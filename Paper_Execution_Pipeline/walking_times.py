"""Load and validate the paper's station walking-time input."""
from __future__ import annotations

import csv
from pathlib import Path


def _distance_station_codes(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Origin_Code", "Destination_Code"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return {
            str(value).strip().upper()
            for row in reader
            for value in (row["Origin_Code"], row["Destination_Code"])
            if str(value).strip()
        }


def load_uniform_walking_minutes(
    path: Path,
    distance_path: Path,
    expected_minutes: int,
) -> dict[str, object]:
    """Validate complete station coverage and return the uniform CSV value."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"StationCode", "WalkingMinutes"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        values: dict[str, int] = {}
        for number, row in enumerate(reader, 2):
            station = str(row["StationCode"]).strip().upper()
            if not station:
                raise ValueError(f"{path}:{number}: blank StationCode")
            if station in values:
                raise ValueError(f"{path}:{number}: duplicate station {station}")
            raw_minutes = float(row["WalkingMinutes"])
            minutes = int(raw_minutes)
            if raw_minutes != minutes or minutes < 0:
                raise ValueError(
                    f"{path}:{number}: WalkingMinutes must be a nonnegative integer"
                )
            values[station] = minutes

    expected_stations = _distance_station_codes(distance_path)
    actual_stations = set(values)
    if actual_stations != expected_stations:
        raise ValueError(
            f"{path}: station coverage mismatch; "
            f"missing={sorted(expected_stations - actual_stations)}, "
            f"extra={sorted(actual_stations - expected_stations)}"
        )
    distinct = sorted(set(values.values()))
    if distinct != [expected_minutes]:
        raise ValueError(
            f"{path}: all WalkingMinutes must equal {expected_minutes}; "
            f"found={distinct}"
        )
    return {
        "path": str(path.resolve()),
        "station_count": len(values),
        "walking_minutes": distinct[0],
        "all_rows_uniform": True,
    }
