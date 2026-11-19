"""Build the paper's corridor-constrained 30/60 scenario reduction.

This file is intentionally self-contained.  It does not import any module from
the original workspace.  It reads disruption CSVs created by step 01, builds
the historical corridor and duration priors, compares four reduction methods,
and writes the configured selected-K support used by Stage 1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import t as student_t
from scipy.stats import wasserstein_distance_nd
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bbconfig import cfg  # noqa: E402
import network_data  # noqa: E402


BASE = Path(__file__).resolve().parent
DEFAULT_SCENARIO_DIR = cfg.results_dir / cfg.get("paths.disruptions")
DEFAULT_OUTPUT = cfg.results_dir / cfg.get("paths.clustering_comparison")
DEFAULT_PRODUCTION_KMEANS_DIR = cfg.results_dir / cfg.get("paths.clustering")
# Guard (paper: Sec. 4.3, 1,058 retained cases); 0 disables the check (synthetic).
EXPECTED_RETAINED_STATION_HOURS = int(cfg.get("clustering.guards.retained_cases"))

WINDOWS = tuple(int(value) for value in cfg.get("time.windows"))
DURATIONS = tuple(int(value) for value in cfg.get("time.durations_min"))
HOURS = WINDOWS
SELECTED_K = int(cfg.get("clustering.k_per_window"))
CLUSTERING_METHOD = str(cfg.get("clustering.method"))
STATION_SMOOTHING_ALPHA = float(cfg.get("probability.station_smoothing_beta"))
SCENARIO_LOGIC_VERSION = str(cfg.get("scenarios.logic_version"))
DURATION_CUTOFF = int(cfg.get("probability.duration_cutoff_min"))
# Cohort prior (Table 3: pi_30 = 208/303, pi_60 = 95/303).
COHORT_DURATION_COUNTS = {
    int(key): int(value) for key, value in cfg.get("probability.duration_counts").items()
}
DURATION_WEIGHT = {
    duration: COHORT_DURATION_COUNTS[duration] / sum(COHORT_DURATION_COUNTS.values())
    for duration in DURATIONS
}
REGULAR_INCIDENT_GUARD = int(cfg.get("clustering.guards.regular_incidents"))
REGULAR_TRIP_PATTERN = str(cfg.get("prediction.cohort.regular_trip_pattern"))
IMPACT_VARIABLES = tuple(cfg.get("clustering.reduction.impact_variables"))
K_VALUES = tuple(int(value) for value in cfg.get("clustering.reduction.k_candidates"))
GLOBAL_FFS_PROFILE_K_VALUES = tuple(
    int(value) for value in cfg.get("clustering.reduction.global_ffs_profile_k")
)
SEED = int(cfg.get("clustering.reduction.seed"))
N_INIT = int(cfg.get("clustering.reduction.n_init"))

# Display names of the four implemented methods, in the fixed order
# K-means, K-medoids, hierarchical clustering, fast-forward reduction.
METHODS = tuple(str(value) for value in cfg.get("clustering.reduction.methods"))
if len(METHODS) != 4:
    raise ValueError("clustering.reduction.methods must name the four implemented methods")


def corridor_order_all() -> list[str]:
    """Central group first, then the corridors in the order of inputs.corridor_tracks."""
    return [network_data.central_group_id()] + [
        corridor for corridor in network_data.corridor_order()
        if corridor != network_data.central_group_id()
    ]


def output_name(key: str) -> str:
    """File name of a reduction output, built from [clustering.files]."""
    return str(cfg.get(f"clustering.files.{key}")).format(
        k=SELECTED_K, durations="_".join(str(value) for value in DURATIONS)
    )


def portable(path) -> str:
    """Path for audit files without machine-specific prefixes (<results>/..., <data>/..., or the name)."""
    path = Path(path).resolve()
    for tag, base in (("<results>", cfg.results_dir), ("<data>", cfg.data_dir), ("<package>", BASE.parent)):
        try:
            return f"{tag}/{path.relative_to(Path(base).resolve()).as_posix()}"
        except ValueError:
            continue
    return path.name


CORRIDOR_ORDER = tuple(corridor_order_all())
SCENARIO_PATTERN = re.compile(
    r"^Disrupt_(?P<hour>\d+)_(?P<station>.+)_(?P<duration>\d+)\.csv$"
)
SCENARIO_ID_PATTERN = re.compile(
    r"^(?P<hour>\d+)_(?P<station>.+)_(?P<duration>\d+)$"
)
FULL_SCENARIO_COLUMNS = {
    "Variable", "Omega", "Origin", "Dest", "Time", "StationID",
    "TrainID", "Value",
}

GRID_COLOR = "#C7CCD1"
GRID_LINESTYLE = "--"
GRID_LINEWIDTH = 0.8
LEGEND_EDGE_COLOR = "#BFBFBF"


def require_file(path: Path, description: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def build_corridor_map() -> tuple[dict[str, str], dict[str, list[str]]]:
    """Station -> corridor id (central group for shared stations) from network_data."""
    central = network_data.central_shared_stations()  # checked against network.central_shared_expected
    station_corridor: dict[str, str] = {}
    station_families: dict[str, list[str]] = {}
    for station, corridors in network_data.station_corridors().items():
        code = str(station).strip().upper()
        station_families[code] = sorted(corridors)
        station_corridor[code] = (
            network_data.central_group_id() if station in central
            else network_data.station_group(station)
        )
    return station_corridor, station_families


def incident_corridor_labels(
    stations: pd.Series, labels: pd.Series, station_corridor: dict[str, str]
) -> dict[str, str]:
    """Incident-file corridor label -> corridor id, learned from incidents whose
    station lies on the modelled network (the relation must be one-to-one)."""
    known = stations.isin(station_corridor)
    pairs = pd.DataFrame({
        "label": labels[known].astype(str).str.strip(),
        "corridor": stations[known].map(station_corridor),
    }).drop_duplicates()
    if pairs["label"].duplicated().any() or pairs["corridor"].duplicated().any():
        raise ValueError(
            "incident corridor labels are not one-to-one with the network corridors:\n"
            + pairs.sort_values(["label", "corridor"]).to_string(index=False)
        )
    return dict(zip(pairs["label"], pairs["corridor"]))


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
        try:
            return [value.strip() for value in next(csv.reader(handle))]
        except StopIteration as exc:
            raise RuntimeError(f"empty disruption file: {path}") from exc


def discover_scenarios(
    scenario_dir: Path,
    read_workers: int,
    location_support: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    scenario_dir = scenario_dir.resolve()
    if not scenario_dir.is_dir():
        raise FileNotFoundError(f"scenario directory not found: {scenario_dir}")

    rows: list[dict] = []
    all_counts = {(hour, duration): 0 for hour in HOURS for duration in DURATIONS}
    positive_counts = {
        (hour, duration): 0 for hour in HOURS for duration in DURATIONS
    }
    no_affected_counts = {
        (hour, duration): 0 for hour in HOURS for duration in DURATIONS
    }
    candidates = []
    for path in sorted(scenario_dir.glob("Disrupt_*.csv")):
        match = SCENARIO_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        hour = int(match.group("hour"))
        duration = int(match.group("duration"))
        if hour not in HOURS or duration not in DURATIONS:
            continue
        station = match.group("station").strip().upper()
        if not station:
            raise RuntimeError(f"empty station code in {path.name}")
        candidates.append((path, hour, station, duration))
    available = {
        f"{hour}_{station}_{duration}": path
        for path, hour, station, duration in candidates
    }

    manifest_path = scenario_dir / "_scenario_stats.csv"
    support_source = "file_headers"
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path, low_memory=False)
        required = {"scenario", "checkpoint"}
        missing = required.difference(manifest.columns)
        if missing:
            raise RuntimeError(
                f"scenario manifest misses columns: {sorted(missing)}"
            )
        latest = manifest.drop_duplicates("scenario", keep="last")
        classified = []
        for row in latest.itertuples(index=False):
            scenario_id = str(row.scenario).strip()
            match = SCENARIO_ID_PATTERN.fullmatch(scenario_id)
            if match is None:
                continue
            hour = int(match.group("hour"))
            duration = int(match.group("duration"))
            if hour not in HOURS or duration not in DURATIONS:
                continue
            station = match.group("station").strip().upper()
            path = available.get(f"{hour}_{station}_{duration}")
            if path is None:
                raise FileNotFoundError(
                    scenario_dir / f"Disrupt_{hour}_{station}_{duration}.csv"
                )
            checkpoint = str(row.checkpoint).strip()
            if checkpoint not in {"EXTRACTED", "NO_AFFECTED"}:
                raise RuntimeError(
                    f"incomplete scenario {scenario_id}: checkpoint={checkpoint}"
                )
            if "logic_version" in manifest.columns:
                logic_version = str(getattr(row, "logic_version", "")).strip()
                if logic_version != SCENARIO_LOGIC_VERSION:
                    raise RuntimeError(
                        f"stale scenario {scenario_id}: "
                        f"logic_version={logic_version!r}"
                    )
            classified.append((path, hour, station, duration, checkpoint))
        support_source = "_scenario_stats.csv"
    else:
        def classify(candidate):
            path, hour, station, duration = candidate
            header = set(_read_header(path))
            if FULL_SCENARIO_COLUMNS.issubset(header):
                checkpoint = "EXTRACTED"
            elif "Info" in header:
                checkpoint = "NO_AFFECTED"
            else:
                raise RuntimeError(
                    f"unrecognized disruption schema in {path}: {sorted(header)}"
                )
            return path, hour, station, duration, checkpoint

        with ThreadPoolExecutor(max_workers=read_workers) as executor:
            classified = list(executor.map(classify, candidates))

    valid_rows: list[dict] = []
    for path, hour, station, duration, checkpoint in classified:
        key = (hour, duration)
        all_counts[key] += 1
        valid_rows.append({
            "window": hour,
            "duration": duration,
            "station": station,
            "scenario_id": f"{hour}_{station}_{duration}",
            "path": path,
            "branch_status": checkpoint,
        })
        if checkpoint == "EXTRACTED":
            positive_counts[key] += 1
            rows.append({
                "window": hour,
                "duration": duration,
                "station": station,
                "scenario_id": f"{hour}_{station}_{duration}",
                "path": path,
            })
        elif checkpoint == "NO_AFFECTED":
            no_affected_counts[key] += 1
        else:
            raise AssertionError(f"unexpected branch status: {checkpoint}")

    missing_file_blocks = [key for key, count in all_counts.items() if count == 0]
    if missing_file_blocks:
        raise RuntimeError(f"scenario blocks have no files: {missing_file_blocks}")

    support = pd.DataFrame(rows).sort_values(
        ["window", "duration", "station"]
    ).reset_index(drop=True)
    valid_support = pd.DataFrame(valid_rows).sort_values(
        ["window", "duration", "station"]
    ).reset_index(drop=True)
    if support.duplicated(["window", "duration", "station"]).any():
        duplicate = support.loc[
            support.duplicated(["window", "duration", "station"], keep=False),
            ["window", "duration", "station"],
        ]
        raise RuntimeError(
            "duplicate disruption scenarios: "
            f"{duplicate.head(10).to_dict('records')}"
        )
    if valid_support.duplicated(["window", "duration", "station"]).any():
        duplicate = valid_support.loc[
            valid_support.duplicated(
                ["window", "duration", "station"], keep=False,
            ),
            ["window", "duration", "station", "branch_status"],
        ]
        raise RuntimeError(
            "duplicate valid duration branches: "
            f"{duplicate.head(10).to_dict('records')}"
        )

    complete_by_hour = {}
    any_duration_by_hour = {}
    valid_station_by_hour = {}
    complete_positive_by_hour = {}
    for hour in HOURS:
        positive = {
            duration: set(support.loc[
                support["window"].eq(hour)
                & support["duration"].eq(duration),
                "station",
            ])
            for duration in DURATIONS
        }
        valid = {
            duration: set(valid_support.loc[
                valid_support["window"].eq(hour)
                & valid_support["duration"].eq(duration),
                "station",
            ])
            for duration in DURATIONS
        }
        positive_complete = set.intersection(*positive.values())
        any_duration = set.union(*positive.values())
        valid_complete = set.intersection(*valid.values())
        valid_any = set.union(*valid.values())
        eligible = any_duration & (
            valid_complete
            if location_support == "all-duration" else valid_any
        )
        complete_by_hour[str(hour)] = len(eligible)
        any_duration_by_hour[str(hour)] = len(any_duration)
        valid_station_by_hour[str(hour)] = len(valid_complete)
        complete_positive_by_hour[str(hour)] = len(positive_complete)
        if len(eligible) < SELECTED_K:
            raise RuntimeError(
                f"hour {hour}: only {len(eligible)} stations satisfy "
                f"location-support={location_support}; "
                f"K={SELECTED_K} is infeasible"
            )

    audit = {
        "scenario_directory": portable(scenario_dir),
        "support_source": support_source,
        "recognized_files": int(sum(all_counts.values())),
        "valid_scenarios": int(len(valid_support)),
        "positive_scenarios": int(len(support)),
        "no_affected_files": int(sum(no_affected_counts.values())),
        "positive_by_hour_duration": {
            f"{hour}_{duration}": int(positive_counts[(hour, duration)])
            for hour in HOURS for duration in DURATIONS
        },
        "complete_station_count_by_hour": complete_by_hour,
        "valid_all_duration_station_count_by_hour": valid_station_by_hour,
        "positive_all_duration_station_count_by_hour": complete_positive_by_hour,
        "any_duration_impact_station_count_by_hour": any_duration_by_hour,
        "clustering_observation_rule": (
            "station-hour with EXTRACTED impact in at least one duration"
        ),
        "valid_duration_branch_statuses": ["EXTRACTED", "NO_AFFECTED"],
        "no_affected_feature_value": "zero vector",
        "location_support_rule": location_support,
    }
    return support, valid_support, audit


def load_history(
    augmented_path: Path,
    raw_path: Path,
    cohort_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    augmented = pd.read_csv(augmented_path, low_memory=False)
    required_augmented = {
        "IncidentID", "Station", "StationCorridor", "MaxDelayMinutes"
    }
    missing = required_augmented.difference(augmented.columns)
    if missing:
        raise ValueError(f"augmented incident file misses: {sorted(missing)}")
    if augmented["IncidentID"].duplicated().any():
        raise ValueError("augmented incident file has duplicate IncidentID values")

    raw = pd.read_csv(
        raw_path,
        usecols=["ItemId", "TripId", "PrimaryServiceForItem"],
        low_memory=False,
    )
    raw["IncidentID"] = pd.to_numeric(raw["ItemId"], errors="coerce").astype(
        "Int64"
    )
    primary = raw[
        raw["IncidentID"].notna()
        & pd.to_numeric(raw["PrimaryServiceForItem"], errors="coerce").eq(1)
    ].copy()
    primary["IncidentID"] = primary["IncidentID"].astype(int)
    primary["IsRegular8xxx"] = (
        primary["TripId"].astype(str).str.strip().str.fullmatch(REGULAR_TRIP_PATTERN)
    )
    eligibility = primary.groupby("IncidentID")["IsRegular8xxx"].all()
    selected_ids = set(eligibility[eligibility].index.astype(int))

    augmented = augmented.copy()
    augmented["IncidentID"] = pd.to_numeric(
        augmented["IncidentID"], errors="raise").astype(int)
    history = augmented[augmented["IncidentID"].isin(selected_ids)].copy()
    regular_8xxx_rows = int(len(history))
    if REGULAR_INCIDENT_GUARD and regular_8xxx_rows != REGULAR_INCIDENT_GUARD:
        raise ValueError(
            f"expected {REGULAR_INCIDENT_GUARD} regular incidents "
            f"(clustering.guards.regular_incidents), observed {len(history)}"
        )

    cohort_audit = {
        "cohort_source": "all_regular_8xxx",
        "cohort_file": None,
        "cohort_rows": regular_8xxx_rows,
    }
    cohort_states = None
    if cohort_path is not None:
        cohort = pd.read_csv(cohort_path, low_memory=False)
        required_cohort = {"IncidentID", "Eligible", "RealizedState"}
        missing_cohort = required_cohort.difference(cohort.columns)
        if missing_cohort:
            raise ValueError(
                f"operational cohort file misses: {sorted(missing_cohort)}"
            )
        cohort = cohort.copy()
        cohort["IncidentID"] = pd.to_numeric(
            cohort["IncidentID"], errors="raise").astype(int)
        if cohort["IncidentID"].duplicated().any():
            raise ValueError("operational cohort has duplicate IncidentID values")
        if pd.api.types.is_bool_dtype(cohort["Eligible"]):
            eligible = cohort["Eligible"].fillna(False)
        else:
            eligible = (
                cohort["Eligible"].astype(str).str.strip().str.lower()
                .isin({"1", "true", "yes"})
            )
        cohort = cohort.loc[eligible].copy()
        cohort_ids = set(cohort["IncidentID"])
        missing_ids = sorted(cohort_ids - set(history["IncidentID"]))
        if missing_ids:
            raise ValueError(
                "operational cohort contains incidents outside the regular "
                f"8xxx population: {missing_ids[:10]}"
            )
        history = history[history["IncidentID"].isin(cohort_ids)].copy()
        if len(history) != len(cohort_ids):
            raise AssertionError("operational cohort filtering lost incidents")
        cohort_states = (
            cohort.set_index("IncidentID")["RealizedState"]
            .pipe(pd.to_numeric, errors="raise").astype(int)
        )
        cohort_audit = {
            "cohort_source": "eligible_operational_cases",
            "cohort_file": portable(cohort_path),
            "cohort_file_rows": int(len(eligible)),
            "cohort_eligible_rows": int(eligible.sum()),
            "cohort_rows": int(len(history)),
        }
    history["station"] = history["Station"].astype(str).str.strip().str.upper()
    label_map = incident_corridor_labels(
        history["station"], history["StationCorridor"], build_corridor_map()[0]
    )
    history["corridor_hint"] = (
        history["StationCorridor"].astype("string").str.strip().map(label_map)
    )
    if history["corridor_hint"].isna().any():
        values = sorted(
            history.loc[
                history["corridor_hint"].isna(), "StationCorridor"
            ].astype(str).unique()
        )
        raise ValueError(f"unrecognized historical corridor labels: {values}")
    maximum_delay = pd.to_numeric(history["MaxDelayMinutes"], errors="raise")
    # Sec. 4.4: long (last duration state) iff MaxDelayMinutes >= the duration cutoff.
    history["duration_state"] = np.where(
        maximum_delay.ge(DURATION_CUTOFF), DURATIONS[-1], DURATIONS[0]
    )
    if cohort_states is not None:
        derived = history.set_index("IncidentID")["duration_state"].astype(int)
        expected = cohort_states.reindex(derived.index)
        mismatch = derived.ne(expected)
        if mismatch.any():
            bad = pd.DataFrame({
                "derived": derived[mismatch],
                "cohort": expected[mismatch],
            })
            raise ValueError(
                "operational cohort duration state disagrees with the "
                f"clustering history:\n{bad.head(10)}"
            )
    observed = {
        duration: int(history["duration_state"].eq(duration).sum())
        for duration in DURATIONS
    }
    return history[
        ["station", "corridor_hint", "duration_state"]
    ].reset_index(drop=True), {
        "augmented_rows": int(len(augmented)),
        "regular_8xxx_rows": regular_8xxx_rows,
        **cohort_audit,
        "duration_counts": {str(key): value for key, value in observed.items()},
        "incident_corridor_labels": label_map,
    }


def historical_corridor_probability(
    history: pd.DataFrame,
    station_corridor: dict[str, str],
) -> tuple[dict[str, float], pd.DataFrame, int, dict[str, int], int]:
    # Corridor is available for every retained incident. Station frequencies
    # use the subset whose station maps to the modeled network.
    station_known = history["station"].isin(station_corridor)
    mapped = history.loc[station_known, "station"].map(station_corridor)
    hinted = history.loc[station_known, "corridor_hint"]
    if not mapped.reset_index(drop=True).equals(hinted.reset_index(drop=True)):
        mismatch = history.loc[
            station_known & history["station"].map(station_corridor).ne(
                history["corridor_hint"]
            ),
            ["station", "corridor_hint"],
        ]
        raise RuntimeError(
            "station and historical corridor disagree:\n"
            + mismatch.to_string(index=False)
        )
    corridor_total = int(len(history))
    station_total = int(station_known.sum())
    if corridor_total <= 0 or station_total <= 0:
        raise RuntimeError(
            "historical location support is empty: "
            f"corridor={corridor_total}, station={station_total}"
        )
    counts = history["corridor_hint"].value_counts()
    probabilities = {
        corridor: float(counts.get(corridor, 0)) / corridor_total
        for corridor in CORRIDOR_ORDER
    }
    if not np.isclose(sum(probabilities.values()), 1.0, atol=1e-12):
        raise AssertionError("historical corridor probabilities do not sum to one")
    table = pd.DataFrame([
        {
            "corridor": corridor,
            "HistoricalCount": int(counts.get(corridor, 0)),
            "HistoricalProbability": probabilities[corridor],
        }
        for corridor in CORRIDOR_ORDER
    ])
    station_counts = {
        str(station): int(count)
        for station, count in history.loc[
            station_known, "station"
        ].value_counts().items()
    }
    if sum(station_counts.values()) != station_total:
        raise AssertionError(
            "historical station counts do not sum to the retained support"
        )
    return (
        probabilities, table, corridor_total, station_counts, station_total
    )


def read_impact(row) -> dict[str, float]:
    path = Path(row.path)
    frame = pd.read_csv(
        path,
        usecols=["Variable", "StationID", "Value"],
        low_memory=False,
    )
    spatial = frame[
        frame["Variable"].isin(IMPACT_VARIABLES)
    ].dropna(subset=["StationID"])
    if spatial.empty:
        return {}
    spatial = spatial.copy()
    spatial["StationID"] = (
        spatial["StationID"].astype(str).str.strip().str.upper()
    )
    spatial["Value"] = pd.to_numeric(spatial["Value"], errors="raise")
    result = spatial.groupby("StationID")["Value"].sum().astype(float).to_dict()
    if not np.isfinite(np.asarray(list(result.values()), dtype=float)).all():
        raise ValueError(f"non-finite impact value in {path}")
    return result


def load_impact_features(
    support: pd.DataFrame,
    read_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    if read_workers <= 0:
        raise ValueError("--read-workers must be positive")
    rows = tuple(support.itertuples(index=False))
    impacts: list[dict[str, float]] = []
    feature_names: set[str] = set()
    with ThreadPoolExecutor(max_workers=read_workers) as executor:
        for index, impact in enumerate(executor.map(read_impact, rows), start=1):
            impacts.append(impact)
            feature_names.update(impact)
            if index % 250 == 0 or index == len(rows):
                print(
                    f"Read {index:,}/{len(rows):,} disruption files",
                    flush=True,
                )
    features = np.asarray(sorted(feature_names), dtype=object)
    if not len(features):
        raise RuntimeError("all Queue/Walking impact vectors are empty")
    feature_index = {str(station): index for index, station in enumerate(features)}
    raw = np.zeros((len(support), len(features)), dtype=float)
    for row_index, impact in enumerate(impacts):
        for station, value in impact.items():
            raw[row_index, feature_index[station]] = float(value)
    if not np.isfinite(raw).all():
        raise ValueError("impact feature matrix contains non-finite values")
    if (raw < -1.0).any():
        raise ValueError("impact feature matrix contains values below -1")
    return features, raw


def build_hour(
    support: pd.DataFrame,
    valid_support: pd.DataFrame,
    raw: np.ndarray,
    hour: int,
    location_support: str,
) -> dict:
    selected = support[
        support["window"].eq(hour) & support["duration"].isin(DURATIONS)
    ]
    stations = sorted(selected["station"].astype(str).unique())
    station_index = {station: index for index, station in enumerate(stations)}
    n = len(stations)
    duration_features: dict[int, np.ndarray] = {}
    positive_impact: dict[int, np.ndarray] = {}
    valid_branch: dict[int, np.ndarray] = {}
    branch_status: dict[int, np.ndarray] = {}

    for duration in DURATIONS:
        mask = (
            support["window"].eq(hour)
            & support["duration"].eq(duration)
        ).to_numpy()
        global_rows = np.flatnonzero(mask)
        frame = support.iloc[global_rows].reset_index(drop=True)
        block = np.zeros((n, raw.shape[1]), dtype=float)
        present = np.zeros(n, dtype=bool)
        for local_row, row in enumerate(frame.itertuples(index=False)):
            position = station_index[str(row.station)]
            block[position] = raw[global_rows[local_row]]
            present[position] = True
        duration_features[duration] = np.log1p(block)
        positive_impact[duration] = present

        valid_frame = valid_support[
            valid_support["window"].eq(hour)
            & valid_support["duration"].eq(duration)
            & valid_support["station"].isin(stations)
        ]
        valid = np.zeros(n, dtype=bool)
        status = np.full(n, "MISSING", dtype=object)
        for row in valid_frame.itertuples(index=False):
            position = station_index[str(row.station)]
            valid[position] = True
            status[position] = str(row.branch_status)
        if not np.array_equal(present, status == "EXTRACTED"):
            raise AssertionError(
                f"hour {hour}, duration {duration}: positive-impact support "
                "does not match EXTRACTED valid-branch status"
            )
        valid_branch[duration] = valid
        branch_status[duration] = status

    all_duration_valid = np.logical_and.reduce([
        valid_branch[value] for value in DURATIONS
    ])
    any_duration_valid = np.logical_or.reduce([
        valid_branch[value] for value in DURATIONS
    ])
    all_duration_positive = np.logical_and.reduce([
        positive_impact[value] for value in DURATIONS
    ])
    any_duration_positive = np.logical_or.reduce([
        positive_impact[value] for value in DURATIONS
    ])
    representative_eligible = (
        all_duration_valid
        if location_support == "all-duration" else any_duration_valid
    )
    if int(representative_eligible.sum()) < SELECTED_K:
        raise RuntimeError(
            f"hour {hour}: fewer than {SELECTED_K} candidates satisfy "
            f"location-support={location_support}"
        )
    return {
        "hour": int(hour),
        "stations": stations,
        "duration_log_features": duration_features,
        "valid_branch": valid_branch,
        "branch_status": branch_status,
        "positive_impact": positive_impact,
        "complete": representative_eligible,
        "all_duration_valid": all_duration_valid,
        "any_duration_valid": any_duration_valid,
        "all_duration_positive_impact": all_duration_positive,
        "any_duration_positive_impact": any_duration_positive,
        "location_support": location_support,
        "duration_weights": dict(DURATION_WEIGHT),
    }


def apply_corridor_probability(
    data: dict,
    station_corridor: dict[str, str],
    corridor_probability: dict[str, float],
    station_counts: dict[str, int],
    smoothing_alpha: float,
) -> list[dict]:
    if smoothing_alpha < 0:
        raise ValueError("station smoothing alpha must be nonnegative")
    corridors = np.asarray(
        [station_corridor[station] for station in data["stations"]],
        dtype=object,
    )
    active = [
        corridor for corridor in CORRIDOR_ORDER
        if np.any(corridors == corridor)
    ]
    active_mass = float(sum(corridor_probability[value] for value in active))
    if active_mass <= 0:
        raise RuntimeError(f"hour {data['hour']}: active corridor mass is zero")
    station_probability = np.zeros(len(corridors), dtype=float)
    rows = []
    for corridor in active:
        members = np.flatnonzero(corridors == corridor)
        conditional_corridor = corridor_probability[corridor] / active_mass
        counts = np.asarray([
            station_counts.get(str(data["stations"][position]), 0)
            for position in members
        ], dtype=float)
        smoothed = counts + smoothing_alpha
        smoothed_total = float(smoothed.sum())
        if smoothed_total <= 0:
            raise ValueError(
                f"corridor {corridor} has no historical station count; "
                "a positive station smoothing value is required"
            )
        station_conditional = smoothed / smoothed_total
        station_probability[members] = (
            conditional_corridor * station_conditional
        )
        if not np.isclose(
            station_probability[members].sum(),
            conditional_corridor,
            atol=1e-12,
        ):
            raise AssertionError(
                f"hour {data['hour']} corridor {corridor}: mass changed"
            )
        for position, count, weight, conditional in zip(
            members, counts, smoothed, station_conditional
        ):
            rows.append({
                "hour": int(data["hour"]),
                "corridor": corridor,
                "station": str(data["stations"][position]),
                "HistoricalProbability": corridor_probability[corridor],
                "ActiveHourProbability": conditional_corridor,
                "ActiveStationCount": int(len(members)),
                "HistoricalStationCount": int(count),
                "SmoothingAlpha": float(smoothing_alpha),
                "SmoothedStationWeight": float(weight),
                "StationConditionalProbability": float(conditional),
                "StationProbability": float(
                    station_probability[position]
                ),
            })
    if not np.isclose(station_probability.sum(), 1.0, atol=1e-12):
        raise AssertionError("hourly station probabilities do not sum to one")

    data["corridor"] = corridors
    data["station_probability"] = station_probability
    data["conditional_probability"] = {
        duration: station_probability.copy() for duration in DURATIONS
    }
    data["joint_probability"] = {
        duration: DURATION_WEIGHT[duration] * station_probability
        for duration in DURATIONS
    }
    data["x"] = np.concatenate(
        [
            np.sqrt(DURATION_WEIGHT[duration])
            * data["duration_log_features"][duration]
            for duration in DURATIONS
        ],
        axis=1,
    )
    return rows


def force_membership(distance: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    labels = np.argmin(distance[:, medoids], axis=1).astype(int)
    labels[medoids] = np.arange(len(medoids), dtype=int)
    return labels


def fit_weighted_kmedoids(
    distance: np.ndarray,
    weights: np.ndarray,
    k: int,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    selected: list[int] = []
    remaining = set(range(len(distance)))
    nearest = np.full(len(distance), np.inf)
    for _ in range(k):
        best = min(
            remaining,
            key=lambda candidate: (
                float(np.dot(
                    weights,
                    np.minimum(nearest, distance[:, candidate]),
                )),
                int(candidate),
            ),
        )
        selected.append(int(best))
        remaining.remove(best)
        nearest = np.minimum(nearest, distance[:, best])

    medoids = np.asarray(selected, dtype=int)
    seen: set[tuple[int, ...]] = set()
    for _ in range(max_iter):
        signature = tuple(map(int, medoids))
        if signature in seen:
            break
        seen.add(signature)
        labels = force_membership(distance, medoids)
        updated = medoids.copy()
        for cluster in range(k):
            members = np.flatnonzero(labels == cluster)
            local = distance[np.ix_(members, members)]
            costs = (weights[members, None] * local).sum(axis=0)
            updated[cluster] = members[int(np.argmin(costs))]
        if np.array_equal(updated, medoids):
            break
        medoids = updated
    return medoids, force_membership(distance, medoids)


def weighted_silhouette(
    distance: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    baseline_groups: np.ndarray | None = None,
) -> float:
    labels = np.asarray(labels, dtype=int)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    if baseline_groups is None:
        baseline_groups = np.zeros(len(labels), dtype=int)
    else:
        baseline_groups = np.asarray(baseline_groups, dtype=object)
        if len(baseline_groups) != len(labels):
            raise ValueError("baseline_groups must align with observations")
    values = np.zeros(len(labels), dtype=float)
    for index in range(len(labels)):
        eligible = baseline_groups == baseline_groups[index]
        own = labels[index]
        own_members = np.flatnonzero(eligible & (labels == own))
        own_members = own_members[own_members != index]
        if not len(own_members):
            continue
        own_mass = float(weights[own_members].sum())
        if own_mass <= 1e-15:
            continue
        a = float(
            np.dot(weights[own_members], distance[index, own_members])
            / own_mass
        )
        alternatives = []
        for cluster in np.unique(labels[eligible]):
            if cluster == own:
                continue
            members = np.flatnonzero(eligible & (labels == cluster))
            mass = float(weights[members].sum())
            if mass > 1e-15:
                alternatives.append(float(
                    np.dot(weights[members], distance[index, members]) / mass
                ))
        if not alternatives:
            continue
        b = min(alternatives)
        denominator = max(a, b)
        values[index] = (
            (b - a) / denominator if denominator > 1e-12 else 0.0
        )
    return float(np.dot(weights, values))


def unweighted_silhouette(
    distance: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Return the ordinary observation-level mean silhouette coefficient."""
    return weighted_silhouette(
        distance,
        labels,
        np.ones(len(labels), dtype=float),
    )


def unweighted_sse(x: np.ndarray, labels: np.ndarray) -> float:
    """Return raw within-cluster SSE with every observation weighted equally."""
    within = 0.0
    for cluster in np.unique(labels):
        members = np.flatnonzero(labels == cluster)
        center = x[members].mean(axis=0)
        within += float(np.square(x[members] - center).sum())
    return within


def weighted_sse(
    x: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Return within-cluster SSE without renormalizing global probability."""
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(x) or (weights < 0).any():
        raise ValueError("weights must be nonnegative and align with x")
    within = 0.0
    for cluster in np.unique(labels):
        members = np.flatnonzero(labels == cluster)
        local_weights = weights[members]
        if float(local_weights.sum()) <= 0:
            raise ValueError(f"cluster {cluster} has zero probability mass")
        center = np.average(x[members], axis=0, weights=local_weights)
        within += float(
            (local_weights[:, None] * np.square(x[members] - center)).sum()
        )
    return within


def weighted_sse_r2(
    x: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    baseline_groups: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Return weighted SSE and R² against the eligible baseline partition.

    For corridor-constrained clustering, ``baseline_groups`` contains the
    corridor labels. Total dispersion is then measured around each corridor
    centroid, excluding between-corridor variation that clustering cannot
    explain.
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    if baseline_groups is None:
        overall = np.average(x, axis=0, weights=weights)
        total = float((weights[:, None] * np.square(x - overall)).sum())
    else:
        baseline_groups = np.asarray(baseline_groups, dtype=object)
        if len(baseline_groups) != len(x):
            raise ValueError("baseline_groups must align with observations")
        total = 0.0
        for group in np.unique(baseline_groups):
            members = np.flatnonzero(baseline_groups == group)
            local_weights = weights[members]
            center = np.average(x[members], axis=0, weights=local_weights)
            total += float(
                (
                    local_weights[:, None]
                    * np.square(x[members] - center)
                ).sum()
            )
    within = 0.0
    for cluster in np.unique(labels):
        members = np.flatnonzero(labels == cluster)
        local_weights = weights[members]
        center = np.average(x[members], axis=0, weights=local_weights)
        within += float(
            (local_weights[:, None] * np.square(x[members] - center)).sum()
        )
    r2 = float(1.0 - within / total) if total > 1e-12 else 1.0
    return within, total, r2


def weighted_cluster_centers(
    x: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    k: int,
) -> np.ndarray:
    centers = np.zeros((k, x.shape[1]), dtype=float)
    for cluster in range(k):
        members = np.flatnonzero(labels == cluster)
        if not len(members):
            raise AssertionError(f"cluster {cluster} is empty")
        centers[cluster] = np.average(
            x[members], axis=0, weights=weights[members]
        )
    return centers


def fit_weighted_ward(
    x: np.ndarray,
    weights: np.ndarray,
    k_values: tuple[int, ...],
) -> dict[int, np.ndarray]:
    x = np.asarray(x, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    n, dimension = x.shape
    targets = {int(k) for k in k_values}
    if min(targets) < 1 or max(targets) > n:
        raise ValueError("weighted Ward target K is outside 1..N")

    centers = np.zeros((2 * n - 1, dimension), dtype=float)
    masses = np.zeros(2 * n - 1, dtype=float)
    centers[:n] = x
    masses[:n] = weights
    members: dict[int, np.ndarray] = {
        index: np.asarray([index], dtype=int) for index in range(n)
    }
    active = list(range(n))
    labels_by_k: dict[int, np.ndarray] = {}

    def save_cut() -> None:
        cluster_count = len(active)
        if cluster_count not in targets:
            return
        labels = np.empty(n, dtype=int)
        for label, node in enumerate(active):
            labels[members[node]] = label
        labels_by_k[cluster_count] = labels

    save_cut()
    next_node = n
    while len(active) > min(targets):
        ids = np.asarray(active, dtype=int)
        active_centers = centers[ids]
        active_masses = masses[ids]
        squared_distance = cdist(
            active_centers, active_centers, metric="sqeuclidean"
        )
        factor = (
            active_masses[:, None] * active_masses[None, :]
            / (active_masses[:, None] + active_masses[None, :])
        )
        merge_cost = factor * squared_distance
        merge_cost[np.tril_indices_from(merge_cost)] = np.inf
        row, column = np.unravel_index(
            int(np.argmin(merge_cost)), merge_cost.shape
        )
        left, right = int(ids[row]), int(ids[column])
        combined_mass = masses[left] + masses[right]
        masses[next_node] = combined_mass
        centers[next_node] = (
            masses[left] * centers[left] + masses[right] * centers[right]
        ) / combined_mass
        members[next_node] = np.concatenate([members[left], members[right]])
        active = [node for node in active if node not in (left, right)]
        active.append(next_node)
        next_node += 1
        save_cut()

    missing = targets.difference(labels_by_k)
    if missing:
        raise AssertionError(f"weighted Ward cuts missing for K={sorted(missing)}")
    return labels_by_k


def centroid_representatives(
    data: dict,
    labels: np.ndarray,
    centers: np.ndarray,
    k: int,
) -> np.ndarray:
    representatives = np.empty(k, dtype=int)
    for cluster in range(k):
        candidates = np.flatnonzero(
            (labels == cluster) & np.asarray(data["complete"], dtype=bool)
        )
        if not len(candidates):
            raise RuntimeError(
                f"cluster {cluster} has no valid in-cluster representative"
            )
        distance = np.linalg.norm(
            data["x"][candidates] - centers[cluster], axis=1
        )
        representatives[cluster] = int(candidates[int(np.argmin(distance))])
    if not np.array_equal(
        labels[representatives], np.arange(k, dtype=int)
    ):
        raise AssertionError("representative is not a member of its cluster")
    return representatives


def cost_matrix(x: np.ndarray, p: int = 1) -> np.ndarray:
    """Return the canonical p=1 FFS cost or the p>1 FM cost."""
    x = np.asarray(x, dtype=float)
    difference = x[:, None, :] - x[None, :, :]
    distance = np.sqrt(np.square(difference).sum(axis=-1))
    if p == 1:
        return distance
    center = x.mean(axis=0)
    radius = np.linalg.norm(x - center, axis=1) ** (p - 1)
    multiplier = np.maximum(1.0, np.maximum(radius[:, None], radius[None, :]))
    return multiplier * distance


def redistribute(
    costs: np.ndarray,
    probabilities: np.ndarray,
    kept: np.ndarray,
) -> np.ndarray:
    kept = np.asarray(kept)
    probabilities = np.asarray(probabilities, dtype=float)
    result = probabilities[kept].copy()
    deleted = np.setdiff1d(np.arange(len(probabilities)), kept)
    if deleted.size:
        nearest = np.argmin(costs[np.ix_(deleted, kept)], axis=1)
        np.add.at(result, nearest, probabilities[deleted])
    return result


def fast_forward_selection(
    costs: np.ndarray,
    probabilities: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    costs = np.asarray(costs, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1]:
        raise ValueError("costs must be a square matrix")
    if len(probabilities) != len(costs):
        raise ValueError("probabilities must align with costs")
    if (probabilities < 0).any() or not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite and nonnegative")
    total = costs.shape[0]
    n = int(min(n, total))
    if n >= total:
        return np.arange(total), probabilities.copy(), 0.0

    current = costs.copy()
    remaining = np.arange(total)
    kept: list[int] = []
    for _ in range(n):
        subset = current[np.ix_(remaining, remaining)]
        values = (probabilities[remaining][:, None] * subset).sum(axis=0)
        selected = int(remaining[int(np.argmin(values))])
        kept.append(selected)
        current = np.minimum(current, costs[:, selected][:, None])
        remaining = remaining[remaining != selected]
    kept_array = np.asarray(sorted(kept), dtype=int)
    deleted = np.setdiff1d(np.arange(total), kept_array)
    error = float(
        (
            probabilities[deleted]
            * costs[np.ix_(deleted, kept_array)].min(axis=1)
        ).sum()
    ) if deleted.size else 0.0
    return (
        kept_array,
        redistribute(costs, probabilities, kept_array),
        error,
    )


def fit_canonical_ffs_partition(
    x: np.ndarray,
    probabilities: np.ndarray,
    k: int,
) -> dict:
    probabilities = np.asarray(probabilities, dtype=float)
    costs = cost_matrix(x, p=1)
    representatives, reduced_probability, error = fast_forward_selection(
        costs, probabilities, k,
    )
    labels = force_membership(costs, representatives)
    mapped_probability = np.bincount(
        labels, weights=probabilities, minlength=k,
    )
    if not np.allclose(
        mapped_probability, reduced_probability, atol=1e-12,
    ):
        raise AssertionError("FFS redistributed probability changed")
    if not np.array_equal(
        labels[representatives], np.arange(k, dtype=int),
    ):
        raise AssertionError("FFS selected support lost its own mass")
    normalized = probabilities / probabilities.sum()
    return {
        "labels": labels,
        "centers": weighted_cluster_centers(x, labels, normalized, k),
        "representatives": representatives.astype(int),
        "reduced_probability": reduced_probability.astype(float),
        "native_reduction_error": float(error),
    }


def candidate_partitions(
    data: dict,
    maximum_k: int,
) -> dict[str, dict[str, dict[int, dict]]]:
    candidates: dict[str, dict[str, dict[int, dict]]] = {
        method: {} for method in METHODS
    }
    corridor_array = np.asarray(data["corridor"], dtype=object)
    for corridor in CORRIDOR_ORDER:
        indices = np.flatnonzero(corridor_array == corridor)
        if not len(indices):
            continue
        max_local_k = min(int(maximum_k), len(indices))
        if max_local_k < 1:
            raise RuntimeError(f"hour {data['hour']}: {corridor} is empty")
        x = data["x"][indices]
        unconditional_probability = np.asarray(
            data["station_probability"][indices], dtype=float
        )
        corridor_probability = float(unconditional_probability.sum())
        if corridor_probability <= 0:
            raise RuntimeError(
                f"hour {data['hour']}: {corridor} probability is zero"
            )
        clustering_weights = unconditional_probability / corridor_probability
        euclidean = cdist(x, x, metric="euclidean")
        ward = fit_weighted_ward(
            x, clustering_weights, tuple(range(1, max_local_k + 1))
        )

        for method in METHODS:
            candidates[method][corridor] = {}
        for k in range(1, max_local_k + 1):
            kmeans = KMeans(
                n_clusters=k,
                random_state=SEED,
                n_init=N_INIT,
                algorithm="lloyd",
            ).fit(x, sample_weight=clustering_weights)
            medoids, medoid_labels = fit_weighted_kmedoids(
                euclidean, clustering_weights, k
            )
            forward = fit_canonical_ffs_partition(
                x, unconditional_probability, k,
            )

            labels_by_method = {
                METHODS[0]: kmeans.labels_.astype(int),
                METHODS[1]: medoid_labels.astype(int),
                METHODS[2]: ward[k].astype(int),
                METHODS[3]: forward["labels"],
            }
            for method, labels in labels_by_method.items():
                if len(np.unique(labels)) != k:
                    continue
                centers = weighted_cluster_centers(
                    x, labels, clustering_weights, k
                )
                if method == METHODS[1]:
                    representatives = medoids.astype(int)
                    allocation_cost = float(np.dot(
                        unconditional_probability,
                        euclidean[
                            np.arange(len(x)), representatives[labels]
                        ],
                    ))
                    representative_rule = "actual weighted K-medoids medoid"
                    native_probability = np.bincount(
                        labels,
                        weights=unconditional_probability,
                        minlength=k,
                    )
                    native_error = np.nan
                elif method == METHODS[3]:
                    representatives = forward["representatives"]
                    allocation_cost = float(forward["native_reduction_error"])
                    representative_rule = (
                        "actual canonical p=1 fast-forward selected support"
                    )
                    native_probability = forward["reduced_probability"]
                    native_error = float(forward["native_reduction_error"])
                else:
                    local_data = {
                        "x": x,
                        "complete": data["complete"][indices],
                    }
                    representatives = centroid_representatives(
                        local_data, labels, centers, k
                    )
                    allocation_cost = weighted_sse(
                        x, labels, unconditional_probability,
                    )
                    representative_rule = (
                        "valid in-cluster station nearest weighted centroid"
                    )
                    native_probability = np.bincount(
                        labels,
                        weights=unconditional_probability,
                        minlength=k,
                    )
                    native_error = np.nan
                if not np.array_equal(
                    labels[representatives], np.arange(k, dtype=int)
                ):
                    raise AssertionError(
                        f"{method} representative/cluster mismatch"
                    )
                candidates[method][corridor][k] = {
                    "indices": indices,
                    "labels": labels,
                    "centers": centers,
                    "representatives": representatives,
                    "reduced_probability": native_probability,
                    "native_reduction_error": native_error,
                    "allocation_cost": float(allocation_cost),
                    "allocation_objective": {
                        METHODS[0]: "unconditional probability-weighted SSE",
                        METHODS[1]: (
                            "unconditional probability-weighted medoid L2 loss"
                        ),
                        METHODS[2]: "unconditional probability-weighted SSE",
                        METHODS[3]: (
                            "unconditional canonical p=1 FFS reduction error"
                        ),
                    }[method],
                    "representative_rule": representative_rule,
                }
    return candidates


def allocate_k(
    options: dict[str, dict[int, dict]],
    total_k: int,
) -> dict[str, int]:
    states: dict[int, tuple[float, dict[str, int]]] = {0: (0.0, {})}
    for corridor in CORRIDOR_ORDER:
        if corridor not in options:
            continue
        next_states: dict[int, tuple[float, dict[str, int]]] = {}
        for used, (cost, allocation) in states.items():
            for local_k, fitted in options[corridor].items():
                new_total = used + local_k
                if new_total > total_k:
                    continue
                new_cost = cost + fitted["allocation_cost"]
                previous = next_states.get(new_total)
                if previous is None or new_cost < previous[0] - 1e-15:
                    next_states[new_total] = (
                        new_cost,
                        {**allocation, corridor: int(local_k)},
                    )
        states = next_states
    if total_k not in states:
        raise RuntimeError(f"cannot allocate total K={total_k}")
    return states[total_k][1]


def combine_partition(
    data: dict,
    method_options: dict[str, dict[int, dict]],
    total_k: int,
) -> tuple[dict, list[dict]]:
    allocation = allocate_k(method_options, total_k)
    labels = np.full(len(data["stations"]), -1, dtype=int)
    centers = []
    representatives = []
    reduced_probability = []
    native_reduction_error = 0.0
    native_error_available = True
    allocation_objectives: set[str] = set()
    representative_rules: set[str] = set()
    rows = []
    offset = 0
    for corridor in CORRIDOR_ORDER:
        if corridor not in allocation:
            continue
        local_k = allocation[corridor]
        fitted = method_options[corridor][local_k]
        indices = fitted["indices"]
        labels[indices] = fitted["labels"] + offset
        if not np.array_equal(
            labels[indices] - offset, fitted["labels"]
        ):
            raise AssertionError("original cluster membership changed")
        centers.extend(fitted["centers"])
        representatives.extend(
            indices[fitted["representatives"]].astype(int).tolist()
        )
        reduced_probability.extend(
            np.asarray(fitted["reduced_probability"], dtype=float).tolist()
        )
        local_native_error = float(fitted["native_reduction_error"])
        if np.isfinite(local_native_error):
            native_reduction_error += local_native_error
        else:
            native_error_available = False
        allocation_objectives.add(str(fitted["allocation_objective"]))
        representative_rules.add(str(fitted["representative_rule"]))
        rows.append({
            "hour": int(data["hour"]),
            "k": int(total_k),
            "corridor": corridor,
            "allocated_k": int(local_k),
            "station_count": int(len(indices)),
            "complete_candidate_count": int(data["complete"][indices].sum()),
            "allocation_cost": float(fitted["allocation_cost"]),
            "allocation_objective": str(fitted["allocation_objective"]),
            "native_ffs_reduction_error": (
                local_native_error if np.isfinite(local_native_error) else np.nan
            ),
            "representative_membership_repairs": 0,
        })
        offset += local_k
    if offset != total_k or np.any(labels < 0):
        raise AssertionError("combined corridor partition is incomplete")
    representative_array = np.asarray(representatives, dtype=int)
    if not np.array_equal(
        labels[representative_array], np.arange(total_k, dtype=int)
    ):
        raise AssertionError("representative is not in its original cluster")
    reduced_probability_array = np.asarray(reduced_probability, dtype=float)
    expected_probability = np.bincount(
        labels,
        weights=np.asarray(data["station_probability"], dtype=float),
        minlength=total_k,
    )
    if not np.allclose(
        reduced_probability_array, expected_probability, atol=1e-12,
    ):
        raise AssertionError("reduced probability is inconsistent with mapping")
    if not np.isclose(reduced_probability_array.sum(), 1.0, atol=1e-12):
        raise AssertionError("reduced probability does not sum to one")
    if len(allocation_objectives) != 1 or len(representative_rules) != 1:
        raise AssertionError("method rules changed across corridors")
    return {
        "labels": labels,
        "centers": np.asarray(centers, dtype=float),
        "representatives": representative_array,
        "reduced_probability": reduced_probability_array,
        "native_reduction_error": (
            float(native_reduction_error)
            if native_error_available else np.nan
        ),
        "allocation_objective": next(iter(allocation_objectives)),
        "representative_rule": next(iter(representative_rules)),
    }, rows


def build_corridor_canonical_ffs_options(
    data: dict,
    maximum_k: int,
) -> dict[str, dict[int, dict]]:
    options: dict[str, dict[int, dict]] = {}
    corridor_array = np.asarray(data["corridor"], dtype=object)
    for corridor in CORRIDOR_ORDER:
        indices = np.flatnonzero(corridor_array == corridor)
        if not len(indices):
            continue
        x = data["x"][indices]
        probabilities = data["station_probability"][indices]
        options[corridor] = {}
        for local_k in range(1, min(maximum_k, len(indices)) + 1):
            fitted = fit_canonical_ffs_partition(
                x, probabilities, local_k,
            )
            options[corridor][local_k] = {
                "indices": indices,
                **fitted,
                "allocation_cost": fitted["native_reduction_error"],
                "allocation_objective": (
                    "unconditional canonical p=1 FFS reduction error"
                ),
                "representative_rule": (
                    "actual canonical p=1 fast-forward selected support"
                ),
            }
    return options


def build_corridor_canonical_ffs(
    data: dict,
    total_k: int,
    options: dict[str, dict[int, dict]] | None = None,
) -> tuple[dict, list[dict]]:
    if options is None:
        options = build_corridor_canonical_ffs_options(data, total_k)
    return combine_partition(data, options, total_k)


def build_global_canonical_ffs(data: dict, k: int) -> dict:
    result = fit_canonical_ffs_partition(
        data["x"], data["station_probability"], k,
    )
    return {
        **result,
        "allocation_objective": (
            "global unconditional canonical p=1 FFS reduction error"
        ),
        "representative_rule": (
            "actual global canonical p=1 fast-forward selected support"
        ),
    }


def evaluate(
    data: dict, result: dict, method: str, k: int,
) -> tuple[dict, list[dict]]:
    labels = result["labels"]
    representatives = result["representatives"]
    weights = data["station_probability"]
    distance = cdist(data["x"], data["x"], metric="euclidean")
    global_silhouette = unweighted_silhouette(distance, labels)
    representative_corridor = data["corridor"][representatives]
    component_rows: list[dict] = []
    by_duration = {duration: 0.0 for duration in DURATIONS}
    joint_wasserstein = 0.0
    sse = 0.0
    silhouette = 0.0

    for corridor in np.unique(data["corridor"]):
        members = np.flatnonzero(data["corridor"] == corridor)
        corridor_mass = float(weights[members].sum())
        conditional_sse = unweighted_sse(
            data["x"][members], labels[members]
        )
        conditional_silhouette = unweighted_silhouette(
            distance[np.ix_(members, members)],
            labels[members],
        )
        sse_component = conditional_sse
        silhouette_component = len(members) * conditional_silhouette
        sse += sse_component
        silhouette += silhouette_component

        representative_clusters = np.flatnonzero(
            representative_corridor == corridor
        )
        if set(np.unique(labels[members])) != set(representative_clusters):
            raise AssertionError("corridor cluster/representative mismatch")
        row = {
            "hour": int(data["hour"]),
            "method": method,
            "k": int(k),
            "corridor": str(corridor),
            "corridor_probability": corridor_mass,
            "station_count": int(len(members)),
            "allocated_clusters": int(len(representative_clusters)),
            "corridor_sse": float(conditional_sse),
            "sse_component": float(sse_component),
            "corridor_mean_silhouette": float(conditional_silhouette),
            "silhouette_sum_component": float(silhouette_component),
        }
        conditional_joint_wasserstein = float(wasserstein_distance_nd(
            data["x"][members],
            data["x"][representatives[representative_clusters]],
            u_weights=weights[members],
            v_weights=result["reduced_probability"][
                representative_clusters
            ],
        ))
        joint_component = corridor_mass * conditional_joint_wasserstein
        joint_wasserstein += joint_component
        row["conditional_joint_wasserstein"] = (
            conditional_joint_wasserstein
        )
        row["weighted_joint_wasserstein_component"] = joint_component
        for duration in DURATIONS:
            duration_weights = data["conditional_probability"][duration]
            duration_cluster_mass = np.bincount(
                labels, weights=duration_weights, minlength=k
            )
            original_mass = float(duration_weights[members].sum())
            reduced_mass = float(
                duration_cluster_mass[representative_clusters].sum()
            )
            if not np.isclose(original_mass, reduced_mass, atol=1e-12):
                raise AssertionError("corridor probability mass was not preserved")
            conditional_wasserstein = float(wasserstein_distance_nd(
                data["duration_log_features"][duration][members],
                data["duration_log_features"][duration][
                    representatives[representative_clusters]
                ],
                u_weights=duration_weights[members],
                v_weights=duration_cluster_mass[representative_clusters],
            ))
            wasserstein_component = original_mass * conditional_wasserstein
            by_duration[duration] += wasserstein_component
            row[f"conditional_wasserstein_{duration}"] = (
                conditional_wasserstein
            )
            row[f"weighted_wasserstein_{duration}_component"] = (
                wasserstein_component
            )
        row["weighted_wasserstein_component"] = float(sum(
            DURATION_WEIGHT[duration]
            * row[f"weighted_wasserstein_{duration}_component"]
            for duration in DURATIONS
        ))
        component_rows.append(row)

    duration_weighted_wasserstein = float(sum(
        DURATION_WEIGHT[duration] * by_duration[duration]
        for duration in DURATIONS
    ))
    native_error = float(result["native_reduction_error"])
    native_delta = (
        abs(float(joint_wasserstein) - native_error)
        if np.isfinite(native_error) else np.nan
    )
    if method == METHODS[3] and native_delta > 1e-9:
        raise AssertionError(
            "canonical FFS error does not equal corridor-constrained joint W1: "
            f"hour={data['hour']}, K={k}, delta={native_delta:.3e}"
        )
    counts = np.bincount(labels, minlength=k)
    return {
        "hour": int(data["hour"]),
        "method": method,
        "k": int(k),
        "station_count": int(len(labels)),
        "wasserstein_distance": float(joint_wasserstein),
        "duration_weighted_wasserstein_diagnostic": (
            duration_weighted_wasserstein
        ),
        "wasserstein_30": by_duration[30],
        "wasserstein_60": by_duration[60],
        "native_reduction_error": native_error,
        "native_error_wasserstein_abs_delta": native_delta,
        "allocation_objective": result["allocation_objective"],
        "clustering_sse": float(sse),
        "silhouette": float(silhouette / len(labels)),
        "global_silhouette_diagnostic": global_silhouette,
        "minimum_cluster_size": int(counts.min()),
        "singleton_clusters": int((counts == 1).sum()),
        "representative_fallbacks": 0,
        "representative_membership_repairs": 0,
    }, component_rows


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["method", "k"], as_index=False, sort=False)
        .agg(
            HourCount=("hour", "nunique"),
            MeanWasserstein=("wasserstein_distance", "mean"),
            SDWasserstein=("wasserstein_distance", "std"),
            MeanDurationWeightedWassersteinDiagnostic=(
                "duration_weighted_wasserstein_diagnostic", "mean",
            ),
            MeanWasserstein30=("wasserstein_30", "mean"),
            MeanWasserstein60=("wasserstein_60", "mean"),
            MeanNativeReductionError=("native_reduction_error", "mean"),
            MaxNativeErrorWassersteinAbsDelta=(
                "native_error_wasserstein_abs_delta", "max",
            ),
            AllocationObjective=("allocation_objective", "first"),
            MeanSSE=("clustering_sse", "mean"),
            SDSSE=("clustering_sse", "std"),
            MeanSilhouette=("silhouette", "mean"),
            SDSilhouette=("silhouette", "std"),
            TotalFallbacks=("representative_fallbacks", "sum"),
            TotalMembershipRepairs=(
                "representative_membership_repairs", "sum",
            ),
        )
    )


def build_outputs(
    hour_data: dict[int, dict],
    selected: dict[int, dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    representative_rows = []
    branch_rows = []
    assignment_rows = []
    for hour in HOURS:
        data = hour_data[hour]
        result = selected[hour]
        labels = result["labels"]
        representatives = result["representatives"]
        representative_names = [
            data["stations"][int(index)] for index in representatives
        ]
        for cluster, representative_index in enumerate(representatives):
            representative_rows.append({
                "hour": int(hour),
                "k": SELECTED_K,
                "cluster": int(cluster),
                "representative_station": representative_names[cluster],
                "selection_rule": str(result["representative_rule"]),
                "centroid_fallback_used": False,
                "representative_membership_repair_used": False,
                "all_duration_source_available": bool(all(
                    data["valid_branch"][duration][representative_index]
                    for duration in DURATIONS
                )),
                "at_least_one_duration_source_available": bool(any(
                    data["valid_branch"][duration][representative_index]
                    for duration in DURATIONS
                )),
                "all_duration_positive_impact": bool(all(
                    data["positive_impact"][duration][representative_index]
                    for duration in DURATIONS
                )),
                "at_least_one_duration_positive_impact": bool(any(
                    data["positive_impact"][duration][representative_index]
                    for duration in DURATIONS
                )),
                "station_marginal_probability": float(
                    data["station_probability"][representative_index]
                ),
                "reduced_probability": float(
                    result["reduced_probability"][cluster]
                ),
                "distance_to_centroid": float(np.linalg.norm(
                    data["x"][representative_index]
                    - result["centers"][cluster]
                )),
            })
        for duration in DURATIONS:
            conditional = data["conditional_probability"][duration]
            for cluster, representative_station in enumerate(
                representative_names
            ):
                members = np.flatnonzero(labels == cluster)
                conditional_mass = float(conditional[members].sum())
                if not np.isclose(
                    conditional_mass,
                    result["reduced_probability"][cluster],
                    atol=1e-12,
                ):
                    raise AssertionError("branch probability differs from q")
                branch_rows.append({
                    "hour": int(hour),
                    "k": SELECTED_K,
                    "cluster": int(cluster),
                    "representative_station": representative_station,
                    "duration": int(duration),
                    "cluster_size": int(len(members)),
                    "conditional_cluster_probability": conditional_mass,
                    "reduced_probability": float(
                        result["reduced_probability"][cluster]
                    ),
                    "duration_weight": DURATION_WEIGHT[duration],
                    "scenario_probability": (
                        DURATION_WEIGHT[duration] * conditional_mass
                    ),
                    "source_scenario_available": bool(
                        data["valid_branch"][duration][
                            representatives[cluster]
                        ]
                    ),
                    "source_branch_status": str(
                        data["branch_status"][duration][
                            representatives[cluster]
                        ]
                    ),
                    "source_has_positive_impact": bool(
                        data["positive_impact"][duration][
                            representatives[cluster]
                        ]
                    ),
                })
            for station_position, station in enumerate(data["stations"]):
                cluster = int(labels[station_position])
                assignment_rows.append({
                    "hour": int(hour),
                    "duration": int(duration),
                    "station": station,
                    "station_marginal_probability": float(
                        data["station_probability"][station_position]
                    ),
                    "conditional_station_probability": float(
                        conditional[station_position]
                    ),
                    "cluster": cluster,
                    "representative_station": representative_names[cluster],
                })
    return (
        pd.DataFrame(representative_rows),
        pd.DataFrame(branch_rows),
        pd.DataFrame(assignment_rows),
    )


def validate_outputs(
    representatives: pd.DataFrame,
    branches: pd.DataFrame,
    assignments: pd.DataFrame,
) -> None:
    if len(representatives) != len(HOURS) * SELECTED_K:
        raise AssertionError("representative table is incomplete")
    if representatives.duplicated(["hour", "representative_station"]).any():
        raise AssertionError("duplicate representative within an hour")
    if (representatives["reduced_probability"] <= 0).any():
        raise AssertionError("representative has nonpositive reduced probability")
    if not np.allclose(
        representatives.groupby("hour")["reduced_probability"].sum(),
        1.0,
        atol=1e-12,
    ):
        raise AssertionError("hourly reduced probabilities are invalid")
    if not representatives["all_duration_source_available"].all():
        raise AssertionError("representative lacks a valid duration branch")
    if len(branches) != len(HOURS) * SELECTED_K * len(DURATIONS):
        raise AssertionError("branch table is incomplete")
    if not branches["source_scenario_available"].all():
        raise AssertionError("representative branch source is unavailable")
    if not branches["source_branch_status"].isin(
        {"EXTRACTED", "NO_AFFECTED"}
    ).all():
        raise AssertionError("representative branch status is invalid")
    if not branches["source_has_positive_impact"].eq(
        branches["source_branch_status"].eq("EXTRACTED")
    ).all():
        raise AssertionError("branch status and impact flag disagree")
    if not np.allclose(
        branches["conditional_cluster_probability"],
        branches["reduced_probability"],
        atol=1e-12,
    ):
        raise AssertionError("branch probability differs from reduced q")
    if not np.allclose(
        branches.groupby(["hour", "duration"])[
            "conditional_cluster_probability"
        ].sum(),
        1.0,
        atol=1e-10,
    ):
        raise AssertionError("conditional cluster probabilities are invalid")
    if not np.allclose(
        branches.groupby("hour")["scenario_probability"].sum(),
        1.0,
        atol=1e-10,
    ):
        raise AssertionError("hourly scenario probabilities are invalid")
    duration_mass = branches.groupby(
        ["hour", "duration"]
    )["scenario_probability"].sum()
    for duration in DURATIONS:
        if not np.allclose(
            duration_mass.xs(duration, level="duration"),
            DURATION_WEIGHT[duration],
            atol=1e-10,
        ):
            raise AssertionError(f"duration {duration} mass is invalid")
    if assignments.duplicated(["hour", "duration", "station"]).any():
        raise AssertionError("assignment table has duplicate keys")
    cross = assignments[
        assignments["corridor"] != assignments["representative_corridor"]
    ]
    if len(cross):
        raise AssertionError("a station crossed a corridor boundary")
    if representatives["centroid_fallback_used"].any():
        raise AssertionError("out-of-cluster representative fallback remains")
    station_membership = assignments.drop_duplicates(
        ["hour", "station"]
    )[["hour", "station", "cluster"]]
    representative_membership = representatives.merge(
        station_membership,
        left_on=["hour", "representative_station"],
        right_on=["hour", "station"],
        how="left",
        validate="one_to_one",
        suffixes=("_representative", "_assigned"),
    )
    if representative_membership["cluster_assigned"].isna().any():
        raise AssertionError("representative is absent from its hour support")
    if not representative_membership["cluster_representative"].eq(
        representative_membership["cluster_assigned"]
    ).all():
        raise AssertionError("representative is not a member of its cluster")


def build_ffs_mapping(
    hour_data: dict[int, dict],
    selected: dict[int, dict],
    k: int = SELECTED_K,
    expected_rule: str = (
        "actual canonical p=1 fast-forward selected support"
    ),
) -> pd.DataFrame:
    rows: list[dict] = []
    for hour in HOURS:
        data = hour_data[hour]
        result = selected[hour]
        labels = np.asarray(result["labels"], dtype=int)
        representatives = np.asarray(result["representatives"], dtype=int)
        if result["representative_rule"] != expected_rule:
            raise AssertionError("mapping is not based on requested canonical FFS")
        representative_names = np.asarray(
            [data["stations"][index] for index in representatives],
            dtype=object,
        )
        for position, station in enumerate(data["stations"]):
            cluster = int(labels[position])
            representative_index = int(representatives[cluster])
            rows.append({
                "hour": int(hour),
                "k": int(k),
                "corridor": str(data["corridor"][position]),
                "station": str(station),
                "cluster": cluster,
                "representative_station": str(
                    representative_names[cluster]
                ),
                "representative_corridor": str(
                    data["corridor"][representative_index]
                ),
                "cross_corridor_assignment": bool(
                    data["corridor"][position]
                    != data["corridor"][representative_index]
                ),
                "is_selected_support": bool(position == representative_index),
                "station_probability": float(
                    data["station_probability"][position]
                ),
                "representative_reduced_probability": float(
                    result["reduced_probability"][cluster]
                ),
                "ground_distance": float(np.linalg.norm(
                    data["x"][position] - data["x"][representative_index]
                )),
                "branch_status_30": str(
                    data["branch_status"][30][position]
                ),
                "branch_status_60": str(
                    data["branch_status"][60][position]
                ),
            })
    mapping = pd.DataFrame(rows)
    if mapping.duplicated(["hour", "station"]).any():
        raise AssertionError("FFS mapping has duplicate station-hours")
    if mapping.groupby("hour")["is_selected_support"].sum().ne(
        k
    ).any():
        raise AssertionError("FFS mapping does not contain K selected supports")
    mapped_mass = mapping.groupby(
        ["hour", "cluster"], sort=True,
    )["station_probability"].sum()
    reported_mass = mapping.drop_duplicates(
        ["hour", "cluster"]
    ).set_index(["hour", "cluster"])[
        "representative_reduced_probability"
    ].sort_index()
    if not np.allclose(mapped_mass.sort_index(), reported_mass, atol=1e-12):
        raise AssertionError("FFS mapping does not preserve reduced q")
    return mapping


def build_ffs_validation(
    hour_data: dict[int, dict],
    selected: dict[int, dict],
    metrics: pd.DataFrame,
    representatives: pd.DataFrame,
    branches: pd.DataFrame,
    assignments: pd.DataFrame,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    selected_metrics = metrics[
        metrics["method"].eq(METHODS[3])
        & metrics["k"].eq(SELECTED_K)
    ].sort_values("hour")
    retained = int(sum(len(hour_data[hour]["stations"]) for hour in HOURS))
    all_valid = bool(all(
        hour_data[hour]["all_duration_valid"].all() for hour in HOURS
    ))
    both_no_affected = int((
        mapping["branch_status_30"].eq("NO_AFFECTED")
        & mapping["branch_status_60"].eq("NO_AFFECTED")
    ).sum())
    max_exact_delta = float(
        selected_metrics["native_error_wasserstein_abs_delta"].max()
    )
    expected_representatives = len(HOURS) * SELECTED_K
    per_hour_representatives = representatives.groupby("hour").size()
    per_hour_selected_support = mapping.groupby("hour")[
        "is_selected_support"
    ].sum()
    per_hour_k_valid = bool(
        per_hour_representatives.reindex(HOURS, fill_value=0).eq(
            SELECTED_K
        ).all()
        and per_hour_selected_support.reindex(HOURS, fill_value=0).eq(
            SELECTED_K
        ).all()
    )
    retained_support = mapping.loc[
        mapping["is_selected_support"], ["hour", "station"]
    ].drop_duplicates()
    representative_support = representatives[
        ["hour", "representative_station"]
    ].rename(columns={"representative_station": "station"})
    representative_support_check = representative_support.merge(
        retained_support,
        on=["hour", "station"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    actual_retained_count = int(
        representative_support_check["_merge"].eq("both").sum()
    )
    cross_corridor_count = int((
        assignments["corridor"] != assignments["representative_corridor"]
    ).sum())
    duplicate_count = int(
        representatives.duplicated(["hour", "representative_station"]).sum()
        + mapping.duplicated(["hour", "station"]).sum()
        + assignments.duplicated(["hour", "duration", "station"]).sum()
        + branches.duplicated(["hour", "duration", "cluster"]).sum()
    )
    fallback_count = int(representatives["centroid_fallback_used"].sum())
    repair_count = int(
        representatives["representative_membership_repair_used"].sum()
    )
    representative_rules = representatives["selection_rule"].astype(str)
    rule_valid = bool(
        representative_rules.str.contains(
            "actual canonical p=1 fast-forward selected support",
            regex=False,
        ).all()
        and not representative_rules.str.contains(
            "centroid", case=False, regex=False,
        ).any()
    )
    location_probability_delta = max(
        abs(float(hour_data[hour]["station_probability"].sum()) - 1.0)
        for hour in HOURS
    )
    duration_mass = branches.groupby(
        ["hour", "duration"]
    )["scenario_probability"].sum()
    duration_mass_delta = {
        duration: float(max(
            abs(
                float(duration_mass.loc[(hour, duration)])
                - DURATION_WEIGHT[duration]
            )
            for hour in HOURS
        ))
        for duration in DURATIONS
    }
    mapped_q = mapping.groupby(
        ["hour", "cluster"], sort=True,
    )["station_probability"].sum().sort_index()
    reported_q = mapping.drop_duplicates(
        ["hour", "cluster"]
    ).set_index(["hour", "cluster"])[
        "representative_reduced_probability"
    ].sort_index()
    mapping_q_delta = float(np.max(np.abs(
        mapped_q.to_numpy(dtype=float) - reported_q.to_numpy(dtype=float)
    )))
    mapping_probability_delta = float(max(
        abs(value - 1.0)
        for value in mapping.groupby("hour")["station_probability"].sum()
    ))
    gates = [
        {
            "gate": "retained_station_hours",
            "observed": retained,
            "expected": EXPECTED_RETAINED_STATION_HOURS,
            "passed": (not EXPECTED_RETAINED_STATION_HOURS
                       or retained == EXPECTED_RETAINED_STATION_HOURS),
        },
        {
            "gate": "per_hour_representative_count_equals_k",
            "observed": json.dumps(
                per_hour_representatives.astype(int).to_dict(), sort_keys=True,
            ),
            "expected": SELECTED_K,
            "passed": per_hour_k_valid,
        },
        {
            "gate": "representatives_are_actual_retained_support",
            "observed": actual_retained_count,
            "expected": expected_representatives,
            "passed": actual_retained_count == expected_representatives,
        },
        {
            "gate": "cross_corridor_assignments",
            "observed": cross_corridor_count,
            "expected": 0,
            "passed": cross_corridor_count == 0,
        },
        {
            "gate": "duplicate_output_keys",
            "observed": duplicate_count,
            "expected": 0,
            "passed": duplicate_count == 0,
        },
        {
            "gate": "out_of_cluster_fallbacks",
            "observed": fallback_count,
            "expected": 0,
            "passed": fallback_count == 0,
        },
        {
            "gate": "ffs_actual_support_representative_rule",
            "observed": " | ".join(sorted(set(representative_rules))),
            "expected": "actual selected support; no centroid",
            "passed": rule_valid,
        },
        {
            "gate": "location_probability_sum_per_hour",
            "observed": location_probability_delta,
            "expected": "max abs delta <=1e-12",
            "passed": location_probability_delta <= 1e-12,
        },
        {
            "gate": "d30_scenario_probability_mass",
            "observed": duration_mass_delta[30],
            "expected": f"pi30={DURATION_WEIGHT[30]:.17g}; max abs delta <=1e-12",
            "passed": duration_mass_delta[30] <= 1e-12,
        },
        {
            "gate": "d60_scenario_probability_mass",
            "observed": duration_mass_delta[60],
            "expected": f"pi60={DURATION_WEIGHT[60]:.17g}; max abs delta <=1e-12",
            "passed": duration_mass_delta[60] <= 1e-12,
        },
        {
            "gate": "mapping_q_and_probability_preservation",
            "observed": max(mapping_q_delta, mapping_probability_delta),
            "expected": "max abs delta <=1e-12",
            "passed": (
                mapping_q_delta <= 1e-12
                and mapping_probability_delta <= 1e-12
            ),
        },
        {
            "gate": "all_retained_have_two_valid_duration_branches",
            "observed": all_valid,
            "expected": True,
            "passed": all_valid,
        },
        {
            "gate": "both_no_affected_excluded_from_retained",
            "observed": both_no_affected,
            "expected": 0,
            "passed": both_no_affected == 0,
        },
        {
            "gate": "representative_rows",
            "observed": int(len(representatives)),
            "expected": expected_representatives,
            "passed": len(representatives) == expected_representatives,
        },
        {
            "gate": "branch_rows",
            "observed": int(len(branches)),
            "expected": len(HOURS) * SELECTED_K * len(DURATIONS),
            "passed": len(branches) == len(HOURS) * SELECTED_K * len(DURATIONS),
        },
        {
            "gate": "assignment_rows",
            "observed": int(len(assignments)),
            "expected": retained * len(DURATIONS),
            "passed": len(assignments) == retained * len(DURATIONS),
        },
        {
            "gate": "membership_repairs",
            "observed": repair_count,
            "expected": 0,
            "passed": repair_count == 0,
        },
        {
            "gate": "native_ffs_error_equals_joint_wasserstein",
            "observed": max_exact_delta,
            "expected": "<=1e-9",
            "passed": max_exact_delta <= 1e-9,
        },
    ]
    gate_frame = pd.DataFrame(gates)
    if not gate_frame["passed"].all():
        failed = gate_frame.loc[~gate_frame["passed"], "gate"].tolist()
        raise AssertionError(f"canonical FFS validation failed: {failed}")
    summary = {
        "status": "PASS",
        "method": METHODS[3],
        "ground_cost": (
            "plain Euclidean on joint sqrt(duration probability)-scaled "
            "log1p Queue+Walking features"
        ),
        "retained_station_hours": retained,
        "representatives": int(len(representatives)),
        "branches": int(len(branches)),
        "assignments": int(len(assignments)),
        "maximum_native_error_wasserstein_abs_delta": max_exact_delta,
        "maximum_location_probability_abs_delta": location_probability_delta,
        "maximum_mapping_q_abs_delta": mapping_q_delta,
        "maximum_mapping_probability_abs_delta": mapping_probability_delta,
        "fallbacks": fallback_count,
        "membership_repairs": repair_count,
        "gates": gates,
    }
    return gate_frame, summary


def _mapping_joint_wasserstein(
    data: dict,
    mapping: pd.DataFrame,
) -> float:
    station_position = {
        station: index for index, station in enumerate(data["stations"])
    }
    mapping = mapping.set_index("station").loc[data["stations"]].reset_index()
    total = 0.0
    for corridor in np.unique(data["corridor"]):
        members = np.flatnonzero(data["corridor"] == corridor)
        representative_names = sorted(
            mapping.iloc[members]["representative_station"].unique()
        )
        representative_indices = np.asarray([
            station_position[str(station)] for station in representative_names
        ], dtype=int)
        if not np.all(data["corridor"][representative_indices] == corridor):
            raise AssertionError("production mapping crosses a corridor")
        representative_lookup = {
            station: index for index, station in enumerate(representative_names)
        }
        labels = np.asarray([
            representative_lookup[str(value)]
            for value in mapping.iloc[members]["representative_station"]
        ], dtype=int)
        local_weights = data["station_probability"][members]
        reduced = np.bincount(
            labels, weights=local_weights, minlength=len(representative_names)
        )
        corridor_mass = float(local_weights.sum())
        total += corridor_mass * float(wasserstein_distance_nd(
            data["x"][members],
            data["x"][representative_indices],
            u_weights=local_weights,
            v_weights=reduced,
        ))
    return float(total)


def compare_ffs_with_production_kmeans(
    production_dir: Path,
    hour_data: dict[int, dict],
    metrics: pd.DataFrame,
    ffs_mapping: pd.DataFrame,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
]:
    production_dir = production_dir.resolve()
    assignment_path = require_file(
        production_dir
        / output_name("assignments"),
        "production K-means assignment table",
    )
    production = pd.read_csv(assignment_path, low_memory=False)
    required = {
        "hour", "duration", "station", "cluster",
        "representative_station", "method",
    }
    missing = required.difference(production.columns)
    if missing:
        raise ValueError(
            f"production K-means assignment misses: {sorted(missing)}"
        )
    if set(production["method"].astype(str)) != {METHODS[0]}:
        raise ValueError("production comparison input is not K-means")
    per_station_unique = production.groupby(
        ["hour", "station"]
    )[["cluster", "representative_station"]].nunique()
    if per_station_unique.to_numpy().max() != 1:
        raise AssertionError("production D30/D60 mappings disagree")
    production = production.drop_duplicates(
        ["hour", "station"]
    )[["hour", "station", "cluster", "representative_station"]].rename(
        columns={
            "cluster": "production_kmeans_cluster",
            "representative_station": (
                "production_kmeans_representative_station"
            ),
        }
    )
    detail = ffs_mapping.merge(
        production,
        on=["hour", "station"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not detail["_merge"].eq("both").all():
        raise AssertionError("FFS and production K-means retained keys differ")
    detail = detail.drop(columns="_merge")
    detail["same_mapping_representative"] = detail[
        "representative_station"
    ].eq(detail["production_kmeans_representative_station"])

    representative_change_rows: list[dict] = []
    for hour in HOURS:
        subset = detail[detail["hour"].eq(hour)]
        station_corridor = subset.set_index("station")["corridor"].to_dict()
        ffs_q = subset.groupby("representative_station")[
            "station_probability"
        ].sum().to_dict()
        production_q = subset.groupby(
            "production_kmeans_representative_station"
        )["station_probability"].sum().to_dict()
        union = sorted(set(ffs_q) | set(production_q))
        for station in union:
            in_ffs = station in ffs_q
            in_production = station in production_q
            if in_ffs and in_production:
                change_type = "SHARED"
            elif in_ffs:
                change_type = "NEW_ONLY"
            else:
                change_type = "OLD_ONLY"
            new_q = float(ffs_q.get(station, 0.0))
            old_q = float(production_q.get(station, 0.0))
            representative_change_rows.append({
                "hour": int(hour),
                "representative_station": str(station),
                "corridor": str(station_corridor[str(station)]),
                "change_type": change_type,
                "in_canonical_ffs": bool(in_ffs),
                "in_production_kmeans": bool(in_production),
                "canonical_ffs_reduced_probability": new_q,
                "production_kmeans_reduced_probability": old_q,
                "absolute_reduced_probability_delta": abs(new_q - old_q),
            })
    representative_changes = pd.DataFrame(representative_change_rows)
    observed_corridor_counts = (
        representative_changes.groupby(
            ["hour", "corridor"], as_index=False, sort=True,
        )
        .agg(
            canonical_ffs_representative_count=("in_canonical_ffs", "sum"),
            production_kmeans_representative_count=(
                "in_production_kmeans", "sum",
            ),
            representative_set_overlap_count=(
                "change_type", lambda values: int((values == "SHARED").sum()),
            ),
        )
    )
    corridor_counts = pd.MultiIndex.from_product(
        [HOURS, CORRIDOR_ORDER], names=["hour", "corridor"]
    ).to_frame(index=False).merge(
        observed_corridor_counts,
        on=["hour", "corridor"],
        how="left",
        validate="one_to_one",
    )
    for column in (
        "canonical_ffs_representative_count",
        "production_kmeans_representative_count",
        "representative_set_overlap_count",
    ):
        corridor_counts[column] = corridor_counts[column].fillna(0).astype(int)
    corridor_counts["representative_count_delta"] = (
        corridor_counts["canonical_ffs_representative_count"]
        - corridor_counts["production_kmeans_representative_count"]
    )

    ffs_metrics = metrics[
        metrics["method"].eq(METHODS[3])
        & metrics["k"].eq(SELECTED_K)
    ].set_index("hour")
    hour_rows: list[dict] = []
    for hour in HOURS:
        subset = detail[detail["hour"].eq(hour)]
        ffs_representatives = set(subset["representative_station"])
        production_representatives = set(
            subset["production_kmeans_representative_station"]
        )
        if len(ffs_representatives) != SELECTED_K:
            raise AssertionError(f"hour {hour}: FFS representative count changed")
        if len(production_representatives) != SELECTED_K:
            raise AssertionError(
                f"hour {hour}: production K-means representative count changed"
            )
        production_wasserstein = _mapping_joint_wasserstein(
            hour_data[hour],
            production[production["hour"].eq(hour)].rename(columns={
                "production_kmeans_representative_station": (
                    "representative_station"
                ),
            }),
        )
        ffs_wasserstein = float(
            ffs_metrics.loc[hour, "wasserstein_distance"]
        )
        hour_rows.append({
            "hour": int(hour),
            "station_count": int(len(subset)),
            "same_mapping_representative_count": int(
                subset["same_mapping_representative"].sum()
            ),
            "same_mapping_representative_percent": float(
                100.0 * subset["same_mapping_representative"].mean()
            ),
            "representative_set_overlap_count": int(
                len(ffs_representatives & production_representatives)
            ),
            "representative_set_overlap_percent": float(
                100.0
                * len(ffs_representatives & production_representatives)
                / SELECTED_K
            ),
            "ffs_joint_wasserstein": ffs_wasserstein,
            "production_kmeans_joint_wasserstein": production_wasserstein,
            "ffs_minus_production_kmeans_joint_wasserstein": float(
                ffs_wasserstein - production_wasserstein
            ),
            "ffs_native_reduction_error": float(
                ffs_metrics.loc[hour, "native_reduction_error"]
            ),
            "representative_q_union_mean_abs_delta": float(
                representative_changes.loc[
                    representative_changes["hour"].eq(hour),
                    "absolute_reduced_probability_delta",
                ].mean()
            ),
            "representative_q_union_max_abs_delta": float(
                representative_changes.loc[
                    representative_changes["hour"].eq(hour),
                    "absolute_reduced_probability_delta",
                ].max()
            ),
        })
    by_hour = pd.DataFrame(hour_rows)
    total_overlap = int(
        representative_changes["change_type"].eq("SHARED").sum()
    )
    old_only = int(
        representative_changes["change_type"].eq("OLD_ONLY").sum()
    )
    new_only = int(
        representative_changes["change_type"].eq("NEW_ONLY").sum()
    )
    expected_slots = len(HOURS) * SELECTED_K
    summary = pd.DataFrame([{
        "hour_count": int(len(by_hour)),
        "station_hour_count": int(len(detail)),
        "same_mapping_representative_count": int(
            detail["same_mapping_representative"].sum()
        ),
        "same_mapping_representative_percent": float(
            100.0 * detail["same_mapping_representative"].mean()
        ),
        "representative_slots_per_method": expected_slots,
        "representative_set_overlap_count": total_overlap,
        "representative_set_overlap_percent": float(
            100.0 * total_overlap / expected_slots
        ),
        "production_kmeans_old_only_representative_count": old_only,
        "canonical_ffs_new_only_representative_count": new_only,
        "mean_representative_set_overlap_count": float(
            by_hour["representative_set_overlap_count"].mean()
        ),
        "representative_q_union_mean_abs_delta": float(
            representative_changes[
                "absolute_reduced_probability_delta"
            ].mean()
        ),
        "representative_q_union_max_abs_delta": float(
            representative_changes[
                "absolute_reduced_probability_delta"
            ].max()
        ),
        "mean_ffs_joint_wasserstein": float(
            by_hour["ffs_joint_wasserstein"].mean()
        ),
        "mean_production_kmeans_joint_wasserstein": float(
            by_hour["production_kmeans_joint_wasserstein"].mean()
        ),
        "mean_ffs_minus_production_kmeans_joint_wasserstein": float(
            by_hour[
                "ffs_minus_production_kmeans_joint_wasserstein"
            ].mean()
        ),
        "production_kmeans_directory": portable(production_dir),
    }])
    return detail, representative_changes, corridor_counts, by_hour, summary


def _mapping_unrestricted_joint_wasserstein(
    data: dict,
    mapping: pd.DataFrame,
) -> float:
    station_position = {
        station: index for index, station in enumerate(data["stations"])
    }
    ordered = mapping.set_index("station").loc[data["stations"]].reset_index()
    representative_names = sorted(ordered["representative_station"].unique())
    representative_indices = np.asarray([
        station_position[str(station)] for station in representative_names
    ], dtype=int)
    lookup = {
        station: index for index, station in enumerate(representative_names)
    }
    labels = np.asarray([
        lookup[str(value)] for value in ordered["representative_station"]
    ], dtype=int)
    reduced = np.bincount(
        labels,
        weights=data["station_probability"],
        minlength=len(representative_names),
    )
    return float(wasserstein_distance_nd(
        data["x"],
        data["x"][representative_indices],
        u_weights=data["station_probability"],
        v_weights=reduced,
    ))


def build_retained_key_audit(
    hour_data: dict[int, dict],
    production_dir: Path,
) -> dict:
    assignment_path = require_file(
        production_dir.resolve()
        / output_name("assignments"),
        "production cohort retained assignment table",
    )
    production = pd.read_csv(
        assignment_path,
        usecols=["hour", "station"],
        low_memory=False,
    )
    production["hour"] = pd.to_numeric(
        production["hour"], errors="raise"
    ).astype(int)
    production["station"] = (
        production["station"].astype(str).str.strip().str.upper()
    )
    production = production[
        production["hour"].isin(HOURS)
    ].drop_duplicates(["hour", "station"])
    current = pd.DataFrame([
        {
            "hour": int(hour),
            "station": str(station).strip().upper(),
            "current_station_probability": float(
                hour_data[hour]["station_probability"][index]
            ),
        }
        for hour in HOURS
        for index, station in enumerate(hour_data[hour]["stations"])
    ])
    production_keys = sorted(map(tuple, production[["hour", "station"]].values))
    current_keys = sorted(map(tuple, current[["hour", "station"]].values))

    probability_path = require_file(
        production_dir.resolve() / output_name("station_probability"),
        "production cohort station probability table",
    )
    production_probability = pd.read_csv(
        probability_path,
        usecols=["hour", "station", "StationProbability"],
        low_memory=False,
    ).rename(columns={"StationProbability": "production_station_probability"})
    production_probability["hour"] = pd.to_numeric(
        production_probability["hour"], errors="raise"
    ).astype(int)
    production_probability["station"] = (
        production_probability["station"].astype(str).str.strip().str.upper()
    )
    probability_comparison = current.merge(
        production_probability,
        on=["hour", "station"],
        how="left",
        validate="one_to_one",
    )
    if probability_comparison["production_station_probability"].isna().any():
        raise AssertionError("production station probability keys are incomplete")
    maximum_probability_delta = float(np.max(np.abs(
        probability_comparison["current_station_probability"].to_numpy()
        - probability_comparison["production_station_probability"].to_numpy()
    )))

    def digest(keys: list[tuple]) -> str:
        payload = "".join(
            f"{int(hour):02d}|{station}\n" for hour, station in keys
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    production_counts = (
        production.groupby("hour").size().reindex(HOURS, fill_value=0)
        .astype(int).to_dict()
    )
    current_counts = (
        current.groupby("hour").size().reindex(HOURS, fill_value=0)
        .astype(int).to_dict()
    )
    audit = {
        "production_assignment_file": portable(assignment_path),
        "production_probability_file": portable(probability_path),
        "current_key_count": len(current_keys),
        "production_key_count": len(production_keys),
        "current_key_sha256": digest(current_keys),
        "production_key_sha256": digest(production_keys),
        "current_count_by_hour": {
            str(key): value for key, value in current_counts.items()
        },
        "production_count_by_hour": {
            str(key): value for key, value in production_counts.items()
        },
        "exact_key_set_match": current_keys == production_keys,
        "exact_per_hour_count_match": current_counts == production_counts,
        "maximum_station_probability_abs_delta": maximum_probability_delta,
        "exact_station_probability_match": maximum_probability_delta <= 1e-12,
    }
    if not audit["exact_key_set_match"]:
        raise AssertionError(
            "global FFS retained station-hour keys differ from production "
            "cohort support"
        )
    if not audit["exact_station_probability_match"]:
        raise AssertionError(
            "global FFS station probabilities differ from production "
            "cohort probabilities"
        )
    return audit


def build_global_vs_corridor_ffs_comparison(
    hour_data: dict[int, dict],
) -> dict[str, object]:
    corridor_results: dict[int, dict[int, dict]] = {
        k: {} for k in K_VALUES
    }
    global_results: dict[int, dict[int, dict]] = {
        k: {} for k in K_VALUES
    }
    for hour in HOURS:
        data = hour_data[hour]
        options = build_corridor_canonical_ffs_options(data, max(K_VALUES))
        for k in K_VALUES:
            corridor_results[k][hour], _rows = build_corridor_canonical_ffs(
                data, k, options,
            )
            global_results[k][hour] = build_global_canonical_ffs(data, k)

    corridor_mapping = pd.concat([
        build_ffs_mapping(
            hour_data,
            corridor_results[k],
            k=k,
            expected_rule=(
                "actual canonical p=1 fast-forward selected support"
            ),
        )
        for k in K_VALUES
    ], ignore_index=True)
    global_mapping = pd.concat([
        build_ffs_mapping(
            hour_data,
            global_results[k],
            k=k,
            expected_rule=(
                "actual global canonical p=1 fast-forward selected support"
            ),
        )
        for k in K_VALUES
    ], ignore_index=True)

    metric_rows: list[dict] = []
    representative_rows: list[dict] = []
    corridor_rows: list[dict] = []
    overlap_rows: list[dict] = []
    gate_rows: list[dict] = []
    for hour in HOURS:
        data = hour_data[hour]
        corridor_mass = {
            corridor: float(data["station_probability"][
                data["corridor"] == corridor
            ].sum())
            for corridor in CORRIDOR_ORDER
        }
        for k in K_VALUES:
            mappings = {
                "GLOBAL": global_mapping[
                    global_mapping["hour"].eq(hour)
                    & global_mapping["k"].eq(k)
                ].copy(),
                "CORRIDOR_CONSTRAINED": corridor_mapping[
                    corridor_mapping["hour"].eq(hour)
                    & corridor_mapping["k"].eq(k)
                ].copy(),
            }
            results = {
                "GLOBAL": global_results[k][hour],
                "CORRIDOR_CONSTRAINED": corridor_results[k][hour],
            }
            scope_metrics: dict[str, dict] = {}
            representative_sets: dict[str, set[str]] = {}
            for scope, mapping in mappings.items():
                result = results[scope]
                unrestricted_w1 = _mapping_unrestricted_joint_wasserstein(
                    data, mapping,
                )
                constrained_w1 = (
                    _mapping_joint_wasserstein(data, mapping)
                    if scope == "CORRIDOR_CONSTRAINED" else np.nan
                )
                native_error = float(result["native_reduction_error"])
                selected_support = mapping[mapping["is_selected_support"]]
                representative_sets[scope] = set(
                    selected_support["station"].astype(str)
                )
                assignment_mean = float((
                    mapping["station_probability"]
                    * mapping["ground_distance"]
                ).sum())
                assignment_max = float(mapping["ground_distance"].max())
                cross = mapping["cross_corridor_assignment"].astype(bool)
                cross_count = int(cross.sum())
                cross_mass = float(
                    mapping.loc[cross, "station_probability"].sum()
                )

                mapped_q = mapping.groupby("cluster")[
                    "station_probability"
                ].sum().sort_index()
                reported_q = mapping.drop_duplicates("cluster").set_index(
                    "cluster"
                )["representative_reduced_probability"].sort_index()
                q_delta = float(np.max(np.abs(
                    mapped_q.to_numpy(dtype=float)
                    - reported_q.to_numpy(dtype=float)
                )))
                probability_delta = abs(
                    float(mapping["station_probability"].sum()) - 1.0
                )
                own_mass_valid = bool(selected_support[
                    "station"
                ].eq(selected_support["representative_station"]).all())
                selected_count = int(len(selected_support))
                expected_w1 = (
                    unrestricted_w1
                    if scope == "GLOBAL" else constrained_w1
                )
                error_delta = abs(native_error - expected_w1)
                gates = {
                    "retained_station_count": (
                        len(mapping) == len(data["stations"])
                    ),
                    "selected_support_count": selected_count == k,
                    "selected_support_keeps_own_mass": own_mass_valid,
                    "probability_mass_preserved": probability_delta <= 1e-12,
                    "reduced_q_exact": q_delta <= 1e-12,
                    "native_error_exact": error_delta <= 1e-9,
                    "corridor_mapping_has_no_cross_assignment": (
                        scope != "CORRIDOR_CONSTRAINED" or cross_count == 0
                    ),
                }
                for gate, passed in gates.items():
                    gate_rows.append({
                        "hour": int(hour),
                        "k": int(k),
                        "scope": scope,
                        "gate": gate,
                        "observed": {
                            "probability_mass_preserved": probability_delta,
                            "reduced_q_exact": q_delta,
                            "native_error_exact": error_delta,
                        }.get(gate, int(passed)),
                        "passed": bool(passed),
                    })

                corridor_counts_for_scope: list[int] = []
                corridor_q_for_scope: list[float] = []
                for corridor in CORRIDOR_ORDER:
                    rep_part = selected_support[
                        selected_support["representative_corridor"].eq(corridor)
                    ]
                    rep_count = int(len(rep_part))
                    rep_q = float(
                        rep_part["representative_reduced_probability"].sum()
                    )
                    corridor_counts_for_scope.append(rep_count)
                    corridor_q_for_scope.append(rep_q)
                    corridor_rows.append({
                        "hour": int(hour),
                        "k": int(k),
                        "scope": scope,
                        "corridor": corridor,
                        "original_corridor_probability": corridor_mass[corridor],
                        "representative_count": rep_count,
                        "representative_count_share": rep_count / k,
                        "representative_reduced_probability": rep_q,
                        "zero_representative_corridor": rep_count == 0,
                    })
                zero_corridors = [
                    corridor for corridor, count in zip(
                        CORRIDOR_ORDER, corridor_counts_for_scope
                    ) if count == 0 and corridor_mass[corridor] > 0
                ]
                zero_mass = float(sum(
                    corridor_mass[corridor] for corridor in zero_corridors
                ))
                scope_metrics[scope] = {
                    "native_error": native_error,
                    "unrestricted_w1": unrestricted_w1,
                    "constrained_w1": constrained_w1,
                    "assignment_mean": assignment_mean,
                    "assignment_max": assignment_max,
                    "cross_count": cross_count,
                    "cross_mass": cross_mass,
                    "zero_count": len(zero_corridors),
                    "zero_mass": zero_mass,
                    "max_rep_count": max(corridor_counts_for_scope),
                    "max_rep_count_share": max(corridor_counts_for_scope) / k,
                    "max_rep_q": max(corridor_q_for_scope),
                }
                for row in selected_support.itertuples(index=False):
                    representative_rows.append({
                        "hour": int(hour),
                        "k": int(k),
                        "scope": scope,
                        "representative_station": str(row.station),
                        "representative_corridor": str(row.corridor),
                        "reduced_probability": float(
                            row.representative_reduced_probability
                        ),
                    })

            overlap = representative_sets["GLOBAL"] & representative_sets[
                "CORRIDOR_CONSTRAINED"
            ]
            overlap_rows.extend({
                "hour": int(hour),
                "k": int(k),
                "representative_station": station,
                "overlap": station in overlap,
                "in_global": station in representative_sets["GLOBAL"],
                "in_corridor_constrained": (
                    station in representative_sets["CORRIDOR_CONSTRAINED"]
                ),
            } for station in sorted(
                representative_sets["GLOBAL"]
                | representative_sets["CORRIDOR_CONSTRAINED"]
            ))
            global_metric = scope_metrics["GLOBAL"]
            corridor_metric = scope_metrics["CORRIDOR_CONSTRAINED"]
            metric_rows.append({
                "hour": int(hour),
                "k": int(k),
                "global_unrestricted_joint_w1": global_metric["unrestricted_w1"],
                "global_native_error": global_metric["native_error"],
                "corridor_unrestricted_joint_w1": corridor_metric[
                    "unrestricted_w1"
                ],
                "corridor_constrained_joint_w1": corridor_metric[
                    "constrained_w1"
                ],
                "corridor_native_error": corridor_metric["native_error"],
                "global_cross_corridor_assignment_count": global_metric[
                    "cross_count"
                ],
                "global_cross_corridor_probability_mass": global_metric[
                    "cross_mass"
                ],
                "global_probability_weighted_mean_assignment_distance": (
                    global_metric["assignment_mean"]
                ),
                "global_max_assignment_distance": global_metric[
                    "assignment_max"
                ],
                "corridor_probability_weighted_mean_assignment_distance": (
                    corridor_metric["assignment_mean"]
                ),
                "corridor_max_assignment_distance": corridor_metric[
                    "assignment_max"
                ],
                "global_zero_representative_corridor_count": global_metric[
                    "zero_count"
                ],
                "global_zero_representative_corridor_probability_mass": (
                    global_metric["zero_mass"]
                ),
                "corridor_zero_representative_corridor_count": corridor_metric[
                    "zero_count"
                ],
                "corridor_zero_representative_corridor_probability_mass": (
                    corridor_metric["zero_mass"]
                ),
                "global_max_corridor_representative_count": global_metric[
                    "max_rep_count"
                ],
                "global_max_corridor_representative_count_share": global_metric[
                    "max_rep_count_share"
                ],
                "global_max_corridor_representative_q": global_metric[
                    "max_rep_q"
                ],
                "corridor_max_corridor_representative_count": corridor_metric[
                    "max_rep_count"
                ],
                "corridor_max_corridor_representative_count_share": corridor_metric[
                    "max_rep_count_share"
                ],
                "corridor_max_corridor_representative_q": corridor_metric[
                    "max_rep_q"
                ],
                "representative_set_overlap_count": len(overlap),
                "representative_set_overlap_percent": 100.0 * len(overlap) / k,
            })

    metrics_by_window = pd.DataFrame(metric_rows).sort_values(["k", "hour"])
    numeric_metrics = [
        column for column in metrics_by_window.columns
        if column not in {"hour", "k"}
    ]
    aggregate_rows: list[dict] = []
    for k, part in metrics_by_window.groupby("k", sort=True):
        row: dict = {"k": int(k), "window_count": int(len(part))}
        for column in numeric_metrics:
            row[f"mean_{column}"] = float(part[column].mean())
            row[f"max_{column}"] = float(part[column].max())
        aggregate_rows.append(row)
    metrics_by_k = pd.DataFrame(aggregate_rows)

    marginal_rows: list[dict] = []
    for hour, part in metrics_by_window.groupby("hour", sort=True):
        part = part.sort_values("k")
        for scope, column in (
            ("GLOBAL", "global_native_error"),
            ("CORRIDOR_CONSTRAINED", "corridor_native_error"),
        ):
            previous = None
            for row in part.itertuples(index=False):
                current = float(getattr(row, column))
                if previous is not None:
                    marginal_rows.append({
                        "hour": int(hour),
                        "k": int(row.k),
                        "scope": scope,
                        "previous_k": int(row.k) - 1,
                        "native_error_reduction": previous - current,
                        "native_error_reduction_percent": (
                            100.0 * (previous - current) / previous
                            if previous > 0 else 0.0
                        ),
                    })
                previous = current
    marginal = pd.DataFrame(marginal_rows)
    gates = pd.DataFrame(gate_rows)
    if not gates["passed"].all():
        failed = gates.loc[~gates["passed"]].head(20).to_dict("records")
        raise AssertionError(f"global/corridor FFS validation failed: {failed}")
    retained_per_k = global_mapping.groupby("k").size()
    expected_per_k = EXPECTED_RETAINED_STATION_HOURS or int(retained_per_k.iloc[0])
    if not retained_per_k.eq(expected_per_k).all():
        raise AssertionError("global retained station-hour count changed")
    summary = {
        "status": "PASS",
        "scope": "reduction-only global versus corridor-constrained FFS",
        "hours": list(HOURS),
        "k_values": list(K_VALUES),
        "retained_station_hours_per_k": expected_per_k,
        "ground_cost": "canonical p=1 plain Euclidean on common joint feature",
        "common_probability": "actual historical station probability",
        "other_clustering_methods_run": False,
        "plots_written": False,
        "pdf_written": False,
        "stage1_or_stage2_run": False,
        "gate_count": int(len(gates)),
    }
    return {
        "global_mapping": global_mapping,
        "corridor_mapping": corridor_mapping,
        "metrics_by_window_k": metrics_by_window,
        "metrics_by_k": metrics_by_k,
        "representatives": pd.DataFrame(representative_rows),
        "corridor_summary": pd.DataFrame(corridor_rows),
        "representative_overlap": pd.DataFrame(overlap_rows),
        "marginal_reduction": marginal,
        "validation_gates": gates,
        "summary": summary,
    }


def build_global_ffs_reduction_profile(
    hour_data: dict[int, dict],
    retained_key_audit: dict | None = None,
    cohort_audit: dict | None = None,
) -> dict[str, object]:
    results: dict[int, dict[int, dict]] = {k: {} for k in K_VALUES}
    for hour in HOURS:
        for k in K_VALUES:
            results[k][hour] = build_global_canonical_ffs(
                hour_data[hour], k,
            )
    mapping = pd.concat([
        build_ffs_mapping(
            hour_data,
            results[k],
            k=k,
            expected_rule=(
                "actual global canonical p=1 fast-forward selected support"
            ),
        )
        for k in K_VALUES
    ], ignore_index=True)

    metric_rows: list[dict] = []
    representative_rows: list[dict] = []
    branch_rows: list[dict] = []
    corridor_rows: list[dict] = []
    gate_rows: list[dict] = []
    for hour in HOURS:
        data = hour_data[hour]
        corridor_probability = {
            corridor: float(data["station_probability"][
                data["corridor"] == corridor
            ].sum())
            for corridor in CORRIDOR_ORDER
        }
        for k in K_VALUES:
            result = results[k][hour]
            part = mapping[
                mapping["hour"].eq(hour) & mapping["k"].eq(k)
            ].copy()
            selected_support = part[part["is_selected_support"]].copy()
            unrestricted_w1 = _mapping_unrestricted_joint_wasserstein(
                data, part,
            )
            native_error = float(result["native_reduction_error"])
            native_delta = abs(native_error - unrestricted_w1)
            assignment_mean = float((
                part["station_probability"] * part["ground_distance"]
            ).sum())
            assignment_max = float(part["ground_distance"].max())
            mapped_q = part.groupby("cluster")[
                "station_probability"
            ].sum().sort_index()
            reported_q = part.drop_duplicates("cluster").set_index(
                "cluster"
            )["representative_reduced_probability"].sort_index()
            q_delta = float(np.max(np.abs(
                mapped_q.to_numpy(dtype=float)
                - reported_q.to_numpy(dtype=float)
            )))
            probability_delta = abs(float(part["station_probability"].sum()) - 1.0)
            cross = part["cross_corridor_assignment"].astype(bool)
            cross_count = int(cross.sum())
            cross_mass = float(part.loc[cross, "station_probability"].sum())
            valid_status = bool(
                part["branch_status_30"].isin(
                    {"EXTRACTED", "NO_AFFECTED"}
                ).all()
                and part["branch_status_60"].isin(
                    {"EXTRACTED", "NO_AFFECTED"}
                ).all()
            )
            both_no_affected = int((
                part["branch_status_30"].eq("NO_AFFECTED")
                & part["branch_status_60"].eq("NO_AFFECTED")
            ).sum())

            corridor_counts: list[int] = []
            corridor_q: list[float] = []
            for corridor in CORRIDOR_ORDER:
                reps = selected_support[
                    selected_support["representative_corridor"].eq(corridor)
                ]
                count = int(len(reps))
                q = float(
                    reps["representative_reduced_probability"].sum()
                )
                corridor_counts.append(count)
                corridor_q.append(q)
                corridor_rows.append({
                    "hour": int(hour),
                    "k": int(k),
                    "corridor": corridor,
                    "original_corridor_probability": corridor_probability[
                        corridor
                    ],
                    "representative_count": count,
                    "representative_count_share": count / k,
                    "representative_reduced_probability": q,
                    "zero_representative_corridor": count == 0,
                })
            zero_corridors = [
                corridor for corridor, count in zip(
                    CORRIDOR_ORDER, corridor_counts
                )
                if count == 0 and corridor_probability[corridor] > 0
            ]
            zero_corridor_mass = float(sum(
                corridor_probability[corridor] for corridor in zero_corridors
            ))

            for row in selected_support.itertuples(index=False):
                representative_rows.append({
                    "hour": int(hour),
                    "k": int(k),
                    "representative_station": str(row.station),
                    "representative_corridor": str(row.corridor),
                    "reduced_probability": float(
                        row.representative_reduced_probability
                    ),
                    "selection_rule": (
                        "actual global canonical p=1 fast-forward selected support"
                    ),
                })
                for duration in DURATIONS:
                    status = str(getattr(row, f"branch_status_{duration}"))
                    branch_rows.append({
                        "hour": int(hour),
                        "k": int(k),
                        "representative_station": str(row.station),
                        "representative_corridor": str(row.corridor),
                        "duration": int(duration),
                        "reduced_probability": float(
                            row.representative_reduced_probability
                        ),
                        "duration_weight": DURATION_WEIGHT[duration],
                        "scenario_probability": float(
                            DURATION_WEIGHT[duration]
                            * row.representative_reduced_probability
                        ),
                        "source_branch_status": status,
                        "source_has_positive_impact": status == "EXTRACTED",
                    })
            local_branches = branch_rows[-k * len(DURATIONS):]
            branch_mass = {
                duration: float(sum(
                    row["scenario_probability"]
                    for row in local_branches
                    if row["duration"] == duration
                ))
                for duration in DURATIONS
            }
            gate_values = {
                "retained_observations": (
                    len(part), len(data["stations"]),
                    len(part) == len(data["stations"]),
                ),
                "actual_selected_support_count": (
                    len(selected_support), k, len(selected_support) == k,
                ),
                "selected_support_keeps_own_mass": (
                    int(selected_support["station"].eq(
                        selected_support["representative_station"]
                    ).sum()),
                    k,
                    selected_support["station"].eq(
                        selected_support["representative_station"]
                    ).all(),
                ),
                "reduced_q_exact": (q_delta, "<=1e-12", q_delta <= 1e-12),
                "location_probability_mass": (
                    probability_delta, "<=1e-12", probability_delta <= 1e-12,
                ),
                "native_error_equals_unrestricted_w1": (
                    native_delta, "<=1e-9", native_delta <= 1e-9,
                ),
                "native_error_equals_weighted_assignment_distance": (
                    abs(native_error - assignment_mean),
                    "<=1e-12",
                    abs(native_error - assignment_mean) <= 1e-12,
                ),
                "d30_branch_mass": (
                    abs(branch_mass[30] - DURATION_WEIGHT[30]),
                    "<=1e-12",
                    abs(branch_mass[30] - DURATION_WEIGHT[30]) <= 1e-12,
                ),
                "d60_branch_mass": (
                    abs(branch_mass[60] - DURATION_WEIGHT[60]),
                    "<=1e-12",
                    abs(branch_mass[60] - DURATION_WEIGHT[60]) <= 1e-12,
                ),
                "valid_duration_branch_status": (
                    int(valid_status), 1, valid_status,
                ),
                "both_no_affected_excluded": (
                    both_no_affected, 0, both_no_affected == 0,
                ),
                "no_centroid_repair_or_corridor_dp": (
                    result["representative_rule"],
                    "actual global selected support",
                    (
                        "global canonical p=1 fast-forward selected support"
                        in result["representative_rule"]
                        and "centroid" not in result["representative_rule"].lower()
                    ),
                ),
            }
            for gate, (observed, expected, passed) in gate_values.items():
                gate_rows.append({
                    "hour": int(hour),
                    "k": int(k),
                    "gate": gate,
                    "observed": observed,
                    "expected": expected,
                    "passed": bool(passed),
                })
            metric_rows.append({
                "hour": int(hour),
                "k": int(k),
                "unrestricted_joint_wasserstein": unrestricted_w1,
                "native_reduction_error": native_error,
                "native_error_wasserstein_abs_delta": native_delta,
                "probability_weighted_mean_assignment_distance": assignment_mean,
                "max_assignment_distance": assignment_max,
                "cross_corridor_assignment_count": cross_count,
                "cross_corridor_assignment_percent": 100.0 * cross_count / len(part),
                "cross_corridor_probability_mass": cross_mass,
                "zero_representative_corridor_count": len(zero_corridors),
                "zero_representative_corridor_probability_mass": (
                    zero_corridor_mass
                ),
                "max_corridor_representative_count": max(corridor_counts),
                "max_corridor_representative_count_share": (
                    max(corridor_counts) / k
                ),
                "max_corridor_representative_q": max(corridor_q),
            })

    metrics_by_window = pd.DataFrame(metric_rows).sort_values(["k", "hour"])
    numeric_metrics = [
        column for column in metrics_by_window.columns
        if column not in {"hour", "k"}
    ]
    aggregate_rows: list[dict] = []
    for k, part in metrics_by_window.groupby("k", sort=True):
        row: dict = {"k": int(k), "window_count": int(len(part))}
        for column in numeric_metrics:
            row[f"mean_{column}"] = float(part[column].mean())
            row[f"max_{column}"] = float(part[column].max())
        aggregate_rows.append(row)
    metrics_by_k = pd.DataFrame(aggregate_rows)

    marginal_rows: list[dict] = []
    for hour, part in metrics_by_window.groupby("hour", sort=True):
        previous = None
        for row in part.sort_values("k").itertuples(index=False):
            current = float(row.native_reduction_error)
            if previous is not None:
                marginal_rows.append({
                    "hour": int(hour),
                    "k": int(row.k),
                    "previous_k": int(row.k) - 1,
                    "native_error_reduction": previous - current,
                    "native_error_reduction_percent": (
                        100.0 * (previous - current) / previous
                        if previous > 0 else 0.0
                    ),
                })
            previous = current
        reductions = [
            row["native_error_reduction"]
            for row in marginal_rows if row["hour"] == int(hour)
        ]
        minimum_reduction = min(reductions) if reductions else 0.0
        gate_rows.append({
            "hour": int(hour),
            "k": -1,
            "gate": "native_error_monotone_nonincreasing_k1_20",
            "observed": minimum_reduction,
            "expected": ">=-1e-12",
            "passed": minimum_reduction >= -1e-12,
        })
    gates = pd.DataFrame(gate_rows)
    for k, count in mapping.groupby("k").size().items():
        passed = (not EXPECTED_RETAINED_STATION_HOURS
                  or int(count) == EXPECTED_RETAINED_STATION_HOURS)
        gates.loc[len(gates)] = {
            "hour": -1,
            "k": int(k),
            "gate": "retained_station_hours_all_windows",
            "observed": int(count),
            "expected": EXPECTED_RETAINED_STATION_HOURS,
            "passed": passed,
        }
    if retained_key_audit is not None:
        gates.loc[len(gates)] = {
            "hour": -1,
            "k": -1,
            "gate": "production_cohort_retained_key_set_exact",
            "observed": retained_key_audit["current_key_sha256"],
            "expected": retained_key_audit["production_key_sha256"],
            "passed": retained_key_audit["exact_key_set_match"],
        }
        gates.loc[len(gates)] = {
            "hour": -1,
            "k": -1,
            "gate": "production_cohort_per_hour_counts_exact",
            "observed": json.dumps(
                retained_key_audit["current_count_by_hour"], sort_keys=True,
            ),
            "expected": json.dumps(
                retained_key_audit["production_count_by_hour"], sort_keys=True,
            ),
            "passed": retained_key_audit["exact_per_hour_count_match"],
        }
        gates.loc[len(gates)] = {
            "hour": -1,
            "k": -1,
            "gate": "production_cohort_station_probability_exact",
            "observed": retained_key_audit[
                "maximum_station_probability_abs_delta"
            ],
            "expected": "<=1e-12",
            "passed": retained_key_audit["exact_station_probability_match"],
        }
    if cohort_audit is not None:
        gates.loc[len(gates)] = {
            "hour": -1,
            "k": -1,
            "gate": "operational_cohort_eligible_incidents",
            "observed": cohort_audit["eligible_incidents"],
            "expected": sum(COHORT_DURATION_COUNTS.values()),
            "passed": cohort_audit["eligible_incidents"] == sum(COHORT_DURATION_COUNTS.values()),
        }
        gates.loc[len(gates)] = {
            "hour": -1,
            "k": -1,
            "gate": "operational_cohort_duration_counts",
            "observed": json.dumps(
                cohort_audit["duration_counts"], sort_keys=True,
            ),
            "expected": json.dumps(
                {str(key): value for key, value in
                 COHORT_DURATION_COUNTS.items()},
                sort_keys=True,
            ),
            "passed": cohort_audit["duration_counts"] == {
                str(key): value
                for key, value in COHORT_DURATION_COUNTS.items()
            },
        }
    if not gates["passed"].all():
        failed = gates.loc[~gates["passed"]].head(20).to_dict("records")
        raise AssertionError(f"global FFS reduction validation failed: {failed}")
    summary = {
        "status": "PASS",
        "scope": "planning-window-wise global canonical FFS reduction profile",
        "hours": list(HOURS),
        "k_values": list(K_VALUES),
        "profile_k_range": f"{min(K_VALUES)}..{max(K_VALUES)}",
        "final_support_k": None,
        "final_support_status": "pending profile review",
        "retained_station_hours_per_k": EXPECTED_RETAINED_STATION_HOURS,
        "retained_key_audit": retained_key_audit,
        "operational_cohort_audit": cohort_audit,
        "ground_cost": "canonical p=1 plain Euclidean on common joint feature",
        "probability": "actual historical station probability",
        "duration_probability": {
            str(duration): DURATION_WEIGHT[duration]
            for duration in DURATIONS
        },
        "valid_duration_branch_statuses": ["EXTRACTED", "NO_AFFECTED"],
        "no_affected_feature_semantics": "zero vector",
        "both_no_affected_observations_retained": False,
        "representative_rule": (
            "actual global canonical p=1 fast-forward selected support"
        ),
        "corridor_use": "post-hoc diagnostics only",
        "corridor_reduction_or_dp_run": False,
        "other_clustering_methods_run": False,
        "centroid_or_membership_repair_used": False,
        "plots_written": False,
        "pdf_written": False,
        "production_or_manuscript_overwritten": False,
        "stage1_or_stage2_run": False,
        "gate_count": int(len(gates)),
    }
    return {
        "mapping": mapping,
        "metrics_by_window_k": metrics_by_window,
        "metrics_by_k": metrics_by_k,
        "representatives": pd.DataFrame(representative_rows),
        "branches": pd.DataFrame(branch_rows),
        "corridor_diagnostics": pd.DataFrame(corridor_rows),
        "marginal_reduction": pd.DataFrame(marginal_rows),
        "validation_gates": gates,
        "summary": summary,
    }


def apply_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 13,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_metrics(summary: pd.DataFrame, output: Path) -> None:
    apply_plot_style()
    colors = dict(zip(
        METHODS, ("#2F6DAE", "#E6862A", "#3A923A", "#C43C39")
    ))
    markers = dict(zip(METHODS, ("o", "s", "^", "P")))
    line_styles = dict(zip(METHODS, ("-", "-.", "--", ":")))
    z_orders = {
        "K-means": 5,
        "K-medoids": 3,
        "Hierarchical clustering": 7,
        "Fast-forward scenario reduction": 4,
    }
    marker_intervals = {
        "K-means": (0, 2),
        "Hierarchical clustering": (1, 2),
    }
    figure, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), sharex=True)
    panels = (
        (
            "MeanWasserstein", "SDWasserstein",
            "Wasserstein distance",
        ),
        ("MeanSSE", "SDSSE", "SSE"),
        (
            "MeanSilhouette", "SDSilhouette",
            "Silhouette coefficient",
        ),
    )
    for axis, (metric, sd_metric, title) in zip(axes, panels):
        for method in METHODS:
            part = summary[summary["method"].eq(method)].sort_values("k")
            x = part["k"].to_numpy(dtype=float)
            mean = part[metric].to_numpy(dtype=float)
            count = part["HourCount"].to_numpy(dtype=float)
            critical = np.asarray([
                student_t.ppf(0.975, df=int(value) - 1) for value in count
            ])
            half = (
                critical * part[sd_metric].to_numpy(dtype=float)
                / np.sqrt(count)
            )
            axis.fill_between(
                x,
                mean - half,
                mean + half,
                color=colors[method],
                alpha=0.11,
                linewidth=0,
                zorder=2 if method == "Hierarchical clustering" else 1,
            )
            axis.plot(
                x,
                mean,
                marker=markers[method],
                markersize=4.0,
                linewidth=1.8,
                linestyle=line_styles[method],
                color=colors[method],
                label=method,
                zorder=z_orders[method],
                markevery=marker_intervals.get(method),
            )
        axis.axvline(
            SELECTED_K, color="#555555", linestyle=":", linewidth=1.25
        )
        axis.set_ylabel(title, fontsize=11.3)
        axis.set_xlabel("K (representative stations per hour)")
        axis.set_xticks(range(6, 21, 2))
        axis.set_axisbelow(True)
        axis.grid(
            axis="y",
            linestyle=GRID_LINESTYLE,
            linewidth=GRID_LINEWIDTH,
            color=GRID_COLOR,
            alpha=0.9,
        )
        axis.grid(False, axis="x")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor=LEGEND_EDGE_COLOR,
        fontsize=9.0,
    )
    figure.subplots_adjust(
        left=0.055, right=0.992, top=0.91, bottom=0.24, wspace=0.19
    )
    stem = output / f"Fig_four_methods_wasserstein_sse_silhouette_K{min(K_VALUES)}_{max(K_VALUES)}"
    figure.savefig(stem.with_suffix(".png"), dpi=300)
    figure.savefig(
        output / "clustering_method_metrics_{}.png".format("_".join(map(str, DURATIONS))),
        dpi=300,
    )
    plt.close(figure)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR
    )
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument(
        "--production-kmeans-dir",
        type=Path,
        default=DEFAULT_PRODUCTION_KMEANS_DIR,
        help="K-means output of 02_scenario_reduction.py, used only for comparison",
    )
    result.add_argument(
        "--history-cohort-file",
        type=Path,
        default=cfg.path("inputs.incident_cohort"),
        help=("incident-level cohort file (default: inputs.incident_cohort); rows "
              "with Eligible=true define the historical location and duration priors"),
    )
    result.add_argument(
        "--windows",
        default=",".join(str(hour) for hour in WINDOWS),
        help="planning-window start hours within time.windows, e.g. 6-20, 6, or 6,8,17",
    )
    result.add_argument(
        "--read-workers", type=int,
        default=int(cfg.get("clustering.reduction.read_workers")),
    )
    result.add_argument(
        "--station-smoothing-alpha",
        type=float,
        default=STATION_SMOOTHING_ALPHA,
        help=("nonnegative Laplace smoothing added to historical station "
              "counts; zero uses unsmoothed empirical frequencies"),
    )
    result.add_argument(
        "--selected-method",
        choices=METHODS,
        default=CLUSTERING_METHOD,
        help="method whose clustering.k_per_window representative support is written",
    )
    result.add_argument(
        "--selected-k-only",
        action="store_true",
        help=("evaluate clustering.k_per_window only instead of the diagnostic "
              "clustering.reduction.k_candidates sweep"),
    )
    result.add_argument(
        "--location-support",
        choices=("all-duration", "any-duration"),
        default="all-duration",
        help=(
            "require a valid EXTRACTED/NO_AFFECTED source in every duration "
            "or at least one duration; NO_AFFECTED impact features are zero"
        ),
    )
    result.add_argument(
        "--check-only",
        action="store_true",
        help="validate inputs and 30/60 support without writing outputs",
    )
    result.add_argument(
        "--write-diagnostic-plot",
        action="store_true",
        help="optionally write the K-sweep PNG; no PDF is ever written",
    )
    result.add_argument(
        "--global-ffs-reduction-profile-only",
        action="store_true",
        help=(
            "run only planning-window-wise global canonical p=1 FFS for "
            "clustering.reduction.global_ffs_profile_k; skip corridor reduction, other methods, and plots"
        ),
    )
    return result


def parse_windows(value: str) -> tuple[int, ...]:
    selected: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = (int(item) for item in part.split("-", 1))
            if left > right:
                raise ValueError("window range start must not exceed its end")
            selected.update(range(left, right + 1))
        else:
            selected.add(int(part))
    result = tuple(sorted(selected))
    if not result or any(hour not in WINDOWS for hour in result):
        raise ValueError(f"--windows must select start hours within time.windows {WINDOWS}")
    return result


def missing_scenario_blocks(
    scenario_dir: Path, hours: tuple[int, ...]
) -> list[str]:
    missing = []
    for hour in hours:
        for duration in DURATIONS:
            if not any(scenario_dir.glob(f"Disrupt_{hour}_*_{duration}.csv")):
                missing.append(f"{hour:02d}/{duration}")
    return missing


def main(argv: list[str] | None = None) -> None:
    global HOURS, K_VALUES, DURATION_WEIGHT
    args = parser().parse_args(argv)
    HOURS = parse_windows(args.windows)
    if args.global_ffs_reduction_profile_only and HOURS != WINDOWS:
        raise ValueError(
            f"global FFS profile requires all planning windows {WINDOWS}"
        )
    if args.global_ffs_reduction_profile_only and args.selected_k_only:
        raise ValueError(
            "global FFS profile requires the complete "
            "clustering.reduction.global_ffs_profile_k sweep"
        )
    if (
        args.global_ffs_reduction_profile_only
        and args.history_cohort_file is None
    ):
        raise ValueError(
            "global FFS profile requires --history-cohort-file for the "
            "operational cohort"
        )
    if args.global_ffs_reduction_profile_only:
        K_VALUES = GLOBAL_FFS_PROFILE_K_VALUES
    elif args.selected_k_only:
        K_VALUES = (SELECTED_K,)
    if args.read_workers <= 0:
        raise ValueError("--read-workers must be positive")
    if args.station_smoothing_alpha < 0:
        raise ValueError("--station-smoothing-alpha must be nonnegative")
    augmented_file = require_file(
        cfg.path("inputs.incidents_gt10"), "augmented incident data"
    )
    raw_incident_file = require_file(
        cfg.path("inputs.incidents_raw"), "raw incident data"
    )

    station_corridor, station_families = build_corridor_map()
    cohort_file = (
        require_file(args.history_cohort_file, "operational cohort")
        if args.history_cohort_file is not None else None
    )
    history, history_audit = load_history(
        augmented_file, raw_incident_file, cohort_file)
    duration_counts = {
        duration: int(history["duration_state"].eq(duration).sum())
        for duration in DURATIONS
    }
    duration_total = sum(duration_counts.values())
    if duration_total <= 0 or any(
            duration_counts[duration] <= 0 for duration in DURATIONS):
        raise ValueError(
            f"cohort must contain both duration states: {duration_counts}")
    if args.global_ffs_reduction_profile_only:
        if len(history) != sum(COHORT_DURATION_COUNTS.values()):
            raise ValueError(
                "global FFS profile requires the eligible incidents of "
                "probability.duration_counts; "
                f"observed={len(history)}"
            )
        if duration_counts != COHORT_DURATION_COUNTS:
            raise ValueError(
                "global FFS profile duration counts changed: "
                f"expected={COHORT_DURATION_COUNTS}, "
                f"observed={duration_counts}"
            )
    DURATION_WEIGHT = {
        duration: duration_counts[duration] / duration_total
        for duration in DURATIONS
    }
    (
        corridor_probability,
        corridor_table,
        corridor_history_n,
        station_counts,
        station_history_n,
    ) = historical_corridor_probability(history, station_corridor)

    scenario_dir = args.scenario_dir.resolve()
    missing_blocks = missing_scenario_blocks(scenario_dir, HOURS)
    if args.check_only and missing_blocks:
        print(json.dumps({
            "mode": "check_only",
            "status": "PENDING_PREVIOUS_STAGE",
            "hours": list(HOURS),
            "durations": list(DURATIONS),
            "missing_scenario_blocks": missing_blocks,
            "history": {
                **history_audit,
                "corridor_probability_rows": corridor_history_n,
                "station_frequency_rows": station_history_n,
                "station_missing_rows": int(len(history) - station_history_n),
            },
        }, ensure_ascii=False, indent=2))
        return

    support, valid_support, support_audit = discover_scenarios(
        scenario_dir, int(args.read_workers), args.location_support
    )
    unknown = sorted(
        set(support["station"]).union(valid_support["station"])
        - set(station_corridor)
    )
    if unknown:
        raise RuntimeError(f"stations missing from corridor map: {unknown}")

    check_report = {
        "mode": "check_only" if args.check_only else "run",
        "selected_k": (
            None if args.global_ffs_reduction_profile_only else SELECTED_K
        ),
        "selected_method": (
            "planning-window-wise global canonical FFS profile"
            if args.global_ffs_reduction_profile_only else args.selected_method
        ),
        "hours": list(HOURS),
        "durations": list(DURATIONS),
        "duration_probability": {
            str(key): value for key, value in DURATION_WEIGHT.items()
        },
        "station_smoothing_alpha": float(args.station_smoothing_alpha),
        "location_support_rule": args.location_support,
        "central_shared": sorted(network_data.central_shared_stations()),
        "scenario_support": support_audit,
        "history": {
            **history_audit,
            "corridor_probability_rows": corridor_history_n,
            "station_frequency_rows": station_history_n,
            "station_missing_rows": int(len(history) - station_history_n),
        },
    }
    if args.check_only:
        print(json.dumps(check_report, ensure_ascii=False, indent=2))
        return

    output = args.output.resolve()
    if (
        args.global_ffs_reduction_profile_only
        and output == DEFAULT_OUTPUT.resolve()
    ):
        raise ValueError(
            "global FFS profile requires an explicit isolated --output directory"
        )
    if (
        (
            args.selected_method == METHODS[3]
            or args.global_ffs_reduction_profile_only
        )
        and output == args.production_kmeans_dir.resolve()
    ):
        raise ValueError(
            "canonical FFS output must not overwrite production K-means"
        )
    output.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    _feature_names, raw_features = load_impact_features(
        support, int(args.read_workers)
    )

    if args.global_ffs_reduction_profile_only:
        hour_data: dict[int, dict] = {}
        hourly_probability_rows: list[dict] = []
        for hour in HOURS:
            data = build_hour(
                support, valid_support, raw_features, hour,
                args.location_support,
            )
            hourly_probability_rows.extend(apply_corridor_probability(
                data,
                station_corridor,
                corridor_probability,
                station_counts,
                float(args.station_smoothing_alpha),
            ))
            hour_data[hour] = data
        retained_key_audit = build_retained_key_audit(
            hour_data, args.production_kmeans_dir,
        )
        cohort_audit = {
            "cohort_file": portable(cohort_file),
            "eligible_incidents": int(len(history)),
            "duration_counts": {
                str(key): value for key, value in duration_counts.items()
            },
        }
        result = build_global_ffs_reduction_profile(
            hour_data, retained_key_audit, cohort_audit,
        )
        write_csv(
            pd.DataFrame(hourly_probability_rows),
            output / output_name("station_probability"),
        )
        for key, filename in (
            ("mapping", "global_ffs_mapping_by_window_k.csv"),
            (
                "metrics_by_window_k",
                "global_ffs_metrics_by_window_k.csv",
            ),
            ("metrics_by_k", "global_ffs_metrics_by_k.csv"),
            ("representatives", "global_ffs_representatives.csv"),
            ("branches", "global_ffs_branches_by_window_k.csv"),
            (
                "corridor_diagnostics",
                "global_ffs_posthoc_corridor_diagnostics.csv",
            ),
            (
                "marginal_reduction",
                "global_ffs_marginal_reduction.csv",
            ),
            (
                "validation_gates",
                "global_ffs_validation_gates.csv",
            ),
        ):
            write_csv(result[key], output / filename)
        profile_audit = {
            **check_report,
            **result["summary"],
            "mode": "global_ffs_reduction_profile_only",
            "output_directory": portable(output),
        }
        (output / "global_ffs_reduction_profile_summary.json").write_text(
            json.dumps(profile_audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            "Planning-window-wise global canonical FFS profile (clustering.reduction.global_ffs_profile_k) "
            f"saved to {output}",
            flush=True,
        )
        return

    hour_data: dict[int, dict] = {}
    metric_rows: list[dict] = []
    corridor_metric_rows: list[dict] = []
    allocation_rows: list[dict] = []
    hourly_probability_rows: list[dict] = []
    selected_by_method: dict[str, dict[int, dict]] = {
        method: {} for method in METHODS
    }
    for hour in HOURS:
        data = build_hour(
            support, valid_support, raw_features, hour, args.location_support
        )
        hourly_probability_rows.extend(apply_corridor_probability(
            data,
            station_corridor,
            corridor_probability,
            station_counts,
            float(args.station_smoothing_alpha),
        ))
        hour_data[hour] = data
        all_candidates = candidate_partitions(data, max(K_VALUES))
        for method in METHODS:
            for k in K_VALUES:
                result, rows = combine_partition(
                    data, all_candidates[method], k
                )
                metric, corridor_metrics = evaluate(
                    data, result, method, k
                )
                metric_rows.append(metric)
                corridor_metric_rows.extend(corridor_metrics)
                for row in rows:
                    row["method"] = method
                    allocation_rows.append(row)
                if k == SELECTED_K:
                    selected_by_method[method][hour] = result
        print(
            f"hour {hour}: four methods K={min(K_VALUES)}..{max(K_VALUES)}",
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    expected_metrics = len(HOURS) * len(K_VALUES) * len(METHODS)
    if len(metrics) != expected_metrics:
        raise AssertionError(
            f"metric grid incomplete: expected={expected_metrics}, "
            f"observed={len(metrics)}"
        )
    summary = summarize(metrics)
    comparison = summary[summary["k"].eq(SELECTED_K)].sort_values(
        ["MeanWasserstein", "MeanSilhouette"],
        ascending=[True, False],
    ).reset_index(drop=True)
    if len(comparison) != len(METHODS):
        raise AssertionError(
            f"K={SELECTED_K} method comparison is incomplete"
        )
    comparison.insert(0, "PriorityRank", np.arange(1, len(comparison) + 1))
    if args.selected_method not in set(comparison["method"]):
        raise AssertionError(
            f"selected clustering method is unavailable: {args.selected_method}"
        )
    comparison["SelectedForIsolatedOutput"] = comparison["method"].eq(
        args.selected_method
    )
    selected_method = args.selected_method

    representatives, branches, assignments = build_outputs(
        hour_data, selected_by_method[selected_method]
    )
    representatives["method"] = selected_method
    branches["method"] = selected_method
    assignments["method"] = selected_method
    for frame, station_column in (
        (representatives, "representative_station"),
        (branches, "representative_station"),
        (assignments, "station"),
    ):
        frame["corridor"] = frame[station_column].map(station_corridor)
    assignments["representative_corridor"] = assignments[
        "representative_station"
    ].map(station_corridor)
    validate_outputs(representatives, branches, assignments)

    ffs_mapping = None
    validation_gates = None
    validation_summary = None
    production_detail = None
    production_representative_changes = None
    production_corridor_counts = None
    production_by_hour = None
    production_summary = None
    if selected_method == METHODS[3]:
        ffs_mapping = build_ffs_mapping(
            hour_data, selected_by_method[selected_method]
        )
        validation_gates, validation_summary = build_ffs_validation(
            hour_data,
            selected_by_method[selected_method],
            metrics,
            representatives,
            branches,
            assignments,
            ffs_mapping,
        )
        (
            production_detail,
            production_representative_changes,
            production_corridor_counts,
            production_by_hour,
            production_summary,
        ) = compare_ffs_with_production_kmeans(
            args.production_kmeans_dir,
            hour_data,
            metrics,
            ffs_mapping,
        )

    corridor_definition = pd.DataFrame([
        {
            "station": station,
            "corridor": station_corridor[station],
            "track_families": ";".join(station_families[station]),
        }
        for station in sorted(station_corridor)
    ])
    write_csv(
        corridor_definition,
        output / output_name("corridor_definition"),
    )
    write_csv(
        corridor_table,
        output / output_name("corridor_probability"),
    )
    write_csv(
        pd.DataFrame(hourly_probability_rows),
        output / output_name("station_probability"),
    )
    write_csv(metrics, output / "method_k_metrics_by_hour.csv")
    write_csv(
        pd.DataFrame(corridor_metric_rows),
        output / "method_k_metrics_by_hour_corridor.csv",
    )
    write_csv(summary, output / "method_k_metrics_summary.csv")
    write_csv(
        comparison, output / output_name("comparison")
    )
    write_csv(
        pd.DataFrame(allocation_rows),
        output / "corridor_k_allocation_by_method_hour.csv",
    )
    write_csv(
        representatives,
        output
        / output_name("representatives"),
    )
    write_csv(
        branches,
        output / output_name("branches"),
    )
    write_csv(
        assignments,
        output / output_name("assignments"),
    )
    if ffs_mapping is not None:
        write_csv(
            ffs_mapping,
            output / output_name("ffs_mapping"),
        )
        write_csv(
            validation_gates,
            output / "canonical_ffs_validation_gates.csv",
        )
        (output / "canonical_ffs_validation_summary.json").write_text(
            json.dumps(validation_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_csv(
            production_detail,
            output / "ffs_vs_production_kmeans_station_mapping.csv",
        )
        write_csv(
            production_representative_changes,
            output / "ffs_vs_production_kmeans_representative_changes.csv",
        )
        write_csv(
            production_corridor_counts,
            output / "ffs_vs_production_kmeans_representative_corridor_counts.csv",
        )
        write_csv(
            production_by_hour,
            output / "ffs_vs_production_kmeans_by_hour.csv",
        )
        write_csv(
            production_summary,
            output / "ffs_vs_production_kmeans_summary.csv",
        )
    if args.write_diagnostic_plot and not args.selected_k_only:
        plot_metrics(summary, output)

    audit = {
        **check_report,
        "mode": "completed",
        "selected_method": selected_method,
        "selected_k": SELECTED_K,
        "method_selection_policy": (
            f"user-prespecified {selected_method}; cross-method metrics are "
            "reported as diagnostics only"
        ),
        "method_ranking_priority": [
            "mean corridor-constrained joint-feature discrete 1-Wasserstein (lower)",
            "mean unweighted within-corridor silhouette (higher)",
        ],
        "sse_policy": (
            "raw unweighted within-cluster squared Euclidean distances, "
            "summed across corridors; compactness and elbow diagnostic only "
            "and excluded from cross-method ranking because its scale depends "
            "on the feature representation"
        ),
        "silhouette_scope": (
            "alternative clusters are restricted to the same corridor; "
            "every station scenario has equal weight and observations in a "
            "corridor with one cluster contribute zero"
        ),
        "metric_aggregation": (
            "SSE is summed across within-corridor clusters without probability "
            "weights; silhouette is the unweighted mean across station "
            "scenarios after within-corridor evaluation; joint Wasserstein is "
            "the corridor-probability-weighted sum of conditional transport"
        ),
        "probability_weighting_by_metric": {
            "SSE": False,
            "Silhouette coefficient": False,
            "Wasserstein distance": True,
        },
        "clustering_observation_weighting": (
            "actual historical station probability for all four methods"
        ),
        "clustering_duration_weighting": (
            "sqrt(P(duration))-scaled feature blocks, so squared Euclidean "
            "distance contributions are proportional to historical "
            "duration probability"
        ),
        "corridor_k_allocation_objective_by_method": {
            METHODS[0]: "unconditional probability-weighted SSE",
            METHODS[1]: (
                "unconditional probability-weighted medoid L2 loss"
            ),
            METHODS[2]: "unconditional probability-weighted SSE",
            METHODS[3]: (
                "unconditional canonical p=1 FFS reduction error"
            ),
        },
        "allocation_objective_cross_method_ranking_allowed": False,
        "wasserstein_transport_scope": (
            "original probability mass can be transported only to "
            "representatives in the same corridor"
        ),
        "corridor_metric_output": (
            "method_k_metrics_by_hour_corridor.csv"
        ),
        "r2_policy": "omitted from reported metrics and method ranking",
        "methods": list(METHODS),
        "k_values": list(K_VALUES),
        "location_probability": (
            f"cohort P(corridor) from all {corridor_history_n} retained "
            "incidents, then "
            + ("Laplace-smoothed " if args.station_smoothing_alpha > 0 else
               "unsmoothed empirical ")
            + f"cohort P(station|corridor) from {station_history_n} "
            "station-known incidents; inactive corridors are removed and remaining "
            "corridor mass is renormalized"
        ),
        "laplace_smoothing_used": bool(args.station_smoothing_alpha > 0),
        "station_smoothing_alpha": float(args.station_smoothing_alpha),
        "machine_learning_probability_used": False,
        "cross_corridor_assignments": 0,
        "feature": (
            "sqrt(P(duration))-scaled concatenation of 30/60 log1p "
            "Queue+Walking vectors"
        ),
        "canonical_ffs_ground_cost": (
            "H-R p=1 plain Euclidean distance on the joint feature"
        ),
        "legacy_ground_cost_not_used": (
            "p=2 Fortet-Mourier generalized cost with radius multiplier"
        ),
        "representative_rule_by_method": {
            METHODS[0]: "valid in-cluster station nearest weighted centroid",
            METHODS[1]: "actual weighted K-medoids medoid",
            METHODS[2]: "valid in-cluster station nearest weighted centroid",
            METHODS[3]: "actual canonical p=1 fast-forward selected support",
        },
        "selected_representative_rule": (
            selected_by_method[selected_method][HOURS[0]][
                "representative_rule"
            ]
        ),
        "out_of_cluster_representative_fallback_allowed": False,
        "out_of_cluster_representative_fallbacks": int(
            representatives["centroid_fallback_used"].sum()
        ),
        "representative_membership_repairs": int(
            representatives["representative_membership_repair_used"].sum()
        ),
        "representative_membership_repair_policy": (
            "none; representatives are selected only from their original cluster"
        ),
        "canonical_ffs_validation": validation_summary,
        "production_kmeans_comparison_directory": (
            portable(args.production_kmeans_dir)
            if selected_method == METHODS[3] else None
        ),
        "diagnostic_plot_written": bool(
            args.write_diagnostic_plot and not args.selected_k_only
        ),
        "pdf_written": False,
        "output_directory": portable(output),
    }
    (output / "clustering_comparison_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"\nK={SELECTED_K} comparison (Wasserstein, silhouette)",
        flush=True,
    )
    print(comparison.to_string(index=False), flush=True)
    print(f"\nSelected: {selected_method}; saved to {output}", flush=True)


if __name__ == "__main__":
    main()
