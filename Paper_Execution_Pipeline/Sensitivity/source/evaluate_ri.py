"""Evaluate resilience and passenger-time outcomes.

Disruption files contain physically affected ODs.  Causal Stage-2 files may
instead contain the station-level union of the D30/D60 supports.  For the
resilience index (RI), ODs outside the applicable output support are restored
from the matching Normal window when full-OD evaluation is selected.
Passenger-time outcomes are kept separate: Queue, Load, and Walking inventories
are filtered back to physically affected ODs and divided by their passenger
demand present at t0-1 plus demand arriving from t0 onward.  Both definitions
use the post-disruption horizon t = time.evaluation_start_min .. time.horizon_min
(paper: 10..240).  All paths and settings come from the active bbconfig profile.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parents[3]))
from bbconfig import cfg  # noqa: E402

DURATIONS = tuple(int(value) for value in cfg.get("time.durations_min"))
# Disruption rows weight a duration branch by the Table 3 duration prior pi_tau
# (probability.duration_counts: pi_30 = 208/303, pi_60 = 95/303 in the paper).
# The value enters only the Duration/Scenario_Probability columns of
# Disruption_RI.csv; Bridging rows use each case's own Stage 2 duration weights.
_PRIOR_COUNTS = {
    int(key): int(value)
    for key, value in cfg.get("probability.duration_counts").items()
}
DURATION_WEIGHTS = {
    duration: _PRIOR_COUNTS[duration] / sum(_PRIOR_COUNTS.values())
    for duration in DURATIONS
}
OD_HOURLY_MIN_TRIPS = float(cfg.get("operations.od_hourly_min_trips"))
WINDOWS = tuple(int(value) for value in cfg.get("time.windows"))
WINDOW_HOURS = int(cfg.get("time.horizon_min")) // 60

RESULTS = cfg.results_dir
CORRECTED_RESULTS = RESULTS / cfg.get("paths.sensitivity")
NORMAL_DIR = RESULTS / cfg.get("paths.normal")
DISRUPTION_DIR = RESULTS / cfg.get("paths.disruptions")
OD_FILE = cfg.path("inputs.od_matrix")
SCENARIO_STATS = DISRUPTION_DIR / "_scenario_stats.csv"
STATION_PROBABILITY_FILE = (
    cfg.package_path("pinned.clustering.station_probability")
    if cfg.get("pinned.use_pinned")
    else RESULTS / cfg.get("paths.clustering")
    / cfg.get("clustering.files.station_probability")
)
BASELINE_DIR = RESULTS / cfg.get("paths.stage2_proposed")
FIXED_SPE_DIR = RESULTS / cfg.get("paths.stage2_benchmark")
PROPOSED_CASE = str(cfg.get("evaluation.ri.proposed_case"))
BENCHMARK_CASE = f"FIXED_{cfg.get('evaluation.benchmark_depot')}"
LOGIC_VERSION = str(cfg.get("scenarios.logic_version"))
TARGET_VARIABLES = frozenset({"Queue", "Load", "Walking"})
TIME_INDEX = pd.Index(range(int(cfg.get("time.horizon_min")) + 1), name="Time")
DISRUPTION_START = int(cfg.get("time.disruption_onset_min"))
CONDITION_CACHE_SCHEMA = "ri-condition-data-v1"
CONDITION_CACHE_ENABLED = True
CONDITION_CACHE_DIR = Path(
    os.path.abspath(
        os.environ.get("PAPER_RI_CACHE_DIR")
        or str(RESULTS / cfg.get("paths.cache") / "ri")
    )
)
CONDITION_CACHE_STATS = {
    "hits": 0,
    "misses": 0,
    "writes": 0,
    "read_errors": 0,
    "write_errors": 0,
    "bypassed": 0,
}
_OD_INPUT_MEMORY_CACHE: dict[tuple[str, int, int], pd.DataFrame] = {}
SCENARIO_FILE_RE = re.compile(
    r"^Disrupt_(?P<window>\d+)_(?P<station>.+)_(?P<duration>\d+)\.csv$"
)


@dataclass(frozen=True)
class Scenario:
    window: int
    station: str
    duration: int
    disruption_file: Path | None

    @property
    def scenario_id(self) -> str:
        return f"{self.window:02d}_{self.station}_{self.duration}"


@dataclass
class NormalWindow:
    source_file: Path
    od_set: frozenset[str]
    qlw_by_od_time: pd.Series
    total_qlw: pd.Series
    state_by_od_variable_time: pd.Series
    total_state_by_variable_time: pd.Series
    demand_by_od: pd.Series
    initial_passengers_by_od: pd.Series
    post_disruption_demand_by_od: pd.Series
    total_demand: float


@dataclass
class ConditionData:
    od_set: frozenset[str]
    total_qlw: pd.Series
    state_by_od_variable_time: pd.Series
    state_by_variable_time: pd.Series


@dataclass
class ScenarioMerge:
    scenario: Scenario
    normal: NormalWindow
    affected_ods: frozenset[str]
    common_support_ods: frozenset[str]
    reference_qlw: pd.Series
    unaffected_qlw: pd.Series
    common_unaffected_qlw: pd.Series
    unaffected_state_by_variable_time: pd.Series
    affected_passenger_demand: float
    disrupted_ri: float
    disrupted_time_metrics: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root", type=Path, default=CORRECTED_RESULTS,
        help="isolated sensitivity result root",
    )
    parser.add_argument(
        "--baseline-dir", type=Path, default=BASELINE_DIR,
        help="matching causal Stage-2 baseline",
    )
    parser.add_argument(
        "--benchmark-dir", "--fixed-spe-dir", dest="fixed_spe_dir",
        type=Path, default=FIXED_SPE_DIR,
        help=f"fixed benchmark Stage-2 results (case id {BENCHMARK_CASE})",
    )
    parser.add_argument("--normal-dir", type=Path, default=NORMAL_DIR)
    parser.add_argument(
        "--station-probability-file", type=Path,
        default=STATION_PROBABILITY_FILE,
        help="historical station probability per (hour, station); "
             "only the Station/Scenario_Probability columns depend on it",
    )
    parser.add_argument("--disruption-dir", type=Path, default=DISRUPTION_DIR)
    parser.add_argument(
        "--cases", nargs="+",
        help=(f"case IDs: {PROPOSED_CASE}, {BENCHMARK_CASE}, BASELINE and/or "
              "sensitivity case IDs; default = every case under --results-root"),
    )
    parser.add_argument(
        "--windows", nargs="+", type=int,
        default=[int(value) for value in cfg.get("evaluation.sensitivity.windows")],
    )
    parser.add_argument("--stations", nargs="+", help="optional station-code filter")
    parser.add_argument("--durations", nargs="+", type=int, default=list(DURATIONS))
    parser.add_argument(
        "--od-scope",
        choices=["full", "affected"],
        default=str(cfg.get("evaluation.ri.od_scope")),
        help="evaluate RI over the full Normal OD universe or affected ODs only",
    )
    parser.add_argument(
        "--evaluation-start",
        type=int,
        default=int(cfg.get("time.evaluation_start_min")),
        help="first minute included in the RI evaluation horizon",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="skip missing Bridging files instead of failing",
    )
    parser.add_argument(
        "--condition-cache-dir",
        type=Path,
        default=None,
        help=(
            "persistent local cache for parsed DecisionVars/disruption CSVs; "
            "defaults to <results>/<paths.cache>/ri of the active profile"
        ),
    )
    parser.add_argument(
        "--no-condition-cache",
        action="store_true",
        help="read source CSVs directly without the persistent local cache",
    )
    args = parser.parse_args()
    args.normal_dir = args.normal_dir.resolve()
    args.disruption_dir = args.disruption_dir.resolve()
    invalid_windows = sorted(set(args.windows) - set(WINDOWS))
    invalid_durations = sorted(set(args.durations) - set(DURATIONS))
    if invalid_windows:
        parser.error(f"unsupported windows: {invalid_windows}")
    if invalid_durations:
        parser.error(f"unsupported durations: {invalid_durations}")
    if not TIME_INDEX.min() <= args.evaluation_start < TIME_INDEX.max():
        parser.error(
            f"evaluation start must be in "
            f"[{TIME_INDEX.min()}, {TIME_INDEX.max() - 1}]"
        )
    if args.stations:
        args.stations = [str(value).strip().upper() for value in args.stations]
    args.results_root = args.results_root.resolve()
    args.baseline_dir = args.baseline_dir.resolve()
    args.fixed_spe_dir = args.fixed_spe_dir.resolve()
    if args.condition_cache_dir is not None:
        args.condition_cache_dir = Path(
            os.path.abspath(os.fspath(args.condition_cache_dir))
        )
    args.output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.results_root / cfg.get("evaluation.ri.sensitivity_output")
    )
    return args


def clean_od_values(frame: pd.DataFrame) -> frozenset[str]:
    values = frame["Omega"].dropna().astype(str).str.strip()
    return frozenset(values[values.ne("")])


def state_totals(frame: pd.DataFrame) -> pd.Series:
    state = frame[frame["Variable"].isin(TARGET_VARIABLES)].copy()
    state["Time"] = pd.to_numeric(state["Time"], errors="coerce")
    state["Value"] = pd.to_numeric(state["Value"], errors="coerce")
    state = state.dropna(subset=["Time", "Value"])
    state = state[state["Time"].between(TIME_INDEX.min(), TIME_INDEX.max())]
    state["Time"] = state["Time"].astype(int)
    return state.groupby(["Variable", "Time"], sort=True)["Value"].sum()


def state_by_od_totals(frame: pd.DataFrame) -> pd.Series:
    state = frame[frame["Variable"].isin(TARGET_VARIABLES)].copy()
    state = state[state["Omega"].notna()]
    state["Omega"] = state["Omega"].astype(str).str.strip()
    state["Time"] = pd.to_numeric(state["Time"], errors="coerce")
    state["Value"] = pd.to_numeric(state["Value"], errors="coerce")
    state = state.dropna(subset=["Time", "Value"])
    state = state[
        state["Omega"].ne("")
        & state["Time"].between(TIME_INDEX.min(), TIME_INDEX.max())
    ]
    state["Time"] = state["Time"].astype(int)
    return state.groupby(
        ["Omega", "Variable", "Time"], sort=False
    )["Value"].sum()


def qlw_totals(state_by_variable_time: pd.Series) -> pd.Series:
    if state_by_variable_time.empty:
        return pd.Series(0.0, index=TIME_INDEX, dtype=float)
    return (
        state_by_variable_time.groupby(level="Time").sum()
        .reindex(TIME_INDEX, fill_value=0.0)
        .astype(float)
    )


def absolute_path_text(path: Path) -> str:
    """Return an absolute path without Drive-backed canonicalization I/O."""
    return os.path.abspath(os.fspath(path))


def source_signature(path: Path) -> dict[str, object]:
    source = Path(path)
    stat = source.stat()
    return {
        "path": os.path.normcase(absolute_path_text(source)),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "cache_schema": CONDITION_CACHE_SCHEMA,
        "pandas_version": pd.__version__,
    }


def condition_cache_file(
    signature: dict[str, object],
    cache_dir: Path,
) -> Path:
    # One cache file per canonical source path.  Size/mtime remain in the
    # record and invalidate its payload, while replacement avoids accumulating
    # stale generations whenever a result CSV is regenerated in place.
    identity = {
        "path": signature["path"],
        "cache_schema": signature["cache_schema"],
        "pandas_version": signature["pandas_version"],
    }
    encoded = json.dumps(
        identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return cache_dir / CONDITION_CACHE_SCHEMA / digest[:2] / f"{digest}.pkl"


def condition_payload(condition: ConditionData) -> dict[str, object]:
    # Do not pickle ConditionData itself: the evaluator is dynamically imported
    # under different module names by OOF and standalone workflows.
    return {
        "od_set": tuple(sorted(condition.od_set)),
        "total_qlw": condition.total_qlw,
        "state_by_od_variable_time": condition.state_by_od_variable_time,
        "state_by_variable_time": condition.state_by_variable_time,
    }


def condition_from_payload(payload: object) -> ConditionData:
    if not isinstance(payload, dict):
        raise ValueError("condition cache payload is not a mapping")
    required = {
        "od_set", "total_qlw", "state_by_od_variable_time",
        "state_by_variable_time",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"condition cache payload misses {sorted(missing)}")
    series_fields = (
        "total_qlw", "state_by_od_variable_time", "state_by_variable_time",
    )
    if any(not isinstance(payload[name], pd.Series) for name in series_fields):
        raise ValueError("condition cache payload has a non-Series state")
    return ConditionData(
        frozenset(str(value) for value in payload["od_set"]),
        payload["total_qlw"],
        payload["state_by_od_variable_time"],
        payload["state_by_variable_time"],
    )


def write_pickle_atomic(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_condition_csv(path: Path) -> ConditionData:
    required = {"Variable", "Omega", "Time", "Value"}
    frame = pd.read_csv(path, usecols=lambda column: column in required)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    od_set = clean_od_values(frame)
    state = state_totals(frame)
    state_by_od = state_by_od_totals(frame)
    return ConditionData(od_set, qlw_totals(state), state_by_od, state)


def read_condition(
    path: Path,
    *,
    cache_dir: Path | None = None,
    use_cache: bool | None = None,
) -> ConditionData:
    enabled = CONDITION_CACHE_ENABLED if use_cache is None else bool(use_cache)
    root = CONDITION_CACHE_DIR if cache_dir is None else Path(cache_dir)
    if not enabled:
        CONDITION_CACHE_STATS["bypassed"] += 1
        return read_condition_csv(path)

    signature = source_signature(path)
    cache_file = condition_cache_file(signature, root)
    if cache_file.is_file():
        try:
            with cache_file.open("rb") as stream:
                record = pickle.load(stream)
            if (
                not isinstance(record, dict)
                or record.get("schema") != CONDITION_CACHE_SCHEMA
                or record.get("source") != signature
            ):
                raise ValueError("condition cache metadata mismatch")
            condition = condition_from_payload(record.get("payload"))
            CONDITION_CACHE_STATS["hits"] += 1
            return condition
        except (
            OSError, EOFError, pickle.PickleError, AttributeError,
            ImportError, TypeError, ValueError,
        ):
            CONDITION_CACHE_STATS["read_errors"] += 1

    CONDITION_CACHE_STATS["misses"] += 1
    condition = read_condition_csv(path)
    record = {
        "schema": CONDITION_CACHE_SCHEMA,
        "source": signature,
        "payload": condition_payload(condition),
    }
    try:
        write_pickle_atomic(record, cache_file)
        CONDITION_CACHE_STATS["writes"] += 1
    except OSError:
        # Cache availability must never change the RI/PT calculation result.
        CONDITION_CACHE_STATS["write_errors"] += 1
    return condition


def condition_state_for_ods(
    condition: ConditionData,
    ods: frozenset[str],
) -> pd.Series:
    """Return Q/L/W state aggregated only over the requested physical ODs."""
    state = condition.state_by_od_variable_time
    if state.empty or not ods:
        return pd.Series(dtype=float)
    selected = state[
        state.index.get_level_values("Omega").isin(ods)
    ]
    if selected.empty:
        return pd.Series(dtype=float)
    return selected.groupby(["Variable", "Time"], sort=True).sum()


def bridging_condition_for_ri(
    bridging: ConditionData,
    physical_affected_ods: frozenset[str],
    physical_unaffected_qlw: pd.Series,
    common_unaffected_qlw: pd.Series,
    od_scope: str,
    declared_support_mode: str | None = None,
) -> tuple[pd.Series, pd.Series, str]:
    """Build the RI state without counting common-only Normal ODs twice."""
    physical_state = condition_state_for_ods(
        bridging, physical_affected_ods
    )
    physical_qlw = qlw_totals(physical_state)
    if declared_support_mode is None:
        support_mode = (
            "common-support-v4"
            if bridging.od_set - physical_affected_ods
            else "physical-support-legacy"
        )
    elif declared_support_mode in {
        "common-support-v4",
        "physical-support-legacy",
        "physical-support-bus-only",
    }:
        support_mode = declared_support_mode
    else:
        raise ValueError(
            f"unsupported RI output support mode: {declared_support_mode}"
        )
    if od_scope == "affected":
        condition_qlw = physical_qlw
    elif support_mode == "common-support-v4":
        condition_qlw = bridging.total_qlw + common_unaffected_qlw
    else:
        condition_qlw = bridging.total_qlw + physical_unaffected_qlw
    return condition_qlw, physical_state, support_mode


def post_disruption_demand_by_od(
    od: pd.DataFrame,
    window: int,
    od_set: frozenset[str],
    disruption_start: int = DISRUPTION_START,
) -> pd.Series:
    """Reconstruct sum_{t>=t0} D from the exact hourly demand convention."""
    required = {"OriginID", "DestinationID", "Hour", "Trips_Hourly"}
    missing = required - set(od.columns)
    if missing:
        raise ValueError(f"OD demand input is missing columns {sorted(missing)}")
    selected = od[list(required)].copy()
    selected["Hour"] = pd.to_numeric(selected["Hour"], errors="coerce")
    selected["Trips_Hourly"] = pd.to_numeric(
        selected["Trips_Hourly"], errors="coerce"
    )
    selected = selected.dropna(subset=["Hour", "Trips_Hourly"])
    selected["Hour"] = selected["Hour"].astype(int)
    selected = selected[
        selected["Hour"].between(window, window + WINDOW_HOURS - 1)
    ]
    selected = (
        selected.groupby(
            ["OriginID", "DestinationID", "Hour"], as_index=False
        )["Trips_Hourly"].sum()
    )
    selected = selected[
        selected["Trips_Hourly"] > OD_HOURLY_MIN_TRIPS
    ].copy()
    selected["Omega"] = (
        selected["OriginID"].astype(str).str.strip()
        + "_"
        + selected["DestinationID"].astype(str).str.strip()
    )
    selected = selected[selected["Omega"].isin(od_set)]
    if selected.empty:
        return pd.Series(0.0, index=sorted(od_set), dtype=float)

    hour_start = (selected["Hour"] - window) * 60
    first_included = np.maximum(hour_start, disruption_start)
    last_included = np.minimum(hour_start + 59, int(TIME_INDEX.max()))
    selected["IncludedMinutes"] = np.maximum(
        0, last_included - first_included + 1
    )
    selected["PostDemand"] = (
        selected["Trips_Hourly"] * selected["IncludedMinutes"] / 60.0
    )
    return (
        selected.groupby("Omega")["PostDemand"].sum()
        .reindex(sorted(od_set), fill_value=0.0)
        .astype(float)
    )


def load_od_input(path: Path | None = None) -> pd.DataFrame:
    source = OD_FILE if path is None else Path(path)
    signature = source_signature(source)
    key = (
        str(signature["path"]),
        int(signature["size"]),
        int(signature["mtime_ns"]),
    )
    if key not in _OD_INPUT_MEMORY_CACHE:
        _OD_INPUT_MEMORY_CACHE.clear()
        _OD_INPUT_MEMORY_CACHE[key] = pd.read_csv(
            source,
            usecols=["OriginID", "DestinationID", "Hour", "Trips_Hourly"],
            low_memory=False,
        )
    return _OD_INPUT_MEMORY_CACHE[key]


def load_normal(
    window: int,
    od_input: pd.DataFrame | None = None,
) -> NormalWindow:
    path = NORMAL_DIR / f"Normal_{window:02d}_{window + WINDOW_HOURS:02d}.csv"
    required = {"Variable", "Omega", "Time", "Value"}
    frame = pd.read_csv(path, usecols=lambda column: column in required)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    od_set = clean_od_values(frame)
    frame["Omega"] = frame["Omega"].astype(str).str.strip()
    frame["Time"] = pd.to_numeric(frame["Time"], errors="coerce")
    frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    frame = frame.dropna(subset=["Time", "Value"])
    frame = frame[frame["Time"].between(TIME_INDEX.min(), TIME_INDEX.max())]
    frame["Time"] = frame["Time"].astype(int)
    state = frame[frame["Variable"].isin(TARGET_VARIABLES)]
    by_od_variable_time = state.groupby(
        ["Omega", "Variable", "Time"], sort=False
    )["Value"].sum()
    by_od_time = by_od_variable_time.groupby(
        ["Omega", "Time"], sort=False
    ).sum()
    total_state = by_od_variable_time.groupby(
        ["Variable", "Time"], sort=True
    ).sum()
    total = qlw_totals(total_state)

    initial_passengers = (
        state[state["Time"].eq(DISRUPTION_START - 1)]
        .groupby("Omega")["Value"].sum()
        .reindex(sorted(od_set), fill_value=0.0)
        .astype(float)
    )
    od = load_od_input() if od_input is None else od_input
    post_demand = post_disruption_demand_by_od(
        od, window, od_set, DISRUPTION_START
    )
    demand_by_od = initial_passengers.add(post_demand, fill_value=0.0)
    return NormalWindow(
        path,
        od_set,
        by_od_time,
        total,
        by_od_variable_time,
        total_state,
        demand_by_od,
        initial_passengers,
        post_demand,
        float(demand_by_od.sum()),
    )


def performance_ratio(
    normal: pd.Series,
    condition: pd.Series,
    time_index: pd.Index = TIME_INDEX,
) -> np.ndarray:
    normal_values = normal.reindex(time_index, fill_value=0.0).to_numpy(float)
    condition_values = condition.reindex(time_index, fill_value=0.0).to_numpy(float)
    ratio = np.ones_like(normal_values)
    positive = condition_values > 1e-12
    ratio[positive] = normal_values[positive] / condition_values[positive]
    ratio[(~positive) & (normal_values > 1e-12)] = 1.0
    return np.clip(ratio, 0.0, 1.0)


def integrate(
    values: np.ndarray | pd.Series,
    time_index: pd.Index = TIME_INDEX,
) -> float:
    if isinstance(values, pd.Series):
        values = values.reindex(time_index, fill_value=0.0)
    array = np.asarray(values, dtype=float)
    times = time_index.to_numpy(dtype=float)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(array, times))


def resilience_index(
    normal: pd.Series,
    condition: pd.Series,
    evaluation_start: int = 0,
) -> float:
    time_index = pd.Index(
        range(evaluation_start, int(TIME_INDEX.max()) + 1),
        name="Time",
    )
    ratios = performance_ratio(normal, condition, time_index)
    return float(ratios.sum()) / float(len(time_index))


def normal_affected_qlw(
    normal: NormalWindow,
    affected_ods: frozenset[str],
) -> pd.Series:
    affected_rows = normal.qlw_by_od_time[
        normal.qlw_by_od_time.index.get_level_values("Omega").isin(affected_ods)
    ]
    return (
        affected_rows.groupby(level="Time").sum()
        .reindex(TIME_INDEX, fill_value=0.0)
        .astype(float)
    )


def normal_unaffected_qlw(
    normal: NormalWindow, affected_ods: frozenset[str]
) -> pd.Series:
    affected_normal = normal_affected_qlw(normal, affected_ods)
    unaffected = normal.total_qlw - affected_normal
    unaffected[unaffected.abs() < 1e-9] = 0.0
    if (unaffected < -1e-7).any():
        raise AssertionError("unaffected Normal Q+L+W became negative")
    return unaffected.clip(lower=0.0)


def normal_affected_state(
    normal: NormalWindow, affected_ods: frozenset[str]
) -> pd.Series:
    affected = normal.state_by_od_variable_time[
        normal.state_by_od_variable_time.index.get_level_values("Omega").isin(
            affected_ods
        )
    ]
    if affected.empty:
        return pd.Series(dtype=float)
    return affected.groupby(["Variable", "Time"]).sum()


def normal_unaffected_state(
    normal: NormalWindow, affected_ods: frozenset[str]
) -> pd.Series:
    unaffected = normal.total_state_by_variable_time.subtract(
        normal_affected_state(normal, affected_ods), fill_value=0.0
    )
    unaffected[unaffected.abs() < 1e-9] = 0.0
    if (unaffected < -1e-7).any():
        raise AssertionError("unaffected Normal passenger state became negative")
    return unaffected.clip(lower=0.0)


def passenger_time_metrics(
    state_by_variable_time: pd.Series,
    passenger_demand: float,
    evaluation_index: pd.Index,
) -> dict[str, float]:
    if not np.isfinite(passenger_demand) or passenger_demand < 0.0:
        raise ValueError("passenger-time metrics require nonnegative modeled demand")

    def area(variable: str) -> float:
        if (
            state_by_variable_time.empty
            or variable not in state_by_variable_time.index.get_level_values(
                "Variable"
            )
        ):
            values = pd.Series(0.0, index=evaluation_index, dtype=float)
        else:
            values = state_by_variable_time.xs(
                variable, level="Variable"
            ).reindex(evaluation_index, fill_value=0.0)
        # Each state is a one-minute passenger inventory.  The discrete sum
        # therefore gives passenger-minutes exactly as defined in Eq. PT.
        return float(values.sum())

    queue_area = area("Queue")
    onboard_area = area("Load")
    walking_area = area("Walking")
    total_area = queue_area + onboard_area + walking_area
    if passenger_demand <= 0.0:
        if max(abs(queue_area), abs(onboard_area), abs(walking_area)) > 1e-9:
            raise ValueError("zero affected demand has nonzero passenger-time state")
        # An implicit zero-impact branch has no physical affected passenger.
        # Passenger-time averages are undefined, while its zero numerators and
        # scenario probability remain available for probability-weighted sums.
        average_waiting = average_onboard = average_walking = average_total = (
            float("nan")
        )
        time_defined = False
    else:
        average_waiting = queue_area / passenger_demand
        average_onboard = onboard_area / passenger_demand
        average_walking = walking_area / passenger_demand
        average_total = total_area / passenger_demand
        time_defined = True
    return {
        "Passenger_Demand": float(passenger_demand),
        "Passenger_Time_Defined": time_defined,
        "Queue_Passenger_Minutes": queue_area,
        "Onboard_Passenger_Minutes": onboard_area,
        "Walking_Passenger_Minutes": walking_area,
        "Total_Travel_Passenger_Minutes": total_area,
        "Average_Waiting_Time_min": average_waiting,
        "Average_InVehicle_Time_min": average_onboard,
        "Average_Walking_Time_min": average_walking,
        "Average_Total_Travel_Time_min": average_total,
    }


def percent_change(value: float, reference: float) -> float:
    if not np.isfinite(reference) or abs(reference) <= 1e-12:
        return float("nan")
    return 100.0 * (value - reference) / reference


def discover_scenarios(
    args: argparse.Namespace,
    durations: list[int] | None = None,
) -> list[Scenario]:
    stats = pd.read_csv(SCENARIO_STATS, low_memory=False)
    required = {
        "window", "target", "duration", "checkpoint", "logic_version"
    }
    missing = required - set(stats.columns)
    if missing:
        raise ValueError(f"{SCENARIO_STATS}: missing columns {sorted(missing)}")
    stats["start_hour"] = (
        stats["window"].astype(str).str.split("_").str[0].astype(int)
    )
    stats["station"] = stats["target"].astype(str).str.strip().str.upper()
    selected = stats[
        stats["start_hour"].isin(args.windows)
        & stats["duration"].isin(DURATION_WEIGHTS)
        & stats["checkpoint"].eq("EXTRACTED")
        & stats["logic_version"].eq(LOGIC_VERSION)
    ].copy()
    if args.stations:
        selected = selected[selected["station"].isin(args.stations)]
    keys = ["start_hour", "station", "duration"]
    if selected.duplicated(keys, keep=False).any():
        raise ValueError("duplicate current disruption scenarios in manifest")
    lookup = {
        (int(row.start_hour), str(row.station), int(row.duration))
        for row in selected.itertuples(index=False)
    }
    station_hours = sorted({
        (int(row.start_hour), str(row.station))
        for row in selected[selected["duration"].eq(max(DURATIONS))].itertuples(index=False)
    })
    scenarios = []
    requested_durations = args.durations if durations is None else durations
    for window, station in station_hours:
        for duration in sorted(requested_durations):
            key = (window, station, duration)
            path = (
                DISRUPTION_DIR / f"Disrupt_{window}_{station}_{duration}.csv"
                if key in lookup else None
            )
            if path is None and duration != min(DURATIONS):
                raise ValueError(f"missing required disruption branch: {key}")
            scenarios.append(Scenario(window, station, duration, path))
    if not scenarios:
        raise ValueError("no matching disruption scenarios")
    return scenarios


def discover_cases(requested: list[str] | None) -> dict[str, Path]:
    available = {str(cfg.get("evaluation.sensitivity.baseline_case")): BASELINE_DIR}
    stage2_root = CORRECTED_RESULTS / "stage2"
    if stage2_root.is_dir():
        for directory in sorted(stage2_root.iterdir()):
            if directory.is_dir() and (directory / "DecisionVars").is_dir():
                available[directory.name.upper()] = directory
    if requested:
        available.update({
            PROPOSED_CASE: BASELINE_DIR,
            BENCHMARK_CASE: FIXED_SPE_DIR,
        })
        names = [str(value).strip().upper() for value in requested]
        unknown = sorted(set(names) - set(available))
        if unknown:
            raise ValueError(f"unknown or not-yet-created case IDs: {unknown}")
        return {name: available[name] for name in names}
    return available


def common_row(
    scenario: Scenario,
    normal: NormalWindow,
    normal_reference: pd.Series,
    affected_ods: frozenset[str],
    common_support_ods: frozenset[str],
    source_file: Path | None,
    od_scope: str,
    evaluation_start: int,
    probabilities: tuple[float, float, float],
) -> dict[str, object]:
    evaluation_index = pd.Index(
        range(evaluation_start, int(TIME_INDEX.max()) + 1),
        name="Time",
    )
    station_probability, duration_probability, scenario_probability = probabilities
    return {
        "Scenario": scenario.scenario_id,
        "Window": scenario.window,
        "Station": scenario.station,
        "Duration": scenario.duration,
        "Station_Probability": station_probability,
        "Duration_Probability": duration_probability,
        "Scenario_Probability": scenario_probability,
        "Normal_OD_Count": len(normal.od_set),
        "Affected_OD_Count": len(affected_ods),
        "Physical_Affected_OD_Count": len(affected_ods),
        "Common_Support_OD_Count": len(common_support_ods),
        "Common_Only_OD_Count": len(common_support_ods - affected_ods),
        "Unaffected_OD_Count": len(normal.od_set) - len(affected_ods),
        "Outside_Common_Support_OD_Count": (
            len(normal.od_set) - len(common_support_ods)
        ),
        "Full_OD_Count": len(normal.od_set),
        "Evaluation_OD_Scope": od_scope,
        "Time_Start": int(evaluation_index.min()),
        "Time_End": int(evaluation_index.max()),
        "Time_Points": len(evaluation_index),
        "Normal_QLW_Area": integrate(normal_reference, evaluation_index),
        "Source_File": absolute_path_text(source_file) if source_file else None,
    }


def load_scenario_probabilities(
    scenarios: list[Scenario],
) -> dict[str, tuple[float, float, float]]:
    required = {"hour", "station", "StationProbability"}
    frame = pd.read_csv(
        STATION_PROBABILITY_FILE,
        usecols=lambda column: column in required,
    )
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{STATION_PROBABILITY_FILE}: missing columns {sorted(missing)}"
        )
    frame["hour"] = pd.to_numeric(frame["hour"], errors="raise").astype(int)
    frame["station"] = frame["station"].astype(str).str.strip().str.upper()
    frame["StationProbability"] = pd.to_numeric(
        frame["StationProbability"], errors="raise"
    )
    if frame.duplicated(["hour", "station"]).any():
        raise ValueError(f"{STATION_PROBABILITY_FILE}: duplicate hour/station rows")
    lookup = frame.set_index(["hour", "station"])["StationProbability"]
    result: dict[str, tuple[float, float, float]] = {}
    for scenario in scenarios:
        key = (scenario.window, scenario.station)
        if key not in lookup.index:
            raise ValueError(f"missing historical station probability for {key}")
        station_probability = float(lookup.loc[key])
        duration_probability = float(DURATION_WEIGHTS[scenario.duration])
        scenario_probability = station_probability * duration_probability
        if not 0.0 <= scenario_probability <= 1.0:
            raise ValueError(
                f"invalid scenario probability for {scenario.scenario_id}: "
                f"{scenario_probability}"
            )
        result[scenario.scenario_id] = (
            station_probability,
            duration_probability,
            scenario_probability,
        )
    return result


def load_case_scenario_probabilities(
    scenarios: list[Scenario],
    case_dir: Path,
    station_probabilities: dict[str, tuple[float, float, float]],
) -> tuple[
    dict[str, tuple[float, float, float]],
    dict[tuple[int, str], dict],
]:
    summaries: dict[tuple[int, str], dict] = {}
    result: dict[str, tuple[float, float, float]] = {}
    for scenario in scenarios:
        key = (scenario.window, scenario.station)
        if key not in summaries:
            path = (
                case_dir / "Performance" /
                f"FullMILP_{scenario.window:02d}_{scenario.station}_summary.json"
            )
            try:
                summaries[key] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"cannot read Stage-2 summary {path}") from error
        weights = summaries[key].get("duration_weights") or {}
        duration_probability = float(
            weights.get(str(scenario.duration), weights.get(scenario.duration))
        )
        station_probability = station_probabilities[scenario.scenario_id][0]
        scenario_probability = station_probability * duration_probability
        result[scenario.scenario_id] = (
            station_probability,
            duration_probability,
            scenario_probability,
        )
    return result, summaries


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def write_json_atomic(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    global CORRECTED_RESULTS, BASELINE_DIR, FIXED_SPE_DIR
    global NORMAL_DIR, DISRUPTION_DIR, SCENARIO_STATS, STATION_PROBABILITY_FILE
    global CONDITION_CACHE_DIR, CONDITION_CACHE_ENABLED
    args = parse_args()
    CORRECTED_RESULTS = args.results_root
    BASELINE_DIR = args.baseline_dir
    FIXED_SPE_DIR = args.fixed_spe_dir
    NORMAL_DIR = args.normal_dir
    STATION_PROBABILITY_FILE = args.station_probability_file.resolve()
    DISRUPTION_DIR = args.disruption_dir
    SCENARIO_STATS = DISRUPTION_DIR / "_scenario_stats.csv"
    CONDITION_CACHE_ENABLED = not args.no_condition_cache
    if args.condition_cache_dir is not None:
        CONDITION_CACHE_DIR = args.condition_cache_dir
    for key in CONDITION_CACHE_STATS:
        CONDITION_CACHE_STATS[key] = 0
    evaluation_index = pd.Index(
        range(args.evaluation_start, int(TIME_INDEX.max()) + 1),
        name="Time",
    )
    scenarios = discover_scenarios(args)
    scenario_probabilities = load_scenario_probabilities(scenarios)
    if args.stations is None and set(args.durations) == set(DURATION_WEIGHTS):
        for window in args.windows:
            probability_sum = sum(
                scenario_probabilities[scenario.scenario_id][2]
                for scenario in scenarios
                if scenario.window == window
            )
            if not np.isclose(probability_sum, 1.0, atol=1e-10):
                raise ValueError(
                    f"H{window:02d} historical scenario probability sums to "
                    f"{probability_sum}, expected 1"
                )
    case_dirs = discover_cases(args.cases)
    od_input = load_od_input()
    normal_cache = {
        window: load_normal(window, od_input)
        for window in sorted(args.windows)
    }

    # Always construct the station-level support from every retained duration,
    # even when the command evaluates only one duration.  An absent D30 file is
    # the model's implicit empty physical branch.
    support_scenarios = (
        scenarios
        if set(args.durations) == set(DURATION_WEIGHTS)
        else discover_scenarios(args, sorted(DURATION_WEIGHTS))
    )
    disruption_cache: dict[str, ConditionData | None] = {}
    common_support_by_station: dict[tuple[int, str], frozenset[str]] = {}
    for support_scenario in support_scenarios:
        condition = (
            None
            if support_scenario.disruption_file is None
            else read_condition(support_scenario.disruption_file)
        )
        disruption_cache[support_scenario.scenario_id] = condition
        key = (support_scenario.window, support_scenario.station)
        branch_support = frozenset() if condition is None else condition.od_set
        common_support_by_station[key] = (
            common_support_by_station.get(key, frozenset()) | branch_support
        )
    for (window, station), common_support in common_support_by_station.items():
        unknown = common_support - normal_cache[window].od_set
        if unknown:
            raise ValueError(
                f"H{window:02d}_{station}: common support has {len(unknown)} "
                "ODs outside the matching Normal OD universe"
            )

    normal_rows = []
    if args.od_scope == "full":
        for window, normal in normal_cache.items():
            row = {
                "Condition": "Normal",
                "Scenario": f"NORMAL_{window:02d}",
                "Window": window,
                "Normal_OD_Count": len(normal.od_set),
                "Full_OD_Count": len(normal.od_set),
                "Evaluation_OD_Scope": args.od_scope,
                "Time_Start": int(evaluation_index.min()),
                "Time_End": int(evaluation_index.max()),
                "Time_Points": len(evaluation_index),
                "QLW_Area": integrate(normal.total_qlw, evaluation_index),
                "RI": 1.0,
                "RI_Normal": 1.0,
                "Source_File": absolute_path_text(normal.source_file),
            }
            row.update(passenger_time_metrics(
                normal.total_state_by_variable_time,
                normal.total_demand,
                evaluation_index,
            ))
            row["Passenger_Time_OD_Scope"] = "full-normal-reference"
            normal_rows.append(row)

    disruption_rows = []
    merge_cache: dict[str, ScenarioMerge] = {}
    for scenario in scenarios:
        normal = normal_cache[scenario.window]
        if scenario.disruption_file is None:
            affected_ods = frozenset()
            affected_normal_qlw = pd.Series(
                0.0, index=TIME_INDEX, dtype=float
            )
            unaffected_qlw = normal.total_qlw.copy()
            affected_normal_state = (
                normal.total_state_by_variable_time.iloc[0:0].copy()
            )
            unaffected_state = normal.total_state_by_variable_time.copy()
            disruption_qlw = pd.Series(
                0.0, index=TIME_INDEX, dtype=float
            )
            disruption_state = (
                normal.total_state_by_variable_time.iloc[0:0].copy()
            )
        else:
            disruption = disruption_cache[scenario.scenario_id]
            if disruption is None:
                raise AssertionError("explicit disruption branch was not loaded")
            affected_ods = disruption.od_set
            unknown = affected_ods - normal.od_set
            if unknown:
                raise ValueError(
                    f"{scenario.scenario_id}: disruption has {len(unknown)} ODs "
                    "outside the matching Normal OD universe"
                )
            affected_normal_qlw = normal_affected_qlw(normal, affected_ods)
            unaffected_qlw = normal_unaffected_qlw(normal, affected_ods)
            affected_normal_state = normal_affected_state(normal, affected_ods)
            unaffected_state = normal_unaffected_state(normal, affected_ods)
            disruption_qlw = disruption.total_qlw
            disruption_state = disruption.state_by_variable_time
        common_support_ods = common_support_by_station[
            (scenario.window, scenario.station)
        ]
        common_unaffected_qlw = normal_unaffected_qlw(
            normal, common_support_ods
        )
        affected_passenger_demand = float(normal.demand_by_od.reindex(
            sorted(affected_ods), fill_value=0.0
        ).sum())
        if args.od_scope == "affected":
            normal_reference = affected_normal_qlw
            normal_reference_state = affected_normal_state
            condition_qlw = disruption_qlw
        else:
            normal_reference = normal.total_qlw
            normal_reference_state = normal.total_state_by_variable_time
            condition_qlw = disruption_qlw + unaffected_qlw
        if args.od_scope == "affected":
            normal_row = {
                "Condition": "Normal",
                "Scenario": scenario.scenario_id,
                "Window": scenario.window,
                "Station": scenario.station,
                "Duration": scenario.duration,
                "Normal_OD_Count": len(normal.od_set),
                "Affected_OD_Count": len(affected_ods),
                "Full_OD_Count": len(normal.od_set),
                "Evaluation_OD_Scope": args.od_scope,
                "Time_Start": int(evaluation_index.min()),
                "Time_End": int(evaluation_index.max()),
                "Time_Points": len(evaluation_index),
                "QLW_Area": integrate(normal_reference, evaluation_index),
                "RI": 1.0,
                "RI_Normal": 1.0,
                "Source_File": absolute_path_text(normal.source_file),
            }
            normal_row.update(passenger_time_metrics(
                normal_reference_state, affected_passenger_demand, evaluation_index
            ))
            normal_row["Passenger_Time_OD_Scope"] = "affected"
            normal_rows.append(normal_row)
        ri = resilience_index(
            normal_reference,
            condition_qlw,
            args.evaluation_start,
        )
        row = common_row(
            scenario,
            normal,
            normal_reference,
            affected_ods,
            common_support_ods,
            scenario.disruption_file,
            args.od_scope,
            args.evaluation_start,
            scenario_probabilities[scenario.scenario_id],
        )
        row.update({
            "Condition": "Disruption",
            "Passenger_Time_OD_Scope": "affected",
            "QLW_Area": integrate(condition_qlw, evaluation_index),
            "RI": ri,
            "RI_Normal": 1.0,
            "RI_Disruption": ri,
        })
        disruption_time_metrics = passenger_time_metrics(
            disruption_state, affected_passenger_demand, evaluation_index
        )
        row.update(disruption_time_metrics)
        disruption_rows.append(row)
        merge_cache[scenario.scenario_id] = ScenarioMerge(
            scenario,
            normal,
            affected_ods,
            common_support_ods,
            normal_reference,
            unaffected_qlw,
            common_unaffected_qlw,
            unaffected_state,
            affected_passenger_demand,
            ri,
            disruption_time_metrics,
        )

    bridging_rows = []
    missing_files = []
    for case_id, case_dir in case_dirs.items():
        case_probabilities, case_summaries = load_case_scenario_probabilities(
            scenarios, case_dir, scenario_probabilities
        )
        strict_support_by_station: dict[
            tuple[int, str], frozenset[str]
        ] = {}
        for scenario in scenarios:
            decision_file = (
                case_dir
                / "DecisionVars"
                / f"Disrupt_{scenario.window:02d}_{scenario.station}_{scenario.duration}.csv"
            )
            try:
                bridging = read_condition(decision_file)
            except FileNotFoundError:
                missing_files.append((case_id, scenario.scenario_id, decision_file))
                if args.allow_incomplete:
                    continue
                raise FileNotFoundError(decision_file)
            merge = merge_cache[scenario.scenario_id]
            output_ods = bridging.od_set
            unknown_output = output_ods - merge.normal.od_set
            if unknown_output:
                raise ValueError(
                    f"{case_id}/{scenario.scenario_id}: Bridging output contains "
                    f"{len(unknown_output)} ODs outside the matching Normal "
                    "OD universe"
                )
            outside_physical_common = output_ods - merge.common_support_ods
            common_only_output = output_ods - merge.affected_ods
            summary_stats = (
                case_summaries[(scenario.window, scenario.station)].get("stats")
                or {}
            )
            formulation_mode = summary_stats.get("FormulationMode")
            sharing_marker = summary_stats.get("PassengerFlowVariableSharing")
            if formulation_mode == "BUS_ONLY_WAIT_AND_SEE":
                declared_support_mode = "physical-support-bus-only"
            elif sharing_marker is None:
                declared_support_mode = (
                    "common-support-v4"
                    if common_only_output
                    else "physical-support-legacy"
                )
            else:
                declared_support_mode = (
                    "common-support-v4"
                    if bool(sharing_marker)
                    else "physical-support-legacy"
                )
            missing_common_support = merge.common_support_ods - output_ods
            if (
                declared_support_mode == "common-support-v4"
                and missing_common_support
            ):
                raise ValueError(
                    f"{case_id}/{scenario.scenario_id}: v4 output omits "
                    f"{len(missing_common_support)} ODs from the declared common "
                    "support"
                )
            if declared_support_mode == "common-support-v4":
                support_key = (scenario.window, scenario.station)
                prior_support = strict_support_by_station.setdefault(
                    support_key, output_ods
                )
                if prior_support != output_ods:
                    raise ValueError(
                        f"{case_id}/H{scenario.window:02d}_{scenario.station}: "
                        "strict D30/D60 output supports differ"
                    )
                declared_count = summary_stats.get("CommonAffectedODCount")
                if (
                    declared_count is not None
                    and int(declared_count) != len(output_ods)
                ):
                    raise ValueError(
                        f"{case_id}/{scenario.scenario_id}: strict output has "
                        f"{len(output_ods)} ODs but summary declares "
                        f"{declared_count}"
                    )
            if (
                declared_support_mode in {
                "physical-support-legacy", "physical-support-bus-only"}
                and common_only_output
            ):
                raise ValueError(
                    f"{case_id}/{scenario.scenario_id}: physical-support "
                    "summary conflicts "
                    "with common-only ODs in the output"
                )
            ri_model_support = (
                output_ods
                if declared_support_mode == "common-support-v4"
                else merge.affected_ods
            )
            ri_model_unaffected_qlw = normal_unaffected_qlw(
                merge.normal, ri_model_support
            )
            condition_qlw, bridging_physical_state, support_mode = (
                bridging_condition_for_ri(
                    bridging,
                    merge.affected_ods,
                    merge.unaffected_qlw,
                    ri_model_unaffected_qlw,
                    args.od_scope,
                    declared_support_mode,
                )
            )
            ri = resilience_index(
                merge.reference_qlw,
                condition_qlw,
                args.evaluation_start,
            )
            delta = ri - merge.disrupted_ri
            relative_pct = (
                100.0 * delta / merge.disrupted_ri
                if merge.disrupted_ri > 1e-12
                else np.nan
            )
            row = common_row(
                scenario,
                merge.normal,
                merge.reference_qlw,
                merge.affected_ods,
                merge.common_support_ods,
                decision_file,
                args.od_scope,
                args.evaluation_start,
                case_probabilities[scenario.scenario_id],
            )
            row.update({
                "Condition": "Bridging",
                "Passenger_Time_OD_Scope": "physical-affected",
                "Case_ID": case_id,
                "RI_Output_Support_Mode": support_mode,
                "RI_Model_Support_OD_Count": len(ri_model_support),
                "Output_OD_Count": len(output_ods),
                "Output_Affected_OD_Count": len(
                    output_ods & merge.affected_ods
                ),
                "Output_Physical_Affected_OD_Count": len(
                    output_ods & merge.affected_ods
                ),
                "Output_Common_Only_OD_Count": len(common_only_output),
                "Output_Outside_Physical_Common_Support_OD_Count": len(
                    outside_physical_common
                ),
                "Missing_Affected_OD_Count": len(merge.affected_ods - output_ods),
                "Missing_Physical_Affected_OD_Count": len(
                    merge.affected_ods - output_ods
                ),
                "Missing_Common_Support_OD_Count": len(
                    missing_common_support
                ),
                "QLW_Area": integrate(condition_qlw, evaluation_index),
                "RI": ri,
                "RI_Normal": 1.0,
                "RI_Disruption": merge.disrupted_ri,
                "RI_Bridging": ri,
                "RI_Improvement": delta,
                "RI_Improvement_pct": relative_pct,
            })
            bridging_time_metrics = passenger_time_metrics(
                bridging_physical_state,
                merge.affected_passenger_demand,
                evaluation_index,
            )
            row.update(bridging_time_metrics)
            for metric in (
                "Queue_Passenger_Minutes",
                "Onboard_Passenger_Minutes",
                "Walking_Passenger_Minutes",
                "Total_Travel_Passenger_Minutes",
            ):
                row[f"Disruption_{metric}"] = (
                    merge.disrupted_time_metrics[metric]
                )
            row["PT_Reduction_Passenger_Minutes"] = (
                merge.disrupted_time_metrics[
                    "Total_Travel_Passenger_Minutes"
                ]
                - bridging_time_metrics["Total_Travel_Passenger_Minutes"]
            )
            for metric in (
                "Average_Waiting_Time_min",
                "Average_InVehicle_Time_min",
                "Average_Walking_Time_min",
                "Average_Total_Travel_Time_min",
            ):
                disruption_value = merge.disrupted_time_metrics[metric]
                metric_stem = metric.removeprefix("Average_").removesuffix(
                    "_min"
                )
                row[f"Disruption_{metric}"] = disruption_value
                row[f"{metric_stem}_Change_vs_Disruption_min"] = (
                    bridging_time_metrics[metric] - disruption_value
                )
                row[f"{metric_stem}_Change_vs_Disruption_pct"] = percent_change(
                    bridging_time_metrics[metric], disruption_value
                )
            bridging_rows.append(row)

    if missing_files and not args.allow_incomplete:
        raise AssertionError("unreachable missing-file validation")

    normal_frame = pd.DataFrame(normal_rows).sort_values(["Window"])
    disruption_frame = pd.DataFrame(disruption_rows).sort_values(
        ["Window", "Station", "Duration"]
    )
    bridging_frame = pd.DataFrame(bridging_rows).sort_values(
        ["Case_ID", "Window", "Station", "Duration"]
    )
    output_dir = Path(absolute_path_text(args.output_dir))
    write_csv_atomic(normal_frame, output_dir / "Normal_RI.csv")
    write_csv_atomic(disruption_frame, output_dir / "Disruption_RI.csv")
    write_csv_atomic(bridging_frame, output_dir / "Bridging_RI.csv")
    write_json_atomic(
        {
            "schema": "ri-io-audit-v1",
            "condition_cache_schema": CONDITION_CACHE_SCHEMA,
            "condition_cache_enabled": CONDITION_CACHE_ENABLED,
            "condition_cache_dir": absolute_path_text(CONDITION_CACHE_DIR),
            "condition_cache_stats": dict(CONDITION_CACHE_STATS),
            "od_input_memory_cache_entries": len(_OD_INPUT_MEMORY_CACHE),
        },
        output_dir / "RI_IO_Audit.json",
    )
    print(
        f"RI_COMPLETE normal={len(normal_frame)} "
        f"disruption={len(disruption_frame)} bridging={len(bridging_frame)} "
        f"skipped={len(missing_files)} "
        f"cache_hits={CONDITION_CACHE_STATS['hits']} "
        f"cache_misses={CONDITION_CACHE_STATS['misses']} "
        f"output={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
