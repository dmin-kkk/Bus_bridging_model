"""Legacy names of the executable configuration, read from the active profile.

Every value comes from ``bbconfig.cfg`` (config/profiles/<profile>/*.toml); the names are
kept because many pipeline modules import them.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bbconfig import cfg  # noqa: E402


def _int_keys(mapping) -> dict[int, int]:
    return {int(key): int(value) for key, value in mapping.items()}


WINDOWS = tuple(int(window) for window in cfg.get("time.windows"))
DURATIONS = tuple(int(duration) for duration in cfg.get("time.durations_min"))
# Executable operational cohort used by the Long30 model and downstream OOF
# evaluation (Table 3: pi_30, pi_60).
COHORT_DURATION_COUNTS = _int_keys(cfg.get("probability.duration_counts"))
COHORT_DURATION_WEIGHTS = {
    duration: count / sum(COHORT_DURATION_COUNTS.values())
    for duration, count in COHORT_DURATION_COUNTS.items()
}

REPRESENTATIVES_PER_DURATION = int(cfg.get("clustering.k_per_window"))
CLUSTERING_METHOD = cfg.get("clustering.method")
STATION_SMOOTHING_ALPHA = float(cfg.get("probability.station_smoothing_beta"))
FLEET_SIZE = int(cfg.get("operations.fleet_size"))
DEPOT_ACCESS_BOUND_MIN = int(cfg.get("operations.depot_access_bound_min"))
DISPATCH_BUFFER_MINUTES = int(cfg.get("time.clearance_buffer_min"))
MIN_TRIP_BOARDING = float(cfg.get("operations.min_trip_boarding"))
OD_HOURLY_MIN_TRIPS = float(cfg.get("operations.od_hourly_min_trips"))
WALKING_MINUTES = int(cfg.get("operations.walking_transfer_min"))
WALKING_TIME_FILENAME = cfg.get("inputs.walking_times")
PREDICTION_CUTOFFS = tuple(int(value) for value in cfg.get("probability.cutoff_alternatives_min"))
PREDICTION_LONG_COUNTS = _int_keys(cfg.get("probability.long_counts_by_cutoff"))

SCENARIO_LOGIC_VERSION = cfg.get("scenarios.logic_version")
ROUTE_FILENAME = cfg.get("routes.library_file")
REPRESENTATIVE_BRANCH_FILENAME = (
    f"selected_representative_branches_k{REPRESENTATIVES_PER_DURATION}_"
    + "_".join(str(duration) for duration in DURATIONS) + ".csv"
)
