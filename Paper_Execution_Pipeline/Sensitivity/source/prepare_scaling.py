"""Prepare corridor-constrained K-means/internal representatives for K scaling.

For every run of ``evaluation.scaling.runs`` (window + K list; Sec. 6.8,
Table 8) the production clustering features are rebuilt for that window, the
frozen K = clustering.k_per_window reduction is reproduced as a gate, and the
representatives for each larger K are written to
``<results>/<paths.scaling>/inputs/h<WW>/``.  Without arguments every
configured window is prepared; ``--window`` selects one.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


SCRIPT = Path(__file__).resolve()
SOURCE = SCRIPT.parent
SENSITIVITY = SOURCE.parent
PIPELINE = SENSITIVITY.parent
sys.path.insert(0, str(PIPELINE.parent))
sys.path.insert(0, str(PIPELINE))
from bbconfig import cfg  # noqa: E402
import network_data  # noqa: E402

CLUSTERING = PIPELINE / "02_reduction_method_comparison.py"
RUNS = {
    int(run["window"]): tuple(int(k) for k in run["k_values"])
    for run in cfg.get("evaluation.scaling.runs")
}
BASELINE_K = int(cfg.get("clustering.k_per_window"))       # Sec. 5.1: K = 14
METHOD = str(cfg.get("clustering.method"))                  # Sec. 5.1: K-means
SEED = int(cfg.get("clustering.reduction.seed"))
N_INIT = int(cfg.get("clustering.reduction.n_init"))
LOCATION_SUPPORT = str(cfg.get("clustering.reduction.location_support"))
READ_WORKERS = int(cfg.get("clustering.reduction.read_workers"))
SMOOTHING_BETA = float(cfg.get("probability.station_smoothing_beta"))  # Table 3: beta
DURATION_SOURCE = str(cfg.get("probability.duration_source"))
DURATIONS = tuple(int(value) for value in cfg.get("time.durations_min"))
DURATION_TAG = "_".join(str(value) for value in DURATIONS)
_COUNTS = {
    int(state): int(count)
    for state, count in cfg.get("probability.duration_counts").items()
}
# Table 3: pi_30 = 208/303, pi_60 = 95/303.
DURATION_WEIGHT = {
    duration: _COUNTS[duration] / sum(_COUNTS.values()) for duration in DURATIONS
}
# Corridor order of the dynamic program and of the cluster numbering: central
# group first, then the corridors in input-file order (network_data).
CORRIDORS = (network_data.central_group_id(), *network_data.corridor_order())
COST_TOLERANCE = 1e-15
TIE_BREAK = 1e-12
PROBABILITY_TOLERANCE = 1e-12
# Set per window in prepare_window().
WINDOW = 0
K_VALUES: tuple[int, ...] = ()
OUTPUT = Path()
BASELINE_FILES: dict[str, Path] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window", type=int, choices=sorted(RUNS),
        help="one window of evaluation.scaling.runs (default: every window)",
    )
    parser.add_argument(
        "--k-values", nargs="+", type=int,
        help="override the K list of --window (default: evaluation.scaling.runs)",
    )
    parser.add_argument(
        "--disruption-dir", type=Path,
        default=cfg.results_dir / cfg.get("paths.disruptions"),
        help="disruption scenarios (_scenario_stats.csv and Disrupt_*.csv)",
    )
    parser.add_argument(
        "--clustering-dir", type=Path,
        help=("folder with the K = clustering.k_per_window reduction used by "
              "the reproduction gate; default: the pinned files when "
              "pinned.use_pinned, else <results>/<paths.clustering>"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=cfg.results_dir / cfg.get("paths.scaling") / "inputs",
        help="representatives are written to <output-root>/h<WW>/",
    )
    args = parser.parse_args()
    if args.k_values is not None:
        if args.window is None:
            parser.error("--k-values requires --window")
        if any(value <= 0 for value in args.k_values):
            parser.error("--k-values must contain positive integers")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def baseline_files(clustering_dir: Path | None) -> dict[str, Path]:
    """Frozen K = clustering.k_per_window reduction (stations/branches/assignments)."""
    names = ("representatives", "branches", "assignments")
    if clustering_dir is None and bool(cfg.get("pinned.use_pinned")):
        files = {}
        for name in names:
            path = cfg.package_path(f"pinned.clustering.{name}")
            expected = str(cfg.get(f"pinned.clustering.{name}_sha256")).upper()
            if sha256(path) != expected:
                raise ValueError(f"pinned file checksum differs: {path}")
            files[name] = path
        return files
    folder = (
        clustering_dir if clustering_dir is not None
        else cfg.results_dir / cfg.get("paths.clustering")
    )
    return {
        name: folder / str(cfg.get(f"clustering.files.{name}")).format(
            k=BASELINE_K, durations=DURATION_TAG)
        for name in names
    }


def load_module():
    spec = importlib.util.spec_from_file_location("legacy_h16_clustering", CLUSTERING)
    if spec is None or spec.loader is None:
        raise ImportError(CLUSTERING)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.HOURS = (WINDOW,)
    module.SELECTED_K = BASELINE_K
    module.DURATION_WEIGHT = dict(DURATION_WEIGHT)
    return module


def station_corridor_map() -> dict[str, str]:
    """Station -> corridor id, or the central group id for shared stations."""
    return {
        str(station).strip().upper(): network_data.station_group(station)
        for station in network_data.station_corridors()
    }


def fit_options(module, data: dict, maximum_k: int) -> dict:
    options = {}
    corridor_array = np.asarray(data["corridor"], dtype=object)
    for corridor in CORRIDORS:
        indices = np.flatnonzero(corridor_array == corridor)
        if not len(indices):
            continue
        complete_count = int(data["complete"][indices].sum())
        options[corridor] = {}
        x = data["x"][indices]
        for local_k in range(1, min(maximum_k, complete_count) + 1):
            fitted = KMeans(
                n_clusters=local_k,
                random_state=SEED,
                n_init=N_INIT,
                algorithm="lloyd",
            ).fit(x)
            labels = fitted.labels_.astype(int)
            centers = fitted.cluster_centers_
            options[corridor][local_k] = {
                "indices": indices,
                "labels": labels,
                "centers": centers,
                "sse": float(np.square(x - centers[labels]).sum()),
            }
    return options


def allocate(options: dict, total_k: int) -> dict[str, int]:
    states: dict[int, tuple[float, dict[str, int]]] = {0: (0.0, {})}
    for corridor in CORRIDORS:
        if corridor not in options:
            continue
        next_states = {}
        for used, (cost, allocation) in states.items():
            for local_k, fitted in options[corridor].items():
                new_total = used + local_k
                if new_total > total_k:
                    continue
                new_cost = cost + fitted["sse"]
                previous = next_states.get(new_total)
                if previous is None or new_cost < previous[0] - COST_TOLERANCE:
                    next_states[new_total] = (
                        new_cost, {**allocation, corridor: local_k}
                    )
        states = next_states
    if total_k not in states:
        raise RuntimeError(f"K={total_k} cannot be allocated")
    return states[total_k][1]


def select_representatives(data: dict, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    corridor_array = np.asarray(data["corridor"], dtype=object)
    representatives = np.full(len(centers), -1, dtype=int)
    for corridor in CORRIDORS:
        if not (corridor_array == corridor).any():
            continue
        clusters = np.unique(labels[corridor_array == corridor])
        candidates = np.flatnonzero(
            (corridor_array == corridor) & np.asarray(data["complete"], dtype=bool)
        )
        distance = np.linalg.norm(
            centers[clusters][:, None, :] - data["x"][candidates][None, :, :],
            axis=2,
        )
        penalty = (float(distance.max()) + 1.0) * 1000.0
        cost = distance + (labels[candidates][None, :] != clusters[:, None]) * penalty
        # Reproduce the frozen station-code tie rule, lexicographic order of station codes.
        cost = cost + np.arange(len(candidates), dtype=float)[None, :] * TIE_BREAK
        rows, columns = linear_sum_assignment(cost)
        representatives[clusters[rows]] = candidates[columns]
    if (representatives < 0).any():
        raise AssertionError("representative assignment is incomplete")
    return representatives


def fit_k(data: dict, options: dict, total_k: int) -> dict:
    allocation = allocate(options, total_k)
    labels = np.full(len(data["stations"]), -1, dtype=int)
    centers = []
    offset = 0
    for corridor in CORRIDORS:
        if corridor not in allocation:
            continue
        fitted = options[corridor][allocation[corridor]]
        indices = fitted["indices"]
        labels[indices] = fitted["labels"] + offset
        centers.extend(fitted["centers"])
        offset += allocation[corridor]
    centers = np.asarray(centers, dtype=float)
    if offset != total_k or (labels < 0).any():
        raise AssertionError("combined partition is incomplete")
    representatives = select_representatives(data, labels, centers)
    probability = np.bincount(
        labels, weights=data["station_probability"], minlength=total_k
    )
    return {
        "allocation": allocation,
        "labels": labels,
        "centers": centers,
        "representatives": representatives,
        "probability": probability,
    }


def frames(data: dict, result: dict, k: int):
    labels = result["labels"]
    reps = result["representatives"]
    names = [data["stations"][index] for index in reps]
    representative_rows = []
    branch_rows = []
    assignment_rows = []
    for cluster, index in enumerate(reps):
        representative_rows.append({
            "hour": WINDOW,
            "k": k,
            "cluster": cluster,
            "representative_station": names[cluster],
            "selection_rule": (
                "complete 30/60 station nearest corridor-constrained "
                "K-means cluster unweighted centroid"
            ),
            "centroid_fallback_used": bool(labels[index] != cluster),
            "all_duration_source_available": True,
            "station_marginal_probability": float(data["station_probability"][index]),
            "distance_to_centroid": float(
                np.linalg.norm(data["x"][index] - result["centers"][cluster])
            ),
            "method": METHOD,
            "corridor": str(data["corridor"][index]),
        })
    for duration in DURATIONS:
        for cluster, name in enumerate(names):
            members = np.flatnonzero(labels == cluster)
            mass = float(data["station_probability"][members].sum())
            branch_rows.append({
                "hour": WINDOW,
                "k": k,
                "cluster": cluster,
                "representative_station": name,
                "duration": duration,
                "cluster_size": len(members),
                "conditional_cluster_probability": mass,
                "duration_weight": DURATION_WEIGHT[duration],
                "scenario_probability": DURATION_WEIGHT[duration] * mass,
                "source_scenario_available": True,
                "method": METHOD,
                "corridor": str(data["corridor"][reps[cluster]]),
            })
        for station_index, station in enumerate(data["stations"]):
            cluster = int(labels[station_index])
            assignment_rows.append({
                "hour": WINDOW,
                "duration": duration,
                "station": station,
                "station_marginal_probability": float(
                    data["station_probability"][station_index]
                ),
                "conditional_station_probability": float(
                    data["station_probability"][station_index]
                ),
                "cluster": cluster,
                "representative_station": names[cluster],
                "method": METHOD,
                "corridor": str(data["corridor"][station_index]),
                "representative_corridor": str(
                    data["corridor"][reps[cluster]]
                ),
            })
    return tuple(map(pd.DataFrame, (representative_rows, branch_rows, assignment_rows)))


def baseline_gate(data: dict, options: dict) -> dict:
    result = fit_k(data, options, BASELINE_K)
    representatives, branches, assignments = frames(data, result, BASELINE_K)
    stored_representatives = pd.read_csv(BASELINE_FILES["representatives"])
    stored_branches = pd.read_csv(BASELINE_FILES["branches"])
    stored_assignments = pd.read_csv(BASELINE_FILES["assignments"])
    stored_representatives = stored_representatives[
        stored_representatives["hour"].eq(WINDOW)
    ].sort_values("cluster")
    stored_branches = stored_branches[
        stored_branches["hour"].eq(WINDOW)
    ].sort_values(["duration", "cluster"])
    stored_assignments = stored_assignments[
        stored_assignments["hour"].eq(WINDOW)
    ].sort_values(["duration", "station"])
    representatives = representatives.sort_values("cluster")
    branches = branches.sort_values(["duration", "cluster"])
    assignments = assignments.sort_values(["duration", "station"])
    stored_cluster = stored_assignments.drop_duplicates("station").set_index(
        "station"
    )["cluster"].astype(int)
    stored_internal = all(
        int(stored_cluster.loc[row.representative_station]) == int(row.cluster)
        for row in stored_representatives.itertuples(index=False)
    )
    checks = {
        "stored_internal_representatives": stored_internal,
        "stored_unique_representatives": (
            stored_representatives["representative_station"].nunique() == BASELINE_K
        ),
        "labels": assignments["cluster"].tolist()
        == stored_assignments["cluster"].astype(int).tolist(),
        "branch_probability": bool(np.allclose(
            branches["scenario_probability"],
            stored_branches["scenario_probability"],
            atol=PROBABILITY_TOLERANCE,
        )),
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen K{BASELINE_K} reproduction failed: {checks}")
    return checks


def prepare_window(args: argparse.Namespace, window: int, k_values: tuple[int, ...]) -> None:
    global WINDOW, K_VALUES, OUTPUT
    WINDOW = int(window)
    K_VALUES = tuple(dict.fromkeys(int(value) for value in k_values))
    OUTPUT = args.output_root.resolve() / f"h{WINDOW:02d}"
    module = load_module()
    support, valid_support, support_audit = module.discover_scenarios(
        args.disruption_dir, READ_WORKERS, LOCATION_SUPPORT
    )
    _, raw = module.load_impact_features(support, read_workers=READ_WORKERS)
    station_corridor = station_corridor_map()
    history, _ = module.load_history(
        cfg.path("inputs.incidents_gt10"),
        cfg.path("inputs.incidents_raw"),
        cfg.path("inputs.incident_cohort"),
    )
    corridor_probability, _, history_rows, station_counts, _ = (
        module.historical_corridor_probability(history, station_corridor)
    )
    data = module.build_hour(support, valid_support, raw, WINDOW, LOCATION_SUPPORT)
    data["complete"] = data["all_duration_positive_impact"]
    module.apply_corridor_probability(
        data, station_corridor, corridor_probability, station_counts,
        SMOOTHING_BETA,
    )
    maximum_exact_k = int(data["complete"].sum())
    options = fit_options(module, data, max(K_VALUES))
    data["complete"] = data["all_duration_valid"]
    checks = baseline_gate(data, options)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for k in K_VALUES:
        result = fit_k(data, options, k)
        representatives, branches, assignments = frames(data, result, k)
        if len(representatives) != k or len(branches) != len(DURATIONS) * k:
            raise AssertionError(f"K={k} output size mismatch")
        if representatives["centroid_fallback_used"].any():
            raise AssertionError(f"K={k} has an out-of-cluster representative")
        if not np.allclose(
            branches.groupby("duration")["scenario_probability"].sum().sort_index(),
            [DURATION_WEIGHT[duration] for duration in sorted(DURATIONS)],
            atol=PROBABILITY_TOLERANCE,
        ):
            raise AssertionError(f"K={k} duration mass mismatch")
        suffix = f"k{k}_{DURATION_TAG}_H{WINDOW:02d}.csv"
        station_path = OUTPUT / f"selected_representative_stations_{suffix}"
        branch_path = OUTPUT / f"selected_representative_branches_{suffix}"
        assignment_path = OUTPUT / f"selected_assignments_{suffix}"
        representatives.to_csv(station_path, index=False, encoding="utf-8-sig")
        branches.to_csv(branch_path, index=False, encoding="utf-8-sig")
        assignments.to_csv(assignment_path, index=False, encoding="utf-8-sig")
        outputs[str(k)] = {
            "allocation": result["allocation"],
            "representatives": representatives["representative_station"].tolist(),
            "fallbacks": int(representatives["centroid_fallback_used"].sum()),
            "branch_file": str(branch_path.resolve()),
        }
    audit = {
        "schema": (
            f"cohort-kmeans-internal-h{WINDOW:02d}-scaling-input-v1"
        ),
        "window": WINDOW,
        "k_values": list(K_VALUES),
        "method": "corridor-constrained unweighted K-means",
        "duration_weights": {str(k): v for k, v in DURATION_WEIGHT.items()},
        "observation_count": len(data["stations"]),
        "complete_candidate_count": maximum_exact_k,
        "internal_representative_candidate_count": int(data["complete"].sum()),
        "maximum_exact_k": maximum_exact_k,
        "historical_rows": int(history_rows),
        "baseline_reproduction": checks,
        "baseline_k": BASELINE_K,
        "support_audit": support_audit,
        "outputs": outputs,
    }
    (OUTPUT / (
        f"{DURATION_SOURCE}_kmeans_internal_h{WINDOW:02d}_scaling_audit.json"
    )).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def main() -> int:
    global BASELINE_FILES
    args = parse_args()
    args.disruption_dir = args.disruption_dir.resolve()
    BASELINE_FILES = baseline_files(
        args.clustering_dir.resolve() if args.clustering_dir is not None else None
    )
    if args.window is not None:
        selected = {args.window: tuple(args.k_values or RUNS[args.window])}
    else:
        selected = dict(RUNS)
    for window, k_values in selected.items():
        prepare_window(args, window, k_values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
