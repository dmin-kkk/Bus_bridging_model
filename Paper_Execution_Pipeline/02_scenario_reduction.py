"""Scenario reduction used by Stage 1 (Sec. 4.3, Sec. 5.1).

Profiles
--------
* ``pinned.use_pinned = true`` (paper): the frozen reduction files listed in
  ``[pinned.clustering]`` are verified by SHA-256 (any mismatch stops the run)
  and copied to ``cfg.results_dir / paths.clustering`` under the names in
  ``[clustering.files]``.  ``--recompute`` rebuilds the reduction from the
  step-01 scenarios and diffs it against the pinned files; the recomputed files
  go to a ``recompute_check`` sub-folder and never replace the pinned inputs.
* ``pinned.use_pinned = false`` (synthetic): the reduction is always recomputed
  and written to ``cfg.results_dir / paths.clustering``.

Reduction rule (see docs/MODEL_CODE_MAP.md): unweighted
corridor-constrained K-means per corridor, local K capped by the number of
stations with a positive impact in every duration, corridor budgets allocated by minimum total SSE, and one
internal representative per cluster chosen by an assignment problem with the
station-code tie rule.  Only this rule reproduces the paper memberships; the
four-method comparison of ``02_reduction_method_comparison.py`` breaks three exact
distance ties differently (H13, H15, H16).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
from bbconfig import cfg  # noqa: E402

FLOAT_RTOL = 1e-9    # comparison tolerance for recomputed vs pinned floats
FLOAT_ATOL = 1e-12
TIE_EPSILON = 1e-12  # station-order tie breaker of the representative rule
PINNED_KEYS = ("representatives", "branches", "assignments", "station_probability")
DESCRIPTIVE_COLUMNS = {"selection_rule"}   # free-text descriptions, reported but not compared
COMPARE_KEYS = {
    "representatives": ["hour", "cluster"],
    "branches": ["hour", "cluster", "duration"],
    "assignments": ["hour", "duration", "station"],
    "station_probability": ["hour", "station"],
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def durations() -> list[int]:
    return [int(value) for value in cfg.get("time.durations_min")]


def output_name(key: str) -> str:
    return str(cfg.get(f"clustering.files.{key}")).format(
        k=int(cfg.get("clustering.k_per_window")),
        durations="_".join(str(value) for value in durations()),
    )


def duration_weights() -> dict[int, float]:
    counts = {int(key): int(value) for key, value in cfg.get("probability.duration_counts").items()}
    if sorted(counts) != durations():
        raise ValueError(f"probability.duration_counts {counts} does not match time.durations_min {durations()}")
    total = sum(counts.values())
    return {duration: counts[duration] / total for duration in durations()}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def verified_pinned_files() -> dict[str, Path]:
    """Pinned reduction files after SHA-256 verification (raises on any mismatch)."""
    files = {}
    for key in PINNED_KEYS:
        path = cfg.package_path(f"pinned.clustering.{key}")
        expected = str(cfg.get(f"pinned.clustering.{key}_sha256")).upper()
        if not path.is_file():
            raise FileNotFoundError(f"pinned file missing: {path}")
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for pinned file {path.name}: expected {expected}, observed {observed}")
        files[key] = path
    return files


# ---------------------------------------------------------------------------
# Corridor-constrained K-means rule (see module docstring)
# ---------------------------------------------------------------------------
def fit_options(data: dict, maximum_k: int, corridors: list[str], seed: int, n_init: int) -> dict:
    options = {}
    corridor_array = np.asarray(data["corridor"], dtype=object)
    for corridor in corridors:
        indices = np.flatnonzero(corridor_array == corridor)
        if not len(indices):
            continue
        complete_count = int(data["complete"][indices].sum())
        options[corridor] = {}
        x = data["x"][indices]
        for local_k in range(1, min(maximum_k, complete_count) + 1):
            fitted = KMeans(n_clusters=local_k, random_state=seed, n_init=n_init, algorithm="lloyd").fit(x)
            labels = fitted.labels_.astype(int)
            centers = fitted.cluster_centers_
            options[corridor][local_k] = {
                "indices": indices, "labels": labels, "centers": centers,
                "sse": float(np.square(x - centers[labels]).sum()),
            }
    return options


def allocate(options: dict, total_k: int) -> dict[str, int]:
    """Corridor budgets with minimum total SSE (dynamic programme over corridors)."""
    states: dict[int, tuple[float, dict[str, int]]] = {0: (0.0, {})}
    for corridor, corridor_options in options.items():
        next_states = {}
        for used, (cost, allocation) in states.items():
            for local_k, fitted in corridor_options.items():
                new_total = used + local_k
                if new_total > total_k:
                    continue
                new_cost = cost + fitted["sse"]
                previous = next_states.get(new_total)
                if previous is None or new_cost < previous[0] - 1e-15:
                    next_states[new_total] = (new_cost, {**allocation, corridor: local_k})
        states = next_states
    if total_k not in states:
        raise RuntimeError(f"K={total_k} cannot be allocated across corridors {list(options)}")
    return states[total_k][1]


def select_representatives(data: dict, labels: np.ndarray, centers: np.ndarray, corridors: list[str]) -> np.ndarray:
    corridor_array = np.asarray(data["corridor"], dtype=object)
    representatives = np.full(len(centers), -1, dtype=int)
    for corridor in corridors:
        clusters = np.unique(labels[corridor_array == corridor])
        if not len(clusters):
            continue
        candidates = np.flatnonzero((corridor_array == corridor) & np.asarray(data["complete"], dtype=bool))
        distance = np.linalg.norm(centers[clusters][:, None, :] - data["x"][candidates][None, :, :], axis=2)
        penalty = (float(distance.max()) + 1.0) * 1000.0
        cost = distance + (labels[candidates][None, :] != clusters[:, None]) * penalty
        # Frozen station-code tie rule (lexicographic order of station codes).
        cost = cost + np.arange(len(candidates), dtype=float)[None, :] * TIE_EPSILON
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
    for corridor in options:
        fitted = options[corridor][allocation[corridor]]
        labels[fitted["indices"]] = fitted["labels"] + offset
        centers.extend(fitted["centers"])
        offset += allocation[corridor]
    centers = np.asarray(centers, dtype=float)
    if offset != total_k or (labels < 0).any():
        raise AssertionError("combined partition is incomplete")
    representatives = select_representatives(data, labels, centers, list(options))
    return {"allocation": allocation, "labels": labels, "centers": centers, "representatives": representatives}


def frames(data: dict, result: dict, k: int, hour: int, weights: dict[int, float], method: str):
    labels = result["labels"]
    reps = result["representatives"]
    names = [data["stations"][index] for index in reps]
    representative_rows, branch_rows, assignment_rows = [], [], []
    for cluster, index in enumerate(reps):
        representative_rows.append({
            "hour": hour, "k": k, "cluster": cluster,
            "representative_station": names[cluster],
            "selection_rule": "complete 30/60 station nearest corridor-constrained K-means cluster unweighted centroid",
            "centroid_fallback_used": bool(labels[index] != cluster),
            "all_duration_source_available": True,
            "station_marginal_probability": float(data["station_probability"][index]),
            "distance_to_centroid": float(np.linalg.norm(data["x"][index] - result["centers"][cluster])),
            "method": method,
            "corridor": str(data["corridor"][index]),
        })
    for duration in weights:
        for cluster, name in enumerate(names):
            members = np.flatnonzero(labels == cluster)
            mass = float(data["station_probability"][members].sum())
            branch_rows.append({
                "hour": hour, "k": k, "cluster": cluster, "representative_station": name,
                "duration": duration, "cluster_size": len(members),
                "conditional_cluster_probability": mass,
                "duration_weight": weights[duration],
                "scenario_probability": weights[duration] * mass,
                "source_scenario_available": True,
                "method": method,
                "corridor": str(data["corridor"][reps[cluster]]),
            })
        for station_index, station in enumerate(data["stations"]):
            cluster = int(labels[station_index])
            assignment_rows.append({
                "hour": hour, "duration": duration, "station": station,
                "station_marginal_probability": float(data["station_probability"][station_index]),
                "conditional_station_probability": float(data["station_probability"][station_index]),
                "cluster": cluster, "representative_station": names[cluster],
                "method": method,
                "corridor": str(data["corridor"][station_index]),
                "representative_corridor": str(data["corridor"][reps[cluster]]),
            })
    return tuple(map(pd.DataFrame, (representative_rows, branch_rows, assignment_rows)))


def recompute(windows: tuple[int, ...], scenario_dir: Path, read_workers: int) -> tuple[dict[str, pd.DataFrame], dict]:
    k = int(cfg.get("clustering.k_per_window"))
    seed = int(cfg.get("clustering.reduction.seed"))
    n_init = int(cfg.get("clustering.reduction.n_init"))
    beta = float(cfg.get("probability.station_smoothing_beta"))
    method = str(cfg.get("clustering.method"))
    location_support = str(cfg.get("clustering.reduction.location_support"))
    weights = duration_weights()
    core = load_module("production_clustering_core", ROOT / "02_reduction_method_comparison.py")
    core.HOURS = tuple(windows)
    core.SELECTED_K = k
    core.DURATION_WEIGHT = dict(weights)
    corridors = core.corridor_order_all()

    station_corridor, families = core.build_corridor_map()
    history, history_audit = core.load_history(
        cfg.path("inputs.incidents_gt10"), cfg.path("inputs.incidents_raw"), cfg.path("inputs.incident_cohort"))
    observed = {int(d): int(history["duration_state"].eq(d).sum()) for d in durations()}
    expected = {int(key): int(value) for key, value in cfg.get("probability.duration_counts").items()}
    if observed != expected:
        raise ValueError(f"cohort duration counts {observed} differ from probability.duration_counts {expected}")
    corridor_probability, corridor_table, _, station_counts, _ = core.historical_corridor_probability(
        history, station_corridor)
    support, valid_support, support_audit = core.discover_scenarios(scenario_dir, read_workers, location_support)
    _, raw = core.load_impact_features(support, read_workers)

    tables: list[list[pd.DataFrame]] = [[], [], []]
    probability_rows = []
    for hour in windows:
        data = core.build_hour(support, valid_support, raw, hour, location_support)
        probability_rows.extend(core.apply_corridor_probability(
            data, station_corridor, corridor_probability, station_counts, beta))
        # Candidate rule: local K capped by the stations with a positive impact in every duration.
        data["complete"] = data["all_duration_positive_impact"]
        options = fit_options(data, k, corridors, seed, n_init)
        data["complete"] = data["all_duration_valid"]
        result = fit_k(data, options, k)
        if not np.array_equal(result["labels"][result["representatives"]], np.arange(k)):
            raise AssertionError(f"H{hour:02d}: out-of-cluster representative")
        for destination, frame in zip(tables, frames(data, result, k, hour, weights, method)):
            destination.append(frame)
        print(f"H{hour:02d}: K={k}, internal representatives, allocation {result['allocation']}", flush=True)
    representatives, branches, assignments = [pd.concat(parts, ignore_index=True) for parts in tables]

    retained = assignments[["hour", "station"]].drop_duplicates().shape[0]
    guard = int(cfg.get("clustering.guards.retained_cases"))
    if guard and tuple(windows) == tuple(cfg.get("time.windows")) and retained != guard:
        raise AssertionError(f"retained location-window cases {retained} != clustering.guards.retained_cases {guard}")
    if len(representatives) != k * len(windows) or len(branches) != k * len(windows) * len(weights):
        raise AssertionError("reduced support size differs from K x windows (x durations)")
    if not np.allclose(branches.groupby("hour").scenario_probability.sum(), 1.0):
        raise AssertionError("branch probabilities do not sum to one per window")
    corridor_definition = pd.DataFrame([
        {"station": station, "corridor": station_corridor[station], "track_families": ";".join(families[station])}
        for station in sorted(station_corridor)])
    outputs = {
        "representatives": representatives,
        "branches": branches,
        "assignments": assignments,
        "station_probability": pd.DataFrame(probability_rows),
        "corridor_probability": corridor_table,
        "corridor_definition": corridor_definition,
    }
    audit = {
        "method": "corridor-constrained unweighted K-means (production rule)",
        "production_rule": cfg.get("clustering.reduction.production_rule"),
        "seed": seed, "n_init": n_init, "k": k, "windows": list(windows),
        "duration_weights": {str(key): value for key, value in weights.items()},
        "station_smoothing_beta": beta,
        "retained_location_window_cases": int(retained),
        "features": "concatenated sqrt(duration probability) * log1p(impact)",
        "station_probabilities_used_in_fit": False,
        "station_probabilities_used_in_reduced_mass": True,
        "representatives_must_be_cluster_members": True,
        "history": history_audit, "support": support_audit,
    }
    return outputs, audit


def write_outputs(outputs: dict[str, pd.DataFrame], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for key, frame in outputs.items():
        frame.to_csv(folder / output_name(key), index=False, encoding="utf-8-sig")


def compare_with_pinned(outputs: dict[str, pd.DataFrame], pinned: dict[str, Path], windows) -> pd.DataFrame:
    """Row-level diff of the recomputed reduction against the pinned files (common columns)."""
    rows = []
    for key, columns in COMPARE_KEYS.items():
        new = outputs[key].copy()
        old = pd.read_csv(pinned[key], encoding="utf-8-sig")
        old = old[old["hour"].isin(windows)].copy()
        common = [c for c in new.columns
                  if c in old.columns and c not in columns and c not in DESCRIPTIVE_COLUMNS]
        merged = old.merge(new, on=columns, how="outer", suffixes=("_pinned", "_recomputed"), indicator=True)
        for hour, part in merged.groupby("hour"):
            missing = int((part["_merge"] != "both").sum())
            both = part[part["_merge"] == "both"]
            differing = {}
            for column in common:
                a, b = both[f"{column}_pinned"], both[f"{column}_recomputed"]
                if pd.api.types.is_float_dtype(a) or pd.api.types.is_float_dtype(b):
                    bad = ~np.isclose(a.astype(float), b.astype(float), rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
                else:
                    bad = a.astype(str).to_numpy() != b.astype(str).to_numpy()
                if bad.any():
                    differing[column] = int(bad.sum())
            rows.append({"file": key, "hour": int(hour), "rows_pinned": int((part["_merge"] != "right_only").sum()),
                         "rows_recomputed": int((part["_merge"] != "left_only").sum()), "unmatched_rows": missing,
                         "differing_columns": json.dumps(differing), "identical": missing == 0 and not differing})
    return pd.DataFrame(rows)


def parse_windows(value: str | None) -> tuple[int, ...]:
    allowed = tuple(int(w) for w in cfg.get("time.windows"))
    if not value:
        return allowed
    selected = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            left, right = (int(item) for item in part.split("-", 1))
            selected.update(range(left, right + 1))
        elif part:
            selected.add(int(part))
    result = tuple(sorted(selected))
    if not result or any(w not in allowed for w in result):
        raise ValueError(f"--windows must be within time.windows {allowed}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recompute", action="store_true",
                        help="with pinned.use_pinned: rebuild the reduction and diff it against the pinned files")
    parser.add_argument("--windows", default=None, help="windows to recompute, e.g. 8,17 (default: time.windows)")
    parser.add_argument("--scenario-dir", type=Path, default=cfg.results_dir / cfg.get("paths.disruptions"),
                        help="step-01 disruption scenarios (default: <results>/paths.disruptions)")
    parser.add_argument("--read-workers", type=int, default=int(cfg.get("clustering.reduction.read_workers")))
    args = parser.parse_args(argv)
    output = cfg.results_dir / cfg.get("paths.clustering")
    use_pinned = bool(cfg.get("pinned.use_pinned"))

    if use_pinned:
        pinned = verified_pinned_files()
        output.mkdir(parents=True, exist_ok=True)
        for key, path in pinned.items():
            shutil.copyfile(path, output / output_name(key))
            print(f"pinned {path.name} (SHA-256 verified) -> {output / output_name(key)}", flush=True)
        (output / output_name("audit")).write_text(json.dumps({
            "source": "pinned", "files": {key: {"pinned": cfg.get(f"pinned.clustering.{key}"),
                                                 "sha256": cfg.get(f"pinned.clustering.{key}_sha256")}
                                          for key in pinned},
            "note": "frozen Stage 1 reduction; recompute with --recompute for verification only",
        }, indent=2), encoding="utf-8")
        if not args.recompute:
            return 0
    windows = parse_windows(args.windows)
    outputs, audit = recompute(windows, args.scenario_dir.resolve(), args.read_workers)
    if not use_pinned:
        write_outputs(outputs, output)
        audit["source"] = "recomputed"
        (output / output_name("audit")).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"reduction written to {output}", flush=True)
        return 0
    check = output / "recompute_check"
    write_outputs(outputs, check)
    report = compare_with_pinned(outputs, pinned, windows)
    report.to_csv(check / "recompute_vs_pinned.csv", index=False, encoding="utf-8-sig")
    audit["source"] = "recomputed for verification"
    audit["identical_to_pinned_by_file_window"] = bool(report["identical"].all())
    (check / output_name("audit")).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report.to_string(index=False), flush=True)
    print(f"recompute check written to {check}; identical={bool(report['identical'].all())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
