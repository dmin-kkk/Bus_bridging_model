"""Run the exact finite-pool nonanticipative Stage-1 block CG.

This runner is intentionally window-by-window.  The correctness gate is
``test_cg_stage1.py``; run it before a real window.  Production uses certified
dynamic partial pricing: promising omitted columns are priced in deterministic
batches, while a complete canonical-pool pass is still required before the
root LP can be reported as certified.

Examples
--------
Run the synthetic correctness gate::

    python -X utf8 -m unittest -v test_cg_stage1.py

Prepare and solve one real window (LP CG only)::

    python -X utf8 stage1_exact_cg.py \
        --window 20 \
        --route-file BusRoutes_LOC14_EXP_K3-5_T120.csv \
        --allow-unaccelerated-large-run

Add ``--final-milp`` to obtain a restricted-pool incumbent UB after a certified
LP convergence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import importlib.util
import json
import math
import os
import re
import signal
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

BASE = Path(__file__).resolve().parent
PIPELINE_ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parents[2]))   # package root (bbconfig)
from bbconfig import cfg  # noqa: E402
PIPELINE_INPUTS = Path(os.environ.get("PAPER_PIPELINE_INPUT_DIR")
                       or cfg.data_dir).resolve()
PIPELINE_RESULTS = Path(os.environ.get("PAPER_PIPELINE_RESULTS_DIR")
                        or cfg.results_dir).resolve()
sys.path.insert(0, str(BASE / "CG_Stage1_Stage2"))
import cg_stage1 as cg
import cg_problem_builder as problem_builder

NA_FILE = BASE / "stage1_kmeans_runner.py"
DEPOT_CANDIDATE_FILE = PIPELINE_INPUTS / cfg.get("inputs.depots")
MATCHED_REPRESENTATIVE_FILE = (
    BASE / "Matched_KMeans_Representatives" /
    "matched_cluster_representatives_primary.csv")
EXACT_DISRUPT_BUFFER = cg.DEFAULT_DISRUPT_BUFFER
# Eq. (21) [eq:deadhead_bound]: depot access bound mu_max (Table 3: 120 min).
EXACT_DEPOT_ACCESS_BOUND_MIN = float(cfg.get("operations.depot_access_bound_min"))
DEFAULT_OUTPUT = PIPELINE_RESULTS / cfg.get("paths.stage1")
UNACCELERATED_POOL_LIMIT = 5_000
RUN_SIGNATURE_SCHEMA = "stage1-exact-cg-production-v6-exact-structural-speed"
DEFAULT_MODELED_DURATIONS = (30, 60, 90)
FINAL_MIP_GAP_TOLERANCE = 1e-9
ACCEPTED_FINAL_MILP_STATUSES = frozenset({"OPTIMAL"})
IMPLEMENTATION_FILES = (
    Path(__file__).resolve(),
    BASE / "CG_Stage1_Stage2" / "cg_stage1.py",
    BASE / "CG_Stage1_Stage2" / "state_compression.py",
    BASE / "CG_Stage1_Stage2" / "queue_compression.py",
    BASE / "CG_Stage1_Stage2" / "walking_elimination.py",
    BASE / "CG_Stage1_Stage2" / "cg_problem_builder.py",
    NA_FILE,
    BASE / "stage1_network_model.py",
)
DECISION_VARIABLE_FIELDS = (
    "Variable", "Omega", "Origin", "Dest", "Time", "StationID",
    "TrainID", "Value",
)


def load_na_module():
    spec = importlib.util.spec_from_file_location("stage1_na_exact_cg", NA_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_candidate_depots(path: Path = DEPOT_CANDIDATE_FILE) -> tuple[str, ...]:
    """Return exactly the depot codes listed in the explicit candidate file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"depot candidate file not found: {path}")
    depots = []
    seen = set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = {"Code"} - fields
        if missing:
            raise ValueError(
                f"depot candidate file is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            depot = str(row.get("Code", "")).strip().upper()
            if not depot:
                raise ValueError(
                    f"depot candidate file line {line_number}: blank Code")
            if depot in seen:
                raise ValueError(f"duplicate depot candidate: {depot}")
            seen.add(depot)
            depots.append(depot)
    if not depots:
        raise ValueError("depot candidate file contains no station")
    return tuple(sorted(depots))


def _resolve_input_path(value) -> Path:
    """Resolve command-line inputs relative to this runner, not the shell cwd."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = BASE / path
    return path.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _display_input_path(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(BASE).as_posix()
    except ValueError:
        return str(path)


def _modeled_durations(args) -> tuple[int, ...]:
    """Read modeled durations from an explicit representative-branch table."""
    if not args.representative_branches:
        return DEFAULT_MODELED_DURATIONS
    path = _resolve_input_path(args.representative_branches)
    if not path.is_file():
        raise FileNotFoundError(f"representative branch file not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "duration" not in set(reader.fieldnames or ()):
            raise ValueError("representative branch file lacks duration column")
        durations = sorted({int(row["duration"]) for row in reader})
    if not durations or any(duration <= 0 for duration in durations):
        raise ValueError(f"invalid modeled durations: {durations}")
    return tuple(durations)


def _hash_file_group(paths) -> dict:
    """Hash a deterministic file group without storing a huge manifest."""
    resolved = sorted(
        {Path(path).resolve() for path in paths},
        key=lambda path: str(path).casefold())
    if not resolved:
        raise FileNotFoundError("run-signature input group is empty")
    items = []
    total_bytes = 0
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(f"run-signature input not found: {path}")
        size = int(path.stat().st_size)
        total_bytes += size
        items.append({
            "path": _display_input_path(path),
            "bytes": size,
            "sha256": _sha256_file(path),
        })
    result = {
        "file_count": len(items),
        "total_bytes": total_bytes,
        "sha256": _canonical_sha256(items),
    }
    if len(items) == 1:
        result["path"] = items[0]["path"]
        result["file_sha256"] = items[0]["sha256"]
    return result


def _current_disruption_input_files(hour: int, disruption_dir: Path,
                                    manifests, modeled_durations) -> tuple[Path, ...]:
    """Mirror the core's manifest gate and return only consumed result CSVs.

    The result directory also contains old postprocessed files matching the
    broad ``Disrupt_H_*`` glob.  The preprocessing core reads only filenames
    whose final token is a modeled duration and whose latest manifest record is
    the current logic version with checkpoint ``EXTRACTED``.  Hashing exactly
    that set, plus the manifests themselves, preserves stale-result safety
    without rereading ignored gigabytes before every window.
    """
    manifest = {}
    required = {"scenario", "logic_version", "checkpoint"}
    for path in manifests:
        with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not required.issubset(set(reader.fieldnames or ())):
                continue
            for row in reader:
                scenario_id = str(row.get("scenario", "")).strip()
                if scenario_id:
                    manifest[scenario_id] = (
                        str(row.get("logic_version", "")).strip(),
                        str(row.get("checkpoint", "")).strip(),
                    )

    current_version = str(problem_builder.EXPECTED_SCENARIO_LOGIC_VERSION)
    modeled_durations = {int(value) for value in modeled_durations}
    selected = []
    prefix = f"Disrupt_{int(hour)}_"
    for path in sorted(
            Path(disruption_dir).glob(f"{prefix}*.csv"),
            key=lambda candidate: candidate.name.casefold()):
        middle = path.name[len(prefix):-4]
        separator = middle.rfind("_")
        if separator < 0:
            continue
        try:
            duration = int(middle[separator + 1:])
        except ValueError:
            continue
        if duration not in modeled_durations:
            continue
        scenario_id = f"{int(hour)}_{middle}"
        record = manifest.get(scenario_id)
        if record == (current_version, "EXTRACTED"):
            selected.append(path.resolve())
    if not selected:
        raise FileNotFoundError(
            f"window {hour}: no current EXTRACTED disruption inputs")
    return tuple(selected)


def _production_input_groups(args) -> dict[str, tuple[Path, ...]]:
    """Return every data-file family that can affect one production window."""
    hour = int(args.window)
    normal = PIPELINE_RESULTS / cfg.get("paths.normal") / (
        f"Normal_{hour:02d}_{hour + 4:02d}.csv")
    disruption_dir = PIPELINE_RESULTS / cfg.get("paths.disruptions")
    manifests = tuple(path for path in (
        disruption_dir / "_scenario_stats.csv",
        disruption_dir / f"_scenario_stats_w{hour:02d}.csv",
    ) if path.is_file())
    disruption_files = _current_disruption_input_files(
        hour, disruption_dir, manifests, _modeled_durations(args))
    groups = {
        "route_library": (_resolve_input_path(args.route_file),),
        "depot_candidates": (DEPOT_CANDIDATE_FILE.resolve(),),
        "scenario_probabilities": (
            Path(problem_builder.SCENARIO_PROBABILITY_FILE).resolve(),),
        "train_schedule": (PIPELINE_INPUTS / cfg.get("inputs.timetable"),),
        "hourly_od": (PIPELINE_INPUTS / cfg.get("inputs.od_matrix"),),
        "station_distance_matrix": (
            PIPELINE_INPUTS / cfg.get("inputs.road_times"),),
        "corridor_topology": (PIPELINE_INPUTS / cfg.get("inputs.corridors"),),
        "normal_window": (normal,),
        "disruption_manifests": manifests,
        "disruption_window": disruption_files,
    }
    if args.representative_branches:
        groups["representative_branches"] = (
            _resolve_input_path(args.representative_branches),)
    depot_start_path = _resolved_depot_start_path(args)
    if depot_start_path is not None and depot_start_path.is_file():
        groups["previous_depot_start"] = (depot_start_path,)
    return groups


def _resolved_depot_start_path(args) -> Path | None:
    if not bool(args.final_milp_previous_depot_start):
        return None
    if args.final_milp_depot_start_file:
        return _resolve_input_path(args.final_milp_depot_start_file)
    if int(args.window) <= 6:
        return None
    return (
        Path(args.output).expanduser().resolve() / "Allocation" /
        f"DepotAlloc_{int(args.window) - 1:02d}.csv")


def _load_depot_start(path: Path | None, depots, max_total_buses: int) -> dict:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = {str(row["Depot"]): float(row["Count"]) for row in rows}
    expected = {str(depot) for depot in depots}
    if set(counts) != expected:
        raise ValueError(
            f"depot start set differs from current candidates: {path}")
    if any(not math.isfinite(value) or value < 0 or
           abs(value - round(value)) > 1e-9 for value in counts.values()):
        raise ValueError(f"depot start contains invalid counts: {path}")
    if math.fsum(counts.values()) > int(max_total_buses) + 1e-9:
        raise ValueError(f"depot start exceeds fleet budget: {path}")
    return {depot: int(round(value)) for depot, value in counts.items()}


def _resolved_column_start_path(args) -> Path | None:
    raw = getattr(args, "final_milp_column_start", None)
    return _resolve_input_path(raw) if raw else None


def _column_start_files(path: Path | None) -> tuple[Path, ...]:
    if path is None:
        return ()
    path = Path(path)
    if path.is_dir():
        return tuple(sorted(path.glob("BusOperations_*.csv")))
    return (path,)


def _load_column_start(path: Path | None) -> set:
    files = _column_start_files(path)
    if not files:
        return set()
    selected = set()
    for source in files:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                value = float(row.get("Y", 0.0) or 0.0)
                if value <= 0.5:
                    continue
                cell = tuple(int(token) for token in re.findall(
                    r"\d+", str(row.get("InformationCell", ""))))
                if not cell:
                    raise ValueError(
                        f"column start has no information cell: {source}")
                selected.add((
                    str(row["Block"]), str(row["TripID"]), cell))
    if not selected:
        raise ValueError(f"column start contains no selected y variables: {path}")
    return selected


def _effective_run_config(args) -> dict:
    """Serialize every run switch that may alter the model or stopping rule."""
    modeled_durations = _modeled_durations(args)
    return {
        "signature_schema": RUN_SIGNATURE_SCHEMA,
        "window": int(args.window),
        "route_pool_provenance": problem_builder.ROUTE_POOL_PROVENANCE,
        "route_file": str(_resolve_input_path(args.route_file)),
        "representatives_per_duration": int(args.k),
        "modeled_durations": list(modeled_durations),
        "representative_branches": (
            str(_resolve_input_path(args.representative_branches))
            if args.representative_branches else None),
        "disrupt_buffer_minutes": int(args.buffer),
        "seed_per_block": int(args.seed_per_block),
        "add_per_iteration": int(args.add_per_iteration),
        "partial_pricing_batch_size": int(args.partial_pricing_batch_size),
        "partial_pricing_reoptimize_after": int(
            args.partial_pricing_reoptimize_after),
        "pricing_workers": int(args.pricing_workers),
        "pricing_model_cache_size": int(args.pricing_model_cache_size),
        "pricing_model_cache_policy": args.pricing_model_cache_policy,
        "rotate_partial_pricing": bool(args.rotate_partial_pricing),
        "search_rmp_optimality_tolerance": (
            None if args.search_rmp_optimality_tol is None else
            float(args.search_rmp_optimality_tol)),
        "search_rc_epsilon": (
            None if args.search_rc_epsilon is None else
            float(args.search_rc_epsilon)),
        "rmp_method": args.rmp_method,
        "rmp_lp_warm_start": args.rmp_lp_warm_start,
        "final_method": args.final_method,
        "final_mip_focus": args.final_mip_focus,
        "final_cuts": args.final_cuts,
        "final_symmetry": args.final_symmetry,
        "final_heuristics": args.final_heuristics,
        "independent_trips": bool(args.independent_trips),
        "final_mip_gap": args.final_mip_gap,
        "final_milp_rmp_hints": bool(args.final_milp_rmp_hints),
        "final_milp_previous_depot_start": bool(
            args.final_milp_previous_depot_start),
        "final_milp_depot_start_file": (
            str(_resolved_depot_start_path(args))
            if _resolved_depot_start_path(args) is not None else None),
        "final_milp_column_start": (
            str(_resolved_column_start_path(args))
            if _resolved_column_start_path(args) is not None else None),
        "final_milp_column_start_sha256": (
            _hash_file_group(_column_start_files(
                _resolved_column_start_path(args)))["sha256"]
            if _resolved_column_start_path(args) is not None else None),
        "max_iterations": int(args.max_iterations),
        "gurobi_time_limit": "unlimited",
        "final_milp_time_limit_seconds": args.final_milp_time_limit_seconds,
        "threads": int(args.threads),
        "gurobi_version": ".".join(
            str(part) for part in cg.gp.gurobi.version()),
        "final_milp": bool(args.final_milp),
        "cg_stall_minutes": args.cg_stall_minutes,
        "final_milp_stall_minutes": args.final_milp_stall_minutes,
        "box_screening": bool(args.box_screening),
        "persistent_rmp": not bool(args.legacy_rebuild_rmp),
        "validate_persistent_final": bool(args.validate_persistent_final),
        "fleet_size": int(os.getenv("TSB_MAX_BUSES")
                          or cfg.get("operations.fleet_size")),
        "mip_gap": float(cfg.get("solver.stage1.mip_gap")),
        "route_source": (os.getenv("TSB_ROUTE_SOURCE")
                         or cfg.get("routes.source_mode")).strip().lower(),
        "bus_board_mode": (os.getenv("TSB_BUS_BOARD")
                           or cfg.get("routes.bus_boarding_mode")).strip().lower(),
        "depot_access_bound_minutes": EXACT_DEPOT_ACCESS_BOUND_MIN,
        "max_depots_per_route": int(os.getenv("TSB_MAX_DEPOTS_PER_ROUTE")
                                    or cfg.get("stage1.max_depots_per_route")),
        "cg_optimality_tolerance": float(os.getenv("TSB_CG_OPT_TOL")
                                         or cfg.get("solver.stage1.cg_opt_tol")),
        "cg_feasibility_tolerance": float(os.getenv("TSB_CG_FEAS_TOL")
                                          or cfg.get("solver.stage1.cg_feas_tol")),
        "eliminate_w": True,
        "eliminate_destination_states": True,
        "compress_load_states": True,
        "compress_queue_states": True,
        "hybrid_bus_load_states": True,
        "state_reformulation": (
            "exact-event-hybrid-final-load-depot-strengthening-v5"),
        "depot_overlap_rows": "distinct-clipped-interval-starts-exact",
        "terminal_empty_rows": "one-nonnegative-sum-row-per-vehicle-exact",
        "directed_route_timing": "raw-directed-cumulative-rounding",
        "scenario_topology": (
            "duration-k-terminals_then-common-station-union-support"),
        "weighted_terminal_count": len(modeled_durations) * int(args.k),
    }


def _root_cg_config_hash(args) -> str:
    """Hash only switches that can alter root CG or its certificate."""
    config = _effective_run_config(args)
    final_only = {
        "final_method", "final_mip_focus", "final_cuts", "final_symmetry",
        "final_heuristics", "final_milp_rmp_hints",
        "final_milp_previous_depot_start", "final_milp_depot_start_file",
        "final_milp_column_start", "final_milp_column_start_sha256",
        "final_milp_time_limit_seconds", "final_mip_gap", "final_milp",
        "final_milp_stall_minutes", "mip_gap",
    }
    return _canonical_sha256({
        key: value for key, value in config.items()
        if key not in final_only
    })


def build_run_signature(args, *, input_groups=None,
                        implementation_files=None) -> dict:
    """Build content-based hashes used by checkpoints and batch skip guards."""
    config = _effective_run_config(args)
    groups = (_production_input_groups(args) if input_groups is None
              else input_groups)
    inputs = {
        str(name): _hash_file_group(paths)
        for name, paths in sorted(groups.items())
    }
    implementation = _hash_file_group(
        IMPLEMENTATION_FILES if implementation_files is None
        else implementation_files)
    config_hash = _canonical_sha256(config)
    input_hash = _canonical_sha256(inputs)
    implementation_hash = str(implementation["sha256"])
    engine_path = Path(cg.__file__).resolve()
    engine = {
        "path": str(engine_path),
        "sha256": _sha256_file(engine_path),
    }
    run_hash = _canonical_sha256({
        "schema": RUN_SIGNATURE_SCHEMA,
        "config_sha256": config_hash,
        "input_sha256": input_hash,
        "implementation_sha256": implementation_hash,
        "engine_sha256": engine["sha256"],
    })
    return {
        "schema": RUN_SIGNATURE_SCHEMA,
        "config": config,
        "inputs": inputs,
        "implementation": implementation,
        "engine": engine,
        "config_sha256": config_hash,
        "input_sha256": input_hash,
        "implementation_sha256": implementation_hash,
        "run_sha256": run_hash,
    }


def _normalise_reduced_dispatch_windows(
        scenarios, buffer: int = EXACT_DISRUPT_BUFFER):
    """Apply the exact-path dispatch buffer to every reduced duration branch.

    In particular, the preprocessing adapter historically synthesized
    zero-impact children with no post-disruption buffer.  Normalising all
    branches here, before the common trip pool is built, prevents those
    children from silently losing physical candidates.
    """
    buffer = int(buffer)
    if buffer < 0:
        raise ValueError("dispatch buffer must be nonnegative")
    for scenario in scenarios:
        start = int(scenario["DISRUPT_START_T"])
        duration = int(scenario["duration"])
        scenario["DISRUPT_END_T"] = start + duration
        scenario["DISPATCH_END_MIN"] = start + duration + buffer
    return scenarios


def _load_legacy_k10_duration_representatives(core, raw, start_hour: int,
                                              k: int, wdata):
    """Return the selected terminal rows, without triplet expansion.

    The published matched-representative file contains the 30 original
    duration-cluster choices for K=10.  Its companion ``matched_station_triplets``
    file deliberately expands unique stations to other durations and is not
    read here.
    """
    if int(k) != 10:
        raise ValueError("legacy matched-representative loader supports only K=10")
    path = MATCHED_REPRESENTATIVE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"matched representative file not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle)
                if int(row["hour"]) == int(start_hour)]
    if len(rows) != 3 * int(k):
        raise RuntimeError(
            f"window {start_hour}: matched representative file must contain "
            f"exactly {3 * int(k)} rows, found {len(rows)}")

    raw_by_key = {}
    for scenario in raw:
        key = (str(scenario["station"]).strip().upper(),
               int(scenario["duration"]))
        if key in raw_by_key:
            raise RuntimeError(f"duplicate raw terminal scenario: {key}")
        raw_by_key[key] = scenario

    selected = []
    seen_clusters = set()
    for row in rows:
        duration = int(row["duration"])
        cluster = int(row["cluster"])
        station = str(row["representative_station"]).strip().upper()
        cluster_key = (duration, cluster)
        if cluster_key in seen_clusters:
            raise RuntimeError(
                f"duplicate matched duration-cluster row: {cluster_key}")
        seen_clusters.add(cluster_key)
        key = (station, duration)
        source = raw_by_key.get(key)
        if source is None:
            # The raw loader drops a disruption file when its recomputed
            # affected-OD set is empty.  Retain the selected terminal row as a
            # zero-impact scenario; do not replace it with, or expand it to,
            # another duration.  A same-station sibling supplies only the
            # common window metadata.
            siblings = [scenario for (raw_station, _), scenario
                        in raw_by_key.items() if raw_station == station]
            if siblings:
                scenario = max(
                    siblings, key=lambda item: int(item["duration"])).copy()
                disrupt_start = int(scenario["DISRUPT_START_T"])
            else:
                # K-means is fitted to every disruption result file, whereas
                # load_disruption_scenarios deliberately omits rows whose
                # affected-OD set is empty.  Such a station can consequently
                # be a valid selected representative without leaving any raw
                # same-station sibling.  Its window start is common to every
                # scenario, so materialise the selected row as the intended
                # passenger-flow-free, resource-only information branch.
                if wdata is None or "START_MIN" not in wdata:
                    raise RuntimeError(
                        "cannot restore zero-impact matched representative "
                        f"without window START_MIN: {key}")
                scenario = {}
                disrupt_start = (
                    int(wdata["START_MIN"])
                    + int(getattr(core, "DISRUPT_OFFSET", 10))
                )
            scenario.update({
                "id": f"{int(start_hour)}_{station}_{duration}",
                "station": station,
                "duration": duration,
                "DISRUPT_START_T": disrupt_start,
                "DISRUPT_END_T": disrupt_start + duration,
                "DISPATCH_END_MIN": disrupt_start + duration,
                "canceled_trains": set(),
                "theta": {},
                "Omega_m": [],
                "spare_capacity": {},
                "pair_score": {},
                "strand_score": {},
                "impact_by_station": {},
                "zero_impact_branch": True,
            })
        else:
            scenario = source.copy()
            scenario["zero_impact_branch"] = False
        scenario["probability"] = float(row["original_scenario_probability"])
        scenario["prob_share"] = float(row["cluster_station_probability"])
        scenario["cluster_id"] = cluster
        scenario["cluster_size"] = int(row["cluster_size"])
        scenario["representative_id"] = scenario["id"]
        scenario["distance_to_feature_centroid"] = float(
            row["distance_to_centroid"])
        scenario["selected_duration_count"] = int(
            row["selected_duration_count"])
        selected.append(scenario)
    return selected, "matched_duration_cluster_representatives"


def _load_corridor_representative_branches(
        core, raw, start_hour: int, k: int, wdata, path: Path):
    """Load a validated common-station corridor-reduction branch table."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"representative branch file not found: {path}")
    required = {
        "hour", "k", "cluster", "representative_station", "duration",
        "cluster_size", "conditional_cluster_probability", "duration_weight",
        "scenario_probability", "method",
    }
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"representative branch file lacks columns: {sorted(missing)}")
        rows = [row for row in reader
                if int(row["hour"]) == int(start_hour)]
    durations = tuple(int(value) for value in core.DURATIONS)
    expected_rows = len(durations) * int(k)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"window {start_hour}: expected {expected_rows} representative "
            f"branches, found {len(rows)}")
    if any(int(row["k"]) != int(k) for row in rows):
        raise ValueError(
            f"window {start_hour}: representative branch K does not match {k}")

    stations_by_duration = {}
    for duration in durations:
        duration_rows = [row for row in rows
                         if int(row["duration"]) == duration]
        if len(duration_rows) != int(k):
            raise RuntimeError(
                f"window {start_hour}, duration {duration}: expected K={k}, "
                f"found {len(duration_rows)}")
        clusters = {int(row["cluster"]) for row in duration_rows}
        if len(clusters) != int(k):
            raise ValueError(
                f"window {start_hour}, duration {duration}: duplicate cluster")
        stations = {
            str(row["representative_station"]).strip().upper()
            for row in duration_rows
        }
        if len(stations) != int(k):
            raise ValueError(
                f"window {start_hour}, duration {duration}: duplicate station")
        stations_by_duration[duration] = stations
    reference_stations = stations_by_duration[durations[0]]
    if any(stations_by_duration[duration] != reference_stations
           for duration in durations[1:]):
        raise ValueError(
            f"window {start_hour}: representative stations differ by duration")

    raw_by_key = {}
    for scenario in raw:
        key = (str(scenario["station"]).strip().upper(),
               int(scenario["duration"]))
        if key in raw_by_key:
            raise ValueError(f"duplicate raw disruption scenario: {key}")
        raw_by_key[key] = scenario
    disrupt_start = (
        int(wdata["START_MIN"]) + int(getattr(core, "DISRUPT_OFFSET", 10)))

    selected = []
    seen = set()
    for row in sorted(rows, key=lambda item: (
            int(item["duration"]), int(item["cluster"]))):
        station = str(row["representative_station"]).strip().upper()
        duration = int(row["duration"])
        key = (station, duration)
        if key in seen:
            raise ValueError(f"duplicate representative branch: {key}")
        seen.add(key)
        source = raw_by_key.get(key)
        if source is None:
            branch = {
                "id": f"{int(start_hour)}_{station}_{duration}",
                "logic_version": getattr(core, "SCENARIO_LOGIC_VERSION", None),
                "station": station,
                "duration": duration,
                "DISRUPT_START_T": disrupt_start,
                "DISRUPT_END_T": disrupt_start + duration,
                "DISPATCH_END_MIN": disrupt_start + duration,
                "canceled_trains": set(),
                "theta": {},
                "Omega_m": [],
                "spare_capacity": {},
                "pair_score": {},
                "strand_score": {},
                "impact_by_station": {},
                "zero_impact_branch": True,
            }
        else:
            branch = source.copy()
            branch["zero_impact_branch"] = bool(
                source.get("zero_impact_branch", False))

        probability = float(row["scenario_probability"])
        conditional = float(row["conditional_cluster_probability"])
        duration_weight = float(row["duration_weight"])
        cluster_method = str(row["method"]).strip()
        if not cluster_method:
            raise ValueError(f"missing clustering method at {key}")
        if (not math.isfinite(probability) or probability < 0 or
                not math.isfinite(conditional) or conditional < 0 or
                not math.isfinite(duration_weight) or duration_weight < 0):
            raise ValueError(f"invalid representative probability at {key}")
        branch.update({
            "station": station,
            "duration": duration,
            "block_station": station,
            "probability": probability,
            "prob_station": conditional,
            "prob_share": conditional,
            "duration_weight": duration_weight,
            "cluster_id": int(row["cluster"]),
            "cluster_size": int(row["cluster_size"]),
            "representative_id": branch["id"],
            "selected_duration_count": len(durations),
            "feature_transform": "duration-probability-weighted log1p impact",
            "cluster_method": cluster_method,
            "cluster_probability_weighted": True,
            "duration_probability_weighted": True,
            "representative_branch_file": str(path),
        })
        selected.append(branch)

    total_mass = math.fsum(float(row["probability"]) for row in selected)
    if not math.isclose(total_mass, 1.0, abs_tol=1e-10):
        raise AssertionError(
            f"representative probabilities sum to {total_mass}, not one")
    for duration in durations:
        block = [row for row in selected if int(row["duration"]) == duration]
        mass = math.fsum(float(row["probability"]) for row in block)
        expected = {float(row["duration_weight"]) for row in block}
        if len(expected) != 1 or not math.isclose(
                mass, next(iter(expected)), abs_tol=1e-10):
            raise AssertionError(
                f"duration {duration} mass {mass} does not match {expected}")
    return selected


def _load_duration_representatives(
        core, raw, start_hour: int, k: int, wdata,
        representative_branches: Path | None = None):
    """Load explicit common-station branches or use the legacy reducer."""
    if representative_branches is not None:
        return (
            _load_corridor_representative_branches(
                core, raw, start_hour, k, wdata, representative_branches),
            "corridor_duration_station_probability_weighted_selected_method",
        )
    return problem_builder.reduce_duration_scenarios_kmeans(
        raw, int(k), int(start_hour)), (
        "duration_specific_unweighted_log1p_kmeans")


def _expand_terminal_station_union(core, raw, terminals, start_hour: int,
                                   wdata):
    """Expand K-per-duration terminals to complete physical station blocks.

    Clustering still selects exactly K weighted ``(station, duration)``
    terminals independently within each duration.  Nonanticipativity requires
    a common 30/60/90 physical branch set at every station in the representative
    union, however.  Existing but unselected raw branches therefore enter with
    probability zero; a missing raw branch is restored as a zero-impact,
    resource-only branch.  Only the original terminals retain cluster mass.
    """
    durations = tuple(int(value) for value in core.DURATIONS)
    raw_by_key = {
        (str(scenario["station"]).strip().upper(), int(scenario["duration"])):
        scenario
        for scenario in raw
    }
    terminal_by_key = {}
    for terminal in terminals:
        key = (str(terminal["station"]).strip().upper(),
               int(terminal["duration"]))
        if key in terminal_by_key:
            raise ValueError(f"duplicate selected terminal representative: {key}")
        terminal_by_key[key] = terminal

    stations = tuple(sorted({station for station, _duration in terminal_by_key}))
    disrupt_start = (
        int(wdata["START_MIN"]) + int(getattr(core, "DISRUPT_OFFSET", 10)))
    selected_counts = {
        station: sum(key[0] == station for key in terminal_by_key)
        for station in stations
    }
    expanded = []
    for station in stations:
        for duration in durations:
            key = (station, duration)
            terminal = terminal_by_key.get(key)
            source = terminal if terminal is not None else raw_by_key.get(key)
            if source is None:
                branch = {
                    "id": (f"{int(start_hour)}_{station}_{duration}"
                           "__support_zero"),
                    "logic_version": getattr(core, "SCENARIO_LOGIC_VERSION", None),
                    "station": station,
                    "duration": duration,
                    "DISRUPT_START_T": disrupt_start,
                    "DISRUPT_END_T": disrupt_start + duration,
                    "DISPATCH_END_MIN": disrupt_start + duration,
                    "canceled_trains": set(),
                    "theta": {},
                    "Omega_m": [],
                    "spare_capacity": {},
                    "pair_score": {},
                    "strand_score": {},
                    "impact_by_station": {},
                    "zero_impact_branch": True,
                }
            else:
                branch = source.copy()
                branch["zero_impact_branch"] = bool(
                    source.get("zero_impact_branch", False))

            is_terminal = terminal is not None
            terminal_probability = (
                float(terminal.get("probability", 0.0)) if is_terminal else 0.0)
            branch.update({
                "station": station,
                "duration": duration,
                "block_station": station,
                "probability": terminal_probability,
                "weighted_terminal": is_terminal,
                "support_only_branch": not is_terminal,
                "selected_duration_count": selected_counts[station],
            })
            if not is_terminal:
                branch["prob_station"] = 0.0
                branch["prob_share"] = 0.0
            expanded.append(branch)

    if len(expanded) != len(stations) * len(durations):
        raise AssertionError("station-union branch expansion is incomplete")
    terminal_mass = math.fsum(
        float(branch["probability"]) for branch in expanded
        if branch["weighted_terminal"])
    support_mass = math.fsum(
        float(branch["probability"]) for branch in expanded
        if not branch["weighted_terminal"])
    if not math.isclose(terminal_mass, 1.0, abs_tol=1e-12):
        raise AssertionError(
            f"weighted terminal probability must sum to one, got {terminal_mass}")
    if support_mass != 0.0:
        raise AssertionError(
            f"support-only branch probability must be zero, got {support_mass}")
    return expanded


def _attach_pre_onset_normal_actions(core, scenarios, start_hour, wdata):
    """Attach the normal pre-onset actions of each branch's affected ODs.

    Sec. 3.2.2: the disrupted CFM is restricted to t >= t0 with the states
    immediately before onset fixed to their normal values.  ``solve_master``
    fixes the b/a/od actions of every minute t < t0 to these values; the
    stations of the normal pre-onset path are added to N^omega.
    """
    normal = core.load_normal_trajectory(int(start_hour))
    onset = int(wdata["START_MIN"]) + int(core.DISRUPT_OFFSET)
    for scenario in scenarios:
        omegas = set(scenario.get("Omega_m", ()))
        support = defaultdict(set)
        for omega, stations in scenario.get(
                "required_station_support", {}).items():
            support[omega].update(stations)
        for omega, stations in cg.pre_onset_normal_station_support(
                normal, omegas, onset).items():
            support[omega].update(stations)
        scenario["required_station_support"] = {
            omega: tuple(sorted(stations))
            for omega, stations in support.items()}
        scenario["normal_pre_onset"] = {
            family: {
                key: value for key, value in normal.get(family, {}).items()
                if key[0] in omegas
                and int(key[1 if family == "od" else 2]) < onset}
            for family in ("b", "a", "od")}


def prepare_window(core, na, start_hour,
                   buffer: int = EXACT_DISRUPT_BUFFER,
                   k: int = problem_builder.DEFAULT_K_PER_DURATION,
                   representative_branches: Path | None = None):
    """Build K weighted terminals per duration and complete station blocks.

    The reducer selects exactly ``len(DURATIONS) * k`` weighted terminal
    outcomes. Their station union is then expanded to all modeled branches for the
    common nonanticipative solve.  Unselected branches have probability zero;
    missing raw branches are zero-impact and resource-only.
    """
    if int(k) <= 0:
        raise ValueError("k must be positive")
    # Keep the paper's duration-wise fixed-K reducer.  These assignments are
    # explicit because the preprocessing module also exposes a common-station
    # adapter used by a different experiment.
    core.REDUCTION = "duration"
    core.K_MODE = "fixed"
    core.K_PER_DURATION = int(k)
    schedule = core.load_schedule(core.FILE_SCHEDULE)
    od = core.load_od(core.FILE_OD)
    bus_tt = core.load_distance_matrix(core.FILE_DISTANCE)
    routes = core.load_route_library(
        core.FILE_PREDEFINED_ROUTES,
        distance_path=core.FILE_DISTANCE,
        validate_forward_times=True)
    if not routes:
        raise RuntimeError(f"route library is empty: {core.FILE_PREDEFINED_ROUTES}")
    wdata = core.build_window_data(start_hour, schedule, od)
    raw = core.load_disruption_scenarios(start_hour, wdata)
    terminals, reduction_source = _load_duration_representatives(
        core, raw, start_hour, int(k), wdata, representative_branches)
    expected_durations = tuple(int(duration) for duration in core.DURATIONS)
    terminal_duration_counts = {
        duration: sum(
            int(scenario["duration"]) == duration for scenario in terminals)
        for duration in expected_durations
    }
    expected_counts = {
        duration: min(
            int(k),
            sum(int(scenario["duration"]) == duration for scenario in raw),
        )
        for duration in expected_durations
    }
    if terminal_duration_counts != expected_counts:
        raise RuntimeError(
            "duration-wise representative count mismatch: "
            f"expected={expected_counts}, actual={terminal_duration_counts}")
    if any(expected_counts[duration] != int(k)
           for duration in expected_durations):
        raise RuntimeError(
            f"window {start_hour} has fewer than K={int(k)} raw scenarios in "
            f"at least one duration: {expected_counts}")

    scenarios = _expand_terminal_station_union(
        core, raw, terminals, start_hour, wdata)
    for scenario in scenarios:
        scenario["reduction_source"] = reduction_source
    _normalise_reduced_dispatch_windows(scenarios, buffer)
    _attach_pre_onset_normal_actions(core, scenarios, start_hour, wdata)
    depots = list(load_candidate_depots())
    by_scenario = core.build_block_trips(
        scenarios, start_hour, depots, bus_tt, routes)
    for scenario in scenarios:
        sid = scenario["id"]
        if sid not in by_scenario:
            raise RuntimeError(f"missing selected-duration trip pool for {sid}")
        _selected, trips = by_scenario[sid]
        scenario["bus_trips"] = trips
    return depots, scenarios, wdata, bus_tt


def build_full_caches(core, na, scenarios, wdata, progress_hook=None,
                      config=None):
    """Build and exactly compact the fixed full-pool row/key support once.

    When the production exact-state reformulation is enabled, prepare and
    release each scenario's minute l/q/w key grids immediately.  This avoids
    retaining all scenarios' explicit grids until a second global pass.
    """
    full_caches = {}
    for index, scenario in enumerate(scenarios, start=1):
        sid = scenario["id"]
        trips = scenario["bus_trips"]
        raw_cache = core.build_stage2_cache(
            scenario, wdata, trips, include_bus_load_keys=False)
        # Compact only provably fixed-zero keys, once, from the complete pool.
        # Keep every physical trip id so eventless duration branches retain the
        # shared dispatch variable and depot occupancy.
        compact = cg.compact_full_pool_cache(
            raw_cache, scenario, wdata, sigma=2, preserve_trip_ids=True)
        if config is not None and (
                config.compress_load_states or
                config.compress_queue_states or config.eliminate_w):
            compact = cg.prepare_exact_state_support(
                {sid: compact}, (scenario,), wdata, config,
                release_source_keys=True)[sid]
        full_caches[sid] = compact
        if progress_hook is not None:
            progress_hook({
                "phase": "BUILDING_FULL_POOL_CACHE",
                "completed_scenarios": index,
                "total_scenarios": len(scenarios),
                "scenario_id": sid,
            })
    cg.validate_shared_passenger_action_cache_support(
        full_caches, scenarios)
    return full_caches


def _write_history(path, history):
    rows = list(history)
    if not rows:
        _write_csv_atomic(path, [], ["iteration", "phase"])
        return
    _write_csv_atomic(path, rows, list(rows[0]))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp",
                delete=False) as handle:
            temporary_path = Path(handle.name)
            json.dump(_jsonable(value), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_certified_cg_checkpoint(
        path: Path, *, args, problem, config, pool) -> dict:
    """Load a hash-matched certified root-CG result for final-MILP resume."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "stage1-exact-cg-active-columns-v1":
        raise ValueError(f"unsupported CG checkpoint schema: {path}")
    expected = {
        "window": int(args.window),
        "root_cg_config_hash": _root_cg_config_hash(args),
        "input_hash": args._run_signature["input_sha256"],
        "implementation_hash": args._run_signature["implementation_sha256"],
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"CG checkpoint {field} mismatch: "
                f"{payload.get(field)!r} != {value!r}")
    pool_by_key = {column.key: column for column in pool}
    expected_pool_hash = cg._pool_hash(pool_by_key)
    if payload.get("pool_hash") != expected_pool_hash:
        raise ValueError("CG checkpoint pool hash mismatch")
    keys = []
    for row in payload.get("columns", ()):
        key = (
            str(row["block"]), str(row["trip_id"]),
            tuple(int(value) for value in row["information_cell"]))
        if key not in pool_by_key:
            raise ValueError(f"CG checkpoint contains an unknown column: {key}")
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError("CG checkpoint contains duplicate active columns")
    stored = dict(payload.get("cg_result", {}))
    if not payload.get("certified") or not stored.get("certified"):
        raise ValueError("CG checkpoint is not certified")
    if stored.get("termination_reason") != "NO_NEGATIVE_OMITTED_COLUMN":
        raise ValueError("CG checkpoint has no full omitted-column certificate")
    hint_payload = payload.get("rmp_primal_hint", {})
    hint_y = {}
    for row in hint_payload.get("y", ()):
        key = (
            str(row["block"]), str(row["trip_id"]),
            tuple(int(value) for value in row["information_cell"]))
        if key in pool_by_key:
            hint_y[key] = float(row["value"])
    stored.update({
        "pool": tuple(pool),
        "active_columns": frozenset(keys),
        "history": tuple(stored.get("history", ())),
        "final_rmp_primal_hint": {
            "x": {str(key): float(value)
                  for key, value in hint_payload.get("x", {}).items()},
            "y": hint_y,
        },
        "final_duals": {},
        "final_omitted_prices": {},
        "depot_row_keys": cg.build_depot_row_keys(problem, config),
    })
    return stored


def _write_csv_atomic(path: Path, rows, fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", newline="", encoding="utf-8-sig",
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
                delete=False) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames),
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _safe_filename_token(value) -> str:
    """Return a readable Windows-safe filename component."""
    raw = str(value)
    invalid = '<>:"/\\|?*'
    token = "".join(
        "_" if character in invalid or ord(character) < 32 else character
        for character in raw)
    token = "_".join(token.split()).strip(" ._")
    return token or "NA"


def _decision_row_dict(row) -> dict:
    if isinstance(row, dict):
        converted = {field: row.get(field) for field in DECISION_VARIABLE_FIELDS}
    else:
        values = tuple(row)
        if len(values) != len(DECISION_VARIABLE_FIELDS):
            raise ValueError(
                f"decision-variable row must have 8 fields, got {len(values)}")
        converted = dict(zip(DECISION_VARIABLE_FIELDS, values))
    minute = int(converted["Time"])
    value = float(converted["Value"])
    if not cg.DECISION_TIME_MIN <= minute <= cg.DECISION_TIME_MAX:
        raise ValueError(f"decision-variable minute outside 0..240: {minute}")
    if value <= cg.DECISION_ROW_EPS:
        raise ValueError(
            f"decision-variable export received non-positive row: {value}")
    converted["Time"] = minute
    converted["Value"] = value
    return converted


def _write_decision_variable_outputs(
        output: Path, scenario_results, scenarios, window: int):
    """Write exactly one legacy-format decision CSV per scenario."""
    directory = Path(output) / "DecisionVars"
    metadata = {str(scenario["id"]): scenario for scenario in scenarios}
    per_scenario_files = {}
    used_names = {}
    row_count = 0
    for sid, result in sorted(scenario_results.items(), key=lambda item: str(item[0])):
        sid = str(sid)
        scenario = metadata.get(sid, {})
        location = scenario.get(
            "block_station", scenario.get("station", sid))
        duration = scenario.get("duration", "NA")
        rows = [_decision_row_dict(row) for row in result.get("dv_rows", ())]
        filename = (
            f"Disrupt_{int(window):02d}_{_safe_filename_token(location)}_"
            f"{_safe_filename_token(duration)}.csv")
        if filename in used_names:
            raise ValueError(
                "two scenarios map to the same output filename "
                f"{filename}: {used_names[filename]!r}, {sid!r}")
        used_names[filename] = sid
        path = directory / filename
        _write_csv_atomic(path, rows, DECISION_VARIABLE_FIELDS)
        per_scenario_files[sid] = str(path.resolve())
        row_count += len(rows)
    return per_scenario_files, None, row_count


def _write_scenario_csv_outputs(output: Path, scenario_rows, bus_rows,
                                scenarios, window: int,
                                scenario_fields, bus_fields):
    """Write summary and bus-operation CSVs separately for every scenario."""
    metadata = {str(scenario["id"]): scenario for scenario in scenarios}
    buses_by_scenario = {}
    for row in bus_rows:
        buses_by_scenario.setdefault(str(row["ScenarioID"]), []).append(row)
    scenario_files, bus_files, used_names = {}, {}, {}
    for row in scenario_rows:
        sid = str(row["ScenarioID"])
        scenario = metadata.get(sid, {})
        location = scenario.get(
            "block_station", scenario.get("station", sid))
        duration = scenario.get("duration", row.get("Duration", "NA"))
        token = (
            f"{int(window):02d}_{_safe_filename_token(location)}_"
            f"{_safe_filename_token(duration)}")
        if token in used_names:
            raise ValueError(
                f"two scenarios map to output token {token}: "
                f"{used_names[token]!r}, {sid!r}")
        used_names[token] = sid
        scenario_path = Path(output) / "ScenarioResults" / f"Scenario_{token}.csv"
        bus_path = Path(output) / "BusOperations" / f"BusOperations_{token}.csv"
        _write_csv_atomic(scenario_path, [row], scenario_fields)
        _write_csv_atomic(
            bus_path, buses_by_scenario.get(sid, ()), bus_fields)
        scenario_files[sid] = str(scenario_path.resolve())
        bus_files[sid] = str(bus_path.resolve())
    return scenario_files, bus_files


def _flat_row(row):
    return {
        key: (json.dumps(_jsonable(value), ensure_ascii=False)
              if isinstance(value, (dict, list, tuple, set, frozenset))
              else _json_number(value))
        for key, value in row.items()
    }


def _write_excel_atomic(path: Path, sheets: dict[str, list[dict]]) -> None:
    """Write compact reporting tables; very large decision rows stay in CSV."""
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.stem}.", suffix=".xlsx",
                delete=False) as handle:
            temporary_path = Path(handle.name)
        with pd.ExcelWriter(temporary_path, engine="openpyxl") as writer:
            for raw_name, rows in sheets.items():
                name = str(raw_name)[:31]
                frame = pd.DataFrame([_flat_row(row) for row in rows])
                frame.to_excel(writer, sheet_name=name, index=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_depot_allocation(path, depots, depot_counts):
    """Atomically persist a validated integer Stage-1 depot allocation."""
    integer_counts = {}
    for depot in depots:
        value = depot_counts.get(depot, 0)
        numeric = float(value)
        rounded = int(round(numeric))
        if (not math.isfinite(numeric) or numeric < -1e-6 or
                abs(numeric - rounded) > 1e-6):
            raise ValueError(
                f"non-integer final MILP depot count for {depot}: {value!r}")
        integer_counts[str(depot)] = max(rounded, 0)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", newline="", encoding="utf-8-sig",
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
                delete=False) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=["Depot", "Count"])
            writer.writeheader()
            writer.writerows(
                {"Depot": depot, "Count": integer_counts[str(depot)]}
                for depot in depots)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return integer_counts


def _json_number(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _terminal_reporting_view(scenario_results, scenarios):
    """Return only probability-bearing clustering terminals for publication."""
    terminals = [
        scenario for scenario in scenarios
        if bool(scenario.get("weighted_terminal", False))]
    terminal_ids = {str(scenario["id"]) for scenario in terminals}
    results = {
        str(sid): value for sid, value in scenario_results.items()
        if str(sid) in terminal_ids}
    return results, terminals


def _scenario_rows(scenario_results, scenarios, window):
    metadata = {str(scenario["id"]): scenario for scenario in scenarios}
    rows = []
    for sid, result in sorted(scenario_results.items()):
        scenario = metadata.get(str(sid), {})
        rows.append({
            "Window": int(window),
            "ScenarioID": str(sid),
            "Block": scenario.get("block_station", scenario.get("station")),
            "Station": scenario.get("station"),
            "Duration": scenario.get("duration"),
            "Probability": scenario.get("probability"),
            "WeightedTerminal": bool(
                scenario.get("weighted_terminal", False)),
            "Status": result.get("status"),
            "Objective": result.get("objective"),
            "BoardedAllVehicles": result.get("boarded"),
            "QueueArea": result.get("queue_area"),
            "OnboardArea": result.get("onboard_area"),
            "WalkingArea": result.get("walking_area"),
            "SelectedTrips": len(result.get("selected_trip_ids", ())),
        })
    return rows


def _assign_fleet_bus_ids(bus_rows, depot_counts):
    """Keep endogenous chains; color only legacy rows without chain IDs."""
    warnings = []
    selected_rows = [row for row in bus_rows if row.get("Selected", False)]
    if selected_rows and all(row.get("FleetBusID") for row in selected_rows):
        return warnings
    by_scenario_depot = {}
    for row in bus_rows:
        if not row.get("Selected", False):
            continue
        by_scenario_depot.setdefault(
            (str(row.get("ScenarioID")), str(row.get("Depot"))), []).append(row)
    for (sid, depot), rows in by_scenario_depot.items():
        count = int(round(float(depot_counts.get(depot, 0))))
        available = list(range(1, count + 1))
        heapq.heapify(available)
        busy = []
        for row in sorted(
                rows, key=lambda item: (
                    int(item.get("DispatchStart", 0)),
                    int(item.get("ServiceEnd", item.get("DispatchStart", 0))),
                    str(item.get("TripID")))):
            start = int(row.get("DispatchStart", 0))
            end = int(row.get("ServiceEnd", start))
            while busy and busy[0][0] <= start:
                _release, slot = heapq.heappop(busy)
                heapq.heappush(available, slot)
            if not available:
                row["FleetBusID"] = None
                row["FleetBusIDScope"] = "scenario_postprocess_failed"
                warnings.append(
                    f"{sid}/{depot}: full-interval coloring exceeds x={count} "
                    f"at trip {row.get('TripID')}")
                continue
            slot = heapq.heappop(available)
            heapq.heappush(busy, (end, slot))
            row["FleetBusID"] = f"{depot}_BUS_{slot:03d}"
            row["FleetBusIDScope"] = "scenario_reporting_only_legacy"
    return warnings


def _bus_rows(scenario_results, depot_counts, window):
    rows = []
    for _sid, result in sorted(scenario_results.items()):
        for operation in result.get("bus_operations", ()):
            row = {"Window": int(window), **dict(operation)}
            cell = tuple(row.get("InformationCell", ()))
            row["InformationCell"] = "-".join(map(str, cell))
            chain_cell = tuple(row.get("ChainInformationCell", ()))
            row["ChainInformationCell"] = "-".join(map(str, chain_cell))
            row["RouteStops"] = json.dumps(
                _jsonable(row.get("RouteStops", ())), ensure_ascii=False)
            row["StopTimeList"] = json.dumps(
                _jsonable(row.get("StopTimeList", ())), ensure_ascii=False)
            row["Legs"] = json.dumps(
                _jsonable(row.get("Legs", ())), ensure_ascii=False)
            rows.append(row)
    warnings = _assign_fleet_bus_ids(rows, depot_counts)
    return rows, warnings


def _performance_rows(result, final_stats=None):
    rows = []
    for history in result.get("history", ()):
        rows.append({"Phase": "RMP", **dict(history)})
    if final_stats:
        rows.append({
            "Phase": "FINAL_RESTRICTED_MILP",
            "iteration": None,
            "objective": final_stats.get("ObjVal"),
            "objective_bound": final_stats.get("ObjBound"),
            "mip_gap": final_stats.get("MIPGap"),
            "num_vars": final_stats.get("NumVars"),
            "num_constraints": final_stats.get("NumConstrs"),
            "num_binary": final_stats.get("NumBinVars"),
            "runtime": final_stats.get("Runtime"),
            "build_seconds": final_stats.get("BuildSec"),
            "status": final_stats.get("Status"),
            "vehicle_chain_enforced": final_stats.get(
                "VehicleChainEnforced"),
            "vehicle_chain_vars": final_stats.get("VehicleChainVars"),
            "vehicle_chain_constraints": final_stats.get(
                "VehicleChainConstrs"),
            "vehicle_chain_links": final_stats.get("VehicleChainLinks"),
            "return_to_home_depot": final_stats.get("ReturnToHomeDepot"),
            "return_within_service_horizon": final_stats.get(
                "ReturnWithinServiceHorizon"),
        })
    return rows


def assess_final_milp_completion(status, stats, allocation_file,
                                 target_mip_gap: float, *,
                                 require_allocation: bool = True,
                                 allow_time_limit_incumbent: bool = False) -> dict:
    """Enforce the production completion contract for a requested final MILP.

    ``require_allocation=False`` is the solver-only gate used before exposing
    a production ``DepotAlloc_XX.csv``.  The default, file-aware gate is then
    run after the accepted incumbent has been written atomically.
    """
    reasons = []
    stats = stats or {}
    status = str(status or "")
    termination = str(stats.get("TerminationReason", ""))
    time_limit_incumbent = bool(
        allow_time_limit_incumbent and status == "TIME_LIMIT" and
        termination == "TIME_LIMIT")
    if status not in ACCEPTED_FINAL_MILP_STATUSES and not time_limit_incumbent:
        reasons.append(f"status={status or 'MISSING'}")
    if (termination not in ACCEPTED_FINAL_MILP_STATUSES and
            not time_limit_incumbent):
        reasons.append(f"termination={termination or 'MISSING'}")
    if (not bool(stats.get("CertifiedOptimal", False)) and
            not time_limit_incumbent):
        reasons.append("CertifiedOptimal is not true")
    if int(stats.get("SolCount", 0) or 0) <= 0:
        reasons.append("no incumbent")
    if stats.get("EarlyStopTriggered") is not False:
        reasons.append("EarlyStopTriggered is not explicitly false")
    try:
        actual_gap = float(stats.get("MIPGap", math.nan))
    except (TypeError, ValueError):
        actual_gap = math.nan
    if (not math.isfinite(actual_gap) or
            (actual_gap > float(target_mip_gap) + FINAL_MIP_GAP_TOLERANCE and
             not time_limit_incumbent)):
        reasons.append(
            f"MIPGap={actual_gap!r} exceeds {float(target_mip_gap):.12g}")
    allocation_path = (
        None if not allocation_file else Path(str(allocation_file)))
    if (require_allocation and
            (allocation_path is None or not allocation_path.is_file())):
        reasons.append("allocation file is missing")
    allocation_exists = bool(
        allocation_path is not None and allocation_path.is_file())
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "allowed_statuses": sorted(ACCEPTED_FINAL_MILP_STATUSES),
        "status": status or None,
        "termination_reason": termination or None,
        "sol_count": int(stats.get("SolCount", 0) or 0),
        "mip_gap": actual_gap if math.isfinite(actual_gap) else None,
        "target_mip_gap": float(target_mip_gap),
        "accepted_by": (
            "TIME_LIMIT_INCUMBENT" if time_limit_incumbent else "MIP_GAP"),
        "time_limit_incumbent_allowed": bool(allow_time_limit_incumbent),
        "early_stop_triggered": stats.get("EarlyStopTriggered"),
        "allocation_required": bool(require_allocation),
        "allocation_file": (str(allocation_path.resolve())
                            if allocation_exists else None),
        "allocation_sha256": (
            _sha256_file(allocation_path) if allocation_exists else None),
    }


def build_cg_config(
        core, *, rmp_method=None, rmp_lp_warm_start=None,
        final_method=None,
        final_mip_focus=None, final_cuts=None, final_cut_passes=None,
        final_concurrent_mip=None,
        final_symmetry=None,
        final_heuristics=None, final_no_rel_heur_time=None,
        final_improve_start_time=None, final_best_obj_stop=None,
        independent_trips=False):
    """Copy every model choice needed by the standalone CG engine."""
    return cg.CGConfig(
        sigma=int(cg.SIGMA),          # Table 3: sigma_ij = 2 min (operations.walking_transfer_min)
        disrupt_buffer=int(core.DISRUPT_BUFFER),
        bus_capacity=float(core.BUS_CAPACITY),
        train_capacity=float(core.TRAIN_CAPACITY),
        min_trip_boarding=float(core.MIN_TRIP_BOARDING),
        max_total_buses=int(core.MAX_TOTAL_BUSES),
        mip_gap=float(core.MIP_GAP),
        time_limit=(None if core.TIME_LIMIT is None
                    else float(core.TIME_LIMIT)),
        output_flag=0,
        optimality_tolerance=float(os.getenv("TSB_CG_OPT_TOL")
                                   or cfg.get("solver.stage1.cg_opt_tol")),
        feasibility_tolerance=float(os.getenv("TSB_CG_FEAS_TOL")
                                    or cfg.get("solver.stage1.cg_feas_tol")),
        pricing_output_flag=0,
        rc_epsilon=float(cg.RC_EPS),
        ignore_names=bool(cfg.get("solver.stage1.ignore_names")),
        eliminate_w=True,
        eliminate_destination_states=True,
        compress_load_states=True,
        compress_queue_states=True,
        hybrid_bus_load_states=True,
        rmp_method=rmp_method,
        rmp_lp_warm_start=rmp_lp_warm_start,
        final_method=final_method,
        final_mip_focus=final_mip_focus,
        final_cuts=final_cuts,
        final_cut_passes=final_cut_passes,
        final_concurrent_mip=final_concurrent_mip,
        final_symmetry=final_symmetry,
        final_heuristics=final_heuristics,
        final_no_rel_heur_time=final_no_rel_heur_time,
        final_improve_start_time=final_improve_start_time,
        final_best_obj_stop=final_best_obj_stop,
        enforce_vehicle_chains=not independent_trips,
    )


def _run_impl(args, checkpoint):
    checkpoint({"phase": "LOADING_PREPROCESSOR"})
    route_file = _resolve_input_path(args.route_file)
    os.environ["TSB_DEPOT_ACCESS_BOUND"] = str(EXACT_DEPOT_ACCESS_BOUND_MIN)
    na = load_na_module()
    core = na.load_core()
    core.DURATIONS = list(_modeled_durations(args))
    # ``core`` is used only to prepare raw scenarios/caches.  The exact master
    # and pricing models below are implemented entirely inside ``cg_stage1``.
    # Retain Gurobi's unlimited default.  CG certification and the final MILP
    # are terminated only by their mathematical criteria (MIPGap for the MILP).
    core.TIME_LIMIT = None
    # The exact Stage-1 path uses one explicit post-disruption dispatch buffer.
    core.DISRUPT_BUFFER = int(args.buffer)
    for name in (
            "FILE_SCHEDULE", "FILE_OD", "FILE_DISTANCE", "DIR_NORMAL",
            "DIR_DISRUPT"):
        value = Path(str(getattr(core, name)))
        if not value.is_absolute():
            setattr(core, name, str((BASE / value).resolve()))
    core.FILE_PREDEFINED_ROUTES = str(route_file)

    depots, scenarios, wdata, bus_tt = prepare_window(
        core, na, args.window, args.buffer, args.k,
        (_resolve_input_path(args.representative_branches)
         if args.representative_branches else None))
    terminal_scenarios = [
        scenario for scenario in scenarios
        if bool(scenario.get("weighted_terminal", False))]
    support_scenarios = [
        scenario for scenario in scenarios
        if not bool(scenario.get("weighted_terminal", False))]
    terminal_duration_counts = {
        int(duration): sum(
            int(scenario["duration"]) == int(duration)
            for scenario in terminal_scenarios)
        for duration in core.DURATIONS
    }
    internal_duration_counts = {
        int(duration): sum(
            int(scenario["duration"]) == int(duration)
            for scenario in scenarios)
        for duration in core.DURATIONS
    }
    information_block_count = len({
        str(scenario["block_station"]) for scenario in scenarios})
    checkpoint({
        "phase": "WINDOW_PREPARED",
        "terminal_scenario_count": len(terminal_scenarios),
        "internal_branch_count": len(scenarios),
        "support_only_branch_count": len(support_scenarios),
        "information_block_count": information_block_count,
        "terminal_duration_counts": terminal_duration_counts,
        "internal_duration_counts": internal_duration_counts,
        "reduction": scenarios[0].get(
            "reduction_source", "duration_specific_fixed_k"),
        "depot_count": len(depots),
        "k": int(args.k),
        "route_pool_provenance": problem_builder.ROUTE_POOL_PROVENANCE,
        "route_file": str(route_file),
        "route_timing": "raw-directed-cumulative-rounding",
        "depot_candidate_file": str(DEPOT_CANDIDATE_FILE),
        "disrupt_buffer_minutes": int(args.buffer),
        "depot_access_bound_minutes": EXACT_DEPOT_ACCESS_BOUND_MIN,
    })
    config = build_cg_config(
        core, rmp_method=args.rmp_method,
        rmp_lp_warm_start=args.rmp_lp_warm_start,
        final_method=args.final_method,
        final_mip_focus=args.final_mip_focus,
        final_cuts=args.final_cuts,
        final_symmetry=args.final_symmetry,
        final_heuristics=args.final_heuristics,
        independent_trips=args.independent_trips)
    metadata_problem = cg.CGProblem(
        depots=tuple(depots), scenarios=tuple(scenarios), full_caches={},
        wdata=wdata, travel_times=bus_tt)
    candidate_count = cg.count_macro_columns(metadata_problem, config)
    if args.audit_pool:
        audit = {
            "window": int(args.window),
            "k": int(args.k),
            "route_pool_provenance": problem_builder.ROUTE_POOL_PROVENANCE,
            "reduction": scenarios[0].get(
                "reduction_source", "duration_specific_fixed_k"),
            "terminal_duration_counts": terminal_duration_counts,
            "internal_duration_counts": internal_duration_counts,
            "terminal_scenario_count": len(terminal_scenarios),
            "internal_branch_count": len(scenarios),
            "support_only_branch_count": len(support_scenarios),
            "zero_impact_support_branch_count": sum(
                bool(scenario.get("zero_impact_branch", False))
                for scenario in support_scenarios),
            "information_block_count": information_block_count,
            "representative_scenarios": [
                {
                    "duration": int(scenario["duration"]),
                    "cluster": int(scenario.get("cluster_id", -1)),
                    "station": str(scenario["station"]),
                    "scenario_id": str(scenario["id"]),
                    "probability": float(scenario.get("probability", 0.0)),
                    "zero_impact_branch": bool(
                        scenario.get("zero_impact_branch", False)),
                }
                for scenario in sorted(
                    terminal_scenarios,
                    key=lambda item: (
                        int(item["duration"]),
                        int(item.get("cluster_id", -1)),
                        str(item["station"])))
            ],
            "modeled_duration_blocks": information_block_count,
            "modeled_probability": sum(
                float(s.get("probability", 0.0))
                for s in terminal_scenarios),
            "candidate_macro_columns": candidate_count,
            "candidate_depot_count": len(depots),
            "disrupt_buffer_minutes": int(args.buffer),
            "depot_access_bound_minutes": EXACT_DEPOT_ACCESS_BOUND_MIN,
            "pricing_mode": (
                "certified-dynamic-partial-pricing"
                if int(args.partial_pricing_batch_size) > 0 else
                "full-pool-every-iteration"),
            "partial_pricing_batch_size": int(
                args.partial_pricing_batch_size),
            "partial_pricing_reoptimize_after": int(
                args.partial_pricing_reoptimize_after),
            "pricing_workers": int(args.pricing_workers),
            "pricing_model_cache_size": int(
                args.pricing_model_cache_size),
            "pricing_model_cache_policy": args.pricing_model_cache_policy,
            "rotate_partial_pricing": bool(args.rotate_partial_pricing),
            "search_rmp_optimality_tolerance": (
                args.search_rmp_optimality_tol),
            "search_rc_epsilon": args.search_rc_epsilon,
            "rmp_method": args.rmp_method,
            "rmp_lp_warm_start": args.rmp_lp_warm_start,
            "final_method": args.final_method,
            "final_mip_focus": args.final_mip_focus,
            "final_cuts": args.final_cuts,
            "final_symmetry": args.final_symmetry,
            "final_heuristics": args.final_heuristics,
            "final_milp_rmp_hints": bool(args.final_milp_rmp_hints),
            "final_milp_previous_depot_start": bool(
                args.final_milp_previous_depot_start),
            "certified_box_screening": bool(args.box_screening),
            "bus_load_balance": "hybrid-aggregate-load-od-prefix-exact",
            "hybrid_bus_load_states": True,
            "depot_overlap_rows": (
                "distinct-clipped-interval-starts-exact"),
            "independent_trips": bool(args.independent_trips),
            "vehicle_resource_model": (
                "INDEPENDENT_TRIP_INTERVALS_NO_RETURN_DEADHEAD"
                if args.independent_trips else "ENDOGENOUS_TRIP_CHAINS"),
            "final_mip_gap_target": float(
                config.mip_gap if args.final_mip_gap is None
                else args.final_mip_gap),
            "terminal_empty_rows": (
                "one-nonnegative-sum-row-per-vehicle-exact"),
            "route_file": str(route_file),
            "depot_candidate_file": str(DEPOT_CANDIDATE_FILE),
            "representative_branch_file": (
                str(_resolve_input_path(args.representative_branches))
                if args.representative_branches else None),
        }
        checkpoint({"phase": "AUDIT_COMPLETE", "audit": audit})
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0
    if args.audit_cache:
        full_caches = build_full_caches(
            core, na, scenarios, wdata, progress_hook=checkpoint,
            config=config)
        audit_problem = cg.CGProblem(
            depots=tuple(depots), scenarios=tuple(scenarios),
            full_caches=full_caches, wdata=wdata, travel_times=bus_tt)
        pool = cg.build_column_pool(audit_problem, config)
        lazy_bus_load_keys = sum(
            len(group["times"])
            for cache in full_caches.values()
            for by_omega in cg.aggregate_bus_load_groups(cache).values()
            for group in by_omega.values()
        )
        family_counts = {}
        for family in (
                "b_var_keys", "a_var_keys", "l_var_keys", "q_var_keys",
                "w_var_keys", "od_var_keys"):
            reference_count_key = {
                "l_var_keys": "l_reference_key_count",
                "q_var_keys": "q_reference_key_count",
                "w_var_keys": "w_reference_key_count",
            }.get(family)
            family_counts[family] = sum(
                int(cache.get(reference_count_key, len(cache.get(family, ()))))
                if reference_count_key is not None else
                len(cache.get(family, ()))
                for cache in full_caches.values())
        compact_q_count = sum(
            int(cache["_queue_compression"].model_key_count)
            for cache in full_caches.values()
            if "_queue_compression" in cache)
        compact_train_l_count = sum(
            len(cache.get("_train_load_segments", ()))
            for cache in full_caches.values())
        audit = {
            "window": int(args.window),
            "k": int(args.k),
            "route_pool_provenance": problem_builder.ROUTE_POOL_PROVENANCE,
            "disrupt_buffer_minutes": int(args.buffer),
            "depot_access_bound_minutes": EXACT_DEPOT_ACCESS_BOUND_MIN,
            "candidate_macro_columns": len(pool),
            "candidate_depot_count": len(depots),
            "route_file": str(route_file),
            "depot_candidate_file": str(DEPOT_CANDIDATE_FILE),
            "materialized_bus_l_keys": sum(
                str(key[1]).startswith("B_")
                for cache in full_caches.values()
                for key in cache["l_var_keys"]),
            "lazy_bus_l_keys_avoided": lazy_bus_load_keys,
            "fixed_support_key_counts": family_counts,
            "exact_reformulation": {
                "queue_reference_keys": family_counts["q_var_keys"],
                "queue_model_keys": compact_q_count,
                "train_load_reference_keys": family_counts["l_var_keys"],
                "train_load_model_keys": compact_train_l_count,
                "walking_reference_keys": family_counts["w_var_keys"],
                "walking_model_keys": 0,
                "queue_reference_lists_released": True,
                "walking_reference_lists_released": True,
            },
            "estimated_full_pool_flow_variables": (
                sum(family_counts.values()) + lazy_bus_load_keys),
            "bus_b_keys_retained": sum(
                str(key[1]).startswith("B_")
                for cache in full_caches.values()
                for key in cache["b_var_keys"]),
            "bus_a_keys_retained": sum(
                str(key[1]).startswith("B_")
                for cache in full_caches.values()
                for key in cache["a_var_keys"]),
            "train_l_keys_retained": sum(
                int(cache.get(
                    "l_reference_key_count",
                    sum(not str(key[1]).startswith("B_")
                        for key in cache["l_var_keys"])))
                for cache in full_caches.values()),
            "bus_load_balance": "hybrid-aggregate-load-od-prefix-exact",
            "hybrid_bus_load_states": True,
            "depot_overlap_rows": (
                "distinct-clipped-interval-starts-exact"),
            "independent_trips": bool(args.independent_trips),
            "vehicle_resource_model": (
                "INDEPENDENT_TRIP_INTERVALS_NO_RETURN_DEADHEAD"
                if args.independent_trips else "ENDOGENOUS_TRIP_CHAINS"),
            "final_mip_gap_target": float(
                config.mip_gap if args.final_mip_gap is None
                else args.final_mip_gap),
            "terminal_empty_rows": (
                "one-nonnegative-sum-row-per-vehicle-exact"),
        }
        checkpoint({"phase": "AUDIT_COMPLETE", "audit": audit})
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0
    if (candidate_count > UNACCELERATED_POOL_LIMIT and
            int(args.partial_pricing_batch_size) == 0 and
            not args.allow_unaccelerated_large_run):
        warning = (
            f"pricing pool has {candidate_count:,} macro-columns; certified "
            "box screening is enabled and exact LPs are solved only for "
            "survivors")
        print(f"  [warning] {warning}")
        checkpoint({"phase": "LARGE_POOL_WARNING", "warning": warning})
    full_caches = build_full_caches(
        core, na, scenarios, wdata, progress_hook=checkpoint,
        config=config)
    prepared_problem = cg.CGProblem(
        depots=tuple(depots), scenarios=tuple(scenarios),
        full_caches=full_caches, wdata=wdata, travel_times=bus_tt)
    pool = cg.build_column_pool(prepared_problem, config)
    if len(pool) != candidate_count:
        raise AssertionError(
            f"metadata/cache pool count mismatch: {candidate_count} != {len(pool)}")
    checkpoint({
        "phase": "FULL_POOL_READY",
        "candidate_macro_columns": len(pool),
        "cache_scenarios": len(full_caches),
        "exact_state_support": {
            "train_load_reference_keys": sum(
                int(cache.get("l_reference_key_count", 0))
                for cache in full_caches.values()),
            "train_load_model_keys": sum(
                len(cache.get("_train_load_segments", ()))
                for cache in full_caches.values()),
            "queue_reference_keys": sum(
                int(cache.get("q_reference_key_count", 0))
                for cache in full_caches.values()),
            "queue_model_keys": sum(
                int(cache["_queue_compression"].model_key_count)
                for cache in full_caches.values()),
            "destination_queue_keys_eliminated": sum(
                int(cache.get("destination_q_eliminated_count", 0))
                for cache in full_caches.values()),
            "od_exit_keys_eliminated": sum(
                int(cache.get("od_eliminated_count", 0))
                for cache in full_caches.values()),
            "walking_reference_keys": sum(
                int(cache.get("w_reference_key_count", 0))
                for cache in full_caches.values()),
            "walking_model_keys": 0,
            "reference_key_lists_released": True,
        },
    })

    output = Path(args.output)
    performance = output / "Performance"
    performance.mkdir(parents=True, exist_ok=True)

    env = core.gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", int(args.threads))
    env.start()
    try:
        problem = cg.CGProblem(
            depots=tuple(depots), scenarios=tuple(scenarios),
            full_caches=full_caches, wdata=wdata, env=env,
            travel_times=bus_tt)
        if args.resume_certified_cg:
            resume_path = _resolve_input_path(args.resume_certified_cg)
            result = _load_certified_cg_checkpoint(
                resume_path, args=args, problem=problem, config=config,
                pool=pool)
            checkpoint({
                "phase": "ROOT_CG_RESUMED",
                "active_columns": len(result["active_columns"]),
                "active_columns_file": str(resume_path),
                "active_columns_sha256": _sha256_file(resume_path),
            })
        else:
            result = cg.run_column_generation(
                problem, config,
                pool=pool,
                seed_per_block=args.seed_per_block,
                add_per_iteration=args.add_per_iteration,
                partial_pricing_batch_size=(
                    int(args.partial_pricing_batch_size)
                    if int(args.partial_pricing_batch_size) > 0 else None),
                partial_pricing_reoptimize_after=(
                    int(args.partial_pricing_reoptimize_after)
                    if int(args.partial_pricing_reoptimize_after) > 0 else None),
                pricing_workers=int(args.pricing_workers),
                pricing_model_cache_size=int(args.pricing_model_cache_size),
                pricing_model_cache_policy=args.pricing_model_cache_policy,
                search_optimality_tolerance=args.search_rmp_optimality_tol,
                search_rc_epsilon=args.search_rc_epsilon,
                rotate_partial_pricing=bool(args.rotate_partial_pricing),
                max_iterations=args.max_iterations,
                use_box_screening=bool(args.box_screening),
                persistent_rmp=not args.legacy_rebuild_rmp,
                validate_persistent_final=args.validate_persistent_final,
                retain_final_diagnostics=False,
                stall_seconds=(args.cg_stall_minutes * 60.0
                               if args.cg_stall_minutes is not None else None),
                progress_hook=checkpoint,
            )

        active_columns_path = (
            performance / f"ExactCG_{int(args.window):02d}_active_columns.json")
        active_columns_payload = {
            "schema": "stage1-exact-cg-active-columns-v1",
            "window": int(args.window),
            "config_hash": args._run_signature["config_sha256"],
            "root_cg_config_hash": _root_cg_config_hash(args),
            "input_hash": args._run_signature["input_sha256"],
            "implementation_hash": args._run_signature[
                "implementation_sha256"],
            "pool_hash": result["pool_hash"],
            "certified": bool(result["certified"]),
            "termination_reason": result["termination_reason"],
            "columns": [
                {
                    "block": key[0],
                    "trip_id": key[1],
                    "information_cell": list(key[2]),
                }
                for key in sorted(result["active_columns"])
            ],
            "rmp_primal_hint": {
                "x": dict(result.get(
                    "final_rmp_primal_hint", {}).get("x", {})),
                "y": [
                    {
                        "block": key[0],
                        "trip_id": key[1],
                        "information_cell": list(key[2]),
                        "value": float(value),
                    }
                    for key, value in sorted(result.get(
                        "final_rmp_primal_hint", {}).get("y", {}).items())
                ],
            },
            "cg_result": {
                key: value for key, value in result.items()
                if key not in {
                    "pool", "active_columns", "final_rmp_primal_hint",
                    "final_duals", "final_omitted_prices", "depot_row_keys",
                }
            },
        }
        _write_json_atomic(active_columns_path, active_columns_payload)
        active_columns_sha256 = _sha256_file(active_columns_path)
        checkpoint({
            "phase": "ROOT_CG_COLUMNS_SAVED",
            "active_columns": len(result["active_columns"]),
            "active_columns_file": str(active_columns_path),
            "active_columns_sha256": active_columns_sha256,
        })

        summary = {
            "window": int(args.window),
            "k": int(args.k),
            "representatives_per_duration": int(args.k),
            "route_pool_provenance": problem_builder.ROUTE_POOL_PROVENANCE,
            "route_file": str(route_file),
            "route_file_sha256": args._run_signature[
                "inputs"]["route_library"]["file_sha256"],
            "config_hash": args._run_signature["config_sha256"],
            "input_hash": args._run_signature["input_sha256"],
            "implementation_hash": args._run_signature[
                "implementation_sha256"],
            "run_signature": args._run_signature["run_sha256"],
            "engine_path": args._run_signature["engine"]["path"],
            "engine_hash": args._run_signature["engine"]["sha256"],
            "depot_candidate_file": str(DEPOT_CANDIDATE_FILE),
            "candidate_depot_count": len(depots),
            "disrupt_buffer_minutes": int(args.buffer),
            "depot_access_bound_minutes": EXACT_DEPOT_ACCESS_BOUND_MIN,
            "terminal_scenario_count": len(terminal_scenarios),
            "internal_branch_count": len(scenarios),
            "support_only_branch_count": len(support_scenarios),
            "terminal_duration_counts": terminal_duration_counts,
            "internal_duration_counts": internal_duration_counts,
            "weighted_terminal_probability": math.fsum(
                float(scenario.get("probability", 0.0))
                for scenario in terminal_scenarios),
            "candidate_macro_columns": len(pool),
            "generated_macro_columns": len(result["active_columns"]),
            "active_columns_file": str(active_columns_path),
            "active_columns_sha256": active_columns_sha256,
            "pool_complete": bool(result["pool_complete"]),
            "pool_hash": result["pool_hash"],
            "bus_load_balance": "hybrid-aggregate-load-od-prefix-exact",
            "hybrid_bus_load_states": True,
            "depot_overlap_rows": (
                "distinct-clipped-interval-starts-exact"),
            "independent_trips": bool(args.independent_trips),
            "vehicle_resource_model": (
                "INDEPENDENT_TRIP_INTERVALS_NO_RETURN_DEADHEAD"
                if args.independent_trips else "ENDOGENOUS_TRIP_CHAINS"),
            "final_mip_gap_target": float(
                config.mip_gap if args.final_mip_gap is None
                else args.final_mip_gap),
            "terminal_empty_rows": (
                "one-nonnegative-sum-row-per-vehicle-exact"),
            "iterations": len(result["history"]),
            "persistent_RMP": bool(result.get("persistent_rmp")),
            "persistent_validation_passed": result.get(
                "persistent_validation_passed"),
            "box_screening": bool(result.get("box_screening")),
            "seed_per_block": int(args.seed_per_block),
            "add_per_iteration": int(args.add_per_iteration),
            "partial_pricing_batch_size": int(
                args.partial_pricing_batch_size),
            "partial_pricing_reoptimize_after": int(
                args.partial_pricing_reoptimize_after),
            "pricing_workers": int(args.pricing_workers),
            "pricing_model_cache_size": int(
                args.pricing_model_cache_size),
            "pricing_model_cache_policy": args.pricing_model_cache_policy,
            "pricing_model_cache_stats": result.get(
                "pricing_model_cache_stats"),
            "rotate_partial_pricing": bool(args.rotate_partial_pricing),
            "search_rmp_optimality_tolerance": (
                args.search_rmp_optimality_tol),
            "search_rc_epsilon": args.search_rc_epsilon,
            "tight_certification_started": bool(
                result.get("tight_certification_started")),
            "rmp_method": args.rmp_method,
            "rmp_lp_warm_start": args.rmp_lp_warm_start,
            "gurobi_time_limit": (
                "unlimited" if args.final_milp_time_limit_seconds is None else
                float(args.final_milp_time_limit_seconds)),
            "final_milp_time_limit_seconds": (
                args.final_milp_time_limit_seconds),
            "cg_stall_minutes": args.cg_stall_minutes,
            "final_milp_stall_minutes": args.final_milp_stall_minutes,
            "certified": bool(result["certified"]),
            "certified_LB": _json_number(result["lb"]),
            "final_RMP_LP_value": _json_number(result["rmp_lp_value"]),
            "pricing_bound_correction": _json_number(
                result["pricing_bound_correction"]),
            "maximum_unresolved_LP_improvement": _json_number(
                result["maximum_unresolved_lp_improvement"]),
            "termination_reason": result["termination_reason"],
            "interrupted": bool(result.get("interrupted")),
            "initial_master_objective": _json_number(
                result.get("initial_rmp_objective")),
            "final_master_objective": _json_number(
                result.get("last_rmp_objective")),
            "master_improvement_absolute": _json_number(
                result.get("master_objective_improvement_absolute")),
            "master_improvement_relative": _json_number(
                result.get("master_objective_improvement_relative")),
            "master_objective_improved": bool(
                result.get("master_objective_improved")),
            "last_RMP_stats": result.get("last_rmp_stats"),
            "final_minimum_omitted_certificate_value": _json_number(
                result["final_minimum_omitted_certificate_value"]),
            "final_pricing_complete": bool(
                result.get("final_pricing_complete")),
            "final_priced_columns": int(
                result.get("final_priced_columns", 0)),
            "final_omitted_columns": int(
                result.get("final_omitted_columns", 0)),
            "final_box_screened_columns": int(
                result["final_box_screened_columns"]),
            "restricted_MILP_status": None,
            "UB": None,
            "relative_gap": None,
            "depot_counts": None,
            "allocation_file": None,
            "allocation_sha256": None,
            "partial_allocation_file": None,
            "partial_allocation_sha256": None,
            "total_fixed_buses": None,
            "final_MILP_stats": None,
            "final_MILP_solver_acceptance": None,
            "final_MILP_acceptance": None,
            "final_MILP_accepted": None,
            "final_active_macro_columns": None,
            "final_milp_column_start": None,
            "final_milp_column_start_sha256": None,
            "final_milp_column_start_selected": 0,
            "final_milp_column_start_added_columns": 0,
            "production_accepted": False,
            "scenario_results_file": None,
            "scenario_results_files": {},
            "bus_operations_file": None,
            "bus_operations_files": {},
            "decision_variables_files": {},
            "decision_variables_combined_file": None,
            "decision_variable_rows": 0,
            "performance_file": None,
            "workbook_file": None,
        }

        final_stats = None
        scenario_results = {}
        integer_counts = {}
        status = None
        if args.final_milp:
            if not result["certified"]:
                raise RuntimeError(
                    "final MILP was requested but the LP lower bound is not certified")
            checkpoint({"phase": "SOLVING_FINAL_RESTRICTED_MILP"})
            final_config = replace(
                config,
                time_limit=args.final_milp_time_limit_seconds,
                mip_gap=(config.mip_gap if args.final_mip_gap is None
                         else args.final_mip_gap))
            depot_start_path = _resolved_depot_start_path(args)
            depot_start = _load_depot_start(
                depot_start_path, depots, final_config.max_total_buses)
            summary["final_milp_depot_start_file"] = (
                str(depot_start_path) if depot_start else None)
            summary["final_milp_depot_start_sha256"] = (
                _sha256_file(depot_start_path) if depot_start else None)
            column_start_path = _resolved_column_start_path(args)
            column_start = _load_column_start(column_start_path)
            pool_keys = {column.key for column in pool}
            unknown_start = column_start - pool_keys
            if unknown_start:
                raise ValueError(
                    "column start contains variables outside the canonical "
                    f"pool: {sorted(unknown_start)[:3]}")
            final_active_columns = (
                set(result["active_columns"]) | set(column_start))
            summary["final_active_macro_columns"] = len(final_active_columns)
            summary["final_milp_column_start"] = (
                str(column_start_path) if column_start else None)
            summary["final_milp_column_start_sha256"] = (
                _hash_file_group(_column_start_files(
                    column_start_path))["sha256"]
                if column_start else None)
            summary["final_milp_column_start_selected"] = len(column_start)
            summary["final_milp_column_start_added_columns"] = (
                len(final_active_columns) - len(result["active_columns"]))
            depot_counts, status, scenario_results, stats = cg.solve_restricted_milp(
                problem, final_config, final_active_columns,
                result["depot_row_keys"],
                primal_hint=(
                    result.get("final_rmp_primal_hint")
                    if args.final_milp_rmp_hints else None),
                depot_start=depot_start,
                column_start=(column_start if column_start else None),
                stall_seconds=(args.final_milp_stall_minutes * 60.0
                               if args.final_milp_stall_minutes is not None
                               else None),
                progress_hook=checkpoint)
            final_stats = stats
            ub = float(stats["ObjVal"]) if stats and stats.get("SolCount", 0) else None
            lb = float(result["lb"])
            if ub is not None and ub + 1e-6 < lb:
                raise AssertionError(f"invalid bound order: UB={ub} < LB={lb}")
            summary["restricted_MILP_status"] = status
            summary["final_MILP_stats"] = stats
            summary["UB"] = ub
            if ub is not None and lb > 0:
                summary["relative_gap"] = (ub - lb) / lb
            has_incumbent = bool(
                stats and int(stats.get("SolCount", 0)) > 0 and
                status in {"OPTIMAL", "SUBOPTIMAL", "TIME_LIMIT", "INTERRUPTED"})
            solver_acceptance = assess_final_milp_completion(
                status, final_stats, None, final_config.mip_gap,
                require_allocation=False,
                allow_time_limit_incumbent=(
                    args.final_milp_time_limit_seconds is not None))
            summary["final_MILP_solver_acceptance"] = solver_acceptance
            if solver_acceptance["accepted"]:
                allocation_path = (output / "Allocation" /
                                   f"DepotAlloc_{int(args.window):02d}.csv")
                integer_counts = _write_depot_allocation(
                    allocation_path, depots, depot_counts)
                summary["depot_counts"] = integer_counts
                summary["allocation_file"] = str(allocation_path.resolve())
                summary["allocation_sha256"] = _sha256_file(allocation_path)
                summary["total_fixed_buses"] = sum(integer_counts.values())
            elif has_incumbent:
                # Preserve a diagnostically useful incumbent without exposing
                # it under the production filename consumed by Stage 2.
                partial_path = (output / "Allocation" /
                                f"DepotAlloc_{int(args.window):02d}_PARTIAL.csv")
                integer_counts = _write_depot_allocation(
                    partial_path, depots, depot_counts)
                summary["partial_allocation_file"] = str(partial_path.resolve())
                summary["partial_allocation_sha256"] = _sha256_file(partial_path)

        if args.final_milp:
            final_acceptance = assess_final_milp_completion(
                status, final_stats, summary.get("allocation_file"),
                final_config.mip_gap,
                allow_time_limit_incumbent=(
                    args.final_milp_time_limit_seconds is not None))
            summary["final_MILP_acceptance"] = final_acceptance
            summary["final_MILP_accepted"] = bool(
                final_acceptance["accepted"])
            if (summary["final_MILP_accepted"] and
                    final_acceptance["allocation_sha256"] !=
                    summary["allocation_sha256"]):
                raise AssertionError(
                    "final allocation changed between write and acceptance gate")
        summary["production_accepted"] = bool(
            result["certified"] and
            (not args.final_milp or summary["final_MILP_accepted"] is True))

        stem = f"ExactCG_{args.window:02d}"
        iterations_path = performance / f"{stem}_iterations.csv"
        performance_path = performance / f"{stem}_model_performance.csv"
        workbook_path = output / f"{stem}_results.xlsx"
        summary_path = performance / f"{stem}_summary.json"

        _write_history(iterations_path, result["history"])
        reporting_results, reporting_scenarios = _terminal_reporting_view(
            scenario_results, scenarios)
        if len(reporting_scenarios) != len(terminal_scenarios):
            raise AssertionError("weighted-terminal reporting view changed")
        scenario_rows = _scenario_rows(
            reporting_results, reporting_scenarios, args.window)
        bus_rows, bus_id_warnings = _bus_rows(
            reporting_results,
            integer_counts or depot_counts if args.final_milp else {},
            args.window)
        scenario_fields = [
            "Window", "ScenarioID", "Block", "Station", "Duration",
            "Probability", "WeightedTerminal", "Status", "Objective",
            "BoardedAllVehicles",
            "QueueArea", "OnboardArea", "WalkingArea", "SelectedTrips",
        ]
        bus_fields = [
            "Window", "ScenarioID", "Duration", "Block", "InformationCell",
            "FleetBusID", "FleetBusIDScope", "BusID", "TripID", "Y",
            "Selected", "AssignedDepot", "Depot", "DispatchStart", "RunStart",
            "RunEnd", "ServiceEnd", "ResourceEnd", "RouteKey", "Route",
            "Direction", "RouteDir", "RouteType", "StartStation", "EndStation",
            "ChainPosition", "ChainLength", "PreviousTripID", "NextTripID",
            "IsFirstTrip", "IsLastTrip", "ChainDecisionTime",
            "ChainInformationCell", "DeadheadBeforeFrom",
            "DeadheadBeforeTo", "DeadheadBeforeStart", "DeadheadBeforeEnd",
            "DeadheadBeforeMinutes", "WaitBeforeRun", "ReturnDeadhead",
            "ReturnStart", "ReturnEnd", "VehicleDutyStart", "VehicleDutyEnd",
            "RouteStops", "StopTimeList", "Legs", "BoardedPassengers",
            "MaxOnboard", "PeakLoadMinute", "ActualLastPassengerMinute",
        ]
        scenario_files, bus_files = _write_scenario_csv_outputs(
            output, scenario_rows, bus_rows, reporting_scenarios, args.window,
            scenario_fields, bus_fields)
        decision_files, decision_combined, decision_count = (
            _write_decision_variable_outputs(
                output, reporting_results, reporting_scenarios, args.window))
        performance_rows = _performance_rows(result, final_stats)
        performance_fields = list(dict.fromkeys(
            key for row in performance_rows for key in row))
        _write_csv_atomic(
            performance_path, performance_rows,
            performance_fields or ["Phase"])

        summary.update({
            "bus_id_assignment_warnings": bus_id_warnings,
            "iterations_file": str(iterations_path.resolve()),
            "scenario_results_file": None,
            "scenario_results_files": scenario_files,
            "bus_operations_file": None,
            "bus_operations_files": bus_files,
            "decision_variables_files": decision_files,
            "decision_variables_combined_file": decision_combined,
            "decision_variable_rows": decision_count,
            "performance_file": str(performance_path.resolve()),
            "workbook_file": str(workbook_path.resolve()),
            "summary_file": str(summary_path.resolve()),
        })
        workbook_error = None
        try:
            _write_excel_atomic(workbook_path, {
                "Summary": [summary],
                "Performance": performance_rows,
                "Scenarios": scenario_rows,
                "BusOperations": bus_rows,
            })
        except Exception as exc:  # CSV/JSON remain the durable primary output.
            workbook_error = f"{type(exc).__name__}: {exc}"
            summary["workbook_file"] = None
            summary["workbook_error"] = workbook_error
        _write_json_atomic(summary_path, summary)
        completion_phase = (
            "COMPLETED" if summary["production_accepted"] else
            ("FINAL_MILP_REJECTED" if args.final_milp and result["certified"]
             else "STOPPED_UNCERTIFIED"))
        checkpoint({
            "phase": completion_phase,
            "summary": summary,
        })
        print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))
        if result.get("interrupted"):
            return 130
        if summary["production_accepted"]:
            return 0
        return 3 if args.final_milp and result["certified"] else 2
    finally:
        env.dispose()


def run(args):
    """Run one window with an atomic, interruption-safe progress journal."""
    output = Path(args.output).expanduser().resolve()
    performance = output / "Performance"
    performance.mkdir(parents=True, exist_ok=True)
    stem = f"ExactCG_{int(args.window):02d}"
    checkpoint_path = performance / f"{stem}_checkpoint.json"
    iterations_path = performance / f"{stem}_iterations.csv"
    wall_started = time.time()
    state = {
        "phase": "STARTING",
        "window": int(args.window),
        "disrupt_buffer_minutes": int(args.buffer),
        "started_epoch": wall_started,
        "output": str(output),
        "route_pool_provenance": problem_builder.ROUTE_POOL_PROVENANCE,
        "route_file": str(_resolve_input_path(args.route_file)),
        "seed_per_block": int(args.seed_per_block),
        "add_per_iteration": int(args.add_per_iteration),
        "partial_pricing_batch_size": int(args.partial_pricing_batch_size),
        "partial_pricing_reoptimize_after": int(
            args.partial_pricing_reoptimize_after),
        "pricing_workers": int(args.pricing_workers),
        "pricing_model_cache_size": int(args.pricing_model_cache_size),
        "pricing_model_cache_policy": args.pricing_model_cache_policy,
        "rotate_partial_pricing": bool(args.rotate_partial_pricing),
        "search_rmp_optimality_tolerance": args.search_rmp_optimality_tol,
        "search_rc_epsilon": args.search_rc_epsilon,
        "hybrid_bus_load_states": True,
        "rmp_method": args.rmp_method,
        "rmp_lp_warm_start": args.rmp_lp_warm_start,
        "box_screening": bool(args.box_screening),
        "persistent_rmp": not args.legacy_rebuild_rmp,
        "cg_stall_minutes": args.cg_stall_minutes,
        "final_milp_stall_minutes": args.final_milp_stall_minutes,
        "final_milp_time_limit_seconds": args.final_milp_time_limit_seconds,
    }

    def checkpoint(update):
        payload = dict(update or {})
        history = payload.get("history")
        state.update(payload)
        state["updated_epoch"] = time.time()
        state["elapsed_seconds"] = state["updated_epoch"] - wall_started
        if history is not None:
            _write_history(iterations_path, history)
            state["iterations_file"] = str(iterations_path)
        _write_json_atomic(checkpoint_path, state)

    checkpoint({"phase": "STARTING"})
    previous_handlers = {}

    def interrupted_by_signal(signum, _frame):
        checkpoint({"phase": "INTERRUPT_REQUESTED", "signal": int(signum)})
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupted_by_signal)
        except (AttributeError, OSError, ValueError):
            pass
    try:
        checkpoint({"phase": "HASHING_RUN_INPUTS"})
        args._run_signature = build_run_signature(args)
        checkpoint({
            "phase": "RUN_SIGNATURE_READY",
            "config_hash": args._run_signature["config_sha256"],
            "input_hash": args._run_signature["input_sha256"],
            "implementation_hash": args._run_signature[
                "implementation_sha256"],
            "run_signature": args._run_signature["run_sha256"],
            "engine_path": args._run_signature["engine"]["path"],
            "engine_hash": args._run_signature["engine"]["sha256"],
            "run_manifest": args._run_signature,
        })
        exit_code = _run_impl(args, checkpoint)
        checkpoint({"exit_code": int(exit_code)})
        return exit_code
    except KeyboardInterrupt:
        checkpoint({
            "phase": "INTERRUPTED",
            "exit_code": 130,
            "certified": False,
        })
        return 130
    except BaseException as exc:
        checkpoint({
            "phase": "FAILED",
            "exit_code": 1,
            "certified": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise
    finally:
        for signum, handler in previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except (OSError, ValueError):
                pass


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument(
        "--route-file", required=True,
        help=("explicit route-library CSV; relative paths are resolved from "
              "the runner directory"))
    parser.add_argument(
        "--k", type=int, default=problem_builder.DEFAULT_K_PER_DURATION,
                        help="weighted representative branches per duration")
    parser.add_argument(
        "--representative-branches",
        help=("explicit common-station representative branch CSV; when set, "
              "the internal duration-specific reducer is not used"))
    parser.add_argument(
        "--buffer", type=int, default=EXACT_DISRUPT_BUFFER,
        help="post-end dispatch buffer in minutes (default: 20)")
    parser.add_argument("--seed-per-block", type=int, default=1)
    parser.add_argument("--add-per-iteration", type=int, default=50)
    parser.add_argument(
        "--partial-pricing-batch-size", type=int, default=2000,
        help=("number of omitted candidates priced before an early RMP "
              "reoptimization; 0 prices the complete pool every iteration"))
    parser.add_argument(
        "--partial-pricing-reoptimize-after", type=int, default=0,
        help=("number of negative columns needed to stop a partial pricing "
              "pass; 0 uses --add-per-iteration (legacy behavior)"))
    parser.add_argument(
        "--pricing-workers", type=int, default=1,
        help=("independent one-thread Gurobi pricing workers; the RMP still "
              "uses --threads"))
    parser.add_argument(
        "--pricing-model-cache-size", type=int, default=0,
        help=("maximum retained candidate pricing LPs across all workers; "
              "0 rebuilds every pricing LP"))
    parser.add_argument(
        "--pricing-model-cache-policy", choices=("stable", "lru"),
        default="stable",
        help=("stable preserves the deterministic hot prefix and avoids "
              "sequential-scan LRU thrashing"))
    parser.add_argument(
        "--rotate-partial-pricing", action="store_true",
        help=("prefer omitted candidates not recently priced; final full "
              "pricing certification remains unchanged"))
    parser.add_argument(
        "--search-rmp-optimality-tol", type=float, default=None,
        help=("optional loose OptimalityTol for intermediate RMP searches; "
              "a tight RMP re-solve and full pricing pass are forced before "
              "certification"))
    parser.add_argument(
        "--search-rc-epsilon", type=float, default=None,
        help=("negative reduced-cost margin during loose search; final "
              "certification always uses the engine's tight RC epsilon"))
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument(
        "--threads", type=int, default=0,
        help="Gurobi thread limit; 0 lets Gurobi choose automatically")
    parser.add_argument(
        "--rmp-method", type=int, choices=range(-1, 7), default=None,
        help=("optional Gurobi Method for LP masters only; omitted preserves "
              "the solver default (-1=automatic, 0=primal, 1=dual)"))
    parser.add_argument(
        "--rmp-lp-warm-start", type=int, choices=(0, 1, 2), default=None,
        help=("optional Gurobi LPWarmStart for LP masters only; omitted "
               "preserves the solver default"))
    parser.add_argument(
        "--final-method", type=int, choices=range(-1, 6), default=None,
        help=("optional Gurobi Method for the final MILP root relaxation; "
              "omitted preserves automatic selection, 2 selects barrier"))
    parser.add_argument(
        "--final-mip-focus", type=int, choices=(0, 1, 2, 3), default=None)
    parser.add_argument(
        "--final-cuts", type=int, choices=(-1, 0, 1, 2, 3), default=None)
    parser.add_argument(
        "--final-symmetry", type=int, choices=(-1, 0, 1, 2), default=None)
    parser.add_argument(
        "--final-heuristics", type=float, default=None)
    parser.add_argument(
        "--independent-trips", action="store_true",
        help=("remove inter-trip vehicle chains and enforce depot capacity on "
              "each complete one-way or round-trip service independently; "
              "return-to-depot deadhead is not added"))
    parser.add_argument(
        "--final-mip-gap", type=float, default=None,
        help="relative gap target for the final restricted MILP only")
    parser.add_argument("--final-milp-depot-start-file")
    parser.add_argument(
        "--final-milp-column-start",
        help=("BusOperations CSV or directory from a prior incumbent; all "
              "restricted y starts are set to 0/1"))
    hint_group = parser.add_mutually_exclusive_group()
    hint_group.add_argument(
        "--final-milp-rmp-hints", dest="final_milp_rmp_hints",
        action="store_true")
    hint_group.add_argument(
        "--no-final-milp-rmp-hints", dest="final_milp_rmp_hints",
        action="store_false")
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument(
        "--final-milp-previous-depot-start",
        dest="final_milp_previous_depot_start", action="store_true")
    start_group.add_argument(
        "--no-final-milp-previous-depot-start",
        dest="final_milp_previous_depot_start", action="store_false")
    parser.set_defaults(
        final_milp_rmp_hints=True,
        final_milp_previous_depot_start=False)
    parser.add_argument(
        "--resume-certified-cg",
        help=("hash-matched ExactCG active-columns JSON; skips root CG and "
              "rebuilds only the requested final MILP"))
    parser.add_argument("--final-milp", action="store_true")
    parser.add_argument("--audit-pool", action="store_true",
                        help="count real macro-columns without building flow caches")
    parser.add_argument("--audit-cache", action="store_true",
                        help="build lazy exact caches and report retained key counts")
    parser.add_argument("--allow-unaccelerated-large-run", action="store_true")
    screening = parser.add_mutually_exclusive_group()
    screening.add_argument(
        "--box-screening", dest="box_screening", action="store_true",
        help="use the certified box lower-bound screen (production default)")
    screening.add_argument(
        "--no-box-screening", dest="box_screening", action="store_false",
        help="solve every pricing LP (validation/debug only)")
    parser.set_defaults(box_screening=True)
    parser.add_argument("--legacy-rebuild-rmp", action="store_true",
                        help="rebuild each RMP iteration (validation fallback)")
    parser.add_argument("--validate-persistent-final", action="store_true",
                        help="compare the final persistent RMP with one legacy rebuild")
    parser.add_argument(
        "--cg-stall-minutes", type=float, default=None,
        help=("optional non-certified root-CG objective-stall cutoff; exact "
              "production runs leave this unset"))
    parser.add_argument(
        "--final-milp-stall-minutes", type=float, default=None,
        help=("optional final restricted-MILP incumbent-stall cutoff; "
              "production runs leave this unset and stop at MIPGap only"))
    parser.add_argument(
        "--final-milp-time-limit-seconds", type=float, default=None,
        help=("optional wall-clock limit for the final restricted MILP only; "
              "a time-limit incumbent is retained with its achieved gap"))
    parser.add_argument(
        "--print-run-signature", action="store_true",
        help="hash effective configuration and inputs, print JSON, and exit")
    parser.add_argument("--output", default=os.getenv(
        "TSB_EXACT_CG_OUTPUT_ROOT") or str(DEFAULT_OUTPUT))
    return parser


def _validate_args(parser, args):
    if args.buffer < 0:
        parser.error("--buffer must be nonnegative")
    if args.k <= 1:
        parser.error("--k must be at least 2")
    if args.add_per_iteration <= 0:
        parser.error("--add-per-iteration must be positive")
    if args.partial_pricing_batch_size < 0:
        parser.error("--partial-pricing-batch-size must be nonnegative")
    if args.partial_pricing_reoptimize_after < 0:
        parser.error(
            "--partial-pricing-reoptimize-after must be nonnegative")
    if (args.partial_pricing_reoptimize_after > 0 and
            args.partial_pricing_reoptimize_after > args.add_per_iteration):
        parser.error(
            "--partial-pricing-reoptimize-after cannot exceed "
            "--add-per-iteration")
    if args.pricing_workers <= 0:
        parser.error("--pricing-workers must be positive")
    if args.pricing_model_cache_size < 0:
        parser.error("--pricing-model-cache-size must be nonnegative")
    if (args.pricing_model_cache_size > 0 and
            args.pricing_model_cache_policy == "stable" and
            args.rotate_partial_pricing):
        parser.error(
            "stable pricing-model caching cannot be combined with "
            "--rotate-partial-pricing")
    if (args.search_rmp_optimality_tol is not None and
            args.search_rmp_optimality_tol <= 0):
        parser.error("--search-rmp-optimality-tol must be positive")
    if args.search_rc_epsilon is not None and args.search_rc_epsilon < 0:
        parser.error("--search-rc-epsilon must be nonnegative")
    if args.cg_stall_minutes is not None and args.cg_stall_minutes <= 0:
        parser.error("--cg-stall-minutes must be positive")
    if (args.final_milp_stall_minutes is not None and
            args.final_milp_stall_minutes <= 0):
        parser.error("--final-milp-stall-minutes must be positive")
    if (args.final_milp_time_limit_seconds is not None and
            args.final_milp_time_limit_seconds <= 0):
        parser.error("--final-milp-time-limit-seconds must be positive")
    if (args.final_mip_gap is not None and
            (not math.isfinite(args.final_mip_gap) or
             not 0.0 <= args.final_mip_gap < 1.0)):
        parser.error("--final-mip-gap must be finite and in [0, 1)")
    if args.resume_certified_cg and not args.final_milp:
        parser.error("--resume-certified-cg requires --final-milp")
    if args.max_iterations <= 0:
        parser.error("--max-iterations must be positive")
    if args.seed_per_block <= 0:
        parser.error("--seed-per-block must be positive")
    if args.threads < 0:
        parser.error("--threads must be nonnegative (0 selects Gurobi automatic)")
    if (args.final_heuristics is not None and
            not 0.0 <= args.final_heuristics <= 1.0):
        parser.error("--final-heuristics must be in [0, 1]")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    if args.print_run_signature:
        print(json.dumps(
            build_run_signature(args), ensure_ascii=False, indent=2))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
