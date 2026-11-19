"""Run the Stage 1 K scaling (Sec. 6.8, Table 8) for the configured windows.

Every run of ``evaluation.scaling.runs`` (window + K list) is solved with the
representatives written by ``prepare_scaling.py``.  Without
arguments all configured windows are processed; ``--window`` selects one.
Outputs: ``<results>/<paths.scaling>/h<WW>/S1_K<k>/`` plus
``scaling_status_h<WW>.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCRIPT = Path(__file__).resolve()
SOURCE = SCRIPT.parent
SENSITIVITY = SOURCE.parent
PIPELINE = SENSITIVITY.parent
sys.path.insert(0, str(PIPELINE.parent))
from bbconfig import cfg  # noqa: E402

PYTHON = Path(sys.executable).resolve()
WORKER = SOURCE / "run_stage1_full_pool_milp.py"
SCALING_ROOT = cfg.results_dir / cfg.get("paths.scaling")
DURATION_SOURCE = str(cfg.get("probability.duration_source"))
FLEET_SIZE = int(cfg.get("operations.fleet_size"))
MIN_TRIP_BOARDING = float(cfg.get("operations.min_trip_boarding"))
BUFFER = int(cfg.get("time.clearance_buffer_min"))
TIME_LIMIT = float(cfg.get("evaluation.solver.stage1_time_limit_s"))
TARGET_GAP = float(cfg.get("evaluation.solver.mip_gap"))
AUTO_PARAMETERS = bool(cfg.get("evaluation.solver.stage1_auto_parameters"))
WORKERS = int(cfg.get("parallel.scaling.workers"))
GAP_TOLERANCE = 1e-9
DURATION_TAG = "_".join(str(int(value)) for value in cfg.get("time.durations_min"))
RUNS = {
    int(run["window"]): tuple(int(k) for k in run["k_values"])
    for run in cfg.get("evaluation.scaling.runs")
}
_COUNTS = {
    str(state): int(count)
    for state, count in cfg.get("probability.duration_counts").items()
}
DURATION_WEIGHTS = {
    state: count / sum(_COUNTS.values()) for state, count in _COUNTS.items()
}
# Set per window in run_window().
WINDOW = 0
K_VALUES: tuple[int, ...] = ()
INPUT = Path()
OUTPUT = Path()
STATUS = Path()
THREADS = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window", type=int, choices=sorted(RUNS),
        help="one window of evaluation.scaling.runs (default: every window)",
    )
    parser.add_argument(
        "--threads", type=int, default=int(cfg.get("parallel.scaling.threads"))
    )
    parser.add_argument(
        "--k-values", nargs="+", type=int,
        help="override the K list of --window (default: evaluation.scaling.runs)",
    )
    parser.add_argument(
        "--input-root", type=Path, default=SCALING_ROOT / "inputs",
        help="folder with h<WW>/ representatives from the prepare step",
    )
    parser.add_argument(
        "--output-root", type=Path, default=SCALING_ROOT,
        help="solves are written to <output-root>/h<WW>/S1_K<k>",
    )
    parser.add_argument(
        "--route-file", type=Path,
        default=(cfg.results_dir / cfg.get("paths.routes")
                 / cfg.get("routes.library_file")),
        help="route library (Sec. 5.2: 14 local + 35 express routes)",
    )
    parser.add_argument(
        "--warm-start-dir", type=Path,
        default=cfg.results_dir / cfg.get("paths.stage1"),
        help="production Stage 1 output (depot allocation used as start)",
    )
    parser.add_argument(
        "--parameter-file",
        type=Path,
        help="optional Gurobi parameter file for large-K memory controls",
    )
    parser.add_argument(
        "--warm-start-hint-only",
        action=argparse.BooleanOptionalAction,
        default=bool(cfg.get("evaluation.scaling.warm_start_hint_only")),
        help="apply the compatible depot allocation as variable hints only",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the Stage 1 commands without solving or writing status",
    )
    args = parser.parse_args()
    if args.k_values is not None:
        if args.window is None:
            parser.error("--k-values requires --window")
        if any(value <= 0 for value in args.k_values):
            parser.error("--k-values must contain positive integers")
    return args


def write_status(value: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(STATUS)


def summary(k: int) -> dict:
    path = (
        OUTPUT / f"S1_K{k}" / "Performance" /
        f"FullMILP_{WINDOW:02d}_summary.json"
    )
    if not path.is_file():
        return {"k": k, "summary_exists": False}
    value = json.loads(path.read_text(encoding="utf-8"))
    stats = value.get("stats") or {}
    gap = float(stats.get("MIPGap", value.get("relative_gap", math.inf)))
    min_trip_boarding = float(value.get("min_trip_boarding", math.nan))
    return {
        "k": k,
        "summary_exists": True,
        "status": value.get("status"),
        "termination_reason": value.get("termination_reason"),
        "has_incumbent": value.get("has_incumbent"),
        "target_met": (
            math.isfinite(gap)
            and gap <= TARGET_GAP + GAP_TOLERANCE
            and min_trip_boarding == MIN_TRIP_BOARDING
        ),
        "gap": gap if math.isfinite(gap) else None,
        "gap_pct": 100.0 * gap if math.isfinite(gap) else None,
        "objective": value.get("objective"),
        "bound": value.get("bound"),
        "runtime_seconds": stats.get("Runtime"),
        "build_seconds": stats.get("BuildSec"),
        "variables": stats.get("NumVars"),
        "constraints": stats.get("NumConstrs"),
        "depot_total": value.get("depot_total"),
        "min_trip_boarding": min_trip_boarding,
        "allocation_file": value.get("allocation_file"),
        "summary_file": str(path.resolve()),
    }


def run_window(args: argparse.Namespace, window: int, k_values: tuple[int, ...]) -> int:
    global WINDOW, K_VALUES, INPUT, OUTPUT, STATUS, THREADS
    THREADS = args.threads
    WINDOW = int(window)
    K_VALUES = tuple(dict.fromkeys(int(value) for value in k_values))
    INPUT = args.input_root.resolve() / f"h{WINDOW:02d}"
    OUTPUT = args.output_root.resolve() / f"h{WINDOW:02d}"
    STATUS = OUTPUT / f"scaling_status_h{WINDOW:02d}.json"
    route = args.route_file.resolve()
    warm_start = args.warm_start_dir.resolve()
    audit_path = INPUT / (
        f"{DURATION_SOURCE}_kmeans_internal_h{WINDOW:02d}_scaling_audit.json"
    )
    required_paths = [PYTHON, WORKER, route, audit_path, warm_start]
    if args.parameter_file is not None:
        required_paths.append(args.parameter_file)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("baseline_k14_reproduction") != {
        "stored_internal_representatives": True,
        "stored_unique_representatives": True,
        "labels": True,
        "branch_probability": True,
    }:
        raise RuntimeError("frozen K reproduction gate failed")
    if int(audit.get("maximum_exact_k", 0)) < max(K_VALUES):
        raise RuntimeError(
            f"H{WINDOW:02d} supports at most K={audit.get('maximum_exact_k')}, "
            f"below requested K={max(K_VALUES)}"
        )

    state = {
        "schema": f"operational303-kmeans-internal-h{WINDOW:02d}-scaling-v1",
        "state": "RUNNING",
        "window": WINDOW,
        "k_values": list(K_VALUES),
        "method": "corridor-constrained unweighted K-means",
        "duration_probability_source": DURATION_SOURCE,
        "duration_weights": dict(DURATION_WEIGHTS),
        "time_limit_seconds": TIME_LIMIT,
        "target_gap": TARGET_GAP,
        "min_trip_boarding": MIN_TRIP_BOARDING,
        "completion_rule": (
            f"run until the {TARGET_GAP:.0%} gap is met "
            f"(time limit {TIME_LIMIT:g} s)"
        ),
        "workers": WORKERS,
        "threads": THREADS,
        "output_root": str(OUTPUT),
        "gurobi_parameters": (
            str(args.parameter_file.resolve())
            if args.parameter_file is not None
            else "automatic defaults"
        ),
        "warm_start": str(warm_start),
        "warm_start_mode": f"compatible H{WINDOW:02d} depot allocation only",
        "warm_start_application": (
            "VARIABLE_HINT_ONLY" if args.warm_start_hint_only
            else "MIP_START_AND_VARIABLE_HINT"
        ),
        "baseline_k14_reused": str(
            warm_start / "Performance" / f"FullMILP_{WINDOW:02d}_summary.json"
        ),
        "results": [],
    }
    if not args.dry_run:
        write_status(state)
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TSB_MAX_BUSES": str(FLEET_SIZE),
        "TSB_DEPOT_ACCESS_BOUND": str(cfg.get("operations.depot_access_bound_min")),
        # Core mode switches (code paths, not parameters).
        "TSB_ROUTE_SOURCE": "predefined",
        "TSB_BUS_BOARD": "general",
    })
    pending = []
    for k in K_VALUES:
        existing = summary(k)
        if existing.get("summary_exists") and existing.get("target_met"):
            existing["run_action"] = "SKIPPED_EXISTING_NO_RERUN"
            state["results"].append(existing)
        else:
            pending.append(k)
    if not args.dry_run:
        write_status(state)

    def command_for(k: int) -> list[str]:
        command = [
            str(PYTHON), "-X", "utf8", "-u", str(WORKER),
            "--case-id", f"OP303_KMEANS_INTERNAL_H{WINDOW:02d}_K{k}",
            "--window", str(WINDOW),
            "--route-file", str(route),
            "--representative-branches", str(INPUT / (
                f"selected_representative_branches_k{k}_"
                f"{DURATION_TAG}_H{WINDOW:02d}.csv")),
            "--output", str(OUTPUT / f"S1_K{k}"),
            "--k", str(k),
            "--fleet-size", str(FLEET_SIZE),
            "--buffer", str(BUFFER),
            "--min-trip-boarding", f"{MIN_TRIP_BOARDING:g}",
            "--time-limit-seconds", str(TIME_LIMIT),
            "--mip-gap", str(TARGET_GAP),
            "--threads", str(THREADS),
            "--duration-probability-source", DURATION_SOURCE,
            "--warm-start-dir", str(warm_start),
            "--warm-start-depot-only",
        ]
        if args.warm_start_hint_only:
            command.append("--warm-start-hint-only")
        if args.parameter_file is None:
            if AUTO_PARAMETERS:
                command.append("--auto-parameters")
        else:
            command.extend([
                "--parameter-file", str(args.parameter_file.resolve()),
            ])
        return command

    def solve(k: int) -> dict:
        output = OUTPUT / f"S1_K{k}"
        command = command_for(k)
        branch = Path(command[command.index("--representative-branches") + 1])
        if not branch.is_file():
            raise FileNotFoundError(branch)
        output.mkdir(parents=True, exist_ok=True)
        log = output / f"one_pass_h{WINDOW:02d}_stdout_stderr.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=OUTPUT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        result = summary(k)
        return {
            **result,
            "return_code": int(completed.returncode),
            "run_action": "RUN_COMPLETED",
            "log": str(log.resolve()),
        }

    if args.dry_run:
        for k in pending:
            print("COMMAND " + subprocess.list2cmdline(command_for(k)), flush=True)
        return 0

    with ThreadPoolExecutor(max_workers=min(WORKERS, len(pending) or 1)) as pool:
        futures = {pool.submit(solve, k): k for k in pending}
        for future in as_completed(futures):
            state["results"].append(future.result())
            state["results"] = sorted(state["results"], key=lambda row: row["k"])
            write_status(state)
    failures = [
        row for row in state["results"]
        if not row.get("summary_exists") or not row.get("target_met")
    ]
    state["state"] = "COMPLETE" if not failures else "COMPLETE_WITH_FAILURES"
    state["failures"] = len(failures)
    write_status(state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if not failures else 1



def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.window is not None:
        selected = {args.window: tuple(args.k_values or RUNS[args.window])}
    else:
        selected = dict(RUNS)
    status = 0
    for window, k_values in selected.items():
        status = max(status, run_window(args, window, k_values))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
