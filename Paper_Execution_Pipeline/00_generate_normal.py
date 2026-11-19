"""Generate normal-condition commuter-flow outputs for the windows of the active profile."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import (
    CORE_DIR, NORMAL_DIR, cfg, ensure_layout, parse_windows,
    require_files, require_packages, run_window_processes,
)
from pipeline_config import WINDOWS


CORE = CORE_DIR / "normal_generation.py"
WINDOW_HOURS = int(cfg.get("time.horizon_min")) // 60


def normal_file(window: int) -> Path:
    return NORMAL_DIR / f"Normal_{window:02d}_{window + WINDOW_HOURS:02d}.csv"


def merge_statistics() -> Path:
    parts = sorted(NORMAL_DIR.glob("_normal_stats_w*.csv"))
    if not parts:
        raise FileNotFoundError("normal window statistics were not generated")
    frame = pd.concat((pd.read_csv(path) for path in parts), ignore_index=True)
    frame = frame.sort_values("window").drop_duplicates("window", keep="last")
    output = NORMAL_DIR / "_normal_stats.csv"
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def check(windows: tuple[int, ...]) -> None:
    ensure_layout()
    require_packages(("pandas", "psutil", "gurobipy"))
    require_files((
        CORE,
        cfg.path("inputs.timetable"),
        cfg.path("inputs.od_matrix"),
        cfg.path("inputs.walking_times"),
    ))
    parse_windows(",".join(str(window) for window in windows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parallel = cfg.get("parallel.normal")
    parser.add_argument("--windows", default=",".join(str(w) for w in WINDOWS),
                        help="comma list or ranges of window start hours (default: time.windows)")
    parser.add_argument("--workers", type=int, default=parallel["workers"])
    parser.add_argument("--threads", type=int, default=parallel["threads"])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    windows = parse_windows(args.windows)
    check(windows)
    if args.check_only:
        print(f"Normal preflight OK: {len(windows)} windows")
        return
    run_window_processes(
        script=CORE, windows=windows, workers=args.workers,
        threads=args.threads, log_prefix="00_normal",
        env_overrides={"PYTHONHASHSEED": str(cfg.get("scenarios.python_hash_seed"))},
    )
    stats = merge_statistics()
    missing = [normal_file(window) for window in windows
               if not normal_file(window).is_file()]
    if missing:
        raise RuntimeError(f"normal outputs missing after solve: {missing}")
    print(f"Normal generation complete: {stats}")


if __name__ == "__main__":
    main()
