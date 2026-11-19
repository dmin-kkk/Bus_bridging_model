"""Generate disruption scenarios (time.durations_min) for the windows of the active profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import (
    CORE_DIR, DISRUPTION_DIR, LOG_DIR, NORMAL_DIR, cfg, ensure_layout,
    parse_windows, require_files, require_packages, run_window_processes,
)
from pipeline_config import DURATIONS, WINDOWS


CORE = CORE_DIR / "disruption_generation.py"
WINDOW_HOURS = int(cfg.get("time.horizon_min")) // 60


def normal_file(window: int) -> Path:
    return NORMAL_DIR / f"Normal_{window:02d}_{window + WINDOW_HOURS:02d}.csv"


def merge_statistics(output_dir: Path = DISRUPTION_DIR) -> Path:
    output_dir = Path(output_dir).resolve()
    parts = sorted(output_dir.glob("_scenario_stats_w*.csv"))
    if not parts:
        raise FileNotFoundError("disruption window statistics were not generated")
    frame = pd.concat((pd.read_csv(path) for path in parts), ignore_index=True)
    keys = [column for column in ("window", "target", "duration") if column in frame]
    if keys:
        frame = frame.drop_duplicates(keys, keep="last")
        frame = frame.sort_values(keys)
    output = output_dir / "_scenario_stats.csv"
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def check(
    windows: tuple[int, ...], *, require_upstream: bool = True
) -> list[Path]:
    ensure_layout()
    require_packages(("pandas", "psutil", "openpyxl", "gurobipy"))
    require_files((
        CORE,
        cfg.path("inputs.timetable"),
        cfg.path("inputs.od_matrix"),
        cfg.path("inputs.corridors"),
        cfg.path("inputs.stations"),
        cfg.path("inputs.walking_times"),
    ))
    missing_upstream = [normal_file(window) for window in windows
                        if not normal_file(window).is_file()]
    if require_upstream:
        require_files(missing_upstream)
    return missing_upstream


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parallel = cfg.get("parallel.disruptions")
    parser.add_argument("--windows", default=",".join(str(w) for w in WINDOWS),
                        help="comma list or ranges of window start hours (default: time.windows)")
    parser.add_argument("--workers", type=int, default=parallel["workers"])
    parser.add_argument("--threads", type=int, default=parallel["threads"])
    parser.add_argument(
        "--output-dir", type=Path, default=DISRUPTION_DIR,
        help="isolated disruption-scenario output directory",
    )
    parser.add_argument(
        "--log-dir", type=Path,
        help=("isolated per-window log directory; custom outputs default to "
              "<output-dir>/_logs"),
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    windows = parse_windows(args.windows)
    output_dir = args.output_dir.resolve()
    log_dir = (
        args.log_dir.resolve() if args.log_dir is not None else
        (LOG_DIR.resolve() if output_dir == DISRUPTION_DIR.resolve()
         else output_dir / "_logs")
    )
    missing_upstream = check(
        windows, require_upstream=not args.check_only
    )
    if args.check_only:
        print(json.dumps({
            "status": (
                "PENDING_PREVIOUS_STAGE" if missing_upstream else "READY"
            ),
            "windows": list(windows),
            "durations": list(DURATIONS),
            "output_dir": str(output_dir),
            "log_dir": str(log_dir),
            "missing_normal_outputs": [
                str(path.resolve()) for path in missing_upstream
            ],
        }, ensure_ascii=False, indent=2))
        return
    run_window_processes(
        script=CORE, windows=windows, workers=args.workers,
        threads=args.threads, log_prefix="01_disruption", log_dir=log_dir,
        env_overrides={"PAPER_DISRUPTION_DIR": str(output_dir),
                       "PYTHONHASHSEED": str(cfg.get("scenarios.python_hash_seed"))},
    )
    stats = merge_statistics(output_dir)
    frame = pd.read_csv(stats)
    observed = set(pd.to_numeric(frame["duration"], errors="coerce").dropna().astype(int))
    if not observed.issubset(set(DURATIONS)):
        raise RuntimeError(f"unexpected duration values in fresh scenario stats: {observed}")
    print(f"Disruption generation complete: {stats}")


if __name__ == "__main__":
    main()
