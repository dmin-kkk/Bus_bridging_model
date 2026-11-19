"""Shared paths and lightweight launch helpers for the paper pipeline.

All folders and the values exported to child processes come from the active
profile (``bbconfig.cfg``): inputs from ``cfg.data_dir``, outputs from
``cfg.results_dir / cfg.get("paths.<stage>")``.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bbconfig import cfg  # noqa: E402

from pipeline_config import (  # noqa: E402
    DURATIONS, FLEET_SIZE, DEPOT_ACCESS_BOUND_MIN, OD_HOURLY_MIN_TRIPS, WINDOWS,
)


ROOT = Path(__file__).resolve().parent
INPUT_DIR = cfg.data_dir
CORE_DIR = ROOT / "_core"
RESULTS_DIR = cfg.results_dir
LOG_DIR = RESULTS_DIR / "logs"

NORMAL_DIR = RESULTS_DIR / cfg.get("paths.normal")
DISRUPTION_DIR = RESULTS_DIR / cfg.get("paths.disruptions")
CLUSTERING_DIR = RESULTS_DIR / cfg.get("paths.clustering")
ROUTE_DIR = RESULTS_DIR / cfg.get("paths.routes")
STAGE1_DIR = RESULTS_DIR / cfg.get("paths.stage1")
PREDICTION_DIR = RESULTS_DIR / cfg.get("paths.ml")
STAGE2_DIR = RESULTS_DIR / cfg.get("paths.stage2_proposed")

PYTHON = Path(sys.executable).resolve()


def ensure_layout() -> None:
    for path in (
        RESULTS_DIR, LOG_DIR, NORMAL_DIR,
        DISRUPTION_DIR, CLUSTERING_DIR, ROUTE_DIR, STAGE1_DIR,
        PREDICTION_DIR, STAGE2_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def parse_windows(value: str | None) -> tuple[int, ...]:
    if value is None or not value.strip():
        return WINDOWS
    parsed: set[int] = set()
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(item) for item in token.split("-", 1))
            parsed.update(range(start, end + 1))
        else:
            parsed.add(int(token))
    result = tuple(sorted(parsed))
    if not result or any(window not in WINDOWS for window in result):
        raise ValueError(f"windows must be among time.windows {list(WINDOWS)}")
    return result


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  " + "\n  ".join(missing))


def require_packages(names: Iterable[str]) -> None:
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Missing Python packages in the active interpreter "
            f"({PYTHON}): {', '.join(missing)}"
        )


def pipeline_env(
    threads: int = 0, overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "PYTHONUTF8": "1",
        "PAPER_INPUT_DIR": str(INPUT_DIR),
        "PAPER_NORMAL_DIR": str(NORMAL_DIR),
        "PAPER_DISRUPTION_DIR": str(DISRUPTION_DIR),
        "PAPER_CLUSTERING_DIR": str(CLUSTERING_DIR),
        "PAPER_ROUTE_DIR": str(ROUTE_DIR),
        "PAPER_STAGE1_DIR": str(STAGE1_DIR),
        "PAPER_PREDICTION_DIR": str(PREDICTION_DIR),
        "PAPER_STAGE2_DIR": str(STAGE2_DIR),
        "PAPER_DURATIONS": ",".join(str(duration) for duration in DURATIONS),
        "PAPER_OD_HOURLY_MIN_TRIPS": str(OD_HOURLY_MIN_TRIPS),
        "TSB_ROUTE_SOURCE": str(cfg.get("routes.source_mode")),
        "TSB_BUS_BOARD": str(cfg.get("routes.bus_boarding_mode")),
        "TSB_MAX_BUSES": str(FLEET_SIZE),
        "TSB_DEPOT_ACCESS_BOUND": str(DEPOT_ACCESS_BOUND_MIN),
        "BB_PROFILE": cfg.name,
        "GRB_THREADS": str(max(0, int(threads))),
    })
    if overrides:
        env.update({str(name): str(value) for name, value in overrides.items()})
    return env


def run_logged(
    command: list[str], *, log_path: Path, env: dict[str, str] | None = None,
    cwd: Path = ROOT,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    return int(process.returncode)


def run_window_processes(
    *, script: Path, windows: tuple[int, ...], workers: int,
    threads: int, log_prefix: str, log_dir: Path = LOG_DIR,
    env_overrides: Mapping[str, str] | None = None,
) -> None:
    if workers < 1:
        raise ValueError("workers must be at least one")
    effective_threads = max(1, int(threads)) if threads else max(
        1, (os.cpu_count() or 1) // workers
    )
    log_dir = Path(log_dir).resolve()

    def launch(window: int) -> tuple[int, int, Path]:
        log = log_dir / f"{log_prefix}_{window:02d}.log"
        code = run_logged(
            [str(PYTHON), "-X", "utf8", str(script), "--window", str(window)],
            log_path=log,
            env=pipeline_env(effective_threads, env_overrides),
        )
        return window, code, log

    failures: list[tuple[int, Path]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(launch, window): window for window in windows}
        for future in as_completed(futures):
            window, code, log = future.result()
            print(f"window {window:02d}: {'OK' if code == 0 else 'FAIL'} ({log})", flush=True)
            if code:
                failures.append((window, log))
    if failures:
        detail = ", ".join(f"{window:02d} ({log})" for window, log in failures)
        raise RuntimeError(f"one or more windows failed: {detail}")
