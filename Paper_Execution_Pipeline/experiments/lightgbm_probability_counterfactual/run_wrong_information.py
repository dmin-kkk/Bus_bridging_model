"""Run the OOF opposite-state (wrong-information) stress test with bus-only NA.

Sec. 6.6: the wrong-information setting assigns probability one to the
duration state that is not realized.  The experiment keeps the supplied Stage-1 allocation and passenger wait-and-see
information structure.  Incidents with the same hour, station, and realized
duration share one solved policy, but every incident remains in the evaluation
cohort.  Canonical OOF result directories are read-only inputs.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


SCRIPT = Path(__file__).resolve()
PIPELINE = SCRIPT.parents[2]
PROJECT = PIPELINE.parent
sys.path.insert(0, str(PROJECT))

from bbconfig import cfg  # noqa: E402

STAGE2_RUNNER = PIPELINE / "06_stage2_core.py"
RESULTS = cfg.results_dir
ALLOCATION_DIR = RESULTS / cfg.get("paths.stage1") / "Allocation"
DEFAULT_OUTPUT = RESULTS / cfg.get("paths.oof_wrong")
DEFAULT_OOF_RESULTS = RESULTS / cfg.get("paths.oof")
# Set from --oof-results-root in main().
CASES_FILE = DEFAULT_OOF_RESULTS / "inputs" / "oof_incident_cases.csv"
HISTORICAL_ROOT = DEFAULT_OOF_RESULTS / "solves" / "Historical"
LEGACY_ML_ROOT = DEFAULT_OOF_RESULTS / "solves" / "ML"
WINDOWS = tuple(int(hour) for hour in cfg.get("time.windows"))
SHORT, LONG = (int(value) for value in cfg.get("time.durations_min"))
P_NEAR = float(cfg.get("evaluation.oof.wrong_p_near"))
P_FAR = float(cfg.get("evaluation.oof.wrong_p_far"))
# The Stage 2 runner turns a duration prior into counts; counts of
# p * PSEUDO_COUNT_SCALE reproduce p at the 1e-5 resolution of P_FAR.
PSEUDO_COUNT_SCALE = 100000
DURATION_COUNTS = {
    int(duration): int(count)
    for duration, count in cfg.get("probability.duration_counts").items()
}
EXPECTED_INCIDENTS = sum(DURATION_COUNTS.values())
EXPECTED_SHORT = DURATION_COUNTS[SHORT]
EXPECTED_LONG = DURATION_COUNTS[LONG]
# 0 = not checked (the count depends on the cohort).
EXPECTED_POLICY_JOBS = int(cfg.get("evaluation.oof.expected_wrong_policy_jobs"))


@dataclass(frozen=True)
class Job:
    job_id: str
    kind: str
    hour: int
    station: str
    realized_state: int
    p30: float
    p60: float
    incident_ids: tuple[int, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_cases() -> pd.DataFrame:
    cases = pd.read_csv(CASES_FILE)
    required = {
        "IncidentID", "Hour", "Station", "RealizedState",
        "MLP30", "MLP60", "HistoricalJobID", "MLJobID", "Eligible",
    }
    missing = sorted(required - set(cases.columns))
    if missing:
        raise AssertionError(f"case table missing columns: {missing}")
    cases = cases.copy()
    cases["IncidentID"] = cases["IncidentID"].astype(int)
    cases["Hour"] = cases["Hour"].astype(int)
    cases["Station"] = cases["Station"].astype(str).str.strip().str.upper()
    cases["RealizedState"] = cases["RealizedState"].astype(int)
    if len(cases) != EXPECTED_INCIDENTS or cases["IncidentID"].nunique() != EXPECTED_INCIDENTS:
        raise AssertionError(
            f"OOF cohort must contain {EXPECTED_INCIDENTS} unique incidents")
    state_counts = cases["RealizedState"].value_counts().to_dict()
    if state_counts != {SHORT: EXPECTED_SHORT, LONG: EXPECTED_LONG}:
        raise AssertionError(f"unexpected realized-state counts: {state_counts}")
    if not cases["RealizedState"].isin([SHORT, LONG]).all():
        raise AssertionError("realized states must be the short or long state")
    # Only incidents inside a planning window are evaluated (paper: all of them).
    cases = cases.loc[
        cases["Eligible"].astype(str).str.strip().str.lower().isin(
            {"true", "1"})
    ].copy()
    probability_error = (
        cases["MLP30"].astype(float) + cases["MLP60"].astype(float) - 1.0
    ).abs().max()
    if float(probability_error) > 1e-10:
        raise AssertionError("LightGBM probabilities do not sum to one")
    return cases.sort_values("IncidentID").reset_index(drop=True)


def extreme_jobs(cases: pd.DataFrame) -> list[Job]:
    jobs: list[Job] = []
    grouped = cases.groupby(["Hour", "Station", "RealizedState"], sort=True)
    for (hour, station, state), block in grouped:
        state = int(state)
        if state == SHORT:
            p30, p60 = P_FAR, P_NEAR
        else:
            p30, p60 = P_NEAR, P_FAR
        jobs.append(Job(
            job_id=f"WRONG_H{int(hour):02d}_{station}_D{state}",
            kind="extreme-wrong",
            hour=int(hour),
            station=str(station),
            realized_state=state,
            p30=p30,
            p60=p60,
            incident_ids=tuple(sorted(block["IncidentID"].astype(int).tolist())),
        ))
    if EXPECTED_POLICY_JOBS and len(jobs) != EXPECTED_POLICY_JOBS:
        raise AssertionError(
            f"expected {EXPECTED_POLICY_JOBS} unique policies, found {len(jobs)}")
    return jobs


def legacy_ml_complete(row: pd.Series) -> bool:
    hour = int(row["Hour"])
    station = str(row["Station"])
    state = int(row["RealizedState"])
    root = LEGACY_ML_ROOT / str(row["MLJobID"])
    summary = root / "summary.json"
    decision = root / "DecisionVars" / f"Disrupt_{hour:02d}_{station}_{state}.csv"
    if not summary.is_file() or not decision.is_file():
        return False
    value = json.loads(summary.read_text(encoding="utf-8"))
    return bool(value.get("has_incumbent"))


def missing_ml_jobs(cases: pd.DataFrame) -> list[Job]:
    missing = cases.loc[~cases.apply(legacy_ml_complete, axis=1)]
    jobs = [
        Job(
            job_id=str(row.MLJobID),
            kind="ml-completion",
            hour=int(row.Hour),
            station=str(row.Station),
            realized_state=int(row.RealizedState),
            p30=float(row.MLP30),
            p60=float(row.MLP60),
            incident_ids=(int(row.IncidentID),),
        )
        for row in missing.itertuples(index=False)
    ]
    return jobs


def output_dir(root: Path, job: Job) -> Path:
    family = "ExtremeWrong" if job.kind == "extreme-wrong" else "MLCompletion"
    return root / "solves" / family / job.job_id


def result_paths(root: Path, job: Job) -> tuple[Path, Path]:
    directory = output_dir(root, job)
    summary = (
        directory / "Performance"
        / f"FullMILP_{job.hour:02d}_{job.station}_summary.json"
    )
    decision = (
        directory / "DecisionVars"
        / f"Disrupt_{job.hour:02d}_{job.station}_{job.realized_state}.csv"
    )
    return summary, decision


def _result_core_complete(root: Path, job: Job) -> bool:
    summary, decision = result_paths(root, job)
    if not summary.is_file() or not decision.is_file():
        return False
    value = json.loads(summary.read_text(encoding="utf-8"))
    weights = value.get("duration_weights") or {}
    stats = value.get("stats") or {}
    passenger_sharing = value.get(
        "passenger_flow_variable_sharing",
        stats.get("PassengerFlowVariableSharing"),
    )
    passenger_recourse = value.get(
        "passenger_recourse", stats.get("PassengerRecourse"),
    )
    return bool(
        value.get("has_incumbent")
        and value.get("nonanticipativity_mode") == "bus-only"
        and passenger_recourse == "WAIT_AND_SEE"
        and passenger_sharing is False
        and abs(float(weights.get(str(SHORT), weights.get(SHORT, -1))) - job.p30) <= 1e-12
        and abs(float(weights.get(str(LONG), weights.get(LONG, -1))) - job.p60) <= 1e-12
    )


def result_complete(root: Path, job: Job, allocation_dir: Path) -> bool:
    if not _result_core_complete(root, job):
        return False
    allocation = Path(allocation_dir) / f"DepotAlloc_{job.hour:02d}.csv"
    manifest_path = output_dir(root, job) / "policy_manifest.json"
    if not allocation.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        Path(str(manifest.get("allocation", ""))).resolve()
        == allocation.resolve()
        and str(manifest.get("allocation_sha256", "")).lower()
        == sha256_file(allocation).lower()
    )


def load_runner():
    pipeline_path = str(PIPELINE)
    if pipeline_path not in sys.path:
        sys.path.insert(0, pipeline_path)
    name = f"oof303_extreme_wrong_stage2_{os.getpid()}"
    specification = importlib.util.spec_from_file_location(name, STAGE2_RUNNER)
    if specification is None or specification.loader is None:
        raise ImportError(STAGE2_RUNNER)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def run_one(
        root: Path, job: Job, threads: int, time_limit: float,
        mip_gap: float, allocation_dir: Path) -> int:
    allocation_dir = Path(allocation_dir).resolve()
    if result_complete(root, job, allocation_dir):
        print(f"JOB_ALREADY_COMPLETE {job.job_id}", flush=True)
        return 0
    allocation = allocation_dir / f"DepotAlloc_{job.hour:02d}.csv"
    mip_start = (
        HISTORICAL_ROOT / f"HIST_H{job.hour:02d}_{job.station}" / "policy.csv"
    )
    for path in (allocation, mip_start):
        if not path.is_file():
            raise FileNotFoundError(path)

    runner = load_runner()
    source = f"opposite-state-{job.job_id.lower()}"
    counts = {
        SHORT: max(1, int(round(job.p30 * PSEUDO_COUNT_SCALE))),
        LONG: max(1, int(round(job.p60 * PSEUDO_COUNT_SCALE))),
    }

    def fixed_prior(requested_source: str):
        if requested_source != source:
            raise AssertionError(requested_source)
        return dict(counts), {SHORT: job.p30, LONG: job.p60}

    runner.duration_prior = fixed_prior
    directory = output_dir(root, job)
    args = Namespace(
        single_case=True,
        check_only=False,
        nonanticipativity_mode="bus-only",
        exact_state_compression=True,
        exact_trip_cell_pruning=False,
        window=job.hour,
        station=job.station,
        oracle_duration=None,
        duration_p60=None,
        local_routes_only=False,
        start_window=job.hour,
        end_window=job.hour,
        buffer=int(cfg.get("time.clearance_buffer_min")),
        time_limit_seconds=float(time_limit),
        mip_gap=float(mip_gap),
        threads=int(threads),
        mip_focus=int(cfg.get("evaluation.solver.mip_focus")),
        heuristics=float(cfg.get("evaluation.solver.heuristics")),
        cuts=int(cfg.get("evaluation.solver.cuts")),
        mip_start_policy=mip_start.resolve(),
        workers=1,
        allocation_dir=allocation_dir,
        output_dir=directory.resolve(),
        case_label=job.job_id,
        duration_probability_source=source,
    )
    exit_code = int(runner.run_single_case(args))
    if exit_code not in (0, 3) or not _result_core_complete(root, job):
        raise RuntimeError(f"{job.job_id} failed with exit code {exit_code}")
    manifest = {
        "schema": "oof303-opposite-state-bus-only-policy-v1",
        "job_id": job.job_id,
        "kind": job.kind,
        "incident_ids": list(job.incident_ids),
        "window": job.hour,
        "station": job.station,
        "realized_state": job.realized_state,
        "probabilities": {f"D{SHORT}": job.p30, f"D{LONG}": job.p60},
        "nonanticipativity_mode": "bus-only",
        "passenger_recourse": "WAIT_AND_SEE",
        "allocation": str(allocation.resolve()),
        "allocation_sha256": sha256_file(allocation),
        "mip_start": str(mip_start.resolve()),
        "mip_start_sha256": sha256_file(mip_start),
        "runner": str(STAGE2_RUNNER.resolve()),
        "runner_sha256": sha256_file(STAGE2_RUNNER),
        "completed_utc": utc_now(),
    }
    write_json_atomic(directory / "policy_manifest.json", manifest)
    if not result_complete(root, job, allocation_dir):
        raise RuntimeError(f"{job.job_id}: allocation provenance gate failed")
    print(f"JOB_COMPLETE {job.job_id}", flush=True)
    return 0


def job_lookup(cases: pd.DataFrame, extreme_only: bool = False) -> dict[str, Job]:
    jobs = [*extreme_jobs(cases)]
    if not extreme_only:
        jobs.extend(missing_ml_jobs(cases))
    lookup = {job.job_id: job for job in jobs}
    if len(lookup) != len(jobs):
        raise AssertionError("duplicate job IDs")
    return lookup


def preflight(
        cases: pd.DataFrame, jobs: dict[str, Job],
        allocation_dir: Path) -> dict:
    allocation_dir = Path(allocation_dir).resolve()
    required = [CASES_FILE, STAGE2_RUNNER]
    required.extend(
        allocation_dir / f"DepotAlloc_{hour:02d}.csv" for hour in WINDOWS
    )
    for job in jobs.values():
        required.append(
            HISTORICAL_ROOT
            / f"HIST_H{job.hour:02d}_{job.station}"
            / "policy.csv"
        )
    missing = sorted(str(path) for path in set(required) if not path.is_file())
    extreme = [job for job in jobs.values() if job.kind == "extreme-wrong"]
    ml_completion = [job for job in jobs.values() if job.kind == "ml-completion"]
    return {
        "schema": "oof303-opposite-state-bus-only-preflight-v1",
        "ready": not missing,
        "case_rows": len(cases),
        "case_ids": int(cases["IncidentID"].nunique()),
        "d30_incidents": int(cases["RealizedState"].eq(SHORT).sum()),
        "d60_incidents": int(cases["RealizedState"].eq(LONG).sum()),
        "extreme_policy_jobs": len(extreme),
        "ml_completion_jobs": len(ml_completion),
        "ml_completion_ids": [job.incident_ids[0] for job in ml_completion],
        "case_table": str(CASES_FILE.resolve()),
        "case_table_sha256": sha256_file(CASES_FILE),
        "allocation_dir": str(allocation_dir),
        "nonanticipativity_mode": "bus-only",
        "passenger_recourse": "WAIT_AND_SEE",
        "missing_inputs": missing,
    }


def child_command(cli: argparse.Namespace, job_id: str) -> list[str]:
    return [
        sys.executable, "-X", "utf8", "-u", str(SCRIPT),
        "--single-job", job_id,
        "--output-dir", str(cli.output_dir),
        "--threads", str(cli.threads),
        "--time-limit-seconds", str(cli.time_limit_seconds),
        "--mip-gap", str(cli.mip_gap),
        "--allocation-dir", str(cli.allocation_dir),
        "--oof-results-root", str(cli.oof_results_root),
        *(["--extreme-only"] if cli.extreme_only else []),
    ]


def run_child(cli: argparse.Namespace, job_id: str) -> tuple[str, int]:
    logs = Path(cli.output_dir) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    with (logs / f"{job_id}.out.log").open("a", encoding="utf-8") as stdout, (
        logs / f"{job_id}.err.log"
    ).open("a", encoding="utf-8") as stderr:
        completed = subprocess.run(
            child_command(cli, job_id),
            cwd=str(PROJECT),
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return job_id, int(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allocation-dir", type=Path, default=ALLOCATION_DIR)
    parser.add_argument(
        "--oof-results-root", type=Path, default=DEFAULT_OOF_RESULTS,
        help="fresh OOF root supplying cases and Historical MIP starts",
    )
    parser.add_argument(
        "--workers", type=int, default=cfg.get("parallel.oof.workers"))
    parser.add_argument(
        "--threads", type=int, default=cfg.get("parallel.oof.threads"))
    parser.add_argument(
        "--time-limit-seconds", type=float,
        default=cfg.get("evaluation.solver.stage2_time_limit_s"))
    parser.add_argument(
        "--mip-gap", type=float, default=cfg.get("evaluation.solver.mip_gap"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--extreme-only", action="store_true",
        help="run only the opposite-state policies (no ML completion solves)")
    parser.add_argument("--single-job", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    global CASES_FILE, HISTORICAL_ROOT, LEGACY_ML_ROOT
    cli = parse_args()
    cli.allocation_dir = cli.allocation_dir.resolve()
    cli.oof_results_root = cli.oof_results_root.resolve()
    CASES_FILE = cli.oof_results_root / "inputs" / "oof_incident_cases.csv"
    HISTORICAL_ROOT = cli.oof_results_root / "solves" / "Historical"
    LEGACY_ML_ROOT = cli.oof_results_root / "solves" / "ML"
    if cli.workers < 1 or cli.threads < 1 or cli.time_limit_seconds <= 0:
        raise ValueError("workers, threads, and time limit must be positive")
    cases = load_cases()
    jobs = job_lookup(cases, cli.extreme_only)
    audit = preflight(cases, jobs, cli.allocation_dir)
    if cli.check_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
        return 0 if audit["ready"] else 1
    if not audit["ready"]:
        raise RuntimeError(f"preflight failed: {audit['missing_inputs']}")
    root = Path(cli.output_dir).resolve()
    if cli.single_job:
        if cli.single_job not in jobs:
            raise KeyError(cli.single_job)
        return run_one(
            root, jobs[cli.single_job], cli.threads,
            cli.time_limit_seconds, cli.mip_gap, cli.allocation_dir,
        )
    if root.exists() and not cli.resume:
        raise FileExistsError(f"output exists; use --resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    extreme = [job for job in jobs.values() if job.kind == "extreme-wrong"]
    case_map = pd.DataFrame([
        {
            "IncidentID": incident_id,
            "ExtremeWrongJobID": job.job_id,
            "Hour": job.hour,
            "Station": job.station,
            "RealizedState": job.realized_state,
            "P30": job.p30,
            "P60": job.p60,
        }
        for job in extreme
        for incident_id in job.incident_ids
    ]).sort_values("IncidentID")
    if len(case_map) != len(cases) or case_map["IncidentID"].nunique() != len(cases):
        raise AssertionError("extreme policy mapping does not cover all incidents")
    case_map.to_csv(root / "incident_policy_map.csv", index=False)
    write_json_atomic(root / "preflight.json", audit)

    pending = [
        job_id for job_id, job in jobs.items()
        if not result_complete(root, job, cli.allocation_dir)
    ]
    failures: dict[str, int] = {}
    progress_path = root / "BatchStatus" / "batch_progress.json"

    def save_progress() -> None:
        completed = sum(
            result_complete(root, job, cli.allocation_dir)
            for job in jobs.values()
        )
        write_json_atomic(progress_path, {
            "schema": "oof303-opposite-state-bus-only-batch-v1",
            "state": "COMPLETE" if completed == len(jobs) and not failures else "RUNNING",
            "incident_count": len(cases),
            "extreme_policy_jobs": len(extreme),
            "ml_completion_jobs": len(jobs) - len(extreme),
            "total_jobs": len(jobs),
            "completed_jobs": completed,
            "remaining_jobs": len(jobs) - completed,
            "failures": failures,
            "workers": cli.workers,
            "threads_per_worker": cli.threads,
            "time_limit_seconds": cli.time_limit_seconds,
            "target_mip_gap": cli.mip_gap,
            "updated_utc": utc_now(),
        })

    save_progress()
    with ThreadPoolExecutor(max_workers=cli.workers) as executor:
        futures = {executor.submit(run_child, cli, job_id): job_id for job_id in pending}
        for future in as_completed(futures):
            job_id = futures[future]
            try:
                _, exit_code = future.result()
            except Exception:
                exit_code = 99
            if (exit_code != 0 or not result_complete(
                    root, jobs[job_id], cli.allocation_dir)):
                failures[job_id] = exit_code
            save_progress()
            print(
                f"BATCH_PROGRESS job={job_id} exit={exit_code} "
                f"failures={len(failures)}",
                flush=True,
            )
    save_progress()
    final = json.loads(progress_path.read_text(encoding="utf-8"))
    return 0 if final["state"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
