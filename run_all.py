"""Run the whole pipeline for one configuration profile.

    python run_all.py --profile synthetic              # end-to-end run on the synthetic network
    python run_all.py --profile paper --check          # check inputs, hashes and packages only
    python run_all.py --profile synthetic --dry-run    # print the plan
    python run_all.py --profile synthetic --from-stage 4 --to-stage 6

Every parameter comes from ``config/profiles/<profile>/``. The paper profile reads its
inputs from ``$BB_DATA_DIR`` (restricted data, not distributed); the synthetic profile
reads ``data/synthetic/``. Outputs go to ``results/<profile>/`` unless ``$BB_RESULTS_DIR``
is set. Each task runs in its own process with ``BB_PROFILE`` set; its log is written to
``<results>/logs/``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import bbconfig  # noqa: E402

PIPELINE = ROOT / "Paper_Execution_Pipeline"
SOURCE = PIPELINE / "Sensitivity" / "source"
COUNTER = PIPELINE / "experiments" / "lightgbm_probability_counterfactual"
SCRIPTS = ROOT / "scripts"
PACKAGES = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "joblib", "openpyxl", "psutil",
            "gurobipy", "xgboost", "lightgbm", "catboost", "optuna", "adjustText"]


def build_plan(cfg) -> list[dict]:
    """Ordered tasks. Scripts read every model/solver setting from the profile themselves;
    run_all only passes process layout (workers/threads) and task selectors."""
    res = cfg.results_dir
    out = lambda key: res / cfg.get(f"paths.{key}")  # noqa: E731
    plan: list[dict] = []

    def add(stage, name, script, args=(), outputs=()):
        script = Path(script)
        plan.append({"stage": stage, "name": name, "script": str(script.relative_to(ROOT)),
                     "command": [sys.executable, "-X", "utf8", "-u", str(script), *map(str, args)],
                     "outputs": [str(p) for p in outputs]})

    def layout(key):
        item = cfg.get(f"parallel.{key}")
        return ["--workers", item["workers"], "--threads", item["threads"]]

    add(0, "normal", PIPELINE / "00_generate_normal.py", layout("normal"), [out("normal")])
    add(1, "disruptions", PIPELINE / "01_generate_disruptions.py", layout("disruptions"), [out("disruptions")])
    add(2, "clustering", PIPELINE / "02_reproduce_production_clustering.py", (), [out("clustering")])
    add(2, "clustering_comparison", PIPELINE / "02_cluster_scenarios.py", (), [out("clustering_comparison")])
    add(3, "routes", PIPELINE / "03_generate_routes.py", (), [out("routes")])
    add(4, "stage1", PIPELINE / "04_run_stage1.py", layout("stage1"), [out("stage1")])
    threads = ["--threads", cfg.get("parallel.ml_threads")]
    add(5, "ml_models", PIPELINE / "05_compare_models_long30_303_optuna.py", threads, [out("ml")])
    add(5, "ml_cutoffs", PIPELINE / "05_lightgbm_303_thresholds_and_permutation.py", threads, [out("ml")])
    add(5, "ml_no_type", PIPELINE / "05_lightgbm_303_no_incident_type.py", threads, [out("ml")])
    add(5, "ml_type_only", PIPELINE / "05_lightgbm_303_incident_type_only.py", threads, [out("ml")])
    add(5, "ml_full_grid", PIPELINE / "09_generate_full_grid_lightgbm_probabilities.py", threads, [out("probabilities")])
    add(6, "stage2_proposed", PIPELINE / "10_run_full_grid_ml_stage2.py", layout("stage2"), [out("stage2_proposed")])
    add(7, "benchmark_allocations", SOURCE / "prepare_fixed_benchmark_allocations.py", (), [out("benchmark_allocations")])
    add(7, "stage2_benchmark", PIPELINE / "10_run_full_grid_ml_stage2.py",
        ["--benchmark", *layout("stage2")], [out("stage2_benchmark")])
    add(8, "oof", PIPELINE / "08_run_oof_operational_value.py", layout("oof"), [out("oof")])
    add(8, "oof_wrong", COUNTER / "run_oof303_extreme_wrong_bus_only.py",
        ["--resume", "--extreme-only", *layout("oof")], [out("oof_wrong")])
    add(8, "oof_evaluation", COUNTER / "evaluate_oof303_extreme_wrong_bus_only.py", (), [out("oof_wrong")])
    add(9, "sensitivity_inputs", SOURCE / "prepare_inputs.py", (), [out("sensitivity")])
    add(9, "sensitivity", SOURCE / "run_sensitivity.py", ["--skip-prepare", *layout("sensitivity")], [out("sensitivity")])
    add(9, "sensitivity_summary", SOURCE / "summarize_results.py", (), [out("sensitivity")])
    add(10, "scaling_inputs", SOURCE / "prepare_legacy_kmeans_h16_scaling.py", (), [out("scaling")])
    add(10, "scaling", SOURCE / "run_legacy_kmeans_h16_scaling.py",
        ["--threads", cfg.get("parallel.scaling")["threads"]], [out("scaling")])
    windows = [str(w) for w in cfg.get("time.windows")]
    weights = cfg.get("pinned.clustering.station_probability_evaluation")
    weight_args = ["--station-probability-file", ROOT / weights] if cfg.get("pinned.use_pinned") and weights else []
    benchmark_case = f"FIXED_{cfg.get('evaluation.benchmark_depot')}"
    add(11, "evaluate_proposed", SOURCE / "evaluate_ri.py",
        ["--cases", "PROPOSED", "--windows", *windows, *weight_args,
         "--output-dir", out("evaluation") / "proposed"], [out("evaluation") / "proposed"])
    add(11, "evaluate_benchmark", SOURCE / "evaluate_ri.py",
        ["--cases", "PROPOSED", benchmark_case, "--windows", *windows, *weight_args,
         "--output-dir", out("evaluation") / "benchmark_comparison"], [out("evaluation") / "benchmark_comparison"])
    add(11, "evaluate_sensitivity", SOURCE / "evaluate_ri.py", weight_args, [out("sensitivity")])
    add(11, "archive", SCRIPTS / "build_archive.py", ["--from-results", res, "--out", res / "archive"], [res / "archive"])
    add(11, "figures", SCRIPTS / "make_paper_figures.py", ["--archive", res / "archive"], [out("figures")])
    return plan


def child_environment(cfg) -> dict:
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PAPER_", "TSB_", "RD_", "RG_", "LR_"))
           and key not in {"PYTHONHOME", "MPLCONFIGDIR", "MATPLOTLIBRC"}}
    env.update({
        "BB_PROFILE": cfg.name,
        # set iteration order (and hence constraint order and the reported optimum among ties)
        # depends on the string-hash seed; fixing it makes reruns bit-reproducible
        "PYTHONHASHSEED": str(cfg.get("scenarios.python_hash_seed")),
        "PYTHONPATH": os.pathsep.join([str(ROOT), str(PIPELINE)]),
        "PYTHONUTF8": "1", "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MPLCONFIGDIR": str(cfg.results_dir / cfg.get("paths.cache") / "matplotlib"),
    })
    return env


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def check(cfg, plan) -> None:
    errors = []
    for task in plan:
        if not (ROOT / task["script"]).is_file():
            errors.append(f"missing script: {task['script']}")
    try:
        data_dir = cfg.data_dir
    except bbconfig.ConfigError as exc:
        errors.append(str(exc))
        data_dir = None
    if data_dir is not None:
        expected = cfg.get("inputs_sha256", default={})
        for key, name in cfg.get("inputs").items():
            path = data_dir / name
            if not path.is_file():
                errors.append(f"missing input inputs.{key}: {path.name}")
            elif expected.get(key) and sha256(path) != expected[key].upper():
                errors.append(f"input differs from the reference hash: inputs.{key} ({path.name})")
    if cfg.get("pinned.use_pinned"):
        for group in ("clustering", "probabilities"):
            for key, value in cfg.get(f"pinned.{group}").items():
                if key.endswith("_sha256"):
                    continue
                path = ROOT / value
                if not path.is_file() or sha256(path) != cfg.get(f"pinned.{group}.{key}_sha256").upper():
                    errors.append(f"pinned file missing or modified: {value}")
    for name in PACKAGES:
        if importlib.util.find_spec(name) is None:
            errors.append(f"missing Python package: {name}")
    if sys.version_info < (3, 11):
        errors.append("Python 3.11 or newer is required (tested with 3.13)")
    if errors:
        raise SystemExit("CHECK FAILED\n" + "\n".join(f"  - {e}" for e in errors))
    print(f"CHECK OK: profile '{cfg.name}', {len(cfg.get('inputs'))} inputs, {len(plan)} tasks")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the bus-bridging pipeline for one profile.")
    parser.add_argument("--profile", default=os.environ.get("BB_PROFILE", "paper"),
                        choices=bbconfig.available_profiles())
    parser.add_argument("--check", action="store_true", help="check inputs and environment, then stop")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, then stop")
    parser.add_argument("--from-stage", type=int, default=0)
    parser.add_argument("--to-stage", type=int, default=11)
    parser.add_argument("--only", nargs="*", default=None, help="task names to run")
    args = parser.parse_args()

    cfg = bbconfig.load(args.profile)
    plan = [t for t in build_plan(cfg)
            if args.from_stage <= t["stage"] <= args.to_stage and (not args.only or t["name"] in args.only)]
    if args.dry_run:
        for task in plan:
            print(f"[{task['stage']:02d}] {task['name']:<22} {task['script']} {' '.join(task['command'][5:])}")
        return 0
    check(cfg, plan)
    if args.check:
        return 0

    results = cfg.results_dir
    logs = results / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = child_environment(cfg)
    status = {"profile": cfg.name, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "tasks": []}
    status_file = results / "run_status.json"
    for task in plan:
        print(f"==> [{task['stage']:02d}] {task['name']}", flush=True)
        start = time.time()
        log_path = logs / f"{task['stage']:02d}_{task['name']}.log"
        with log_path.open("w", encoding="utf-8") as log:
            code = subprocess.call(task["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record = {"stage": task["stage"], "name": task["name"], "return_code": code,
                  "seconds": round(time.time() - start, 1), "log": str(log_path.relative_to(results))}
        missing = [p for p in task["outputs"] if not Path(p).exists()]
        if code == 0 and missing:
            record["missing_outputs"] = [str(Path(p).relative_to(results)) for p in missing]
        status["tasks"].append(record)
        status_file.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"    return code {code}, {record['seconds']} s, log {record['log']}", flush=True)
        if code != 0 or record.get("missing_outputs"):
            print(f"STOPPED at {task['name']}; see {log_path}", file=sys.stderr)
            return 1
    status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    status_file.write_text(json.dumps(status, indent=2), encoding="utf-8")
    total = sum(t["seconds"] for t in status["tasks"])
    print(f"DONE: {len(plan)} tasks in {total / 60:.1f} min; status in {status_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
