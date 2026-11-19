"""AP trajectories of one out-of-fold case under the four duration-information settings (Fig. 11c/d type).

    python Visualization_Code/plot_oof_ap_case.py                     # first case of archive.oof_cases
    python Visualization_Code/plot_oof_ap_case.py --station <code> --hour <onset hour> --state <minutes>

It re-evaluates the Stage 2 decision files of the OOF runs (stage 08) of the active profile's results
(synthetic run: ``BB_PROFILE=synthetic``), so it needs the profile inputs and the OOF solve folders;
it is not part of ``run_all.py`` stage 11. A case is selected by its descriptors (station, onset
hour, realized duration state), as in ``archive.oof_cases`` / ``oof_table7_cases.csv``; no incident
identifier, date or observed duration appears in the console output or the file names. Output: one
PNG and a policy RI summary (CSV) in ``cfg.results_dir / paths.figures``.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "Paper_Execution_Pipeline"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bbconfig import cfg  # noqa: E402
from plot_style import apply_global_style  # noqa: E402

EVALUATOR = PIPELINE / "Sensitivity" / "source" / "evaluate_ri.py"
# Solve-folder patterns written by the stage 08 OOF runners.
HISTORICAL = "Historical/HIST_H{hour:02d}_{station}"
LIGHTGBM = "ML/ML_I{case_key}"
ORACLE = "Oracle/ORACLE_H{hour:02d}_{station}_D{state}"
WRONG = "ExtremeWrong/WRONG_H{hour:02d}_{station}_D{state}"
STYLE = {  # label: (colour, marker, line width) -- plot styling only
    "Normal": ("#4D4D4D", None, 1.20), "Disruption": ("#D55E00", "x", 1.10),
    "Wrong information": ("#CC79A7", "D", 1.10), "Historical": ("#E69F00", "s", 1.15),
    "LightGBM": ("#0072B2", "o", 1.25), "Perfect information": ("#009E73", "^", 1.35),
}


def parse_args() -> argparse.Namespace:
    default = (cfg.get("archive.oof_cases") or [{}])[0]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--station", default=default.get("station"))
    parser.add_argument("--hour", type=int, default=default.get("hour"))
    parser.add_argument("--state", type=int, default=default.get("state"))
    parser.add_argument("--oof-dir", type=Path, default=cfg.results_dir / cfg.get("paths.oof"))
    parser.add_argument("--wrong-dir", type=Path, default=cfg.results_dir / cfg.get("paths.oof_wrong"))
    parser.add_argument("--disruption-dir", type=Path, default=cfg.results_dir / cfg.get("paths.disruptions"))
    parser.add_argument("--output-dir", type=Path, default=cfg.results_dir / cfg.get("paths.figures"))
    args = parser.parse_args()
    if args.station is None or args.hour is None or args.state is None:
        parser.error("give --station, --hour and --state (no archive.oof_cases in the profile)")
    args.station = str(args.station).upper()
    return args


def load_evaluator():
    sys.path.insert(0, str(PIPELINE))
    spec = importlib.util.spec_from_file_location("oof_ap_evaluate_ri", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def case_key(oof_dir: Path, station: str, hour: int, state: int) -> str:
    """Key of the LightGBM solve folder for the unique OOF case with these descriptors (kept in memory only)."""
    cases = pd.read_csv(oof_dir / "inputs" / "oof_incident_cases.csv")
    match = cases[cases["Station"].astype(str).str.upper().eq(station) & cases["Hour"].astype(int).eq(hour)
                  & cases["RealizedState"].astype(int).eq(state)]
    if len(match) != 1:
        raise SystemExit(f"{len(match)} OOF cases match station {station}, hour {hour}, state {state}; need exactly one")
    return str(match.iloc[0]["IncidentID"])   # input-schema column; the value never leaves memory


def main() -> None:
    args = parse_args()
    station, hour, state = args.station, args.hour, args.state
    start, end = int(cfg.get("time.evaluation_start_min")), int(cfg.get("time.horizon_min"))
    scenario = f"Disrupt_{hour:02d}_{station}_{state}.csv"
    solves = args.oof_dir / "solves"
    policies = {
        "Wrong information": args.wrong_dir / "solves" / WRONG.format(hour=hour, station=station, state=state),
        "Historical": solves / HISTORICAL.format(hour=hour, station=station),
        "LightGBM": solves / LIGHTGBM.format(case_key=case_key(args.oof_dir, station, hour, state)),
        "Perfect information": solves / ORACLE.format(hour=hour, station=station, state=state),
    }
    files = {label: folder / "DecisionVars" / scenario for label, folder in policies.items()}
    disruption_file = args.disruption_dir / scenario
    if not disruption_file.is_file():   # stage 01 does not zero-pad hours < 10
        disruption_file = args.disruption_dir / f"Disrupt_{hour}_{station}_{state}.csv"
    missing = [label for label, path in [("Disruption", disruption_file), *files.items()] if not path.is_file()]
    if missing:
        raise SystemExit("missing decision files for: " + ", ".join(missing))

    evaluator = load_evaluator()
    normal = evaluator.load_normal(hour)
    disruption = evaluator.read_condition(disruption_file)
    unaffected = evaluator.normal_unaffected_qlw(normal, disruption.od_set)
    conditions = {"Disruption": disruption.total_qlw + unaffected}
    for label, path in files.items():
        condition = evaluator.read_condition(path)
        if condition.od_set != disruption.od_set:
            raise AssertionError(f"{label}: OD universe differs from the disruption run")
        conditions[label] = condition.total_qlw + unaffected
    time_index = evaluator.TIME_INDEX
    curves = pd.DataFrame({"Time": time_index.to_numpy(dtype=int), "Normal": 1.0})
    for label, values in conditions.items():
        curves[label] = evaluator.performance_ratio(normal.total_qlw, values, time_index)

    window = curves["Time"].between(start, end)
    summary = pd.DataFrame({"Condition": list(STYLE), "RI": [float(curves.loc[window, c].mean()) for c in STYLE]})
    base = float(summary.loc[summary["Condition"].eq("Disruption"), "RI"].iloc[0])
    summary["RI_Improvement_pct"] = 100.0 * (summary["RI"] - base) / base
    summary.insert(0, "Case", f"{station} {hour:02d}:00 D{state}")

    apply_global_style()
    fig, axis = plt.subplots(figsize=(10.0, 6.0))
    axis.axvspan(start, start + state, facecolor="#F4A3A3", alpha=0.28, zorder=0, label="Disruption period")
    marks = np.flatnonzero(curves["Time"].to_numpy(dtype=int) % 2 == 0)
    for label, (color, marker, width) in STYLE.items():
        axis.plot(curves["Time"], curves[label], color=color, linewidth=width, alpha=0.9, marker=marker,
                  markevery=marks if marker else None, markersize=2.3, markerfacecolor="white",
                  markeredgecolor=color, markeredgewidth=0.65, label=label, zorder=3)
    axis.set_xlabel("Time (min)")
    axis.set_ylabel("AP")
    axis.set_xlim(0, end)
    axis.set_ylim(0, 1.05)
    axis.grid(axis="both", linestyle="--", linewidth=0.8, color="#C7CCD1", alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="lower right", ncol=2, frameon=True, facecolor="white", edgecolor="#BFBFBF")
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fig11_oof_ap_{station.lower()}_h{hour:02d}_d{state}"
    fig.savefig(args.output_dir / f"{stem}.png", dpi=int(cfg.get("archive.figure_dpi")), bbox_inches="tight")
    plt.close(fig)
    summary.to_csv(args.output_dir / f"{stem}_ri.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
