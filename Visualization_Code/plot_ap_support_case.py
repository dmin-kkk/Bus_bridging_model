"""AP trajectories of one location-window support case (Fig. 10a/b type).

    python Visualization_Code/plot_ap_support_case.py      # case of archive.ap_case, results of the active profile

The case (station, planning window), the comparison Stage 2 run and the legend labels are set in
``archive.ap_case`` of the profile. For every duration realization the AP curve
(Normal Q+L+W divided by the condition's Q+L+W, network-wide, minute by minute) is computed for the
disrupted condition and for the decision files of the comparison run and of the proposed run, as in
``Sensitivity/source/evaluate_ri.py`` (full OD set: ODs outside a file's support are restored from
the Normal window). The RI of a curve is its mean over ``time.evaluation_start_min`` .. horizon.

``scripts/build_archive.py`` calls ``compute_curves`` and stores the curves in the archive;
``scripts/make_paper_figures.py`` calls ``plot_case`` (Fig. 10a = shortest, Fig. 10b = longest
duration). Run directly, the script computes the curves from ``cfg.results_dir`` and writes both
panels to ``cfg.results_dir / paths.figures``.
"""
from __future__ import annotations

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
CONDITIONS = ("Normal", "Disruption", "Benchmark", "Proposed")
STYLE = {  # condition: (colour, marker, line width) -- plot styling only
    "Normal": ("#4D4D4D", None, 1.20), "Disruption": ("#D55E00", "x", 1.10),
    "Benchmark": ("#E69F00", "s", 1.15), "Proposed": ("#0072B2", "o", 1.25),
}


def load_evaluator(results: Path):
    sys.path.insert(0, str(PIPELINE))
    spec = importlib.util.spec_from_file_location("ap_case_evaluate_ri", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.NORMAL_DIR = results / cfg.get("paths.normal")
    return module


def compute_curves(results: Path) -> pd.DataFrame:
    """AP per minute of the configured case: columns Station, Window, Duration, Time, Normal, Disruption,
    Benchmark, Proposed."""
    case = cfg.get("archive.ap_case")
    station, window = str(case["station"]).upper(), int(case["window"])
    if not station:
        raise FileNotFoundError("no archive.ap_case.station in the profile")
    results = Path(results)
    runs = {"Benchmark": results / cfg.get(f"paths.{case['benchmark']}"),
            "Proposed": results / cfg.get("paths.stage2_proposed")}
    evaluator = load_evaluator(results)
    normal = evaluator.load_normal(window)
    frames = []
    for duration in (int(d) for d in cfg.get("time.durations_min")):
        name = f"Disrupt_{window:02d}_{station}_{duration}.csv"
        disruptions = results / cfg.get("paths.disruptions")
        unpadded = disruptions / f"Disrupt_{window}_{station}_{duration}.csv"   # stage 01 does not zero-pad hours < 10
        files = {"Disruption": unpadded if unpadded.is_file() else disruptions / name,
                 **{label: run / "DecisionVars" / name for label, run in runs.items()}}
        missing = [label for label, path in files.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{name}: no file for " + ", ".join(missing))
        disruption = evaluator.read_condition(files["Disruption"], use_cache=False)
        curves = {"Time": evaluator.TIME_INDEX.to_numpy(dtype=int), "Normal": 1.0}
        for label, path in files.items():
            condition = disruption if label == "Disruption" else evaluator.read_condition(path, use_cache=False)
            support = condition.od_set | disruption.od_set
            qlw = condition.total_qlw + evaluator.normal_unaffected_qlw(normal, support)
            curves[label] = evaluator.performance_ratio(normal.total_qlw, qlw)
        frame = pd.DataFrame(curves)
        frame.insert(0, "Duration", duration)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "Window", window)
    out.insert(0, "Station", station)
    return out[["Station", "Window", "Duration", "Time", *CONDITIONS]]


def ri_summary(curves: pd.DataFrame) -> pd.DataFrame:
    """RI (mean AP over the evaluation period) and improvement over the disrupted condition."""
    start = int(cfg.get("time.evaluation_start_min"))
    rows = []
    for (station, window, duration), part in curves.groupby(["Station", "Window", "Duration"], sort=True):
        ri = part.loc[part["Time"].ge(start), list(CONDITIONS)].mean()
        for condition in CONDITIONS:
            rows.append({"Station": station, "Window": int(window), "Duration": int(duration),
                         "Condition": condition, "RI": float(ri[condition]),
                         "RI_Improvement_pct": 100.0 * (ri[condition] - ri["Disruption"]) / ri["Disruption"]})
    return pd.DataFrame(rows)


def plot_case(curves: pd.DataFrame, duration: int, path: Path, dpi: int) -> Path:
    case = cfg.get("archive.ap_case")
    labels = {"Normal": "Normal", "Disruption": "Disruption",
              "Benchmark": case["benchmark_label"], "Proposed": case["proposed_label"]}
    onset, horizon = int(cfg.get("time.disruption_onset_min")), int(cfg.get("time.horizon_min"))
    part = curves[curves["Duration"].eq(duration)].sort_values("Time")
    if part.empty:
        raise FileNotFoundError(f"no AP curves for duration {duration}")
    apply_global_style()
    fig, axis = plt.subplots(figsize=(10.0, 6.0))
    marks = np.flatnonzero(part["Time"].to_numpy(dtype=int) % 2 == 0)
    for condition in CONDITIONS:
        color, marker, width = STYLE[condition]
        axis.plot(part["Time"], part[condition], color=color, linewidth=width, marker=marker,
                  markevery=marks if marker else None, markersize=2.3, markerfacecolor="white",
                  markeredgecolor=color, markeredgewidth=0.65, label=labels[condition], zorder=3)
    axis.axvspan(onset, onset + duration, facecolor="#F4A3A3", alpha=0.28, zorder=0, label="Disruption period")
    axis.set_xlabel("Time (min)")
    axis.set_ylabel("AP")
    axis.set_xlim(0, horizon)
    axis.set_ylim(0, 1.05)
    axis.set_xticks(range(0, horizon + 1, 20))
    axis.grid(axis="both", linestyle="--", linewidth=0.8, color="#C7CCD1", alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#BFBFBF")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    curves = compute_curves(cfg.results_dir)
    out = cfg.results_dir / cfg.get("paths.figures")
    outputs = cfg.get("archive.outputs")
    durations = sorted(curves["Duration"].unique())
    for key, duration in (("fig10a", durations[0]), ("fig10b", durations[-1])):
        print("wrote", plot_case(curves, duration, out / outputs[key], int(cfg.get("archive.figure_dpi"))).name)
    print(ri_summary(curves).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
