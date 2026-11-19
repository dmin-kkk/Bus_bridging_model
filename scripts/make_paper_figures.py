"""Regenerate the script-generated tables and figures from the result archive only.

    python scripts/make_paper_figures.py            # archive of the active profile: cfg.results_dir / archive
    python scripts/make_paper_figures.py --archive <results>/archive

Reads the CSV files written by ``scripts/build_archive.py`` (names in
``config/profiles/<profile>/archive.toml``); no solver, no restricted input.
Figures (PNG) go to ``cfg.results_dir / paths.figures``; tables (CSV + LaTeX)
to its ``tables`` sub-folder.  Items whose archive file is missing are skipped
with a message, so the script also runs on partial (e.g. synthetic) archives.
A status file ``figure_status.csv`` lists what was produced.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Visualization_Code"))
from bbconfig import cfg  # noqa: E402
from plot_ap_support_case import plot_case, ri_summary  # noqa: E402
from plot_style import LEGEND_FONT_SIZE, apply_global_style  # noqa: E402

FILES = cfg.get("archive.files")
OUT = cfg.get("archive.outputs")
DPI = int(cfg.get("archive.figure_dpi"))
DURATIONS = tuple(int(d) for d in cfg.get("time.durations_min"))
# Case ids in the evaluation output: proposed run and fixed centralized benchmark (FIXED_<depot>).
BENCHMARK_CASES = ("BASELINE", f"FIXED_{str(cfg.get('evaluation.benchmark_depot')).upper()}")
DURATION_COLORS = ["#4C88CF", "#E06B65"]            # plot styling (short, long)
DURATION_LABEL = "{kind} disruption ({minutes} min)"
CASE_COLORS = ["#2F6DAE", "#D6604D"]                # plot styling (proposed, benchmark)
WINDOW_COLORS = ["#1f77b4", "#ff7f0e"]              # plot styling (sensitivity windows)
POLICY_COLORS = ["#C44E52", "#8C8C8C", "#4C72B0", "#55A868"]
STATUS: list[dict] = []


# ---------------------------------------------------------------- io helpers
class Ctx:
    def __init__(self, archive: Path, figures: Path):
        self.archive = archive
        self.figures = figures
        self.tables = figures / OUT["tables_subdir"]
        self.figures.mkdir(parents=True, exist_ok=True)
        self.tables.mkdir(parents=True, exist_ok=True)

    def load(self, key: str) -> pd.DataFrame | None:
        path = self.archive / FILES[key]
        if not path.is_file():
            return None
        return pd.read_csv(path, encoding="utf-8")

    def need(self, *keys: str) -> list[pd.DataFrame]:
        frames = [self.load(k) for k in keys]
        missing = [FILES[k] for k, f in zip(keys, frames) if f is None]
        if missing:
            raise FileNotFoundError("archive file(s) missing: " + ", ".join(missing))
        return frames

    def save_fig(self, fig, key: str) -> Path:
        path = self.figures / OUT[key]
        fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.10)
        plt.close(fig)
        return path

    def save_table(self, frame: pd.DataFrame, stem: str, latex: bool = True, float_format: str = "%.3f") -> Path:
        path = self.tables / f"{stem}.csv"
        frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        if latex:
            write_latex(frame, self.tables / f"{stem}.tex", float_format)
        return path


def write_latex(frame: pd.DataFrame, path: Path, float_format: str) -> None:
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "--" if not np.isfinite(value) else float_format % value
        return str(value).replace("_", r"\_").replace("%", r"\%").replace("\n", " ")
    lines = [r"\begin{tabular}{" + "l" * len(frame.columns) + "}", r"\toprule",
             " & ".join(cell(c) for c in frame.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(cell(v) for v in row) + r" \\" for row in frame.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def duration_label(duration: int) -> str:
    kind = "Short" if duration == min(DURATIONS) else "Long"
    return DURATION_LABEL.format(kind=kind, minutes=duration)


def corridor_axis(ctx: Ctx, present) -> tuple[list[str], list[str]]:
    labels = ctx.load("corridor_labels")
    if labels is None:
        order = sorted(present)
        return order, order
    labels = labels.sort_values("Order")
    order = [c for c in labels["Corridor"] if c in set(present)]
    order += sorted(c for c in present if c not in order)
    mapping = dict(zip(labels["Corridor"], labels["Label"].astype(str).str.replace("\\n", "\n", regex=False)))
    return order, [mapping.get(c, c) for c in order]


def stage2_source(ctx: Ctx, kind: str) -> tuple[str, pd.DataFrame | None]:
    """Fig. 8 uses the archived Fig. 8 run when present, else the proposed run."""
    for key in (f"fig8_{kind}", f"stage2_{kind}"):
        frame = ctx.load(key)
        if frame is not None:
            return FILES[key], frame
    return "", None


# ---------------------------------------------------------------- figures
def fig5a(ctx: Ctx) -> Path:
    (frame,) = ctx.need("data_hourly")
    apply_global_style()
    fig, axis = plt.subplots(figsize=(8.4, 5.8))
    for column, color, name in (("Boarding", "#2F6DAE", "Boarding"), ("Alighting", "#D6604D", "Alighting")):
        axis.plot(frame["Hour"], frame[column], color=color, linewidth=2.0, marker="o", markersize=4, label=name)
        axis.fill_between(frame["Hour"], frame[column] - frame[f"{column}_CI"], frame[column] + frame[f"{column}_CI"],
                          color=color, alpha=0.18, linewidth=0)
    axis.set_xlabel("Hour of day")
    axis.set_ylabel("Passengers per hour")
    axis.set_xticks(range(0, 24, 2))
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.legend(frameon=True)
    return ctx.save_fig(fig, "fig5a")


def fig5b(ctx: Ctx) -> Path:
    (frame,) = ctx.need("data_corridor")
    apply_global_style()
    order, labels = corridor_axis(ctx, frame["Corridor"].unique())
    frame = frame.set_index("Corridor").reindex(order)
    fig, axis = plt.subplots(figsize=(8.4, 5.8))
    x, width = np.arange(len(order)), 0.38
    for offset, column, color in ((-width / 2, "Boarding", "#3E73B9"), (width / 2, "Alighting", "#D6514B")):
        bars = axis.bar(x + offset, frame[column], width, color=color, edgecolor="#303030", linewidth=0.8, label=column)
        axis.bar_label(bars, labels=[f"{v:,.0f}" for v in frame[column]], padding=3, fontsize=10, rotation=90)
    axis.set_xticks(x, labels)
    axis.set_xlabel("Corridor")
    axis.set_ylabel("Daily average passengers")
    axis.set_ylim(0, 1.18 * float(frame[["Boarding", "Alighting"]].to_numpy().max()))
    axis.grid(axis="y", color="#D5D9DD", linewidth=0.65, alpha=0.75)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", frameon=True, facecolor="white")
    return ctx.save_fig(fig, "fig5b")


def fig6(ctx: Ctx) -> Path:
    (summary,) = ctx.need("clustering_metrics")
    from scipy.stats import t as student_t
    apply_global_style()
    methods = list(dict.fromkeys(summary["method"]))
    colors = ["#2F6DAE", "#E6862A", "#3A923A", "#C43C39"]
    markers, styles = ["o", "s", "^", "P"], ["-", "-.", "--", ":"]
    selected_k = int(cfg.get("clustering.k_per_window"))
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), sharex=True)
    panels = (("MeanWasserstein", "SDWasserstein", "Wasserstein distance"), ("MeanSSE", "SDSSE", "SSE"),
              ("MeanSilhouette", "SDSilhouette", "Silhouette coefficient"))
    for axis, (metric, sd, title) in zip(axes, panels):
        for i, method in enumerate(methods):
            part = summary[summary["method"].eq(method)].sort_values("k")
            x, mean, n = part["k"].to_numpy(float), part[metric].to_numpy(float), part["HourCount"].to_numpy(float)
            half = np.asarray([student_t.ppf(0.975, df=max(int(v) - 1, 1)) for v in n]) * part[sd].to_numpy(float) / np.sqrt(n)
            axis.fill_between(x, mean - half, mean + half, color=colors[i % 4], alpha=0.11, linewidth=0)
            axis.plot(x, mean, marker=markers[i % 4], markersize=4.0, linewidth=1.8, linestyle=styles[i % 4],
                      color=colors[i % 4], label=method)
        axis.axvline(selected_k, color="#555555", linestyle=":", linewidth=1.25)
        axis.set_ylabel(title, fontsize=11.3)
        axis.set_xlabel("K (representative stations per hour)")
        axis.grid(axis="y", linestyle="--", alpha=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    return ctx.save_fig(fig, "fig6")


def fig7(ctx: Ctx) -> Path:
    (frame,) = ctx.need("ml_importance")
    apply_global_style()
    frame = frame.sort_values("BrierScoreIncreaseMean")
    fig, axis = plt.subplots(figsize=(8.0, 5.6))
    axis.barh(frame["DisplayFeature"], frame["BrierScoreIncreaseMean"], xerr=frame["BrierScoreIncreaseCI95"],
              color="#4C72B0", edgecolor="black", linewidth=0.6, capsize=3)
    axis.axvline(0.0, color="black", linewidth=0.7)
    axis.set_xlabel("Increase in Brier score after permutation")
    axis.grid(axis="x", linestyle="--", alpha=0.4)
    return ctx.save_fig(fig, "fig7")


def grouped_duration_bars(axis, table, categories, column, metric, labels, low, high):
    x = np.arange(len(categories), dtype=float)
    width = 0.36
    for offset, duration, color in zip((-width / 2, width / 2), DURATIONS, DURATION_COLORS):
        indexed = table[table["Duration"].eq(duration)].set_index(column)
        values = indexed[metric].reindex(categories).to_numpy(float)
        yerr = np.vstack((values - indexed[low].reindex(categories).to_numpy(float),
                          indexed[high].reindex(categories).to_numpy(float) - values))
        bars = axis.bar(x + offset, values, width, color=color, edgecolor="black", linewidth=0.6,
                        label=duration_label(duration), yerr=yerr, capsize=3,
                        error_kw={"ecolor": "black", "elinewidth": 0.9})
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=LEGEND_FONT_SIZE)
    axis.set_xticks(x, labels)
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.grid(axis="y", linestyle=":", alpha=0.45)
    axis.legend(frameon=True)


def fig8_corridor(ctx: Ctx, metric: str, ylabel: str, key: str) -> Path:
    source, table = stage2_source(ctx, "corridor")
    if table is None:
        raise FileNotFoundError("no corridor metrics in the archive")
    apply_global_style()
    order, labels = corridor_axis(ctx, table["Corridor"].unique())
    fig, axis = plt.subplots(figsize=(10.0, 6.0))
    grouped_duration_bars(axis, table, order, "Corridor", metric, labels, f"{metric}_CI_Low", f"{metric}_CI_High")
    axis.set_xlabel("Corridor")
    axis.set_ylabel(ylabel)
    fig.tight_layout()
    return ctx.save_fig(fig, key)


def top_stations(cases: pd.DataFrame) -> list[str]:
    ranking = cases.groupby("Station")["RI_Disruption"].mean().sort_values(kind="stable")
    return ranking.head(int(cfg.get("archive.top_stations"))).index.tolist()


def fig8c(ctx: Ctx) -> Path:
    source, cases = stage2_source(ctx, "cases")
    if cases is None:
        raise FileNotFoundError("no case metrics in the archive")
    apply_global_style()
    stations = top_stations(cases)
    hours = sorted(cases["Window"].unique())
    subset = cases[cases["Station"].isin(stations)]
    values = {d: subset[subset["Duration"].eq(d)].pivot(index="Station", columns="Window", values="RI_Improvement_pct")
              .reindex(index=stations, columns=hours).clip(lower=0.0).to_numpy(float) for d in DURATIONS}
    vmax = max(max(float(np.nanmax(v)) if np.isfinite(v).any() else 0.0 for v in values.values()), 1e-9)
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad("#E8E8E8")
    fig, axes = plt.subplots(1, len(DURATIONS), figsize=(15.5, 6.2), sharey=True, constrained_layout=True)
    image = None
    for axis, duration in zip(np.atleast_1d(axes), DURATIONS):
        grid = values[duration]
        image = axis.imshow(grid, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid[i, j]
                text = "--" if not np.isfinite(value) else f"{value:.2f}"
                color = "#6E6E6E" if not np.isfinite(value) else ("white" if value >= 0.6 * vmax else "#222222")
                axis.text(j, i, text, ha="center", va="center", fontsize=LEGEND_FONT_SIZE - 2, color=color)
        axis.set_xticks(np.arange(len(hours)), [str(h) for h in hours])
        axis.set_yticks(np.arange(len(stations)), stations)
        axis.set_xlabel("Hour of day")
        axis.set_title(duration_label(duration), fontweight="bold", pad=10)
    np.atleast_1d(axes)[0].set_ylabel("Station")
    bar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.012, shrink=0.82, aspect=28)
    bar.set_label("RI improvement (%)")
    return ctx.save_fig(fig, "fig8c")


def fig9(ctx: Ctx, key_file: str, ylabel: str, key: str) -> Path:
    (hourly,) = ctx.need(key_file)
    apply_global_style()
    cases = list(BENCHMARK_CASES)
    names = list(cfg.get("archive.benchmark_labels"))
    hours = sorted(hourly["Window"].unique())
    fig, axis = plt.subplots(figsize=(16.0, 7.0))
    width, x = 0.34, np.arange(len(hours), dtype=float)
    for offset, case, name, color in zip((-width / 2, width / 2), cases, names, CASE_COLORS):
        indexed = hourly[hourly["Case_ID"].eq(case)].set_index("Window")
        means = indexed["mean"].reindex(hours).to_numpy(float)
        errors = indexed["ci95"].reindex(hours).to_numpy(float)
        bars = axis.bar(x + offset, means, width, yerr=errors, color=color, edgecolor="white", linewidth=0.8,
                        ecolor="#333333", capsize=4, label=f"{name} (95% CI)", zorder=3)
        for b, value, error in zip(bars, means, errors):
            if np.isfinite(value):
                axis.text(b.get_x() + b.get_width() / 2, value + (error if np.isfinite(error) else 0) + 0.02,
                          f"{value:.2f}", ha="center", va="bottom", fontsize=LEGEND_FONT_SIZE)
    axis.set_xticks(x, [str(h) for h in hours])
    axis.set_xlabel("Hour of day")
    axis.set_ylabel(ylabel)
    axis.set_axisbelow(True)
    axis.grid(axis="y", linestyle="--", linewidth=0.8, color="#C7CCD1", alpha=0.9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=True, facecolor="white", edgecolor="#BFBFBF")
    upper = float(np.nanmax(hourly["mean"] + hourly["ci95"].fillna(0)))
    axis.set_ylim(0.0, upper * 1.18 if upper > 0 else 1.0)
    fig.tight_layout()
    return ctx.save_fig(fig, key)


def fig10(ctx: Ctx, duration: int, key: str) -> Path:
    """Fig. 10a/b: AP trajectories of the support case (Visualization_Code/plot_ap_support_case.py)."""
    (curves,) = ctx.need("ap_case")
    return plot_case(curves, duration, ctx.figures / OUT[key], DPI)


def ap_case_ri(ctx: Ctx) -> Path:
    """Sec. 5.5: RI of the support-case trajectories and improvement over the disrupted condition."""
    (curves,) = ctx.need("ap_case")
    return ctx.save_table(ri_summary(curves), "fig10_ap_support_case_ri", float_format="%.4f")


def fig11(ctx: Ctx, metric_names: tuple[str, ...], ylabel: str, key: str) -> Path:
    (summary,) = ctx.need("oof_policy_summary")
    apply_global_style()
    part = summary[summary["Metric"].isin(metric_names)]
    order = [p for p in cfg.get("archive.oof_policy_order") if p in set(part["Policy"])]
    order += [p for p in part["Policy"] if p not in order]
    part = part.set_index("Policy").reindex(order)
    fig, axis = plt.subplots(figsize=(8.6, 5.6))
    x = np.arange(len(order))
    values = part["Value"].to_numpy(float)
    err = np.vstack((values - part["CI_Lower"].to_numpy(float), part["CI_Upper"].to_numpy(float) - values))
    bars = axis.bar(x, values, 0.6, yerr=err, capsize=4, color=POLICY_COLORS[:len(order)], edgecolor="black",
                    linewidth=0.6)
    axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=LEGEND_FONT_SIZE)
    axis.set_xticks(x, [p.replace(" ", "\n") for p in order])
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    return ctx.save_fig(fig, key)


def fig_sensitivity(ctx: Ctx, value: str, low: str, high: str, ylabel: str, key: str) -> Path:
    (values,) = ctx.need("sensitivity")
    apply_global_style()
    xlabels = {panel["caption"]: panel["xlabel"] for panel in cfg.get("archive.sensitivity_panels")}
    panels = list(dict.fromkeys(values["panel"]))
    windows = sorted(values["window"].unique())
    ymax = np.ceil(float(values[high].max()) * 1.40 * 2.0) / 2.0
    fig = plt.figure(figsize=(16.4, 10.4))
    rows = int(np.ceil(len(panels) / 3))
    axes = np.atleast_1d(fig.subplots(rows, 3)).ravel()
    for axis, panel in zip(axes, panels):
        part = values[values["panel"].eq(panel)]
        positions = sorted(part["position"].unique())
        labels = [part[part["position"].eq(p)]["parameter_label"].iloc[0] for p in positions]
        x = np.arange(len(positions), dtype=float)
        for index, (offset, window) in enumerate(zip((-0.18, 0.18), windows)):
            sub = part[part["window"].eq(window)].set_index("position").reindex(positions)
            y = sub[value].to_numpy(float)
            bars = axis.bar(x + offset, y, 0.36, color=WINDOW_COLORS[index % 2], edgecolor="black", linewidth=0.6,
                            yerr=np.vstack((y - sub[low].to_numpy(float), sub[high].to_numpy(float) - y)), capsize=3,
                            label=f"{window}:00 (95% CI)")
            for b, top in zip(bars, sub[high].to_numpy(float)):
                axis.annotate(f"{b.get_height():.2f}", xy=(b.get_x() + b.get_width() / 2, top),
                              xytext=(0, 3 + 9 * index), textcoords="offset points", ha="center", va="bottom",
                              fontsize=LEGEND_FONT_SIZE, clip_on=False)
        axis.set_xticks(x, labels)
        axis.set_xlabel(xlabels.get(panel, ""))
        axis.set_ylabel(ylabel, labelpad=8)
        axis.set_title(panel, pad=12)
        axis.set_ylim(0, ymax if ymax > 0 else 1)
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE, frameon=True)
    for axis in axes[len(panels):]:
        axis.set_visible(False)
    fig.subplots_adjust(left=0.065, right=0.990, top=0.955, bottom=0.070, wspace=0.34, hspace=0.42)
    return ctx.save_fig(fig, key)


# ---------------------------------------------------------------- tables
def table4(ctx: Ctx) -> Path:
    (alloc,) = ctx.need("stage1_allocations")
    wide = alloc.pivot_table(index="Depot", columns="Window", values="Buses", aggfunc="sum").fillna(0).astype(int)
    groups = cfg.get("archive.table4_depot_groups")
    rows = []
    listed = set()
    for group in groups:
        for depot in group["depots"]:
            if depot in wide.index:
                rows.append({"Corridor": group["group"], "Depot": depot, **wide.loc[depot].to_dict()})
                listed.add(depot)
    for depot in sorted(set(wide.index) - listed):
        if wide.loc[depot].sum() > 0:
            rows.append({"Corridor": "", "Depot": depot, **wide.loc[depot].to_dict()})
    table = pd.DataFrame(rows)
    shown = table.copy()
    for column in wide.columns:
        shown[column] = shown[column].map(lambda v: "--" if v == 0 else str(v))
    ctx.save_table(table, "table4_depot_allocation", latex=False)
    write_latex(shown, ctx.tables / "table4_depot_allocation.tex", "%.0f")
    return ctx.tables / "table4_depot_allocation.csv"


def table5(ctx: Ctx) -> Path:
    (frame,) = ctx.need("ml_cutoffs")
    table = pd.DataFrame({"Cutoff (min)": frame["CutoffMinutes"], "Long/Short": frame["Long"].astype(str) + "/" + frame["Short"].astype(str),
                          "Brier score": frame["BrierScore"], "Brier skill": frame["BrierSkillScore"],
                          "Brier skill (from rounded Brier)": 1 - frame["BrierScore"].round(3) / (frame["ObservedRate"] * (1 - frame["ObservedRate"])),
                          "Log loss": frame["LogLoss"], "ROC-AUC": frame["ROC_AUC"], "PR-AUC": frame["PR_AUC"],
                          "ECE": frame["ECE10"]})
    return ctx.save_table(table, "table5_cutoff_metrics")


def table6(ctx: Ctx) -> Path:
    (frame,) = ctx.need("ml_models")
    table = frame[["Model", "BrierScore", "LogLoss", "ROC_AUC", "PR_AUC", "ECE10"]].rename(
        columns={"BrierScore": "Brier", "LogLoss": "Log loss", "ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC", "ECE10": "ECE"})
    return ctx.save_table(table, "table6_classifier_comparison")


def table7(ctx: Ctx) -> Path:
    (frame,) = ctx.need("oof_cases")
    table = frame[["Case", "Policy", "InputP30", "InputP60", "RI", "RI_Improvement_pct", "PT_Reduction_min"]]
    return ctx.save_table(table, "table7_oof_cases", float_format="%.4f")


def table8(ctx: Ctx) -> Path:
    scaling, alloc = ctx.need("scaling", "scaling_allocations")
    wide = alloc.pivot_table(index="K", columns="Depot", values="Buses", aggfunc="sum").fillna(0).astype(int)
    depots = list(cfg.get("archive.scaling_depots")) or [d for d in wide.columns if wide[d].sum() > 0]
    table = pd.DataFrame({"K": scaling["K"], "Variables (1e6)": scaling["NumVars"] / 1e6,
                          "Constraints (1e6)": scaling["NumConstrs"] / 1e6, "Runtime (s)": scaling["BuildSolve_s"],
                          "Gap (%)": scaling["MIPGap_pct"], "Total elapsed (s)": scaling["TotalElapsed_s"]})
    table = table.merge(wide.reindex(columns=depots, fill_value=0), left_on="K", right_index=True, how="left")
    return ctx.save_table(table, "table8_stage1_scaling", float_format="%.2f")


def table_routes(ctx: Ctx) -> Path:
    (routes,) = ctx.need("routes")
    for kind, stem in (("LOC", "tableC1_local_routes"), ("EXP", "tableC2_express_routes")):
        part = routes[routes["Type"].eq(kind)][["Route", "BaseLine", "NumStops", "StopSequence", "RouteTime_min"]]
        ctx.save_table(part, stem, float_format="%.0f")
    return ctx.tables / "tableC1_local_routes.csv"


def runtime_summary(ctx: Ctx) -> Path:
    rows = []
    for key, stage in (("stage1_summary", "Stage 1"), ("stage2_runtime", "Stage 2 proposed"),
                       ("benchmark_runtime", "Stage 2 fixed benchmark"), ("scaling", "Stage 1 scaling")):
        frame = ctx.load(key)
        if frame is None:
            continue
        for basis in ("TotalElapsed_s", "Elapsed_s", "BuildSolve_s", "Solver_s"):
            if basis not in frame:
                continue
            values = pd.to_numeric(frame[basis], errors="coerce").dropna()
            if values.empty:
                continue
            top = frame.loc[values.idxmax()]
            rows.append({"stage": stage, "basis": basis, "n": len(values), "mean_s": values.mean(),
                         "median_s": values.median(), "p95_s": values.quantile(0.95), "max_s": values.max(),
                         "argmax": " ".join(str(top[c]) for c in ("Window", "Station", "K") if c in frame and pd.notna(top[c])),
                         "over_60s": int(values.gt(60).sum()), "over_60s_pct": 100 * values.gt(60).mean()})
    if not rows:
        raise FileNotFoundError("no runtime files in the archive")
    return ctx.save_table(pd.DataFrame(rows), "runtime_summary", float_format="%.2f")


def relocation_table(ctx: Ctx) -> Path:
    (alloc,) = ctx.need("stage1_allocations")
    wide = alloc.pivot_table(index="Window", columns="Depot", values="Buses", aggfunc="sum").fillna(0).sort_index()
    rows = []
    windows = list(wide.index)
    for before, after in zip(windows[:-1], windows[1:]):
        change = wide.loc[after] - wide.loc[before]
        rows.append({"from_window": before, "to_window": after, "buses_relocated_min": int(change.clip(lower=0).sum()),
                     "depots_losing": int((change < 0).sum()), "depots_gaining": int((change > 0).sum())})
    table = pd.DataFrame(rows)
    table.loc[len(table)] = {"from_window": "mean", "to_window": "", "buses_relocated_min": table["buses_relocated_min"].mean(),
                             "depots_losing": table["depots_losing"].mean(), "depots_gaining": table["depots_gaining"].mean()}
    return ctx.save_table(table, "relocation_bus_counts", float_format="%.2f")


def text_numbers(ctx: Ctx) -> Path:
    """Numbers quoted in the text, recomputed from the archive (for the manuscript comparison)."""
    rows = []
    add = lambda item, value, note="": rows.append({"item": item, "value": value, "note": note})  # noqa: E731
    _, corridor = stage2_source(ctx, "corridor")
    if corridor is not None:
        for row in corridor.itertuples(index=False):
            add(f"corridor {row.Corridor} D{row.Duration} RI improvement (%)", row.RI_Improvement_pct)
            add(f"corridor {row.Corridor} D{row.Duration} PT reduction (min)", row.PT_Reduction_min)
    _, cases = stage2_source(ctx, "cases")
    if cases is not None:
        for duration in DURATIONS:
            part = cases[cases["Duration"].eq(duration)]
            stations = top_stations(cases)
            top = part[part["Station"].isin(stations)].sort_values("RI_Improvement_pct", ascending=False).head(3)
            for rank, row in enumerate(top.itertuples(index=False), 1):
                add(f"top-10 heatmap D{duration} rank {rank}", row.RI_Improvement_pct, f"{row.Station} {row.Window}:00")
    for key, name in (("benchmark_hourly_ri", "RI improvement (%)"), ("benchmark_hourly_pt", "PT reduction (min)")):
        frame = ctx.load(key)
        if frame is not None:
            for row in frame.itertuples(index=False):
                add(f"hourly {row.Case_ID} {row.Window}:00 {name}", row.mean)
    data = ctx.load("data_corridor")
    if data is not None:
        for row in data.itertuples(index=False):
            add(f"boarding share {row.Corridor} (%)", row.Boarding_share_pct)
    oof = ctx.load("oof_policy_summary")
    if oof is not None:
        for row in oof.itertuples(index=False):
            add(f"OOF {row.Policy} {row.Metric}", row.Value)
    sens = ctx.load("sensitivity")
    if sens is not None:
        for row in sens.itertuples(index=False):
            add(f"sensitivity {row.case_id} {row.window}:00 RI improvement (%)", row.ri_improvement_pct, row.panel)
            add(f"sensitivity {row.case_id} {row.window}:00 PT reduction (min)", row.pt_reduction_min_per_affected_passenger, row.panel)
    path = ctx.tables / "text_numbers.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


# ---------------------------------------------------------------- main
def run(item: str, function, *args) -> None:
    try:
        output = function(*args)
        STATUS.append({"item": item, "status": "ok", "output": Path(output).name, "message": ""})
        print(f"ok    {item}: {Path(output).name}")
    except FileNotFoundError as exc:
        STATUS.append({"item": item, "status": "skipped", "output": "", "message": str(exc)})
        print(f"skip  {item}: {exc}")
    except Exception as exc:  # keep going; the status file records the failure
        STATUS.append({"item": item, "status": "error", "output": "", "message": f"{type(exc).__name__}: {exc}"})
        print(f"ERROR {item}: {exc}")
        traceback.print_exc()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=cfg.results_dir / "archive")
    parser.add_argument("--out", type=Path, default=cfg.results_dir / cfg.get("paths.figures"))
    args = parser.parse_args()
    if not args.archive.is_dir():
        print(f"archive folder not found: {args.archive}")
        return 1
    ctx = Ctx(args.archive.resolve(), args.out.resolve())
    run("Fig. 5a", fig5a, ctx)
    run("Fig. 5b", fig5b, ctx)
    run("Fig. 6", fig6, ctx)
    run("Fig. 7", fig7, ctx)
    run("Fig. 8a", fig8_corridor, ctx, "RI_Improvement_pct", "RI improvement (%)", "fig8a")
    run("Fig. 8b", fig8_corridor, ctx, "PT_Reduction_min", "PT reduction (min/affected passenger)", "fig8b")
    run("Fig. 8c", fig8c, ctx)
    run("Fig. 9a", fig9, ctx, "benchmark_hourly_ri", "RI improvement (%)", "fig9a")
    run("Fig. 9b", fig9, ctx, "benchmark_hourly_pt", "PT reduction (min/affected passenger)", "fig9b")
    run("Fig. 10a", fig10, ctx, min(DURATIONS), "fig10a")
    run("Fig. 10b", fig10, ctx, max(DURATIONS), "fig10b")
    run("Sec. 5.5 support-case RI", ap_case_ri, ctx)
    run("Fig. 11a", fig11, ctx, ("RI improvement", "RI gain"), "RI improvement (%)", "fig11a")
    run("Fig. 11b", fig11, ctx, ("PT reduction",), "PT reduction (min/affected passenger)", "fig11b")
    run("Fig. 12", fig_sensitivity, ctx, "ri_improvement_pct", "ci_low", "ci_high", "RI improvement (%)", "fig12")
    run("Fig. 13", fig_sensitivity, ctx, "pt_reduction_min_per_affected_passenger", "pt_reduction_ci_low",
        "pt_reduction_ci_high", "PT reduction (min/affected passenger)", "fig13")
    run("Table 4", table4, ctx)
    run("Table 5", table5, ctx)
    run("Table 6", table6, ctx)
    run("Table 7", table7, ctx)
    run("Table 8", table8, ctx)
    run("Tables C.1-C.2", table_routes, ctx)
    run("Sec. 5.8 runtimes", runtime_summary, ctx)
    run("Sec. 5.2 relocation counts", relocation_table, ctx)
    run("Text numbers", text_numbers, ctx)
    pd.DataFrame(STATUS).to_csv(ctx.figures / "figure_status.csv", index=False, encoding="utf-8", lineterminator="\n")
    failed = [s for s in STATUS if s["status"] == "error"]
    print(f"done: {sum(s['status'] == 'ok' for s in STATUS)} ok, "
          f"{sum(s['status'] == 'skipped' for s in STATUS)} skipped, {len(failed)} errors -> {ctx.figures}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
