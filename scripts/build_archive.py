"""Aggregate the result folders of a run into its result archive (``results/<profile>/archive``).

The archive holds only aggregated values (allocations, solver statistics,
case-level RI/PT ratios, policy/sensitivity/scaling summaries, clustering and
classifier metrics, route-level times printed in the paper).  It never holds
OD-level flows, passenger counts, per-incident records, incident/trip/vehicle
identifiers, dates, timestamps, free text or file paths.  Every figure and
table script (``scripts/make_paper_figures.py``) reads the archive only.
The package is synthetic-only: the archive is built from a synthetic run
(``run_all.py --profile synthetic``, stage 11).

Typical use
    python scripts/build_archive.py --from-results results/synthetic --out results/synthetic/archive
    python scripts/build_archive.py --stage1 <dir> --stage2-proposed <dir> ... --out <archive dir>

``--from-results`` derives every source from the package result layout
(``config/profiles/<profile>/paths.toml``); any explicit source argument
overrides the derived one.  Missing sources are skipped with a message.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bbconfig import cfg  # noqa: E402

FILES = cfg.get("archive.files")
# Case ids in the evaluation output: proposed run and fixed centralized benchmark (FIXED_<depot>).
BENCHMARK_CASES = ("BASELINE", f"FIXED_{str(cfg.get('evaluation.benchmark_depot')).upper()}")

# Column names that must never reach the archive (restricted-data screen).
FORBIDDEN_COLUMN = re.compile(
    r"(incident_?id|job_?id|trip_?id|vehicle|(^|_)date($|_)|timestamp|time_start|time_end|"
    r"file|path|demand|passenger_minutes|od_count|(^|_)od($|_)|flow|segtime|cumtime|seg_time|"
    r"cum_time|qlw|description|free_?text|comment|email|actualduration)", re.IGNORECASE)
FORBIDDEN_VALUE = re.compile(r"(?:[A-Za-z]:\\|/Users/|AppData|\\Users\\|@[A-Za-z0-9-]+\.)")


# ---------------------------------------------------------------- helpers
def label(path: Path | None) -> str:
    """Source label without any absolute or personal path component.

    Paths inside the package are written relative to the package root; other
    paths keep only the part below the folder named ``archive.source_root_marker``,
    else the last three components."""
    if path is None:
        return ""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        pass
    parts = path.parts
    marker = cfg.get("archive.source_root_marker")
    if marker in parts:
        index = len(parts) - 1 - parts[::-1].index(marker)
        return "/".join(parts[index + 1:])
    return "/".join(parts[-3:])


def first_existing(*candidates) -> Path | None:
    for item in candidates:
        if item is not None and Path(item).exists():
            return Path(item)
    return None


def find_file(base: Path | None, name: str, prefer: tuple[str, ...] = ()) -> Path | None:
    """File ``name`` (a glob pattern is allowed) in ``base``, its ``prefer`` sub-folders or below."""
    if base is None or not Path(base).exists():
        return None
    base = Path(base)
    if base.is_file():
        return base if base.name == name else None
    for sub in prefer:
        if (base / sub / name).is_file():
            return base / sub / name
    if (base / name).is_file():
        return base / name
    matches = sorted(base.rglob(name))
    return matches[0] if matches else None


def pick_ri(bases, required: set[str], any_prefix: str | None = None) -> Path | None:
    """First Bridging_RI.csv below ``bases`` whose Case_ID values contain ``required``."""
    for base in bases:
        if base is None or not Path(base).exists():
            continue
        base = Path(base)
        files = [base] if base.is_file() else sorted(base.rglob("Bridging_RI.csv"))
        for path in files:
            try:
                ids = pd.read_csv(path, usecols=["Case_ID"], encoding="utf-8-sig")["Case_ID"]
                cases = set(ids.astype(str).str.upper().replace({"PROPOSED": "BASELINE"}))
            except ValueError:          # no Case_ID column: a single-case (baseline) file
                cases = {"BASELINE"}
            if required <= cases and (any_prefix is None or any(c.startswith(any_prefix) for c in cases)):
                return path
    return None


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Archive:
    def __init__(self, out: Path):
        self.out = out
        self.rows: list[dict] = []
        self.dropped: list[dict] = []
        self.skipped: list[str] = []

    def drop(self, file_key: str, columns, reason: str) -> None:
        columns = [c for c in columns]
        if columns:
            self.dropped.append({"file": FILES[file_key], "columns": "; ".join(columns), "reason": reason})

    def skip(self, item: str, why: str) -> None:
        message = f"SKIP {item}: {why}"
        print(message)
        self.skipped.append(message)

    def write(self, key: str, frame: pd.DataFrame, description: str, source: str) -> None:
        bad = [c for c in frame.columns if FORBIDDEN_COLUMN.search(str(c))]
        if bad:
            raise ValueError(f"{key}: restricted column names {bad}")
        for column in frame.columns:
            if frame[column].dtype == object:
                hits = frame[column].astype(str).str.contains(FORBIDDEN_VALUE, regex=True)
                if hits.any():
                    raise ValueError(f"{key}: column {column} holds a path or e-mail value")
        path = self.out / FILES[key]
        frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.rows.append({"file": path.name, "description": description, "source": source,
                          "rows": len(frame), "columns": ";".join(map(str, frame.columns)),
                          "sha256": digest})
        print(f"wrote {path.name}: {len(frame)} rows")


# ---------------------------------------------------------------- aggregation formulas
# Aggregation formulas of the paper figures (case means and station-cluster bootstrap intervals).
def station_bootstrap_ci(group: pd.DataFrame, value: str | None, numerator: str | None,
                         weight: str | None, seed: int, reps: int, quantile: bool):
    nums, dens = [], []
    for _, station in group.groupby("Station", sort=True):
        if numerator is None:   # unweighted mean of a case-level value
            values = pd.to_numeric(station[value], errors="coerce").to_numpy(float)
            valid = np.isfinite(values)
            if valid.any():
                nums.append(float(values[valid].sum()))
                dens.append(float(valid.sum()))
        else:                   # passenger-minute ratio (optionally probability weighted)
            w = station[weight].to_numpy(float) if weight else np.ones(len(station))
            nums.append(float(np.sum(w * station[numerator].to_numpy(float))))
            dens.append(float(np.sum(w * station["Passenger_Demand"].to_numpy(float))))
    if not nums:
        return float("nan"), float("nan"), float("nan")
    num, den = np.asarray(nums), np.asarray(dens)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(num), size=(reps, len(num)))
    sden = den[sampled].sum(axis=1)
    valid = sden > 0.0
    estimates = num[sampled].sum(axis=1)[valid] / sden[valid]
    if estimates.size == 0:
        return float("nan"), float("nan"), float("nan")
    if quantile:
        low, high = np.quantile(estimates, (0.025, 0.975))
        return float(low), float(high), float("nan")
    std = float(np.std(estimates, ddof=1))
    return float("nan"), float("nan"), std


def prepare_ri(frame: pd.DataFrame, case_id: str | None = "BASELINE") -> pd.DataFrame:
    frame = frame.copy()
    if "Case_ID" in frame and case_id is not None:
        frame["Case_ID"] = frame["Case_ID"].astype(str).str.upper().replace({"PROPOSED": "BASELINE"})
    if "Evaluation_OD_Scope" in frame and set(frame["Evaluation_OD_Scope"].dropna()) - {"full"}:
        raise ValueError("RI file is not evaluated on the full OD universe")
    frame["Station"] = frame["Station"].astype(str).str.strip().str.upper()
    frame["Window"] = pd.to_numeric(frame["Window"]).astype(int)
    frame["Duration"] = pd.to_numeric(frame["Duration"]).astype(int)
    if "PT_Reduction_Passenger_Minutes" not in frame:
        frame["PT_Reduction_Passenger_Minutes"] = (frame["Disruption_Total_Travel_Passenger_Minutes"]
                                                   - frame["Total_Travel_Passenger_Minutes"])
    demand = pd.to_numeric(frame["Passenger_Demand"], errors="coerce").fillna(0.0)
    frame["Passenger_Demand"] = demand
    frame["PT_Reduction_min"] = np.where(demand > 0, frame["PT_Reduction_Passenger_Minutes"] / demand.where(demand > 0), np.nan)
    return frame


def attach_corridor(frame: pd.DataFrame, corridor_map: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(corridor_map, on=["Window", "Station"], how="left", validate="many_to_one")
    if merged["Corridor"].isna().any():
        fallback = corridor_map.drop_duplicates("Station").set_index("Station")["Corridor"]
        merged["Corridor"] = merged["Corridor"].fillna(merged["Station"].map(fallback))
    return merged


def case_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["Window", "Station", "Corridor", "Duration", "Scenario_Probability", "RI_Disruption",
            "RI_Bridging", "RI_Improvement_pct", "PT_Reduction_min"]
    if "Case_ID" in frame:
        cols = ["Case_ID", *cols]
    return frame[cols].sort_values([c for c in ("Case_ID", "Window", "Station", "Duration") if c in cols]).reset_index(drop=True)


def corridor_table(frame: pd.DataFrame, seed: int, reps: int) -> pd.DataFrame:
    rows = []
    for (corridor, duration), group in frame.groupby(["Corridor", "Duration"], sort=True):
        ri_values = pd.to_numeric(group["RI_Improvement_pct"], errors="coerce")
        demand = float(group["Passenger_Demand"].sum())
        ri_low, ri_high, _ = station_bootstrap_ci(group, "RI_Improvement_pct", None, None, seed, reps, True)
        pt_low, pt_high, _ = station_bootstrap_ci(group, None, "PT_Reduction_Passenger_Minutes", None, seed, reps, True)
        rows.append({"Corridor": corridor, "Duration": int(duration), "Scenario_Count": len(group),
                     "Affected_Scenario_Count": int((group["Passenger_Demand"] > 0).sum()),
                     "RI_Disruption": float(group["RI_Disruption"].mean()),
                     "RI_Bridging": float(group["RI_Bridging"].mean()),
                     "RI_Improvement_pct": float(ri_values[np.isfinite(ri_values)].mean()),
                     "RI_Improvement_pct_CI_Low": ri_low, "RI_Improvement_pct_CI_High": ri_high,
                     "PT_Reduction_min": (float(group["PT_Reduction_Passenger_Minutes"].sum()) / demand
                                          if demand > 0 else float("nan")),
                     "PT_Reduction_min_CI_Low": pt_low, "PT_Reduction_min_CI_High": pt_high})
    return pd.DataFrame(rows)


def hourly_benchmark_tables(frame: pd.DataFrame, seed: int, reps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ri = (frame.dropna(subset=["RI_Improvement_pct"]).groupby(["Case_ID", "Window"], as_index=False)
          ["RI_Improvement_pct"].agg(count="count", mean="mean", std="std"))
    ri["std"] = ri["std"].fillna(0.0)
    ri["ci95"] = 1.96 * ri["std"] / np.sqrt(ri["count"])
    rows = []
    for (case, window), group in frame.groupby(["Case_ID", "Window"], sort=True):
        weights = group["Scenario_Probability"].to_numpy(float)
        denominator = float(np.sum(weights * group["Passenger_Demand"].to_numpy(float)))
        if denominator <= 0:
            estimate = std = float("nan")
        else:
            estimate = float(np.sum(weights * group["PT_Reduction_Passenger_Minutes"].to_numpy(float)) / denominator)
            std = station_bootstrap_ci(group, None, "PT_Reduction_Passenger_Minutes", "Scenario_Probability",
                                       seed, reps, False)[2]
        rows.append({"Case_ID": case, "Window": int(window), "scenario_count": len(group),
                     "count": int((group["Passenger_Demand"] > 0).sum()), "mean": estimate, "std": std,
                     "ci95": 1.96 * std})
    return ri, pd.DataFrame(rows)


# ---------------------------------------------------------------- items
def stage1_items(arc: Archive, stage1: Path | None) -> None:
    if stage1 is None or not (stage1 / "Allocation").is_dir():
        arc.skip("stage1", f"no Allocation/ folder in {label(stage1) or '(not given)'}")
        return
    allocations, summaries = [], []
    for path in sorted((stage1 / "Allocation").glob("DepotAlloc_*.csv")):
        window = int(path.stem.split("_")[-1])
        table = read_csv(path)
        depot_col, count_col = table.columns[:2]
        for depot, count in table[[depot_col, count_col]].itertuples(index=False):
            allocations.append({"Window": window, "Depot": str(depot).strip(), "Buses": int(count)})
    for path in sorted((stage1 / "Performance").glob("FullMILP_*_summary.json")):
        data = read_json(path)
        stats = data.get("stats", {})
        summaries.append(stage1_summary_row(data, stats))
    if allocations:
        arc.write("stage1_allocations", pd.DataFrame(allocations),
                  "Stage 1 reserve targets: buses per depot and planning window (Table 4)", label(stage1))
    if summaries:
        arc.write("stage1_summary", pd.DataFrame(summaries).sort_values("Window"),
                  "Stage 1 solver statistics per window: objective, bound, gap, runtimes, model size (Sec. 5.8)",
                  label(stage1))
        arc.drop("stage1_summary", ["representative_branches", "allocation_file", "warm_start_source",
                                    "scenario/omega/theta counts by scenario", "decision-variable file lists"],
                 "absolute paths and per-scenario passenger-state counts; not needed for any table")


def stage1_summary_row(data: dict, stats: dict) -> dict:
    runtime, build = stats.get("Runtime"), stats.get("BuildSec")
    gap = stats.get("MIPGap", data.get("relative_gap"))
    return {"Window": int(data["window"]), "K": data.get("k"), "Status": data.get("status", stats.get("Status")),
            "Objective": data.get("objective", stats.get("ObjVal")), "Bound": data.get("bound", stats.get("ObjBound")),
            "MIPGap_pct": None if gap is None else 100.0 * gap,
            "Solver_s": runtime, "Build_s": build,
            "BuildSolve_s": None if runtime is None or build is None else runtime + build,
            "TotalElapsed_s": data.get("total_elapsed_seconds", data.get("elapsed_seconds")),
            "NumVars": stats.get("NumVars"), "NumConstrs": stats.get("NumConstrs"), "NumNZs": stats.get("NumNZs"),
            "FleetSize": data.get("fleet_size")}


def stage2_runtime_item(arc: Archive, key: str, run: Path | None, what: str) -> None:
    perf = None if run is None else run / "Performance"
    if perf is None or not perf.is_dir():
        arc.skip(key, f"no Performance/ folder in {label(run) or '(not given)'}")
        return
    rows = []
    for path in sorted(perf.glob("FullMILP_*_*_summary.json")):
        data = read_json(path)
        stats = data.get("stats", {})
        runtime, build = stats.get("Runtime"), stats.get("BuildSec")
        gap = stats.get("MIPGap")
        rows.append({"Window": int(data["window"]), "Station": str(data["station"]).upper(),
                     "Status": data.get("status", stats.get("Status")),
                     "MIPGap_pct": None if gap is None else 100.0 * gap,
                     "Solver_s": runtime, "Build_s": build,
                     "BuildSolve_s": None if runtime is None or build is None else runtime + build,
                     "Elapsed_s": data.get("elapsed_seconds", data.get("total_elapsed_seconds")),
                     "NumVars": stats.get("NumVars"), "NumConstrs": stats.get("NumConstrs")})
    if not rows:
        arc.skip(key, f"no Stage 2 summaries in {label(run)}")
        return
    arc.write(key, pd.DataFrame(rows).sort_values(["Window", "Station"]),
              f"{what}: solver status, gap and runtimes per location-window case (Sec. 5.8)", label(run))
    arc.drop(key, ["objective", "Z30", "Z60", "duration_weights", "policy_file", "decision_variable_files"],
             "objective values are passenger-time totals (demand-scaled); file lists hold absolute paths")


def ri_items(arc: Archive, ri_file: Path | None, corridor_map, key_cases: str, key_corridor: str | None,
             what: str, seed: int, reps: int, case_ids=("BASELINE",)) -> pd.DataFrame | None:
    if ri_file is None:
        arc.skip(key_cases, "no Bridging_RI.csv found")
        return None
    frame = prepare_ri(read_csv(ri_file))
    if "Case_ID" in frame:
        frame = frame[frame["Case_ID"].isin(case_ids)].copy()
    if frame.empty:
        arc.skip(key_cases, f"no rows with Case_ID in {case_ids} in {label(ri_file)}")
        return None
    if corridor_map is None:
        arc.skip(key_cases, "no station-corridor map")
        return None
    frame = attach_corridor(frame, corridor_map)
    cases = case_metrics(frame)
    if len(case_ids) == 1 and "Case_ID" in cases:
        cases = cases.drop(columns="Case_ID")
    arc.write(key_cases, cases, f"{what}: RI and PT reduction per location-window-duration case", label(ri_file))
    arc.drop(key_cases, ["Passenger_Demand", "*_Passenger_Minutes", "*_OD_Count", "Normal_QLW_Area", "QLW_Area",
                         "Source_File", "Time_Start", "Time_End", "average waiting/in-vehicle/walking times"],
             "affected demand and passenger-minute totals are demand magnitudes (OD-derived); OD counts describe "
             "OD-level structure; Source_File is an absolute path; component times are not printed in the paper")
    if key_corridor is not None:
        arc.write(key_corridor, corridor_table(frame, seed, reps),
                  f"{what}: corridor x duration means with 95% station-cluster bootstrap CI (Fig. 8a/b, Sec. 5.4)",
                  label(ri_file))
    return frame


def benchmark_items(arc: Archive, ri_file: Path | None, corridor_map, seed: int, reps: int) -> None:
    case_ids = BENCHMARK_CASES
    frame = ri_items(arc, ri_file, corridor_map, "benchmark_cases", None,
                     "Proposed vs fixed benchmark", seed, reps, case_ids)
    if frame is None:
        return
    ri, pt = hourly_benchmark_tables(frame, seed, reps)
    arc.write("benchmark_hourly_ri", ri, "Hourly mean RI improvement and 95% CI, proposed vs fixed benchmark (Fig. 9a)",
              label(ri_file))
    arc.write("benchmark_hourly_pt", pt,
              "Hourly probability-weighted PT reduction and 95% CI, proposed vs fixed benchmark (Fig. 9b)",
              label(ri_file))


def oof_items(arc: Archive, wrong: Path | None, value: Path | None, cases: Path | None) -> None:
    summary = find_file(wrong, "policy_summary_*.csv") or find_file(wrong, "policy_summary.csv", ("evaluation",))
    if summary is None:
        arc.skip("oof_policy_summary", f"no policy summary in {label(wrong) or '(not given)'}")
    else:
        table = read_csv(summary)
        arc.write("oof_policy_summary", table[["Policy", "Metric", "Value", "CI_Lower", "CI_Upper", "Unit"]],
                  "Out-of-fold policy means with 95% CI, four information settings (Fig. 11a/b, Sec. 5.6)",
                  label(summary.parent.parent))
    value_summary = find_file(value, "policy_summary.csv", ("evaluation",))
    if value_summary is not None:
        table = read_csv(value_summary)
        keep = [c for c in table.columns if not FORBIDDEN_COLUMN.search(c)]
        arc.write("oof_value_policy_summary", table[keep],
                  "Out-of-fold policy summary of the operational-value run (cross-check of Fig. 11a/b)",
                  label(value_summary.parent.parent))
        arc.drop("oof_value_policy_summary", [c for c in table.columns if c not in keep], "restricted column screen")
    results = find_file(cases, "incident_policy_results_*.csv") or find_file(cases, "incident_policy_results.csv", ("evaluation",))
    selected = cfg.get("archive.oof_cases")
    if results is None or not selected:
        arc.skip("oof_cases", "no incident_policy_results file or no archive.oof_cases configured")
        return
    table = read_csv(results)
    rows = []
    for case in selected:
        part = table[table["Station"].astype(str).str.upper().eq(case["station"])
                     & table["Hour"].astype(int).eq(int(case["hour"]))
                     & table["RealizedState"].astype(int).eq(int(case["state"]))]
        if part.empty or part["IncidentID"].nunique() != 1:
            arc.skip("oof_cases", f"case {case['station']} {case['hour']}:00 not unique ({part['IncidentID'].nunique()} incidents)")
            continue
        first = part.iloc[0]
        rows.append({"Case": case["label"], "Station": case["station"], "Hour": int(case["hour"]),
                     "RealizedState": int(case["state"]), "Policy": "Disruption", "InputP30": np.nan,
                     "InputP60": np.nan, "RI": float(first["RI_Disruption"]), "RI_Improvement_pct": np.nan,
                     "PT_Reduction_min": np.nan})
        for item in part.itertuples(index=False):
            demand = float(item.Passenger_Demand)
            rows.append({"Case": case["label"], "Station": case["station"], "Hour": int(case["hour"]),
                         "RealizedState": int(case["state"]), "Policy": item.Policy,
                         "InputP30": float(item.InputP30), "InputP60": float(item.InputP60), "RI": float(item.RI),
                         "RI_Improvement_pct": 100.0 * (float(item.RI) - float(item.RI_Disruption)) / float(item.RI_Disruption),
                         "PT_Reduction_min": float(item.PT_Reduction_Passenger_Minutes) / demand if demand > 0 else np.nan})
    if rows:
        arc.write("oof_cases", pd.DataFrame(rows),
                  "Out-of-fold cases of archive.oof_cases (Table 7: station, hour, policy, input probabilities, RI, PT)",
                  label(results.parent.parent))
        arc.drop("oof_cases", ["IncidentID", "JobID", "DecisionFile", "ActualDurationMinutes", "IncidentType",
                               "StationCorridor", "Passenger_Demand", "PT_Reduction_Passenger_Minutes"],
                 "per-incident identifiers, exact duration, file paths and demand magnitudes; only the values "
                 "printed in Table 7 are kept (incident category is printed in the table caption only)")


SENSITIVITY_KEEP = ["panel", "position", "parameter_label", "case_id", "window", "scenario_count",
                    "affected_scenario_count", "ri_improvement_pct", "ci_low", "ci_high",
                    "pt_reduction_min_per_affected_passenger", "pt_reduction_ci_low", "pt_reduction_ci_high"]


def sensitivity_mean_ci(frame: pd.DataFrame, column: str, seed: int, reps: int):
    """Unweighted case mean, station-cluster bootstrap."""
    valid = frame[["Station", column]].dropna()
    groups = [g[column].to_numpy(float) for _, g in valid.groupby("Station", sort=False)]
    rng = np.random.default_rng(seed)
    sampled = np.empty(reps)
    for repetition in range(reps):
        index = rng.integers(0, len(groups), size=len(groups))
        sampled[repetition] = np.concatenate([groups[i] for i in index]).mean()
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(valid[column].to_numpy(float).mean()), float(low), float(high)


def sensitivity_ratio_ci(frame: pd.DataFrame, seed: int, reps: int):
    """Probability-weighted PT ratio, station-cluster bootstrap."""
    weights = frame["Scenario_Probability"].to_numpy(float)
    denominator = float(np.sum(weights * frame["Passenger_Demand"].to_numpy(float)))
    if denominator <= 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(np.sum(weights * frame["PT_Reduction_Passenger_Minutes"].to_numpy(float)) / denominator)
    totals = []
    for _, g in frame.groupby("Station", sort=False):
        w = g["Scenario_Probability"].to_numpy(float)
        totals.append((float(np.sum(w * g["PT_Reduction_Passenger_Minutes"].to_numpy(float))),
                       float(np.sum(w * g["Passenger_Demand"].to_numpy(float)))))
    totals = np.asarray(totals)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(totals), size=(reps, len(totals)))
    num, den = totals[index, 0].sum(axis=1), totals[index, 1].sum(axis=1)
    valid = den > 0
    low, high = np.quantile(num[valid] / den[valid], [0.025, 0.975])
    return estimate, float(low), float(high)


def sensitivity_from_ri(ri_file: Path, seed: int, reps: int) -> pd.DataFrame:
    ri = prepare_ri(read_csv(ri_file))
    ri["Case_ID"] = ri["Case_ID"].astype(str).str.upper()
    windows = sorted(ri.loc[ri["Case_ID"].str.startswith("S2_"), "Window"].unique())
    rows = []
    for panel in cfg.get("archive.sensitivity_panels"):
        for position, (text, case) in enumerate(zip(panel["labels"], panel["cases"])):
            for window in windows:
                subset = ri[ri["Case_ID"].eq(case) & ri["Window"].eq(window)]
                if subset.empty:
                    continue
                ri_mean, ri_low, ri_high = sensitivity_mean_ci(subset, "RI_Improvement_pct", seed, reps)
                pt, pt_low, pt_high = sensitivity_ratio_ci(subset, seed, reps)
                rows.append({"panel": panel["caption"], "position": position, "parameter_label": text,
                             "case_id": case, "window": int(window), "scenario_count": len(subset),
                             "affected_scenario_count": int((subset["Passenger_Demand"] > 0).sum()),
                             "ri_improvement_pct": ri_mean, "ci_low": ri_low, "ci_high": ri_high,
                             "pt_reduction_min_per_affected_passenger": pt, "pt_reduction_ci_low": pt_low,
                             "pt_reduction_ci_high": pt_high})
    return pd.DataFrame(rows, columns=SENSITIVITY_KEEP)


def sensitivity_item(arc: Archive, sens: Path | None, ri_bases, seed: int, reps: int) -> None:
    path = find_file(sens, "sensitivity_figure_values_fullod_unweighted.csv", ("summary",))
    if path is None:
        ri_file = pick_ri(ri_bases, {"BASELINE"}, any_prefix="S2_")
        if ri_file is None:
            arc.skip("sensitivity", f"no sensitivity summary or sensitivity RI file in {label(sens) or '(not given)'}")
            return
        table = sensitivity_from_ri(ri_file, seed, reps)
        if table.empty:
            arc.skip("sensitivity", f"no configured sensitivity case in {label(ri_file)}")
            return
        arc.write("sensitivity", table, "Sensitivity summaries per panel, setting and window (Figs. 12-13, Sec. 5.7), "
                  "computed from the case-level RI file", label(ri_file))
        arc.drop("sensitivity", ["Passenger_Demand", "*_Passenger_Minutes"], "demand magnitudes (aggregated away)")
        return
    table = read_csv(path)
    keep = SENSITIVITY_KEEP
    arc.write("sensitivity", table[keep], "Sensitivity summaries per panel, setting and window (Figs. 12-13, Sec. 5.7)",
              label(path.parent.parent))
    arc.drop("sensitivity", [c for c in table.columns if c not in keep],
             "probability_weighted_affected_demand is a demand magnitude; component times and method strings "
             "are not plotted")


def scaling_items(arc: Archive, base: Path | None, scaling: list[Path]) -> None:
    window = int(cfg.get("archive.scaling_window"))
    dirs = []
    for item in scaling:
        if item is None or not Path(item).exists():
            continue
        item = Path(item)
        if (item / "Performance").is_dir():
            dirs.append(item)
        else:
            dirs.extend(sorted(p for p in item.rglob("S1_K*") if (p / "Performance").is_dir()))
    if base is not None and (base / "Performance" / f"FullMILP_{window:02d}_summary.json").is_file():
        dirs.insert(0, base)
    rows, allocations = [], []
    for directory in dirs:
        summary = directory / "Performance" / f"FullMILP_{window:02d}_summary.json"
        alloc = directory / "Allocation" / f"DepotAlloc_{window:02d}.csv"
        if not summary.is_file() or not alloc.is_file():
            continue
        data = read_json(summary)
        row = stage1_summary_row(data, data.get("stats", {}))
        if any(r["K"] == row["K"] for r in rows):
            continue
        rows.append(row)
        table = read_csv(alloc)
        for depot, count in table.iloc[:, :2].itertuples(index=False):
            allocations.append({"K": row["K"], "Depot": str(depot).strip(), "Buses": int(count)})
    if not rows:
        arc.skip("scaling", "no scaling runs found")
        return
    arc.write("scaling", pd.DataFrame(rows).sort_values("K"),
              f"Stage 1 scaling at window {window}: model size, runtime, gap per K (Table 8)",
              "; ".join(label(d) for d in dirs))
    arc.write("scaling_allocations", pd.DataFrame(allocations).sort_values(["K", "Depot"]),
              f"Stage 1 scaling at window {window}: buses per depot per K (Table 8)", "; ".join(label(d) for d in dirs))


def simple_copy(arc: Archive, key: str, path: Path | None, keep: list[str] | None, description: str,
                reason: str = "not printed in the paper") -> None:
    if path is None:
        arc.skip(key, "source file not found")
        return
    table = read_csv(path)
    columns = [c for c in (keep or table.columns) if c in table.columns]
    arc.write(key, table[columns], description, label(path))
    arc.drop(key, [c for c in table.columns if c not in columns], reason)


def routes_item(arc: Archive, path: Path | None) -> None:
    if path is None:
        arc.skip("routes", "route library not found")
        return
    table = read_csv(path)
    rows = []
    for route, group in table.sort_values(["Route", "StopSeq"]).groupby("Route", sort=False):
        first = group.iloc[0]
        rows.append({"Route": route, "BaseLine": first["BaseLine"], "Type": first["Type"],
                     "NumStops": len(group), "StopSequence": "-".join(group["StationCode"].astype(str)),
                     "RouteTime_min": float(first["Route_TotalTime"])})
    arc.write("routes", pd.DataFrame(rows), "Bus route library: stop sequence and one-way time per route (Tables C.1-C.2)",
              label(path))
    arc.drop("routes", ["CumTime_min", "SegTime_min", "OrigLine_TotalTime", "TimeSaving_pct", "DisruptionDemand",
                        "SkippedStations", "OrigLine_NumStops"],
             "segment and cumulative times are entries of the restricted road-time matrix; DisruptionDemand is "
             "OD-derived; the rest is not printed in Tables C.1-C.2")


def corridor_map_item(arc: Archive, path: Path | None):
    if path is None:
        arc.skip("corridor_groups", "station-corridor map not found")
        return None
    table = read_csv(path)
    table = table.rename(columns={"hour": "Window", "station": "Station", "corridor": "Corridor"})
    table["Station"] = table["Station"].astype(str).str.strip().str.upper()
    table["Window"] = pd.to_numeric(table["Window"]).astype(int)
    mapping = table[["Window", "Station", "Corridor"]].drop_duplicates()
    stations = mapping.drop_duplicates(["Station", "Corridor"])[["Station", "Corridor"]].sort_values(["Corridor", "Station"])
    arc.write("corridor_groups", stations, "Public network grouping: station -> corridor group (plot labels)", label(path))
    order = list(cfg.get("archive.corridor_order"))
    labels = list(cfg.get("archive.corridor_labels"))
    present = sorted(stations["Corridor"].unique())
    rows = [{"Corridor": c, "Label": labels[i] if i < len(labels) else c, "Order": i}
            for i, c in enumerate(order) if c in present]
    rows += [{"Corridor": c, "Label": c.replace("_", " ").title(), "Order": len(rows) + j}
             for j, c in enumerate(c for c in present if c not in order)]
    arc.write("corridor_labels", pd.DataFrame(rows), "Corridor display labels and plot order", "config archive.toml")
    return mapping


def data_description_item(arc: Archive) -> None:
    """Fig. 5a: network-wide boarding and alighting per hour of the input OD matrix.

    The OD matrix holds one mean day (no day-to-day records), so no CI is available (CI columns empty)."""
    try:
        od_path = cfg.path("inputs.od_matrix")
    except Exception as exc:
        arc.skip("data_hourly", f"inputs not available ({exc})")
        return
    if not od_path.is_file():
        arc.skip("data_hourly", "OD matrix not found in the data directory")
        return
    od = pd.read_csv(od_path, usecols=["OriginID", "DestinationID", "Hour", "Trips_Hourly"])
    trips = od.groupby("Hour")["Trips_Hourly"].sum()   # each trip boards and alights within its hour
    frame = pd.DataFrame({"Hour": trips.index.astype(int), "Boarding": trips.to_numpy(float),
                          "Alighting": trips.to_numpy(float), "Boarding_CI": np.nan, "Alighting_CI": np.nan})
    arc.write("data_hourly", frame, "Network-wide boarding/alighting per hour of the input OD matrix (Fig. 5a)",
              "inputs.od_matrix summed per hour")
    arc.drop("data_hourly", ["OriginID", "DestinationID", "Trips_Hourly"], "only network-wide hourly totals are kept")


def ap_case_item(arc: Archive, results: Path | None) -> None:
    """Fig. 10a/b: AP trajectories of the support case archive.ap_case (network-wide ratios per minute)."""
    if results is None:
        arc.skip("ap_case", "needs --from-results (Stage 2 decision files of the run)")
        return
    sys.path.insert(0, str(ROOT / "Visualization_Code"))
    from plot_ap_support_case import compute_curves
    try:
        curves = compute_curves(Path(results))
    except FileNotFoundError as exc:
        arc.skip("ap_case", str(exc))
        return
    case = cfg.get("archive.ap_case")
    arc.write("ap_case", curves, "AP per minute of the support case (station, window) for each duration realization: "
              "Normal, Disruption, comparison run, proposed run (Fig. 10a/b, Sec. 5.5)",
              f"{cfg.get('paths.disruptions')}, {cfg.get('paths.' + case['benchmark'])}, "
              f"{cfg.get('paths.stage2_proposed')} (DecisionVars)")
    arc.drop("ap_case", ["Q/L/W passenger states per OD"], "only the network-wide AP ratio per minute is kept")


def data_corridor_item(arc: Archive) -> None:
    """Fig. 5b / Sec. 4.1: daily boarding and alighting per corridor group (needs the restricted OD matrix)."""
    try:
        od_path = cfg.path("inputs.od_matrix")
    except Exception as exc:  # profile whose data directory is not set
        arc.skip("data_corridor", f"inputs not available ({exc})")
        return
    if not od_path.is_file():
        arc.skip("data_corridor", "OD matrix not found in the data directory")
        return
    sys.path.insert(0, str(ROOT / "Paper_Execution_Pipeline"))
    import network_data
    od = pd.read_csv(od_path, usecols=["OriginID", "DestinationID", "Trips_Hourly"])
    for column in ("OriginID", "DestinationID"):
        od[column] = od[column].astype(str).str.strip().str.upper()
    stations = set(od["OriginID"]) | set(od["DestinationID"])
    group = {station: network_data.station_group(station) for station in stations}
    boarding = od.groupby(od["OriginID"].map(group))["Trips_Hourly"].sum()
    alighting = od.groupby(od["DestinationID"].map(group))["Trips_Hourly"].sum()
    frame = pd.DataFrame({"Corridor": boarding.index, "Boarding": boarding.to_numpy(float),
                          "Alighting": alighting.reindex(boarding.index).to_numpy(float)})
    frame["Boarding_share_pct"] = 100.0 * frame["Boarding"] / frame["Boarding"].sum()
    arc.write("data_corridor", frame, "Daily boarding and alighting per corridor group (Fig. 5b, Sec. 4.1)",
              "inputs.od_matrix aggregated to corridor groups (network_data.station_group)")
    arc.drop("data_corridor", ["OriginID", "DestinationID", "Trips_Hourly"],
             "OD-level flows are restricted; only corridor-group totals (as plotted in Fig. 5b) are kept")


# ---------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-results", type=Path, help="results root in the package layout (paths.toml)")
    p.add_argument("--out", type=Path, required=True)
    for name in ("stage1", "stage2-proposed", "stage2-benchmark", "benchmark-comparison", "stage2-corridor-run", "evaluation",
                 "oof-value", "oof-wrong", "oof-cases", "sensitivity", "clustering", "ml-cutoffs", "ml-models",
                 "routes", "corridor-map"):
        p.add_argument(f"--{name}", type=Path)
    p.add_argument("--scaling", type=Path, nargs="*", default=[], help="S1_K* folders or their parent(s)")
    return p.parse_args()


def derive(args) -> None:
    res = args.from_results
    if res is None:
        return
    res = Path(res)
    path = lambda key: res / cfg.get(f"paths.{key}")  # noqa: E731
    defaults = {"stage1": path("stage1"), "stage2_proposed": path("stage2_proposed"),
                "stage2_benchmark": path("stage2_benchmark"), "benchmark_comparison": path("evaluation"),
                "evaluation": path("evaluation"),
                "oof_value": path("oof"), "oof_wrong": path("oof_wrong"), "oof_cases": path("oof_wrong"),
                "sensitivity": path("sensitivity"), "clustering": path("clustering_comparison"),
                "ml_cutoffs": path("ml"), "ml_models": path("ml"),
                "routes": path("routes") / str(cfg.get("routes.library_file", default="")),
                "corridor_map": path("clustering") / "hourly_corridor_station_probability.csv"}
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if not args.scaling:
        args.scaling = [path("scaling")]
    if not Path(args.corridor_map).is_file():          # clustering output elsewhere in the run
        found = sorted(res.rglob("hourly_corridor_station_probability.csv")) if res.is_dir() else []
        args.corridor_map = found[0] if found else args.corridor_map


def main() -> int:
    args = parse_args()
    derive(args)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.csv"):
        stale.unlink()
    arc = Archive(out)
    seed, reps = int(cfg.get("archive.bootstrap_seed")), int(cfg.get("archive.bootstrap_repetitions"))

    corridor_map = corridor_map_item(arc, first_existing(args.corridor_map))

    stage1_items(arc, first_existing(args.stage1))
    stage2_runtime_item(arc, "stage2_runtime", first_existing(args.stage2_proposed), "Stage 2 proposed")
    stage2_runtime_item(arc, "benchmark_runtime", first_existing(args.stage2_benchmark), "Stage 2 fixed benchmark")

    proposed_ri = pick_ri([first_existing(args.stage2_proposed), first_existing(args.evaluation)], {"BASELINE"})
    ri_items(arc, proposed_ri, corridor_map, "stage2_cases", "stage2_corridor", "Stage 2 proposed", seed, reps)
    corridor_run = first_existing(args.stage2_corridor_run)
    if corridor_run is not None:
        ri_items(arc, find_file(corridor_run, "Bridging_RI.csv", ("Evaluation_RI_PT",)), corridor_map,
                 "fig8_cases", "fig8_corridor", "Stage 2 run behind Fig. 8 / Sec. 5.4", seed, reps)
    else:
        arc.skip("fig8_cases", "no --stage2-corridor-run given; figures use the proposed run")
    comparison = pick_ri([first_existing(args.benchmark_comparison), first_existing(args.evaluation),
                          first_existing(args.stage2_benchmark)], set(BENCHMARK_CASES))
    benchmark_items(arc, comparison, corridor_map, seed, reps)

    oof_items(arc, first_existing(args.oof_wrong), first_existing(args.oof_value), first_existing(args.oof_cases))
    sensitivity_item(arc, first_existing(args.sensitivity),
                     [first_existing(args.sensitivity), first_existing(args.evaluation)], seed, reps)
    scaling_items(arc, first_existing(args.stage1), args.scaling)

    simple_copy(arc, "clustering_metrics", find_file(first_existing(args.clustering), "method_k_metrics_summary.csv"),
                None, "Scenario-reduction metrics per method and K, averaged over windows (Fig. 6)")
    simple_copy(arc, "ml_cutoffs", find_file(first_existing(args.ml_cutoffs), "lightgbm_*_threshold_metrics.csv"),
                ["CutoffMinutes", "N", "Long", "Short", "BrierScore", "LogLoss", "ROC_AUC", "PR_AUC", "ECE10",
                 "ObservedRate", "ReferenceBrierScore", "BrierSkillScore"],
                "LightGBM probability metrics per duration cutoff (Table 5)")
    simple_copy(arc, "ml_models", find_file(first_existing(args.ml_models), "model_comparison.csv"),
                ["Model", "N", "Long", "Short", "BrierScore", "LogLoss", "ROC_AUC", "PR_AUC", "ECE10",
                 "ObservedRate", "BrierSkillScore"], "Classifier comparison at the 30-min cutoff (Table 6)")
    simple_copy(arc, "ml_importance",
                find_file(first_existing(args.ml_cutoffs), "grouped_permutation_importance_lightgbm_n*_long30.csv"),
                None, "Grouped permutation importance of LightGBM, mean and fold SD/CI (Fig. 7)")
    routes_item(arc, first_existing(args.routes))
    data_description_item(arc)
    data_corridor_item(arc)
    ap_case_item(arc, args.from_results)

    manifest = pd.DataFrame(arc.rows)
    manifest[["file", "description", "source", "rows", "sha256"]].to_csv(out / "MANIFEST.csv", index=False,
                                                                       lineterminator="\n")
    write_readme(out, manifest, arc)
    print(f"archive: {len(arc.rows)} files in {out.name}; {len(arc.skipped)} items skipped")
    return 0


def write_readme(out: Path, manifest: pd.DataFrame, arc: Archive) -> None:
    lines = ["# Result archive", "",
             "Aggregated results behind the tables and figures of the manuscript. Built by",
             "`scripts/build_archive.py`; read by `scripts/make_paper_figures.py`. Profile: "
             f"`{cfg.name}`.", "",
             "The archive contains no restricted data: no OD-level flows or passenger counts, no per-incident",
             "records, identifiers, dates or timestamps, no road-time or running-time entries, no file paths.",
             "Case-level values are ratios (RI, minutes per affected passenger) per location-window-duration case.",
             "", "## Files", "", "| file | rows | description | source |", "|---|---:|---|---|"]
    for row in manifest.itertuples(index=False):
        lines.append(f"| `{row.file}` | {row.rows} | {row.description} | {row.source} |")
    lines += ["", "Columns of every file: see the header row; checksums in `MANIFEST.csv`.", "",
              "## Columns dropped from the sources", "", "| file | dropped | reason |", "|---|---|---|"]
    for row in arc.dropped:
        lines.append(f"| `{row['file']}` | {row['columns']} | {row['reason']} |")
    if arc.skipped:
        lines += ["", "## Items not built", ""] + [f"- {s}" for s in arc.skipped]
    (out / "ARCHIVE_INFO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
