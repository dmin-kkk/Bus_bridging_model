"""Evaluate the OOF opposite-state (wrong-information) bus-only stress test.

Sec. 6.6 / Fig. 11: mean RI improvement and PT reduction per affected
passenger under wrong information, the historical prior, LightGBM and perfect
information, with paired incident-bootstrap 95% confidence intervals.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
PIPELINE = SCRIPT.parents[2]
PROJECT = PIPELINE.parent
sys.path.insert(0, str(PROJECT))

from bbconfig import cfg  # noqa: E402

VISUALIZATION = PROJECT / "Visualization_Code"
EVALUATOR = PIPELINE / "Sensitivity" / "source" / "evaluate_ri.py"
RESULTS = cfg.results_dir
DEFAULT_OOF_RESULTS = RESULTS / cfg.get("paths.oof")
# OOF case table and solves; re-pointed to --oof-results-root in main().
CASES_FILE = DEFAULT_OOF_RESULTS / "inputs" / "oof_incident_cases.csv"
LEGACY_EVALUATION = (
    DEFAULT_OOF_RESULTS / "evaluation" / "incident_policy_results.csv"
)
HISTORICAL_ORACLE_ROOT = DEFAULT_OOF_RESULTS / "solves"
LEGACY_ML_ROOT = DEFAULT_OOF_RESULTS / "solves" / "ML"
DEFAULT_RESULTS = RESULTS / cfg.get("paths.oof_wrong")
DEFAULT_NORMAL_DIR = RESULTS / cfg.get("paths.normal")
DEFAULT_DISRUPTION_DIR = RESULTS / cfg.get("paths.disruptions")
RI_CACHE_DIR = RESULTS / cfg.get("paths.cache") / "ri"
OD_FILE = cfg.path("inputs.od_matrix")
FINAL_FIGURES = RESULTS / cfg.get("paths.figures")
SHORT, LONG = (int(value) for value in cfg.get("time.durations_min"))
P_NEAR = float(cfg.get("evaluation.oof.wrong_p_near"))
P_FAR = float(cfg.get("evaluation.oof.wrong_p_far"))
# Output labels of the four information settings (Policy column, Fig. 11 order).
POLICIES = (
    "Wrong information",
    "Historical",
    "LightGBM",
    "Perfect information",
)
POLICY_COLORS = {
    "Wrong information": "#CC79A7",
    "Historical": "#7A7A7A",
    "LightGBM": "#2878B5",
    "Perfect information": "#E07A2D",
}
EVALUATION_START = int(cfg.get("time.evaluation_start_min"))
EVALUATION_END = int(cfg.get("time.horizon_min"))
BOOTSTRAP_SAMPLES = int(cfg.get("evaluation.oof.bootstrap_samples"))
BOOTSTRAP_SEED = int(cfg.get("evaluation.oof.bootstrap_seed"))


def load_module(name: str, path: Path):
    pipeline_path = str(PIPELINE)
    if pipeline_path not in sys.path:
        sys.path.insert(0, pipeline_path)
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def load_style() -> None:
    path = str(VISUALIZATION)
    if path not in sys.path:
        sys.path.insert(0, path)
    from plot_style import apply_global_style

    apply_global_style()


def as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def decision_path(
    results_root: Path,
    oof_results_root: Path | None,
    row: pd.Series,
    policy: str,
    policy_map: dict[int, str],
) -> Path:
    incident_id = int(row["IncidentID"])
    hour = int(row["Hour"])
    station = str(row["Station"])
    state = int(row["RealizedState"])
    filename = f"Disrupt_{hour:02d}_{station}_{state}.csv"
    if policy == "Historical":
        if oof_results_root is not None:
            return (
                oof_results_root / "solves" / "Historical" /
                str(row["HistoricalJobID"]) / "DecisionVars" / filename
            )
        return (
            HISTORICAL_ORACLE_ROOT / "Historical" / str(row["HistoricalJobID"])
            / "DecisionVars" / filename
        )
    if policy == "Perfect information":
        if oof_results_root is not None:
            return (
                oof_results_root / "solves" / "Oracle" /
                str(row["OracleJobID"]) / "DecisionVars" / filename
            )
        return (
            HISTORICAL_ORACLE_ROOT / "Oracle" / str(row["OracleJobID"])
            / "DecisionVars" / filename
        )
    if policy == "LightGBM":
        if oof_results_root is not None:
            return (
                oof_results_root / "solves" / "ML" /
                str(row["MLJobID"]) / "DecisionVars" / filename
            )
        legacy = LEGACY_ML_ROOT / str(row["MLJobID"]) / "DecisionVars" / filename
        if legacy.is_file():
            return legacy
        return (
            results_root / "solves" / "MLCompletion" / str(row["MLJobID"])
            / "DecisionVars" / filename
        )
    job_id = policy_map[incident_id]
    return (
        results_root / "solves" / "ExtremeWrong" / job_id
        / "DecisionVars" / filename
    )


def bootstrap_summary(rows: pd.DataFrame) -> pd.DataFrame:
    ordered = rows.sort_values(["IncidentID", "Policy"])
    incident_ids = sorted(ordered["IncidentID"].unique())
    if not incident_ids:
        raise AssertionError("bootstrap requires at least one incident")
    pivot_ri = ordered.pivot(
        index="IncidentID", columns="Policy", values="RI_Improvement"
    ).reindex(index=incident_ids, columns=POLICIES)
    pivot_num = ordered.pivot(
        index="IncidentID", columns="Policy", values="PT_Reduction_Passenger_Minutes"
    ).reindex(index=incident_ids, columns=POLICIES)
    demand = (
        ordered.drop_duplicates("IncidentID")
        .set_index("IncidentID")["Passenger_Demand"]
        .reindex(incident_ids)
    )
    if pivot_ri.isna().any().any() or pivot_num.isna().any().any() or demand.isna().any():
        raise AssertionError("policy panel is incomplete")
    ri_values = pivot_ri.to_numpy(dtype=float)
    pt_numerators = pivot_num.to_numpy(dtype=float)
    demand_values = demand.to_numpy(dtype=float)
    points_ri = 100.0 * ri_values.mean(axis=0)
    points_pt = pt_numerators.sum(axis=0) / demand_values.sum()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_ri = np.empty((BOOTSTRAP_SAMPLES, len(POLICIES)), dtype=float)
    boot_pt = np.empty_like(boot_ri)
    for index in range(BOOTSTRAP_SAMPLES):
        sample = rng.integers(0, len(incident_ids), len(incident_ids))
        boot_ri[index] = 100.0 * ri_values[sample].mean(axis=0)
        denominator = demand_values[sample].sum()
        boot_pt[index] = pt_numerators[sample].sum(axis=0) / denominator
    records = []
    # 2.5% and 97.5% quantiles: 95% CI (Fig. 11).
    for column, policy in enumerate(POLICIES):
        records.extend((
            {
                "Policy": policy,
                "Metric": "RI improvement",
                "Value": points_ri[column],
                "CI_Lower": float(np.quantile(boot_ri[:, column], 0.025)),
                "CI_Upper": float(np.quantile(boot_ri[:, column], 0.975)),
                "Unit": "percentage points",
            },
            {
                "Policy": policy,
                "Metric": "PT reduction",
                "Value": points_pt[column],
                "CI_Lower": float(np.quantile(boot_pt[:, column], 0.025)),
                "CI_Upper": float(np.quantile(boot_pt[:, column], 0.975)),
                "Unit": "min/affected passenger",
            },
        ))
    return pd.DataFrame(records)


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    stem: str,
    output_dir: Path,
) -> None:
    selected = summary.loc[summary["Metric"].eq(metric)].set_index("Policy").loc[
        list(POLICIES)
    ]
    load_style()
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    x = np.arange(len(POLICIES))
    values = selected["Value"].to_numpy(dtype=float)
    ci_upper = selected["CI_Upper"].to_numpy(dtype=float)
    lower = values - selected["CI_Lower"].to_numpy(dtype=float)
    upper = ci_upper - values
    bars = axis.bar(
        x,
        values,
        width=0.62,
        color=[POLICY_COLORS[policy] for policy in POLICIES],
        edgecolor="black",
        linewidth=0.7,
        yerr=np.vstack([lower, upper]),
        error_kw={"ecolor": "black", "elinewidth": 1.0, "capsize": 4},
        zorder=3,
    )
    axis.set_xticks(x, POLICIES)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    offset = max(float(ci_upper.max()), 1.0) * 0.018
    for bar, value, upper_bound in zip(bars, values, ci_upper):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            upper_bound + offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )
    axis.set_ylim(
        bottom=min(0.0, float(values.min()) - offset * 3.0),
        top=float(ci_upper.max()) + offset * 3.5,
    )
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg.get("archive.figure_dpi"))
    figure.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    final_names = {
        "oof_ri_improvement_extreme_wrong": (
            "OOF_average_RI.png"
        ),
        "oof_passenger_time_reduction_extreme_wrong": (
            "OOF_average_PT.png"
        ),
    }
    final_name = final_names.get(stem)
    if final_name is not None:
        FINAL_FIGURES.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            FINAL_FIGURES / final_name, dpi=dpi, bbox_inches="tight"
        )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--oof-results-root", type=Path,
        help=(
            "fresh OOF root supplying the case table and the Historical, "
            "LightGBM, and Oracle policies; when omitted, the OOF stage output "
            "of the profile is read in the legacy layout and the RI of the "
            "three shared policies is checked against its evaluation"
        ))
    parser.add_argument(
        "--evaluation-dir", type=Path,
        help="output folder (default: <results-root>/evaluation)")
    parser.add_argument("--normal-dir", type=Path, default=DEFAULT_NORMAL_DIR)
    parser.add_argument(
        "--disruption-dir", type=Path, default=DEFAULT_DISRUPTION_DIR)
    parser.add_argument(
        "--no-condition-cache", action="store_true",
        help="read the CSV outputs directly instead of the RI condition cache")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    global CASES_FILE
    args = parse_args()
    root = args.results_root.resolve()
    oof_root = (
        args.oof_results_root.resolve()
        if args.oof_results_root is not None else None
    )
    if oof_root is not None:
        CASES_FILE = oof_root / "inputs" / "oof_incident_cases.csv"
    evaluation_dir = (
        args.evaluation_dir.resolve()
        if args.evaluation_dir is not None else root / "evaluation"
    )
    normal_dir = args.normal_dir.resolve()
    disruption_dir = args.disruption_dir.resolve()
    progress_path = root / "BatchStatus" / "batch_progress.json"
    policy_map_path = root / "incident_policy_map.csv"
    prerequisites = [CASES_FILE, progress_path, policy_map_path]
    oof_progress_path = None
    if oof_root is None:
        prerequisites.append(LEGACY_EVALUATION)
    else:
        oof_progress_path = oof_root / "BatchStatus" / "batch_progress.json"
        prerequisites.append(oof_progress_path)
    missing = [str(path) for path in prerequisites if not path.is_file()]
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file() else {}
    )
    oof_progress = (
        json.loads(oof_progress_path.read_text(encoding="utf-8"))
        if oof_progress_path is not None and oof_progress_path.is_file()
        else {}
    )
    ready = bool(
        not missing and progress.get("state") == "COMPLETE"
        and (
            oof_root is None
            or oof_progress.get("state") in {"COMPLETE", "FINISHED"}
        )
    )
    check = {
        "schema": "oof-opposite-state-evaluation-preflight-v1",
        "ready": ready,
        "batch_state": progress.get("state"),
        "completed_jobs": progress.get("completed_jobs"),
        "total_jobs": progress.get("total_jobs"),
        "oof_results_root": str(oof_root) if oof_root is not None else None,
        "oof_batch_state": oof_progress.get("state") if oof_root is not None else None,
        "missing_inputs": missing,
    }
    if args.check_only:
        print(json.dumps(check, ensure_ascii=False, indent=2), flush=True)
        return 0 if ready else 1
    if not ready:
        raise RuntimeError(f"evaluation is not ready: {check}")

    evaluator = load_module("oof_extreme_wrong_evaluator", EVALUATOR)
    # The RI/PT helpers read these module paths.
    evaluator.NORMAL_DIR = normal_dir
    evaluator.OD_FILE = OD_FILE
    evaluator.CONDITION_CACHE_DIR = RI_CACHE_DIR
    if args.no_condition_cache:
        evaluator.CONDITION_CACHE_ENABLED = False
    cases = pd.read_csv(CASES_FILE).sort_values("IncidentID")
    # Incidents inside a planning window (paper: all incidents).
    cases = cases.loc[cases["Eligible"].map(as_bool)].copy()
    zero_impact_mask = cases["ZeroImpactRealized"].map(as_bool)
    zero_impact_excluded = int(zero_impact_mask.sum())
    evaluation_cases = cases.loc[~zero_impact_mask].copy()
    mapping_frame = pd.read_csv(policy_map_path)
    policy_map = dict(zip(
        mapping_frame["IncidentID"].astype(int),
        mapping_frame["ExtremeWrongJobID"].astype(str),
    ))
    if len(policy_map) != len(cases) or set(policy_map) != set(
            cases["IncidentID"].astype(int)):
        raise AssertionError(
            f"policy map covers {len(policy_map)} incidents, "
            f"case table has {len(cases)} eligible incidents")
    legacy_lookup = (
        pd.read_csv(LEGACY_EVALUATION).set_index(["IncidentID", "Policy"])
        if oof_root is None else None
    )
    normal_cache = {
        int(hour): evaluator.load_normal(int(hour))
        for hour in sorted(cases["Hour"].astype(int).unique())
    }
    scenario_cache: dict[tuple[int, str, int], dict] = {}
    condition_cache = {}
    evaluation_index = pd.Index(range(EVALUATION_START, EVALUATION_END + 1), name="Time")
    output_rows = []
    legacy_max_ri_error = 0.0

    for _, case in evaluation_cases.iterrows():
        incident_id = int(case["IncidentID"])
        hour = int(case["Hour"])
        station = str(case["Station"])
        state = int(case["RealizedState"])
        zero_impact = as_bool(case["ZeroImpactRealized"])
        scenario_key = (hour, station, state)
        if scenario_key not in scenario_cache:
            normal = normal_cache[hour]
            disruption_path = (
                disruption_dir / f"Disrupt_{hour}_{station}_{state}.csv"
            )
            if not zero_impact:
                if not disruption_path.is_file():
                    raise FileNotFoundError(disruption_path)
                disruption = evaluator.read_condition(disruption_path)
                affected = disruption.od_set
                unaffected = evaluator.normal_unaffected_qlw(normal, affected)
                disrupted_qlw = disruption.total_qlw + unaffected
                disrupted_state = disruption.state_by_variable_time
            else:
                disruption = None
                affected = frozenset()
                unaffected = normal.total_qlw.copy()
                disrupted_qlw = normal.total_qlw.copy()
                disrupted_state = normal.total_state_by_variable_time.iloc[0:0].copy()
            demand = float(normal.demand_by_od.reindex(sorted(affected), fill_value=0.0).sum())
            disruption_metrics = evaluator.passenger_time_metrics(
                disrupted_state, demand, evaluation_index
            )
            scenario_cache[scenario_key] = {
                "normal": normal,
                "affected": affected,
                "unaffected": unaffected,
                "demand": demand,
                "disruption_ri": evaluator.resilience_index(
                    normal.total_qlw, disrupted_qlw, EVALUATION_START
                ),
                "disruption_metrics": disruption_metrics,
            }
        scenario = scenario_cache[scenario_key]

        for policy in POLICIES:
            path = decision_path(root, oof_root, case, policy, policy_map)
            if policy == "Historical":
                job_id = str(case["HistoricalJobID"])
                p30, p60 = float(case["HistoricalP30"]), float(case["HistoricalP60"])
            elif policy == "LightGBM":
                job_id = str(case["MLJobID"])
                p30, p60 = float(case["MLP30"]), float(case["MLP60"])
            elif policy == "Perfect information":
                job_id = str(case["OracleJobID"])
                p30, p60 = (1.0, 0.0) if state == SHORT else (0.0, 1.0)
            else:
                job_id = policy_map[incident_id]
                p30, p60 = (P_FAR, P_NEAR) if state == SHORT else (P_NEAR, P_FAR)
            if not scenario["affected"]:
                ri = 1.0
                metrics = evaluator.passenger_time_metrics(
                    scenario["normal"].total_state_by_variable_time.iloc[0:0].copy(),
                    0.0,
                    evaluation_index,
                )
            else:
                if not path.is_file():
                    raise FileNotFoundError(path)
                key = str(path.resolve())
                if key not in condition_cache:
                    condition_cache[key] = evaluator.read_condition(path)
                bridging = condition_cache[key]
                if bridging.od_set != scenario["affected"]:
                    raise AssertionError(
                        f"{incident_id} {policy}: decision support differs from physical support"
                    )
                ri = evaluator.resilience_index(
                    scenario["normal"].total_qlw,
                    bridging.total_qlw + scenario["unaffected"],
                    EVALUATION_START,
                )
                metrics = evaluator.passenger_time_metrics(
                    bridging.state_by_variable_time,
                    scenario["demand"],
                    evaluation_index,
                )
            disruption_total = float(
                scenario["disruption_metrics"]["Total_Travel_Passenger_Minutes"]
            )
            policy_total = float(metrics["Total_Travel_Passenger_Minutes"])
            row = {
                "IncidentID": incident_id,
                "Hour": hour,
                "Station": station,
                "StationCorridor": case["StationCorridor"],
                "IncidentType": case["IncidentType"],
                "ActualDurationMinutes": float(case["ActualDurationMinutes"]),
                "RealizedState": state,
                "Policy": policy,
                "JobID": job_id,
                "InputP30": p30,
                "InputP60": p60,
                "RI_Disruption": scenario["disruption_ri"],
                "RI": ri,
                "RI_Improvement": ri - scenario["disruption_ri"],
                "Passenger_Demand": scenario["demand"],
                "PT_Reduction_Passenger_Minutes": disruption_total - policy_total,
                "DecisionFile": str(path.resolve()) if path.is_file() else None,
            }
            output_rows.append(row)
            legacy_policy = {
                "Historical": "Historical",
                "LightGBM": "ML",
                "Perfect information": "Oracle",
            }.get(policy)
            lookup_key = (incident_id, legacy_policy)
            if (legacy_lookup is not None and legacy_policy
                    and lookup_key in legacy_lookup.index):
                legacy_ri = float(legacy_lookup.loc[lookup_key, "RI"])
                legacy_max_ri_error = max(legacy_max_ri_error, abs(ri - legacy_ri))

    result = pd.DataFrame(output_rows).sort_values(["IncidentID", "Policy"])
    expected_rows = len(evaluation_cases) * 4
    if len(result) != expected_rows:
        raise AssertionError(f"expected {expected_rows} rows, found {len(result)}")
    if result.groupby("IncidentID")["Policy"].nunique().ne(4).any():
        raise AssertionError("not every incident has four policies")
    if legacy_lookup is not None and legacy_max_ri_error > 1e-9:
        raise AssertionError(f"legacy RI reproduction error: {legacy_max_ri_error}")

    figures_dir = evaluation_dir / "figures"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(evaluation_dir / "incident_policy_results_extreme_wrong.csv", index=False)
    summary = bootstrap_summary(result)
    summary.to_csv(evaluation_dir / "policy_summary_extreme_wrong.csv", index=False)
    ri_values = summary.loc[
        summary["Metric"].eq("RI improvement")
    ].set_index("Policy")["Value"]
    pt_values = summary.loc[summary["Metric"].eq("PT reduction")].set_index("Policy")["Value"]
    ri_order = bool(np.all(np.diff(ri_values.loc[list(POLICIES)].to_numpy()) > 0.0))
    pt_order = bool(np.all(np.diff(pt_values.loc[list(POLICIES)].to_numpy()) > 0.0))
    audit = {
        "schema": "oof-opposite-state-bus-only-evaluation-v1",
        "input_incidents": len(cases),
        "evaluated_incidents": len(evaluation_cases),
        "zero_impact_incidents_excluded": zero_impact_excluded,
        "policy_rows": len(result),
        "policies": list(POLICIES),
        "requested_policy_order": (
            "Wrong information < Historical < LightGBM < Perfect information"
        ),
        "requested_order_is_execution_gate": False,
        "incident_weighting": "uniform",
        "ri_estimand": "100 * mean incident absolute RI improvement",
        "pt_estimand": "sum passenger-minute reduction / sum affected demand",
        "probability_weighting": "none",
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "resampling_unit": "paired IncidentID",
        },
        "legacy_max_ri_reproduction_error": legacy_max_ri_error,
        "legacy_reproduction_gate_applied": legacy_lookup is not None,
        "oof_results_root": str(oof_root) if oof_root is not None else None,
        "requested_ri_order_holds": ri_order,
        "requested_pt_order_holds": pt_order,
    }
    (evaluation_dir / "evaluation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_metric(
        summary,
        "RI improvement",
        "RI improvement (%)",
        "oof_ri_improvement_extreme_wrong",
        figures_dir,
    )
    plot_metric(
        summary,
        "PT reduction",
        "PT reduction (min/affected passenger)",
        "oof_passenger_time_reduction_extreme_wrong",
        figures_dir,
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
