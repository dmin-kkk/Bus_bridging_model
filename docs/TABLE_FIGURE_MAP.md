# Table and figure map

Every table and figure of the manuscript mapped to the script that produces it in this package and
to the synthetic input/output behind it.

**The package is synthetic-only.** Every script-generated item is produced from the synthetic dataset
(`data/synthetic/`) by `run_all.py --profile synthetic`. The values therefore differ from the
manuscript; only the form of each table and figure is reproduced. Nothing derived from the restricted
data (passenger records, incident records, real network case results) is distributed, and no pre-built
archive is shipped.

## How the outputs are produced

```
results/synthetic/<stage folders> ──scripts/build_archive.py──> results/synthetic/archive/*.csv
                                   ──scripts/make_paper_figures.py──> results/synthetic/figures/ (+ tables/)
```

* `run_all.py --profile synthetic`, stage 11:
  * `scripts/build_archive.py --from-results results/synthetic --out results/synthetic/archive`
    aggregates the run's result folders (aggregated values only; `archive/MANIFEST.csv` lists rows,
    checksums and the source folder of each file).
  * `scripts/make_paper_figures.py` reads only the archive (default `cfg.results_dir / archive`) and
    writes PNG figures to `results/synthetic/figures/` and tables (CSV, plus LaTeX `.tex`) to
    `results/synthetic/figures/tables/`. Missing archive files are skipped;
    `figures/figure_status.csv` lists the outcome of every item.
* `render_results.py` runs the same two steps (`--from-results <dir>` rebuilds the archive first).
* Archive/output file names, the Table 7 cases, the Fig. 10 support case and the sensitivity panels
  are set in `config/profiles/synthetic/archive.toml`.
* Function names refer to `scripts/make_paper_figures.py`. Stage folders are those of
  `config/profiles/synthetic/paths.toml` below `results/synthetic/`.

Status: **synthetic** = produced from the synthetic run by stage 11; **synthetic (standalone)** =
produced from the synthetic run by a separate script (not part of stage 11); **conceptual** =
conceptual illustration, not script-generated; **typed** = table typed in the manuscript
(definitions, settings, literature), not script-generated; **not produced** = reason given.

## Map

| Item | Status | Script / function | Synthetic input (below `results/synthetic/`) | Output (below `results/synthetic/figures/`) |
|---|---|---|---|---|
| Fig. 1 | conceptual | conceptual illustration, not script-generated | - | - |
| Table 1 | typed | literature comparison, not script-generated | - | - |
| Fig. 2 | conceptual | conceptual illustration, not script-generated | - | - |
| Fig. 3 | conceptual | conceptual illustration (flow diagram), not script-generated | - | - |
| Table 2 | typed | feature list, not script-generated (features built in `Paper_Execution_Pipeline/05_*.py`) | - | - |
| Fig. 4a | conceptual | hand-made GIS map of the real network, not script-generated | - | - |
| Fig. 4b | conceptual | hand-made GIS map of the real network, not script-generated | - | - |
| Fig. 5a | synthetic | `fig5a` | `data/synthetic/od_matrix_hourly.csv` summed per hour (`archive/data_hourly_boarding_alighting.csv`) | `fig5a_hourly_boarding_alighting.png` |
| Fig. 5b | synthetic | `fig5b` | `data/synthetic/od_matrix_hourly.csv` summed per corridor group (`archive/data_corridor_boarding_alighting.csv`) | `fig5b_corridor_boarding_alighting.png` |
| Fig. 5c | not produced | - | - | - (no plotting script in the package; the published panel describes the restricted incident records) |
| Fig. 5d | not produced | - | - | - (as Fig. 5c) |
| Table 3 | typed | settings table; the values used are in `config/profiles/*/*.toml` and `docs/MODEL_CODE_MAP.md` | - | - |
| Fig. 6 | synthetic | `fig6` | `02_clustering_comparison/method_k_metrics_summary.csv` | `fig6_clustering_metrics.png` |
| Table 4 | synthetic | `table4` | `04_stage1/Allocation/DepotAlloc_*.csv` | `tables/table4_depot_allocation.{csv,tex}` |
| Sec. 5.2 text | synthetic (bus counts only) | `relocation_table` | as Table 4 | `tables/relocation_bus_counts.csv` (relocation travel times are not computed) |
| Table 5 | synthetic | `table5` | `05_ml/lightgbm_cutoffs_permutation/lightgbm_*_threshold_metrics.csv` | `tables/table5_cutoff_metrics.{csv,tex}` |
| Table 6 | synthetic | `table6` | `05_ml/long30_optuna/model_comparison.csv` | `tables/table6_classifier_comparison.{csv,tex}` |
| Fig. 7 | synthetic | `fig7` | `05_ml/lightgbm_cutoffs_permutation/grouped_permutation_importance_lightgbm_n*_long30.csv` | `fig7_grouped_permutation_importance.png` |
| Fig. 8a | synthetic | `fig8_corridor` | `11_evaluation/proposed/Bridging_RI.csv` + corridor map `02_clustering*/hourly_corridor_station_probability.csv` | `fig8a_corridor_ri.png` |
| Fig. 8b | synthetic | `fig8_corridor` | as Fig. 8a | `fig8b_corridor_pt.png` |
| Fig. 8c | synthetic | `fig8c` | as Fig. 8a | `fig8c_top10_station_ri.png` |
| Sec. 5.4 text | synthetic | `text_numbers` | as Fig. 8a | `tables/text_numbers.csv` |
| Fig. 9a | synthetic | `fig9` | `11_evaluation/benchmark_comparison/Bridging_RI.csv` (cases `PROPOSED`, `FIXED_<evaluation.benchmark_depot>`) | `fig9a_hourly_ri_proposed_vs_fixed.png` |
| Fig. 9b | synthetic | `fig9` | as Fig. 9a | `fig9b_hourly_pt_proposed_vs_fixed.png` |
| Sec. 5.5 text | synthetic | `text_numbers` | as Fig. 9a | `tables/text_numbers.csv` |
| Fig. 10a | synthetic | `fig10` -> `Visualization_Code/plot_ap_support_case.py` | `01_disruptions/`, `06_stage2_proposed/DecisionVars/`, `07_stage2_fixed_benchmark/DecisionVars/` for `archive.ap_case` (station, window) -> `archive/stage2_ap_support_case.csv` | `fig10a_ap_support_case_short.png` |
| Fig. 10b | synthetic | as Fig. 10a | as Fig. 10a | `fig10b_ap_support_case_long.png` |
| Sec. 5.5 text | synthetic | `ap_case_ri` | as Fig. 10a | `tables/fig10_ap_support_case_ri.{csv,tex}` |
| Sec. 5.5 text | not produced | - | - | - (taken from the Table D schedules; no script in the package) |
| Fig. 11a | synthetic | `fig11` | `08_oof_wrong/evaluation/policy_summary_*.csv` | `fig11a_oof_ri.png` |
| Fig. 11b | synthetic | `fig11` | as Fig. 11a | `fig11b_oof_pt.png` |
| Sec. 5.6 text | synthetic | `text_numbers` | as Fig. 11a | `tables/text_numbers.csv` |
| Fig. 11c | synthetic (standalone) | `Visualization_Code/plot_oof_ap_case.py` (first case of `archive.oof_cases`) | `08_oof/solves/`, `08_oof_wrong/solves/`, `01_disruptions/`, `00_normal/` | `fig11_oof_ap_<station>_h<hour>_d<state>.png` (+ `_ri.csv`) |
| Fig. 11d | synthetic (standalone) | `Visualization_Code/plot_oof_ap_case.py --station <code> --hour <h> --state <min>` (second case of `archive.oof_cases`) | as Fig. 11c | as Fig. 11c |
| Table 7 | synthetic | `table7` | `08_oof_wrong/evaluation/incident_policy_results_*.csv`, cases of `archive.oof_cases` | `tables/table7_oof_cases.{csv,tex}` |
| Fig. 12 | synthetic | `fig_sensitivity` | `09_sensitivity/RI_FullOD/Bridging_RI.csv` (variants of `evaluation.sensitivity.cases`) | `fig12_sensitivity_ri.png` |
| Fig. 13 | synthetic | `fig_sensitivity` | as Fig. 12 | `fig13_sensitivity_pt.png` |
| Sec. 5.7 text | synthetic (except beta) | `text_numbers` | as Fig. 12 | `tables/text_numbers.csv`; the location-smoothing (beta) paragraph has no run or script in the package |
| Table 8 | synthetic | `table8` | `04_stage1/` and `10_scaling/h<window>/S1_K*/` (window `archive.scaling_window`) | `tables/table8_stage1_scaling.{csv,tex}` |
| Sec. 5.8 text | synthetic | `runtime_summary` | `04_stage1/Performance/`, `06_stage2_proposed/Performance/`, `07_stage2_fixed_benchmark/Performance/`, `10_scaling/` | `tables/runtime_summary.csv` |
| Table A.1 | typed | metric definitions, not script-generated (computed in `Paper_Execution_Pipeline/05_*.py`) | - | - |
| Table A.2 | typed | search space typed from `Paper_Execution_Pipeline/05_duration_model_comparison.py` | - | - |
| Table C.1 | synthetic | `table_routes` | `03_routes/<routes.library_file>` | `tables/tableC1_local_routes.{csv,tex}` |
| Table C.2 | synthetic | `table_routes` | as Table C.1 | `tables/tableC2_express_routes.{csv,tex}` |
| Table D.1 | not produced | - | - | - (no schedule-table script in the package) |
| Table D.2 | not produced | - | - | - (as Table D.1; also needs a local-route-only Stage 2 run) |

## Notes on the synthetic versions

* **Fig. 5a**: the synthetic OD matrix holds one mean day, so there is no confidence band, and each
  trip boards and alights within its hour (the two curves coincide).
* **Fig. 8**: the archive has no separate "Fig. 8 run"; Fig. 8 uses the proposed run
  (`stage2_proposed_*` archive files).
* **Fig. 10**: the support case is `archive.ap_case` (synthetic: a central station in the evening
  window). The manuscript compares the proposed strategy with a local-route benchmark; the synthetic
  run has no local-route-only Stage 2 run, so the comparison line is the fixed centralized
  benchmark (`archive.ap_case.benchmark = "stage2_benchmark"`). AP(t) is the Normal Q+L+W divided by
  the condition's Q+L+W over the full OD set (ODs outside a file's support restored from the Normal
  window), as in `Paper_Execution_Pipeline/Sensitivity/source/evaluate_ri.py`; the RI of each curve
  (mean over `time.evaluation_start_min`..horizon) equals the evaluation output for the same case.
* **Table 7 / Fig. 11c-d**: two synthetic OOF cases with a long realized duration (`archive.oof_cases`);
  each descriptor triple matches one incident of the synthetic cohort.
* **Figs. 12-13**: one variant per axis, as run by the synthetic profile; tick labels follow the
  synthetic baseline values (`archive.sensitivity_panels`).

## Archive files not used by a figure or table

| file | purpose |
|---|---|
| `benchmark_case_metrics.csv` | Case-level values behind Fig. 9 |
| `oof_value_policy_summary.csv` | Policy summary of the operational-value OOF run (cross-check of Fig. 11a/b) |
| `network_corridor_groups.csv` | Station -> corridor grouping used for the corridor aggregation |
