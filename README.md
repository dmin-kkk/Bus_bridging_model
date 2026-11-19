# Reserve bus positioning and bus bridging for regional rail

Reproducibility package for a two-stage stochastic optimisation framework that positions a
reserve bus fleet at depots before rail disruptions occur (Stage 1) and dispatches bridging
buses once an incident occurs (Stage 2). Incident-duration probabilities come from a
classifier trained on historical incidents.

The package runs every stage of the pipeline on a synthetic network with the same file
structure as the original case-study data. No original operational data is included.

## Contents

| Path | Content |
|---|---|
| `run_all.py` | single entry point: runs all stages in order |
| `config/profiles/synthetic/` | every model, solver and data setting, one commented key per value |
| `bbconfig/` | loader for the configuration profile |
| `Paper_Execution_Pipeline/` | stage scripts `00`–`08` and the model builders in `_core/`, sensitivity and scaling runners in `Sensitivity/source/`, out-of-fold experiments in `experiments/` |
| `data/synthetic/` | synthetic 12-station, two-corridor network: timetable, OD demand, road travel times, depot candidates, incident records, weather |
| `scripts/make_synthetic_inputs.py` | seeded generator of `data/synthetic/` |
| `models/synthetic/` | Stage 1 and Stage 2 MILP instances (MPS) with solutions and solver summaries, trained duration classifier |
| `scripts/build_archive.py`, `scripts/make_paper_figures.py` | aggregate the results and draw the figures and tables |
| `docs/TABLE_FIGURE_MAP.md` | each table and figure of the article → script → input and output files |
| `docs/MODEL_CODE_MAP.md` | each set, parameter, variable and equation of the formulation → code location |
| `synthetic_results/` | figures, tables and run log of a complete synthetic run (see below) |
| `tests/` | quick checks (`python -m unittest discover -s tests`) |

## Results without running the code

`synthetic_results/` holds the output of one complete run of `python run_all.py` on the synthetic
network, so the results can be inspected before anything is executed:

* `figures/` — the article's figures (Figs. 5–13) regenerated from the synthetic run, with
  `figure_status.csv`;
* `tables/` — the article's tables (Tables 4–8, C.1–C.2) and summary numbers, as CSV and LaTeX;
* `run_status.json` — the runtime of each of the 27 pipeline tasks.

The numbers differ from the article because the network and demand are synthetic; the figures
and tables have the same structure and are produced by the same scripts.

## Requirements

* Python 3.11 or newer (tested with 3.13).
* `pip install -r requirements.txt`
* Gurobi 12 with a valid licence (free academic licences are available from Gurobi).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python run_all.py                 # full pipeline on the synthetic network (about 70 minutes)
python -m unittest discover -s tests
```

Results are written to `results/synthetic/`: one folder per stage, logs in `logs/`, the figures
and tables in `figures/`, and the run status in `run_status.json`.

* `python run_all.py --dry-run` prints the plan.
* `--from-stage N --to-stage M` and `--only <task>` run part of it.
* `BB_RESULTS_DIR` sets another output folder.

## Pipeline

| Stage | Script | Output |
|---|---|---|
| 0 | `00_normal_scenarios.py` | passenger flows without disruption |
| 1 | `01_disruption_scenarios.py` | passenger flows under each disruption scenario |
| 2 | `02_scenario_reduction.py`, `02_reduction_method_comparison.py` | representative incident locations per window |
| 3 | `03_bus_routes.py` | local and express bridging-route library |
| 4 | `04_stage1_positioning.py` | depot allocation of the reserve fleet per window |
| 5 | `05_duration_*.py` | duration classifiers and probabilities per location and window |
| 6 | `06_stage2_dispatch.py` | bus-bridging dispatch per location and window |
| 7 | `06_stage2_dispatch.py --benchmark` | dispatch from a fixed centralised allocation |
| 8 | `08_oof_evaluation.py`, `experiments/` | out-of-fold and wrong-information evaluation |
| 9–10 | `Sensitivity/source/` | sensitivity analysis and representative-location scaling |
| 11 | `Sensitivity/source/evaluate_ri.py`, `scripts/` | resilience metrics, figures and tables |

## Reproducibility

The model builders iterate over sets of identifiers, so the order of constraints follows
Python's string-hash seed. `run_all.py` fixes `PYTHONHASHSEED` (`scenarios.python_hash_seed`)
for every task. With the same seed, thread count and Gurobi version, reruns give identical
results.

## Data

The case study uses operational data of a regional rail operator: incident records, running
records and an origin–destination demand matrix, plus a road travel-time matrix. These data are
not redistributed. The synthetic dataset reproduces their file structure so that every stage
can be executed.
