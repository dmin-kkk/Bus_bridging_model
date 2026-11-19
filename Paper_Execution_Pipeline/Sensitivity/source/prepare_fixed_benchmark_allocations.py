"""Create audited allocations that place the whole reserve fleet at one benchmark depot.

The fixed centralized benchmark (Sec. 5.2) keeps the Stage 2 formulation and trip pool
and replaces the Stage 1 positioning by ``evaluation.benchmark_fleet`` buses at
``evaluation.benchmark_depot`` in every window of ``evaluation.benchmark_windows``.
Output: ``DepotAlloc_HH.csv`` files under ``cfg.results_dir / paths.benchmark_allocations``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bbconfig import cfg  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depot", default=str(cfg.get("evaluation.benchmark_depot")))
    parser.add_argument("--fleet", type=int, default=int(cfg.get("evaluation.benchmark_fleet")))
    parser.add_argument("--windows", nargs="+", type=int,
                        default=[int(w) for w in cfg.get("evaluation.benchmark_windows")])
    parser.add_argument("--depot-file", type=Path, default=cfg.path("inputs.depots"))
    parser.add_argument("--output-dir", type=Path,
                        default=cfg.results_dir / cfg.get("paths.benchmark_allocations"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    depot = str(args.depot).strip().upper()
    depots = pd.read_csv(args.depot_file, usecols=["Code"])
    codes = sorted(
        {
            str(value).strip().upper()
            for value in depots["Code"]
            if str(value).strip()
        }
    )
    if depot not in codes:
        raise ValueError(f"{depot} is not in {args.depot_file}")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(
        {
            "Depot": codes,
            "Count": [args.fleet if code == depot else 0 for code in codes],
        }
    )
    if int(rows["Count"].sum()) != args.fleet:
        raise AssertionError("fixed allocation does not preserve fleet size")
    files = []
    for window in args.windows:
        path = output / f"DepotAlloc_{window:02d}.csv"
        temporary = path.with_name(path.name + ".tmp")
        rows.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
        files.append(path.name)
    audit = {
        "schema": "fixed-benchmark-allocation-v1",
        "case_label": f"FIXED_{depot}{args.fleet}",
        "fixed_depot": depot,
        "fleet_size": args.fleet,
        "windows": list(args.windows),
        "all_other_depots": 0,
        "depot_candidate_file": args.depot_file.name,
        "allocation_files": files,
    }
    audit_path = output / "allocation_audit.json"
    temporary = audit_path.with_name(audit_path.name + ".tmp")
    temporary.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, audit_path)
    print(f"FIXED_BENCHMARK_ALLOCATIONS_COMPLETE label=FIXED_{depot}{args.fleet} "
          f"files={len(files)} output={output}")


if __name__ == "__main__":
    main()
