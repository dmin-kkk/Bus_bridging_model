"""Build the sensitivity inputs (Sec. 6.7) and their provenance manifest.

Everything is written below ``cfg.results_dir / paths.sensitivity / inputs``: the
inputs are derived from restricted data (road-time matrix, OD-based routes) and
therefore never stored in the package.  Windows, travel-time factors and route
pools come from ``[evaluation.sensitivity]`` of the active profile.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve()
PACKAGE = SCRIPT.parents[3]
sys.path.insert(0, str(PACKAGE))

from bbconfig import cfg  # noqa: E402


SENSITIVITY_ROOT = cfg.results_dir / cfg.get("paths.sensitivity")
INPUTS = SENSITIVITY_ROOT / "inputs"
STAGE2_WINDOWS = tuple(int(w) for w in cfg.get("evaluation.sensitivity.windows"))
STAGE1_K_WINDOWS = tuple(int(w) for w in cfg.get("evaluation.sensitivity.stage1_k_windows"))
DURATIONS = tuple(int(d) for d in cfg.get("time.durations_min"))
K_BASELINE = int(cfg.get("clustering.k_per_window"))
FLEET_BASELINE = int(cfg.get("operations.fleet_size"))
CASES = list(cfg.get("evaluation.sensitivity.cases"))


def travel_time_label(multiplier: float) -> str:
    """File label of a road-time multiplier (0.8 -> TT080)."""
    return f"TT{round(100 * float(multiplier)):03d}"


TRAVEL_FACTORS = {
    travel_time_label(case["travel_time_multiplier"]): float(case["travel_time_multiplier"])
    for case in CASES
    if "travel_time_multiplier" in case
}
ROUTE_POOL_CASES = [case["id"] for case in CASES if "route_pool" in case]


def representative_branch_file() -> Path:
    """Stage 1 representative branches of the clustering stage (``clustering.files.branches``).

    With pinned.use_pinned, the clustering stage copies the SHA-256-verified pinned files there.
    """
    name = str(cfg.get("clustering.files.branches")).format(
        k=K_BASELINE, durations="_".join(str(d) for d in DURATIONS))
    return cfg.results_dir / cfg.get("paths.clustering") / name


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def scaled_distance(source: Path, destination: Path, factor: float) -> pd.DataFrame:
    frame = pd.read_csv(source)
    required = {"Origin_Code", "Destination_Code", "Duration_min"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")
    frame["Duration_min"] = (
        pd.to_numeric(frame["Duration_min"], errors="raise") * factor
    ).round(6)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8-sig")
    return frame


def retime_routes(
    route_source: Path,
    distance_frame: pd.DataFrame,
    destination: Path,
) -> None:
    routes = pd.read_csv(route_source)
    required = {
        "Route", "StopSeq", "StationCode", "CumTime_min", "SegTime_min"
    }
    missing = required - set(routes.columns)
    if missing:
        raise ValueError(f"{route_source}: missing columns {sorted(missing)}")
    times = {}
    for row in distance_frame.itertuples(index=False):
        key = (str(row.Origin_Code).strip(), str(row.Destination_Code).strip())
        value = float(row.Duration_min)
        if key not in times or value < times[key]:
            times[key] = value

    for route, group in routes.groupby("Route", sort=False):
        ordered = group.sort_values("StopSeq")
        stops = ordered["StationCode"].astype(str).tolist()
        raw_cumulative = [0.0]
        for origin, destination_code in zip(stops, stops[1:]):
            key = (origin, destination_code)
            if key not in times:
                raise KeyError(f"{route}: missing directed road time {key}")
            raw_cumulative.append(raw_cumulative[-1] + times[key])
        cumulative = [int(round(value)) for value in raw_cumulative]
        segments = [0] + [
            right - left for left, right in zip(cumulative, cumulative[1:])
        ]
        if any(value <= 0 for value in segments[1:]):
            raise ValueError(
                f"{route}: scaled cumulative rounding produced {segments}"
            )
        routes.loc[ordered.index, "CumTime_min"] = cumulative
        routes.loc[ordered.index, "SegTime_min"] = segments
        if "Route_TotalTime" in routes.columns:
            routes.loc[ordered.index, "Route_TotalTime"] = cumulative[-1]

    destination.parent.mkdir(parents=True, exist_ok=True)
    routes.to_csv(destination, index=False, encoding="utf-8-sig")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    INPUTS.mkdir(parents=True, exist_ok=True)

    base_representatives = representative_branch_file()
    copy_file(
        base_representatives,
        INPUTS / "representatives" / base_representatives.name,
    )

    allocation_source = cfg.results_dir / cfg.get("paths.stage1") / "Allocation"
    for window in STAGE2_WINDOWS:
        copy_file(
            allocation_source / f"DepotAlloc_{window:02d}.csv",
            INPUTS
            / "allocations"
            / f"K{K_BASELINE}_N{FLEET_BASELINE}"
            / f"DepotAlloc_{window:02d}.csv",
        )

    route_dir = INPUTS / "routes"
    route_results = cfg.results_dir / cfg.get("paths.routes")
    library_name = str(cfg.get("routes.library_file"))
    base_route = route_results / library_name
    copy_file(base_route, route_dir / library_name)
    if ROUTE_POOL_CASES:
        upper_name = str(cfg.get("evaluation.sensitivity.upper_route_pool_file"))
        lower_name = str(cfg.get("evaluation.sensitivity.lower_route_pool_file"))
        lower_suffix = str(cfg.get("evaluation.sensitivity.lower_route_pool_suffix"))
        full_route = route_results / upper_name
        full_route_frame = pd.read_csv(full_route, encoding="utf-8-sig")
        lower_mask = (
            full_route_frame["Type"].astype(str).str.upper().eq("LOC")
            | full_route_frame["Route"].astype(str).str.endswith(lower_suffix)
        )
        full_route_frame.loc[lower_mask].to_csv(
            route_dir / lower_name,
            index=False, encoding="utf-8-sig",
        )
        copy_file(full_route, route_dir / upper_name)
    depot_source = cfg.path("inputs.depots")
    copy_file(depot_source, INPUTS / "static" / depot_source.name)

    distance_source = cfg.path("inputs.road_times")
    travel_dir = INPUTS / "travel_time"
    copy_file(
        distance_source,
        travel_dir / f"{distance_source.stem}_{travel_time_label(1)}.csv",
    )
    for label, factor in TRAVEL_FACTORS.items():
        distance_path = travel_dir / f"{distance_source.stem}_{label}.csv"
        scaled = scaled_distance(distance_source, distance_path, factor)
        retime_routes(
            base_route,
            scaled,
            route_dir / f"{Path(library_name).stem}_{label}.csv",
        )

    # Stage-2 sensitivity uses the Stage 1 support of the main analysis; the
    # K scaling of Table 8 has its own preparation step.
    representative_metadata = {
        "method": "production K-means with in-cluster representatives",
        "source": str(base_representatives),
        "k": K_BASELINE,
    }
    files = sorted(
        path for path in INPUTS.rglob("*")
        if path.is_file() and path.name != "input_manifest.json"
    )
    manifest = {
        "schema": "peak-sensitivity-inputs-cohort-v2",
        "profile": cfg.name,
        "stage1_k_windows": list(STAGE1_K_WINDOWS),
        "stage2_windows": list(STAGE2_WINDOWS),
        "baseline_allocation": (
            f"matching K{K_BASELINE}/N{FLEET_BASELINE} Stage1 allocation per window"
        ),
        "travel_time_definition": (
            "scale raw directed road Duration_min; recompute route cumulative "
            "and segment times using cumulative rounding"
        ),
        "representative_generation": representative_metadata,
        "files": [
            {
                "path": str(path.relative_to(SENSITIVITY_ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    (INPUTS / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Prepared {len(files)} sensitivity input files under {INPUTS}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
