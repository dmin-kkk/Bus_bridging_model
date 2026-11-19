"""Derive nested Stage-2 route pools from the validated Kmin--Kchain route file.

The source file is never modified.  Each derived pool retains all LOC routes and
includes every EXP route whose ``_K#`` suffix is at most the requested maximum K
(routes.pool_max_stops).  This makes the K sensitivity inputs strictly nested
while leaving all non-route Stage-2 inputs unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bbconfig import cfg  # noqa: E402

ROUTE_DIR = Path(
    os.environ.get("PAPER_ROUTE_DIR") or cfg.results_dir / cfg.get("paths.routes")
).resolve()
N_LOCAL = int(cfg.get("routes.local_routes_expected"))
KMIN = int(cfg.get("routes.express_min_stops"))
TMAX = int(cfg.get("routes.express_max_travel_min"))
DEFAULT_SOURCE = ROUTE_DIR / (
    f"BusRoutes_LOC{N_LOCAL}_EXP_K{KMIN}-{cfg.get('routes.express_chain_max_stops')}_T{TMAX}.csv")
DEFAULT_OUTPUT_DIR = ROUTE_DIR
K_PATTERN = re.compile(r"_K(?P<k>\d+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_name(max_k: int) -> str:
    span = f"K{KMIN}" if max_k == KMIN else f"K{KMIN}-{max_k}"
    return f"BusRoutes_LOC{N_LOCAL}_EXP_{span}_T{TMAX}.csv"


def build(source: Path, output_dir: Path, max_values: tuple[int, ...]) -> list[dict]:
    source = source.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        required = {"Route", "Type", "StationCode", "StopSeq"}
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(f"route file missing columns: {sorted(missing)}")
        rows = list(reader)

    route_meta: dict[str, tuple[str, int | None]] = {}
    for row in rows:
        route = str(row["Route"]).strip()
        route_type = str(row["Type"]).strip().upper()
        if route_type == "LOC":
            k_value = None
        elif route_type == "EXP":
            match = K_PATTERN.search(route)
            if match is None:
                raise ValueError(f"EXP route has no _K# suffix: {route}")
            k_value = int(match.group("k"))
        else:
            raise ValueError(f"unknown route type {route_type!r}: {route}")
        previous = route_meta.setdefault(route, (route_type, k_value))
        if previous != (route_type, k_value):
            raise ValueError(f"inconsistent metadata within route {route}")

    loc_routes = {route for route, (kind, _) in route_meta.items() if kind == "LOC"}
    if len(loc_routes) != N_LOCAL:
        raise ValueError(
            f"{source}: {len(loc_routes)} LOC routes, routes.local_routes_expected={N_LOCAL}")
    manifests: list[dict] = []
    previous_routes: set[str] = set()
    output_dir.mkdir(parents=True, exist_ok=True)
    for max_k in sorted(set(max_values)):
        if max_k < KMIN:
            raise ValueError(f"maximum K must be at least routes.express_min_stops={KMIN}")
        selected_routes = {
            route for route, (kind, k_value) in route_meta.items()
            if kind == "LOC" or (k_value is not None and k_value <= max_k)
        }
        if not loc_routes.issubset(selected_routes):
            raise AssertionError("a nested pool dropped a LOC route")
        if previous_routes and not previous_routes.issubset(selected_routes):
            raise AssertionError("route pools are not nested")
        previous_routes = selected_routes

        selected_rows = [row for row in rows if row["Route"].strip() in selected_routes]
        output_path = output_dir / _output_name(max_k)
        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected_rows)

        exp_routes = selected_routes.difference(loc_routes)
        manifests.append({
            "label": f"new_k{max_k}",
            "max_k": max_k,
            "route_file": str(output_path),
            "sha256": _sha256(output_path),
            "loc_routes": len(loc_routes),
            "exp_routes": len(exp_routes),
            "total_routes": len(selected_routes),
            "rows": len(selected_rows),
        })

    manifest_path = output_dir / cfg.get("routes.pool_manifest_file")
    manifest_path.write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifests, ensure_ascii=False, indent=2))
    print(f"Saved manifest: {manifest_path}")
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-k", type=int, nargs="+",
                        default=tuple(cfg.get("routes.pool_max_stops")))
    args = parser.parse_args()
    build(args.source, args.output_dir, tuple(args.max_k))


if __name__ == "__main__":
    main()
