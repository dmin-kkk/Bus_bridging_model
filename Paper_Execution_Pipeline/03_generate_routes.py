"""Generate the bridging-route library: local routes plus nested express variants.

Every value comes from the active profile (``config/profiles/<profile>/routes.toml``):
local stop sequences are ``inputs.local_routes``, road times ``inputs.road_times``,
OD demand ``inputs.od_matrix``; K range, travel-time cap, saving threshold and
endpoint policy are ``routes.express_*``. The final library is ``routes.library_file``.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from pipeline_common import (
    CORE_DIR, LOG_DIR, PYTHON, ROOT, ROUTE_DIR, cfg, ensure_layout,
    pipeline_env, require_files, require_packages, run_logged,
)
from pipeline_config import ROUTE_FILENAME


LOCAL_BUILDER = CORE_DIR / "build_local_routes.py"
EXPRESS_BUILDER = CORE_DIR / "build_express_variants.py"
POOL_BUILDER = CORE_DIR / "build_route_pool.py"

N_LOCAL = int(cfg.get("routes.local_routes_expected"))
N_EXPRESS = int(cfg.get("routes.express_routes_expected"))   # -1: not checked
KMIN = int(cfg.get("routes.express_min_stops"))
KMAX = int(cfg.get("routes.express_max_stops"))
KCHAIN = int(cfg.get("routes.express_chain_max_stops"))
TMAX = int(cfg.get("routes.express_max_travel_min"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def library_name(max_k: int) -> str:
    """Pool file name for express routes with Kmin..max_k stops (as build_route_pool.py)."""
    span = f"K{KMIN}" if max_k == KMIN else f"K{KMIN}-{max_k}"
    return f"BusRoutes_LOC{N_LOCAL}_EXP_{span}_T{TMAX}.csv"


def check() -> None:
    ensure_layout()
    require_packages(("pandas",))
    require_files((
        LOCAL_BUILDER, EXPRESS_BUILDER, POOL_BUILDER,
        cfg.path("inputs.local_routes"),
        cfg.path("inputs.od_matrix"),
        cfg.path("inputs.road_times"),
    ))
    if library_name(KMAX) != ROUTE_FILENAME:
        raise ValueError(
            f"routes.library_file={ROUTE_FILENAME} differs from the pool name "
            f"{library_name(KMAX)} implied by routes.express_* and routes.local_routes_expected")


def execute(command: list[str], log_name: str) -> None:
    code = run_logged(
        command, log_path=LOG_DIR / log_name,
        env=pipeline_env(), cwd=ROOT,
    )
    if code:
        raise RuntimeError(f"route command failed; see {LOG_DIR / log_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    check()
    if args.check_only:
        print("Route-generation preflight OK")
        return

    local = ROUTE_DIR / cfg.get("routes.local_detail_file")
    full = ROUTE_DIR / library_name(KCHAIN)
    audit = ROUTE_DIR / f"EXP_K{KMIN}-{KCHAIN}_T{TMAX}_audit.csv"
    final = ROUTE_DIR / ROUTE_FILENAME

    execute(
        [str(PYTHON), "-X", "utf8", str(LOCAL_BUILDER)],
        "03_routes_local.log",
    )
    execute(
        [
            str(PYTHON), "-X", "utf8", str(EXPRESS_BUILDER),
            "--local", str(local),
            "--od", str(cfg.path("inputs.od_matrix")),
            "--distance", str(cfg.path("inputs.road_times")),
            "--out", str(full),
            "--audit", str(audit),
            "--kmin", str(KMIN), "--kmax", str(KCHAIN),
            "--max-route-min", str(TMAX),
            "--min-saving-min", str(cfg.get("routes.express_min_saving_min")),
            "--endpoint-policy", str(cfg.get("routes.express_endpoint_policy")),
        ],
        "03_routes_express.log",
    )
    # Only the K <= n_s library is written here (as in the paper run); the other
    # nested pools (routes.pool_max_stops) come from build_route_pool.py run directly.
    execute(
        [
            str(PYTHON), "-X", "utf8", str(POOL_BUILDER),
            "--source", str(full), "--output-dir", str(ROUTE_DIR),
            "--max-k", str(KMAX),
        ],
        "03_routes_pool.log",
    )

    require_files((local, full, audit, final))
    frame = pd.read_csv(final, encoding="utf-8-sig")
    routes = frame[["Route", "Type"]].drop_duplicates()
    counts = routes["Type"].str.upper().value_counts().to_dict()
    n_loc, n_exp = counts.get("LOC", 0), counts.get("EXP", 0)
    if n_loc != N_LOCAL or (N_EXPRESS >= 0 and n_exp != N_EXPRESS) \
            or len(routes) != n_loc + n_exp:
        raise RuntimeError(
            f"unexpected final route universe: {counts}, total={len(routes)} "
            f"(routes.local_routes_expected={N_LOCAL}, "
            f"routes.express_routes_expected={N_EXPRESS})")
    print(
        f"Route generation complete: {final}\n"
        f"routes={len(routes)} (LOC={n_loc}, EXP={n_exp}), sha256={sha256(final)}"
    )


if __name__ == "__main__":
    main()
