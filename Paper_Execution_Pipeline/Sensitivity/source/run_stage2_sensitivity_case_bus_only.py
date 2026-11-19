"""Run one BUS_ONLY_WAIT_AND_SEE Stage-2 sensitivity case."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve()
LEGACY_RUNNER = SCRIPT.with_name("run_stage2_sensitivity_case.py")


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "stage2_sensitivity_case_legacy_core", LEGACY_RUNNER
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {LEGACY_RUNNER}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def main() -> int:
    runner = load_runner()
    runner.PRODUCTION_SCHEMA = (
        "paper-stage2-full-pool-joint-30-60-bus-only-na-v5"
    )
    runner.SENSITIVITY_SCHEMA = (
        "production-core-stage2-sensitivity-v7-bus-only-wait-and-see"
    )
    runner.NONANTICIPATIVITY_MODE = "bus-only"
    runner.FORMULATION_MODE = "BUS_ONLY_WAIT_AND_SEE"
    return int(runner.main())


if __name__ == "__main__":
    raise SystemExit(main())
