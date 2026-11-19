"""Render the paper figures and tables (compatibility entry point).

    python render_results.py                                   # figures from cfg.results_dir / archive
    python render_results.py --from-results results/synthetic   (with BB_PROFILE=synthetic)

Without ``--from-results`` the figures and tables are regenerated from the archive of the active
profile (``cfg.results_dir / archive``, written by ``run_all.py`` stage 11). With ``--from-results``
the archive is first rebuilt from a run in the package result layout (``scripts/build_archive.py``)
into ``<results>/archive``. The package is synthetic-only; no pre-built archive is shipped.
Outputs: ``cfg.results_dir / paths.figures`` (see ``docs/TABLE_FIGURE_MAP.md``).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


def run(script: Path, *args) -> None:
    subprocess.run([sys.executable, "-X", "utf8", str(script), *map(str, args)], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-results", type=Path, help="rebuild the archive from this results root first")
    parser.add_argument("--archive", type=Path, help="archive folder to read (default: cfg.results_dir / archive)")
    args = parser.parse_args()
    archive = args.archive
    if args.from_results is not None:
        archive = archive or args.from_results / "archive"
        run(SCRIPTS / "build_archive.py", "--from-results", args.from_results, "--out", archive)
    run(SCRIPTS / "make_paper_figures.py", *(["--archive", archive] if archive else []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
