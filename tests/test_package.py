"""Release checks that run in seconds and never start a full-size optimisation.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import hashlib
import importlib
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Paper_Execution_Pipeline"))
import bbconfig  # noqa: E402

TEXT_SUFFIXES = {".py", ".toml", ".md", ".csv", ".json", ".txt", ".ps1"}
PERSONAL = [
    re.compile(r"\b[A-Za-z]:[\\/](?!n)"),      # drive-letter paths (allow escaped "\n" style text)
    re.compile(r"[\\/]Users[\\/]", re.I),
    re.compile(r"AppData", re.I),
    re.compile(r"gurobi\.lic|LICENSEID|WLSACCESSID|WLSSECRET", re.I),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}"),
]
SKIP = {"__pycache__", "results", ".git"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


class ProfileTests(unittest.TestCase):
    def test_profiles_have_identical_keys(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_profiles.py")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_pinned_files_match_their_hashes(self):
        # The pinned paper files derive from restricted data and are not distributed;
        # check them only when a user has placed them in the package.
        if not (ROOT / "pinned").is_dir():
            self.skipTest("pinned/ (paper profile, restricted) is not distributed")
        paper = bbconfig.load("paper")
        for group in ("clustering", "probabilities"):
            for key, value in paper.get(f"pinned.{group}").items():
                if key.endswith("_sha256"):
                    continue
                path = ROOT / value
                self.assertTrue(path.is_file(), path)
                self.assertEqual(sha256(path), paper.get(f"pinned.{group}.{key}_sha256").upper(), path)

    def test_synthetic_inputs_exist_and_match_manifest(self):
        synthetic = bbconfig.load("synthetic")
        for key, name in synthetic.get("inputs").items():
            self.assertTrue((synthetic.data_dir / name).is_file(), f"inputs.{key}: {name}")
        manifest = synthetic.data_dir / "MANIFEST.sha256"
        self.assertTrue(manifest.is_file())
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            digest, name = line.split(maxsplit=1)
            name = name.strip().lstrip("*")
            self.assertEqual(sha256(synthetic.data_dir / name), digest.upper(), name)

    def test_synthetic_central_group_is_derived_from_data(self):
        bbconfig.load("synthetic")
        import network_data
        importlib.reload(network_data)
        self.assertEqual(network_data.central_shared_stations(),
                         set(bbconfig.load("synthetic").get("network.central_shared_expected")))


class ReleaseHygieneTests(unittest.TestCase):
    def test_no_personal_paths_licences_or_emails(self):
        offenders = []
        for path in ROOT.rglob("*"):
            rel = path.relative_to(ROOT)
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if any(part in SKIP for part in rel.parts) or rel.name == "test_package.py":
                continue
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            scanner = rel.as_posix() in {"scripts/audit_package.py", "scripts/build_archive.py"}
            for number, line in enumerate(text.splitlines(), 1):
                if scanner and "re.compile(" in line:
                    continue  # detection pattern of a scanning tool
                if any(p.search(line) for p in PERSONAL):
                    offenders.append(f"{rel}:{number}: {line.strip()[:120]}")
        self.assertEqual(offenders, [], "\n".join(offenders[:50]))

    def test_no_model_or_licence_files_outside_synthetic_models(self):
        bad = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
               if p.suffix.lower() in {".lic", ".mps", ".lp", ".sol", ".mst", ".ilp"}
               and not p.relative_to(ROOT).parts[:2] == ("models", "synthetic")
               and not any(part in SKIP for part in p.relative_to(ROOT).parts)]
        self.assertEqual(bad, [])

    def test_plan_builds_for_every_profile(self):
        for name in bbconfig.available_profiles():
            env = {**os.environ, "BB_PROFILE": name, "BB_DATA_DIR": str(ROOT), "PYTHONUTF8": "1"}
            result = subprocess.run([sys.executable, str(ROOT / "run_all.py"), "--profile", name, "--dry-run"],
                                    capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
            self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
