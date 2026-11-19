"""Release audit of the package.

    python scripts/audit_package.py --data-dir <folder with the paper inputs> [--out audit.md]

Scans every text file of the package for

* absolute or personal paths, licence strings and e-mail addresses (all files);
* station, depot, corridor and line identifiers of the paper network inside code (``*.py``);
  the identifiers are read at run time from the paper inputs, so this script names none;
* paper parameter values written as literals in code;
* vehicle identifiers and incident free text from the restricted incident registry (all files);
* restricted file types (real-instance model files) and restricted column names in shipped CSVs.

Configuration (``config/``), documentation (``*.md``) and data folders may name stations;
they are reported for information only. Code hits are violations unless justified.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import tokenize
import tomllib
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bbconfig  # noqa: E402

TEXT_SUFFIXES = {".py", ".toml", ".md", ".csv", ".json", ".txt", ".ps1", ".cfg", ".ini", ".yaml", ".yml", ".tex"}
SKIP_DIRS = {"__pycache__", ".git", "results"}
CODE_SUFFIXES = {".py", ".ps1"}
SELF_SCANNERS = {"scripts/audit_package.py", "tests/test_package.py", "scripts/build_archive.py"}

PATH_PATTERNS = [
    re.compile(r"\b[A-Za-z]:[\\/]"),          # drive-letter paths
    re.compile(r"[\\/]Users[\\/]", re.I),
    re.compile(r"AppData", re.I),
    re.compile("\ub0b4 \ub4dc\ub77c\uc774\ube0c"),  # Google Drive folder name (Korean)
    re.compile(r"[\\/]home[\\/][a-z]"),
]
LICENSE_PATTERNS = [
    re.compile(r"gurobi\.lic", re.I),
    re.compile(r"\bLICENSEID\b|\bWLSACCESSID\b|\bWLSSECRET\b|\bTOKENSERVER\b|\bLICENSEKEY\b", re.I),
    re.compile(r"grbgetkey", re.I),
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I),
]
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+")
RESTRICTED_COLUMNS = re.compile(
    r"^(vehicle_?id|trip_?id|tripid|item_?description|orgtext|incident_?id|origin|destination|"
    r"o_?code|d_?code|origin_code|destination_code|pax|passengers?|date|tripdate|actual.*)$",
    re.I,
)
# paper parameter values that must live in configuration only
NUMBER_PATTERNS = [
    (re.compile(r"(?<![\w.])1,?058(?![\w.])"), "instance count 1,058"),
    (re.compile(r"(?<![\w.])(303|404)(?![\w.])"), "cohort size"),
    (re.compile(r"(?<![\w.])(208|267|137)(?![\w.])"), "prior counts"),
    (re.compile(r"(?<![\w.])0\.(686|314|661)\d*(?![\w.])"), "duration probabilities"),
    (re.compile(r"\brange\(\s*6\s*,\s*21\s*\)|['\"]6-20['\"]"), "paper windows 6..20"),
    (re.compile(r"(?:default\s*=\s*|=\s*)(14|15|30|50|60|120|240|500|600)\b(?!\s*[*/]\s*60)"), "parameter literal"),
    (re.compile(r"(?:default\s*=\s*|=\s*)(0\.01|0\.05|0\.5|1e100)\b"), "solver/probability literal"),
]


def iter_files():
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            rel = path.relative_to(ROOT)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            yield path, rel


def category(rel: Path) -> str:
    top = rel.parts[0]
    if rel.suffix.lower() in CODE_SUFFIXES:
        return "code"
    if top in {"config"}:
        return "config"
    if top in {"data", "archive", "pinned", "models"}:
        return "data"
    if rel.suffix.lower() in {".md", ".tex", ".txt"}:
        return "docs"
    return "other"


def network_tokens(data_dir: Path, profile) -> dict[str, set[str]]:
    """Identifiers of the paper network, read from the paper inputs."""
    tokens: dict[str, set[str]] = defaultdict(set)
    corridors = pd.read_csv(data_dir / profile.get("inputs.corridors"), dtype=str)
    for column in corridors.columns:
        tokens["corridor/track name"].add(column.strip())
        tokens["station code"].update(s.strip() for s in corridors[column].dropna() if s.strip())
    tracks_file = data_dir / profile.get("inputs.corridor_tracks")
    if tracks_file.is_file():
        tracks = pd.read_csv(tracks_file, dtype=str)
        tokens["corridor/track name"].update(tracks["Corridor"].dropna())
        tokens["corridor label"].update(tracks["CorridorLabel"].dropna())
    book = pd.ExcelFile(data_dir / profile.get("inputs.stations"))
    for sheet in book.sheet_names:
        tokens["line name"].add(sheet.strip())
        frame = book.parse(sheet)
        if "Code" in frame.columns:
            tokens["station code"].update(frame["Code"].dropna().astype(str).str.strip())
        for column in frame.columns:
            if str(column).lower() in {"station", "name", "station_name", "stationname"}:
                tokens["station name"].update(frame[column].dropna().astype(str).str.strip())
    depots = pd.read_csv(data_dir / profile.get("inputs.depots"), dtype=str)
    tokens["depot code"].update(depots.iloc[:, 0].dropna().str.strip())
    for column in depots.columns:
        if column.lower() in {"station", "name"}:
            tokens["station name"].update(depots[column].dropna().str.strip())
    # very short or generic names are too ambiguous to audit
    for key in list(tokens):
        tokens[key] = {t for t in tokens[key] if len(t) >= 3}
    return tokens


def restricted_tokens(data_dir: Path, profile) -> tuple[set[str], list[str]]:
    raw = pd.read_csv(data_dir / profile.get("inputs.incidents_raw"), dtype=str, low_memory=False)
    vehicles = set()
    for column in raw.columns:
        if "vehicle" in column.lower():
            vehicles.update(v.strip() for v in raw[column].dropna() if len(v.strip()) >= 4)
    texts = []
    for column in raw.columns:
        if column.lower() in {"itemdescription", "orgtext", "description", "comments"}:
            texts.extend(t for t in raw[column].dropna().astype(str) if len(t) >= 30)
    return vehicles, texts


def shingles(text: str, n: int = 6) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def code_strings_and_comments(text: str):
    """(line, kind, value) for string literals and comments of a Python file."""
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.STRING:
                yield tok.start[0], "string", tok.string
            elif tok.type == tokenize.COMMENT:
                yield tok.start[0], "comment", tok.string
            elif tok.type == tokenize.NAME:
                yield tok.start[0], "name", tok.string
    except (tokenize.TokenError, IndentationError, SyntaxError):
        for number, line in enumerate(text.splitlines(), 1):
            yield number, "line", line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True, help="folder with the paper inputs")
    parser.add_argument("--out", type=Path, default=None, help="write the report here (Markdown)")
    parser.add_argument("--root", type=Path, default=None,
                        help="folder to scan (default: this package); used for before/after comparisons")
    args = parser.parse_args()
    global ROOT
    if args.root is not None:
        ROOT = args.root.resolve()

    profile = bbconfig.load("paper")
    allow_file = Path(__file__).resolve().parents[1] / "config" / "audit_allowlist.toml"
    rules = tomllib.loads(allow_file.read_text(encoding="utf-8")).get("allow", []) if allow_file.is_file() else []

    def allowed(token: str, kind: str, value: str) -> str | None:
        for rule in rules:
            if rule["token"] != token:
                continue
            if rule.get("kind") and rule["kind"] != kind:
                continue
            if rule.get("context") and not re.search(rule["context"], value):
                continue
            return rule["reason"]
        return None
    net = network_tokens(args.data_dir, profile)
    vehicles, texts = restricted_tokens(args.data_dir, profile)
    text_shingles = set()
    for t in texts:
        text_shingles |= shingles(t)

    findings: dict[str, list[str]] = defaultdict(list)
    word_res = {
        kind: re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(sorted(map(re.escape, values), key=len, reverse=True)) + r")(?![A-Za-z0-9_])")
        for kind, values in net.items() if values
    }
    vehicle_re = (re.compile(r"(?<![A-Za-z0-9.])(" + "|".join(map(re.escape, sorted(vehicles, key=len, reverse=True))) + r")(?![A-Za-z0-9]|\.\d)")
                  if vehicles else None)

    for path, rel in iter_files():
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        cat = category(rel)
        lines = text.splitlines()
        self_scan = rel.as_posix() in SELF_SCANNERS
        for number, line in enumerate(lines, 1):
            if self_scan and "re.compile(" in line:
                continue  # detection pattern of a scanning tool, not a leak
            for pattern in PATH_PATTERNS:
                if pattern.search(line):
                    findings["absolute/personal path"].append(f"{rel}:{number} [{cat}] {line.strip()[:160]}")
                    break
            for pattern in LICENSE_PATTERNS:
                if pattern.search(line):
                    findings["licence string"].append(f"{rel}:{number} [{cat}] {line.strip()[:160]}")
                    break
            if EMAIL.search(line):
                findings["e-mail address"].append(f"{rel}:{number} [{cat}] {line.strip()[:160]}")
            if vehicle_re is not None and vehicle_re.search(line):
                findings["vehicle identifier"].append(f"{rel}:{number} [{cat}] {line.strip()[:160]}")
        if text_shingles and (shingles(text) & text_shingles):
            hit = next(iter(shingles(text) & text_shingles))
            findings["incident free text"].append(f"{rel} [{cat}] shares the 6-word sequence '{' '.join(hit)}'")
        if rel.suffix == ".csv" and cat == "data" and rel.parts[0] in {"archive", "pinned", "models"}:
            header = lines[0] if lines else ""
            bad = [c for c in header.split(",") if RESTRICTED_COLUMNS.match(c.strip().strip('"'))]
            if bad:
                findings["restricted column name in shipped CSV"].append(f"{rel}: {bad}")
        if cat == "code" and rel.suffix == ".py":
            if rel.parts[:2] == ("scripts", "audit_package.py"):
                continue
            for number, kind, value in code_strings_and_comments(text):
                for net_kind, regex in word_res.items():
                    match = regex.search(value)
                    if match and not (kind == "name" and net_kind != "station code"):
                        reason = allowed(match.group(1), kind, value)
                        label = (f"allowed coincidence ({reason})" if reason
                                 else f"network identifier in code ({net_kind})")
                        findings[label].append(
                            f"{rel}:{number} [{kind}] '{match.group(1)}' in {value.strip()[:120]}")
            for number, line in enumerate(lines, 1):
                stripped = line.split("#", 1)[0]
                for regex, label in NUMBER_PATTERNS:
                    if regex.search(stripped):
                        findings[f"literal in code ({label})"].append(f"{rel}:{number} {line.strip()[:160]}")
        elif cat in {"docs", "config"}:
            for net_kind, regex in word_res.items():
                if net_kind in {"station code", "depot code"}:
                    count = len(regex.findall(text))
                    if count:
                        findings[f"info: network identifiers in {cat}"].append(f"{rel}: {count} occurrences")

    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if path.suffix.lower() in {".mps", ".lp", ".sol", ".mst", ".ilp", ".gz", ".lic"} and not str(rel).startswith(("models\\synthetic", "models/synthetic")):
            if not any(part in SKIP_DIRS for part in rel.parts):
                findings["model/licence file outside models/synthetic"].append(str(rel))

    out = io.StringIO()
    out.write("# Package audit\n\n")
    out.write(f"Network identifiers checked: { {k: len(v) for k, v in net.items()} }; "
              f"vehicle ids: {len(vehicles)}; free-text records: {len(texts)}\n\n")
    if not findings:
        out.write("No findings.\n")
    for key in sorted(findings):
        out.write(f"## {key} ({len(findings[key])})\n\n")
        for item in findings[key][:400]:
            out.write(f"- {item}\n")
        if len(findings[key]) > 400:
            out.write(f"- ... {len(findings[key]) - 400} more\n")
        out.write("\n")
    report = out.getvalue()
    if args.out:
        args.out.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
