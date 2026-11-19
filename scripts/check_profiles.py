"""Check that every profile defines the same configuration keys.

    python scripts/check_profiles.py

Exit code 0 when the key sets agree; otherwise the differing keys are printed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bbconfig  # noqa: E402


def leaf_keys(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaf_keys(value, f"{prefix}{key}.")
    else:
        yield prefix[:-1]


# Tables whose keys are data (where the inputs live, case ids, duration states) may differ.
IGNORE_PREFIXES = ("profile.",)


def main() -> int:
    profiles = bbconfig.available_profiles()
    keys = {}
    for name in profiles:
        profile = bbconfig.load(name)
        keys[name] = {k for k in leaf_keys({k: profile[k] for k in profile})
                      if not k.startswith(IGNORE_PREFIXES)}
    reference = profiles[0]
    status = 0
    for name in profiles[1:]:
        only_ref = sorted(keys[reference] - keys[name])
        only_other = sorted(keys[name] - keys[reference])
        # value tables keyed by data (duration states, case ids) may differ in content
        if only_ref or only_other:
            status = 1
            print(f"[{reference}] only: {only_ref}")
            print(f"[{name}] only: {only_other}")
    if status == 0:
        print(f"OK: {len(keys[reference])} keys in each of {profiles}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
