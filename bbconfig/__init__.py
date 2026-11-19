"""Profile-based configuration for the bus-bridging reproducibility package.

Every run-time value (windows, fleet size, solver settings, network
definitions, ...) is read from a *profile* under ``config/profiles/<name>/``.
A profile is a directory of TOML files; each file contributes its top-level
tables, and the files are merged into one nested mapping.  No value used by
the pipeline may be hard-coded in a script: scripts call ``bbconfig.cfg``.

Usage
-----
>>> from bbconfig import cfg
>>> cfg.get("operations.fleet_size")
30
>>> cfg.path("inputs.timetable")          # resolved against the data directory
>>> cfg.package_path("pinned.clustering.branches")   # resolved against the package
>>> cfg.results_dir / "04_stage1"

Profile selection (first match wins):
1. ``bbconfig.load(name)`` called explicitly;
2. environment variable ``BB_PROFILE``;
3. ``paper``.

Data directory (restricted inputs are never stored in the package):
* the profile's ``[profile].data_dir_env`` names an environment variable
  (``BB_DATA_DIR`` for the paper profile) that must point to the inputs, or
* ``[profile].data_dir`` gives a path relative to the package root
  (the synthetic profile ships its generated data in ``data/synthetic``).

Results directory: ``BB_RESULTS_DIR`` if set, otherwise
``[profile].results_dir`` relative to the package root.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Iterator, Mapping

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = PACKAGE_ROOT / "config" / "profiles"
_MISSING = object()


class ConfigError(KeyError):
    """Raised when a required configuration key is absent."""


def _merge(target: dict, source: Mapping, origin: str) -> None:
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, Mapping):
            _merge(target[key], value, origin)
        elif key in target:
            raise ConfigError(f"duplicate configuration key '{key}' (second definition in {origin})")
        else:
            target[key] = dict(value) if isinstance(value, Mapping) else value


class Profile(Mapping):
    """Read-only nested view of one merged profile."""

    def __init__(self, name: str, data: dict, files: list[Path]):
        self.name = name
        self._data = data
        self.files = files

    # Mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # Dotted access ----------------------------------------------------
    def get(self, dotted: str, default: Any = _MISSING) -> Any:  # type: ignore[override]
        node: Any = self._data
        for part in dotted.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            elif default is _MISSING:
                raise ConfigError(f"profile '{self.name}' has no key '{dotted}'")
            else:
                return default
        return node

    def require(self, *dotted: str) -> list[Any]:
        return [self.get(key) for key in dotted]

    # Paths ------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        section = self._data.get("profile", {})
        env_name = section.get("data_dir_env")
        if env_name:
            value = os.environ.get(env_name)
            if not value:
                raise ConfigError(
                    f"profile '{self.name}' reads its inputs from the directory in "
                    f"environment variable {env_name}; set it to the folder that holds "
                    "the input files listed in [inputs]")
            return Path(value).expanduser().resolve()
        relative = section.get("data_dir")
        if not relative:
            raise ConfigError(f"profile '{self.name}' defines neither data_dir nor data_dir_env")
        return (PACKAGE_ROOT / relative).resolve()

    @property
    def results_dir(self) -> Path:
        override = os.environ.get("BB_RESULTS_DIR")
        if override:
            return Path(override).expanduser().resolve()
        return (PACKAGE_ROOT / self.get("profile.results_dir")).resolve()

    def path(self, dotted: str) -> Path:
        """File name stored under ``dotted`` resolved against the data directory."""
        return self.data_dir / str(self.get(dotted))

    def package_path(self, dotted: str) -> Path:
        """Path stored under ``dotted`` resolved against the package root."""
        return (PACKAGE_ROOT / str(self.get(dotted))).resolve()


_ACTIVE: Profile | None = None


def available_profiles() -> list[str]:
    return sorted(p.name for p in PROFILE_ROOT.iterdir() if p.is_dir())


def load(name: str | None = None) -> Profile:
    """Load (and activate) a profile; ``name`` defaults to $BB_PROFILE or 'paper'."""
    global _ACTIVE
    name = name or os.environ.get("BB_PROFILE") or "paper"
    folder = PROFILE_ROOT / name
    if not folder.is_dir():
        raise ConfigError(f"unknown profile '{name}'; available: {available_profiles()}")
    data: dict = {}
    files = sorted(folder.glob("*.toml"))
    for file in files:
        with file.open("rb") as handle:
            _merge(data, tomllib.load(handle), file.name)
    _ACTIVE = Profile(name, data, files)
    os.environ["BB_PROFILE"] = name  # propagate to child processes
    return _ACTIVE


class _LazyActive:
    """``cfg`` proxy: loads the active profile on first use."""

    def _profile(self) -> Profile:
        return _ACTIVE if _ACTIVE is not None else load()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._profile(), item)

    def __getitem__(self, key: str) -> Any:
        return self._profile()[key]

    def __contains__(self, key: str) -> bool:
        return key in self._profile()

    def __repr__(self) -> str:
        return f"<bbconfig profile {self._profile().name!r}>"


cfg = _LazyActive()

__all__ = ["cfg", "load", "Profile", "ConfigError", "PACKAGE_ROOT", "available_profiles"]
