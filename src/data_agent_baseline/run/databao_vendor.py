"""Runtime application of the local Databao graph patches.

The starter kit depends on the published ``databao-agent`` package, but our current
best runner relies on two small internal graph changes.  Keeping the patched files
inside this repository makes a fresh checkout reproducible without manually editing
``.venv``.
"""

from __future__ import annotations

import importlib.util
import os
from importlib.resources import files
from pathlib import Path


DATABAO_VENDOR_PATCH_ENV = "DATABAO_APPLY_VENDOR_PATCHES"


def _vendor_patches_enabled() -> bool:
    value = os.environ.get(DATABAO_VENDOR_PATCH_ENV, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _databao_package_root() -> Path:
    spec = importlib.util.find_spec("databao")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("databao package is not installed; run `uv sync` first.")
    return Path(next(iter(spec.submodule_search_locations)))


def ensure_vendor_databao_patches() -> dict[str, str]:
    """Copy vendored Databao graph patches into the installed package if needed."""

    if not _vendor_patches_enabled():
        return {"enabled": "false", "status": "skipped"}

    package_root = _databao_package_root()
    patch_root = files("data_agent_baseline._vendor.databao_patches")
    patch_map = {
        "lighthouse_graph.py": package_root / "agent" / "executors" / "lighthouse" / "graph.py",
        "separate_graph.py": package_root / "agent" / "executors" / "separate" / "graph.py",
    }
    updated: list[str] = []
    for patch_name, target in patch_map.items():
        source = patch_root.joinpath(patch_name)
        source_bytes = source.read_bytes()
        if target.exists() and target.read_bytes() == source_bytes:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_bytes)
        updated.append(str(target))
    return {
        "enabled": "true",
        "status": "updated" if updated else "already_current",
        "package_root": str(package_root),
        "updated_files": ";".join(updated),
    }
