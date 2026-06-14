"""Run provenance for reproducible experiments.

Collects the metadata needed to reproduce and audit a run: a content hash of the
effective configuration, library and platform versions, and the current git
commit. This is the manifest backbone for observability — every run records
*exactly* what produced it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def collect_versions() -> dict[str, str]:
    """Return versions of the runtime and the core scientific dependencies."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for module_name, label in (
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("torch_geometric", "torch_geometric"),
        ("networkx", "networkx"),
        ("matplotlib", "matplotlib"),
    ):
        try:
            module = __import__(module_name)
            versions[label] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # pragma: no cover - optional dependency probing
            versions[label] = "missing"
    return versions


def git_sha(cwd: str | Path | None = None) -> str | None:
    """Return the current git commit SHA, or ``None`` if unavailable.

    Args:
        cwd: Repository directory; defaults to the process working directory.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git absent
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def config_hash(config: Mapping[str, Any]) -> str:
    """Return a stable SHA256 hash of a configuration mapping.

    The mapping is canonicalised (sorted keys, ``default=str`` for non-JSON
    values) before hashing, so logically identical configs hash identically
    regardless of key order or container type.
    """
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def collect_provenance(
    config: Mapping[str, Any],
    *,
    cwd: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a provenance record for a run.

    Args:
        config: Effective configuration dictionary for the run (hashed).
        cwd: Repository directory for the git SHA lookup.
        extra: Additional fields to merge into the record.

    Returns:
        A JSON-serialisable provenance dictionary with a UTC timestamp, the
        config hash, library/platform versions, and the git commit.
    """
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config_hash": config_hash(config),
        "git_sha": git_sha(cwd),
        "versions": collect_versions(),
    }
    if extra:
        record.update(dict(extra))
    return record
