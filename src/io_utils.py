"""Input/output utilities for saving artifacts and manifests.

Centralized I/O operations for experiment outputs, ensuring consistent
formatting and provenance tracking.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def write_csv_table(
    path: Path,
    header: list[str],
    rows: list[list[Any]],
) -> None:
    """Write a CSV table with header and data rows.

    Args:
        path: Output file path.
        header: Column names.
        rows: Data rows (must match header length).

    Raises:
        ValueError: If row lengths are inconsistent.
    """
    if not rows:
        raise ValueError("Cannot write empty table")

    expected_cols = len(header)
    for i, row in enumerate(rows):
        if len(row) != expected_cols:
            raise ValueError(
                f"Row {i} has {len(row)} columns, expected {expected_cols} "
                f"to match header {header}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        path: Path to file.

    Returns:
        Hexadecimal SHA256 hash string.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash non-existent file: {path}")

    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_json_manifest(
    path: Path,
    manifest: dict[str, Any],
) -> None:
    """Save a JSON manifest with pretty printing.

    Args:
        path: Output file path.
        manifest: Dictionary to serialize (must be JSON-serializable).

    Raises:
        TypeError: If manifest contains non-serializable objects.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Validate serializability before writing
    try:
        json.dumps(manifest, default=str)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Manifest is not JSON-serializable: {e}") from e

    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
