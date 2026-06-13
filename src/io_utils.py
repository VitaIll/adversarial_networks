"""Input/output utilities for saving artifacts and manifests.

Centralized I/O operations for experiment outputs, ensuring consistent
formatting and provenance tracking.
"""

from __future__ import annotations

import csv
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np


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


def append_realization_row(path: Path, row: dict[str, Any]) -> None:
    """Append one realization result row to CSV, creating the file/header if needed.

    Args:
        path: CSV file path.
        row: Flat result dictionary. The optional ``history`` key is ignored.

    Raises:
        ValueError: If no serializable columns remain after filtering.
    """
    row_to_write = {key: value for key, value in row.items() if key != "history"}
    if not row_to_write:
        raise ValueError("row must contain at least one serializable key.")

    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    fieldnames = list(row_to_write.keys())
    if file_exists:
        try:
            with path.open("r", newline="", encoding="utf-8") as f_read:
                reader = csv.reader(f_read)
                header = next(reader, None)
                if header:
                    fieldnames = [str(name) for name in header]
        except OSError:
            # Fall back to row keys if header cannot be read.
            pass

    write_header = (not file_exists) or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row_to_write)


def load_completed_realizations(path: Path) -> list[dict[str, Any]]:
    """Load completed Monte Carlo realization rows from a CSV file.

    Args:
        path: CSV file path.

    Returns:
        Parsed rows with numeric/string conversions applied where possible.
    """
    if not path.exists():
        return []

    type_map: dict[str, type] = {
        "realization": int,
        "final_step": int,
        "beta_hat": float,
        "gamma_hat": float,
        "sigma_sq_hat": float,
        "beta_final": float,
        "gamma_final": float,
        "sigma_sq_final": float,
        "loss_d_final": float,
        "loss_g_final": float,
        "loss_d_rolling_final": float,
        "loss_g_rolling_final": float,
        "init_seed": int,
        "init_beta": float,
        "init_gamma": float,
        "init_log_sigma_sq": float,
        "gt_seed": int,
        "train_seed": int,
        "elapsed_seconds": float,
    }
    true_tokens = {"1", "true", "t", "yes", "y"}
    false_tokens = {"0", "false", "f", "no", "n"}

    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_idx, raw_row in enumerate(reader, start=1):
                parsed: dict[str, Any] = {}
                row_valid = True
                for key, value in raw_row.items():
                    if value is None:
                        parsed[key] = value
                        continue
                    value_str = value.strip()
                    if value_str == "":
                        parsed[key] = value_str
                        continue

                    if key == "converged":
                        lower = value_str.lower()
                        if lower in true_tokens:
                            parsed[key] = True
                        elif lower in false_tokens:
                            parsed[key] = False
                        else:
                            warnings.warn(
                                f"Skipping malformed row {row_idx}: invalid converged value "
                                f"{value_str!r}.",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                            row_valid = False
                            break
                        continue

                    caster = type_map.get(key)
                    if caster is None:
                        parsed[key] = value
                        continue

                    try:
                        parsed[key] = caster(value_str)
                    except (TypeError, ValueError):
                        warnings.warn(
                            f"Skipping malformed row {row_idx}: cannot parse {key}={value_str!r}.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        row_valid = False
                        break

                if row_valid:
                    rows.append(parsed)
    except (OSError, csv.Error) as exc:
        warnings.warn(
            f"Could not load completed realizations from {path}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    return rows


def save_realization_history(path: Path, history: dict[str, list[float]]) -> None:
    """Save one realization training history as compressed float32 arrays.

    Args:
        path: Output ``.npz`` path.
        history: Per-step history lists keyed by metric name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        beta=np.asarray(history.get("beta", []), dtype=np.float32),
        gamma=np.asarray(history.get("gamma", []), dtype=np.float32),
        sigma_sq=np.asarray(history.get("sigma_sq", []), dtype=np.float32),
        loss_d=np.asarray(history.get("loss_d", []), dtype=np.float32),
        loss_g=np.asarray(history.get("loss_g", []), dtype=np.float32),
        tau_x=np.asarray(history.get("tau_x", []), dtype=np.float32),
        tau_y=np.asarray(history.get("tau_y", []), dtype=np.float32),
    )
