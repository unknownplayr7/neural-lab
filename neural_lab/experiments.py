"""Safe JSON experiment records and comparison helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any
from uuid import UUID, uuid4

import torch


EXPERIMENT_SCHEMA_VERSION = 1
DATASET_SOURCE = {
    "cifar10": "uoft-cs/cifar10 (Hugging Face Datasets streaming)",
    "cifar100": "uoft-cs/cifar100 (Hugging Face Datasets streaming)",
}


def create_experiment_record(
    run_config: dict[str, Any],
    *,
    architecture: str,
    device: str,
) -> dict[str, Any]:
    """Create a new reproducibility record with a stable UUID and UTC timestamp."""
    dataset_name = run_config.get("dataset_name")
    if dataset_name not in DATASET_SOURCE:
        raise ValueError("Unsupported experiment dataset")
    if architecture not in {"CifarCNN", "CifarResNet"}:
        raise ValueError("Unsupported experiment architecture")
    record = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "run_id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "dataset_source": DATASET_SOURCE[dataset_name],
        "split_strategy": (
            "Seeded shuffled train stream; disjoint validation window immediately "
            "after the training window; separate held-out test split."
        ),
        "architecture": architecture,
        "device": str(device),
        "run_config": dict(run_config),
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
        },
        "history": [],
        "test_metrics": None,
    }
    _validate_record(record)
    return record


def _validate_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict) or record.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError("Unsupported experiment record")
    try:
        UUID(record.get("run_id", ""))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Experiment run_id must be a UUID") from exc
    if not isinstance(record.get("created_at"), str):
        raise ValueError("Experiment created_at must be a string")
    if record.get("dataset_name") not in DATASET_SOURCE:
        raise ValueError("Experiment dataset is unsupported")
    if record.get("architecture") not in {"CifarCNN", "CifarResNet"}:
        raise ValueError("Experiment architecture is unsupported")
    if not isinstance(record.get("run_config"), dict):
        raise ValueError("Experiment run_config must be a mapping")
    if not isinstance(record.get("software"), dict):
        raise ValueError("Experiment software metadata must be a mapping")
    if not isinstance(record.get("history"), list):
        raise ValueError("Experiment history must be a list")
    if record.get("test_metrics") is not None and not isinstance(record["test_metrics"], dict):
        raise ValueError("Experiment test_metrics must be a mapping or null")
    # Reject tensors, NaN/Infinity, and unexpected Python objects before persistence.
    json.dumps(record, allow_nan=False)
    return record


def update_experiment_record(
    record: dict[str, Any],
    history: list[dict[str, Any]],
    test_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an updated JSON-safe copy containing current metrics."""
    _validate_record(record)
    updated = dict(record)
    updated["history"] = [dict(row) for row in history]
    if test_metrics is not None:
        updated["test_metrics"] = {
            key: test_metrics[key]
            for key in ("accuracy", "loss", "examples")
            if key in test_metrics
        }
    _validate_record(updated)
    return updated


def save_experiment_record(
    record: dict[str, Any],
    directory: str | Path = "experiments",
) -> Path:
    """Atomically save a record under its UUID-derived filename."""
    _validate_record(record)
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / (record["run_id"] + ".json")
    fd, temporary_name = tempfile.mkstemp(prefix=".experiment-", suffix=".tmp", dir=target_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return target


def load_experiment_records(directory: str | Path = "experiments") -> list[dict[str, Any]]:
    """Load valid versioned JSON records, skipping malformed or unsupported files."""
    target_dir = Path(directory)
    if not target_dir.is_dir():
        return []
    records = []
    for path in target_dir.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            _validate_record(record)
            if path.stem == record["run_id"]:
                records.append(record)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return sorted(records, key=lambda item: item["created_at"], reverse=True)


def comparison_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten records into a compact, comparable table."""
    rows = []
    for record in records:
        history = record.get("history", [])
        test = record.get("test_metrics") or {}
        best = max(history, key=lambda item: item.get("val_accuracy", float("-inf")), default={})
        rows.append({
            "Run ID": record["run_id"][:8],
            "Created (UTC)": record["created_at"],
            "Dataset": record["dataset_name"].upper(),
            "Architecture": record["architecture"],
            "Device": record["device"],
            "Epochs": len(history),
            "Best validation accuracy": best.get("val_accuracy"),
            "Best validation loss": min(
                (item.get("val_loss", float("inf")) for item in history),
                default=None,
            ),
            "Test accuracy": test.get("accuracy"),
            "Test examples": test.get("examples"),
            "Seed": record["run_config"].get("seed"),
            "Batch size": record["run_config"].get("batch_size"),
            "Learning rate": record["run_config"].get("learning_rate"),
        })
    return rows
