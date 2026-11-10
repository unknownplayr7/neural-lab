"""Tests for persisted reproducibility metadata and run comparison."""

import json
from pathlib import Path

import pytest

from neural_lab.experiments import (
    comparison_rows,
    create_experiment_record,
    load_experiment_records,
    save_experiment_record,
    update_experiment_record,
)


CONFIG = {
    "dataset_name": "cifar10",
    "architecture": "CifarCNN",
    "batch_size": 64,
    "train_limit": 10000,
    "validation_limit": 2500,
    "test_limit": 5000,
    "augment": True,
    "seed": 42,
    "learning_rate": 0.001,
}


def test_experiment_record_contains_reproducibility_metadata(tmp_path: Path):
    record = create_experiment_record(CONFIG, architecture="CifarCNN", device="mps")
    assert len(record["run_id"]) == 36
    assert record["dataset_source"].startswith("uoft-cs/cifar10")
    assert "validation window" in record["split_strategy"]
    assert record["run_config"]["seed"] == 42
    assert record["software"]["python"]
    assert record["software"]["pytorch"]
    assert record["device"] == "mps"


def test_experiment_records_round_trip_and_comparison(tmp_path: Path):
    record = create_experiment_record(CONFIG, architecture="CifarCNN", device="cpu")
    history = [
        {"epoch": 1.0, "train_loss": 1.5, "val_loss": 1.4, "val_accuracy": 0.55},
        {"epoch": 2.0, "train_loss": 1.0, "val_loss": 1.1, "val_accuracy": 0.7},
    ]
    record = update_experiment_record(
        record, history, {"accuracy": 0.68, "loss": 1.2, "examples": 5000}
    )
    saved = save_experiment_record(record, tmp_path)
    assert saved.name == record["run_id"] + ".json"
    loaded = load_experiment_records(tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["run_id"] == record["run_id"]
    rows = comparison_rows(loaded)
    assert rows[0]["Best validation accuracy"] == 0.7
    assert rows[0]["Best validation loss"] == 1.1
    assert rows[0]["Test accuracy"] == 0.68
    assert rows[0]["Epochs"] == 2


def test_loader_skips_malformed_and_unsupported_records(tmp_path: Path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "unsupported.json").write_text(
        json.dumps({"schema_version": 99}), encoding="utf-8"
    )
    assert load_experiment_records(tmp_path) == []


@pytest.mark.parametrize(
    "config,architecture",
    [
        ({"dataset_name": "unknown"}, "CifarCNN"),
        (CONFIG, "UnknownArchitecture"),
    ],
)
def test_create_record_rejects_unsupported_values(config, architecture):
    with pytest.raises(ValueError):
        create_experiment_record(config, architecture=architecture, device="cpu")


def test_record_rejects_non_json_safe_values():
    record = create_experiment_record(CONFIG, architecture="CifarCNN", device="cpu")
    record["unsafe"] = float("nan")
    with pytest.raises(ValueError):
        save_experiment_record(record)
