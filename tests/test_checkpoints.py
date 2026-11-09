"""Offline tests for checkpoint restoration used by the Streamlit upload flow."""

import io

import pytest
import torch

from neural_lab.checkpoints import restore_checkpoint
from neural_lab.model import CifarCNN, save_checkpoint

COMPLETE_CONFIG = {
    "dataset_name": "cifar10",
    "batch_size": 16,
    "train_limit": 1000,
    "validation_limit": 1000,
    "test_limit": 1000,
    "augment": True,
    "seed": 7,
    "learning_rate": 0.001,
    "scheduler_enabled": False,
    "scheduler_patience": 2,
    "scheduler_factor": 0.5,
    "early_stopping_enabled": False,
    "early_stopping_patience": 5,
}


def _checkpoint_bytes(*, config=None, with_optimizer=True, with_scheduler=False):
    model = CifarCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) if with_optimizer else None
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        if with_scheduler and optimizer is not None
        else None
    )
    buffer = io.BytesIO()
    save_checkpoint(
        buffer,
        model,
        epoch=4,
        optimizer=optimizer,
        scheduler=scheduler,
        history=[{"epoch": 1.0}, {"epoch": 2.0}, {"epoch": 3.0}, {"epoch": 4.0}],
        run_config=config,
    )
    return buffer.getvalue()


def test_restore_resumable_checkpoint_restores_model_optimizer_and_progress():
    restored = restore_checkpoint(_checkpoint_bytes(config=COMPLETE_CONFIG), device="cpu")

    assert restored["resumable"] is True
    assert restored["current_epoch"] == 4
    assert len(restored["history"]) == 4
    assert restored["optimizer"] is not None
    assert restored["scheduler"] is None
    assert restored["run_config"] == COMPLETE_CONFIG
    assert all(parameter.device.type == "cpu" for parameter in restored["model"].parameters())


def test_restore_scheduler_enabled_checkpoint_restores_scheduler_state():
    config = {**COMPLETE_CONFIG, "scheduler_enabled": True}
    restored = restore_checkpoint(
        _checkpoint_bytes(config=config, with_scheduler=True)
    )

    assert restored["resumable"] is True
    assert restored["scheduler"] is not None
    assert restored["scheduler"].state_dict()["factor"] == 0.5


def test_restore_incomplete_checkpoint_enters_inference_only_mode():
    restored = restore_checkpoint(
        _checkpoint_bytes(config={"dataset_name": "cifar10"}, with_optimizer=True)
    )

    assert restored["resumable"] is False
    assert restored["optimizer"] is None
    assert restored["scheduler"] is None
    assert restored["current_epoch"] == 0
    assert restored["history"] == []
    assert restored["run_config"] is None
    assert restored["class_names_verified"] is True


def test_restore_rejects_invalid_metadata_before_returning_replacement_state():
    data = _checkpoint_bytes(config=COMPLETE_CONFIG)
    payload = torch.load(io.BytesIO(data), weights_only=True)
    payload["class_names"][0] = "unrelated_label"
    buffer = io.BytesIO()
    torch.save(payload, buffer)

    with pytest.raises(ValueError, match="canonical CIFAR class set"):
        restore_checkpoint(buffer.getvalue())


def test_restore_rejects_oversized_and_empty_uploads():
    with pytest.raises(ValueError, match="25 MiB"):
        restore_checkpoint(b"x" * (25 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="empty"):
        restore_checkpoint(b"")


def test_restore_marks_legacy_custom_labels_as_unverified_inference_only():
    model = CifarCNN()
    buffer = io.BytesIO()
    save_checkpoint(
        buffer,
        model,
        epoch=2,
        class_names=tuple(f"custom_{index}" for index in range(10)),
    )
    restored = restore_checkpoint(buffer.getvalue())

    assert restored["class_names_verified"] is False
    assert restored["resumable"] is False
    assert restored["optimizer"] is None
