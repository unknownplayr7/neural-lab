"""Pure checkpoint restoration helpers shared by the Streamlit UI and tests."""

from __future__ import annotations

import io
from typing import Any

import torch

from neural_lab.model import (
    CifarCNN,
    build_model,
    RESUMABLE_CONFIG_FIELDS,
    checkpoint_class_names_verified,
    checkpoint_is_resumable,
    checkpoint_resume_progress,
    validate_checkpoint_payload,
)

def can_save_resumable_checkpoint(
    optimizer: Any,
    run_config: Any,
    class_names: Any,
) -> bool:
    """Whether the current app state can truthfully be saved as resumable."""
    if optimizer is None or not isinstance(run_config, dict):
        return False
    if not RESUMABLE_CONFIG_FIELDS.issubset(run_config):
        return False
    payload = {
        "class_names": class_names,
        "run_config": run_config,
        "optimizer_state_dict": {},
    }
    return checkpoint_class_names_verified(payload)


MAX_CHECKPOINT_BYTES = 25 * 1024 * 1024
DEFAULT_EARLY_STOPPING_STATE = {
    "best_loss": float("inf"),
    "bad_epochs": 0,
    "stopped": False,
}


def restore_checkpoint(
    checkpoint_bytes: bytes,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Validate and fully construct replacement app state without mutating session state."""
    if not isinstance(checkpoint_bytes, bytes):
        raise TypeError("Checkpoint upload must be bytes")
    if len(checkpoint_bytes) > MAX_CHECKPOINT_BYTES:
        raise ValueError("Checkpoint exceeds the 25 MiB upload limit.")
    if not checkpoint_bytes:
        raise ValueError("Checkpoint upload is empty.")

    payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    validate_checkpoint_payload(payload)

    class_names = tuple(payload["class_names"])
    architecture = payload.get("architecture", "CifarCNN")
    model = build_model(architecture, num_classes=len(class_names))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device)

    saved_config = payload.get("run_config") or {}
    class_names_verified = checkpoint_class_names_verified(payload)
    resumable = checkpoint_is_resumable(payload)
    model_dataset = (
        saved_config.get("dataset_name")
        if saved_config.get("dataset_name") in {"cifar10", "cifar100"}
        else ("cifar100" if len(class_names) == 100 else "cifar10")
    )

    optimizer = None
    scheduler = None
    run_config = None
    early_stopping_state = dict(DEFAULT_EARLY_STOPPING_STATE)
    if resumable:
        run_config = dict(saved_config)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(run_config["learning_rate"]),
            weight_decay=1e-4,
        )
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if run_config["scheduler_enabled"]:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(run_config["scheduler_factor"]),
                patience=int(run_config["scheduler_patience"]),
            )
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        early_stopping_state = dict(
            payload.get("early_stopping_state") or DEFAULT_EARLY_STOPPING_STATE
        )

    restored_epoch, restored_history = checkpoint_resume_progress(payload)
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "early_stopping_state": early_stopping_state,
        "class_names": class_names,
        "model_dataset": model_dataset,
        "current_epoch": restored_epoch,
        "history": restored_history,
        "run_config": run_config,
        "last_eval": None,
        "resumable": resumable,
        "class_names_verified": class_names_verified,
        "experiment_record_id": payload.get("experiment_record_id"),
    }
