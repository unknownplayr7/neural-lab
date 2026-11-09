"""Model architecture, device selection, and checkpoint utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

import torch
from torch import nn

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
CIFAR100_CLASSES = (
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
    "worm",
)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

RESUMABLE_CONFIG_FIELDS = frozenset({
    "dataset_name", "batch_size", "train_limit", "validation_limit", "test_limit",
    "augment", "seed", "learning_rate", "scheduler_enabled", "scheduler_patience",
    "scheduler_factor", "early_stopping_enabled", "early_stopping_patience",
})


class CifarCNN(nn.Module):
    """Compact CNN for 32x32 RGB images from CIFAR-10 or CIFAR-100.

    Adaptive pooling makes the classifier independent of the exact spatial
    size after the convolutional feature extractor.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.25) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected NCHW image batch, got shape {tuple(images.shape)}")
        if images.shape[1] != 3:
            raise ValueError(f"Expected 3 RGB channels, got {images.shape[1]}")
        return self.classifier(self.features(images))


def available_device(preference: str = "auto") -> torch.device:
    """Resolve auto/mps/cpu without selecting unavailable accelerators."""
    preference = preference.lower()
    if preference not in {"auto", "cpu", "mps"}:
        raise ValueError("preference must be one of: auto, cpu, mps")
    if preference == "cpu":
        return torch.device("cpu")
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if preference == "mps":
        if not mps_available:
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")



def _cpu_tree(value: Any) -> Any:
    """Move tensors nested in optimizer state to CPU for portable checkpoints."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def save_checkpoint(
    checkpoint_file: BinaryIO,
    model: CifarCNN,
    *,
    epoch: int,
    class_names: tuple[str, ...] = CIFAR10_CLASSES,
    accuracy: float | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    history: list[dict[str, float]] | None = None,
    run_config: dict[str, Any] | None = None,
    scheduler: Any | None = None,
    early_stopping_state: dict[str, Any] | None = None,
) -> None:
    """Serialize a checkpoint to an already-open binary file.

    Filesystem path selection and directory creation belong to the caller.
    """
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "epoch": int(epoch),
        "class_names": list(class_names),
        "accuracy": accuracy,
        "architecture": "CifarCNN",
        "history": history or [],
        "run_config": run_config or {},
        "optimizer_state_dict": _cpu_tree(optimizer.state_dict()) if optimizer is not None else None,
        "scheduler_state_dict": _cpu_tree(scheduler.state_dict()) if scheduler is not None else None,
        "early_stopping_state": dict(early_stopping_state or {}),
    }
    torch.save(payload, checkpoint_file)


def validate_checkpoint_payload(payload: Any) -> dict[str, Any]:
    """Validate checkpoint metadata before model or optimizer state is restored."""
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a mapping")
    version = payload.get("format_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("Unsupported checkpoint format_version; expected 1")
    if payload.get("architecture") != "CifarCNN":
        raise ValueError("Unsupported checkpoint architecture")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint model_state_dict must be a non-empty mapping")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise ValueError("Checkpoint model_state_dict must map string keys to tensors")

    epoch = payload.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Checkpoint epoch must be a non-negative integer")
    classes = payload.get("class_names")
    if not isinstance(classes, (list, tuple)) or len(classes) not in (10, 100):
        raise ValueError("Checkpoint class_names must contain 10 or 100 class names")
    if not all(isinstance(name, str) and name.strip() for name in classes):
        raise ValueError("Checkpoint class_names must contain non-empty strings")
    normalized_classes = [name.strip().casefold() for name in classes]
    if len(set(normalized_classes)) != len(normalized_classes):
        raise ValueError("Checkpoint class_names must be unique")

    accuracy = payload.get("accuracy")
    if accuracy is not None and (
        isinstance(accuracy, bool) or not isinstance(accuracy, (int, float))
        or not torch.isfinite(torch.tensor(float(accuracy))).item()
        or not 0.0 <= float(accuracy) <= 1.0
    ):
        raise ValueError("Checkpoint accuracy must be between 0 and 1")

    history = payload.get("history", [])
    if not isinstance(history, list):
        raise ValueError("Checkpoint history must be a list")
    for index, row in enumerate(history):
        if not isinstance(row, dict):
            raise ValueError(f"Checkpoint history entry {index} must be a mapping")
        for key, value in row.items():
            if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Checkpoint history entry {index} contains an invalid value")
            if not torch.isfinite(torch.tensor(float(value))).item():
                raise ValueError(f"Checkpoint history entry {index} contains a non-finite value")

    config = payload.get("run_config", {})
    if not isinstance(config, dict):
        raise ValueError("Checkpoint run_config must be a mapping")
    expected_types = {
        "dataset_name": str, "batch_size": int, "train_limit": int,
        "validation_limit": int, "test_limit": int, "augment": bool,
        "seed": int, "learning_rate": (int, float),
        "scheduler_enabled": bool, "scheduler_patience": int,
        "scheduler_factor": (int, float), "early_stopping_enabled": bool,
        "early_stopping_patience": int,
    }
    for key, value in config.items():
        expected = expected_types.get(key)
        if expected is None:
            raise ValueError(f"Checkpoint run_config contains unknown field {key!r}")
        if (isinstance(value, bool) and expected is not bool) or not isinstance(value, expected):
            raise ValueError(f"Checkpoint run_config field {key!r} has an invalid type")
    if config.get("dataset_name", "cifar10") not in {"cifar10", "cifar100"}:
        raise ValueError("Checkpoint run_config dataset_name is unsupported")
    if "dataset_name" in config:
        expected_class_count = 10 if config["dataset_name"] == "cifar10" else 100
        if len(classes) != expected_class_count:
            raise ValueError("Checkpoint dataset_name does not match class count")
        canonical_classes = CIFAR10_CLASSES if config["dataset_name"] == "cifar10" else CIFAR100_CLASSES
        if set(normalized_classes) != {name.casefold() for name in canonical_classes}:
            raise ValueError(
                "Checkpoint class_names must match the canonical CIFAR class set; "
                "class ordering may differ"
            )
    for key in ("batch_size", "train_limit", "validation_limit", "test_limit"):
        if key in config and config[key] < 1:
            raise ValueError(f"Checkpoint run_config field {key!r} must be positive")
    if "seed" in config and config["seed"] < 0:
        raise ValueError("Checkpoint run_config seed cannot be negative")
    if "learning_rate" in config and config["learning_rate"] <= 0:
        raise ValueError("Checkpoint run_config learning_rate must be positive")
    for key in ("scheduler_patience", "early_stopping_patience"):
        if key in config and config[key] < 1:
            raise ValueError(f"Checkpoint run_config field {key!r} must be positive")
    if "scheduler_factor" in config and not 0.0 < config["scheduler_factor"] < 1.0:
        raise ValueError("Checkpoint run_config scheduler_factor must be between 0 and 1")

    scheduler_state = payload.get("scheduler_state_dict")
    if scheduler_state is not None and not isinstance(scheduler_state, dict):
        raise ValueError("Checkpoint scheduler_state_dict must be a mapping or null")
    early_state = payload.get("early_stopping_state", {})
    if not isinstance(early_state, dict):
        raise ValueError("Checkpoint early_stopping_state must be a mapping")
    if early_state:
        best_loss = early_state.get("best_loss")
        bad_epochs = early_state.get("bad_epochs")
        stopped = early_state.get("stopped")
        if (isinstance(best_loss, bool) or not isinstance(best_loss, (int, float))
                or isinstance(bad_epochs, bool) or not isinstance(bad_epochs, int) or bad_epochs < 0
                or not isinstance(stopped, bool)):
            raise ValueError("Checkpoint early_stopping_state is malformed")
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer_state is not None:
        if not isinstance(optimizer_state, dict):
            raise ValueError("Checkpoint optimizer_state_dict must be a mapping or null")
        if not isinstance(optimizer_state.get("state"), dict):
            raise ValueError("Checkpoint optimizer state must contain a state mapping")
        groups = optimizer_state.get("param_groups")
        if not isinstance(groups, list) or not all(
            isinstance(group, dict) and isinstance(group.get("params"), list)
            for group in groups
        ):
            raise ValueError("Checkpoint optimizer state must contain valid parameter groups")
    if optimizer_state is not None and RESUMABLE_CONFIG_FIELDS.issubset(config):
        if config["scheduler_enabled"] != (scheduler_state is not None):
            raise ValueError("Checkpoint scheduler metadata does not match scheduler_enabled")
    return payload


def checkpoint_class_names_verified(payload: dict[str, Any]) -> bool:
    """Whether checkpoint labels match a canonical CIFAR class set."""
    classes = payload.get("class_names")
    if not isinstance(classes, (list, tuple)) or len(classes) not in (10, 100):
        return False
    if not all(isinstance(name, str) and name.strip() for name in classes):
        return False
    expected = CIFAR10_CLASSES if len(classes) == 10 else CIFAR100_CLASSES
    normalized = {name.strip().casefold() for name in classes}
    if len(normalized) != len(classes):
        return False
    if normalized != {name.casefold() for name in expected}:
        return False
    config = payload.get("run_config")
    if isinstance(config, dict) and "dataset_name" in config:
        expected_dataset = "cifar10" if len(classes) == 10 else "cifar100"
        return config["dataset_name"] == expected_dataset
    return True


def checkpoint_is_resumable(payload: dict[str, Any]) -> bool:
    """Whether a validated checkpoint has enough state to resume training exactly."""
    config = payload.get("run_config")
    return (
        payload.get("optimizer_state_dict") is not None
        and checkpoint_class_names_verified(payload)
        and isinstance(config, dict)
        and RESUMABLE_CONFIG_FIELDS.issubset(config)
    )


def checkpoint_resume_progress(payload: dict[str, Any]) -> tuple[int, list[dict[str, float]]]:
    """Return saved progress only when the checkpoint is fully resumable."""
    if not checkpoint_is_resumable(payload):
        return 0, []
    return int(payload["epoch"]), list(payload.get("history", []))


def load_checkpoint(path: str | Path, model: CifarCNN, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load and validate a checkpoint using weights-only deserialization."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    validate_checkpoint_payload(payload)
    if len(payload["class_names"]) != model.classifier[-1].out_features:
        raise ValueError("Checkpoint class count does not match the model")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return payload
