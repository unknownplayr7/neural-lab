"""Model architecture, device selection, and checkpoint utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class CifarCNN(nn.Module):
    """Compact CNN designed for 32x32 RGB CIFAR-10 images.

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
    """Resolve auto/cuda/mps/cpu without selecting unavailable accelerators."""
    preference = preference.lower()
    if preference not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("preference must be one of: auto, cpu, cuda, mps")
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if preference == "mps":
        if not mps_available:
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
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
    path: str | Path,
    model: CifarCNN,
    *,
    epoch: int,
    class_names: tuple[str, ...] = CIFAR10_CLASSES,
    accuracy: float | None = None,
    class_names: tuple[str, ...] = CIFAR10_CLASSES,
    optimizer: torch.optim.Optimizer | None = None,
    history: list[dict[str, float]] | None = None,
) -> Path:
    """Save weights plus small, explicit metadata."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "epoch": int(epoch),
        "class_names": list(class_names),
        "accuracy": accuracy,
        "architecture": "CifarCNN",
        "history": history or [],
        "optimizer_state_dict": _cpu_tree(optimizer.state_dict()) if optimizer is not None else None,
    }
    torch.save(payload, target)
    return target


def load_checkpoint(path: str | Path, model: CifarCNN, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint with weights-only deserialization when supported."""
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # Compatibility with older supported PyTorch releases.
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Not a Neural Lab checkpoint: model_state_dict is missing")
    model.load_state_dict(payload["model_state_dict"])
    return payload
