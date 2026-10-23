"""Training and evaluation primitives independent of the Streamlit UI."""
from __future__ import annotations

import random
import time
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    on_batch: Callable[[int, int, float], None] | None = None,
) -> dict[str, float]:
    """Run one train/eval epoch. Optimizer presence selects training mode."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    started = time.perf_counter()
    context = torch.enable_grad if training else torch.no_grad

    with context():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            count = labels.size(0)
            total_loss += float(loss.detach().item()) * count
            total_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
            total_seen += count
            if on_batch is not None:
                on_batch(batch_index, len(loader), total_loss / total_seen)

    if total_seen == 0:
        raise ValueError("DataLoader produced no batches")
    return {
        "loss": total_loss / total_seen,
        "accuracy": total_correct / total_seen,
        "examples": float(total_seen),
        "seconds": time.perf_counter() - started,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    on_epoch: Callable[[int, dict[str, float], dict[str, float]], None] | None = None,
    on_batch: Callable[[int, int, float], None] | None = None,
) -> list[dict[str, float]]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, on_batch=on_batch
        )
        test_metrics = run_epoch(model, test_loader, criterion, device)
        record = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "seconds": train_metrics["seconds"],
        }
        history.append(record)
        if on_epoch is not None:
            on_epoch(epoch, train_metrics, test_metrics)
    return history



def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> dict[str, object]:
    """Evaluate once and return aggregate metrics plus a true-row confusion matrix."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_loss = 0.0
    total_seen = 0
    started = time.perf_counter()

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            total_loss += float(criterion(logits, labels).item())
            total_seen += int(labels.size(0))
            encoded = labels * num_classes + predictions
            confusion += torch.bincount(encoded.cpu(), minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    if total_seen == 0:
        raise ValueError("DataLoader produced no batches")
    correct = int(confusion.diag().sum().item())
    per_class = []
    for index in range(num_classes):
        support = int(confusion[index].sum().item())
        per_class.append({
            "class_index": index,
            "support": support,
            "accuracy": (int(confusion[index, index].item()) / support) if support else None,
        })
    return {
        "loss": total_loss / total_seen,
        "accuracy": correct / total_seen,
        "examples": total_seen,
        "seconds": time.perf_counter() - started,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }
