"""Streaming CIFAR-10/100 data pipeline.

Hugging Face Datasets streams examples from remote shards instead of downloading
and unpacking the entire dataset onto the user's machine.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from io import BytesIO
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms

from .model import (
    CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD,
)

DATASET_SPECS = {
    "cifar10": {"repo_id": "uoft-cs/cifar10", "train_size": 50_000, "test_size": 10_000},
    "cifar100": {"repo_id": "uoft-cs/cifar100", "train_size": 50_000, "test_size": 10_000},
}


def dataset_stats(dataset_name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if dataset_name == "cifar100":
        return CIFAR100_MEAN, CIFAR100_STD
    return CIFAR10_MEAN, CIFAR10_STD


def image_transforms(augment: bool = False, dataset_name: str = "cifar10") -> transforms.Compose:
    mean, std = dataset_stats(dataset_name)
    steps = []
    if augment:
        steps.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(steps)


class HuggingFaceImageStream(IterableDataset):
    """Small iterable adapter that transforms remote examples one batch at a time."""

    def __init__(
        self,
        repo_id: str,
        split: str,
        image_key: str,
        label_key: str,
        transform: Callable[[Image.Image], torch.Tensor],
        *,
        limit: int,
        offset: int = 0,
        shuffle: bool = False,
        seed: int = 42,
        shuffle_buffer: int = 10_000,
        stream_factory: Callable[[], Iterable[dict[str, Any]]] | None = None,
    ) -> None:
        super().__init__()
        if limit < 1:
            raise ValueError("limit must be positive")
        if offset < 0:
            raise ValueError("offset cannot be negative")
        self.repo_id = repo_id
        self.split = split
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform
        self.limit = limit
        self.offset = offset
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.stream_factory = stream_factory

    def __len__(self) -> int:
        return self.limit

    def __iter__(self) -> Iterator[tuple[torch.Tensor, int]]:
        if self.stream_factory is not None:
            stream = self.stream_factory()
        else:
            from datasets import load_dataset

            stream = load_dataset(self.repo_id, split=self.split, streaming=True)
        if self.shuffle and hasattr(stream, "shuffle"):
            stream = stream.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)

        yielded = 0
        for index, example in enumerate(stream):
            if index < self.offset:
                continue
            if yielded >= self.limit:
                break
            image = example[self.image_key]
            if isinstance(image, dict):
                raw = image.get("bytes")
                path = image.get("path")
                if raw is not None:
                    image = Image.open(BytesIO(raw))
                elif path:
                    image = Image.open(path)
                else:
                    raise ValueError("Streaming dataset returned an undecodable image")
            if not isinstance(image, Image.Image):
                raise TypeError(f"Expected a PIL image, got {type(image).__name__}")
            image = image.convert("RGB")
            yield self.transform(image), int(example[self.label_key])
            yielded += 1


def _dataset_metadata(dataset_name: str) -> tuple[str, str, str, tuple[str, ...]]:
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    from datasets import load_dataset

    repo_id = DATASET_SPECS[dataset_name]["repo_id"]
    stream = load_dataset(repo_id, split="train", streaming=True)
    features = stream.features
    image_key = "img" if "img" in features else "image" if "image" in features else None
    if image_key is None:
        raise ValueError(f"Could not identify image column in {repo_id}")
    label_key = next((key for key in ("label", "fine_label", "labels") if key in features), None)
    if label_key is None:
        raise ValueError(f"Could not identify class-label column in {repo_id}")
    class_names = getattr(features[label_key], "names", None)
    if not class_names:
        class_names = list(CIFAR10_CLASSES) if dataset_name == "cifar10" else []
    if not class_names:
        raise ValueError(f"Dataset {repo_id} does not expose class names")
    return repo_id, image_key, label_key, tuple(class_names)


def make_loaders(
    dataset_name: str = "cifar10",
    batch_size: int = 64,
    *,
    train_limit: int = 10_000,
    validation_limit: int = 2_500,
    test_limit: int = 5_000,
    augment: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, tuple[str, ...]]:
    """Create train, validation, and held-out test loaders without full downloads.

    The train and validation windows are disjoint sections of a seeded shuffled
    training stream. The separate test split is only used for final evaluation.
    """
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"dataset_name must be one of {tuple(DATASET_SPECS)}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers != 0:
        raise ValueError("Streaming loaders currently require num_workers=0")
    if train_limit < 1 or validation_limit < 1 or test_limit < 1:
        raise ValueError("train_limit, validation_limit, and test_limit must be positive")
    spec = DATASET_SPECS[dataset_name]
    if train_limit + validation_limit > spec["train_size"]:
        raise ValueError("Training and validation limits exceed the training split")

    repo_id, image_key, label_key, class_names = _dataset_metadata(dataset_name)
    train_transform = image_transforms(augment, dataset_name)
    eval_transform = image_transforms(False, dataset_name)

    train_data = HuggingFaceImageStream(
        repo_id, "train", image_key, label_key, train_transform,
        limit=train_limit, shuffle=True, seed=seed,
    )
    validation_data = HuggingFaceImageStream(
        repo_id, "train", image_key, label_key, eval_transform,
        limit=validation_limit, offset=train_limit, shuffle=True, seed=seed,
    )
    test_data = HuggingFaceImageStream(
        repo_id, "test", image_key, label_key, eval_transform,
        limit=min(test_limit, spec["test_size"]), shuffle=False, seed=seed,
    )

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=0, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=0)
    return train_loader, validation_loader, test_loader, class_names
