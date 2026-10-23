"""CIFAR-10 datasets and loaders."""
from __future__ import annotations

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .model import CIFAR10_MEAN, CIFAR10_STD


def image_transforms(augment: bool = False) -> transforms.Compose:
    steps = []
    if augment:
        steps.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return transforms.Compose(steps)


def make_loaders(
    root: str = "data",
    batch_size: int = 128,
    *,
    train_limit: int | None = None,
    test_limit: int | None = None,
    augment: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create train/test loaders; dataset download occurs only when needed."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    train_data = datasets.CIFAR10(
        root=root, train=True, download=True, transform=image_transforms(augment)
    )
    test_data = datasets.CIFAR10(
        root=root, train=False, download=True, transform=image_transforms(False)
    )

    if train_limit is not None:
        if train_limit < 1:
            raise ValueError("train_limit must be positive or None")
        train_data = Subset(train_data, range(min(train_limit, len(train_data))))
    if test_limit is not None:
        if test_limit < 1:
            raise ValueError("test_limit must be positive or None")
        test_data = Subset(test_data, range(min(test_limit, len(test_data))))

    generator = __import__("torch").Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=False, generator=generator,
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, test_loader
