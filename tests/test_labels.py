"""Tests for safe labels when checkpoint class-index semantics are uncertain."""

import io

import pytest
import torch

from neural_lab.checkpoints import restore_checkpoint
from neural_lab.labels import display_class_name
from neural_lab.model import (
    CIFAR10_CLASSES,
    CIFAR100_CLASSES,
    CifarCNN,
    save_checkpoint,
)


def _legacy_checkpoint(class_names):
    buffer = io.BytesIO()
    save_checkpoint(
        buffer,
        CifarCNN(num_classes=len(class_names)),
        epoch=2,
        class_names=tuple(class_names),
    )
    return restore_checkpoint(buffer.getvalue(), device="cpu")


@pytest.mark.parametrize(
    "legacy_names",
    [
        tuple(reversed(CIFAR10_CLASSES)),
        tuple(f"custom_{index}" for index in range(10)),
    ],
    ids=["reordered-cifar-labels", "custom-labels"],
)
def test_legacy_checkpoint_labels_are_displayed_as_output_indices(legacy_names):
    restored = _legacy_checkpoint(legacy_names)

    assert restored["class_names_verified"] is False
    assert restored["resumable"] is False
    assert display_class_name(0, restored["class_names"], restored["class_names_verified"]) == "Output 0"
    assert display_class_name(9, restored["class_names"], restored["class_names_verified"]) == "Output 9"


@pytest.mark.parametrize("class_names", [CIFAR10_CLASSES, CIFAR100_CLASSES])
def test_verified_canonical_labels_keep_semantic_names(class_names):
    for index in (0, len(class_names) - 1):
        assert display_class_name(index, class_names, True) == class_names[index]


def test_display_class_name_rejects_invalid_indices():
    with pytest.raises(IndexError):
        display_class_name(10, CIFAR10_CLASSES, False)
    with pytest.raises(TypeError):
        display_class_name(True, CIFAR10_CLASSES, False)
