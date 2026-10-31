from pathlib import Path

import pytest
import torch
from PIL import Image

from neural_lab.data import HuggingFaceImageStream, evaluation_loader_config
from neural_lab.model import CifarCNN, load_checkpoint, save_checkpoint


def _records(count=16):
    return [
        {"image": Image.new("RGB", (4, 4), (index, index, index)), "label": index % 10}
        for index in range(count)
    ]


def _pixel_value(image):
    return torch.tensor([float(image.getpixel((0, 0))[0])])


def _order_for_epoch(epoch):
    dataset = HuggingFaceImageStream(
        "fixture", "train", "image", "label", _pixel_value,
        limit=8, shuffle=True, seed=73, stream_factory=_records,
    )
    dataset.set_epoch(epoch)
    return [int(image.item()) for image, _label in dataset]


def test_epoch_shuffle_changes_and_is_reproducible():
    epoch_zero = _order_for_epoch(0)
    epoch_one = _order_for_epoch(1)
    assert epoch_zero != epoch_one
    assert epoch_zero == _order_for_epoch(0)
    assert epoch_one == _order_for_epoch(1)


def test_train_validation_partitions_remain_disjoint_and_validation_is_stable():
    def read(offset, shuffle, epoch):
        dataset = HuggingFaceImageStream(
            "fixture", "train", "image", "label", _pixel_value,
            limit=8, offset=offset, shuffle=shuffle, seed=19, stream_factory=_records,
        )
        dataset.set_epoch(epoch)
        return [int(image.item()) for image, _label in dataset]

    train_zero = read(0, True, 0)
    train_one = read(0, True, 1)
    validation_zero = read(8, False, 0)
    validation_one = read(8, False, 1)
    assert set(train_zero).isdisjoint(validation_zero)
    assert set(train_one) == set(train_zero)
    assert validation_zero == validation_one


def test_saved_run_settings_are_used_for_evaluation():
    original = {
        "dataset_name": "cifar100", "batch_size": 32, "train_limit": 5000,
        "validation_limit": 2500, "test_limit": 1000, "augment": True,
        "seed": 991, "learning_rate": 0.0003,
    }
    assert evaluation_loader_config(original) == {
        "dataset_name": "cifar100", "batch_size": 32, "train_limit": 5000,
        "validation_limit": 2500, "test_limit": 1000, "augment": True,
        "seed": 991,
    }
    with pytest.raises(ValueError, match="missing"):
        evaluation_loader_config({"dataset_name": "cifar10"})


def test_checkpoint_resume_workflow_restores_weights_optimizer_and_run_config(tmp_path: Path):
    model = CifarCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(2, 3, 32, 32)).sum().backward()
    optimizer.step()
    history = [{
        "epoch": 1.0, "train_loss": 1.2, "train_accuracy": 0.5,
        "val_loss": 1.3, "val_accuracy": 0.4, "seconds": 2.0,
    }]
    config = {
        "dataset_name": "cifar10", "batch_size": 16, "train_limit": 1000,
        "validation_limit": 1000, "test_limit": 1000, "augment": True,
        "seed": 73, "learning_rate": 0.001,
    }
    path = save_checkpoint(
        tmp_path / "resume.pt", model, epoch=1, optimizer=optimizer,
        history=history, run_config=config, accuracy=0.4,
    )

    restored = CifarCNN()
    payload = load_checkpoint(path, restored)
    resumed_optimizer = torch.optim.AdamW(
        restored.parameters(), lr=payload["run_config"]["learning_rate"]
    )
    resumed_optimizer.load_state_dict(payload["optimizer_state_dict"])

    assert payload["epoch"] == 1
    assert payload["history"] == history
    assert payload["run_config"] == config
    assert len(resumed_optimizer.state) > 0
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[key])
