from pathlib import Path

import pytest
import torch

from neural_lab.model import CIFAR10_CLASSES, CifarCNN, available_device, load_checkpoint, save_checkpoint


def test_model_returns_ten_logits_for_cifar_images():
    model = CifarCNN()
    result = model(torch.randn(4, 3, 32, 32))
    assert result.shape == (4, len(CIFAR10_CLASSES))


def test_model_rejects_wrong_input_rank_and_channels():
    model = CifarCNN()
    with pytest.raises(ValueError, match="NCHW"):
        model(torch.randn(3, 32, 32))
    with pytest.raises(ValueError, match="RGB"):
        model(torch.randn(2, 1, 32, 32))


def test_model_validates_configuration():
    with pytest.raises(ValueError, match="num_classes"):
        CifarCNN(num_classes=1)
    with pytest.raises(ValueError, match="dropout"):
        CifarCNN(dropout=1.0)


def test_auto_and_cpu_devices_are_available():
    assert available_device("cpu").type == "cpu"
    assert available_device("auto").type in {"cpu", "cuda", "mps"}


def test_checkpoint_round_trip(tmp_path: Path):
    original = CifarCNN()
    path = save_checkpoint(tmp_path / "model.pt", original, epoch=3, accuracy=0.75)
    restored = CifarCNN()
    metadata = load_checkpoint(path, restored)
    assert metadata["epoch"] == 3
    assert metadata["class_names"] == list(CIFAR10_CLASSES)
    for key, value in original.state_dict().items():
        assert torch.equal(value.cpu(), restored.state_dict()[key].cpu())


def test_checkpoint_rejects_unexpected_payload(tmp_path: Path):
    path = tmp_path / "bad.pt"
    torch.save({"not_model": {}}, path)
    with pytest.raises(ValueError, match="model_state_dict"):
        load_checkpoint(path, CifarCNN())
