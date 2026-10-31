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
    assert available_device("auto").type in {"cpu", "mps"}
    with pytest.raises(ValueError, match="preference"):
        available_device("cuda")


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
    torch.save({"format_version": 1, "architecture": "CifarCNN"}, path)
    with pytest.raises(ValueError, match="model_state_dict"):
        load_checkpoint(path, CifarCNN())



def test_checkpoint_preserves_optimizer_state_and_history(tmp_path: Path):
    model = CifarCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.randn(2, 3, 32, 32)).sum()
    loss.backward()
    optimizer.step()
    history = [{"epoch": 1.0, "train_loss": 2.0, "val_loss": 2.1}]
    path = save_checkpoint(
        tmp_path / "resume.pt", model, epoch=1, optimizer=optimizer, history=history
    )
    restored = CifarCNN()
    payload = load_checkpoint(path, restored)
    assert payload["history"] == history
    assert payload["optimizer_state_dict"] is not None
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(payload["optimizer_state_dict"])
    assert len(resumed_optimizer.state) > 0


def test_checkpoint_loader_rejects_unknown_architecture(tmp_path: Path):
    path = tmp_path / "unknown.pt"
    torch.save({"architecture": "UnknownNet", "model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="architecture"):
        load_checkpoint(path, CifarCNN())


class _UnexpectedPickle:
    def __init__(self, marker: str):
        self.marker = marker

    def __reduce__(self):
        import os
        return os.system, (f"touch {self.marker}",)


def test_checkpoint_loader_never_executes_pickle_payload(tmp_path: Path):
    marker = tmp_path / "pickle-executed"
    path = tmp_path / "untrusted.pt"
    torch.save({"payload": _UnexpectedPickle(str(marker))}, path)
    with pytest.raises(Exception):
        load_checkpoint(path, CifarCNN())
    assert not marker.exists()



def test_checkpoint_rejects_unsupported_format_version(tmp_path: Path):
    path = save_checkpoint(tmp_path / "valid.pt", CifarCNN(), epoch=0)
    payload = torch.load(path, weights_only=True)
    payload["format_version"] = 99
    torch.save(payload, path)
    with pytest.raises(ValueError, match="format_version"):
        load_checkpoint(path, CifarCNN())


@pytest.mark.parametrize("field,value,message", [
    ("epoch", -1, "epoch"),
    ("class_names", ["one"], "class_names"),
    ("history", {"epoch": 1}, "history"),
    ("run_config", [], "run_config"),
    ("optimizer_state_dict", [], "optimizer_state_dict"),
    ("accuracy", 1.5, "accuracy"),
])
def test_checkpoint_rejects_malformed_metadata(tmp_path: Path, field, value, message):
    path = save_checkpoint(tmp_path / "valid.pt", CifarCNN(), epoch=0)
    payload = torch.load(path, weights_only=True)
    payload[field] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match=message):
        load_checkpoint(path, CifarCNN())


def test_checkpoint_rejects_class_count_mismatch(tmp_path: Path):
    path = save_checkpoint(tmp_path / "valid.pt", CifarCNN(), epoch=0)
    with pytest.raises(ValueError, match="class count"):
        load_checkpoint(path, CifarCNN(num_classes=100))
