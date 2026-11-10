"""Architecture factory and residual-network checkpoint regression tests."""

import io

import pytest
import torch

from neural_lab.checkpoints import restore_checkpoint
from neural_lab.model import CifarCNN, CifarResNet, build_model, save_checkpoint


@pytest.mark.parametrize("architecture", ["CifarCNN", "CifarResNet"])
@pytest.mark.parametrize("num_classes", [10, 100])
def test_supported_architectures_return_expected_logits(architecture, num_classes):
    model = build_model(architecture, num_classes=num_classes)
    outputs = model(torch.randn(2, 3, 32, 32))
    assert outputs.shape == (2, num_classes)
    assert model.architecture_name == architecture


def test_residual_model_rejects_invalid_input_shape():
    model = CifarResNet()
    with pytest.raises(ValueError, match="NCHW"):
        model(torch.randn(3, 32, 32))
    with pytest.raises(ValueError, match="RGB"):
        model(torch.randn(2, 1, 32, 32))


def test_model_factory_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="Unsupported"):
        build_model("ArbitraryModel", num_classes=10)


def test_residual_checkpoint_restores_matching_architecture():
    original = CifarResNet()
    buffer = io.BytesIO()
    save_checkpoint(buffer, original, epoch=2)
    restored = restore_checkpoint(buffer.getvalue(), device="cpu")
    assert isinstance(restored["model"], CifarResNet)
    assert restored["model"].architecture_name == "CifarResNet"
    for name, tensor in original.state_dict().items():
        assert torch.equal(tensor.cpu(), restored["model"].state_dict()[name].cpu())


def test_legacy_checkpoint_without_architecture_metadata_defaults_to_cnn():
    model = CifarCNN()
    buffer = io.BytesIO()
    save_checkpoint(buffer, model, epoch=0)
    payload = torch.load(io.BytesIO(buffer.getvalue()), weights_only=True)
    payload.pop("architecture")
    legacy = io.BytesIO()
    torch.save(payload, legacy)
    restored = restore_checkpoint(legacy.getvalue(), device="cpu")
    assert isinstance(restored["model"], CifarCNN)


def test_checkpoint_rejects_disagreeing_architecture_metadata():
    model = CifarResNet()
    buffer = io.BytesIO()
    save_checkpoint(
        buffer,
        model,
        epoch=0,
        run_config={"architecture": "CifarCNN"},
    )
    with pytest.raises(ValueError, match="does not match"):
        restore_checkpoint(buffer.getvalue(), device="cpu")
