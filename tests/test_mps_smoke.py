"""Optional functional smoke test for Apple Silicon's MPS backend."""

import pytest
import torch
from torch import nn

from neural_lab.model import CifarCNN, available_device

MPS_AVAILABLE = bool(
    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
)


@pytest.mark.skipif(not MPS_AVAILABLE, reason="PyTorch MPS backend is unavailable on this machine")
def test_mps_training_step_and_inference_are_finite():
    device = available_device("mps")
    model = CifarCNN(num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images = torch.randn(2, 3, 32, 32, device=device)
    labels = torch.tensor([1, 7], dtype=torch.long, device=device)

    model.train()
    logits = model(images)
    loss = nn.CrossEntropyLoss()(logits, labels)
    assert torch.isfinite(logits).all().item()
    assert torch.isfinite(loss).item()
    loss.backward()
    optimizer.step()

    assert all(parameter.device.type == "mps" for parameter in model.parameters())
    model.eval()
    with torch.no_grad():
        predictions = model(images)
    assert predictions.device.type == "mps"
    assert torch.isfinite(predictions).all().item()


def test_cpu_device_path_remains_available():
    device = available_device("cpu")
    model = CifarCNN(num_classes=10).to(device)
    images = torch.randn(2, 3, 32, 32, device=device)
    with torch.no_grad():
        predictions = model(images)
    assert predictions.device.type == "cpu"
    assert torch.isfinite(predictions).all().item()
