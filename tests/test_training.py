import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from neural_lab.training import run_epoch, seed_everything


def tiny_loader():
    torch.manual_seed(7)
    images = torch.randn(8, 3, 8, 8)
    labels = torch.randint(0, 3, (8,))
    return DataLoader(TensorDataset(images, labels), batch_size=4)


def test_eval_epoch_reports_finite_metrics_and_does_not_train():
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3))
    model.train()
    result = run_epoch(model, tiny_loader(), nn.CrossEntropyLoss(), torch.device("cpu"))
    assert model.training is False
    assert result["examples"] == 8
    assert 0 <= result["accuracy"] <= 1
    assert result["loss"] > 0


def test_train_epoch_updates_weights():
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model[1].weight.detach().clone()
    run_epoch(model, tiny_loader(), nn.CrossEntropyLoss(), torch.device("cpu"), optimizer=optimizer)
    assert not torch.equal(before, model[1].weight.detach())


def test_empty_loader_is_rejected():
    loader = DataLoader(TensorDataset(torch.empty(0, 3, 8, 8), torch.empty(0, dtype=torch.long)))
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3))
    with pytest.raises(ValueError, match="no batches"):
        run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))


def test_seed_everything_is_reproducible():
    seed_everything(123)
    first = torch.rand(4)
    seed_everything(123)
    second = torch.rand(4)
    assert torch.equal(first, second)
