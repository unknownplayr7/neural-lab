import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from neural_lab.training import evaluate_classifier, run_epoch, seed_everything


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



def test_classifier_evaluation_returns_confusion_matrix_and_class_support():
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3))
    metrics = evaluate_classifier(model, tiny_loader(), torch.device("cpu"), num_classes=3)
    matrix = metrics["confusion_matrix"]
    assert len(matrix) == 3
    assert all(len(row) == 3 for row in matrix)
    assert sum(sum(row) for row in matrix) == 8
    assert sum(row["support"] for row in metrics["per_class"]) == 8
    assert 0 <= metrics["accuracy"] <= 1


def test_classifier_evaluation_rejects_empty_loader():
    loader = DataLoader(TensorDataset(torch.empty(0, 3, 8, 8), torch.empty(0, dtype=torch.long)))
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3))
    with pytest.raises(ValueError, match="no batches"):
        evaluate_classifier(model, loader, torch.device("cpu"), num_classes=3)



def test_classifier_evaluation_collects_only_requested_mistakes():
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3))
    metrics = evaluate_classifier(model, tiny_loader(), torch.device("cpu"), num_classes=3, max_misclassified=2)
    assert len(metrics["misclassified"]) <= 2
    for mistake in metrics["misclassified"]:
        assert mistake["image"].shape == (3, 8, 8)
        assert 0 <= mistake["confidence"] <= 1
