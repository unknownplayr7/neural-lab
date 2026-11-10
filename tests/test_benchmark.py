"""Tests for the CPU/MPS inference benchmark and result exports."""

import csv
import io
import json

import pytest
import torch
from torch import nn

from neural_lab.benchmark import (
    benchmark_to_csv,
    benchmark_to_json,
    run_inference_benchmark,
)


class TinyClassifier(nn.Module):
    architecture_name = "TinyClassifier"

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(3, 10)

    def forward(self, images):
        return self.output(self.pool(images).flatten(1))


def test_cpu_benchmark_records_methodology_and_timing():
    model = TinyClassifier()
    original_device = next(model.parameters()).device
    result = run_inference_benchmark(
        model,
        device="cpu",
        dataset_name="cifar10",
        batch_size=4,
        warmup_batches=1,
        iterations=2,
    )
    assert result["device"] == "cpu"
    assert result["architecture"] == "TinyClassifier"
    assert result["images_processed"] == 8
    assert result["elapsed_seconds"] > 0
    assert result["images_per_second"] > 0
    assert result["warmup_batches"] == 1
    assert "synchronizes" in result["methodology"]
    assert next(model.parameters()).device == original_device


def test_benchmark_exports_preserve_interpretation_metadata():
    result = run_inference_benchmark(
        TinyClassifier(),
        device="cpu",
        dataset_name="cifar100",
        batch_size=2,
        warmup_batches=0,
        iterations=1,
    )
    json_result = json.loads(benchmark_to_json([result]))
    assert json_result[0]["dataset"] == "cifar100"
    assert "methodology" in json_result[0]
    csv_text = benchmark_to_csv([result])
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert len(rows) == 1
    assert rows[0]["device"] == "cpu"
    assert rows[0]["pytorch_version"] == str(torch.__version__)
    assert rows[0]["images_processed"] == "2"


def test_empty_csv_export_is_empty():
    assert benchmark_to_csv([]) == ""
    assert json.loads(benchmark_to_json([])) == []


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"device": "cuda"}, "cpu or mps"),
        ({"dataset_name": "unknown"}, "dataset_name"),
        ({"batch_size": 0}, "batch_size"),
        ({"warmup_batches": -1}, "warmup_batches"),
        ({"iterations": 0}, "iterations"),
        ({"iterations": True}, "iterations"),
    ],
)
def test_benchmark_rejects_invalid_parameters(kwargs, message):
    arguments = {
        "device": "cpu",
        "dataset_name": "cifar10",
        "batch_size": 2,
        "warmup_batches": 0,
        "iterations": 1,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        run_inference_benchmark(TinyClassifier(), **arguments)
