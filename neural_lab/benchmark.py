"""Repeatable, metadata-rich inference benchmarks for CPU and Apple MPS."""

from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import io
import json
import platform
import time
from typing import Any

import torch
from torch import nn


METHODOLOGY = (
    "Inference-only benchmark using random normalized-shape inputs (NCHW, 3x32x32). "
    "The model is copied and set to eval mode. Warm-up batches are excluded. "
    "The timed phase measures repeated forward passes and synchronizes the selected "
    "accelerator before and after timing. Data loading, preprocessing, training, and "
    "backpropagation are excluded."
)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def run_inference_benchmark(
    model: nn.Module,
    *,
    device: str | torch.device,
    dataset_name: str,
    batch_size: int = 64,
    warmup_batches: int = 3,
    iterations: int = 20,
) -> dict[str, Any]:
    """Measure model-only inference throughput without mutating the active model."""
    target = torch.device(device)
    if target.type not in {"cpu", "mps"}:
        raise ValueError("Benchmark device must be cpu or mps")
    if target.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS is not available on this system")
    if dataset_name not in {"cifar10", "cifar100"}:
        raise ValueError("dataset_name must be cifar10 or cifar100")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 1024:
        raise ValueError("batch_size must be an integer from 1 to 1024")
    if isinstance(warmup_batches, bool) or not isinstance(warmup_batches, int) or not 0 <= warmup_batches <= 100:
        raise ValueError("warmup_batches must be an integer from 0 to 100")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= 1000:
        raise ValueError("iterations must be an integer from 1 to 1000")

    benchmark_model = copy.deepcopy(model).to(target)
    benchmark_model.eval()
    inputs = torch.randn(batch_size, 3, 32, 32, device=target)

    with torch.inference_mode():
        for _ in range(warmup_batches):
            benchmark_model(inputs)
        _synchronize(target)
        started = time.perf_counter()
        for _ in range(iterations):
            benchmark_model(inputs)
        _synchronize(target)
        elapsed = time.perf_counter() - started

    if elapsed <= 0:
        raise RuntimeError("Benchmark timer returned a non-positive duration")
    total_images = batch_size * iterations
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": target.type,
        "device_label": "Apple MPS" if target.type == "mps" else "CPU",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "dataset": dataset_name,
        "architecture": getattr(model, "architecture_name", model.__class__.__name__),
        "batch_size": batch_size,
        "warmup_batches": warmup_batches,
        "iterations": iterations,
        "images_processed": total_images,
        "elapsed_seconds": elapsed,
        "latency_ms_per_batch": elapsed * 1000.0 / iterations,
        "images_per_second": total_images / elapsed,
        "methodology": METHODOLOGY,
    }


def benchmark_to_json(results: list[dict[str, Any]]) -> str:
    """Serialize benchmark results with all interpretation metadata."""
    return json.dumps(results, indent=2, sort_keys=True, allow_nan=False)


def benchmark_to_csv(results: list[dict[str, Any]]) -> str:
    """Export flat benchmark records as CSV."""
    if not results:
        return ""
    fields = list(results[0].keys())
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
    return output.getvalue()
