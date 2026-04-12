"""Tests for evaluation metrics."""

import torch
from canvit_probes.metrics import mIoUAccumulator


def test_perfect_predictions() -> None:
    acc = mIoUAccumulator(num_classes=3, ignore_index=255, device=torch.device("cpu"))
    preds = torch.tensor([[0, 1, 2, 0]]).reshape(1, 2, 2)
    acc.update(preds, preds)
    assert acc.compute() == 1.0


def test_zero_overlap() -> None:
    acc = mIoUAccumulator(num_classes=2, ignore_index=255, device=torch.device("cpu"))
    preds = torch.zeros(1, 2, 2, dtype=torch.long)
    targets = torch.ones(1, 2, 2, dtype=torch.long)
    acc.update(preds, targets)
    assert acc.compute() == 0.0


def test_ignore_index_excluded() -> None:
    acc = mIoUAccumulator(num_classes=2, ignore_index=255, device=torch.device("cpu"))
    preds = torch.tensor([[0, 0], [255, 1]]).unsqueeze(0)
    targets = torch.tensor([[0, 0], [255, 1]]).unsqueeze(0)
    acc.update(preds, targets)
    assert acc.compute() == 1.0


def test_accumulation_across_batches() -> None:
    acc = mIoUAccumulator(num_classes=2, ignore_index=255, device=torch.device("cpu"))
    # Batch 1: perfect class 0
    acc.update(torch.zeros(1, 2, 2, dtype=torch.long), torch.zeros(1, 2, 2, dtype=torch.long))
    # Batch 2: perfect class 1
    acc.update(torch.ones(1, 2, 2, dtype=torch.long), torch.ones(1, 2, 2, dtype=torch.long))
    assert acc.compute() == 1.0
