"""StaticShapeMIoUAccumulator must be numerically identical to mIoUAccumulator.

The XLA variant replaces boolean-mask filtering + bincount with a fixed-shape
scatter_add; any divergence here would silently corrupt TPU mIoU numbers.
Runs on CPU — the accumulator code is pure torch.
"""

import torch
from canvit_specialize.metrics import mIoUAccumulator

from canvit_eval.xla import StaticShapeMIoUAccumulator

NUM_CLASSES = 150
IGNORE_LABEL = 255


def _random_batch(gen: torch.Generator, *, with_ignore: bool) -> tuple[torch.Tensor, torch.Tensor]:
    preds = torch.randint(0, NUM_CLASSES, (4, 64, 64), generator=gen)
    targets = torch.randint(0, NUM_CLASSES, (4, 64, 64), generator=gen)
    if with_ignore:
        ignore_mask = torch.rand((4, 64, 64), generator=gen) < 0.3
        targets = targets.masked_fill(ignore_mask, IGNORE_LABEL)
    return preds, targets


def test_static_shape_miou_matches_reference() -> None:
    gen = torch.Generator().manual_seed(0)
    device = torch.device("cpu")
    ref = mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)
    xla = StaticShapeMIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)

    for with_ignore in (False, True, True, True):
        preds, targets = _random_batch(gen, with_ignore=with_ignore)
        ref.update(preds, targets)
        xla.update(preds, targets)

    torch.testing.assert_close(ref.intersection, xla.intersection)
    torch.testing.assert_close(ref.union, xla.union)
    assert ref.compute() == xla.compute()


def test_static_shape_miou_all_ignored() -> None:
    device = torch.device("cpu")
    ref = mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)
    xla = StaticShapeMIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)
    preds = torch.randint(0, NUM_CLASSES, (2, 8, 8))
    targets = torch.full((2, 8, 8), IGNORE_LABEL)
    ref.update(preds, targets)
    xla.update(preds, targets)
    torch.testing.assert_close(ref.intersection, xla.intersection)
    torch.testing.assert_close(ref.union, xla.union)
    assert ref.compute() == xla.compute()
