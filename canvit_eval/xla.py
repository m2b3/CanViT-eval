"""TPU (torch_xla) support for the eval pipeline.

torch_xla is imported lazily and only when an "xla" device is requested, so
this module is importable on CUDA/CPU machines without torch_xla installed.

Lazy-tensor constraints this module exists for:
- Graphs must be cut explicitly (`torch_xla.sync()`); without barriers the
  pending IR grows across the whole dataset.
- Data-dependent shapes (boolean-mask indexing, `bincount`) trigger
  per-batch recompilation; `StaticShapeMIoUAccumulator` is a fixed-shape
  equivalent of `mIoUAccumulator.update`.
"""

from pathlib import Path

import torch
from canvit_specialize.metrics import mIoUAccumulator


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("xla"):
        import torch_xla

        return torch_xla.device()
    return torch.device(device_str)


def sync_if_xla(device: torch.device) -> None:
    """Cut the lazy-tensor graph on XLA; no-op elsewhere. Non-blocking."""
    if device.type == "xla":
        import torch_xla

        torch_xla.sync()


class StaticShapeMIoUAccumulator(mIoUAccumulator):
    """mIoUAccumulator with an XLA-friendly `update`.

    The parent filters ignored pixels via boolean-mask indexing, whose output
    size is data-dependent — on XLA that recompiles every batch. Here invalid
    (target == ignore_index) pairs are routed to a spare histogram bin that is
    dropped before the confusion matrix is built, so every op has a static
    shape. Numerically identical to the parent (tests/test_xla_miou.py).
    """

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        assert preds.ndim == 3, f"Expected [B, H, W], got shape {preds.shape}"
        assert preds.shape == targets.shape, f"Shape mismatch: {preds.shape} vs {targets.shape}"
        n = self.num_classes
        p = preds.flatten().long()
        t = targets.flatten().long()
        valid = t != self.ignore_index
        # (t, p) -> flat bin; invalid pairs -> spare bin n*n, dropped below.
        idx = torch.where(valid, t * n + p, torch.full_like(t, n * n))
        counts = torch.zeros(n * n + 1, device=p.device, dtype=self.intersection.dtype)
        counts.scatter_add_(0, idx, torch.ones_like(idx, dtype=counts.dtype))
        cm = counts[: n * n].view(n, n)
        diag = cm.diag()
        self.intersection += diag
        self.union += cm.sum(dim=1) + cm.sum(dim=0) - diag


def make_miou_accumulator(
    num_classes: int, ignore_index: int, device: torch.device
) -> mIoUAccumulator:
    if device.type == "xla":
        return StaticShapeMIoUAccumulator(num_classes, ignore_index, device)
    return mIoUAccumulator(num_classes, ignore_index, device)


def dump_metrics_report(device: torch.device, output: Path) -> None:
    """Write torch_xla's counter/timing report next to the eval output.

    Compile counts, transfer counts, and aten:: fallback counters are the
    recompile/sync diagnostics for TPU runs. No-op off XLA.
    """
    if device.type != "xla":
        return
    import torch_xla.debug.metrics as met

    report_path = output.with_suffix(output.suffix + ".xla_metrics.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(met.metrics_report())
