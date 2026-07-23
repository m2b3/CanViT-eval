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
from canvit_pytorch import Viewpoint
from canvit_specialize.metrics import mIoUAccumulator
from torch import Tensor


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


def _lerp_gather(x: Tensor, coords: Tensor, dim: int) -> Tensor:
    """Bilinear interpolation along `dim` of x at fractional pixel `coords`.

    x: [B, C, H, W]; coords: [B, S] pixel coordinates along dim. Out-of-range
    contributions get zero weight (grid_sample zeros-padding semantics).
    Gather-based: static shapes, native XLA lowering.
    """
    size = x.shape[dim]
    i0 = coords.floor()
    w1 = (coords - i0).to(x.dtype)
    w0 = 1.0 - w1
    i0l = i0.long()
    i1l = i0l + 1
    ok0 = ((i0l >= 0) & (i0l < size)).to(x.dtype)
    ok1 = ((i1l >= 0) & (i1l < size)).to(x.dtype)
    idx0 = i0l.clamp(0, size - 1)
    idx1 = i1l.clamp(0, size - 1)

    B, S = coords.shape
    shape = [1] * x.ndim
    shape[0] = B
    shape[dim] = S
    expand = list(x.shape)
    expand[dim] = S

    def take(idx: Tensor) -> Tensor:
        return torch.gather(x, dim, idx.view(shape).expand(expand))

    return take(idx0) * (w0 * ok0).view(shape) + take(idx1) * (w1 * ok1).view(shape)


def sample_at_viewpoint_xla(*, spatial: Tensor, viewpoint: Viewpoint, glimpse_size_px: int) -> Tensor:
    """XLA-native equivalent of canvit_pytorch.sample_at_viewpoint.

    F.grid_sample has no XLA lowering — the aten::grid_sampler_2d CPU
    fallback pulls the full image batch device->host every timestep (~100 MB
    at bs=32 s512; measured 15.8 s of transfer over 2 batches). The sampling
    grid is axis-aligned (centers + scales * cell-center offsets), so the
    bilinear sample separates into two 1-D lerp-gathers. Matches
    sample_at_viewpoint within fp32 rounding (max abs diff 7.2e-7 over
    random viewpoints at 128 px and 32 px; test_xla_sampler.py).
    """
    B, _, H, W = spatial.shape
    S = glimpse_size_px
    offs = ((torch.arange(S, device=spatial.device, dtype=torch.float32) + 0.5) / S) * 2 - 1
    ny = viewpoint.centers[:, 0:1] + viewpoint.scales[:, None] * offs[None, :]
    nx = viewpoint.centers[:, 1:2] + viewpoint.scales[:, None] * offs[None, :]
    # align_corners=False: pixel = ((n + 1) * size - 1) / 2
    py = ((ny + 1.0) * H - 1.0) / 2.0
    px = ((nx + 1.0) * W - 1.0) / 2.0
    x = spatial.float()
    x = _lerp_gather(x, py, dim=2)
    x = _lerp_gather(x, px, dim=3)
    return x.to(spatial.dtype)


def host_pin_standardizer_flags(model: torch.nn.Module) -> None:
    """Move PositionAwareStandardizer._initialized buffers to CPU.

    The `initialized` property does `.item()` inside every forward — on XLA
    that is one blocking device->host transfer per glimpse-step (measured
    aten::_local_scalar_dense == n_timesteps per batch). The flag is a
    load-time constant; on CPU the `.item()` is free. No-op elsewhere."""
    from canvit_pytorch.standardizers import PositionAwareStandardizer

    for module in model.modules():
        if isinstance(module, PositionAwareStandardizer):
            module._initialized = module._initialized.cpu()


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
