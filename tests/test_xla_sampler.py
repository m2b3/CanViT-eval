"""sample_at_viewpoint_xla must match canvit_pytorch.sample_at_viewpoint.

The XLA path swaps F.grid_sample (no XLA lowering) for a gather-based
separable bilinear sampler; any divergence would silently corrupt TPU
glimpses. Runs on CPU — the sampler is pure torch.
"""

import torch
from canvit_pytorch import Viewpoint, sample_at_viewpoint

from canvit_eval.xla import sample_at_viewpoint_xla


def test_sampler_matches_grid_sample() -> None:
    torch.manual_seed(0)
    for _ in range(10):
        spatial = torch.randn(8, 3, 512, 512)
        centers = (torch.rand(8, 2) * 2 - 1) * 0.9  # includes partially out-of-bounds views
        scales = torch.rand(8) * 0.95 + 0.05
        vp = Viewpoint(centers=centers, scales=scales)
        for size_px in (128, 32):
            ref = sample_at_viewpoint(spatial=spatial, viewpoint=vp, glimpse_size_px=size_px)
            got = sample_at_viewpoint_xla(spatial=spatial, viewpoint=vp, glimpse_size_px=size_px)
            torch.testing.assert_close(ref, got, atol=1e-5, rtol=1e-5)


def test_sampler_full_scene_identity_convention() -> None:
    """Full-scene viewpoint (center 0, scale 1) at output size == input size
    must reproduce the input exactly (cell centers land on cell centers)."""
    torch.manual_seed(1)
    spatial = torch.randn(2, 3, 64, 64)
    vp = Viewpoint(centers=torch.zeros(2, 2), scales=torch.ones(2))
    got = sample_at_viewpoint_xla(spatial=spatial, viewpoint=vp, glimpse_size_px=64)
    torch.testing.assert_close(got, spatial, atol=1e-6, rtol=0)
