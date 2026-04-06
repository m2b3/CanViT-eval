"""Integration tests for the core episode runner.

Uses the real CanViT model from HuggingFace — tiny forward passes on CPU.
"""

import pytest
import torch
from canvit_pytorch import Viewpoint
from canvit_pytorch.model.pretraining.hub import CanViTForPretrainingHFHub
from canvit_probes import SegmentationProbe

from canvit_eval.episode import EpisodeStep, run_episode
from canvit_eval.policies import make_policy

MODEL_REPO = "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"
PROBE_REPO = "canvit/probe-ade20k-40k-s512-c32-in21k"
DEVICE = torch.device("cpu")
CANVAS_GRID = 8  # small for fast tests


@pytest.fixture(scope="module")
def model() -> CanViTForPretrainingHFHub:
    return CanViTForPretrainingHFHub.from_pretrained(MODEL_REPO).eval()


@pytest.fixture(scope="module")
def probe() -> SegmentationProbe:
    return SegmentationProbe.from_pretrained(PROBE_REPO).eval()


class _FullScenePolicy:
    def step(self, t: int, state: object) -> Viewpoint:
        return Viewpoint.full_scene(batch_size=1, device=DEVICE)


def test_run_episode_shapes(model: CanViTForPretrainingHFHub) -> None:
    steps = run_episode(
        model=model, images=torch.randn(1, 3, 512, 512), policy=_FullScenePolicy(),
        n_timesteps=2, canvas_grid=CANVAS_GRID, glimpse_px=128,
    )
    assert len(steps) == 2
    assert all(isinstance(s, EpisodeStep) for s in steps)
    canvas = steps[-1].state.canvas
    assert canvas.shape == (1, 16 + CANVAS_GRID**2, 1024)  # [B, regs+spatial, canvas_dim]


def test_episode_canvas_evolves(model: CanViTForPretrainingHFHub) -> None:
    steps = run_episode(
        model=model, images=torch.randn(1, 3, 256, 256), policy=_FullScenePolicy(),
        n_timesteps=2, canvas_grid=CANVAS_GRID, glimpse_px=128,
    )
    assert not torch.equal(steps[0].state.canvas, steps[1].state.canvas)


def test_all_static_policies(model: CanViTForPretrainingHFHub) -> None:
    """Every static policy produces valid episodes."""
    images = torch.randn(1, 3, 256, 256)
    for name in ["coarse_to_fine", "fine_to_coarse", "random", "full_then_random", "repeated_full_scene"]:
        policy = make_policy(name, batch_size=1, device=DEVICE, n_viewpoints=3, canvas_grid=CANVAS_GRID)
        steps = run_episode(
            model=model, images=images, policy=policy,
            n_timesteps=3, canvas_grid=CANVAS_GRID, glimpse_px=128,
        )
        assert len(steps) == 3, f"{name}: expected 3 steps"
        assert all(s.state.canvas.shape[2] == 1024 for s in steps), f"{name}: wrong canvas_dim"


def test_entropy_guided_c2f(model: CanViTForPretrainingHFHub, probe: SegmentationProbe) -> None:
    """Entropy-guided C2F works end-to-end with real model + probe."""
    images = torch.randn(1, 3, 512, 512)
    T = 5  # level 0 (1) + level 1 (4)
    policy = make_policy(
        "entropy_coarse_to_fine", batch_size=1, device=DEVICE, n_viewpoints=21,
        canvas_grid=CANVAS_GRID, probe=probe, get_spatial_fn=model.get_spatial,
    )
    steps = run_episode(
        model=model, images=images, policy=policy,
        n_timesteps=T, canvas_grid=CANVAS_GRID, glimpse_px=128,
    )
    assert len(steps) == T
    # First step should be full scene (level 0)
    assert steps[0].viewpoint.scales[0].item() == 1.0
