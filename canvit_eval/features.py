"""Feature extractors: produce spatial features from images.

Each returns a FeatureExtractor: Tensor[B,C,H,W] → list[Tensor[B,G,G,D]].
The list has T elements (one per timestep for CanViT, one for passive models).
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from canvit_pytorch.model.pretraining.hub import CanViTForPretrainingHFHub
from canvit_utils.teacher import DINOv3Teacher
from torch import Tensor

from canvit_eval.episode import run_episode
from canvit_eval.policies import PolicyName, make_policy

# Callable[[Tensor], list[Tensor]] — images in, per-timestep features out.
# Not a type alias — just use the Callable annotation directly in signatures.


def canvit_extractor(
    model: CanViTForPretrainingHFHub,
    *,
    policy_name: PolicyName,
    n_timesteps: int,
    canvas_grid: int,
    glimpse_px: int,
    min_scale: float = 0.05,
    max_scale: float = 1.0,
    probe: torch.nn.Module | None = None,
    viewpoint_log: list[dict] | None = None,
) -> "Callable[[Tensor], list[Tensor]]":
    """Create a CanViT feature extractor (multi-timestep episode).

    If viewpoint_log is provided, appends per-batch viewpoint metadata
    (first item in batch) for reproducibility.
    """
    device = next(model.parameters()).device

    def extract(images: Tensor) -> list[Tensor]:
        B = images.shape[0]
        policy = make_policy(
            policy_name, B, device, n_timesteps, canvas_grid=canvas_grid,
            min_scale=min_scale, max_scale=max_scale,
            probe=probe, get_spatial_fn=model.get_spatial,
        )
        steps = run_episode(
            model=model, images=images, policy=policy,
            n_timesteps=n_timesteps, canvas_grid=canvas_grid, glimpse_px=glimpse_px,
        )
        if viewpoint_log is not None and len(viewpoint_log) == 0:
            # Log viewpoints from first batch only (for metadata)
            for step in steps:
                vp = step.viewpoint
                viewpoint_log.append({
                    "t": step.t,
                    "center_y": vp.centers[0, 0].item(),
                    "center_x": vp.centers[0, 1].item(),
                    "scale": vp.scales[0].item(),
                })
        return [
            model.get_spatial(step.state.canvas).view(B, canvas_grid, canvas_grid, -1)
            for step in steps
        ]

    return extract


def dinov3_extractor(
    teacher: DINOv3Teacher,
    *,
    eval_resolution: int,
    patch_size: int = 16,
) -> "Callable[[Tensor], list[Tensor]]":
    """Create a DINOv3 feature extractor (single passive forward pass).

    eval_resolution: the resolution the probe was trained at (e.g. 128px, 512px).
    The teacher is run at this resolution, NOT at the dataset scene_size.
    """
    grid = eval_resolution // patch_size

    def extract(images: Tensor) -> list[Tensor]:
        B = images.shape[0]
        resized = F.interpolate(images, size=(eval_resolution, eval_resolution),
                                mode="bilinear", align_corners=False)
        feats = teacher.forward_norm_features(resized).patches  # [B, N, D] (flat)
        feats = feats.view(B, grid, grid, -1)  # [B, G, G, D]
        return [feats]

    return extract
