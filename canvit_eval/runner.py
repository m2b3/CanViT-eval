"""Shared evaluation runner — the common loop across all CanViT tasks.

All tasks share: load model → iterate batches → make policy → run episode.
Only the per-step processing differs. This module provides the common loop
as a generator, so tasks just iterate and process.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from canvit_pytorch.model.pretraining.hub import CanViTForPretrainingHFHub
from torch.utils.data import DataLoader
from tqdm import tqdm

from canvit_eval.config import EpisodeConfig
from canvit_eval.episode import EpisodeStep, run_episode
from canvit_eval.policies import make_policy

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchResult:
    """One batch's episode results + the raw batch data."""

    steps: list[EpisodeStep]
    batch: tuple  # whatever the DataLoader yields (images, masks) or (images, labels) or (images,)


def load_model(model_repo: str, device: torch.device) -> CanViTForPretrainingHFHub:
    """Load CanViT from HuggingFace Hub. Single call site for model loading."""
    log.info("Loading model: %s", model_repo)
    model = CanViTForPretrainingHFHub.from_pretrained(model_repo).to(device).eval()
    log.info("  canvas_dim=%d, local_dim=%d", model.canvas_dim, model.local_dim)
    return model


def eval_batches(
    *,
    model: CanViTForPretrainingHFHub,
    loader: DataLoader,
    episode_cfg: EpisodeConfig,
    canvas_grid: int,
    device: torch.device,
    amp: bool = True,
    policy_kwargs: dict | None = None,
) -> Iterator[BatchResult]:
    """Iterate over dataset, running episodes. Yields (steps, batch) per batch.

    This is the ONE evaluation loop. All tasks consume this generator.

    Args:
        model: Loaded CanViT model.
        loader: DataLoader yielding tuples (first element = images).
        episode_cfg: Episode parameters (policy, n_timesteps, etc.).
        canvas_grid: Resolved canvas grid size.
        device: Compute device.
        amp: Use automatic mixed precision.
        policy_kwargs: Extra kwargs for make_policy (e.g. probe= for entropy C2F).

    Yields:
        BatchResult with episode steps and the raw batch data.
    """
    T = episode_cfg.n_timesteps
    amp_dtype = torch.bfloat16 if amp else torch.float32
    kw = policy_kwargs or {}

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch[0].to(device, non_blocking=True)
            B = images.shape[0]

            policy = make_policy(
                episode_cfg.policy, B, device, T, canvas_grid=canvas_grid,
                min_scale=episode_cfg.min_scale, max_scale=episode_cfg.max_scale,
                **kw,
            )
            steps = run_episode(
                model=model, images=images, policy=policy,
                n_timesteps=T, canvas_grid=canvas_grid, glimpse_px=episode_cfg.glimpse_px,
            )
            yield BatchResult(steps=steps, batch=batch)
