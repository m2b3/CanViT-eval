"""Core CanViT evaluation loop: run an episode (T timesteps) and collect outputs.

This is THE shared code path for all CanViT evaluations. Every task (segmentation,
classification, reconstruction) calls run_episode() then processes the outputs
differently. No other module should implement the model forward loop.

An episode = a sequence of T glimpses taken from a single scene (image batch),
processed recurrently through a CanViT model with a given viewing policy.
"""

from dataclasses import dataclass
from typing import Protocol

import torch
from canvit_pytorch import CanViTOutput, RecurrentState, Viewpoint, sample_at_viewpoint
from torch import Tensor


class Policy(Protocol):
    """Viewing policy: given timestep and current state, returns next viewpoint."""

    def step(self, t: int, state: RecurrentState) -> Viewpoint: ...


@dataclass(frozen=True)
class EpisodeStep:
    """Output from one timestep of a CanViT episode."""

    t: int
    state: RecurrentState
    output: CanViTOutput
    viewpoint: Viewpoint


def run_episode(
    *,
    model: torch.nn.Module,
    images: Tensor,
    policy: Policy,
    n_timesteps: int,
    canvas_grid: int,
    glimpse_px: int,
    state: RecurrentState | None = None,
) -> list[EpisodeStep]:
    """Run a CanViT episode: T timesteps with one policy, returning all outputs.

    This is the ONLY place where the recurrent model forward loop happens
    during evaluation. All tasks call this, then process the returned steps.

    Args:
        model: CanViT model (CanViTForPretrainingHFHub or similar).
        images: [B, C, H, W] scene images.
        policy: Viewing policy (determines viewpoints at each timestep).
        n_timesteps: Number of timesteps (glimpses) per episode.
        canvas_grid: Canvas spatial resolution (e.g. 32 for 32×32).
        glimpse_px: Glimpse pixel size (e.g. 128).
        state: Optional initial state (default: model.init_state()).

    Returns:
        List of T EpisodeStep, one per timestep.
    """
    B = images.shape[0]
    if state is None:
        state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)

    steps: list[EpisodeStep] = []
    for t in range(n_timesteps):
        vp = policy.step(t, state)
        glimpse = sample_at_viewpoint(spatial=images, viewpoint=vp, glimpse_size_px=glimpse_px)
        out = model(glimpse=glimpse, state=state, viewpoint=vp)
        state = out.state
        steps.append(EpisodeStep(t=t, state=state, output=out, viewpoint=vp))

    return steps
