"""Run a CanViT episode: T glimpses sampled by a policy, recurrent state updated each step."""

from dataclasses import dataclass
from typing import Protocol

from canvit_pytorch import CanViTOutput, RecurrentState, Viewpoint, sample_at_viewpoint
from torch import Tensor

from canvit_eval.xla import sample_at_viewpoint_xla, sync_if_xla


class CanViTModel(Protocol):
    def init_state(self, *, batch_size: int, canvas_grid_size: int) -> RecurrentState: ...
    def __call__(self, *, glimpse: Tensor, state: RecurrentState, viewpoint: Viewpoint) -> CanViTOutput: ...


class Policy(Protocol):
    def step(self, t: int, state: RecurrentState) -> Viewpoint: ...


@dataclass(frozen=True)
class EpisodeStep:
    t: int
    state: RecurrentState
    output: CanViTOutput
    viewpoint: Viewpoint


def run_episode(
    *,
    model: CanViTModel,
    images: Tensor,
    policy: Policy,
    n_timesteps: int,
    canvas_grid: int,
    glimpse_px: int,
    state: RecurrentState | None = None,
) -> list[EpisodeStep]:
    B = images.shape[0]
    if state is None:
        state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)

    # F.grid_sample has no XLA lowering; the gather-based equivalent avoids a
    # ~100 MB/step device->host fallback transfer (see xla.py).
    sample = sample_at_viewpoint_xla if images.device.type == "xla" else sample_at_viewpoint

    steps: list[EpisodeStep] = []
    for t in range(n_timesteps):
        vp = policy.step(t, state)
        glimpse = sample(spatial=images, viewpoint=vp, glimpse_size_px=glimpse_px)
        out = model(glimpse=glimpse, state=state, viewpoint=vp)
        state = out.state
        steps.append(EpisodeStep(t=t, state=state, output=out, viewpoint=vp))
        # On XLA, cut the lazy graph per timestep: shapes are static across t,
        # so one compiled step-graph is reused instead of one 21-step megagraph.
        sync_if_xla(images.device)

    return steps
