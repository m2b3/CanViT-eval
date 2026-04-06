"""CanViT evaluation CLI.

    uv run python -m canvit_eval ade20k-seg --probe-repo canvit/probe-... [--model dinov3 --eval-resolution 128]
    uv run python -m canvit_eval in1k-clf
    uv run python -m canvit_eval reconstruction --model-repo canvit/canvitb16-abl-...
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import torch
import tyro

from canvit_eval.config import TEACHER_REPO, EpisodeConfig

log = logging.getLogger(__name__)


@dataclass
class ADE20kSegCmd:
    """ADE20K segmentation: CanViT (multi-timestep) or DINOv3 (passive baseline)."""

    probe_repo: str
    model: Literal["canvit", "dinov3"] = "canvit"
    output: Path = Path("results/ade20k_seg.pt")
    scene_size: int = 512
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 8
    # CanViT-specific (ignored for DINOv3):
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    # DINOv3-specific (ignored for CanViT):
    eval_resolution: int = 512
    teacher_repo: str = TEACHER_REPO

    def run(self) -> None:
        from canvit_eval.features import canvit_extractor, dinov3_extractor
        from canvit_eval.tasks.ade20k_seg import Config, run

        device = torch.device(self.device)

        if self.model == "canvit":
            from canvit_pytorch.model.pretraining.hub import CanViTForPretrainingHFHub
            m = CanViTForPretrainingHFHub.from_pretrained(self.episode.model_repo).to(device).eval()
            cg = self.episode.canvas_grid or self.scene_size // m.backbone.patch_size_px
            # Load probe for entropy policy
            probe_for_entropy = None
            if self.episode.policy == "entropy_coarse_to_fine":
                from canvit_probes import SegmentationProbe
                probe_for_entropy = SegmentationProbe.from_pretrained(self.probe_repo).to(device).eval()
            extract = canvit_extractor(
                m, policy_name=self.episode.policy, n_timesteps=self.episode.n_timesteps,
                canvas_grid=cg, glimpse_px=self.episode.glimpse_px,
                min_scale=self.episode.min_scale, max_scale=self.episode.max_scale,
                probe=probe_for_entropy,
            )
            meta = {"model_repo": self.episode.model_repo, "canvas_grid": cg,
                    "policy": self.episode.policy, "n_timesteps": self.episode.n_timesteps}
        else:
            from canvit_utils.teacher import load_teacher
            teacher = load_teacher(self.teacher_repo, device)
            extract = dinov3_extractor(teacher, eval_resolution=self.eval_resolution)
            meta = {"model": self.teacher_repo, "eval_resolution": self.eval_resolution}

        cfg = Config(
            probe_repo=self.probe_repo, output=self.output, scene_size=self.scene_size,
            device=self.device, batch_size=self.batch_size, num_workers=self.num_workers,
        )
        run(cfg, extract, metadata=meta)


@dataclass
class IN1KClfCmd:
    """ImageNet-1K classification."""

    episode: EpisodeConfig = field(default_factory=lambda: EpisodeConfig(canvas_grid=32))
    probe_repo: str = "yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe"
    val_dir: Path | None = None
    output: Path = Path("results/in1k_clf.pt")
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 8

    def run(self) -> None:
        from canvit_eval.config import imagenet_val_dir
        from canvit_eval.tasks.in1k_clf import Config, evaluate
        val = self.val_dir or imagenet_val_dir()
        cfg = Config(
            episode=self.episode, probe_repo=self.probe_repo,
            val_dir=val, output=self.output,
            device=self.device, batch_size=self.batch_size, num_workers=self.num_workers,
        )
        evaluate(cfg)


@dataclass
class ReconCmd:
    """Reconstruction quality (cosine sim to DINOv3 teacher)."""

    model_repo: str
    episode: EpisodeConfig = field(default_factory=lambda: EpisodeConfig(policy="random", n_timesteps=10))
    output: Path = Path("results/reconstruction.pt")
    scene_size: int = 512
    device: str = "cuda"
    batch_size: int = 16
    teacher_cache: Path | None = None

    def run(self) -> None:
        from canvit_eval.tasks.reconstruction import Config, evaluate
        cfg = Config(
            model_repo=self.model_repo, episode=self.episode,
            output=self.output, scene_size=self.scene_size,
            device=self.device, batch_size=self.batch_size,
            teacher_cache=self.teacher_cache,
        )
        evaluate(cfg)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cmd = tyro.cli(
        Annotated[ADE20kSegCmd, tyro.conf.subcommand("ade20k-seg")]
        | Annotated[IN1KClfCmd, tyro.conf.subcommand("in1k-clf")]
        | Annotated[ReconCmd, tyro.conf.subcommand("reconstruction")]
    )
    cmd.run()


if __name__ == "__main__":
    main()
