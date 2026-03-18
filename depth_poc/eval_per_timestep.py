"""Per-timestep depth evaluation with C2F viewpoints.

Loads a trained DepthProbe checkpoint and evaluates at each timestep
using coarse-to-fine viewing policy (deterministic). Uses Eigen crop
for metrics (standard NYU eval protocol).

Usage:
    uv run python depth_poc/eval_per_timestep.py \
        --ckpt checkpoints/depth-poc/best_rmse0.4987_step8500.pt \
        --nyu-root /datasets/NYU/nyu \
        --n-timesteps 21
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from dinov3.eval.depth.datasets.datasets_utils import _EvalCropType, make_valid_mask
from dinov3.eval.depth.metrics import calculate_depth_metrics
from dinov3.eval.depth.transforms import make_depth_eval_transforms
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from depth_poc.dataset import NYUDepthV2
from depth_poc.train import MIN_DEPTH, MAX_DEPTH, NYU_DEPTH_NORMALIZATION, DepthProbe

log = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    ckpt: Path = Path("checkpoints/depth-poc/best_rmse0.4987_step8500.pt")
    nyu_root: Path = Path("/datasets/NYU/nyu")
    scene_size: int = 512
    model_repo: str = "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"
    n_timesteps: int = 21
    canvas_grid: int = 32
    glimpse_px: int = 128
    batch_size: int = 4
    num_workers: int = 4
    device: str = "cuda"


@torch.no_grad()
def main(cfg: EvalConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(cfg.device)

    # Load checkpoint.
    ckpt = torch.load(cfg.ckpt, map_location=device, weights_only=False)
    ckpt_cfg = ckpt["config"]
    log.info(f"Loaded checkpoint: {cfg.ckpt}")
    log.info(f"  Original mode: {ckpt_cfg['mode']}, scene: {ckpt_cfg['scene_size']}")
    log.info(f"  Train metrics: {ckpt.get('metrics', 'N/A')}")

    # Load CanViT.
    from canvit import CanViTForPretrainingHFHub, sample_at_viewpoint
    from canvit.policies import coarse_to_fine_viewpoints

    model = CanViTForPretrainingHFHub.from_pretrained(cfg.model_repo).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    embed_dim = model.canvas_dim

    # Load probe.
    probe = DepthProbe(embed_dim, n_bins=ckpt_cfg["n_bins"], dropout=0.0).to(device).eval()
    probe.load_state_dict(ckpt["probe"])
    log.info(f"Probe loaded: {sum(p.numel() for p in probe.parameters()):,} params")

    # Data — eval transforms from dinov3 (no crop, resize to square, normalize).
    eval_tf = make_depth_eval_transforms(
        normalization_constant=NYU_DEPTH_NORMALIZATION,
        img_size=(cfg.scene_size, cfg.scene_size),
        fixed_crop="FULL",
        tta=False,
    )

    def eval_transform(img, depth):
        images, depths = eval_tf(img, depth)
        return images[0], depths[0]

    test_ds = NYUDepthV2(cfg.nyu_root, "test", transform=eval_transform)
    test_loader = DataLoader(test_ds, cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    log.info(f"Test set: {len(test_ds)} samples")

    # Per-timestep evaluation with C2F.
    log.info(f"Evaluating {cfg.n_timesteps} timesteps with C2F policy (Eigen crop)...")
    per_t_metrics: list[dict[str, float]] = [{} for _ in range(cfg.n_timesteps)]
    per_t_counts: list[int] = [0] * cfg.n_timesteps

    for images, depths in tqdm(test_loader, desc="Eval"):
        images, depths = images.to(device), depths.to(device)
        B = images.shape[0]

        # Eigen crop mask for metrics.
        eigen_mask = make_valid_mask(depths.unsqueeze(1), eval_crop=_EvalCropType.NYU_EIGEN).squeeze(1)

        vps = coarse_to_fine_viewpoints(B, device, cfg.n_timesteps)
        state = model.init_state(batch_size=B, canvas_grid_size=cfg.canvas_grid)

        for t, vp in enumerate(vps):
            glimpse = sample_at_viewpoint(spatial=images, viewpoint=vp, glimpse_size_px=cfg.glimpse_px)
            out = model(glimpse=glimpse, state=state, viewpoint=vp)
            state = out.state

            feats = model.get_spatial(state.canvas).view(B, cfg.canvas_grid, cfg.canvas_grid, -1)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred = probe(feats.float())
            pred = F.interpolate(pred, size=depths.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            pred = pred.clamp(MIN_DEPTH, MAX_DEPTH)

            m = calculate_depth_metrics(depths, pred, eigen_mask)
            for field in ("rmse", "abs_rel", "a1"):
                v = getattr(m, field)
                val = v.item() if isinstance(v, Tensor) else float(v)
                per_t_metrics[t][field] = per_t_metrics[t].get(field, 0.0) + val
            per_t_counts[t] += 1

    # Print results.
    log.info("=" * 60)
    log.info("Per-timestep results (C2F policy, Eigen crop):")
    log.info(f"{'t':>3} {'RMSE':>8} {'abs_rel':>8} {'a1':>8}")
    for t in range(cfg.n_timesteps):
        n = per_t_counts[t]
        r = per_t_metrics[t]["rmse"] / n
        a = per_t_metrics[t]["abs_rel"] / n
        d = per_t_metrics[t]["a1"] / n
        log.info(f"{t:3d} {r:8.4f} {a:8.4f} {d:8.4f}")
    log.info("=" * 60)


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
