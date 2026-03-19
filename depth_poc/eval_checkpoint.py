"""Evaluate a trained depth probe with deterministic C2F viewpoints.

Loads a checkpoint and evaluates at each timestep (1..n_timesteps)
using C2F policy + Eigen crop. Reports per-timestep RMSE, abs_rel, a1.

    uv run python depth_poc/eval_checkpoint.py --ckpt checkpoints/canvit-c32/final_step151425.pt
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
from depth_poc.train import MIN_DEPTH, MAX_DEPTH, DEPTH_NORM, DepthProbe

log = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    ckpt: Path = Path("checkpoints/canvit-c32/final_step151425.pt")
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

    ckpt = torch.load(cfg.ckpt, map_location=device, weights_only=False)
    ckpt_cfg = ckpt["config"]
    log.info(f"Checkpoint: {cfg.ckpt}")
    log.info(f"  mode={ckpt_cfg['mode']}, scene={ckpt_cfg['scene_size']}, step={ckpt.get('step', '?')}")
    log.info(f"  train metrics: {ckpt.get('metrics', 'N/A')}")

    from canvit import CanViTForPretrainingHFHub, sample_at_viewpoint
    from canvit.policies import coarse_to_fine_viewpoints

    model = CanViTForPretrainingHFHub.from_pretrained(cfg.model_repo).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    probe = DepthProbe(model.canvas_dim, n_bins=ckpt_cfg["n_bins"], dropout=0.0).to(device).eval()
    probe.load_state_dict(ckpt["probe"])
    log.info(f"Probe: {sum(p.numel() for p in probe.parameters()):,} params")

    s = cfg.scene_size
    _eval_tf = make_depth_eval_transforms(normalization_constant=DEPTH_NORM, img_size=(s, s), fixed_crop="FULL", tta=False)
    eval_tf = lambda img, depth: tuple(x[0] for x in _eval_tf(img, depth))
    test_ds = NYUDepthV2(cfg.nyu_root, "test", transform=eval_tf)
    loader = DataLoader(test_ds, cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    log.info(f"Test: {len(test_ds)} samples, {cfg.n_timesteps} timesteps, C2F policy")

    per_t: list[dict[str, float]] = [{} for _ in range(cfg.n_timesteps)]
    counts: list[int] = [0] * cfg.n_timesteps

    for images, depths in tqdm(loader, desc="Eval"):
        images = images.to(device, dtype=torch.float32)
        depths = depths.to(device, dtype=torch.float32)
        if depths.ndim == 4:
            depths = depths.squeeze(1)
        B = images.shape[0]
        eigen_mask = make_valid_mask(depths.unsqueeze(1), eval_crop=_EvalCropType.NYU_EIGEN).squeeze(1)

        vps = coarse_to_fine_viewpoints(B, device, cfg.n_timesteps)
        state = model.init_state(batch_size=B, canvas_grid_size=cfg.canvas_grid)

        for t, vp in enumerate(vps):
            glimpse = sample_at_viewpoint(spatial=images, viewpoint=vp, glimpse_size_px=cfg.glimpse_px)
            result = model(glimpse=glimpse, state=state, viewpoint=vp)
            state = result.state
            feats = model.get_spatial(state.canvas).view(B, cfg.canvas_grid, cfg.canvas_grid, -1)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred = probe(feats.float())
            pred = F.interpolate(pred, size=depths.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            pred = pred.clamp(MIN_DEPTH, MAX_DEPTH)
            m = calculate_depth_metrics(depths, pred, eigen_mask)
            for k in ("rmse", "abs_rel", "a1"):
                v = getattr(m, k)
                per_t[t][k] = per_t[t].get(k, 0.0) + (v.item() if isinstance(v, Tensor) else float(v))
            counts[t] += 1

    log.info("=" * 60)
    log.info("Per-timestep results (C2F, Eigen crop):")
    log.info(f"{'t':>3} {'RMSE':>8} {'abs_rel':>8} {'a1':>8}")
    for t in range(cfg.n_timesteps):
        n = counts[t]
        log.info(f"{t:3d} {per_t[t]['rmse']/n:8.4f} {per_t[t]['abs_rel']/n:8.4f} {per_t[t]['a1']/n:8.4f}")
    log.info("=" * 60)


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
