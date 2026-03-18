"""NYU Depth v2 linear depth probe.

Trains a linear depth probe on frozen DINOv3 or CanViT features.
Everything except the dataset class and feature extraction is imported
from the dinov3 codebase: transforms, loss, metrics, scheduler, bin
conversion. Source: dinov3/eval/depth/configs/config-nyu.yaml.

Only deviation: square resize (DINOv3 uses native 480×640).

    uv run python depth_poc/train.py --nyu-root /datasets/NYU/nyu
    uv run python depth_poc/train.py --mode canvit --nyu-root /datasets/NYU/nyu
"""

import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from dinov3.eval.depth.datasets.datasets_utils import _EvalCropType, make_valid_mask
from dinov3.eval.depth.loss import SigLoss
from dinov3.eval.depth.metrics import calculate_depth_metrics
from dinov3.eval.depth.models import FeaturesToDepth
from dinov3.eval.depth.transforms import make_depth_eval_transforms, make_depth_train_transforms
from dinov3.eval.segmentation.schedulers import WarmupOneCycleLR
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from depth_poc.dataset import NYUDepthV2

log = logging.getLogger(__name__)

# All from config-nyu.yaml unless noted.
MIN_DEPTH = 0.001
MAX_DEPTH = 10.0
N_BINS = 256
DEPTH_NORM = 1000.0  # uint16 → meters
BRIGHTNESS_RANGE = (0.75, 1.25)
NYU_TRAIN_SIZE = 24231
# DINOv3 config: batch 2 × 8 GPUs = 16, lr 3e-4, 38400 steps = 25 epochs.
DINOV3_BATCH = 16
DINOV3_LR = 3e-4
DINOV3_EPOCHS = 25


@dataclass
class Config:
    mode: Literal["teacher", "canvit"] = "teacher"
    nyu_root: Path = Path("/datasets/NYU/nyu")

    scene_size: int = 512
    crop_size: int = 560  # resize here, then random-crop to scene_size during training

    model_repo: str = "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"
    teacher_repo: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"

    n_bins: int = N_BINS
    dropout: float = 0.1

    batch_size: int = 4
    lr: float = DINOV3_LR * 4 / DINOV3_BATCH  # linear scaling
    weight_decay: float = 1e-4
    grad_clip: float = 35.0
    n_epochs: int = DINOV3_EPOCHS

    n_timesteps: int = 10
    canvas_grid: int = 32
    glimpse_px: int = 128

    eval_every: int = 4000
    log_every: int = 50
    num_workers: int = 4
    comet_project: str = "canvit-depth-poc"
    comet_workspace: str = "m2b3-ava"
    device: str = "cuda"
    amp: bool = True
    ckpt_dir: Path = Path("checkpoints/depth")

    @property
    def max_steps(self) -> int:
        return (NYU_TRAIN_SIZE // self.batch_size) * self.n_epochs

    @property
    def warmup_steps(self) -> int:
        return self.max_steps // 3


class DepthProbe(nn.Module):
    """LN → Dropout2d → BN → Conv1x1(D → bins) → AdaBins depth."""

    def __init__(self, embed_dim: int, n_bins: int = N_BINS, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout2d(dropout)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.conv = nn.Conv2d(embed_dim, n_bins, kernel_size=1)
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv.bias, 0)
        self.to_depth = FeaturesToDepth(min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)

    def forward(self, x: Tensor) -> Tensor:
        """[B, H, W, D] → [B, 1, H, W]."""
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.dropout(x)
        x = self.bn(x)
        return self.to_depth(self.conv(x))


# Images arrive already ImageNet-normalized from dinov3 transforms.

def extract_teacher_features(teacher: nn.Module, images: Tensor) -> Tensor:
    feats = teacher.forward_norm_features(images)
    B, N, D = feats.patches.shape
    H = W = int(N**0.5)
    assert H * W == N
    return feats.patches.view(B, H, W, D)


def extract_canvit_features(model: nn.Module, images: Tensor, cfg: Config) -> list[Tensor]:
    from canvit import sample_at_viewpoint
    from canvit.policies import random_viewpoints
    B, device = images.shape[0], images.device
    vps = random_viewpoints(B, device, cfg.n_timesteps, min_scale=0.05, max_scale=1.0, start_with_full_scene=True)
    state = model.init_state(batch_size=B, canvas_grid_size=cfg.canvas_grid)
    out: list[Tensor] = []
    for vp in vps:
        glimpse = sample_at_viewpoint(spatial=images, viewpoint=vp, glimpse_size_px=cfg.glimpse_px)
        result = model(glimpse=glimpse, state=state, viewpoint=vp)
        state = result.state
        out.append(model.get_spatial(state.canvas).view(B, cfg.canvas_grid, cfg.canvas_grid, -1))
    return out


@torch.no_grad()
def evaluate(probe: nn.Module, backbone: nn.Module, loader: DataLoader,
             cfg: Config, device: torch.device, amp_ctx: torch.amp.autocast) -> dict[str, float]:
    probe.eval()
    sums: dict[str, float] = {}
    n = 0
    for images, depths in loader:
        images, depths = images.to(device), depths.to(device)
        if depths.ndim == 4:
            depths = depths.squeeze(1)
        with amp_ctx:
            feats = (extract_teacher_features(backbone, images) if cfg.mode == "teacher"
                     else extract_canvit_features(backbone, images, cfg)[-1])
            pred = probe(feats.float())
        pred = F.interpolate(pred, size=depths.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        pred = pred.clamp(MIN_DEPTH, MAX_DEPTH)
        mask = make_valid_mask(depths.unsqueeze(1), eval_crop=_EvalCropType.NYU_EIGEN).squeeze(1)
        m = calculate_depth_metrics(depths, pred, mask)
        for k in ("rmse", "abs_rel", "a1"):
            v = getattr(m, k)
            sums[k] = sums.get(k, 0.0) + (v.item() if isinstance(v, Tensor) else float(v))
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


def train(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.device)

    log.info(f"mode={cfg.mode} scene={cfg.scene_size} crop={cfg.crop_size} "
             f"lr={cfg.lr} epochs={cfg.n_epochs} steps={cfg.max_steps}")

    s = cfg.scene_size
    train_tf = make_depth_train_transforms(
        normalization_constant=DEPTH_NORM, img_size=(cfg.crop_size, cfg.crop_size),
        random_crop_size=(s, s), fixed_crop="NYU", brightness_range=BRIGHTNESS_RANGE,
    )
    _eval_tf = make_depth_eval_transforms(
        normalization_constant=DEPTH_NORM, img_size=(s, s), fixed_crop="FULL", tta=False,
    )
    eval_tf = lambda img, depth: tuple(x[0] for x in _eval_tf(img, depth))

    train_ds = NYUDepthV2(cfg.nyu_root, "train", transform=train_tf)
    test_ds = NYUDepthV2(cfg.nyu_root, "test", transform=eval_tf)
    log.info(f"Train: {len(train_ds)}, Test: {len(test_ds)}")
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)

    if cfg.mode == "teacher":
        from canvit_utils.teacher import load_teacher
        backbone = load_teacher(cfg.teacher_repo, device)
        embed_dim = backbone.embed_dim
    else:
        from canvit import CanViTForPretrainingHFHub
        backbone = CanViTForPretrainingHFHub.from_pretrained(cfg.model_repo).to(device).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        embed_dim = backbone.canvas_dim

    probe = DepthProbe(embed_dim, cfg.n_bins, cfg.dropout).to(device)
    opt = AdamW(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = WarmupOneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.max_steps,
        warmup_iters=cfg.warmup_steps, warmup_ratio=1e-6,
        pct_start=0, anneal_strategy="cos", final_div_factor=1000.0,
        use_beta1=False, update_momentum=False,
    )
    criterion = SigLoss(warm_up=True, warm_iter=100)
    log.info(f"Probe: {sum(p.numel() for p in probe.parameters()):,} params")

    import comet_ml
    api_key = os.environ.get("COMET_API_KEY")
    if not api_key:
        kf = Path.home() / "comet_api_key.txt"
        if kf.exists():
            api_key = kf.read_text().strip()
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    if api_key:
        exp = comet_ml.Experiment(api_key=api_key, project_name=cfg.comet_project, workspace=cfg.comet_workspace)
    else:
        log.warning("No Comet API key — offline")
        exp = comet_ml.OfflineExperiment(project_name=cfg.comet_project, offline_directory=str(cfg.ckpt_dir))
    exp.set_name(f"depth_{cfg.mode}_s{cfg.scene_size}_{time.strftime('%Y%m%d_%H%M%S')}")
    exp.log_parameters(asdict(cfg))
    for tag in ["depth", cfg.mode, f"s{cfg.scene_size}", f"e{cfg.n_epochs}"]:
        exp.add_tag(tag)

    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.amp)
    feat_h = cfg.scene_size // 16 if cfg.mode == "teacher" else cfg.canvas_grid
    best_rmse = float("inf")
    step = 0
    train_iter = iter(train_loader)
    pbar = tqdm(total=cfg.max_steps, desc="Training")

    while step < cfg.max_steps:
        try:
            images, depths = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, depths = next(train_iter)
        images, depths = images.to(device), depths.to(device)
        if depths.ndim == 4:
            depths = depths.squeeze(1)
        depth_low = F.interpolate(depths.unsqueeze(1), size=(feat_h, feat_h), mode="nearest").squeeze(1)
        valid_low = (depth_low > MIN_DEPTH) & (depth_low < MAX_DEPTH)

        if step % cfg.eval_every == 0:
            metrics = evaluate(probe, backbone, test_loader, cfg, device, amp_ctx)
            log.info(f"Step {step}: RMSE={metrics['rmse']:.4f} abs_rel={metrics['abs_rel']:.4f} a1={metrics['a1']:.4f}")
            for k, v in metrics.items():
                exp.log_metric(f"test/{k}", v, step=step)
            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                for old in cfg.ckpt_dir.glob("best_*.pt"):
                    old.unlink()
                path = cfg.ckpt_dir / f"best_rmse{best_rmse:.4f}_step{step}.pt"
                torch.save({"step": step, "probe": probe.state_dict(), "metrics": metrics, "config": asdict(cfg)}, path)
                exp.log_model("best_probe", str(path))
                log.info(f"  → new best: {path.name}")
                exp.log_metric("test/best_rmse", best_rmse, step=step)
            probe.train()

        with amp_ctx:
            if cfg.mode == "teacher":
                loss = criterion(probe(extract_teacher_features(backbone, images).float()).squeeze(1), depth_low, valid_low)
            else:
                feats_per_t = extract_canvit_features(backbone, images, cfg)
                loss = torch.stack([criterion(probe(f.float()).squeeze(1), depth_low, valid_low) for f in feats_per_t]).mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(probe.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        step += 1
        pbar.update(1)
        if step % cfg.log_every == 0:
            exp.log_metrics({"train/loss": loss.item(), "lr": sched.get_last_lr()[0]}, step=step)

    pbar.close()
    final_path = cfg.ckpt_dir / f"final_step{step}.pt"
    torch.save({"step": step, "probe": probe.state_dict(), "config": asdict(cfg)}, final_path)
    exp.log_model("final_probe", str(final_path))
    log.info(f"Done. Best test RMSE: {best_rmse:.4f}")
    exp.end()


if __name__ == "__main__":
    train(tyro.cli(Config))
