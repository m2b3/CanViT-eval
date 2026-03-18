"""NYU Depth v2 linear depth probe — DINOv3 teacher & CanViT features.

Standalone POC for the paper. Reuses dinov3 loss/metrics. Designed for
eventual integration into canvit-probes.

Usage (teacher baseline):
    uv run python depth_poc/train.py --nyu-root /datasets/NYU/nyu

Usage (CanViT features):
    uv run python depth_poc/train.py --mode canvit --nyu-root /datasets/NYU/nyu
"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from dinov3.eval.depth.loss import SigLoss
from dinov3.eval.depth.metrics import calculate_depth_metrics
from dinov3.eval.depth.models import FeaturesToDepth
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from depth_poc.dataset import NYUDepthV2

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MIN_DEPTH = 0.001
MAX_DEPTH = 10.0


# ── Config ────────────────────────────────────────────────────────────


@dataclass
class Config:
    mode: Literal["teacher", "canvit"] = "teacher"
    nyu_root: Path = Path("/datasets/NYU/nyu")
    scene_size: int = 512

    # Model repos.
    model_repo: str = "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"
    teacher_repo: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"

    # Probe.
    n_bins: int = 256
    dropout: float = 0.1

    # Training.
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_steps: int = 10_000
    warmup_frac: float = 0.33
    grad_clip: float = 35.0

    # CanViT episode.
    n_timesteps: int = 5
    canvas_grid: int = 32
    glimpse_px: int = 128

    # Logging / checkpointing.
    eval_every: int = 500
    log_every: int = 50
    num_workers: int = 4
    comet_project: str = "canvit-depth-poc"
    comet_workspace: str = "m2b3-ava"
    device: str = "cuda"
    amp: bool = True
    ckpt_dir: Path = Path("checkpoints/depth-poc")


# ── Depth Probe ───────────────────────────────────────────────────────


class DepthProbe(nn.Module):
    """Linear depth probe: LN → Dropout2d → BN → Conv1x1(D→bins) → depth.

    Architecture mirrors SegmentationProbe but outputs bin logits
    converted to metric depth via AdaBins-style weighted sum (dinov3).
    """

    def __init__(self, embed_dim: int, n_bins: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout2d(dropout)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.conv = nn.Conv2d(embed_dim, n_bins, kernel_size=1)
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv.bias, 0)
        self.to_depth = FeaturesToDepth(
            min_depth=MIN_DEPTH, max_depth=MAX_DEPTH,
            bins_strategy="linear", norm_strategy="linear",
        )

    def forward(self, x: Tensor) -> Tensor:
        """[B, H, W, D] spatial features → [B, 1, H, W] depth in meters."""
        B, H, W, D = x.shape
        assert D == self.embed_dim
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.dropout(x)
        x = self.bn(x)
        bins = self.conv(x)
        return self.to_depth(bins)  # [B, 1, H, W]


# ── Feature extraction ────────────────────────────────────────────────


def _normalize(images: Tensor) -> Tensor:
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def extract_teacher_features(teacher: nn.Module, images: Tensor) -> Tensor:
    """DINOv3 ViT → [B, H, W, D] spatial patch features."""
    feats = teacher.forward_norm_features(_normalize(images))
    B, N, D = feats.patches.shape
    H = W = int(N**0.5)
    assert H * W == N, f"Non-square patch grid: {N}"
    return feats.patches.view(B, H, W, D)


def extract_canvit_features(
    model: nn.Module, images: Tensor, cfg: Config,
) -> list[Tensor]:
    """CanViT episode → list of [B, G, G, D] per timestep."""
    from canvit import sample_at_viewpoint
    from canvit.policies import random_viewpoints

    images_norm = _normalize(images)
    B = images_norm.shape[0]
    device = images_norm.device

    vps = random_viewpoints(B, device, cfg.n_timesteps, min_scale=0.05, max_scale=1.0, start_with_full_scene=True)
    state = model.init_state(batch_size=B, canvas_grid_size=cfg.canvas_grid)

    out_list: list[Tensor] = []
    for vp in vps:
        glimpse = sample_at_viewpoint(spatial=images_norm, viewpoint=vp, glimpse_size_px=cfg.glimpse_px)
        out = model(glimpse=glimpse, state=state, viewpoint=vp)
        state = out.state
        out_list.append(model.get_spatial(state.canvas).view(B, cfg.canvas_grid, cfg.canvas_grid, -1))
    return out_list


# ── Evaluation ────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    probe: nn.Module,
    backbone: nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    amp_ctx: torch.amp.autocast,
) -> dict[str, float]:
    """Run test set, return averaged depth metrics."""
    probe.eval()
    sums: dict[str, float] = {}
    n = 0
    for images, depths in loader:
        images, depths = images.to(device), depths.to(device)
        valid = (depths > MIN_DEPTH) & (depths < MAX_DEPTH)
        with amp_ctx:
            if cfg.mode == "teacher":
                feats = extract_teacher_features(backbone, images)
            else:
                feats = extract_canvit_features(backbone, images, cfg)[-1]
            pred = probe(feats.float())
        pred = F.interpolate(pred, size=depths.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        pred = pred.clamp(MIN_DEPTH, MAX_DEPTH)
        m = calculate_depth_metrics(depths, pred, valid.unsqueeze(1))
        for field in ("rmse", "abs_rel", "a1"):
            v = getattr(m, field)
            sums[field] = sums.get(field, 0.0) + (v.item() if isinstance(v, Tensor) else float(v))
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


# ── Training ──────────────────────────────────────────────────────────


def train(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.device)

    log.info("=" * 60)
    log.info(f"NYU Depth v2 POC — mode={cfg.mode}, scene={cfg.scene_size}")
    log.info("=" * 60)

    # ── Data ──
    train_ds = NYUDepthV2(cfg.nyu_root, "train", cfg.scene_size)
    test_ds = NYUDepthV2(cfg.nyu_root, "test", cfg.scene_size)
    log.info(f"Train: {len(train_ds)}, Test: {len(test_ds)}")
    train_loader = DataLoader(
        train_ds, cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(test_ds, cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)

    # ── Backbone (frozen) ──
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

    # ── Probe ──
    probe = DepthProbe(embed_dim, cfg.n_bins, cfg.dropout).to(device)
    opt = AdamW(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.max_steps, pct_start=cfg.warmup_frac)
    criterion = SigLoss(warm_up=True, warm_iter=100)
    n_params = sum(p.numel() for p in probe.parameters())
    log.info(f"Probe: {n_params:,} params, {cfg.n_bins} bins, embed_dim={embed_dim}")

    # ── Comet ──
    exp = comet_ml.Experiment(project_name=cfg.comet_project, workspace=cfg.comet_workspace)
    ts = time.strftime("%Y%m%d_%H%M%S")
    exp.set_name(f"depth_{cfg.mode}_s{cfg.scene_size}_{ts}")
    exp.log_parameters(asdict(cfg))
    exp.add_tag("depth-poc")
    exp.add_tag(cfg.mode)

    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.amp)
    best_rmse = float("inf")
    feat_h = cfg.scene_size // 16 if cfg.mode == "teacher" else cfg.canvas_grid

    # ── Training loop ──
    step = 0
    train_iter = iter(train_loader)
    pbar = tqdm(total=cfg.max_steps, desc="Training")

    while step < cfg.max_steps:
        # Get batch.
        try:
            images, depths = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, depths = next(train_iter)
        images, depths = images.to(device), depths.to(device)

        # Random horizontal flip (joint).
        if torch.rand(1).item() > 0.5:
            images = images.flip(-1)
            depths = depths.flip(-1)

        # Downsample depth to feature resolution for loss.
        depth_low = F.interpolate(
            depths.unsqueeze(1), size=(feat_h, feat_h), mode="nearest",
        ).squeeze(1)
        valid_low = (depth_low > MIN_DEPTH) & (depth_low < MAX_DEPTH)

        # ── Eval ──
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
                log.info(f"  → new best: {path.name}")
                exp.log_metric("test/best_rmse", best_rmse, step=step)
            probe.train()

        # ── Forward ──
        with amp_ctx:
            if cfg.mode == "teacher":
                feats = extract_teacher_features(backbone, images)
                pred = probe(feats.float()).squeeze(1)  # [B, feat_h, feat_h]
                loss = criterion(pred, depth_low, valid_low)
            else:
                feats_per_t = extract_canvit_features(backbone, images, cfg)
                losses = []
                for feats_t in feats_per_t:
                    pred_t = probe(feats_t.float()).squeeze(1)
                    losses.append(criterion(pred_t, depth_low, valid_low))
                loss = torch.stack(losses).mean()

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
    torch.save({"step": step, "probe": probe.state_dict(), "config": asdict(cfg)}, cfg.ckpt_dir / f"final_step{step}.pt")
    log.info(f"Done. Best test RMSE: {best_rmse:.4f}")
    exp.end()


if __name__ == "__main__":
    train(tyro.cli(Config))
