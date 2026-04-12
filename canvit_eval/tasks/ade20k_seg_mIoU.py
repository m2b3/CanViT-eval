"""ADE20K semantic segmentation evaluation.

ONE eval pipeline for ALL models (CanViT, DINOv3, anything).
The only thing that varies is how spatial features are extracted per batch.
Dataset loading, probe application, IoU computation are SHARED.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from canvit_probes import SegmentationProbe
from canvit_probes.datasets.ade20k import IGNORE_LABEL, NUM_CLASSES, ADE20kDataset, ResizeMode, make_val_transforms
from canvit_probes.metrics import mIoUAccumulator
from torch import Tensor
from torch.utils.data import DataLoader

from canvit_eval.config import ade20k_root
from canvit_eval.evaluate import evaluate
from canvit_eval.utils import collect_metadata

log = logging.getLogger(__name__)


@dataclass
class Config:
    """ADE20K segmentation eval — model-agnostic."""

    probe_repo: str
    ade20k_root: Path = field(default_factory=ade20k_root)
    output: Path = Path("results/ade20k_seg.pt")
    scene_size: int = 512
    resize_mode: ResizeMode = "squish"
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 8
    amp: bool = True


class _mIoUMetric:
    """Accumulates per-timestep IoU from spatial features."""

    def __init__(self, probe: SegmentationProbe, device: torch.device) -> None:
        self._probe = probe
        self._device = device
        self._accs: list[mIoUAccumulator] = []

    def update(self, t: int, features: Tensor, batch: tuple) -> None:
        _, masks = batch
        masks = masks.to(self._device, non_blocking=True)
        while len(self._accs) <= t:
            self._accs.append(mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, self._device))

        logits = self._probe(features.float())
        if logits.shape[-1] != masks.shape[-1]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        self._accs[t].update(logits.argmax(dim=1), masks)

    def compute(self) -> dict:
        assert len(self._accs) > 0, "No data accumulated"
        mious = {f"t{t}": acc.compute() for t, acc in enumerate(self._accs)}
        for k, v in mious.items():
            log.info("  %s: %.2f%%", k, 100 * v)
        return {"mious": mious}


def run(
    cfg: Config,
    extract_features: Callable[[Tensor], list[Tensor]],
    *,
    metadata: dict | None = None,
) -> Path:
    """Run ADE20K segmentation eval with ANY feature extractor.

    extract_features: [B, C, H, W] images → list of [B, G, G, D] per timestep.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(cfg.device)

    probe = SegmentationProbe.from_pretrained(cfg.probe_repo).to(device).eval()
    img_tf, mask_tf = make_val_transforms(cfg.scene_size, cfg.resize_mode)
    dataset = ADE20kDataset(root=cfg.ade20k_root, split="validation", img_transform=img_tf, mask_transform=mask_tf)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)

    return evaluate(
        loader=loader,
        extract_features=extract_features,
        metric=_mIoUMetric(probe, device),
        output=cfg.output,
        device=device,
        amp=cfg.amp,
        metadata={**(metadata or {}), **collect_metadata(cfg)},
    )
