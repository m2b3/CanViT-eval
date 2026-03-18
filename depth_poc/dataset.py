"""NYU Depth v2 dataset — BTS format (BinsFormer download).

Reads RGB images + uint16 depth PNGs. Applies NYU border crop,
resizes to square scene_size. Returns (image [3,H,W], depth [H,W]) in meters.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

# NYU border crop: removes unreliable depth at image edges.
# (x1, y1, x2, y2) on original 640×480 images.
NYU_CROP_BOX = (43, 45, 608, 472)


class NYUDepthV2(Dataset):
    """BTS-format NYU Depth v2: 24k train, 654 test samples."""

    def __init__(self, root: Path, split: Literal["train", "test"], scene_size: int) -> None:
        self.root = root
        self.scene_size = scene_size
        split_file = root / f"nyu_{split}.txt"
        assert split_file.exists(), f"Missing: {split_file}"
        self.pairs: list[tuple[str, str]] = []
        for line in split_file.read_text().strip().splitlines():
            img_rel, depth_rel, *_ = line.split()
            self.pairs.append((img_rel.strip("/"), depth_rel.strip("/")))
        self.pairs.sort()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img_rel, depth_rel = self.pairs[idx]
        img = Image.open(self.root / img_rel).convert("RGB")
        depth_pil = Image.open(self.root / depth_rel)

        # NYU border crop then resize to square.
        img = img.crop(NYU_CROP_BOX).resize((self.scene_size, self.scene_size), Image.BILINEAR)
        depth_pil = depth_pil.crop(NYU_CROP_BOX).resize((self.scene_size, self.scene_size), Image.NEAREST)

        img_t = TF.to_tensor(img)  # [3, H, W] float32 ∈ [0, 1]
        depth_t = torch.from_numpy(np.array(depth_pil, dtype=np.float32)) / 1000.0  # meters
        return img_t, depth_t
