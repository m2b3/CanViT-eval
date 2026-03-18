"""NYU Depth v2 dataset — BTS format (BinsFormer download).

Returns raw (PIL image, PIL depth) pairs. All preprocessing
(crops, resize, normalization) is handled by the transform pipeline
from dinov3.eval.depth.transforms.
"""

from pathlib import Path
from typing import Callable, Literal

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class NYUDepthV2(Dataset):
    """BTS-format NYU Depth v2: 24k train, 654 test samples."""

    def __init__(
        self,
        root: Path,
        split: Literal["train", "test"],
        transform: Callable[[Image.Image, Image.Image], tuple[Tensor, Tensor]] | None = None,
    ) -> None:
        self.root = root
        self.transform = transform
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
        depth = Image.open(self.root / depth_rel)
        if self.transform is not None:
            img, depth = self.transform(img, depth)
        return img, depth
