"""NYU Depth v2 — BTS format. Returns raw PIL pairs; caller provides transform."""

from pathlib import Path
from typing import Callable, Literal

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class NYUDepthV2(Dataset):

    def __init__(self, root: Path, split: Literal["train", "test"], transform: Callable) -> None:
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
        return self.transform(img, depth)
