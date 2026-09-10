"""Datasets over an MVTec-AD category folder.

Expected layout (this is MVTec AD's own layout, unchanged):

    <root>/
        train/good/*.png
        test/good/*.png
        test/<defect_type>/*.png
        ground_truth/<defect_type>/<stem>_mask.png
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from defectloc.config import IMG_SIZE


def load_img(path, img_size: int = IMG_SIZE) -> np.ndarray:
    """Read an image as float32 RGB in [0, 1], HWC."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    return img.astype(np.float32) / 255.0


def load_mask(path: Optional[Path], img_size: int = IMG_SIZE) -> np.ndarray:
    """Read a binary ground-truth mask, or an all-zero mask for a good image."""
    if path is None or not Path(path).exists():
        return np.zeros((img_size, img_size), np.float32)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.float32)


class TrainGood(Dataset):
    """Normal images only -- the model never sees a defect during training."""

    def __init__(self, root, img_size: int = IMG_SIZE):
        self.img_size = img_size
        self.paths: List[Path] = sorted((Path(root) / "train" / "good").glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(
                f"no training images under {Path(root) / 'train' / 'good'}. "
                "Run `python scripts/prepare_data.py` first."
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        img = load_img(self.paths[i], self.img_size)
        return torch.from_numpy(img).permute(2, 0, 1)  # CHW


class TestSet(Dataset):
    """All test images, their ground-truth masks, and the image-level label."""

    # Not a pytest test class, despite the name; the name matches MVTec's split.
    __test__ = False

    def __init__(self, root, img_size: int = IMG_SIZE):
        root = Path(root)
        self.img_size = img_size
        self.items: List[Tuple[Path, Optional[Path], int]] = []
        test_dir = root / "test"
        if not test_dir.is_dir():
            raise FileNotFoundError(
                f"no test split under {test_dir}. "
                "Run `python scripts/prepare_data.py` first."
            )
        for sub in sorted(test_dir.iterdir()):
            if not sub.is_dir():
                continue
            defect = sub.name  # 'good' or a defect type
            for p in sorted(sub.glob("*.png")):
                label = 0 if defect == "good" else 1
                mask = (
                    None
                    if defect == "good"
                    else root / "ground_truth" / defect / f"{p.stem}_mask.png"
                )
                self.items.append((p, mask, label))
        if not self.items:
            raise FileNotFoundError(f"{test_dir} contains no .png images")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, mask, label = self.items[i]
        img = torch.from_numpy(load_img(path, self.img_size)).permute(2, 0, 1)
        m = torch.from_numpy(load_mask(mask, self.img_size))
        return img, m, label
