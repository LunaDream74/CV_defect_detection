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


def load_lit(path, aug, img_size: int = IMG_SIZE) -> np.ndarray:
    """Like `load_img`, but apply ``aug`` at full resolution first.

    Mirrors scripts/prepare_data.build_lighting followed by `load_img`, so a
    lit training or calibration image matches how a lit test image was made.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = aug(image=img)["image"]
    img = cv2.resize(img, (img_size, img_size))
    return img.astype(np.float32) / 255.0


def load_mask(path: Optional[Path], img_size: int = IMG_SIZE) -> np.ndarray:
    """Read a binary ground-truth mask, or an all-zero mask for a good image."""
    if path is None or not Path(path).exists():
        return np.zeros((img_size, img_size), np.float32)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.float32)


def split_holdout(paths: List[Path], n_holdout: int, seed: int = 0) -> Tuple[List[Path], List[Path]]:
    """Deterministically split ``paths`` into (train, held-out).

    The held-out normals calibrate the score (Fix 2c). They must be images the
    model never trained on: a training image reconstructs better than an unseen
    normal one, so calibrating on it would understate how hot a normal map runs.
    """
    if not 0 <= n_holdout < len(paths):
        raise ValueError(f"cannot hold out {n_holdout} of {len(paths)} images")
    order = np.random.default_rng(seed).permutation(len(paths))
    held = set(order[:n_holdout].tolist())
    train = [p for i, p in enumerate(paths) if i not in held]
    holdout = [p for i, p in enumerate(paths) if i in held]
    return train, holdout


class TrainGood(Dataset):
    """Normal images only -- the model never sees a defect during training.

    ``holdout`` images are withheld (see `split_holdout`, seeded by
    ``split_seed``, default ``seed``); ``part`` picks which side of the split
    this dataset serves.

    ``lighting`` (an albumentations pipeline, Experiment 3) is applied to each
    sample with probability ``lighting_p``, at full resolution before the
    resize, in the same order the lighting test set was built. The sample *is*
    the lit image, so the model learns to reproduce a bottle under that
    lighting rather than to undo it.
    """

    def __init__(self, root, img_size: int = IMG_SIZE, holdout: int = 0, seed: int = 0,
                 part: str = "train", lighting=None, lighting_p: float = 0.0,
                 split_seed: Optional[int] = None):
        self.img_size = img_size
        self.lighting = lighting
        self.lighting_p = lighting_p if lighting is not None else 0.0
        # Own generator for the apply/skip draw: deterministic with num_workers=0.
        self._rng = np.random.default_rng(seed)
        paths = sorted((Path(root) / "train" / "good").glob("*.png"))
        if not paths:
            raise FileNotFoundError(
                f"no training images under {Path(root) / 'train' / 'good'}. "
                "Run `python scripts/prepare_data.py` first."
            )
        if part not in ("train", "holdout"):
            raise ValueError(f"part must be 'train' or 'holdout', got {part!r}")
        train, held = split_holdout(paths, holdout, seed if split_seed is None else split_seed)
        self.paths: List[Path] = train if part == "train" else held

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        if self.lighting_p > 0 and self._rng.random() < self.lighting_p:
            img = load_lit(self.paths[i], self.lighting, self.img_size)
        else:
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
