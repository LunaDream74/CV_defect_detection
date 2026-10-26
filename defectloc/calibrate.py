"""Fix 2c: calibrate the anomaly map against held-out normal images.

Fix 2b standardised each map against *its own* pixels, so a defect inflated
the divisor meant to measure it. Here the reference comes from normal images
the model never trained on, so the image being scored cannot move it.

The calibration is **per pixel**, not per condition. A single mean and std per
condition is an affine map applied to every score in that condition, and
image-AUROC is blind to that: it would return exactly the raw number. Per-pixel
statistics do change the ranking. Each pixel's error is measured against how
hot that same pixel runs on normal bottles, which is also aimed at the rim
over-firing: the rim fires on normal images too, so its high normal mean and
spread are divided out.

    z(p) = (amap(p) - mu(p)) / max(sigma(p), floor),   floor = median_p sigma(p)

The floor stops near-constant background pixels, whose sigma is tiny, from
turning reconstruction noise into huge z values. It is fixed before any
measurement, not tuned.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from defectloc.anomaly import anomaly_map
from defectloc.augment import build_factory_lighting
from defectloc.config import Config
from defectloc.data import load_img

# Seed 0 builds the lighting TEST set (scripts/prepare_data.py). Calibration
# draws from the same lighting distribution with a different seed, so no
# calibration copy reproduces a test image's exact shift.
CALIB_LIGHTING_SEED = 1


@dataclass
class NormalStats:
    mu: np.ndarray        # H,W mean of normal anomaly maps
    sigma: np.ndarray     # H,W std of normal anomaly maps
    floor: float          # lower bound applied to sigma
    n_maps: int
    condition: str        # "clean" or "lighting"


def _load_lit(path: Path, aug, img_size: int) -> np.ndarray:
    """Mirror prepare_data.build_lighting then data.load_img: stress at full res, then resize."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = aug(image=img)["image"]
    img = cv2.resize(img, (img_size, img_size))
    return img.astype(np.float32) / 255.0


@torch.no_grad()
def normal_maps(model, paths: List[Path], cfg: Config, device, lighting: bool = False,
                copies: int = 1, batch_size: int = 8) -> np.ndarray:
    """Anomaly maps (N,H,W) of held-out normal images, optionally lighting-stressed."""
    if not paths:
        raise ValueError("no held-out normal images: train with --calib-holdout N")
    aug = build_factory_lighting(seed=CALIB_LIGHTING_SEED) if lighting else None
    imgs = []
    for _ in range(copies if lighting else 1):
        for p in paths:
            imgs.append(_load_lit(p, aug, cfg.img_size) if lighting else load_img(p, cfg.img_size))
    maps = []
    for i in range(0, len(imgs), batch_size):
        x = torch.from_numpy(np.stack(imgs[i:i + batch_size])).permute(0, 3, 1, 2).to(device)
        maps.append(anomaly_map(model, x, win=cfg.ssim_window))
    return np.concatenate(maps)


def fit_normal_stats(maps: np.ndarray, condition: str) -> NormalStats:
    if maps.ndim != 3 or len(maps) < 2:
        raise ValueError(f"need at least 2 maps shaped N,H,W, got {maps.shape}")
    mu = maps.mean(axis=0)
    sigma = maps.std(axis=0)
    return NormalStats(mu=mu, sigma=sigma, floor=float(np.median(sigma)),
                       n_maps=len(maps), condition=condition)


def z_map(amap: np.ndarray, stats: Optional[NormalStats]) -> np.ndarray:
    """Per-pixel z-score of ``amap`` against the held-out normal statistics."""
    if stats is None:
        raise ValueError("pixel_z scoring needs NormalStats from held-out normals")
    if amap.shape != stats.mu.shape:
        raise ValueError(f"map shape {amap.shape} != calibration shape {stats.mu.shape}")
    return (amap - stats.mu) / np.maximum(stats.sigma, max(stats.floor, 1e-8))
