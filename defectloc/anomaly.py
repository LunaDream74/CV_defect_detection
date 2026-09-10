"""Anomaly map and image score.

Two separate decisions live here, and the project's open bug came from
conflating them:

* the **map** says *where* the image is anomalous (pixel-AUROC), and
* the **score** reduces that map to one number saying *whether* the image is
  anomalous (image-AUROC).

v3 fixed the map by switching from pixel-MSE to local-SSIM dissimilarity
(+15 points of pixel-AUROC with no retraining). The map survived the lighting
stress test; the score did not. See `image_score` for why, and for the fix.
"""

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

ScoreNorm = Literal["none", "median_mad"]

# Standard SSIM stabilisers for data in [0, 1].
_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


def local_ssim_dissimilarity(x: torch.Tensor, y: torch.Tensor, win: int = 11) -> torch.Tensor:
    """Per-pixel structural dissimilarity between two batches. Higher = more anomalous.

    Windowed SSIM computed on the channel-averaged image, returned as ``1 - SSIM``
    and clamped to [0, 1]. Reading *structure* rather than raw pixel values is
    what stops the map over-firing on normal high-frequency regions -- rims and
    reflections, which pixel-MSE lit up as brightly as real defects.
    """
    xg = x.mean(1, keepdim=True)
    yg = y.mean(1, keepdim=True)
    pad = win // 2

    mu_x = F.avg_pool2d(xg, win, 1, pad)
    mu_y = F.avg_pool2d(yg, win, 1, pad)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sig_x = F.avg_pool2d(xg * xg, win, 1, pad) - mu_x2
    sig_y = F.avg_pool2d(yg * yg, win, 1, pad) - mu_y2
    sig_xy = F.avg_pool2d(xg * yg, win, 1, pad) - mu_xy

    ssim_map = ((2 * mu_xy + _C1) * (2 * sig_xy + _C2)) / (
        (mu_x2 + mu_y2 + _C1) * (sig_x + sig_y + _C2)
    )
    return (1 - ssim_map).clamp(0, 1).squeeze(1)  # B,H,W


@torch.no_grad()
def anomaly_map(model, x: torch.Tensor, win: int = 11) -> np.ndarray:
    """Reconstruct ``x`` and return the B,H,W dissimilarity map as numpy."""
    model.eval()
    recon = model(x)
    return local_ssim_dissimilarity(x, recon, win=win).cpu().numpy()


def normalize_map(amap: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Robust per-image standardisation: ``(amap - median) / MAD``.

    This is the fix for the lighting failure. A global lighting shift moves an
    image's *whole* error baseline -- a dim frame reconstructs worse everywhere,
    so every pixel of its map sits higher. Raw top-k scores therefore stop being
    comparable between images shot under different lighting, and the image-level
    ranking collapses even though the map still points at the right pixels.

    Median and MAD describe that image's own normal background (the defect is a
    few percent of the pixels, so it barely moves either). Dividing it out asks
    "how far above *this image's* background is its hottest region", which is
    the question the score was always meant to answer.

    Median and MAD are used rather than mean and standard deviation precisely
    because the defect is an outlier: it would inflate the mean and the standard
    deviation it is supposed to be measured against.
    """
    med = np.median(amap)
    mad = np.median(np.abs(amap - med))
    return (amap - med) / (mad + eps)


def image_score(amap: np.ndarray, topk_frac: float = 0.01, norm: ScoreNorm = "none") -> float:
    """Reduce an anomaly map to one number: mean of the hottest ``topk_frac`` pixels.

    The top-k mean is more robust than a pure max, which a single hot pixel of
    reconstruction noise can dominate.

    Note that ``norm`` cannot change *which* pixels are selected -- the
    normalisation is monotonic within an image, so the same pixels stay hottest.
    It only changes the units the score is reported in, and that is exactly the
    point: it makes scores comparable *across* images.
    """
    if norm == "median_mad":
        amap = normalize_map(amap)
    elif norm != "none":
        raise ValueError(f"unknown score normalisation: {norm!r}")

    flat = np.sort(amap.ravel())[::-1]
    k = max(1, int(len(flat) * topk_frac))
    return float(flat[:k].mean())
