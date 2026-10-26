"""Experiment 4: flat-field the input against its own low-frequency illumination.

Arm B of Experiment 3 showed the model only handles lighting perturbations it
trained on: a shadow it never saw stayed below chance. A shadow is a smooth
multiplicative gradient, m(p) * I(p). If m varies slowly next to the blur
width, blur(m * I) ~= m * blur(I), so

    flat(I) = I / max(blur(I), floor) * gain

cancels m without knowing it exists. Defects are local (a break, a speck), so
they survive the division. This is a physical prior about illumination, not a
fit to any one perturbation.

Parameters are fixed from training images before any measurement, not tuned:
sigma = 16 px at 256 (1/16 of the frame: rim rings survive, a frame-wide
gradient does not); floor = 0.05 (the darkest blurred training value is ~0.14,
~0.06 under the strongest shadow, so it is a safety bound that never binds on
a normal frame); gain = 0.4 (the 99.9th-percentile ratio on training images is
2.09, so clipping at 1.0 starts only above a ratio of 2.5).
"""

import math

import torch
from torchvision.transforms.functional import gaussian_blur

from defectloc.config import Config


def flatfield(x: torch.Tensor, sigma: float = 16.0, floor: float = 0.05,
              gain: float = 0.4) -> torch.Tensor:
    """Divide a B,C,H,W batch in [0, 1] by its own blur, per channel; output in [0, 1]."""
    k = 2 * math.ceil(3 * sigma) + 1
    if k >= min(x.shape[-2:]):
        raise ValueError(f"blur kernel {k} does not fit a {tuple(x.shape[-2:])} image; lower sigma")
    illum = gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma]).clamp_min(floor)
    return (x / illum * gain).clamp(0.0, 1.0)


def prepare(x: torch.Tensor, cfg: Config) -> torch.Tensor:
    """The model's input transform: flat-field unless ``cfg.flatfield_sigma`` is 0.

    Applied to every image the model sees -- training, test, calibration and
    figures -- so the anomaly map compares like with like. ``flatfield_sigma``
    is in pixels at a 256-px image and scales with ``cfg.img_size``, so the
    blur covers the same fraction of the frame at any resolution.
    """
    if cfg.flatfield_sigma <= 0:
        return x
    sigma = cfg.flatfield_sigma * cfg.img_size / 256
    return flatfield(x, sigma, cfg.flatfield_floor, cfg.flatfield_gain)
