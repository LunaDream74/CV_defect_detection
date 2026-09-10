"""The factory-lighting stress test.

Applied to the **test set only**. Training stays clean, so this measures what
happens when a model trained under one lighting rig meets another -- the single
most common way a working inspection model degrades after installation.

Ranges are chosen to be noticeable but not destructive: a person should still
easily see the bottle and any defect. If a human cannot, the model should not
be blamed for failing and the test is unfair.
"""

import random

import albumentations as A
import numpy as np


class DirectionalShadow(A.ImageOnlyTransform):
    """Darken the image along a random linear gradient.

    Mimics uneven overhead lighting or a shadow cast across the conveyor --
    the production-specific failure mode that a global brightness shift does
    not capture.
    """

    def __init__(self, min_factor: float = 0.4, max_factor: float = 0.85, p: float = 0.5):
        super().__init__(p=p)
        self.min_factor = min_factor  # darkest multiplier at the shadowed edge
        self.max_factor = max_factor

    def apply(self, img, **kwargs):
        h, w = img.shape[:2]
        angle = np.random.uniform(0, 2 * np.pi)
        xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        grad = xx * np.cos(angle) + yy * np.sin(angle)
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
        dark = np.random.uniform(self.min_factor, self.max_factor)
        mask = dark + (1 - dark) * grad  # ranges [dark, 1.0] across the image
        out = img.astype(np.float32) * mask[..., None]
        return np.clip(out, 0, 255).astype(img.dtype)

    def get_transform_init_args_names(self):
        return ("min_factor", "max_factor")


def build_factory_lighting(seed: int | None = None) -> A.Compose:
    """Photometric-only pipeline: masks pass through unchanged.

    The notebook left this unseeded, which meant the lighting test set was a
    different set of images on every run and its numbers were not comparable
    between runs. Seeding it is the difference between a stress test and an
    anecdote.
    """
    transforms = [
        # Overall brightness/contrast drift -- bulbs dimming, haze.
        A.RandomBrightnessContrast(
            brightness_limit=(-0.25, 0.25),   # +/-25%, visible but not blown out
            contrast_limit=(-0.20, 0.20),     # +/-20%
            p=0.9,
        ),
        # Non-linear exposure response, more realistic than linear brightness.
        A.RandomGamma(gamma_limit=(80, 120), p=0.7),   # 0.8x - 1.2x gamma
        # Colour temperature: warm/cool cast from different bulb types. Mild --
        # glass tint should shift without recolouring the bottle entirely.
        A.HueSaturationValue(
            hue_shift_limit=8,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.6,
        ),
        DirectionalShadow(min_factor=0.45, max_factor=0.85, p=0.5),
    ]
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    try:
        # albumentations >= 1.4.21 accepts a per-Compose seed.
        return A.Compose(transforms, additional_targets={}, seed=seed)
    except TypeError:
        # Older releases: the global seeds set above are what make it repeatable.
        return A.Compose(transforms, additional_targets={})
