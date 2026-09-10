"""Single place for the values that define a run.

The notebook scattered these across cells as module-level globals. Collecting
them here is what makes a run reproducible: one object, logged verbatim into
the results JSON so any number in the table can be traced to the settings that
produced it.
"""

from dataclasses import dataclass, asdict, field
from typing import Any, Dict


# The notebook's constants, unchanged. Editing these changes the numbers, so
# they are defaults rather than literals buried in function bodies.
IMG_SIZE = 256


@dataclass
class Config:
    # --- data ---
    img_size: int = IMG_SIZE
    num_workers: int = 0          # 0 is the safe default on Windows

    # --- model ---
    latent_ch: int = 64           # bottleneck 64x8x8 = 4096 values (v2 fix)

    # --- training ---
    epochs: int = 80
    batch_size: int = 16
    lr: float = 2e-4
    noise_sigma: float = 0.1      # denoising objective (v2 fix)
    loss_alpha: float = 0.5       # 0.5*MSE + 0.5*(1 - SSIM)

    # --- anomaly map / scoring ---
    ssim_window: int = 11         # local-SSIM dissimilarity map (v3 fix)
    topk_frac: float = 0.01       # image score = mean of hottest 1% of pixels

    # --- reproducibility ---
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
