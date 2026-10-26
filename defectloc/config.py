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

    # --- Fix 2c: per-pixel calibration on held-out normals ---
    calib_holdout: int = 0        # train/good images withheld from training (0 = none)
    holdout_seed: int = 0         # seeds the split only, so --seed varies training alone
    calib_copies: int = 4         # lighting-stressed copies of each held-out image

    # --- Experiment 3: lighting-augmented training ---
    train_lighting_p: float = 0.0     # per-sample probability of the lighting pipeline
    train_lighting_shadow: bool = True  # include DirectionalShadow in that pipeline
    train_lighting_seed: int = 2      # test set uses 0, calibration 1

    # --- Experiment 4: flat-field input (defectloc/preprocess.py) ---
    # On by default since arm D replicated (2026-10-26). Blur width in px at a
    # 256-px image, scaled with img_size; 0 = off. Checkpoints saved before this
    # field existed load with it off (train.load_checkpoint), as they were trained.
    flatfield_sigma: float = 16.0
    flatfield_floor: float = 0.05
    flatfield_gain: float = 0.4

    # --- reproducibility ---
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
