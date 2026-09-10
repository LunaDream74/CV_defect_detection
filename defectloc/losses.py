"""Training objective: 0.5*MSE + 0.5*(1 - SSIM).

A tight bottleneck produces blurry reconstructions. Pure MSE punishes that
blur everywhere and drags the model toward copying; the SSIM term tolerates
uniform blur while still penalising *structural* difference, which is what a
defect is.
"""

import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_fn


def recon_loss(recon, target, alpha: float = 0.5):
    mse = F.mse_loss(recon, target)
    s = ssim_fn(recon, target, data_range=1.0, size_average=True)
    return alpha * mse + (1 - alpha) * (1 - s)
