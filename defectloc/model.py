"""The autoencoder.

Symmetric 5-stage convolutional autoencoder. The bottleneck is deliberately
tight: v1 used 256x16x16 (65,536 values) and the model simply copied its input
through, so defects survived into the reconstruction and never stood out. The
64x8x8 bottleneck here is 16x smaller and forces the model to learn the
*structure* of a normal bottle instead.
"""

import torch.nn as nn


def _enc_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _dec_block(in_ch: int, out_ch: int, last: bool = False) -> nn.Sequential:
    layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)]
    if last:
        layers.append(nn.Sigmoid())
    else:
        layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class ConvAutoencoder(nn.Module):
    """3x256x256 -> 64x8x8 -> 3x256x256."""

    def __init__(self, latent_ch: int = 64):
        super().__init__()
        self.latent_ch = latent_ch
        self.encoder = nn.Sequential(
            _enc_block(3, 32),
            _enc_block(32, 64),
            _enc_block(64, 128),
            _enc_block(128, 256),
            _enc_block(256, latent_ch),
        )
        self.decoder = nn.Sequential(
            _dec_block(latent_ch, 256),
            _dec_block(256, 128),
            _dec_block(128, 64),
            _dec_block(64, 32),
            _dec_block(32, 3, last=True),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
