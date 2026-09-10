"""Train the autoencoder on normal images only.

    python -m defectloc.train --data-root data/MVTecAD/bottle
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from defectloc import __version__
from defectloc.config import Config
from defectloc.data import TrainGood
from defectloc.losses import recon_loss
from defectloc.model import ConvAutoencoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(root, cfg: Config, device: torch.device, log_every: int = 10, verbose: bool = True):
    """Train on ``<root>/train/good`` and return (model, history)."""
    set_seed(cfg.seed)

    ds = TrainGood(root, img_size=cfg.img_size)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=len(ds) > cfg.batch_size,
    )
    model = ConvAutoencoder(latent_ch=cfg.latent_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = []
    model.train()
    for ep in range(cfg.epochs):
        total, nb = 0.0, 0
        for x in loader:
            x = x.to(device)
            # Denoising objective: reconstruct the CLEAN image from a noised
            # input. Another brake on the identity shortcut -- copying the
            # input through now reproduces the noise too, and is punished.
            noise = torch.randn_like(x) * cfg.noise_sigma
            recon = model(x + noise)
            loss = recon_loss(recon, x, alpha=cfg.loss_alpha)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        epoch_loss = total / max(nb, 1)
        history.append({"epoch": ep + 1, "loss": epoch_loss})
        if verbose and ((ep + 1) % log_every == 0 or ep == 0):
            print(f"epoch {ep + 1:3d}/{cfg.epochs}  loss {epoch_loss:.4f}", flush=True)
    return model, history


def save_checkpoint(path, model, cfg: Config, history, extra=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "history": history,
            "version": __version__,
            **(extra or {}),
        },
        path,
    )
    return path


def load_checkpoint(path, device: torch.device):
    """Rebuild a model from a checkpoint, using the config stored inside it."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["config"])
    model = ConvAutoencoder(latent_ch=cfg.latent_ch).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg, ckpt


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the bottle-defect autoencoder.")
    d = Config()
    p.add_argument("--data-root", required=True, help="MVTec category folder (has train/good)")
    p.add_argument("--out", default="checkpoints/ae_v3.pt", help="checkpoint path")
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--img-size", type=int, default=d.img_size)
    p.add_argument("--latent-ch", type=int, default=d.latent_ch)
    p.add_argument("--noise-sigma", type=float, default=d.noise_sigma)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--log-every", type=int, default=10, help="print loss every N epochs")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config(
        img_size=args.img_size,
        num_workers=args.num_workers,
        latent_ch=args.latent_ch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
    )
    device = pick_device(args.device)
    print(f"device: {device}  |  config: {json.dumps(cfg.to_dict())}", flush=True)

    t0 = time.time()
    model, history = train(args.data_root, cfg, device, log_every=args.log_every)
    elapsed = time.time() - t0

    out = save_checkpoint(
        args.out, model, cfg, history,
        extra={"data_root": str(args.data_root), "train_seconds": round(elapsed, 1),
               "device": str(device)},
    )
    print(f"\ntrained in {elapsed:.1f}s  ->  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
