"""Qualitative check: input / reconstruction / anomaly map / ground truth.

Kept as a first-class entry point rather than a notebook cell because looking
at the maps before trusting a metric is the discipline that produced v3. A
pixel-AUROC can look respectable while the map fires on rims and reflections
instead of the defect; only the picture shows that.

    python -m defectloc.visualize --checkpoint checkpoints/ae_v3.pt \
        --data-root data/MVTecAD/bottle --out results/figures/v3_clean.png
"""

import argparse
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from defectloc.anomaly import anomaly_map, normalize_map  # noqa: E402
from defectloc.data import TestSet  # noqa: E402
from defectloc.train import load_checkpoint, pick_device  # noqa: E402


@torch.no_grad()
def make_figure(model, cfg, root, device, n: int = 3, out=None, normalized: bool = False):
    ds = TestSet(root, img_size=cfg.img_size)
    defective = [i for i, (_, _, lab) in enumerate(ds.items) if lab == 1][:n]
    if not defective:
        raise SystemExit(f"{root} has no defective test images to visualise")

    fig, axes = plt.subplots(len(defective), 4, figsize=(14, 3.4 * len(defective)))
    axes = axes.reshape(len(defective), 4)

    for row, idx in enumerate(defective):
        img, mask, _ = ds[idx]
        x = img.unsqueeze(0).to(device)
        recon = model(x)[0].cpu().permute(1, 2, 0).numpy()
        amap = anomaly_map(model, x, win=cfg.ssim_window)[0]
        if normalized:
            amap = normalize_map(amap)

        defect_type = ds.items[idx][0].parent.name
        panels = [
            (img.permute(1, 2, 0).numpy(), f"input ({defect_type})", None),
            (recon, "reconstruction", None),
            (amap, "anomaly map" + (" (median/MAD)" if normalized else ""), "inferno"),
            (mask.numpy(), "ground truth", "gray"),
        ]
        for col, (data, title, cmap) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(data, cmap=cmap)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default="results/figures/anomaly_maps.png")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--normalized", action="store_true", help="show the median/MAD-scaled map")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    a = p.parse_args(argv)

    device = pick_device(a.device)
    model, cfg, _ = load_checkpoint(a.checkpoint, device)
    make_figure(model, cfg, a.data_root, device, n=a.n, out=a.out, normalized=a.normalized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
