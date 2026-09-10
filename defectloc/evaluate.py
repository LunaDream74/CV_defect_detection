"""Evaluate a checkpoint: image-AUROC and pixel-AUROC, PatchCore-comparable.

Scores both ways in a single pass over the test set -- raw top-k (the v3
notebook behaviour) and median/MAD-normalised (the fix) -- so the two are
always measured on identical maps and the comparison cannot drift.

    python -m defectloc.evaluate --checkpoint checkpoints/ae_v3.pt \
        --clean-root data/MVTecAD/bottle \
        --lighting-root data/MVTecAD_lighting/bottle
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from defectloc.anomaly import anomaly_map, image_score, normalize_map
from defectloc.config import Config
from defectloc.data import TestSet
from defectloc.train import load_checkpoint, pick_device, set_seed

SCORE_VARIANTS = ("none", "median_mad")


def _auroc(y_true, y_score):
    """AUROC, or None when only one class is present (an undefined metric)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


@torch.no_grad()
def evaluate(model, root, cfg: Config, device, batch_size: int = 8) -> dict:
    """Return image/pixel AUROC for every scoring variant, plus a per-type breakdown."""
    ds = TestSet(root, img_size=cfg.img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Defect type per item, in dataset order, for the per-type breakdown.
    types = [p.parent.name for p, _, _ in ds.items]

    pixel_gt = []
    pixel_scores = {v: [] for v in SCORE_VARIANTS}
    img_gt = []
    img_scores = {v: [] for v in SCORE_VARIANTS}

    for imgs, masks, labels in loader:
        amaps = anomaly_map(model, imgs.to(device), win=cfg.ssim_window)  # B,H,W
        masks = masks.numpy()
        for amap, m, lab in zip(amaps, masks, labels.numpy()):
            pixel_gt.append(m.ravel())
            img_gt.append(int(lab))
            for v in SCORE_VARIANTS:
                scored = normalize_map(amap) if v == "median_mad" else amap
                pixel_scores[v].append(scored.ravel())
                img_scores[v].append(image_score(amap, topk_frac=cfg.topk_frac, norm=v))

    pixel_gt_all = np.concatenate(pixel_gt)
    img_gt = np.asarray(img_gt)
    types = np.asarray(types)

    out = {"n_images": len(ds), "n_defective": int(img_gt.sum()), "variants": {}}
    for v in SCORE_VARIANTS:
        img_s = np.asarray(img_scores[v])
        px_s = np.concatenate(pixel_scores[v])

        per_type = {}
        good = types == "good"
        for t in sorted(set(types[~good])):
            sel = good | (types == t)
            px_sel = np.concatenate([pixel_scores[v][i] for i in np.flatnonzero(sel)])
            gt_sel = np.concatenate([pixel_gt[i] for i in np.flatnonzero(sel)])
            per_type[t] = {
                "n": int((types == t).sum()),
                "image_AUROC": _auroc(img_gt[sel], img_s[sel]),
                "pixel_AUROC": _auroc(gt_sel, px_sel),
            }

        out["variants"][v] = {
            "image_AUROC": _auroc(img_gt, img_s),
            "pixel_AUROC": _auroc(pixel_gt_all, px_s),
            "per_defect_type": per_type,
        }
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a bottle-defect autoencoder checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--clean-root", required=True, help="clean MVTec category folder")
    p.add_argument("--lighting-root", default=None, help="lighting-stressed copy (optional)")
    p.add_argument("--out", default="results/runs/latest.json", help="where to write the run JSON")
    p.add_argument("--label", default="Autoencoder v3", help="row label in the results table")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    device = pick_device(args.device)
    model, cfg, ckpt = load_checkpoint(args.checkpoint, device)
    set_seed(cfg.seed)
    print(f"device: {device}  |  checkpoint: {args.checkpoint}", flush=True)

    run = {
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "config": cfg.to_dict(),
        "train_seconds": ckpt.get("train_seconds"),
        "final_train_loss": (ckpt.get("history") or [{}])[-1].get("loss"),
        "conditions": {},
    }

    for name, root in (("clean", args.clean_root), ("lighting", args.lighting_root)):
        if root is None:
            continue
        res = evaluate(model, root, cfg, device, batch_size=args.batch_size)
        res["root"] = str(root)
        run["conditions"][name] = res
        for v in SCORE_VARIANTS:
            m = res["variants"][v]
            print(
                f"{name:9s} [{v:10s}]  image-AUROC {m['image_AUROC']:.4f}"
                f"   pixel-AUROC {m['pixel_AUROC']:.4f}"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
