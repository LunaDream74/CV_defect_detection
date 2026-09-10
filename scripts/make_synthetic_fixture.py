"""Generate a tiny MVTec-shaped dataset for smoke-testing the pipeline.

This exists so the code can be verified end to end -- train, map, score,
evaluate, report -- on a machine that does not have MVTec AD downloaded.

It is a **plumbing check, not a benchmark**. The images are procedural rings,
not bottles. Numbers produced from this fixture are meaningless as results and
must never appear in the results table.

    python scripts/make_synthetic_fixture.py --out data/synthetic/bottle
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def _normal(rng, size):
    """A 'good' part: a bright ring on a dark ground, with mild nuisance variation."""
    img = np.full((size, size, 3), 30, np.float32)
    c = size // 2
    jitter = rng.integers(-2, 3, size=2)
    cv2.circle(img, (c + int(jitter[0]), c + int(jitter[1])), int(size * 0.38),
               (200, 205, 195), thickness=max(2, size // 32))
    cv2.circle(img, (c + int(jitter[0]), c + int(jitter[1])), int(size * 0.22),
               (120, 130, 125), thickness=max(1, size // 48))
    img += rng.normal(0, 4, img.shape).astype(np.float32)   # sensor noise
    img *= rng.uniform(0.93, 1.07)                           # exposure variation
    return np.clip(img, 0, 255).astype(np.uint8)


def _defect(rng, img, kind):
    """Paste a defect and return (image, binary mask)."""
    size = img.shape[0]
    mask = np.zeros((size, size), np.uint8)
    out = img.copy()
    c = size // 2
    if kind == "chip":
        ang = rng.uniform(0, 2 * np.pi)
        r = int(size * 0.38)
        p = (int(c + r * np.cos(ang)), int(c + r * np.sin(ang)))
        rad = max(2, size // 20)
        cv2.circle(out, p, rad, (25, 25, 25), -1)     # material missing
        cv2.circle(mask, p, rad, 255, -1)
    else:  # contamination
        p = (int(rng.integers(size // 4, 3 * size // 4)),
             int(rng.integers(size // 4, 3 * size // 4)))
        ax = (max(2, size // 16), max(2, size // 26))
        rot = float(rng.uniform(0, 180))
        cv2.ellipse(out, p, ax, rot, 0, 360, (90, 60, 40), -1)
        cv2.ellipse(mask, p, ax, rot, 0, 360, 255, -1)
    return out, mask


def build(out_root: Path, size: int = 64, n_train: int = 24, n_good: int = 8,
          n_defect: int = 6, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    out_root = Path(out_root)

    for i in range(n_train):
        p = out_root / "train" / "good" / f"{i:03d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), _normal(rng, size))

    for i in range(n_good):
        p = out_root / "test" / "good" / f"{i:03d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), _normal(rng, size))

    for kind in ("chip", "contamination"):
        for i in range(n_defect):
            img, mask = _defect(rng, _normal(rng, size), kind)
            ip = out_root / "test" / kind / f"{i:03d}.png"
            mp = out_root / "ground_truth" / kind / f"{i:03d}_mask.png"
            ip.parent.mkdir(parents=True, exist_ok=True)
            mp.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ip), img)
            cv2.imwrite(str(mp), mask)

    return out_root


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="data/synthetic/bottle")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    root = build(Path(a.out), size=a.size, seed=a.seed)
    n = len(list(root.rglob("*.png")))
    print(f"wrote {n} images to {root}")
    print("NOTE: smoke-test fixture only -- not a benchmark, never report its numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
