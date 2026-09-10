"""Fetch MVTec AD and build the clean + lighting-stressed roots.

Replaces three fragile notebook cells: a hardcoded ``DATA_ROOT`` that only
matched one Kaggle session, a copy step that failed silently when it did not,
and an unseeded augmentation loop that produced a different stress test on
every run.

Two ways to point it at the data:

    # already have MVTec AD unpacked somewhere
    python scripts/prepare_data.py --source /path/to/mvtec_anomaly_detection

    # or let kagglehub fetch it (needs ~/.kaggle/kaggle.json)
    python scripts/prepare_data.py --kagglehub

Output:

    data/MVTecAD/<category>/          clean copy   (train/, test/, ground_truth/)
    data/MVTecAD_lighting/<category>/ stress copy  (test/ photometrically shifted,
                                                    train/ and ground_truth/ identical)
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defectloc.augment import build_factory_lighting  # noqa: E402

KAGGLE_DATASET = "ipythonx/mvtec-ad"


def resolve_source(source: str | None, use_kagglehub: bool, category: str) -> Path:
    """Find the folder that contains ``<category>/``, or fail with a usable message."""
    if source:
        root = Path(source)
        if (root / category).is_dir():
            return root
        if root.name == category and (root / "train").is_dir():
            return root.parent
        raise SystemExit(
            f"--source {root} does not contain '{category}/'.\n"
            f"Point it at the folder holding the MVTec categories "
            f"(the one with bottle/, cable/, capsule/, ...)."
        )

    if use_kagglehub:
        try:
            import kagglehub
        except ImportError:
            raise SystemExit(
                "kagglehub is not installed.  pip install kagglehub\n"
                "Or download MVTec AD yourself and pass --source."
            )
        try:
            path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        except Exception as e:  # noqa: BLE001 - surface the real cause, whatever it is
            raise SystemExit(
                f"kagglehub could not download {KAGGLE_DATASET}: {e}\n"
                "This usually means no Kaggle credentials. Either place your token at\n"
                "  ~/.kaggle/kaggle.json   (Kaggle > Settings > Create New Token)\n"
                "or download MVTec AD from https://www.mvtec.com/company/research/datasets/mvtec-ad\n"
                "and pass --source /path/to/it."
            )
        print(f"kagglehub resolved the dataset to: {path}")
        if (path / category).is_dir():
            return path
        for sub in path.iterdir():        # some mirrors nest one level deeper
            if sub.is_dir() and (sub / category).is_dir():
                return sub
        raise SystemExit(f"downloaded dataset at {path} has no '{category}/' folder")

    raise SystemExit("pass either --source <path> or --kagglehub")


def copy_clean(src_cat: Path, dst_cat: Path) -> Path:
    if dst_cat.exists():
        print(f"clean copy already present: {dst_cat}")
        return dst_cat
    dst_cat.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_cat, dst_cat)
    print(f"clean copy: {dst_cat}")
    return dst_cat


def build_lighting(clean_cat: Path, out_cat: Path, seed: int = 0) -> Path:
    """Photometrically stress the TEST split only; masks and train copy unchanged."""
    lighting = build_factory_lighting(seed=seed)
    n = 0
    for split_dir in sorted((clean_cat / "test").iterdir()):
        if not split_dir.is_dir():
            continue
        for img_path in sorted(split_dir.glob("*.png")):
            img = cv2.imread(str(img_path))
            if img is None:
                raise SystemExit(f"could not read {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            aug = lighting(image=img)["image"]
            aug = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
            out_path = out_cat / "test" / split_dir.name / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), aug)
            n += 1

    # Photometric shifts do not move defects, so masks copy across unchanged.
    shutil.copytree(clean_cat / "ground_truth", out_cat / "ground_truth", dirs_exist_ok=True)
    # Train split copied clean: the point of the test is that training never
    # saw the new lighting.
    shutil.copytree(clean_cat / "train", out_cat / "train", dirs_exist_ok=True)
    print(f"lighting copy: {out_cat}  ({n} test images stressed, seed={seed})")
    return out_cat


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default=None, help="folder containing the MVTec categories")
    p.add_argument("--kagglehub", action="store_true", help="download via kagglehub instead")
    p.add_argument("--category", default="bottle")
    p.add_argument("--data-dir", default="data", help="where to write the prepared copies")
    p.add_argument("--seed", type=int, default=0, help="seed for the lighting stress test")
    a = p.parse_args(argv)

    src_root = resolve_source(a.source, a.kagglehub, a.category)
    data_dir = Path(a.data_dir)

    clean = copy_clean(src_root / a.category, data_dir / "MVTecAD" / a.category)
    light_cat = data_dir / "MVTecAD_lighting" / a.category
    if light_cat.exists():
        print(f"lighting copy already present: {light_cat}  (delete it to rebuild)")
    else:
        build_lighting(clean, light_cat, seed=a.seed)

    print("\nnext:")
    print(f"  python -m defectloc.train --data-root {clean}")
    print(f"  python -m defectloc.evaluate --checkpoint checkpoints/ae_v3.pt \\")
    print(f"      --clean-root {clean} --lighting-root {light_cat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
