"""Record which lighting test images received the directional shadow.

The lighting stress set is built by one seeded pass (scripts/prepare_data.py),
so it can be replayed exactly. This script replays that pass, records every
call to `DirectionalShadow.apply` with its parameters, and checks that each
replayed image is byte-identical to the stored one. If any differs, the replay
is not the build that produced the test set and the labels would be fiction,
so it fails instead of writing.

    python scripts/label_lighting_subsets.py \
        --clean-root data/MVTecAD/bottle --lighting-root data/MVTecAD_lighting/bottle

Writes results/lighting_subsets.json, keyed by `<defect_type>/<file>`.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defectloc import augment  # noqa: E402


def replay(clean_cat: Path, light_cat: Path, seed: int = 0) -> dict:
    calls = []
    original = augment.DirectionalShadow.apply

    def recording_apply(self, img, **kwargs):
        # Same draws, same order as the original; only the recording is new.
        state = np.random.get_state()
        angle = np.random.uniform(0, 2 * np.pi)
        dark = np.random.uniform(self.min_factor, self.max_factor)
        np.random.set_state(state)
        calls.append({"angle": float(angle), "dark": float(dark)})
        return original(self, img, **kwargs)

    augment.DirectionalShadow.apply = recording_apply
    try:
        lighting = augment.build_factory_lighting(seed=seed)
        labels = {}
        # Same traversal as prepare_data.build_lighting.
        for split_dir in sorted((clean_cat / "test").iterdir()):
            if not split_dir.is_dir():
                continue
            for img_path in sorted(split_dir.glob("*.png")):
                n_before = len(calls)
                img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
                aug = cv2.cvtColor(lighting(image=img)["image"], cv2.COLOR_RGB2BGR)
                stored = cv2.imread(str(light_cat / "test" / split_dir.name / img_path.name))
                if stored is None or stored.shape != aug.shape or not np.array_equal(stored, aug):
                    raise SystemExit(
                        f"replay differs from the stored lighting image {split_dir.name}/"
                        f"{img_path.name}; labels would not describe this test set")
                shadow = len(calls) > n_before
                labels[f"{split_dir.name}/{img_path.name}"] = {
                    "shadow": shadow, **(calls[-1] if shadow else {})}
    finally:
        augment.DirectionalShadow.apply = original
    return labels


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clean-root", required=True)
    p.add_argument("--lighting-root", required=True)
    p.add_argument("--seed", type=int, default=0, help="seed the lighting set was built with")
    p.add_argument("--out", default="results/lighting_subsets.json")
    a = p.parse_args(argv)

    labels = replay(Path(a.clean_root), Path(a.lighting_root), a.seed)
    n = sum(v["shadow"] for v in labels.values())
    good = sum(v["shadow"] for k, v in labels.items() if k.startswith("good/"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": a.seed, "images": labels}, indent=1) + "\n",
                   encoding="utf-8")
    print(f"replay matches all {len(labels)} stored images; {n} shadowed "
          f"({good} good, {n - good} defective) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
