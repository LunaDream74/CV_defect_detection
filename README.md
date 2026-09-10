# Bottle Defect Localization — Reconstruction-Based Anomaly Detection

A from-scratch convolutional autoencoder that **localizes** surface defects on bottles (top-down view), trained only on normal samples. The goal is not just a "good / bad" verdict but a **spatial anomaly map** showing *where* a defect is — so the location can be traced back to the upstream manufacturing station responsible for it.

The v3 prototype was a single Kaggle notebook. It is now a runnable Python package (`defectloc/`) with a seeded pipeline, a test suite, and a generated results table. The original notebook is kept as [`patchcore-base.ipynb`](patchcore-base.ipynb) for provenance.

---

## Results (MVTec AD, `bottle`)

Chance AUROC is **0.500** for both metrics. A photometric "factory-lighting" stress test (brightness/contrast/gamma/color drift + a directional shadow gradient) is applied to the **test set only**; training stays clean.

| Model | Clean image-AUROC | Clean pixel-AUROC | Lighting image-AUROC | Lighting pixel-AUROC |
|-------|:-----------------:|:-----------------:|:--------------------:|:--------------------:|
| PatchCore (baseline, via Anomalib) | 1.000 | 0.986 | 0.947 | 0.973 |
| Autoencoder v1 (large bottleneck) | 0.737 | 0.700 | 0.480 | 0.529 |
| Autoencoder v2 (tight bottleneck + denoising) | 0.867 | 0.741 | 0.560 | 0.562 |
| **Autoencoder v3 (this repo)** | 0.937 | **0.894** | 0.512 | 0.715 |
| Autoencoder v3 + median/MAD score | — | — | — | — |

The autoencoder reaches a clean pixel-AUROC of **0.894** from scratch, against PatchCore's 0.986. PatchCore is a baseline to measure the gap against, not part of this model.

Every number above and its provenance is in [`results/results.md`](results/results.md), generated from [`results/history.json`](results/history.json) by `scripts/make_results_table.py`. The v1–v3 rows come from the Kaggle runs recorded in [`defect_detection_log.md`](defect_detection_log.md); they have not been re-measured since the refactor, and are labelled as such in the table.

**The last row is empty on purpose.** The median/MAD score fix is implemented and unit-tested (see below), but it has not been run against MVTec AD, because the machine this refactor was done on has no copy of the dataset. An unmeasured row shows as `—` rather than being filled with a plausible guess. [Reproducing the table](#reproducing-the-table) is two commands once the data is present.

> **Known open issue (v3):** under the lighting stress test, *pixel*-AUROC stays healthy (0.715) but *image*-AUROC collapses to ~random (0.512). This is a score-normalization problem, not a representation flaw — a global lighting shift moves each image's overall error baseline by different amounts, so per-image scores stop being comparable across lighting conditions.

---

## Why this approach

The realistic industrial setting is: lots of *good* product images, very few *defective* ones, and defects that are rare, unlabeled, and unpredictable. That rules out a supervised detector (which needs labeled defects and only finds defect types it has seen) and motivates **unsupervised anomaly detection**.

The method here is **reconstruction-based**:

1. Train a convolutional autoencoder **only on normal bottles**, so it learns what "normal" looks like.
2. At inference, the model rebuilds the input. Because it only ever learned normal, it reconstructs defective regions poorly.
3. Compare input vs. reconstruction using **local SSIM (structural similarity)**, not raw pixels. Wherever they structurally differ is flagged as a defect.

Because it needs no defect labels, it generalizes to **unseen** defect types and transfers to new product lines by retraining on that line's good images alone.

### How this differs from pixel comparison (golden-template differencing)
Pixel comparison subtracts the test image from one fixed reference good image and flags pixel differences. It is fast and very sensitive but **brittle** — it breaks under any misalignment, natural variation, or lighting shift. This method compares against a reconstruction *generated from a learned model of normal*, so it tolerates the real-world variation pixel comparison cannot. Trade-off: pixel comparison wins on raw sensitivity to tiny defects in a perfectly controlled rig; this method wins on robustness. They fail in opposite situations.

---

## How it was built (iteration summary)

Each change was one hypothesis with a predicted direction, and the reconstruction/anomaly-map images were inspected *before* trusting any metric.

| Version | Change | Clean pixel-AUROC |
|---------|--------|:-----------------:|
| v1 | Base autoencoder, large bottleneck (256×16×16) | 0.700 |
| v2 | **Fix 1:** tighter bottleneck (64×8×8) + denoising objective | 0.741 |
| v3 | **Fix 2:** local-SSIM anomaly map (no retraining) | **0.894** |

- **v1 → v2:** The large bottleneck let the model copy the input through verbatim (the *identity shortcut*) — defects survived into the reconstruction, so they didn't stand out. Shrinking the latent code 16× (from 65,536 to 4,096 values) and adding a denoising objective (reconstruct the clean image from a noised input) forced the model to learn normal *structure* instead of copying.
- **v2 → v3:** Pixel-MSE anomaly maps over-fired on normal high-frequency regions (rims, reflections). Switching to a windowed **local-SSIM dissimilarity** map read structure instead of raw pixel values and produced defect-shaped maps — a +15-point jump in pixel-AUROC from a no-retrain change.

See [`defect_detection_log.md`](defect_detection_log.md) for the full project log, including the design reasoning and diagnostic methodology.

---

## The scoring fix (Fix 2b)

The lighting failure separates cleanly into two questions the code had been answering with one number:

- the **map** says *where* the image is anomalous, and it survived the stress test (pixel-AUROC 0.715);
- the **score** reduces that map to one number saying *whether* the image is defective, and it did not (image-AUROC 0.512).

A dim frame reconstructs worse everywhere, so *every* pixel of its anomaly map sits higher. The raw top-1% score therefore measures the lighting as much as the defect, and images shot under different lighting stop being comparable to each other.

`normalize_map` in [`defectloc/anomaly.py`](defectloc/anomaly.py) divides that baseline out: `(amap − median) / MAD`, per image. Median and MAD describe that image's own normal background, and because the defect covers a few percent of the pixels, it barely moves either. Mean and standard deviation would not work here for exactly that reason — the defect is an outlier, so it would inflate the very quantity it is being measured against.

The normalization is monotonic within an image, so it cannot change *which* pixels are hottest, only the units the score is reported in. That is the whole point: it makes scores comparable **across** images.

Two tests in [`tests/test_smoke.py`](tests/test_smoke.py) pin this down. Under a simulated affine lighting shift the normalized score is exactly invariant while the raw score moves by more than half, and a lighting-shifted *good* map outranks a clean *defective* one under raw scoring but not under normalized scoring. That is the failure mechanism reproduced in an assertion.

What those tests do **not** show is how much image-AUROC comes back on real bottles. Only a run on MVTec AD answers that.

---

## Architecture

A symmetric convolutional autoencoder, 5 down/up-sampling stages, bottleneck of **64 channels × 8 × 8**:

```
Input 3×256×256
  Encoder: 3→32→64→128→256→64   (each stage halves resolution: 256→128→64→32→16→8)
  Bottleneck: 64×8×8  (4,096 values — deliberately tight to prevent copying)
  Decoder: 64→256→128→64→32→3   (each stage doubles resolution: 8→16→32→64→128→256)
Output 3×256×256 (reconstruction)
```

- **Loss:** `0.5·MSE + 0.5·(1 − SSIM)` — the SSIM term tolerates the blur a tight bottleneck produces while still penalizing structural defects.
- **Training objective:** denoising — input is corrupted with Gaussian noise (σ=0.1), target is the clean image.
- **Anomaly map:** per-pixel local-SSIM dissimilarity (window 11), computed on the grayscale-averaged channels.
- **Image score:** mean of the top-1% hottest pixels (more robust than pure max), optionally median/MAD-normalized.

---

## Repository layout

```
defectloc/            the pipeline
  config.py           every value that defines a run, logged into each result
  model.py            the autoencoder
  losses.py           0.5·MSE + 0.5·(1 − SSIM)
  data.py             MVTec-layout datasets
  augment.py          the factory-lighting stress test (seeded)
  anomaly.py          local-SSIM map, top-k score, median/MAD normalization
  train.py            training + checkpoint I/O
  evaluate.py         image/pixel AUROC, both scoring variants, per-defect-type
  visualize.py        input / reconstruction / map / ground-truth figures
scripts/
  prepare_data.py         fetch MVTec AD, build clean + lighting roots
  make_results_table.py   render results/results.md
  patchcore_baseline.py   re-run the PatchCore reference row (optional, heavy)
  make_synthetic_fixture.py  tiny procedural dataset for smoke tests
tests/test_smoke.py   14 plumbing and invariant tests, no GPU or dataset needed
results/              history.json (recorded, with provenance) + generated results.md
```

---

## How to run

```bash
pip install -r requirements.txt
```

Get the data. Either point at a copy you already have, or let `kagglehub` fetch it (needs a Kaggle token at `~/.kaggle/kaggle.json`):

```bash
python scripts/prepare_data.py --source /path/to/mvtec_anomaly_detection
# or
python scripts/prepare_data.py --kagglehub
```

That writes `data/MVTecAD/bottle` (clean) and `data/MVTecAD_lighting/bottle` (stress-tested test split, masks and train split unchanged). The lighting set is seeded, so it is the same set of images on every run.

Train, then evaluate:

```bash
python -m defectloc.train --data-root data/MVTecAD/bottle --epochs 80

python -m defectloc.evaluate \
    --checkpoint checkpoints/ae_v3.pt \
    --clean-root data/MVTecAD/bottle \
    --lighting-root data/MVTecAD_lighting/bottle \
    --out results/runs/ae_v3.json
```

`evaluate` scores both ways in a single pass over the test set, so the raw and median/MAD numbers are always measured on identical anomaly maps:

```
clean     [none      ]  image-AUROC 0.xxxx   pixel-AUROC 0.xxxx
clean     [median_mad]  image-AUROC 0.xxxx   pixel-AUROC 0.xxxx
lighting  [none      ]  image-AUROC 0.xxxx   pixel-AUROC 0.xxxx
lighting  [median_mad]  image-AUROC 0.xxxx   pixel-AUROC 0.xxxx
```

Look at the maps before trusting any of those numbers:

```bash
python -m defectloc.visualize --checkpoint checkpoints/ae_v3.pt \
    --data-root data/MVTecAD/bottle --out results/figures/v3_clean.png
```

Training runs on GPU if one is visible and falls back to CPU otherwise (`--device` forces either). 80 epochs over 209 images takes a few minutes on a GPU.

### Reproducing the table

```bash
python -m defectloc.evaluate --checkpoint checkpoints/ae_v3.pt \
    --clean-root data/MVTecAD/bottle \
    --lighting-root data/MVTecAD_lighting/bottle \
    --out results/runs/ae_v3.json

python scripts/make_results_table.py
```

The second command regenerates `results/results.md`, merging the recorded history with every run JSON in `results/runs/`. Each run contributes two rows, one per scoring variant. To promote a measured number into the headline table above, copy it into `results/history.json` with its provenance.

### Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
```

14 tests, about 10 seconds, no GPU and no dataset. They build a small procedural fixture and run the pipeline end to end. They assert **no accuracy threshold**, because a number measured on procedural rings would mean nothing; accuracy belongs in the results table, measured on MVTec AD.

### PatchCore baseline (optional)

```bash
pip install -r requirements-baseline.txt
python scripts/patchcore_baseline.py --data-root data/MVTecAD --category bottle
```

`anomalib` is a heavy dependency and is not needed for the autoencoder. This script was written against the Anomalib API but has not been executed since the refactor, so treat a first run as debugging rather than as a measurement. The baseline numbers in the table came from the original Kaggle run.

---

## Notes on reproducibility

- **Everything is seeded**, including the lighting stress set. In the notebook it was not, which is why the v2 row in the log (0.867 / 0.741 clean) and a later re-run of the same configuration (0.874 / 0.743) disagree slightly. Run-to-run spread of roughly a point on this dataset is normal, and is worth remembering before reading any single-point comparison too closely.
- **The config is stored inside each checkpoint**, so `evaluate` and `visualize` rebuild the exact model that was trained without repeating any flags.
- **Small test set.** MVTec `bottle` has 83 test images, 20 of them good. AUROC on that few images moves noticeably from a couple of rank swaps, so the differences between neighbouring rows are indicative rather than significant.
- The notebook's hardcoded `DATA_ROOT`, its duplicate cells, and its unseeded augmentation loop are all resolved by `scripts/prepare_data.py`. The Anomalib `RecursionError` workaround (`Engine(enable_progress_bar=False)`) is applied in `scripts/patchcore_baseline.py`.

---

## Roadmap

- [x] **Refactor** the notebook into Python modules.
- [~] **Fix 2b — per-image score normalization** (subtract median / divide by MAD before the top-k image score) to recover lighting image-AUROC. Implemented and unit-tested; not yet measured on MVTec AD.
- [ ] **Masking-based denoising** — replace additive noise with random patch masking/inpainting to erase defects more completely and reduce false-firing.
- [ ] **Synthetic defects (DRAEM-style)** — paste defect-shaped corruptions on normal images for self-supervised, mask-supervised training; expected to push past ~0.95 and improve pixel-F1.
- [ ] **Feature-space reconstruction** — reconstruct pretrained-backbone features instead of raw pixels; historically where reconstruction methods start matching memory-bank methods like PatchCore.
- [ ] **Two-stage pipeline** — a detector (e.g. YOLO) to crop individual bottles from multi-bottle / conveyor frames, feeding each crop to this per-bottle model.
- [ ] **Pixel-F1 alongside AUROC.** PatchCore's clean pixel-AUROC 0.986 came with pixel-F1 0.727, so AUROC alone flatters both models. `evaluate.py` reports AUROC only.

---

## Scope and limitations

- Trained and evaluated on a **single view** (top-down `bottle`). The *method* transfers to other views (side-view label/seal inspection, overhead multi-bottle) but the *trained weights* do not — each view needs its own model trained on that view's good images. Detail-heavy views (labels with text) would likely need a larger bottleneck.
- Reconstruction-based detection is strongest on **textural/appearance** defects (chips, contamination, scratches) and weaker on **logical** defects (wrong label entirely, right part in wrong place), which may need a complementary method.
- Results are on MVTec AD, which is well-aligned and well-lit; real-line performance depends on capture conditions.
- The lighting stress test is **photometric only**. It does not cover blur, defocus, misalignment, or camera pose change, and it is synthetic rather than a second real capture session. It is a lower bound on the difficulty of a real lighting change.
- Every autoencoder row in the results table comes from one training run at one seed. There are no error bars.
