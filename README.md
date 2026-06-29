# Bottle Defect Localization — Reconstruction-Based Anomaly Detection

A from-scratch convolutional autoencoder that **localizes** surface defects on bottles (top-down view), trained only on normal samples. The goal is not just a "good / bad" verdict but a **spatial anomaly map** showing *where* a defect is — so the location can be traced back to the upstream manufacturing station responsible for it.

This repo is the **v3 prototype** (single Kaggle notebook). It will be refactored into Python modules later.

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

## Results (MVTec AD, `bottle`)

A photometric "factory-lighting" stress test (brightness/contrast/gamma/color drift + a directional shadow gradient) is applied to the **test set only** to measure robustness; training stays clean.

| Model | Clean image-AUROC | Clean pixel-AUROC | Lighting image-AUROC | Lighting pixel-AUROC |
|-------|:-----------------:|:-----------------:|:--------------------:|:--------------------:|
| PatchCore (baseline, via Anomalib) | 1.000 | 0.986 | 0.947 | 0.973 |
| **Autoencoder v3 (this repo)** | 0.937 | **0.894** | 0.512 | 0.715 |

The autoencoder reaches a clean pixel-AUROC of **0.894** from scratch. PatchCore is included as a baseline to measure against, not as part of this model.

> **Known open issue (v3):** under the lighting stress test, *pixel*-AUROC stays healthy (0.715) but *image*-AUROC collapses to ~random (0.512). This is a score-normalization problem, not a representation flaw — a global lighting shift moves each image's overall error baseline by different amounts, so per-image scores stop being comparable across lighting conditions. The planned fix (per-image score normalization) is in the roadmap below.

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
- **Image score:** mean of the top-1% hottest pixels (more robust than pure max).

---

## How to run

This is a Kaggle notebook (`patchcore-base.ipynb`). It expects a **GPU** accelerator and **Internet enabled** (Settings → Internet; requires phone verification) for the `pip install` and dataset download steps.

1. Open `patchcore-base.ipynb` in Kaggle (or locally with the dataset present).
2. Enable GPU and Internet in the notebook settings.
3. Run the cells top to bottom. The notebook will:
   - download MVTec AD via `kagglehub` (`ipythonx/mvtec-ad`),
   - copy the `bottle` category into the writable working dir,
   - build the lighting-stress test set,
   - train the autoencoder (~80 epochs, a few minutes on GPU),
   - evaluate clean + lighting and print image/pixel-AUROC,
   - visualize input / reconstruction / anomaly map / ground truth.

### Dependencies
Beyond the Kaggle base image, the notebook installs:
```
anomalib          # PatchCore baseline
pytorch-msssim    # SSIM loss term
```
and uses `torch`, `albumentations`, `opencv-python`, `scikit-learn`, `numpy`, `matplotlib` (preinstalled on Kaggle). `anomalib` is only needed if you also run the PatchCore baseline; the autoencoder itself depends only on `torch`, `pytorch-msssim`, `opencv`, `albumentations`, and `scikit-learn`.

### Notes / gotchas if running outside the original session
- The dataset path is resolved by `kagglehub.dataset_download("ipythonx/mvtec-ad")`, which prints the real path. One early cell still hardcodes `DATA_ROOT = "/kaggle/input/datasets/ipythonx/mvtec-ad"` — if the copy step errors, set `DATA_ROOT` to the path `kagglehub` printed.
- There are a couple of duplicate cells (the augmentation definition and the visualization block appear twice); they're harmless but will be cleaned up in the module refactor.
- Anomalib's rich progress bar can throw a `RecursionError` in the Kaggle notebook; if you run the PatchCore baseline, construct the engine with `Engine(enable_progress_bar=False)`.

---

## Roadmap

- [ ] **Fix 2b — per-image score normalization** (subtract median / divide by MAD before the top-k image score) to recover lighting image-AUROC. Cheap, no retraining.
- [ ] **Masking-based denoising** — replace additive noise with random patch masking/inpainting to erase defects more completely and reduce false-firing.
- [ ] **Synthetic defects (DRAEM-style)** — paste defect-shaped corruptions on normal images for self-supervised, mask-supervised training; expected to push past ~0.95 and improve pixel-F1.
- [ ] **Feature-space reconstruction** — reconstruct pretrained-backbone features instead of raw pixels; historically where reconstruction methods start matching memory-bank methods like PatchCore.
- [ ] **Two-stage pipeline** — a detector (e.g. YOLO) to crop individual bottles from multi-bottle / conveyor frames, feeding each crop to this per-bottle model.
- [ ] **Refactor** the notebook into Python modules (`model.py`, `data.py`, `train.py`, `evaluate.py`, `augment.py`).

---

## Scope and limitations

- Trained and evaluated on a **single view** (top-down `bottle`). The *method* transfers to other views (side-view label/seal inspection, overhead multi-bottle) but the *trained weights* do not — each view needs its own model trained on that view's good images. Detail-heavy views (labels with text) would likely need a larger bottleneck.
- Reconstruction-based detection is strongest on **textural/appearance** defects (chips, contamination, scratches) and weaker on **logical** defects (wrong label entirely, right part in wrong place), which may need a complementary method.
- Results are on MVTec AD, which is well-aligned and well-lit; real-line performance depends on capture conditions.
