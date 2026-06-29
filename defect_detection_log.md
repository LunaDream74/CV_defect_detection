# Defect Detection Prototype — Project Log

**Goal:** Build a from-scratch computer-vision model that *localizes* defects on bottles (top-down view), so that the spatial location of a defect can be traced back to the upstream manufacturing station responsible for it. Binary "good/bad" output is insufficient — the project needs a **spatial anomaly map**.

**Dataset:** MVTec AD, `bottle` category (Kaggle: `ipythonx/mvtec-ad`). Training uses only normal (`train/good`) images; defects are rare and unlabeled at train time, so the problem is framed as **unsupervised anomaly localization**, not supervised segmentation.

**Strategy:** Establish a strong off-the-shelf baseline (PatchCore via Anomalib), then build a self-authored model and improve it through deliberate, measured experiments — looking at the reconstruction/anomaly-map images *before* trusting any metric.

---

## 1. Baseline — PatchCore (via Anomalib)

PatchCore is a memory-bank method using a pretrained backbone; nothing is trained, so it serves as a baseline to beat rather than something to improve.

**Setup notes (Kaggle-specific):**
- Notebook needs **Internet enabled** (Settings → Internet) for `pip install anomalib`; requires phone verification.
- Kaggle `/kaggle/input` is read-only — dataset was copied into writable `/kaggle/working/MVTecAD/bottle`.
- The rich progress bar caused a `RecursionError` in the notebook; fixed with `Engine(enable_progress_bar=False)`.

**PatchCore results (bottle):**

| Condition | image_AUROC | pixel_AUROC |
|-----------|-------------|-------------|
| Clean     | 1.000       | 0.986       |
| Lighting  | 0.947       | 0.973       |

The clean run also reported image_F1 ≈ 0.992, pixel_F1 ≈ 0.727.

---

## 2. The lighting stress test

**Motivation:** MVTec bottles are shot under consistent studio lighting, one centered bottle, top-down. Real production lines have inconsistent lighting (we control the camera, but not lighting). We needed to know whether lighting is a real threat before designing around it.

**Method:** Built a photometric-only augmented copy of the **test set** (training left clean), applying factory-realistic lighting perturbations with Albumentations:
- Brightness/contrast drift (±25% / ±20%)
- Non-linear gamma (0.8–1.2)
- Mild color-temperature shift (hue ±8, small sat/val)
- A custom **directional shadow gradient** (overhead lamp / passing occlusion)

Ranges were tuned by eye so a human can still clearly see the bottle and defect — otherwise model failure would be the test's fault, not the model's. Ground-truth masks were copied unchanged (photometric augmentation doesn't move defect locations).

**Finding (on PatchCore):** pixel-AUROC dropped only 0.986 → 0.973 (~1.3 pts), but pixel-F1 dropped more (0.727 → 0.598). Small AUROC drop + larger F1 drop = a **threshold/calibration problem, not a representation problem**. Conclusion: lighting is a real but *manageable, mostly-recalibratable* effect — not a wall. This meant lighting robustness did not need to be the centerpiece of the self-built model.

---

## 3. The self-built prototype — Convolutional Autoencoder

**Why a reconstruction autoencoder (vs. alternatives):**
- vs. PatchCore: it's a genuinely *trainable, ownable* model — PatchCore has nothing to architect.
- vs. supervised U-Net segmentation: needs no defect masks at train time and generalizes to *unseen* defect types — matches the real "rare, unlabeled, unpredictable defects" scenario.
- vs. GAN/diffusion reconstruction: far simpler to train and debug — the correct *first* prototype.

**Principle:** Train the encoder–decoder only on normal bottles. It learns to rebuild "normal." A defective bottle should reconstruct as a *clean* bottle (the model never learned defects), so **input − reconstruction = the defect**. That difference, as a per-pixel map, is the localization.

**Loss:** Combined MSE + SSIM:
`loss = α·MSE + (1−α)·(1−SSIM)`, α=0.5.
Pure MSE produces blurry reconstructions that false-fire at every normal edge; the SSIM term compares local structure and is the single most important departure from a textbook autoencoder.

**Evaluation (made PatchCore-comparable):**
- Train on `train/good` only.
- Anomaly map per test image, then flatten all pixel scores + all GT-mask pixels across the test set → `roc_auc_score` = **pixel-AUROC**.
- Image score = mean of top-k% hottest pixels (more robust than pure max) → AUROC vs. image labels = **image-AUROC**.
- Run the same lighting-stress test set → robustness gap, directly comparable to PatchCore's 1.3-pt drop.

---

## 4. Iteration history (the important part)

Discipline throughout: **look at the reconstruction + anomaly-map image before reading any metric.** Each change was one hypothesis with a predicted direction.

### v1 — Base autoencoder (latent 256ch × 16×16, additive-free, MSE+SSIM)

| Condition | image_AUROC | pixel_AUROC |
|-----------|-------------|-------------|
| Clean     | 0.737       | 0.700       |
| Lighting  | 0.480       | 0.529       |

- Training loss fell very low (~0.043).
- **Lighting image-AUROC 0.48 < 0.5** — worse than random; signal of a mechanical problem.
- **Diagnosis via visualization:** defects were *faithfully reproduced* in the reconstruction → the **identity shortcut**. The bottleneck (256×16×16 = 65,536 values) was too large — barely a compression — so the model copied the input through, defects included. Low loss was the warning sign.

### Fix 1 — Break the identity shortcut

Two simultaneous changes:
1. **Tighter bottleneck:** dropped `latent_ch` 256 → 64 and added a 5th down/up-sampling stage. Bottleneck went from 256×16×16 (65,536 values) to **64×8×8 (4,096 values)** — 16× tighter. The model physically cannot memorize a defect through that small a code.
   - *(Gotcha corrected during implementation: the `4` in `Conv2d(i,o,4,...)` is the **kernel size**, not a stage count. A stage is added by inserting another `enc(...)`/`dec(...)` block, keeping kernel size 4 so each stage exactly halves/doubles resolution.)*
2. **Denoising objective:** corrupt the input with Gaussian noise, reconstruct the *clean* target:
   ```python
   noise = torch.randn_like(x) * 0.1
   recon = model(x + noise)
   loss  = recon_loss(recon, x)   # target = clean x
   ```
   Prevents copying; forces learning of normal structure.

**Result (v2):**

| Condition | image_AUROC | pixel_AUROC |
|-----------|-------------|-------------|
| Clean     | 0.867       | 0.741       |
| Lighting  | 0.560       | 0.562       |

- Loss plateaued *higher* (~0.062) — a **healthier** sign: the model can no longer cheat.
- **Clean image-AUROC jumped 0.737 → 0.867.** Lighting image-AUROC recovered above 0.5.
- **Visualization confirmed:** reconstructions now blurry/smoothed, defects partially erased — shortcut broken but not fully defeated.
- **Remaining problem:** pixel-AUROC barely moved (0.700 → 0.741). image-AUROC ≫ pixel-AUROC split = the model knows *whether* better than *where*. Anomaly maps fired on rims/reflections (normal high-frequency regions) and only covered defect *fragments*, not full shapes — a **map-quality** problem.

### Fix 2 — Local-SSIM anomaly map (no retraining)

Replaced the smoothed pixel-MSE anomaly map with a **per-pixel local-SSIM dissimilarity** map (windowed SSIM formulation, win=11). Pixel-MSE over-responds to small pixel differences at sharp normal edges; local SSIM compares *structure* in a window and is forgiving of blur-vs-sharp mismatch while still catching genuine structural breaks. This changes only how the map is read from the *existing* model — no retraining.

**Result (v3, current):**

| Condition | image_AUROC | pixel_AUROC |
|-----------|-------------|-------------|
| Clean     | 0.937       | 0.894       |
| Lighting  | 0.512       | 0.715       |

- **Clean pixel-AUROC jumped 0.741 → 0.894** (+15 pts) from a no-retrain change — the biggest single gain. Confirms the defect signal was always in the reconstruction; pixel-MSE was drowning it in edge noise.
- Clean image-AUROC rose to 0.937.
- **Visualization confirmed:** anomaly maps are now broad, connected, and *defect-shaped* — they resemble the ground-truth masks instead of scattered blobs.
- Lighting pixel-AUROC also rose (0.562 → 0.715).

---

## 5. Current standing vs. PatchCore

| Model | Clean image | Clean pixel | Lighting image | Lighting pixel |
|-------|-------------|-------------|----------------|----------------|
| PatchCore baseline | 1.000 | 0.986 | 0.947 | 0.973 |
| Autoencoder v3 (current) | 0.937 | **0.894** | 0.512 | 0.715 |

A from-scratch autoencoder reached **clean pixel-AUROC 0.894** vs PatchCore's 0.986 — a legitimate, fully-understood gap, reached in two improvement steps. Every point of the gap is explainable.

---

## 6. Open issue (to be addressed next — Fix 2b)

**Lighting image-AUROC is stuck at 0.512 (≈ random)** while lighting *pixel*-AUROC is a healthy 0.715. This split is diagnostic:
- The model still localizes defects under lighting (pixel survives).
- But it can't decide *whether* an image is defective across different lighting (image collapses).

**Mechanism:** the image score is the top-k% mean of absolute anomaly values. A global lighting shift raises the *whole* map's error baseline by image-dependent amounts, so per-image scores are no longer comparable across lighting conditions. This is a **score-normalization problem, not a representation flaw**.

**Planned next step (Fix 2b):** per-image score normalization (e.g. subtract median, divide by MAD) before computing the top-k image score, to cancel global lighting shifts. Cheap, no retraining, isolates the scoring variable.

---

## 7. Remaining improvement ladder (planned)

1. **Fix 2b** — per-image score normalization (recover lighting image-AUROC).
2. **Masking-based denoising** — replace additive noise with random patch-masking/inpainting; erases defects more completely, reduces residual false-firing. (Requires retraining.)
3. **Synthetic defects (DRAEM-style)** — paste fake defect-shaped corruptions on normal images for self-supervised, mask-supervised training. Most likely to push past ~0.95 and lift pixel-F1.
4. **Feature-space reconstruction** — reconstruct pretrained-backbone features instead of raw pixels; historically where reconstruction methods start *matching* PatchCore.

---

## Key lessons / methodology notes

- **Look at the image before the metric.** Every diagnosis here (identity shortcut, map-quality, score-normalization) came from *seeing* the reconstruction/map, then confirming with numbers.
- **Read the split between image-AUROC and pixel-AUROC.** image ≫ pixel = "knows whether, not where" (map quality). pixel healthy but image collapsed = scoring/normalization issue. AUROC vs F1 split = representation vs threshold.
- **A low training loss can be a warning** (identity shortcut), and a *higher* plateaued loss can be healthier.
- **Change one thing at a time, with a predicted direction.** That's what makes it research rather than flailing.
