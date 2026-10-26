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
| Autoencoder v3 (notebook run) | 0.937 | 0.894 | 0.512 | 0.715 |
| Autoencoder v3 (re-run post-refactor) | 0.936 | **0.908** | 0.551 | 0.720 |

A from-scratch autoencoder reached **clean pixel-AUROC 0.908** vs PatchCore's 0.986 — a legitimate, fully-understood gap, reached in two improvement steps. Every point of the gap is explainable.

---

## 6. Fix 2b — per-image score normalization: TESTED, REJECTED

**Hypothesis.** Lighting image-AUROC is stuck near random (0.512) while lighting *pixel*-AUROC is a
healthy 0.715. The image score is the top-k% mean of absolute anomaly values, and a global lighting
shift raises the *whole* map's error baseline by image-dependent amounts, so per-image scores stop
being comparable across lighting conditions. A score-normalization problem, not a representation flaw.

**Predicted direction.** Subtract the per-image median and divide by the per-image MAD before the
top-k image score. Lighting image-AUROC rises well clear of 0.5; clean image-AUROC roughly unchanged.

**Result (measured on Kaggle, same checkpoint, both variants scored in one pass over identical maps):**

| Scoring | Clean image | Clean pixel | Lighting image | Lighting pixel |
|---------|-------------|-------------|----------------|----------------|
| raw top-1% | 0.936 | 0.908 | 0.551 | 0.720 |
| median/MAD | 0.642 | 0.885 | **0.451** | 0.728 |

**Wrong in both directions.** Lighting image-AUROC did not recover, it dropped *below chance* (0.451),
meaning the ranking is now actively inverted. Clean image-AUROC lost 0.29. Pixel-AUROC barely moved,
which confirms the map was untouched and this is purely a scoring effect.

**Mechanism, two errors:**

1. **The baseline was not pure nuisance.** An image that reconstructs badly overall is genuinely more
   likely to be defective. Dividing that out discards real signal — which is why the *clean* condition,
   where lighting was never a confound, collapsed hardest.
2. **MAD is not defect-invariant.** The argument was that a defect covering a few percent of pixels
   barely moves median or MAD. The anomaly maps disprove it: the map fires across the whole rim
   annulus, so a defective image has a visibly wider spread and therefore a *larger* MAD. Dividing by
   it shrinks exactly the scores that should rank highest. That inversion is why lighting fell below 0.5.

**What the unit tests did and did not establish.** Two tests assert that the normalized score is exactly
invariant under a simulated affine shift of the anomaly map. They pass, and they were never wrong. The
error was in what they licensed: invariance to a transformation is not the same property as
discriminating defects, and real lighting is not a pure affine shift of the anomaly map. A test can pin
a mechanism correctly and still say nothing about whether that mechanism helps.

**Kept, not deleted.** `normalize_map` stays in `defectloc/anomaly.py` as a non-default option and the
numbers stay in the results table, so the failure is not retried by accident.

**Also confirmed by this run:** a fresh v3 training run reproduced the notebook within run-to-run spread
(clean 0.937/0.894 → 0.936/0.908, lighting 0.512/0.715 → 0.551/0.720), which is the evidence that the
module refactor preserved the numerics.

---

## 6b. Fix 2c — per-pixel calibration on held-out normals: TESTED, REJECTED (2026-10-26)

**Design correction before measuring.** Fix 2c as planned ("calibrate against held-out normals under
the same condition") meant one mean and std per condition. That is one affine map over every score in
the condition, and image-AUROC is invariant to it: it could not have moved the number. Rebuilt as
**per-pixel** calibration: hold out 30 of 209 train images (seeded, recorded in the checkpoint), retrain
on 179, take each pixel's mean and std over the held-out normal maps, score `z = (amap − mu) /
max(sigma, median sigma)` with the usual top-1%. Lighting test images are scored against held-out
normals stressed with lighting seed 1 (`pixel_z`) and against the clean statistics (`pixel_z_cleancal`).

**Predicted direction (written into the README first).** Clean image-AUROC holds and pixel-AUROC rises
from discounting the rim; lighting image-AUROC rises under `pixel_z`. Bar: +0.05 lighting image-AUROC,
≤ 0.02 clean image loss.

**Result (local CPU, one checkpoint, four variants in one pass):**

| Scoring | Clean image | Clean pixel | Lighting image | Lighting pixel |
|---------|-------------|-------------|----------------|----------------|
| raw top-1% | 0.948 | 0.902 | 0.573 | 0.728 |
| `pixel_z`, matched | 0.958 | 0.869 | **0.569** | 0.650 |
| `pixel_z_cleancal` | 0.958 | 0.869 | 0.511 | 0.597 |

**Mechanism.**

1. **The rim is where bottles break.** Clean pixel-AUROC per type: contamination 0.847 → 0.963,
   broken_large 0.927 → 0.800, broken_small 0.971 → 0.894. 35–41% of breakage pixels sit in the
   top-10% sigma zone against 3% of contamination pixels. Dividing out the rim's normal spread divides
   out breakage too. Part of what was called "rim over-firing" is the defect class that lives there.
2. **Lighting swamps the per-pixel structure.** The sigma floor over lighting-stressed normals is 0.260
   against 0.017 clean, so `pixel_z` under lighting is close to a smoothed raw map.

**New observation.** In `results/figures/fix2c_lighting_z.png` row 3 (directional shadow) the
reconstruction itself collapses. One frame, not a measurement, but it means part of the lighting
failure may be representation after all. The healthy lighting pixel-AUROC that supported "score, not
representation" is an average and can hide a subset of frames where the model fails outright.

**Also confirmed:** the Kaggle refactor checkpoint `ae_v3.pt` re-evaluated on the local copy of the data
gives 0.937/0.908 clean and 0.555/0.721 lighting, matching the recorded run.

---

## 6c. Experiment 3 — lighting-augmented training: ACCEPTED (2026-10-26)

**Locating the failure first.** `scripts/label_lighting_subsets.py` replays the seed-0 lighting build,
records which images received `DirectionalShadow`, and checks every replayed image is byte-identical
to the stored one (all 83 are; 32 shadowed). On the 179-image baseline, lighting image-AUROC is
**0.269** on shadowed images (inverted) and 0.731 on unshadowed. The headline 0.573 was an average of
two different failures.

**Hypothesis.** Representation, not scoring: the model has seen one lighting rig and cannot reconstruct
a bottle under another. **Design:** two arms, one change each from the 179-image baseline. A applies
the lighting pipeline (seed 2) to half the training samples, target = the lit image. B does the same
without the shadow. **Bar for A:** lighting image ≥ 0.673, clean image ≥ 0.928, clean pixel ≥ 0.882.

**Result (raw score, one seed per arm, local CPU):**

| | Clean image | Clean pixel | Lighting image | Lighting pixel | Shadowed | Unshadowed |
|---|---|---|---|---|---|---|
| baseline | 0.948 | 0.902 | 0.573 | 0.728 | 0.269 | 0.731 |
| A | 0.937 | 0.898 | **0.802** | 0.839 | **0.740** | 0.838 |
| B (no shadow) | 0.948 | 0.889 | 0.783 | 0.850 | 0.457 | 0.957 |

**Every prediction held.** The shadowed reconstruction that collapsed in the baseline is intact in A
and the map sits on the break (`results/figures/exp3_lighting_*.png`). The original diagnosis, "score
normalisation, not representation", was wrong; the healthy lighting pixel-AUROC that supported it was
an average over a subset where the model failed outright. B shows the robustness is coverage of the
specific perturbation: near-immune to what it trained on, still below chance on the unseen shadow.

**Caveats.** A trains on the test's transform family (different draws), so it shows learnability, not
robustness to unforeseen lighting. One seed per arm. 7 good images in the shadowed subset, so each one
moves that subset's image-AUROC by up to 0.14.

---

## 7. Remaining improvement ladder (planned)

1. **Robustness to unseen lighting** — B shows the model only handles perturbations it was trained on.
   Candidates: a broader photometric family, or flat-fielding the input against its own low-frequency
   illumination before the model sees it. Judge on a held-out perturbation, as B does. Re-run A on two
   more seeds first.
   *(Fix 2b and Fix 2c, both scoring fixes, were rejected; lighting-augmented training was accepted.
   See sections 6, 6b and 6c.)*
2. **Masking-based denoising** — replace additive noise with random patch-masking/inpainting; erases defects more completely, reduces residual false-firing. (Requires retraining.)
3. **Synthetic defects (DRAEM-style)** — paste fake defect-shaped corruptions on normal images for self-supervised, mask-supervised training. Most likely to push past ~0.95 and lift pixel-F1.
4. **Feature-space reconstruction** — reconstruct pretrained-backbone features instead of raw pixels; historically where reconstruction methods start *matching* PatchCore.

---

## Key lessons / methodology notes

- **Look at the image before the metric.** Every diagnosis here (identity shortcut, map-quality, score-normalization) came from *seeing* the reconstruction/map, then confirming with numbers.
- **Read the split between image-AUROC and pixel-AUROC.** image ≫ pixel = "knows whether, not where" (map quality). pixel healthy but image collapsed = scoring/normalization issue. AUROC vs F1 split = representation vs threshold.
- **A low training loss can be a warning** (identity shortcut), and a *higher* plateaued loss can be healthier.
- **Change one thing at a time, with a predicted direction.** That's what makes it research rather than flailing.
- **A correct diagnosis does not make the first fix correct.** The score-normalization diagnosis still
  looks right; the median/MAD implementation of it was wrong, because the divisor it chose was itself
  contaminated by the defect. Diagnosis and remedy need separate evidence.
- **Watch what a unit test actually claims.** The Fix 2b tests proved affine invariance and passed
  throughout. Invariance was never the property that mattered.
- **Check a fix against the metric's invariances before building it.** Fix 2c as first written was an
  affine map per condition, which AUROC cannot see. Five minutes of that reasoning saved a run.
- **An average can hide an inverted subset.** Lighting pixel-AUROC looked healthy for three versions
  and justified two scoring fixes. Splitting by the one perturbation that broke reconstruction showed
  an inverted ranking underneath. Split before diagnosing.
- **"Nuisance" is a claim about where defects are not.** The rim looked like nuisance in the maps, but
  it is where breakage happens. Break results down by defect type before calling a region noise.
