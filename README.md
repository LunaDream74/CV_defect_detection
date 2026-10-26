# Bottle Defect Localization — Reconstruction-Based Anomaly Detection

A from-scratch convolutional autoencoder that **localizes** surface defects on bottles (top-down view), trained only on normal samples. The goal is not just a "good / bad" verdict but a **spatial anomaly map** showing *where* a defect is — so the location can be traced back to the upstream manufacturing station responsible for it.

The v3 prototype was a single Kaggle notebook. It is now a runnable Python package (`defectloc/`) with a seeded pipeline, a test suite, and a generated results table. The original notebook is kept as [`patchcore-base.ipynb`](patchcore-base.ipynb) for provenance.

---

## Results (MVTec AD, `bottle`)

Chance AUROC is **0.500** for both metrics. A photometric "factory-lighting" stress test (brightness/contrast/gamma/color drift + a directional shadow gradient) is applied to the **test set** (seed 0). Every model trains on clean images except the lighting-augmented rows, which train on a different draw of the same pipeline (seed 2).

| Model | Clean image-AUROC | Clean pixel-AUROC | Lighting image-AUROC | Lighting pixel-AUROC |
|-------|:-----------------:|:-----------------:|:--------------------:|:--------------------:|
| PatchCore (baseline, via Anomalib) | 1.000 | 0.986 | 0.947 | 0.973 |
| Autoencoder v1 (large bottleneck) | 0.737 | 0.700 | 0.480 | 0.529 |
| Autoencoder v2 (tight bottleneck + denoising) | 0.867 | 0.741 | 0.560 | 0.562 |
| Autoencoder v3 (original notebook run) | 0.937 | 0.894 | 0.512 | 0.715 |
| **Autoencoder v3 (re-run after refactor)** | 0.936 | **0.908** | 0.551 | 0.720 |
| Autoencoder v3 + median/MAD score (rejected) | 0.642 | 0.885 | 0.451 | 0.728 |
| Autoencoder v3, 179-image train (30 held out) | 0.948 | 0.902 | 0.573 | 0.728 |
| ↳ + per-pixel z, matched calibration (rejected) | 0.958 | 0.869 | 0.569 | 0.650 |
| ↳ + per-pixel z, clean calibration (rejected) | 0.958 | 0.869 | 0.511 | 0.597 |
| **Autoencoder v3 + lighting-augmented training** | 0.937 | 0.898 | **0.802** | **0.839** |
| ↳ same, shadow held out of training | 0.948 | 0.889 | 0.783 | 0.850 |

The autoencoder reaches a clean pixel-AUROC of **0.908** from scratch, against PatchCore's 0.986. PatchCore is a baseline to measure the gap against, not part of this model.

The re-run reproduces the notebook within run-to-run spread (clean 0.937/0.894 → 0.936/0.908, lighting 0.512/0.715 → 0.551/0.720). That agreement is the evidence that the refactor preserved the numerics, and it is why the two v3 rows are both kept rather than one overwriting the other.

Every number and its provenance is in [`results/results.md`](results/results.md), generated from [`results/history.json`](results/history.json) by `scripts/make_results_table.py`.

> **The lighting failure was representation, and training under varied lighting closes most of it.** Under the stress test, image-AUROC sat near chance (0.551) while pixel-AUROC looked healthy (0.720), which was read as a scoring problem. Two scoring fixes built on that reading failed ([Fix 2b](#fix-2b-what-happened), [Fix 2c](#fix-2c-what-happened)). Splitting the lighting set by the replayed shadow labels showed the ranking was *inverted* on shadowed frames, where the reconstruction collapses. Lighting-augmented training lifts lighting image-AUROC from 0.573 to 0.802 at a cost of 0.011 clean ([Experiment 3](#experiment-3-what-happened)). The gain holds for the perturbations the model trained on; a shadow it never saw is still not handled.

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

## Fix 2b: what happened

The lighting failure separates into two questions the code had been answering with one number:

- the **map** says *where* the image is anomalous, and it survived the stress test (pixel-AUROC 0.720);
- the **score** reduces that map to one number saying *whether* the image is defective, and it did not (image-AUROC 0.551).

The hypothesis was that a dim frame reconstructs worse everywhere, so every pixel of its anomaly map sits higher, and the raw top-1% score therefore measures the lighting as much as the defect. Predicted direction: dividing each image's own baseline out should recover lighting image-AUROC at little cost on clean.

`normalize_map` in [`defectloc/anomaly.py`](defectloc/anomaly.py) implements that as `(amap − median) / MAD`, per image.

**It failed, in both conditions.**

| | Clean image-AUROC | Lighting image-AUROC |
|---|:---:|:---:|
| raw top-1% | 0.936 | 0.551 |
| median/MAD | 0.642 | **0.451** |

Lighting did not recover; it fell below chance, which means the ranking is now actively inverted. Clean lost 0.29 outright. Both variants were scored in the same pass over identical anomaly maps from one checkpoint, so nothing about training or the map explains the difference.

Two mistakes in the reasoning, and the second is the interesting one.

**The baseline was not pure nuisance.** An image whose reconstruction is bad overall is genuinely more likely to be defective, so the very quantity being divided out carried real signal. That alone explains the clean-condition collapse, where lighting was never a confound to begin with.

**MAD is not defect-invariant.** The argument for median and MAD was that a defect covering a few percent of the pixels barely moves either. The anomaly maps in [`results/figures/v3_clean.png`](results/figures/v3_clean.png) show that is wrong: the map fires across the whole rim annulus, not just the defect, so a defective image has a visibly wider spread of map values. Its MAD is therefore *larger*, and dividing by it shrinks exactly the scores that should be highest. The normalization penalizes defective images, which is why the lighting number lands below 0.5 rather than merely failing to improve.

Pixel-AUROC barely moved (clean 0.908 → 0.885, lighting 0.720 → 0.728), confirming this is purely a scoring effect and the map is untouched.

### What the tests did and did not establish

Two tests in [`tests/test_smoke.py`](tests/test_smoke.py) still pass, and they were never wrong. Under a simulated affine shift of the anomaly map, the normalized score is exactly invariant while the raw score moves by more than half.

The error was in what that licensed. Invariance to a transformation is not the same property as discriminating defects, and real lighting is not a pure affine shift of the anomaly map. A unit test that pins a mechanism can be entirely correct and still say nothing about whether the mechanism helps. The tests are kept, with their scope stated, because the invariance claim remains true and worth protecting.

### Where to go next

Ordered by how much of the above they address, and stated as hypotheses rather than plans:

1. **Calibrate against normal images, not against the image itself.** Estimate the score distribution on held-out *good* images under the same condition and standardize against that. The divisor then cannot be inflated by the defect it is meant to measure, which is the failure above.
2. **Normalize with an input-side statistic** such as the frame's mean brightness, rather than a statistic of the error map. Same reasoning: keep the defect out of the denominator.
3. **Tighten the map first.** The rim over-firing visible in the figures is what makes MAD defect-sensitive. A map that fired only on defects would make every scoring rule easier, and it is the more fundamental problem.

## Fix 2c: per-pixel calibration on held-out normals

*Design and prediction written 2026-10-26, before the measurement.*

Idea 1 above, as first written, could not have worked. One mean and one standard deviation per condition is a single affine map applied to every score in that condition, and image-AUROC ignores affine maps: the number would come back unchanged. `test_one_mean_and_std_per_condition_cannot_change_the_ranking` pins that.

So the calibration is per pixel. Withhold 30 of the 209 training images (`--calib-holdout 30`, a seeded split recorded in the checkpoint), train on the other 179, and compute the mean `mu(p)` and spread `sigma(p)` of each pixel's anomaly value over the held-out normals. A test map is scored as `z(p) = (amap(p) − mu(p)) / max(sigma(p), floor)`, with `floor` fixed at the median of `sigma` over pixels, then the usual top-1% mean. [`defectloc/calibrate.py`](defectloc/calibrate.py) implements it.

This addresses both mistakes in Fix 2b. The reference comes from other images, so a defect cannot inflate its own divisor. And because the rim fires on normal bottles too, its high `mu` and `sigma` are divided out, which is idea 3 attacked through the score instead of the map.

The held-out images must be unseen. A training image reconstructs better than a new normal one, so calibrating on `ae_v3.pt`'s own training set would understate how hot a normal map runs. That is why this needs a retrained checkpoint, and why its raw score is re-measured on that checkpoint instead of compared with the 0.936 / 0.551 above.

Two lighting variants are scored in one pass:

- **`pixel_z`**: lighting test images against held-out normals stressed by the same lighting *distribution* (seed 1, four copies each; the test set uses seed 0). This assumes the deployment lighting range is known.
- **`pixel_z_cleancal`**: lighting test images against the clean statistics. This assumes nothing about the lighting and is the stricter test.

**Prediction.** On clean images, image-AUROC at least holds and pixel-AUROC rises, from discounting the rim. Under lighting, `pixel_z` raises image-AUROC over the raw score; `pixel_z_cleancal` gains less or nothing. The method removes *where normals fire*, not *how bright this frame is*, so it does not directly cancel a per-image brightness offset. That is the part of the diagnosis it leaves alone, and the reason a small lighting gain would not be a surprise.

**Accept criterion, fixed in advance.** On the same checkpoint, `pixel_z` must raise lighting image-AUROC by at least 0.05 over the raw score while losing no more than 0.02 clean image-AUROC. Anything less is recorded as a rejection, as Fix 2b was.

### Fix 2c: what happened

**Rejected.** Same checkpoint, same anomaly maps, scored four ways in one pass ([`results/runs/ae_v3_holdout30.json`](results/runs/ae_v3_holdout30.json)):

| Score | Clean image | Clean pixel | Lighting image | Lighting pixel |
|---|:---:|:---:|:---:|:---:|
| raw top-1% | 0.948 | 0.902 | 0.573 | 0.728 |
| `pixel_z`, matched calibration | 0.958 | 0.869 | **0.569** | 0.650 |
| `pixel_z_cleancal`, clean calibration | 0.958 | 0.869 | 0.511 | 0.597 |

Lighting image-AUROC moved by −0.004 against a bar of +0.05. The clean-image condition held (+0.010), so the rejection is on the lighting gain alone. Two of the three predictions were wrong: clean pixel-AUROC fell instead of rising, and the calibration did nothing for lighting.

**Discounting the rim cuts both ways, because bottles break at the rim.** The per-defect-type breakdown on clean images splits cleanly:

| Defect type | Pixel-AUROC raw → z | Image-AUROC raw → z | Defect pixels in the top-10% σ zone |
|---|:---:|:---:|:---:|
| contamination | 0.847 → **0.963** | 0.914 → **0.976** | 3% |
| broken_large | 0.927 → 0.800 | 0.995 → 0.970 | 35% |
| broken_small | 0.971 → 0.894 | 0.936 → 0.930 | 41% |

Contamination sits in the interior, away from the high-variance ring, and per-pixel z does exactly what it was designed to do there. Breakage sits *on* the ring, so dividing out the ring's normal spread divides out the defect too ([`results/figures/fix2c_clean_raw.png`](results/figures/fix2c_clean_raw.png) against [`fix2c_clean_z.png`](results/figures/fix2c_clean_z.png): the z map keeps a thin arc of the break and moves the hot spots off it). Some of what earlier figures read as "rim over-firing" was the defect class that lives at the rim. The README's claim that the rim is pure nuisance was too strong.

**Under lighting, the calibration has nothing to work with.** The per-pixel spread over lighting-stressed normals is about 16 times the clean spread (σ floor 0.260 against 0.017). The lighting variance swamps the per-pixel structure the method relies on, so z is close to a smoothed copy of the raw map. The clean-calibrated variant is worse than raw, as expected: it removes neither the rim nor the frame's brightness.

**A new observation that cuts against the original diagnosis.** In [`results/figures/fix2c_lighting_z.png`](results/figures/fix2c_lighting_z.png), row 3 carries the directional shadow, and its *reconstruction* collapses: the bottle comes back as a smeared blur and the whole outer edge fires. No scoring rule can fix that image, because the map itself is wrong. This is one image, not a measurement, but it suggests part of the lighting failure is in the representation, not only the score. The project's working diagnosis ("score normalisation, not representation") had held up until now on the evidence of a healthy lighting pixel-AUROC. That evidence is an average, and an average can hide a subset of frames where the model fails outright.

What is worth keeping: the held-out split, the calibration code and the `--zmap` figure option, all of which the next experiments can reuse; and the contamination result, which says per-pixel normal statistics are a real gain for interior defects.

## Experiment 3: lighting-augmented training

*Design and prediction written 2026-10-26, before training.*

**Where the lighting failure lives.** The lighting set is one seeded pass, so [`scripts/label_lighting_subsets.py`](scripts/label_lighting_subsets.py) replays it, records which images received the directional shadow, and checks that every replayed image is byte-identical to the stored one (all 83 are). On the 179-image baseline checkpoint, raw score:

| Lighting subset | n (good / defective) | Image-AUROC | Pixel-AUROC |
|---|:---:|:---:|:---:|
| shadowed | 32 (7 / 25) | **0.269** | 0.664 |
| not shadowed | 51 (13 / 38) | 0.731 | 0.765 |
| all | 83 (20 / 63) | 0.573 | 0.728 |

The headline 0.573 averages two different failures. On shadowed frames the ranking is inverted, which fits the collapsed reconstruction seen in Fix 2c. On unshadowed frames it is degraded (0.731 against 0.948 clean) but well above chance. The 7 shadowed good images make the first number noisy; it is a direction, not a precise value.

**Hypothesis.** The model has only seen one lighting rig, so it cannot reconstruct a bottle under another one, and the map fires on the lighting. If that is right, it is a representation problem, and training under varied lighting fixes it where scoring could not.

**Design.** Two arms, each one change from the 179-image baseline (same split, seed and 80 epochs):

- **A, full augmentation.** With probability 0.5 per sample per epoch, apply the factory-lighting pipeline before the denoising noise. The target is the *augmented* image, so the model learns to reproduce a bottle under that lighting, not to undo the lighting (undoing it would make the map fire on the lighting itself). Pipeline seed 2; the test set uses 0 and calibration 1.
- **B, shadow held out.** Same as A, without `DirectionalShadow`. The shadowed test subset then measures whether robustness extends to a perturbation the model never saw.

Arm A trains on the same family of transforms the test uses, with different draws. Its result therefore says whether the failure is learnable, not that the model will survive unforeseen lighting. Arm B is the closer test of that.

**Prediction.** A raises lighting image-AUROC substantially, most of all on the shadowed subset, which should end above 0.5. Clean holds within noise, since half the samples are still unaugmented. B recovers the unshadowed subset roughly as A does but recovers much less of the shadowed one.

**Accept criterion for A, fixed in advance.** Raw-score lighting image-AUROC of at least 0.673 (+0.10 over 0.573), with clean image-AUROC at least 0.928 and clean pixel-AUROC at least 0.882 (each at most −0.02). B has no pass/fail; it is read against A.

### Experiment 3: what happened

**Arm A accepted.** Raw top-1% score, one checkpoint per arm ([`ae_v3_lightaug.json`](results/runs/ae_v3_lightaug.json), [`ae_v3_lightaug_noshadow.json`](results/runs/ae_v3_lightaug_noshadow.json)):

| | Clean image | Clean pixel | Lighting image | Lighting pixel | Shadowed image | Unshadowed image |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline (179 images) | 0.948 | 0.902 | 0.573 | 0.728 | 0.269 | 0.731 |
| **A: lighting augmentation** | 0.937 | 0.898 | **0.802** | **0.839** | **0.740** | 0.838 |
| B: same, shadow held out | 0.948 | 0.889 | 0.783 | 0.850 | 0.457 | **0.957** |

Lighting image-AUROC rose by 0.229 against a bar of 0.10, at a cost of 0.011 clean image and 0.004 clean pixel. Lighting pixel-AUROC, which every previous version left near 0.72, rose too. All four parts of the prediction held: the largest gain is on the shadowed subset (+0.471, now well above chance), clean held, and B recovered the unshadowed images while leaving the shadowed ones below chance.

**The failure was representation.** [`results/figures/exp3_lighting_baseline.png`](results/figures/exp3_lighting_baseline.png) against [`exp3_lighting_lightaug.png`](results/figures/exp3_lighting_lightaug.png), row 3 (the shadowed bottle): the baseline reconstructs a smear and fires along the whole outer edge; arm A reconstructs an intact bottle and the map sits on the break. The project's long-standing diagnosis, "score normalisation, not representation", was wrong. The healthy-looking lighting pixel-AUROC that supported it averaged over a subset where the model failed outright. Two scoring fixes were spent on the wrong half of the problem, and the subset split is what showed it.

**What B adds.** Training on brightness, contrast, gamma and colour shifts makes the model nearly immune to them (0.957 unshadowed, above its own clean 0.948). It does not transfer to a shadow it never saw (0.457). Robustness came from coverage of the specific perturbation, not from a general ability to handle lighting.

**How far this goes.** Arm A trained on the same transform family the test uses, with different random draws. It shows the lighting failure is learnable. It does not show the model would survive lighting no one anticipated, and B suggests it would not. Each arm is one seed. Earlier re-runs moved lighting image-AUROC by about 0.05, well under A's +0.229. The shadowed subset has 7 good images, so each one moves its image-AUROC by up to 0.14, and its numbers are directions, not precise values.

On clean images arm A's maps look like the baseline's ([`results/figures/exp3_clean_lightaug.png`](results/figures/exp3_clean_lightaug.png)). Two small differences: the reconstructed background carries a faint warm cast, learned from the colour-temperature transform, and the one-pixel frame border fires slightly more. Neither touches the defect regions, which fits the small clean cost.

Per-pixel z on top of arm A raises lighting image-AUROC to 0.839 but costs clean pixel-AUROC (0.898 → 0.831), the same rim trade as Fix 2c. Not adopted.

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
- **Image score:** mean of the top-1% hottest pixels (more robust than pure max). A median/MAD-normalized variant exists and is reported alongside, but it scored worse and is not the default.

---

## Repository layout

```
defectloc/            the pipeline
  config.py           every value that defines a run, logged into each result
  model.py            the autoencoder
  losses.py           0.5·MSE + 0.5·(1 − SSIM)
  data.py             MVTec-layout datasets, hold-out split, lighting-augmented training
  augment.py          the factory-lighting pipeline (seeded; shadow optional)
  anomaly.py          local-SSIM map, top-k score, median/MAD normalization
  calibrate.py        Fix 2c: per-pixel statistics of held-out normal maps, z map
  train.py            training + checkpoint I/O
  evaluate.py         image/pixel AUROC, every scoring variant, per-defect-type, shadow subsets
  visualize.py        input / reconstruction / map / ground-truth figures
scripts/
  prepare_data.py         fetch MVTec AD, build clean + lighting roots
  make_results_table.py   render results/results.md
  patchcore_baseline.py   re-run the PatchCore reference row (optional, heavy)
  make_synthetic_fixture.py  tiny procedural dataset for smoke tests
  label_lighting_subsets.py  replay the lighting build, label shadowed test images
tests/test_smoke.py   23 plumbing and invariant tests, no GPU or dataset needed
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

`evaluate` scores every variant in a single pass over the test set, so the raw and median/MAD numbers (and, for a hold-out checkpoint, the two `pixel_z` variants) are always measured on identical anomaly maps:

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

Training runs on GPU if one is visible and falls back to CPU otherwise (`--device` forces either). 80 epochs over 209 images takes a few minutes on a GPU; on a laptop CPU, 80 epochs over 179 images took 19 minutes.

Fix 2c needs a checkpoint trained with held-out normals. `evaluate` then adds the two `pixel_z` variants on its own, and `visualize --zmap` draws the calibrated map:

```bash
python -m defectloc.train --data-root data/MVTecAD/bottle --calib-holdout 30 \
    --out checkpoints/ae_v3_holdout30.pt

python -m defectloc.visualize --checkpoint checkpoints/ae_v3_holdout30.pt \
    --data-root data/MVTecAD_lighting/bottle --train-root data/MVTecAD/bottle \
    --zmap lighting --out results/figures/fix2c_lighting_z.png
```

Experiment 3 trains with the lighting pipeline on half the samples (`--no-train-shadow` for arm B) and scores the shadowed and unshadowed lighting images separately. The labels come from replaying the seed-0 lighting build, which fails loudly if any replayed image differs from the stored one:

```bash
python scripts/label_lighting_subsets.py \
    --clean-root data/MVTecAD/bottle --lighting-root data/MVTecAD_lighting/bottle

python -m defectloc.train --data-root data/MVTecAD/bottle --calib-holdout 30 \
    --train-lighting-p 0.5 --out checkpoints/ae_v3_lightaug.pt

python -m defectloc.evaluate --checkpoint checkpoints/ae_v3_lightaug.pt \
    --clean-root data/MVTecAD/bottle --lighting-root data/MVTecAD_lighting/bottle \
    --lighting-subsets results/lighting_subsets.json --out results/runs/ae_v3_lightaug.json
```

On this laptop's CPU, arm A took 27 minutes and arm B 20.

### Reproducing the table

```bash
python -m defectloc.evaluate --checkpoint checkpoints/ae_v3.pt \
    --clean-root data/MVTecAD/bottle \
    --lighting-root data/MVTecAD_lighting/bottle \
    --out results/runs/ae_v3.json

python scripts/make_results_table.py
```

The second command regenerates `results/results.md`, merging the recorded history with every run JSON in `results/runs/`. Each run contributes two rows, one per scoring variant. To promote a measured number into the headline table above, copy it into `results/history.json` with its provenance.

### Filling the table on Kaggle (no local dataset needed)

Kaggle already hosts MVTec AD and gives you a GPU, so the run can happen there and only a
small JSON comes back. `scripts/make_kaggle_kernel.py` generates a self-contained notebook
with the contents of `defectloc/` embedded, so the Kaggle run uses exactly this code.

```bash
pip install kaggle
python scripts/make_kaggle_kernel.py --username <your-kaggle-username>
kaggle kernels push -p kaggle_kernel
kaggle kernels status <your-kaggle-username>/bottle-defect-mad-score
kaggle kernels output <your-kaggle-username>/bottle-defect-mad-score -p results/runs/
python scripts/make_results_table.py
```

Needs a Kaggle API token at `~/.kaggle/kaggle.json` (Kaggle > Settings > Create New Token).
The kernel is created private, GPU on, internet on (one `pip install`), with `ipythonx/mvtec-ad`
attached. Regenerate the kernel after any change to `defectloc/`, or the Kaggle run and this
repo drift apart.

### Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
```

23 tests, about 15 seconds, no GPU and no dataset. They build a small procedural fixture and run the pipeline end to end. They assert **no accuracy threshold**, because a number measured on procedural rings would mean nothing; accuracy belongs in the results table, measured on MVTec AD.

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
- **The measured rows ran on CPU, not GPU.** Kaggle assigned a Tesla P100 (compute capability sm_60) alongside a torch build supporting sm_70 and up. `torch.cuda.is_available()` returns `True` in that situation and the incompatibility only surfaces once a real op runs, so `pick_device` probes the GPU with an actual convolution and falls back to CPU. The run took 1931 s instead of a few minutes. Device choice affects runtime, not results.
- The notebook's hardcoded `DATA_ROOT`, its duplicate cells, and its unseeded augmentation loop are all resolved by `scripts/prepare_data.py`. The Anomalib `RecursionError` workaround (`Engine(enable_progress_bar=False)`) is applied in `scripts/patchcore_baseline.py`.

---

## Roadmap

- [x] **Refactor** the notebook into Python modules.
- [x] **Fix 2b — per-image median/MAD score normalization.** Measured and **rejected**: image-AUROC fell in both conditions (clean 0.936 → 0.642, lighting 0.551 → 0.451). Kept in the code as an option, and in the table as a recorded negative result. See [Fix 2b: what happened](#fix-2b-what-happened).
- [x] **Fix 2c — per-pixel calibration against held-out normal images.** Measured and **rejected**: lighting image-AUROC 0.573 → 0.569 against a pre-registered bar of +0.05. It helps interior contamination (pixel 0.847 → 0.963) and hurts rim breakage (broken_large 0.927 → 0.800). See [Fix 2c: what happened](#fix-2c-what-happened).
- [x] **Lighting-augmented training.** Measured and **accepted**: lighting image-AUROC 0.573 → 0.802, clean −0.011. The shadowed subset went from inverted (0.269) to 0.740. Holding the shadow out of training (arm B) leaves it at 0.457. See [Experiment 3: what happened](#experiment-3-what-happened).
- [ ] **Robustness to unseen lighting** — B shows coverage, not generality. Candidates: a broader or stronger photometric family in training, or a per-image input normalisation (e.g. flat-field the frame against its own low-frequency illumination) that removes a shadow gradient before the model sees it. Judge on a held-out perturbation, as B does.
- [ ] **Seed spread** — re-run arm A on two more seeds before quoting 0.802 as more than one run.
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
- Every autoencoder row comes from one training run at one seed. There are no error bars. The two v3 rows are the only repeat measurement available, and they differ by up to 0.04 on lighting image-AUROC, which is the best available estimate of the noise floor.
