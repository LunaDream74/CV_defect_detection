# Computer_vision · Layer 2 — project router

Reconstruction-based defect **localisation** for industrial inspection. A convolutional autoencoder
trained only on normal bottles produces a spatial anomaly map, so a defect's location can be traced
back to the upstream manufacturing station that caused it. Unsupervised, so it generalises to
unseen defect types.

## Files

| File | Layer | Role |
|------|-------|------|
| [`README.md`](README.md) | 3/4 | Method, results, architecture, how to run, roadmap, scope limits. **Read this first** — it is complete and this file does not restate it. |
| [`results/results.md`](results/results.md) | 4 | Generated results table with provenance for every number |
| [`defect_detection_log.md`](defect_detection_log.md) | 4 | Full project log: design reasoning and diagnostic methodology behind each version |
| [`patchcore-base.ipynb`](patchcore-base.ipynb) | 4 | The original v3 Kaggle notebook, kept for provenance. Superseded by `defectloc/`. |

## The code is here now

The notebook has been refactored into a runnable package. This folder is a git repository.

- `defectloc/` — the pipeline (`config`, `model`, `losses`, `data`, `augment`, `anomaly`, `train`,
  `evaluate`, `visualize`)
- `scripts/` — `prepare_data.py`, `make_results_table.py`, `patchcore_baseline.py`,
  `make_synthetic_fixture.py`
- `tests/` — 19 tests, ~15s, no GPU and no dataset required
- `defectloc/calibrate.py` — Fix 2c per-pixel calibration (rejected, kept for reuse)
- Local data: `data/` (gitignored) holds MVTec bottle, fetched file by file from Kaggle on
  2026-10-26 (`data/raw/bottle`), plus the built clean and lighting roots. Torch is CPU-only here.

`README.md` § How to run has the commands. Nothing here needs Kaggle any more, though
`prepare_data.py --kagglehub` still works if a Kaggle token is present.

## Results

| Model | Clean image-AUROC | Clean pixel-AUROC | Lighting image-AUROC | Lighting pixel-AUROC |
|-------|---|---|---|---|
| PatchCore (baseline, Anomalib) | 1.000 | 0.986 | 0.947 | 0.973 |
| Autoencoder v3 (notebook run) | 0.937 | 0.894 | 0.512 | 0.715 |
| Autoencoder v3 (re-run post-refactor) | 0.936 | **0.908** | **0.551** | 0.720 |
| Autoencoder v3 + median/MAD score (rejected) | 0.642 | 0.885 | 0.451 | 0.728 |
| Autoencoder v3, 179-image train (30 held out) | 0.948 | 0.902 | 0.573 | 0.728 |
| ↳ + per-pixel z, matched calibration (rejected) | 0.958 | 0.869 | 0.569 | 0.650 |
| **Autoencoder v3 + lighting-augmented training (accepted)** | 0.937 | 0.898 | **0.802** | **0.839** |
| ↳ same, shadow held out of training | 0.948 | 0.889 | 0.783 | 0.850 |
| ↳ lighting augmentation, mean of 3 seeds | 0.938 | 0.887 | **0.887** | 0.850 |
| ↳ shadow held out + flat-field (rejected) | 0.947 | 0.909 | 0.860 | 0.867 |
| **Autoencoder v3 + flat-field input (default), 3 seeds** | **0.988** | **0.921** | 0.818 | 0.837 |

Chance AUROC is 0.500. Built from scratch: v1 0.700 → v2 0.741 (tighter bottleneck + denoising,
killed the identity shortcut) → v3 0.894 (local-SSIM anomaly map, no retraining).

The post-refactor re-run reproduces the notebook within run-to-run spread, which is the evidence
that the refactor preserved the numerics.

## The lighting issue, and how the diagnosis changed

Under the lighting stress test, pixel-AUROC held at 0.720 but image-AUROC sat near chance (0.551).
This was read as **score normalisation, not representation**: a global lighting shift moves each
image's error baseline by a different amount, so per-image scores stop being comparable. Two scoring
fixes were built on that reading, then Experiment 3 showed the reading was wrong. History in order:

**The first fix was measured and rejected.** Per-image median/MAD normalisation
(`defectloc/anomaly.py::normalize_map`) made image-AUROC *worse* in both conditions: clean
0.936 → 0.642, lighting 0.551 → 0.451, the latter below chance and therefore inverted. Two reasons,
both in README § Fix 2b: what happened. The overall reconstruction error it divided out was real
signal, and MAD is not defect-invariant because the map fires across the whole rim annulus, so a
defective image gets a larger divisor. The code and the numbers are kept so the failure is not
retried by accident.

**The second fix was measured and rejected too (2026-10-26).** Fix 2c, per-pixel calibration on 30
held-out normal images, moved lighting image-AUROC 0.573 → 0.569 against a pre-registered bar of
+0.05 (README § Fix 2c: what happened). It helps interior contamination (pixel 0.847 → 0.963) and hurts
breakage (broken_large 0.927 → 0.800), because bottles break at the rim the calibration discounts.
Under lighting the normal spread is ~16× the clean spread, so the calibration has nothing to grip.

**The diagnosis itself was wrong (Experiment 3, 2026-10-26).** Replaying the lighting build to label
shadowed frames showed lighting image-AUROC was *inverted* on them (0.269) and 0.731 elsewhere: the
reconstruction collapses under the shadow. Lighting-augmented training (arm A) lifts lighting
image-AUROC 0.573 → 0.802 (shadowed 0.740) at −0.011 clean, passing its pre-registered bar. Arm B,
trained without the shadow, handles the other perturbations (0.957) but not the shadow (0.457): the
robustness is coverage of what was trained on, not general.

## Status

Refactor done and validated by a reproduction run (re-confirmed on local data 2026-10-26). Fix 2b and
Fix 2c (scoring) rejected; Experiment 3 (lighting-augmented training) accepted, three seeds: lighting
image-AUROC 0.887 ± 0.074. Experiment 4 (flat-field input): arm C rejected on its unshadowed floor;
**arm D (flat-field only) replicated on three seeds, clean 0.988 / 0.921, and flat-field is now the
default input** (`flatfield_sigma = 16`). Checkpoints from before it load with it off; reproduce any
earlier row with `--flatfield-sigma 0`.

Next (README § Roadmap): arm A + flat-field on three seeds (`--train-lighting-p 0.5 --seed N`, flat-field
is on by default; ~37 min each on CPU, a stopped run starts over). Also replicate C, and measure the
no-flat-field baseline's seed spread. Everything through the D replicates and the default change is
committed and pushed to `origin/main` (2026-10-26). Checkpoints are local only (gitignored).
No stage folders: the pipeline is a package with a CLI, not a staged workflow.

## Standing constraints

- Layer 3: `../_config/conventions.md` § Experiment discipline. This project is the reference
  example of it: one hypothesis per version, predicted direction stated, anomaly maps inspected
  before the metric was trusted.
- Layer 3: `../_config/conventions.md` § Results tables. `results/results.md` is generated, never
  hand-edited; unmeasured cells show as `—` rather than being filled in.
- Layer 3: `../_config/writing-voice.md` for any writeup. The existing "Scope and limitations"
  section is the model to match.
