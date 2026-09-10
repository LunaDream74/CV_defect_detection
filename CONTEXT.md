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
- `tests/` — 14 tests, ~10s, no GPU and no dataset required

`README.md` § How to run has the commands. Nothing here needs Kaggle any more, though
`prepare_data.py --kagglehub` still works if a Kaggle token is present.

## Results

| Model | Clean image-AUROC | Clean pixel-AUROC | Lighting image-AUROC | Lighting pixel-AUROC |
|-------|---|---|---|---|
| PatchCore (baseline, Anomalib) | 1.000 | 0.986 | 0.947 | 0.973 |
| Autoencoder v3 (notebook run) | 0.937 | 0.894 | 0.512 | 0.715 |
| Autoencoder v3 (re-run post-refactor) | 0.936 | **0.908** | **0.551** | 0.720 |
| Autoencoder v3 + median/MAD score (rejected) | 0.642 | 0.885 | 0.451 | 0.728 |

Chance AUROC is 0.500. Built from scratch: v1 0.700 → v2 0.741 (tighter bottleneck + denoising,
killed the identity shortcut) → v3 0.894 (local-SSIM anomaly map, no retraining).

The post-refactor re-run reproduces the notebook within run-to-run spread, which is the evidence
that the refactor preserved the numerics.

## The open issue, stated precisely

Under the lighting stress test, pixel-AUROC holds at 0.720 but image-AUROC sits near chance (0.551).
This is **score normalisation, not representation**: a global lighting shift moves each image's error
baseline by a different amount, so per-image scores stop being comparable.

**The first fix was measured and rejected.** Per-image median/MAD normalisation
(`defectloc/anomaly.py::normalize_map`) made image-AUROC *worse* in both conditions: clean
0.936 → 0.642, lighting 0.551 → 0.451, the latter below chance and therefore inverted. Two reasons,
both in README § Fix 2b: what happened. The overall reconstruction error it divided out was real
signal, and MAD is not defect-invariant because the map fires across the whole rim annulus, so a
defective image gets a larger divisor. The code and the numbers are kept so the failure is not
retried by accident.

The diagnosis still stands; the remedy does not. Next candidate (Fix 2c) is to calibrate against
held-out normal images rather than against the image's own pixels, so the defect cannot inflate its
own divisor.

## Status

Refactor done and validated by a reproduction run. Fix 2b tested and rejected. The next action is
Fix 2c, or tightening the map's rim over-firing, which is the more fundamental problem. No stage
folders: the pipeline is a package with a CLI, not a staged workflow.

## Standing constraints

- Layer 3: `../_config/conventions.md` § Experiment discipline. This project is the reference
  example of it: one hypothesis per version, predicted direction stated, anomaly maps inspected
  before the metric was trusted.
- Layer 3: `../_config/conventions.md` § Results tables. `results/results.md` is generated, never
  hand-edited; unmeasured cells show as `—` rather than being filled in.
- Layer 3: `../_config/writing-voice.md` for any writeup. The existing "Scope and limitations"
  section is the model to match.
