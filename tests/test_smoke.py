"""Fast checks that the pipeline is wired correctly and the scoring fix holds.

These are plumbing and invariant tests. They deliberately assert **no accuracy
threshold**: they run on a procedural fixture, not on bottles, and a number
from that fixture would mean nothing. Accuracy lives in results/results.md,
measured on MVTec AD.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defectloc.anomaly import anomaly_map, image_score, local_ssim_dissimilarity, normalize_map
from defectloc.augment import build_factory_lighting
from defectloc.config import Config
from defectloc.data import TestSet, TrainGood
from defectloc.model import ConvAutoencoder

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def fixture_root(tmp_path_factory):
    sys.path.insert(0, str(REPO / "scripts"))
    from make_synthetic_fixture import build

    return build(tmp_path_factory.mktemp("mvtec_like") / "bottle", size=64, seed=0)


# --------------------------------------------------------------------------
# model / map plumbing
# --------------------------------------------------------------------------

def test_autoencoder_preserves_shape_and_range():
    model = ConvAutoencoder(latent_ch=64).eval()
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape
    assert 0.0 <= float(y.min()) and float(y.max()) <= 1.0, "sigmoid output must stay in [0,1]"


def test_bottleneck_is_tight():
    """The v1 -> v2 fix. A latent this size is what stops the identity shortcut."""
    model = ConvAutoencoder(latent_ch=64).eval()
    with torch.no_grad():
        z = model.encoder(torch.rand(1, 3, 256, 256))
    assert tuple(z.shape[1:]) == (64, 8, 8)
    assert z[0].numel() == 4096


def test_ssim_dissimilarity_is_zero_for_identical_images():
    x = torch.rand(2, 3, 64, 64)
    d = local_ssim_dissimilarity(x, x)
    assert d.shape == (2, 64, 64)
    assert float(d.max()) < 1e-4, "an image against itself must be structurally identical"


def test_anomaly_map_is_hotter_on_a_corrupted_region():
    x = torch.rand(1, 3, 64, 64) * 0.1 + 0.5
    y = x.clone()
    y[:, :, 20:40, 20:40] = 0.0          # a blatant structural difference
    d = local_ssim_dissimilarity(x, y)[0].numpy()
    assert d[25:35, 25:35].mean() > d[0:10, 0:10].mean()


# --------------------------------------------------------------------------
# the scoring fix -- the point of this repo
# --------------------------------------------------------------------------

def _map_with_defect(rng, lift=0.0, gain=1.0):
    base = rng.normal(0.10, 0.01, (128, 128))
    base[40:60, 40:60] += 0.25
    return base * gain + lift


def test_normalisation_does_not_change_which_pixels_are_hottest():
    """Normalisation is monotonic within an image: same pixels, different units."""
    rng = np.random.default_rng(0)
    amap = _map_with_defect(rng)
    k = 100
    top_raw = np.argsort(amap.ravel())[-k:]
    top_norm = np.argsort(normalize_map(amap).ravel())[-k:]
    assert np.array_equal(np.sort(top_raw), np.sort(top_norm))


def test_median_mad_score_is_invariant_to_a_global_lighting_shift():
    """The regression test for the open v3 bug.

    A lighting shift moves an image's whole error baseline (an affine change to
    its anomaly map). The raw top-k score moves with it; the normalised score
    must not, or image-level ranking collapses across lighting conditions --
    which is exactly the 0.512 image-AUROC in the results table.
    """
    rng = np.random.default_rng(0)
    amap = _map_with_defect(rng)
    shifted = amap * 1.6 + 0.20

    raw, raw_shifted = image_score(amap), image_score(shifted)
    norm = image_score(amap, norm="median_mad")
    norm_shifted = image_score(shifted, norm="median_mad")

    assert raw_shifted > raw * 1.5, "sanity: the raw score should move a lot"
    assert norm_shifted == pytest.approx(norm, rel=1e-6)


def test_raw_score_misranks_across_lighting_but_normalised_score_does_not():
    """The failure mode in one assertion: a lit-up GOOD image outranking a real defect."""
    rng = np.random.default_rng(1)
    good = rng.normal(0.10, 0.01, (128, 128))
    defect = _map_with_defect(rng)
    good_shifted = good * 1.6 + 0.20     # same photometric shift, no defect

    assert image_score(good_shifted) > image_score(defect), "the bug"
    assert image_score(good_shifted, norm="median_mad") < image_score(defect, norm="median_mad")


def test_image_score_rejects_an_unknown_normalisation():
    with pytest.raises(ValueError):
        image_score(np.zeros((8, 8)), norm="zscore")


# --------------------------------------------------------------------------
# Fix 2c: per-pixel calibration on held-out normals
# --------------------------------------------------------------------------

def test_one_mean_and_std_per_condition_cannot_change_the_ranking():
    """Why Fix 2c is per pixel. Uniform statistics are one affine map for every
    image in a condition, and image-AUROC is invariant to that, so the 'calibrate
    against normal images' fix as first written could not have moved the metric."""
    from defectloc.calibrate import NormalStats

    rng = np.random.default_rng(0)
    maps = [rng.normal(0.1, 0.02, (32, 32)) + (i % 3) * 0.05 for i in range(9)]
    uniform = NormalStats(mu=np.full((32, 32), 0.1), sigma=np.full((32, 32), 0.03),
                          floor=0.0, n_maps=9, condition="clean")
    raw = [image_score(m) for m in maps]
    cal = [image_score(m, norm="pixel_z", stats=uniform) for m in maps]
    assert np.array_equal(np.argsort(raw), np.argsort(cal))


def test_pixel_z_discounts_a_rim_that_fires_on_normal_images_too():
    """The rim case. Every normal map is hot on the rim; a defect elsewhere is
    dimmer than the rim. Raw top-k ranks a good image above the defective one,
    per-pixel z against held-out normals does not."""
    from defectloc.calibrate import fit_normal_stats

    rng = np.random.default_rng(0)
    rim = np.zeros((64, 64), bool)
    rim[:, :4] = True

    def base():
        m = rng.normal(0.10, 0.01, (64, 64))
        m[rim] += rng.normal(0.90, 0.10)
        return m

    stats = fit_normal_stats(np.stack([base() for _ in range(30)]), "clean")
    good = base()
    defective = base()
    defective[rim] -= 0.2          # this frame's rim happens to run cooler
    defective[30:36, 30:36] = 0.8  # defect, dimmer than a typical rim

    assert image_score(good) > image_score(defective), "sanity: raw misranks"
    assert (image_score(defective, norm="pixel_z", stats=stats)
            > image_score(good, norm="pixel_z", stats=stats))


def test_pixel_z_needs_matching_statistics():
    from defectloc.calibrate import fit_normal_stats

    stats = fit_normal_stats(np.zeros((3, 8, 8)), "clean")
    with pytest.raises(ValueError):
        image_score(np.zeros((8, 8)), norm="pixel_z")          # no stats
    with pytest.raises(ValueError):
        image_score(np.zeros((16, 16)), norm="pixel_z", stats=stats)  # wrong shape
    # Degenerate calibration (sigma == 0 everywhere) must not divide by zero.
    assert np.isfinite(image_score(np.ones((8, 8)), norm="pixel_z", stats=stats))


def test_holdout_split_is_deterministic_and_disjoint(fixture_root):
    from defectloc.data import split_holdout

    paths = TrainGood(fixture_root).paths
    tr, ho = split_holdout(paths, 5, seed=0)
    tr2, ho2 = split_holdout(paths, 5, seed=0)
    assert (tr, ho) == (tr2, ho2)
    assert len(ho) == 5 and len(tr) == len(paths) - 5 and not set(tr) & set(ho)
    assert TrainGood(fixture_root, holdout=5, part="holdout").paths == ho
    assert TrainGood(fixture_root, holdout=5).paths == tr
    with pytest.raises(ValueError):
        split_holdout(paths, len(paths), seed=0)


# --------------------------------------------------------------------------
# augmentation
# --------------------------------------------------------------------------

def test_lighting_stress_is_seeded_and_actually_changes_the_image():
    img = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    a = build_factory_lighting(seed=7)(image=img)["image"]
    b = build_factory_lighting(seed=7)(image=img)["image"]
    c = build_factory_lighting(seed=8)(image=img)["image"]
    assert np.array_equal(a, b), "same seed must reproduce the same stress test"
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, img)
    assert a.dtype == np.uint8 and a.shape == img.shape


# --------------------------------------------------------------------------
# datasets and end-to-end run
# --------------------------------------------------------------------------

def test_datasets_load_labels_and_masks(fixture_root):
    train = TrainGood(fixture_root, img_size=64)
    test = TestSet(fixture_root, img_size=64)
    assert len(train) == 24
    assert len(test) == 20

    img, mask, label = test[0]
    assert img.shape == (3, 64, 64) and mask.shape == (64, 64)
    assert set(np.unique(mask.numpy())) <= {0.0, 1.0}

    labels = [lab for _, _, lab in test.items]
    assert sum(labels) == 12 and labels.count(0) == 8
    # good images must carry an empty mask
    good_idx = labels.index(0)
    assert test[good_idx][1].sum() == 0


def test_missing_data_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        TrainGood(tmp_path)
    with pytest.raises(FileNotFoundError):
        TestSet(tmp_path)


def test_end_to_end_train_evaluate(fixture_root, tmp_path):
    from defectloc.evaluate import evaluate
    from defectloc.train import load_checkpoint, save_checkpoint, train

    cfg = Config(img_size=64, epochs=2, batch_size=8, num_workers=0, seed=0)
    device = torch.device("cpu")

    model, history = train(fixture_root, cfg, device, verbose=False)
    assert len(history) == 2 and np.isfinite(history[-1]["loss"])

    ckpt = save_checkpoint(tmp_path / "ae.pt", model, cfg, history)
    reloaded, cfg2, _ = load_checkpoint(ckpt, device)
    assert cfg2.latent_ch == cfg.latent_ch

    res = evaluate(reloaded, fixture_root, cfg2, device, batch_size=4)
    assert res["n_images"] == 20 and res["n_defective"] == 12
    for variant in ("none", "median_mad"):
        m = res["variants"][variant]
        assert 0.0 <= m["image_AUROC"] <= 1.0
        assert 0.0 <= m["pixel_AUROC"] <= 1.0
        assert set(m["per_defect_type"]) == {"chip", "contamination"}


def test_end_to_end_with_calibration_holdout(fixture_root, tmp_path):
    """The Fix 2c path through the real CLIs: train with a hold-out, evaluate
    every variant, and refuse a checkpoint whose recorded split does not match."""
    import json

    from defectloc import evaluate as ev
    from defectloc import train as tr

    ckpt = tmp_path / "ae_cal.pt"
    assert tr.main(["--data-root", str(fixture_root), "--out", str(ckpt), "--epochs", "1",
                    "--img-size", "64", "--batch-size", "8", "--calib-holdout", "4",
                    "--device", "cpu"]) == 0
    out = tmp_path / "run.json"
    assert ev.main(["--checkpoint", str(ckpt), "--clean-root", str(fixture_root),
                    "--lighting-root", str(fixture_root), "--out", str(out),
                    "--device", "cpu"]) == 0
    run = json.loads(out.read_text(encoding="utf-8"))
    assert len(run["calibration"]["holdout_files"]) == 4
    assert run["calibration"]["lighting"]["n_maps"] == 4 * Config().calib_copies
    for cond in ("clean", "lighting"):
        assert set(run["conditions"][cond]["variants"]) == {
            "none", "median_mad", "pixel_z", "pixel_z_cleancal"}
    clean = run["conditions"]["clean"]["variants"]
    assert clean["pixel_z"]["image_AUROC"] == clean["pixel_z_cleancal"]["image_AUROC"]

    model, cfg, meta = tr.load_checkpoint(ckpt, torch.device("cpu"))
    meta["calib_holdout_files"] = ["not-a-file.png"] * 4
    tr.save_checkpoint(ckpt, model, cfg, meta["history"],
                       extra={"calib_holdout_files": meta["calib_holdout_files"]})
    with pytest.raises(SystemExit):
        ev.main(["--checkpoint", str(ckpt), "--clean-root", str(fixture_root),
                 "--out", str(out), "--device", "cpu"])


def test_reconstruction_is_deterministic_given_a_seed(fixture_root):
    from defectloc.train import train

    cfg = Config(img_size=64, epochs=1, batch_size=8, num_workers=0, seed=3)
    device = torch.device("cpu")
    a, _ = train(fixture_root, cfg, device, verbose=False)
    b, _ = train(fixture_root, cfg, device, verbose=False)
    x = torch.rand(1, 3, 64, 64)
    assert np.allclose(anomaly_map(a, x), anomaly_map(b, x), atol=1e-6)


# --------------------------------------------------------------------------
# the deliverable itself
# --------------------------------------------------------------------------

def test_results_table_regenerates(tmp_path):
    out = tmp_path / "results.md"
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "make_results_table.py"),
         "--history", str(REPO / "results" / "history.json"),
         "--runs-dir", str(tmp_path / "no_runs"),
         "--out", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    text = out.read_text(encoding="utf-8")
    assert "| Model |" in text
    assert "PatchCore" in text and "0.894" in text
    assert "--" in text, "unmeasured rows must show as -- rather than vanish"
