#!/usr/bin/env python
"""End-to-end smoke test on a synthetic dataset — no download, CPU, ~3 minutes.

Generates a fake 7-class "dermoscopy" dataset in which class identity is
encoded *only* in sub-Nyquist structure (mesh period, blob density, streak
bearing) plus a weak colour cue, and in which the structure contrast is
deliberately modulated by a simulated pigmentation axis. That is the exact
regime the paper argues about, so the smoke test also functions as a sanity
check that the pipeline can detect the effect it claims to detect.

Run:  python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cartomnist import Config
from cartomnist.pipeline import run_all


N = 224


def make_image(cls: int, pigment: float, rng: np.random.Generator) -> np.ndarray:
    """One synthetic lesion.

    `pigment` in [0, 1] darkens the skin and — as in reality — compresses the
    luminance contrast of every structure, so the naive-resize arm should lose
    more at high pigment and the generalized arm should lose less.
    """
    yy, xx = np.mgrid[0:N, 0:N] / N
    base_L = 0.80 - 0.42 * pigment
    contrast = 1.0 - 0.55 * pigment

    lesion = np.exp(-(((xx - .5) ** 2 + (yy - .5) ** 2) / 0.055))
    g = base_L - 0.22 * contrast * lesion

    # --- class-specific sub-Nyquist structure (all below 16 px period)
    mesh_period = [5.0, 6.0, 7.0, 5.5, 6.5, 6.0, 7.5][cls]
    mesh_amp = [0.06, 0.01, 0.05, 0.02, 0.055, 0.045, 0.01][cls]
    g += mesh_amp * contrast * lesion * (
        np.sin(2 * np.pi * xx * N / mesh_period)
        * np.sin(2 * np.pi * yy * N / mesh_period))

    n_blobs = [10, 45, 12, 8, 40, 15, 5][cls]
    for _ in range(n_blobs):
        a = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0, 0.22)
        cx, cy = .5 + r * np.cos(a), .5 + r * np.sin(a)
        g -= 0.16 * contrast * np.exp(
            -(((xx - cx) ** 2 + (yy - cy) ** 2) / 1.4e-5))

    n_streaks = [0, 2, 0, 0, 9, 1, 0][cls]
    theta0 = [0, 0, 0, 0, np.pi / 4, 0, 0][cls]
    for k in range(n_streaks):
        th = theta0 + rng.normal(0, 0.25)
        cx, cy = .5, .5
        d = ((xx - cx) * np.sin(th) - (yy - cy) * np.cos(th)) ** 2
        along = ((xx - cx) * np.cos(th) + (yy - cy) * np.sin(th))
        g -= 0.13 * contrast * np.exp(-d / 8e-6) * np.exp(-(along ** 2) / 0.02)

    rgbs = np.stack([g * 1.06, g * 0.80, g * 0.74], -1)
    # weak, class-correlated colour cue that DOES survive resizing, so the
    # naive arm is not at chance and the comparison is not a straw man
    rgbs[..., 2] += 0.012 * (cls - 3) / 3.0 * lesion[..., None][..., 0]
    rgbs += rng.normal(0, 0.005, rgbs.shape)
    return np.clip(rgbs, 0, 1).astype(np.float32)


def build(n_per_split=(210, 70, 140), seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    # deliberately imbalanced, like DermaMNIST
    p = np.array([0.04, 0.08, 0.11, 0.03, 0.11, 0.60, 0.03])
    for split, n in zip(("train", "val", "test"), n_per_split):
        y = rng.choice(7, size=n, p=p)
        # guarantee every class is present in every split
        y[:7] = np.arange(7)
        pig = rng.beta(1.6, 3.0, size=n)
        x = np.stack([make_image(int(c), float(g), rng) for c, g in zip(y, pig)])
        out[f"{split}_images"] = (x * 255).astype(np.uint8)
        out[f"{split}_labels"] = y.astype(np.int64)
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "smoke_out"
    cfg = Config()
    cfg.paths.root = root
    cfg.paths.mkdirs()
    cfg.apply_fast()
    cfg.max_train = cfg.max_val = cfg.max_test = 0
    # epochs_small=10, not 6: gradient clipping (added to train_eval after a
    # real-data seed was confirmed stuck in a bad trajectory regardless of
    # epoch budget) trades occasional large-but-lucky steps for consistently
    # smaller, safer ones. That is a net win on real data, but it means this
    # synthetic task's very easy, strong signal no longer converges within
    # 6 epochs the way it did when a large early step could get lucky.
    cfg.epochs_small, cfg.epochs_native = 10, 2
    cfg.batch_size, cfg.batch_size_native = 64, 32
    cfg.seeds = [0]
    cfg.topfer_resolutions = [7, 14, 28]
    cfg.topfer_k_values = [0, 2, 5]
    cfg.topfer_epochs = 3
    cfg.ita_bins = "tertile"
    cfg.run_isic_validation = True          # will report unavailable, and that is a pass
    cfg.n_jobs = 2
    cfg.amp = False

    print("building synthetic dataset ...")
    raw = build()
    out = run_all(cfg, raw_data=raw)

    print("\n" + "=" * 72)
    print("SMOKE TEST CHECKS")
    print("=" * 72)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    for k in ("naive", "generalized", "native224"):
        e = out["evaluations"][k]
        check(f"{k} produced finite AUC", np.isfinite(e["macro_auc"]),
              f"= {e['macro_auc']:.4f}")
        check(f"{k} produced an ECE", np.isfinite(e["ece"]), f"= {e['ece']:.4f}")

    check("generalized beats naive on AUC (synthetic ground truth is sub-Nyquist)",
          out["evaluations"]["generalized"]["macro_auc"]
          > out["evaluations"]["naive"]["macro_auc"],
          f"({out['evaluations']['generalized']['macro_auc']:.4f} vs "
          f"{out['evaluations']['naive']['macro_auc']:.4f})")
    check("Nyquist audit flags sub-pixel structures",
          sum(1 for v in out["nyquist"].values()
              if not v["survives_naive_resize"]) >= 3)
    # 210 synthetic training images cannot support a real scale law, so the
    # check is that the fit is well-formed and self-reporting — either a finite
    # alpha, or an explicit diagnostic saying why not.
    law = out["topfer"]["law"]
    check("Töpfer sweep produced a well-formed scale law",
          set(law["kstar"]) == set(cfg.topfer_resolutions)
          and (np.isfinite(law["alpha"]) or bool(law["diagnostic"])),
          f"alpha = {law['alpha']:.3f}" if np.isfinite(law["alpha"])
          else "alpha undefined, diagnostic present (expected at this sample size)")
    check("Mercator analysis ran for all three regimes",
          len(out["mercator"]["summary"]) == 3)
    check("shortcut test present",
          "symbols shuffled across images (shortcut test)" in out["ablations"])
    figs = sorted((root / "carto_out" / "figures").glob("*.png"))
    check("all plates drawn", len(figs) >= 12, f"({len(figs)} files)")
    zp = root / "cartomnist_report.zip"
    check("bundle written", zp.exists(),
          f"({zp.stat().st_size / 1e6:.1f} MB)" if zp.exists() else "")
    check("bundle excludes cached tensors",
          not any("carto_cache" in n for n in
                  __import__("zipfile").ZipFile(zp).namelist()) if zp.exists() else False)

    print("\n" + ("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
