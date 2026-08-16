"""End-to-end orchestration. `run_all(cfg)` produces every number and figure.

Design note: the `naive` regime is not a separately built array. It is channels
0-2 of the generalized tensor, which by construction *are* the area-pooled RGB
that a MedMNIST-style benchmark ships. Selecting channels rather than rebuilding
guarantees that the only difference between the two low-resolution arms is the
presence of the symbol and source-diagram channels — not an interpolation
kernel, not a colour-space round trip, not a random crop.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import figures as F
from . import ita as ITA
from . import mercator as MERC
from . import metrics as M
from . import topfer as TOP
from .config import Config
from .contracts import legend_dict
from .data import CLASS_SHORT, DermaSource, cached
from .filterbanks import STRUCTURE_NAMES, estimate_absolute_scales
from .generalize import (channel_spec, generalize, generalize_batch,
                         generalize_multi_batch, raw_structure_maps_batch)
from .sufficiency import nyquist_table
from .train import EquivariantAug, average_logits, train_eval


class Timer:
    def __init__(self):
        self.t0 = time.time()
        self.marks: List[tuple] = []

    def mark(self, name: str) -> None:
        self.marks.append((name, round(time.time() - self.t0, 1)))
        print(f"  [{self.marks[-1][1]:>7.1f}s] {name}")


def _orient_indices(spec, chans: List[int]):
    cos_i = [chans.index(i) for i, nm in enumerate(spec.names)
             if nm.endswith("::cos2theta") and i in chans]
    sin_i = [chans.index(i) for i, nm in enumerate(spec.names)
             if nm.endswith("::sin2theta") and i in chans]
    return cos_i, sin_i


# ==========================================================================
def run_all(cfg: Optional[Config] = None,
            raw_data: Optional[Dict[str, np.ndarray]] = None) -> Dict:
    cfg = cfg or Config()
    P = cfg.paths.mkdirs()
    T = Timer()
    out: Dict[str, object] = {"config": json.loads(cfg.to_json()),
                              "legend": legend_dict()}
    print("=" * 72)
    print("CARTOGRAPHIC GENERALIZATION FOR MEDMNIST — full run")
    print("=" * 72)

    # ---------------------------------------------------------------- data
    src = DermaSource(cfg, raw=raw_data)
    n_classes = src.n_classes
    rare = src.rare_classes()
    out["class_counts"] = {s: src.class_counts(s) for s in ("train", "val", "test")}
    out["rare_classes"] = [CLASS_SHORT[c] for c in rare]
    T.mark(f"loaded DermaMNIST-224  "
           f"train={len(src.labels('train'))} val={len(src.labels('val'))} "
           f"test={len(src.labels('test'))}")

    spec = channel_spec()
    out["channels"] = spec.names
    out["nyquist"] = nyquist_table(cfg.target_size, cfg.native_size)

    # ------------------------------------------------------- generalization
    tag = f"{cfg.target_size}_{'ef' if cfg.equal_fidelity else 'abs'}"
    gen: Dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        gen[split] = cached(
            P.cache / f"gen_{split}_{tag}_{len(src.labels(split))}.npy",
            lambda s=split: generalize_batch(
                src.images(s), target=cfg.target_size, n_jobs=cfg.n_jobs,
                equal_fidelity=cfg.equal_fidelity,
                exaggeration_gamma=cfg.exaggeration_gamma,
                displacement_lambda=cfg.displacement_lambda,
                coherence_floor=cfg.coherence_floor))
        T.mark(f"generalized {split}: {gen[split].shape}")

    # -------------------------------------------------------------- ITA
    ita: Dict[str, np.ndarray] = {}
    for split in ("train", "test"):
        ita[split] = np.asarray(cached(
            P.cache / f"ita_{split}_{len(src.labels(split))}.npy",
            lambda s=split: ITA.batch_ita(src.images(s), n_jobs=cfg.n_jobs)))
    bin_idx, bin_labels = ITA.make_bins(ita["test"], cfg.ita_bins)
    out["ita_summary"] = ITA.summary(ita["test"])
    out["ita_bins"] = list(bin_labels)
    out["ita_bin_counts"] = {l: int((bin_idx == b).sum())
                             for b, l in enumerate(bin_labels)}
    T.mark(f"ITA computed; strata = {bin_labels}")

    # =================================================== classification arms
    ch_rgb = spec.idx("rgb")
    ch_all = list(range(spec.n))
    cos_i, sin_i = _orient_indices(spec, ch_all)
    aug_gen = EquivariantAug(cos_i, sin_i)
    aug_rgb = EquivariantAug([], [])

    results: Dict[str, List] = {}
    y = {s: src.labels(s) for s in ("train", "val", "test")}

    print("\n-- training: naive 28²  (RGB channels only)")
    results["naive"] = [train_eval(
        gen["train"], y["train"], gen["val"], y["val"], gen["test"], y["test"],
        regime="naive", n_classes=n_classes, cfg=cfg, channels=ch_rgb,
        aug=aug_rgb, seed=s) for s in cfg.seeds]
    T.mark("naive trained")

    print("\n-- training: generalized 28²  (full sheet)")
    results["generalized"] = [train_eval(
        gen["train"], y["train"], gen["val"], y["val"], gen["test"], y["test"],
        regime="generalized", n_classes=n_classes, cfg=cfg, channels=ch_all,
        aug=aug_gen, seed=s) for s in cfg.seeds]
    T.mark("generalized trained")

    print("\n-- training: native 224²  (ResNet-18)")
    results["native224"] = [train_eval(
        src.images("train"), y["train"], src.images("val"), y["val"],
        src.images("test"), y["test"], regime="native224",
        n_classes=n_classes, cfg=cfg, aug=None, seed=s) for s in cfg.seeds]
    T.mark("native224 trained")

    evals, extras = {}, {}
    for key, res in results.items():
        vlog, tlog = average_logits(res)
        ev, ex = M.evaluate(tlog, res[0].test_y, regime=key, n_classes=n_classes,
                            rare_classes=rare, val_logits=vlog, val_y=res[0].val_y,
                            ita_bin=bin_idx, ita_labels=bin_labels,
                            n_bins=cfg.n_reliability_bins,
                            temperature_scale=cfg.temperature_scale)
        ev.extra["n_params"] = float(res[0].n_params)
        ev.extra["epochs"] = float(res[0].epochs)
        ev.extra["n_seeds"] = float(len(res))
        evals[key], extras[key] = ev, ex
        print(f"  {key:>12}: AUC {ev.macro_auc:.4f}  bAcc {ev.balanced_accuracy:.4f}  "
              f"rare {ev.rare_recall:.4f}  ECE {ev.ece:.4f}  ITA gap {ev.ita_gap:.4f}")
    out["evaluations"] = {k: v.as_dict() for k, v in evals.items()}

    # ---- the headline number
    try:
        n_, g_, t_ = (evals[k].rare_recall for k in ("naive", "generalized", "native224"))
        out["gap_recovery"] = {
            "rare_recall_naive": n_, "rare_recall_generalized": g_,
            "rare_recall_native224": t_,
            "fraction_of_224_gap_recovered_at_28":
                float((g_ - n_) / (t_ - n_)) if abs(t_ - n_) > 1e-6 else float("nan"),
        }
        for m in ("balanced_accuracy", "macro_auc", "ita_gap"):
            a, b, c = (float(getattr(evals[k], m))
                       for k in ("naive", "generalized", "native224"))
            out["gap_recovery"][f"{m}_recovery"] = (
                float((b - a) / (c - a)) if abs(c - a) > 1e-6 else float("nan"))
    except Exception as exc:                            # noqa: BLE001
        out["gap_recovery"] = {"error": str(exc)}

    # ============================================================ ablations
    print("\n-- ablations (test-time channel surgery on the generalized model)")
    model_res = results["generalized"]
    ab: Dict[str, Dict[str, float]] = {}

    def _score(tlog, ty, vlog, vy) -> Dict[str, float]:
        ev, _ = M.evaluate(tlog, ty, regime="ab", n_classes=n_classes,
                           rare_classes=rare, val_logits=vlog, val_y=vy,
                           n_bins=cfg.n_reliability_bins)
        return {"macro_auc": ev.macro_auc, "rare_recall": ev.rare_recall,
                "balanced_accuracy": ev.balanced_accuracy, "ece": ev.ece}

    vlog, tlog = average_logits(model_res)
    ab["generalized (full)"] = _score(tlog, model_res[0].test_y, vlog,
                                      model_res[0].val_y)
    nv, nt = average_logits(results["naive"])
    ab["naive (trained w/o symbols)"] = _score(nt, results["naive"][0].test_y, nv,
                                               results["naive"][0].val_y)

    ablations = {
        "symbols zeroed at test": spec.idx("symbols"),
        "source diagram zeroed at test": spec.idx("source_diagram"),
        "RGB zeroed at test (symbols only)": spec.idx("rgb"),
    }
    for name, zero in ablations.items():
        if not zero:
            continue
        res = [train_eval(gen["train"], y["train"], gen["val"], y["val"],
                          gen["test"], y["test"], regime="generalized",
                          n_classes=n_classes, cfg=cfg, channels=ch_all,
                          aug=aug_gen, test_zero_channels=zero, seed=s,
                          verbose=False) for s in cfg.seeds[:1]]
        v, t = average_logits(res)
        ab[name] = _score(t, res[0].test_y, v, res[0].val_y)
        print(f"  {name}: AUC {ab[name]['macro_auc']:.4f}")

    # the shortcut test: destroy the symbols' alignment to the image
    sym_idx = spec.idx("symbols")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y["test"]))
    shuffled = np.array(gen["test"], dtype=np.float32, copy=True)
    shuffled[:, sym_idx] = shuffled[perm][:, sym_idx]
    res = [train_eval(gen["train"], y["train"], gen["val"], y["val"],
                      shuffled, y["test"], regime="generalized",
                      n_classes=n_classes, cfg=cfg, channels=ch_all,
                      aug=aug_gen, seed=s, verbose=False) for s in cfg.seeds[:1]]
    v, t = average_logits(res)
    ab["symbols shuffled across images (shortcut test)"] = _score(
        t, res[0].test_y, v, res[0].val_y)
    del shuffled
    print(f"  shortcut test: AUC "
          f"{ab['symbols shuffled across images (shortcut test)']['macro_auc']:.4f}")
    out["ablations"] = ab
    T.mark("ablations done")

    # ==================================================== certificate / abstain
    cert_ch = [i for i in spec.groups["source_diagram"]
               if spec.names[i].endswith("::certificate")][0]
    dens_ch = [i for i in spec.groups["symbols"]
               if spec.names[i].endswith("::density")]
    test_gen = np.asarray(gen["test"])
    cert_map = test_gen[:, cert_ch]
    dens_map = test_gen[:, dens_ch].sum(axis=1)
    cert_score = np.array([
        float((cert_map[i] * dens_map[i]).sum() / max(1e-8, dens_map[i].sum()))
        for i in range(len(cert_map))], dtype=np.float32)

    probs = extras["generalized"]["probs"]
    ty = results["generalized"][0].test_y
    curves = {
        "source certificate": M.coverage_risk(cert_score, probs, ty),
        "softmax confidence": M.coverage_risk(probs.max(1), probs, ty),
        "random": M.coverage_risk(np.random.default_rng(0).random(len(ty)), probs, ty),
    }
    out["coverage_risk"] = {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                                for kk, vv in v.items()} for k, v in curves.items()}
    out["certificate_vs_confidence"] = {
        k: v["balanced_aurc"] for k, v in curves.items()}
    T.mark("certificate / abstention evaluated")

    # ======================================================= Mercator analysis
    print("\n-- Mercator analysis (classifier-free)")
    n_probe = min(len(y["train"]), 2500 if not cfg.fast else 1200)
    pidx = np.sort(np.random.default_rng(0).choice(len(y["train"]), n_probe,
                                                   replace=False))
    tr_img = src.images("train")[pidx]

    raw_tr = cached(P.cache / f"rawmaps_train_{n_probe}_{cfg.target_size}.npy",
                    lambda: np.stack([raw_structure_maps_batch(
                        tr_img, cfg.target_size, cfg.n_jobs)[s]
                        for s in STRUCTURE_NAMES]))
    raw_te = cached(P.cache / f"rawmaps_test_{len(y['test'])}_{cfg.target_size}.npy",
                    lambda: np.stack([raw_structure_maps_batch(
                        src.images("test"), cfg.target_size, cfg.n_jobs)[s]
                        for s in STRUCTURE_NAMES]))
    targets_tr = {s: np.asarray(raw_tr[i]) for i, s in enumerate(STRUCTURE_NAMES)}
    targets_te = {s: np.asarray(raw_te[i]) for i, s in enumerate(STRUCTURE_NAMES)}
    T.mark("reference structure maps computed")

    abs_scales = estimate_absolute_scales(tr_img, cfg.native_size, n=200)
    out["absolute_scales"] = abs_scales
    gen_abs_tr = cached(P.cache / f"genabs_train_{n_probe}_{cfg.target_size}.npy",
                        lambda: generalize_batch(
                            tr_img, target=cfg.target_size, n_jobs=cfg.n_jobs,
                            equal_fidelity=False, absolute_scales=abs_scales,
                            exaggeration_gamma=cfg.exaggeration_gamma,
                            displacement_lambda=cfg.displacement_lambda,
                            coherence_floor=cfg.coherence_floor))
    gen_abs_te = cached(P.cache / f"genabs_test_{len(y['test'])}_{cfg.target_size}.npy",
                        lambda: generalize_batch(
                            src.images("test"), target=cfg.target_size,
                            n_jobs=cfg.n_jobs, equal_fidelity=False,
                            absolute_scales=abs_scales,
                            exaggeration_gamma=cfg.exaggeration_gamma,
                            displacement_lambda=cfg.displacement_lambda,
                            coherence_floor=cfg.coherence_floor))
    T.mark("absolute-scale arm generalized")

    gen_tr_sub = np.asarray(gen["train"])[pidx]
    regimes = {
        "naive": {"train": gen_tr_sub[:, :3], "test": test_gen[:, :3]},
        "generalized_ef": {"train": gen_tr_sub, "test": test_gen},
        "generalized_abs": {"train": np.asarray(gen_abs_tr),
                            "test": np.asarray(gen_abs_te)},
    }
    mres = MERC.analyse(regimes, targets_tr, targets_te, ita["test"],
                        bin_idx, bin_labels)
    merc_summary = MERC.summarise(mres)
    out["mercator"] = {"per_structure": [m.as_dict() for m in mres],
                       "summary": merc_summary}
    dyn = {r: MERC.symbol_dynamic_range(
        np.asarray(regimes[r]["test"]), spec, ita["test"], bin_idx, bin_labels)
        for r in ("generalized_ef", "generalized_abs")}
    out["symbol_dynamic_range"] = dyn
    for r, s in merc_summary.items():
        print(f"  {r:>18}: mean retention spread across ITA strata "
              f"{s['mean_spread']:.4f}  ({s['n_significant_slopes']} significant slopes)")
    T.mark("Mercator analysis done")

    # ======================================================== Töpfer sweep
    print("\n-- Töpfer sweep")
    n_top = min(len(y["train"]), 3500 if not cfg.fast else 1500)
    tidx = np.sort(np.random.default_rng(1).choice(len(y["train"]), n_top,
                                                   replace=False))
    n_top_te = min(len(y["test"]), 2000)
    teidx = np.sort(np.random.default_rng(2).choice(len(y["test"]), n_top_te,
                                                    replace=False))
    res_list = cfg.topfer_resolutions
    key = "_".join(map(str, res_list))

    def _build_multi(idxs, imgs):
        """One survey pass for ALL resolutions, flattened into a single array.

        Calling generalize_multi_batch inside the per-resolution comprehension
        would re-survey the images once per resolution, which is exactly the
        O(R x cost) the multi-target API exists to avoid.
        """
        d = generalize_multi_batch(
            imgs[idxs], res_list, n_jobs=cfg.n_jobs,
            equal_fidelity=cfg.equal_fidelity,
            exaggeration_gamma=cfg.exaggeration_gamma,
            displacement_lambda=cfg.displacement_lambda,
            coherence_floor=cfg.coherence_floor)
        return np.concatenate([d[r].reshape(len(idxs), -1) for r in res_list], axis=1)

    multi = {}
    for split, idxs, imgs in (("train", tidx, src.images("train")),
                              ("val", np.arange(len(y["val"])), src.images("val")),
                              ("test", teidx, src.images("test"))):
        arr = np.asarray(cached(
            P.cache / f"topfer_{split}_{key}_{len(idxs)}.npy",
            lambda i=idxs, im=imgs: _build_multi(i, im)))
        offs, cur = {}, 0
        for r in res_list:
            sz = spec.n * r * r
            offs[r] = arr[:, cur:cur + sz].reshape(len(idxs), spec.n, r, r)
            cur += sz
        multi[split] = offs
    tensors_by_res = {r: {s: multi[s][r] for s in ("train", "val", "test")}
                      for r in res_list}
    top_labels = {"train": y["train"][tidx], "val": y["val"],
                  "test": y["test"][teidx]}
    T.mark("multi-resolution tensors built")

    imp = TOP.structure_importance(np.asarray(gen["train"])[pidx], y["train"][pidx],
                                   spec)
    order = [k for k, _ in sorted(imp.items(), key=lambda kv: (-kv[1], kv[0]))]
    out["structure_importance"] = imp
    out["selection_order"] = order
    print(f"  selection order (train-only probe): {order}")

    points = TOP.run_sweep(tensors_by_res, top_labels, order, cfg, n_classes)
    law = TOP.fit_scale_law(points)
    out["topfer"] = {"points": [asdict(p) for p in points], "law": law}
    print(f"  fitted K* ∝ r^{law['alpha']:.3f}   (Töpfer predicts r^0.50)")
    T.mark("Töpfer sweep done")

    # ============================================================= ISIC check
    isic_res = {"available": False, "reason": "skipped by config"}
    if cfg.run_isic_validation:
        from . import isic
        isic_res = isic.validate(target=cfg.target_size, native=cfg.native_size,
                                 limit=200 if cfg.fast else 400,
                                 equal_fidelity=cfg.equal_fidelity)
        print(f"  ISIC validation available = {isic_res.get('available')}")
    out["isic_validation"] = isic_res
    T.mark("ISIC validation")

    # ================================================================ figures
    print("\n-- drawing plates")
    figs: List[Path] = []
    rng = np.random.default_rng(7)
    show = []
    for c in range(n_classes):
        cand = np.flatnonzero(y["test"] == c)
        if len(cand):
            show.append(int(rng.choice(cand)))
    show = show[:5]

    figs.append(F.fig_legend(P.figures))
    figs.append(F.fig_three_regimes(P.figures, src.images("test"),
                                    test_gen[:, :3], test_gen, spec,
                                    y["test"], show, cfg.target_size))
    i0 = show[0]
    _, inter = generalize(src.images("test")[i0], target=cfg.target_size,
                          equal_fidelity=cfg.equal_fidelity,
                          exaggeration_gamma=cfg.exaggeration_gamma,
                          displacement_lambda=cfg.displacement_lambda,
                          coherence_floor=cfg.coherence_floor,
                          return_intermediates=True)
    figs.append(F.fig_symbol_atlas(P.figures, src.images("test")[i0],
                                   test_gen[i0], inter, spec,
                                   CLASS_SHORT[int(y["test"][i0])], cfg.target_size))
    figs.append(F.fig_source_diagram(P.figures, src.images("test"), test_gen, spec,
                                     y["test"], show[:4], out["nyquist"],
                                     cfg.target_size))

    fin = np.flatnonzero(np.isfinite(ita["test"]))
    ex = {"dark": int(fin[np.argmin(ita["test"][fin])]),
          "light": int(fin[np.argmax(ita["test"][fin])])}
    figs.append(F.fig_mercator(P.figures, mres, bin_labels, dyn,
                               src.images("test"), ita["test"], ex))
    figs.append(F.fig_reliability(P.figures,
                                 {k: extras[k]["reliability"] for k in extras}, evals))
    figs.append(F.fig_topfer(P.figures, law))
    figs.append(F.fig_certificate(P.figures, curves))
    figs.append(F.fig_rare_class(P.figures, evals, rare, src.class_counts("train")))
    figs.append(F.fig_ablation(P.figures, ab))
    figs.append(F.fig_isic(P.figures, isic_res))
    figs.append(F.fig_headline(P.figures, evals, law, merc_summary, curves,
                               out["nyquist"]))
    out["figures"] = [str(f.name) for f in figs]
    T.mark(f"{len(figs)} plates drawn")

    # ================================================================ results
    from .report import write_all
    out["timing"] = dict(T.marks)
    write_all(out, cfg)
    T.mark("results + report written")
    print("\nDone. Bundle:", P.out)
    return out
