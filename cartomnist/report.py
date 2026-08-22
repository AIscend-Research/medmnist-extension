"""Write the deliverable: JSON, CSVs, a standalone HTML report, and one zip.

The zip contains only analysis, results and figures — never the cached tensors
or model weights, which is why `Paths.cache` lives outside `Paths.out`.
"""
from __future__ import annotations

import base64
import json
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np

from .data import CLASS_SHORT


def _json_safe(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# --------------------------------------------------------------------------
def write_csvs(out: Dict, results_dir: Path) -> List[Path]:
    import csv
    written: List[Path] = []

    # --- headline table
    p = results_dir / "benchmark_table.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "n_test", "n_params", "macro_auc", "auc_lo", "auc_hi",
                    "accuracy", "balanced_accuracy", "rare_recall", "rare_auc",
                    "ece", "ece_uncalibrated", "mce", "brier", "temperature",
                    "ita_gap"])
        for k, e in out["evaluations"].items():
            w.writerow([k, e["n"], int(e["extra"].get("n_params", 0)),
                        e["macro_auc"], e["macro_auc_lo"], e["macro_auc_hi"],
                        e["accuracy"], e["balanced_accuracy"], e["rare_recall"],
                        e["rare_auc"], e["ece"], e["ece_uncalibrated"], e["mce"],
                        e["brier"], e["temperature"], e["ita_gap"]])
    written.append(p)

    # --- per class
    p = results_dir / "per_class.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "class", "recall", "auc", "train_n"])
        counts = out.get("class_counts", {}).get("train", {})
        for k, e in out["evaluations"].items():
            for i, c in enumerate(CLASS_SHORT[:len(e["per_class_recall"])]):
                w.writerow([k, c, e["per_class_recall"][i], e["per_class_auc"][i],
                            counts.get(c, "")])
    written.append(p)

    # --- ITA stratified
    p = results_dir / "ita_stratified.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "ita_stratum", "balanced_accuracy", "macro_auc", "ece"])
        for k, e in out["evaluations"].items():
            for lab, v in e["ita_balanced_accuracy"].items():
                w.writerow([k, lab, v, e["ita_auc"].get(lab, ""),
                            e["ita_ece"].get(lab, "")])
    written.append(p)

    # --- Mercator
    p = results_dir / "mercator_retention.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "structure", "ita_stratum", "retention_r2", "sem", "n",
                    "spread_across_strata", "slope_per_10deg_ita", "slope_p"])
        for m in out["mercator"]["per_structure"]:
            for lab, v in m["per_bin_mean"].items():
                w.writerow([m["regime"], m["structure"], lab, v,
                            m["per_bin_sem"].get(lab, ""), m["n_per_bin"].get(lab, ""),
                            m["spread"], m["slope_per_10deg"], m["slope_p"]])
    written.append(p)

    # --- Töpfer
    p = results_dir / "topfer_sweep.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["resolution", "k", "n_channels", "macro_auc", "balanced_accuracy"])
        for pt in out["topfer"]["points"]:
            w.writerow([pt["resolution"], pt["k"], pt["n_channels"],
                        pt["macro_auc"], pt["balanced_accuracy"]])
    written.append(p)

    # --- ablations
    p = results_dir / "ablations.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ablation", "macro_auc", "rare_recall", "balanced_accuracy", "ece"])
        for k, v in out["ablations"].items():
            w.writerow([k, v.get("macro_auc"), v.get("rare_recall"),
                        v.get("balanced_accuracy"), v.get("ece")])
    written.append(p)

    # --- Nyquist audit
    p = results_dir / "nyquist_audit.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["structure", "period_native_px", "target_cell_px",
                    "nyquist_ratio", "survives_naive_resize"])
        for k, v in out["nyquist"].items():
            w.writerow([k, v["period_native_px"], v["target_cell_px"],
                        v["nyquist_ratio"], v["survives_naive_resize"]])
    written.append(p)
    return written


# --------------------------------------------------------------------------
_CSS = """
:root{--paper:#f5f0e6;--deep:#ebe3d3;--ink:#14202b;--ink2:#54606b;--ink3:#8b939b;
--rule:#c9bfa8;--blue:#2a78d6;--orange:#eb6834;--aqua:#1baf7a;--red:#e34948;
--yellow:#eda100;}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
font-family:"Iowan Old Style",Georgia,"Times New Roman",serif;line-height:1.62;}
.wrap{max-width:1180px;margin:0 auto;padding:56px 26px 96px;}
header{border-top:3px solid var(--ink);border-bottom:1px solid var(--rule);
padding:26px 0 22px;margin-bottom:38px;}
h1{font-size:2.35rem;margin:.1em 0 .18em;letter-spacing:-.01em;line-height:1.14;}
h2{font-size:1.32rem;margin:2.6em 0 .5em;padding-bottom:.3em;
border-bottom:1px solid var(--rule);}
h3{font-size:1.02rem;margin:1.7em 0 .4em;color:var(--ink2);}
.plate{font-size:.7rem;letter-spacing:.22em;text-transform:uppercase;color:var(--ink3);}
.lede{font-size:1.04rem;color:var(--ink2);font-style:italic;max-width:74ch;}
figure{margin:1.8em 0 2.4em;}
figure img{width:100%;height:auto;border:1px solid var(--rule);background:var(--paper);}
figcaption{font-size:.83rem;color:var(--ink2);margin-top:.7em;max-width:88ch;}
.tablewrap{overflow-x:auto;border:1px solid var(--rule);background:var(--deep);}
table{border-collapse:collapse;width:100%;font-size:.85rem;
font-variant-numeric:tabular-nums;}
th,td{padding:.5em .75em;text-align:right;border-bottom:1px solid var(--rule);
white-space:nowrap;}
th:first-child,td:first-child{text-align:left;}
thead th{font-size:.68rem;letter-spacing:.09em;text-transform:uppercase;
color:var(--ink3);border-bottom:1.5px solid var(--ink);}
tbody tr:last-child td{border-bottom:none;}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));
gap:14px;margin:1.6em 0 2.2em;}
.tile{background:var(--deep);border:1px solid var(--rule);padding:18px 16px;}
.tile .v{font-size:1.85rem;font-weight:700;line-height:1.1;}
.tile .l{font-size:.78rem;color:var(--ink2);margin-top:.45em;}
.tile .s{font-size:.7rem;color:var(--ink3);margin-top:.3em;font-style:italic;}
.contract{background:var(--deep);border:1px solid var(--rule);padding:16px 18px;
margin:.8em 0;}
.contract.bad{background:#f2e5e3;border-color:var(--red);}
.contract .op{font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;
font-weight:700;}
.contract .an{font-style:italic;color:var(--ink2);font-size:.87rem;margin:.4em 0 .7em;}
.k{font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;font-weight:700;}
.k.p{color:var(--aqua);} .k.s{color:var(--red);}
.contract ul{margin:.25em 0 .8em 1.1em;padding:0;font-size:.85rem;}
.cite{font-size:.72rem;color:var(--ink3);}
.caveat{border-left:3px solid var(--red);padding:.5em 0 .5em 1em;margin:1.4em 0;
font-size:.9rem;color:var(--ink2);}
footer{margin-top:64px;padding-top:20px;border-top:1px solid var(--rule);
font-size:.78rem;color:var(--ink3);}
code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.85em;
background:var(--deep);padding:.08em .35em;}
@media (prefers-color-scheme:dark){
:root{--paper:#171512;--deep:#211e1a;--ink:#f2ece0;--ink2:#c3bbab;--ink3:#8d857a;
--rule:#3c372f;--blue:#3987e5;--orange:#d95926;--aqua:#199e70;--red:#e66767;
--yellow:#c98500;}
figure img{background:#f5f0e6;}
.contract.bad{background:#2a1f1d;}
}
"""


def _img_tag(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img alt="{path.stem}" src="data:image/png;base64,{b64}">'


def _table(headers: List[str], rows: List[List]) -> str:
    h = "".join(f"<th>{x}</th>" for x in headers)
    b = "".join("<tr>" + "".join(
        f"<td>{'' if v is None else (f'{v:.4f}' if isinstance(v, float) else v)}</td>"
        for v in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def write_html(out: Dict, figures_dir: Path, path: Path) -> Path:
    ev = out["evaluations"]
    gr = out.get("gap_recovery", {})
    law = out["topfer"]["law"]
    ms = out["mercator"]["summary"]
    nyq = out["nyquist"]
    lost = sum(1 for v in nyq.values() if not v["survives_naive_resize"])

    def f(x, d=3):
        try:
            return f"{float(x):.{d}f}"
        except Exception:                                # noqa: BLE001
            return "—"

    tiles = [
        (f"{float(gr.get('fraction_of_224_gap_recovered_at_28', float('nan'))):.0%}"
         if np.isfinite(gr.get("fraction_of_224_gap_recovered_at_28", np.nan))
         else "no gap",
         "of the rare-class gap that 224² buys, recovered at 28²",
         f"naive {f(ev['naive']['rare_recall'])} → generalized "
         f"{f(ev['generalized']['rare_recall'])} → 224 {f(ev['native224']['rare_recall'])}",
         "orange"),
        (f"{lost}/{len(nyq)}", "diagnostic structures that cannot survive 224²→28² "
         "resampling at any quality", "they are symbolized instead", "red"),
        (f"r^{f(law.get('alpha'), 2)}", "measured symbol-budget scale law",
         "Töpfer's radical law predicts r^0.50", "blue"),
        (f"{f(ms.get('naive', {}).get('mean_spread'))} → "
         f"{f(ms.get('generalized_ef', {}).get('mean_spread'))}",
         "retention spread across ITA strata, naive → generalized",
         "0 = equal fidelity by construction", "aqua"),
        (f(ev["generalized"]["ece"], 4), "ECE of the generalized 28² benchmark",
         f"naive 28² = {f(ev['naive']['ece'], 4)}", "orange"),
        (f(out["certificate_vs_confidence"].get("source certificate")),
         "AURC of the label-free source certificate",
         f"model confidence = {f(out['certificate_vs_confidence'].get('softmax confidence'))}",
         "blue"),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="v" style="color:var(--{c})">{v}</div>'
        f'<div class="l">{lab}</div><div class="s">{s}</div></div>'
        for v, lab, s, c in tiles)

    # legend cards
    cards = []
    for group, items in out["legend"].items():
        for c in items:
            bad = c["operator"] == "naive_resize"
            cards.append(
                f'<div class="contract{" bad" if bad else ""}">'
                f'<div class="op">{c["operator"].replace("_", " ")}</div>'
                f'<div class="an">{c["cartographic_analogue"]}</div>'
                f'<div class="k p">Preserves</div><ul>'
                + "".join(f"<li>{x}</li>" for x in c["preserves"]) + "</ul>"
                '<div class="k s">Sacrifices</div><ul>'
                + "".join(f"<li>{x}</li>" for x in c["sacrifices"]) + "</ul>"
                f'<div class="cite">Invariant: {c["invariant"]}<br>{c["citation"]}</div>'
                "</div>")

    bench_rows = [[k, e["n"], int(e["extra"].get("n_params", 0)),
                   f"{e['macro_auc']:.4f} [{e['macro_auc_lo']:.3f}, {e['macro_auc_hi']:.3f}]",
                   e["balanced_accuracy"], e["rare_recall"], e["ece"],
                   e["ece_uncalibrated"], e["brier"], e["ita_gap"]]
                  for k, e in ev.items()]

    nyq_rows = [[k, v["period_native_px"], v["target_cell_px"], v["nyquist_ratio"],
                 "yes" if v["survives_naive_resize"] else "NO — lost"]
                for k, v in nyq.items()]

    ab_rows = [[k, v.get("macro_auc"), v.get("rare_recall"),
                v.get("balanced_accuracy"), v.get("ece")]
               for k, v in out["ablations"].items()]

    merc_rows = [[m["regime"], m["structure"], m["spread"],
                  m["slope_per_10deg"], m["slope_p"]]
                 for m in out["mercator"]["per_structure"]]

    figs = sorted(figures_dir.glob("fig*.png"))
    fig_captions = {
        "fig00_headline": "Frontispiece — the standard, as one sheet.",
        "fig01_legend": "Plate I — the legend: named operators and their fidelity contracts.",
        "fig02_three_regimes": "Plate II — three ways to publish the same lesion at 28×28.",
        "fig03_symbol_atlas": "Plate III — survey sheet: instrument response to symbol.",
        "fig04_source_diagram": "Plate IV — the source diagram and the Nyquist audit.",
        "fig05_mercator": "Plate V — the Mercator result: non-uniform loss by pigmentation.",
        "fig06_reliability": "Plate VI — reliability diagrams, one per regime.",
        "fig07_topfer": "Plate VII — Töpfer's radical law as a benchmark scale law.",
        "fig08_certificate": "Plate VIII — the certificate as an abstention model.",
        "fig09_rare_class": "Plate IX — rare-class recovery.",
        "fig10_ablation": "Plate X — ablations, including the shortcut test.",
        "fig11_isic_validation": "Plate XI — do the symbols track expert annotation?",
    }
    fig_html = "".join(
        f"<figure>{_img_tag(p)}<figcaption>{fig_captions.get(p.stem, p.stem)}</figcaption></figure>"
        for p in figs)

    isic = out.get("isic_validation", {})
    isic_note = (isic.get("verdict") if isic.get("available")
                 else f"<strong>Not run.</strong> {isic.get('reason','')} "
                      "The symbol channels are reported as measured-but-unvalidated.")

    shortcut = out["ablations"].get("symbols shuffled across images (shortcut test)", {})
    full = out["ablations"].get("generalized (full)", {})
    drop = (full.get("macro_auc", np.nan) - shortcut.get("macro_auc", np.nan))

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cartographic Generalization for MedMNIST</title><style>{_CSS}</style></head>
<body><div class="wrap">
<header>
<div class="plate">Benchmark audit · DermaMNIST · {out['config']['native_size']}² → {out['config']['target_size']}²</div>
<h1>Stop resizing medical images.<br>Generalize them like a map.</h1>
<p class="lede">A mapmaker reducing 1:10,000 to 1:1,000,000 faces exactly this problem, and
has for two centuries. The answer was never to photoreduce the sheet. It was a vocabulary of
named operators with explicit contracts, a scale bar in physical units, and — on nautical
charts since the nineteenth century — a source diagram showing where the survey cannot be
trusted. This report applies that discipline to a medical imaging benchmark.</p>
</header>

<div class="tiles">{tile_html}</div>

<h2>1 · The benchmark table</h2>
{_table(["regime", "n test", "params", "macro AUC [95% CI]", "bal. acc",
         "rare recall", "ECE", "ECE (uncal.)", "Brier", "ITA gap"], bench_rows)}
<p class="cite">Temperature fitted on validation only; probabilities averaged across
{int(ev['generalized']['extra'].get('n_seeds', 1))} seed(s). Rare classes:
{', '.join(out['rare_classes'])}.</p>

<h2>2 · The Nyquist audit — what resizing cannot keep</h2>
{_table(["structure", "period (native px)", "cell width (px)", "p / 2Δ",
         "survives naive resize?"], nyq_rows)}
<p class="cite">Δ is the benchmark cell width in native pixels. A structure whose period
falls below 2Δ is below the Nyquist limit of the target grid: no resampling kernel, at any
quality setting, can represent it. Symbolization does not beat the sampling theorem — it
moves the measurement upstream of the reduction, which is a different thing entirely.</p>

<h2>3 · The Mercator result — is the resizing policy neutral?</h2>
<p class="lede">Retention is measured without a classifier: a ridge probe fitted on train
reconstructs each structure's native density map from what a regime kept, scored per test
image, stratified by individual typology angle (ITA). Spread is the max − min retention
across ITA strata for that structure and regime; the slope and its permutation p-value ask
whether retention trends with ITA at all.</p>
{_table(["regime", "structure", "spread across ITA strata", "slope (Δ retention / 10° ITA)",
         "permutation p"], merc_rows)}
<p class="cite">naive is expected to show a negative, significant slope (retention falls with
decreasing ITA — the Mercator distortion); generalized (equal fidelity) is expected to be flat
by construction; generalized (absolute scale) is expected to recover the distortion, showing
equal fidelity is a design choice inside the framework and not automatic.</p>

<h2>4 · The legend</h2>
<p class="lede">Every operator that touches the pixels, and what it admits to destroying.</p>
{''.join(cards)}

<h2>5 · Ablations, including the one that could sink this</h2>
{_table(["ablation", "macro AUC", "rare recall", "bal. acc", "ECE"], ab_rows)}
<div class="caveat"><strong>Shortcut test.</strong> Shuffling the symbol channels across
test images breaks their alignment with the underlying picture. Macro AUC falls by
{f(drop, 4)}. If that number were near zero, the model would have been reading the symbols
as a per-image fingerprint rather than as measurement, and the entire result would be an
artefact.</div>
<div class="caveat"><strong>Filter-bank validity.</strong> {isic_note}</div>

<h2>6 · Plates</h2>
{fig_html}

<h2>7 · What this proposes</h2>
<p>A medical imaging benchmark should ship:</p>
<ol>
<li><strong>Named generalization operators with fidelity contracts</strong> — what was
preserved, what was sacrificed, and a checkable invariant for each.</li>
<li><strong>A scale bar in physical units</strong> — millimetres of tissue per pixel, so a
reader can tell whether a structure is representable before training anything on it.</li>
<li><strong>A reliability diagram</strong> — and its ECE, pre- and post-calibration.</li>
</ol>
<p>Every existing MedMNIST-style dataset is auditable against that standard today. The
source diagram is the abstention model, made explicit and shipped with the data rather than
refitted per model. And the equity result is not a bias audit bolted on afterwards: it falls
out of the framework, because a generalization policy is precisely a decision about whose
features survive.</p>
<p class="cite">Harley, "Deconstructing the Map" (1989); Töpfer &amp; Pillewizer (1966);
McMaster &amp; Shea (1992); IHO S-4/S-57 source diagrams; Del Bino &amp; Bernerd (2013)
for ITA; Argenziano et al. (1998) for the structure vocabulary; Codella et al. (2018) /
Tschandl et al. (2018) for ISIC and HAM10000; Yang et al. (2023) for MedMNIST v2.</p>

<footer>Generated by <code>cartomnist</code>. Configuration and every number in this report
are reproduced in <code>results/results.json</code>; tables are also written as CSV.
This report contains no model weights and no cached tensors.</footer>
</div></body></html>"""
    path.write_text(html, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
def write_all(out: Dict, cfg) -> Dict[str, Path]:
    P = cfg.paths
    (P.results).mkdir(parents=True, exist_ok=True)

    jpath = P.results / "results.json"
    jpath.write_text(json.dumps(out, indent=2, default=_json_safe))
    (P.results / "legend.json").write_text(
        json.dumps(out["legend"], indent=2, default=_json_safe))
    (P.results / "config.json").write_text(
        json.dumps(out["config"], indent=2, default=_json_safe))
    write_csvs(out, P.results)

    hpath = write_html(out, P.figures, P.out / "report.html")

    zpath = P.root / "cartomnist_report.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(P.out.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(P.out.parent))
    print(f"\n  report   : {hpath}")
    print(f"  results  : {P.results}")
    print(f"  figures  : {P.figures}")
    print(f"  BUNDLE   : {zpath}  ({zpath.stat().st_size / 1e6:.1f} MB)")
    return {"json": jpath, "html": hpath, "zip": zpath}
