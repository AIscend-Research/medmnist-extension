#!/usr/bin/env python
"""CLI entry point. `python scripts/run_all.py --fast`"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cartomnist import Config
from cartomnist.pipeline import run_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true",
                    help="subsample and shorten everything (~30 GPU-minutes)")
    ap.add_argument("--root", type=Path, default=None,
                    help="output root (default: /kaggle/working or cwd)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--target", type=int, default=28, help="benchmark grid size")
    ap.add_argument("--no-isic", action="store_true")
    ap.add_argument("--jobs", type=int, default=0, help="0 = all cores")
    ap.add_argument("--ita-bins", default="fitzpatrick", choices=["fitzpatrick", "tertile"])
    a = ap.parse_args()

    cfg = Config()
    if a.root:
        cfg.paths.root = a.root
    cfg.paths.mkdirs()
    if a.fast:
        cfg.apply_fast()
    if a.seeds:
        cfg.seeds = a.seeds
    cfg.device = a.device
    cfg.target_size = a.target
    cfg.run_isic_validation = not a.no_isic
    cfg.n_jobs = a.jobs
    cfg.ita_bins = a.ita_bins

    run_all(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
