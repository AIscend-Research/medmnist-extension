"""The generalization pipeline: native image in, generalized target sheet out.

    survey (native res)  ->  symbolize  ->  aggregate  ->  simplify
                         ->  exaggerate ->  displace   ->  stack with RGB
                         ->  append source diagram

The output is an (C, target, target) float32 tensor in [0, 1]. Channel identity
is declared once, in `channel_spec()`, and every downstream consumer — models,
ablations, figures, the report — reads it from there.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from . import operators as ops
from .filterbanks import STRUCTURES, STRUCTURE_NAMES, survey
from .sufficiency import source_diagram


# --------------------------------------------------------------------------
# Channel layout
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelSpec:
    names: List[str]
    groups: Dict[str, List[int]]     # logical group -> channel indices

    @property
    def n(self) -> int:
        return len(self.names)

    def idx(self, *groups: str) -> List[int]:
        out: List[int] = []
        for g in groups:
            out.extend(self.groups[g])
        return sorted(out)

    def complement(self, *groups: str) -> List[int]:
        drop = set(self.idx(*groups))
        return [i for i in range(self.n) if i not in drop]


def channel_spec(structures: Sequence[str] = tuple(STRUCTURE_NAMES),
                 with_source_diagram: bool = True) -> ChannelSpec:
    names: List[str] = ["rgb_r", "rgb_g", "rgb_b"]
    groups: Dict[str, List[int]] = {"rgb": [0, 1, 2]}
    spec_by_name = {s.name: s for s in STRUCTURES}

    for sname in structures:
        s = spec_by_name[sname]
        idxs = [len(names)]
        names.append(f"{sname}::density")
        if s.oriented:
            idxs += [len(names), len(names) + 1]
            names += [f"{sname}::cos2theta", f"{sname}::sin2theta"]
        if sname == "dots_globules":
            idxs.append(len(names))
            names.append(f"{sname}::scale")
        groups[sname] = idxs

    groups["symbols"] = sorted(i for s in structures for i in groups[s])

    if with_source_diagram:
        start = len(names)
        names += ["source::sampling", "source::contrast", "source::certificate"]
        groups["source_diagram"] = [start, start + 1, start + 2]
    else:
        groups["source_diagram"] = []
    return ChannelSpec(names, groups)


DEFAULT_SPEC = channel_spec()


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------
def generalize(rgb: np.ndarray,
               target: int = 28,
               structures: Sequence[str] = tuple(STRUCTURE_NAMES),
               equal_fidelity: bool = True,
               absolute_scales: Dict[str, float] | None = None,
               exaggeration_gamma: float = 0.55,
               displacement_lambda: float = 0.6,
               coherence_floor: float = 0.15,
               with_source_diagram: bool = True,
               return_intermediates: bool = False):
    """Generalize one native-resolution RGB image onto a `target` grid."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    native = rgb.shape[0]

    # ---- 1. SURVEY at native resolution (no downsampling has happened yet)
    surv = survey(rgb, native_size=native, equal_fidelity=equal_fidelity,
                  absolute_scales=absolute_scales)

    # ---- 2. AGGREGATION of the base image (this alone is the naive baseline)
    base = ops.aggregate(rgb, target)                       # (t, t, 3)
    base = np.transpose(base, (2, 0, 1))

    # ---- 3. SYMBOLIZATION + AGGREGATION of every retained structure
    dens_planes: List[np.ndarray] = []
    extra: Dict[str, List[np.ndarray]] = {}
    spec_by_name = {s.name: s for s in STRUCTURES}

    for sname in structures:
        s = spec_by_name[sname]
        r = surv[sname]
        d = ops.aggregate(r["density"], target)
        dens_planes.append(d)

        cols: List[np.ndarray] = []
        if s.oriented:
            # SIMPLIFICATION happens in double-angle space, after pooling, so
            # the coherence test is applied to the *pooled* bearing estimate.
            c2, s2 = ops.aggregate_orientation(r["theta"], r["density"], target)
            pooled_coh = np.hypot(c2, s2)
            c2, s2 = ops.simplify_orientation(c2, s2, pooled_coh, coherence_floor)
            # map [-1, 1] -> [0, 1] so every channel shares one range
            cols += [(c2 + 1.0) * 0.5, (s2 + 1.0) * 0.5]
        if sname == "dots_globules":
            cols.append(ops.aggregate(r["scale"], target))
        extra[sname] = cols

    # ---- 4. DISPLACEMENT across co-located symbols, then EXAGGERATION
    dens = np.stack(dens_planes, 0) if dens_planes else np.zeros((0, target, target), np.float32)
    dens_disp = ops.displace(dens, displacement_lambda) if dens.size else dens
    dens_final = ops.exaggerate(dens_disp, exaggeration_gamma) if dens.size else dens

    # ---- 5. assemble in the declared channel order
    planes: List[np.ndarray] = [base[0], base[1], base[2]]
    for i, sname in enumerate(structures):
        planes.append(dens_final[i])
        planes.extend(extra[sname])

    # ---- 6. SOURCE DIAGRAM appended last
    sd = None
    if with_source_diagram:
        sd = source_diagram(rgb, target, native_size=native)
        squash = lambda m: (1.0 / (1.0 + np.exp(-m))).astype(np.float32)
        planes.append(squash(sd["sampling"].min(axis=0)))
        planes.append(squash(sd["contrast"].min(axis=0)))
        planes.append(sd["certificate"])

    out = np.stack(planes, 0).astype(np.float32)
    out = np.clip(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    if return_intermediates:
        return out, {"survey": surv, "source": sd,
                     "density_pooled": dens, "density_final": dens_final}
    return out


# --------------------------------------------------------------------------
def generalize_multi(rgb: np.ndarray,
                     targets: Sequence[int],
                     structures: Sequence[str] = tuple(STRUCTURE_NAMES),
                     equal_fidelity: bool = True,
                     absolute_scales: Dict[str, float] | None = None,
                     exaggeration_gamma: float = 0.55,
                     displacement_lambda: float = 0.6,
                     coherence_floor: float = 0.15,
                     with_source_diagram: bool = True) -> Dict[int, np.ndarray]:
    """Generalize one image onto several target grids, surveying only once.

    The survey is the expensive step (filter banks at native resolution) and it
    is independent of the target grid — a chart's source data does not change
    when you choose a different scale to publish at. Only aggregation onward is
    per-target. This makes the Töpfer resolution sweep cost ~1.2x one
    resolution instead of Nx, which is what makes the sweep affordable at all.

    Results are bit-identical to calling `generalize` once per target.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    native = rgb.shape[0]

    surv = survey(rgb, native_size=native, equal_fidelity=equal_fidelity,
                  absolute_scales=absolute_scales)
    spec_by_name = {s.name: s for s in STRUCTURES}
    out: Dict[int, np.ndarray] = {}

    for target in targets:
        base = np.transpose(ops.aggregate(rgb, target), (2, 0, 1))
        dens_planes, extra = [], {}
        for sname in structures:
            s = spec_by_name[sname]
            r = surv[sname]
            dens_planes.append(ops.aggregate(r["density"], target))
            cols = []
            if s.oriented:
                c2, s2 = ops.aggregate_orientation(r["theta"], r["density"], target)
                c2, s2 = ops.simplify_orientation(c2, s2, np.hypot(c2, s2),
                                                  coherence_floor)
                cols += [(c2 + 1.0) * 0.5, (s2 + 1.0) * 0.5]
            if sname == "dots_globules":
                cols.append(ops.aggregate(r["scale"], target))
            extra[sname] = cols

        dens = (np.stack(dens_planes, 0) if dens_planes
                else np.zeros((0, target, target), np.float32))
        if dens.size:
            dens = ops.exaggerate(ops.displace(dens, displacement_lambda),
                                  exaggeration_gamma)

        planes = [base[0], base[1], base[2]]
        for i, sname in enumerate(structures):
            planes.append(dens[i])
            planes.extend(extra[sname])

        if with_source_diagram:
            sd = source_diagram(rgb, target, native_size=native)
            squash = lambda m: (1.0 / (1.0 + np.exp(-m))).astype(np.float32)
            planes += [squash(sd["sampling"].min(axis=0)),
                       squash(sd["contrast"].min(axis=0)), sd["certificate"]]

        t = np.stack(planes, 0).astype(np.float32)
        out[target] = np.clip(np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
    return out


def generalize_multi_batch(rgbs, targets: Sequence[int], n_jobs: int = 0,
                           **kw) -> Dict[int, np.ndarray]:
    n = len(rgbs)
    if n_jobs in (0, None):
        n_jobs = max(1, (os.cpu_count() or 2))

    def _run(idx):
        parts = [generalize_multi(rgbs[i], targets, **kw) for i in idx]
        return {t: np.stack([p[t] for p in parts]) for t in targets}

    if n_jobs == 1:
        return _run(np.arange(n))
    try:
        from joblib import Parallel, delayed
    except ImportError:
        return _run(np.arange(n))
    chunks = [c for c in np.array_split(np.arange(n), n_jobs * 4) if len(c)]
    parts = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_run)(c) for c in chunks)
    return {t: np.concatenate([p[t] for p in parts], axis=0) for t in targets}


# --------------------------------------------------------------------------
def generalize_batch(rgbs, target=28, n_jobs=0, **kw) -> np.ndarray:
    """Generalize a stack of images, in parallel when joblib is available.

    `rgbs` may be a numpy array (N, H, W, 3) or any indexable sequence.
    """
    n = len(rgbs)
    if n_jobs in (0, None):
        import os
        n_jobs = max(1, (os.cpu_count() or 2))
    try:
        from joblib import Parallel, delayed
    except ImportError:
        return np.stack([generalize(rgbs[i], target=target, **kw) for i in range(n)])

    if n_jobs == 1:
        return np.stack([generalize(rgbs[i], target=target, **kw) for i in range(n)])

    chunks = np.array_split(np.arange(n), n_jobs * 4)
    chunks = [c for c in chunks if len(c)]

    def _run(idx):
        return np.stack([generalize(rgbs[i], target=target, **kw) for i in idx])

    parts = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_run)(c) for c in chunks)
    return np.concatenate(parts, axis=0)


# --------------------------------------------------------------------------
def raw_structure_maps(rgb: np.ndarray, target: int = 28,
                       structures: Sequence[str] = tuple(STRUCTURE_NAMES)
                       ) -> Dict[str, np.ndarray]:
    """Un-normalised native structure density, area-pooled to the target grid.

    This is the *reference measurement* the Mercator analysis scores every
    regime against. It deliberately skips every normalisation choice, so it is
    not a restatement of any regime's own preprocessing — otherwise the
    retention experiment would be circular.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    surv = survey(rgb, native_size=rgb.shape[0], raw=True)
    return {s: ops.aggregate(surv[s]["density"], target) for s in structures}


def raw_structure_maps_batch(rgbs, target: int = 28, n_jobs: int = 0,
                             structures: Sequence[str] = tuple(STRUCTURE_NAMES)
                             ) -> Dict[str, np.ndarray]:
    n = len(rgbs)
    if n_jobs in (0, None):
        n_jobs = max(1, (os.cpu_count() or 2))

    def _run(idx):
        parts = [raw_structure_maps(rgbs[i], target, structures) for i in idx]
        return {s: np.stack([p[s] for p in parts]) for s in structures}

    if n_jobs == 1:
        return _run(np.arange(n))
    try:
        from joblib import Parallel, delayed
    except ImportError:
        return _run(np.arange(n))
    chunks = [c for c in np.array_split(np.arange(n), n_jobs * 4) if len(c)]
    parts = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_run)(c) for c in chunks)
    return {s: np.concatenate([p[s] for p in parts], axis=0) for s in structures}


# --------------------------------------------------------------------------
def naive_resize(rgb: np.ndarray, target: int = 28) -> np.ndarray:
    """The baseline, for completeness: area-mean pooling of RGB and nothing else.

    Matches how MedMNIST-style benchmarks are built. Returned as (3, t, t) so it
    is drop-in comparable with `generalize`.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    return np.transpose(ops.aggregate(rgb, target), (2, 0, 1)).astype(np.float32)
