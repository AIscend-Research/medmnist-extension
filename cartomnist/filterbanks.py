"""Hand-designed filter banks for the five validated dermoscopic structures.

No training happens here. These are the "survey instruments": cheap, fixed,
auditable operators run at NATIVE resolution, whose outputs are later
symbolized onto the 28x28 sheet.

Structure vocabulary follows the dermoscopy literature that ISIC 2018 Task 2
annotates (Argenziano's 7-point checklist and successors):

    pigment_network   quasi-periodic reticular mesh   -> Gabor bank
    dots_globules     small round dark blobs          -> scale-normalised LoG
    streaks           radial linear/branched          -> oriented ridge (Sato)
    vessels           red curvilinear                 -> Sato on a redness map
    blue_white_veil   diffuse structureless bluish    -> Lab opponent + texture

Every function takes float RGB in [0, 1], shape (H, W, 3), and returns dense
response maps at the same (H, W) with values in [0, 1] after normalisation.

Each structure declares a *characteristic period* in native pixels. That number
is what the source diagram checks against the Nyquist limit of the target grid,
and it is the only place in the codebase where dermatological prior knowledge is
hard-coded — so it is stated explicitly and cited.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2lab
from skimage.filters import gabor_kernel, sato

# --------------------------------------------------------------------------
# Structure specification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StructureSpec:
    """Prior knowledge about one diagnostic structure, stated in physical units.

    `period_px_at_224` is the characteristic spatial period of the structure
    when a dermoscopic lesion field is rendered at 224x224. It is derived from
    typical dermoscopic geometry: a ~20 mm field of view at 224 px gives
    ~11.2 px/mm, and the literature quantifies these structures in mm.
    """
    name: str
    period_px_at_224: float
    contrast_floor: float      # minimum band-limited RMS contrast (Lab L* units/100)
    oriented: bool
    note: str
    normalize: str = "robust"  # "robust" (band-pass energy) | "membership" ([0,1] already)


STRUCTURES: Tuple[StructureSpec, ...] = (
    StructureSpec(
        "pigment_network", period_px_at_224=6.0, contrast_floor=0.012, oriented=True,
        note="reticular mesh, hole-to-hole spacing ~0.4-0.7 mm",
    ),
    StructureSpec(
        "dots_globules", period_px_at_224=4.0, contrast_floor=0.020, oriented=False,
        note="globules 0.1-0.5 mm; dots < 0.1 mm",
    ),
    StructureSpec(
        "streaks", period_px_at_224=10.0, contrast_floor=0.015, oriented=True,
        note="radial streaming / pseudopods at the periphery, ~1 mm wide",
    ),
    StructureSpec(
        "vessels", period_px_at_224=5.0, contrast_floor=0.010, oriented=True,
        note="dotted / linear-irregular vessels, caliber ~0.05-0.3 mm",
    ),
    StructureSpec(
        "blue_white_veil", period_px_at_224=28.0, contrast_floor=0.008, oriented=False,
        note="diffuse structureless zone, several mm across",
        normalize="membership",
    ),
)

STRUCTURE_NAMES: List[str] = [s.name for s in STRUCTURES]
ORIENTED: List[str] = [s.name for s in STRUCTURES if s.oriented]


def scale_periods(native_size: int) -> Dict[str, float]:
    """Characteristic periods rescaled to an arbitrary native resolution."""
    f = native_size / 224.0
    return {s.name: s.period_px_at_224 * f for s in STRUCTURES}


# --------------------------------------------------------------------------
# Cached Gabor kernels
# --------------------------------------------------------------------------
_GABOR_CACHE: Dict[tuple, list] = {}


def _gabor_bank(period: float, n_theta: int = 6, bandwidth: float = 1.0):
    key = (round(period, 3), n_theta, bandwidth)
    if key not in _GABOR_CACHE:
        freq = 1.0 / max(period, 2.0)
        kernels = []
        for i in range(n_theta):
            theta = np.pi * i / n_theta
            k = gabor_kernel(freq, theta=theta, bandwidth=bandwidth)
            kernels.append((theta, np.real(k).astype(np.float32),
                            np.imag(k).astype(np.float32)))
        _GABOR_CACHE[key] = kernels
    return _GABOR_CACHE[key]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _luminance(lab: np.ndarray) -> np.ndarray:
    return lab[..., 0] / 100.0


def _redness(lab: np.ndarray) -> np.ndarray:
    """a* channel, positive = red. Normalised to roughly [0, 1]."""
    return np.clip((lab[..., 1] + 20.0) / 80.0, 0.0, 1.0)


def _structure_tensor_orientation(gx: np.ndarray, gy: np.ndarray,
                                  sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (theta, coherence) from a smoothed structure tensor.

    theta is in [0, pi) — orientation, not direction. coherence in [0, 1] is
    (l1 - l2) / (l1 + l2), i.e. how confidently a single orientation exists.
    """
    jxx = ndi.gaussian_filter(gx * gx, sigma)
    jyy = ndi.gaussian_filter(gy * gy, sigma)
    jxy = ndi.gaussian_filter(gx * gy, sigma)

    diff = jxx - jyy
    tr = jxx + jyy
    disc = np.sqrt(diff * diff + 4.0 * jxy * jxy) + 1e-12

    # dominant *gradient* orientation; ridge orientation is perpendicular
    theta = 0.5 * np.arctan2(2.0 * jxy, diff)
    theta = np.mod(theta + np.pi / 2.0, np.pi)
    coherence = disc / (tr + 1e-12)
    return theta.astype(np.float32), np.clip(coherence, 0, 1).astype(np.float32)


def _image_noise_sigma(lab: np.ndarray) -> float:
    """Robust per-image noise scale in L*/100 units.

    1.4826 * MAD of a 3x3 Laplacian, divided by the Laplacian's noise gain
    (sqrt(6) for the 4-neighbour kernel). At this scale the response is
    dominated by sensor and compression noise rather than by anatomy.
    """
    L = lab[..., 0] / 100.0
    lap = ndi.laplace(L)
    mad = float(np.median(np.abs(lap - np.median(lap))))
    return max(1.4826 * mad / np.sqrt(6.0), 1e-5)


_NOISE_CACHE: Dict[tuple, float] = {}


def _quantize_sigma(sigma: float) -> float:
    """Bucket the noise estimate so the null below is computed ~20 times total,
    not once per image."""
    s = max(float(sigma), 1e-6)
    return float(10.0 ** (round(np.log10(s) * 8.0) / 8.0))


def noise_ceiling(name: str, period: float, sigma: float, size: int = 128) -> float:
    """The p99.5 response this filter produces on *pure noise* of scale `sigma`.

    This is the permutation null done cheaply and exactly: synthesise an image
    with no structure at all but the same noise amplitude as the real one, run
    the identical filter, and record how large a response noise alone can
    manufacture. Anything at or below that number is not evidence.

    Without this, per-image percentile normalisation is actively harmful: it
    stretches a featureless, noisy image's response to full range and draws a
    confident pigment-network symbol over nothing. Worse for this project's
    argument, it would manufacture the equity result, because low-contrast
    images would get their noise amplified the most.

    Noise is injected in both L* and a* (the two channels the bank reads), with
    a* scaled so that the redness map carries the same variance. The detectors
    are not all linear, so this is a calibrated empirical null rather than an
    analytic gain — which is why it is measured rather than derived.
    """
    sq = _quantize_sigma(sigma)
    key = (name, round(float(period), 2), sq, size)
    if key in _NOISE_CACHE:
        return _NOISE_CACHE[key]
    rng = np.random.default_rng(abs(hash(key)) % (2 ** 31))
    lab = np.empty((size, size, 3), dtype=np.float32)
    lab[..., 0] = 60.0 + 100.0 * sq * rng.standard_normal((size, size))
    lab[..., 1] = 80.0 * sq * rng.standard_normal((size, size))
    lab[..., 2] = 8.0 + 26.0 * sq * rng.standard_normal((size, size))
    d = _DISPATCH[name](lab, period)["density"]
    val = float(np.percentile(d, 99.5))
    _NOISE_CACHE[key] = val
    return val


def _robust01(x: np.ndarray, floor: float, hi_p: float = 99.5) -> np.ndarray:
    """Per-image rescale to [0, 1], anchored at a measured noise floor.

    `floor` is `noise_ceiling(...)`: the largest response noise alone could
    produce. Subtracting it makes 0 mean "indistinguishable from noise", and
    dividing by this image's own remaining headroom is the equal-fidelity move
    — the dynamic range a symbol gets does not depend on the subject's overall
    pigmentation or on the scope's illumination.

    The claim this supports is therefore *conditional*: equal fidelity holds
    wherever the measurement is supportable at all. Where it is not, the symbol
    correctly collapses toward zero and the source diagram reports why. That is
    a weaker claim than "equal fidelity everywhere", and it is the true one.
    """
    v = x[np.isfinite(x)]
    if v.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    hi = float(np.percentile(v, hi_p))
    denom = max(hi - floor, floor, 1e-5)
    return np.clip((x - floor) / denom, 0.0, 1.0).astype(np.float32)


def absolute01(x: np.ndarray, scale: float) -> np.ndarray:
    """Fixed global scale — the deliberately unfair alternative to `_robust01`.

    One number per structure for the entire population, which is what a naive
    implementer would write and what produces the Mercator distortion.
    """
    return np.clip(x / max(scale, 1e-8), 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# The five instruments
# --------------------------------------------------------------------------
def pigment_network(lab: np.ndarray, period: float) -> Dict[str, np.ndarray]:
    """Gabor energy across orientations; the mesh is quasi-periodic, so the
    *sum* of energies (not the max) is the presence evidence, while the
    structure tensor of the winning orientation gives the bearing."""
    L = _luminance(lab)
    Lb = L - ndi.gaussian_filter(L, period * 1.5)      # band-pass, kill shading
    energies = []
    thetas = []
    for theta, kr, ki in _gabor_bank(period):
        re = ndi.convolve(Lb, kr, mode="reflect")
        im = ndi.convolve(Lb, ki, mode="reflect")
        energies.append(np.hypot(re, im))
        thetas.append(theta)
    E = np.stack(energies, 0)                          # (T, H, W)
    density = E.max(0)

    # bearing: circular mean over the bank in double-angle space
    w = E - E.mean(0, keepdims=True)
    w = np.clip(w, 0, None)
    th = np.asarray(thetas, dtype=np.float32)[:, None, None]
    c = (w * np.cos(2 * th)).sum(0)
    s = (w * np.sin(2 * th)).sum(0)
    n = w.sum(0) + 1e-12
    theta_map = 0.5 * np.arctan2(s / n, c / n)
    coherence = np.hypot(c / n, s / n)
    # Gabor bearing is the *normal* to the mesh ridges
    theta_map = np.mod(theta_map + np.pi / 2.0, np.pi)
    return {"density": density.astype(np.float32),
            "theta": theta_map.astype(np.float32),
            "coherence": np.clip(coherence, 0, 1).astype(np.float32)}


def dots_globules(lab: np.ndarray, period: float) -> Dict[str, np.ndarray]:
    """Scale-normalised Laplacian of Gaussian across three scales.

    Dark blobs give positive scale-normalised -LoG. `scale` records the
    argmax sigma, which is the record of *how much exaggeration was applied*
    when the blob is later drawn as a symbol.
    """
    L = _luminance(lab)
    sigmas = np.array([period / 3.0, period / 2.0, period / 1.4], dtype=np.float32)
    sigmas = np.maximum(sigmas, 0.8)
    resp = np.stack([
        -(s ** 2) * ndi.gaussian_laplace(L, s) for s in sigmas
    ], 0)
    resp = np.clip(resp, 0, None)
    idx = resp.argmax(0)
    density = resp.max(0)
    scale = sigmas[idx] / sigmas[-1]
    return {"density": density.astype(np.float32),
            "scale": scale.astype(np.float32)}


def streaks(lab: np.ndarray, period: float) -> Dict[str, np.ndarray]:
    """Elongated dark ridges. Sato gives presence; the structure tensor of the
    ridge response gives bearing (which for streaks is the clinically
    meaningful quantity — radial vs. tangential)."""
    L = _luminance(lab)
    Lb = L - ndi.gaussian_filter(L, period * 2.0)
    sig = np.maximum(np.array([period / 4.0, period / 2.5]), 0.8)
    ridge = sato(-Lb, sigmas=sig, black_ridges=False, mode="reflect")
    ridge = np.nan_to_num(ridge, nan=0.0, posinf=0.0, neginf=0.0)
    gy, gx = np.gradient(ndi.gaussian_filter(Lb, period / 3.0))
    theta, coh = _structure_tensor_orientation(gx, gy, sigma=period)
    return {"density": ridge.astype(np.float32), "theta": theta, "coherence": coh}


def vessels(lab: np.ndarray, period: float) -> Dict[str, np.ndarray]:
    """Curvilinear redness. Run the ridge detector on a* rather than on
    luminance, so pigment does not masquerade as vasculature."""
    R = _redness(lab)
    Rb = R - ndi.gaussian_filter(R, period * 2.0)
    sig = np.maximum(np.array([period / 4.0, period / 2.0]), 0.8)
    ridge = sato(Rb, sigmas=sig, black_ridges=False, mode="reflect")
    ridge = np.nan_to_num(ridge, nan=0.0, posinf=0.0, neginf=0.0)
    gy, gx = np.gradient(ndi.gaussian_filter(Rb, period / 3.0))
    theta, coh = _structure_tensor_orientation(gx, gy, sigma=period)
    return {"density": ridge.astype(np.float32), "theta": theta, "coherence": coh}


def blue_white_veil(lab: np.ndarray, period: float) -> Dict[str, np.ndarray]:
    """Diffuse, structureless, bluish-white.

    Membership = (bluer than this subject's own skin) x (not darker than it)
    x (absence of texture). The texture term is what makes it a *veil* rather
    than merely blue pigment: a veil is defined by what is missing inside it.

    Both colour terms are referenced to the image's own median skin colour
    rather than to an absolute point in Lab. That is the equal-fidelity choice
    and it is also the clinically correct one — a reader judges a veil against
    the surrounding skin, not against a fixed swatch. Referencing it absolutely
    would make "veil" partly a synonym for "pale skin", which is the Mercator
    failure in its purest form.

    Unlike the band-pass detectors, this returns a calibrated membership in
    [0, 1] already, so it is exempt from the per-image robust rescale (see
    `StructureSpec.normalize`) — stretching a membership would report a veil on
    any image that merely contains its own bluest pixel.
    """
    L = _luminance(lab)
    b = lab[..., 2]
    b0 = float(np.median(b))
    L0 = float(np.median(L))
    blueness = np.clip((b0 - b) / 12.0, 0.0, 1.0)
    lightness = np.clip((L - L0 + 0.02) / 0.22, 0.0, 1.0)
    hf = L - ndi.gaussian_filter(L, 3.0)
    texture = np.sqrt(ndi.gaussian_filter(hf * hf, period / 2.0))
    structureless = np.exp(-texture / 0.02)
    veil = ndi.gaussian_filter(blueness * lightness * structureless, period / 3.0)
    return {"density": np.clip(veil, 0.0, 1.0).astype(np.float32)}


_DISPATCH = {
    "pigment_network": pigment_network,
    "dots_globules": dots_globules,
    "streaks": streaks,
    "vessels": vessels,
    "blue_white_veil": blue_white_veil,
}


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def survey(rgb: np.ndarray, native_size: int | None = None,
           equal_fidelity: bool = True,
           absolute_scales: Dict[str, float] | None = None,
           raw: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
    """Run every instrument over one native-resolution image.

    Returns {structure_name: {"density", ["theta", "coherence"], ["scale"]}},
    all at native resolution, densities normalised to [0, 1].

    Parameters
    ----------
    equal_fidelity : bool
        True  -> per-image robust normalisation (equal symbol dynamic range
                 for every subject, by construction).
        False -> fixed global scale from `absolute_scales`, reproducing the
                 Mercator failure mode for the ablation.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    if native_size is None:
        native_size = rgb.shape[0]
    lab = rgb2lab(np.clip(rgb, 0, 1)).astype(np.float32)
    periods = scale_periods(native_size)
    sigma = _image_noise_sigma(lab)

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for spec in STRUCTURES:
        r = _DISPATCH[spec.name](lab, periods[spec.name])
        if not raw:
            if spec.normalize == "membership":
                # already a calibrated [0, 1] membership; rescaling it would
                # invent a veil on any image containing its own bluest pixel
                r["density"] = np.clip(r["density"], 0.0, 1.0).astype(np.float32)
            elif equal_fidelity:
                floor = noise_ceiling(spec.name, periods[spec.name], sigma)
                r["density"] = _robust01(r["density"], floor)
            else:
                scale = (absolute_scales or {}).get(spec.name, 0.05)
                r["density"] = absolute01(r["density"], scale)
        out[spec.name] = r
    return out


def estimate_absolute_scales(rgbs: np.ndarray, native_size: int,
                             n: int = 200, seed: int = 0) -> Dict[str, float]:
    """Fit the fixed global scales used by the `equal_fidelity=False` arm.

    Fitted on a training subsample only, exactly as a naive implementer would
    do it: one number per structure for the entire population.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rgbs), size=min(n, len(rgbs)), replace=False)
    acc: Dict[str, List[float]] = {s.name: [] for s in STRUCTURES}
    for i in idx:
        rgb = np.asarray(rgbs[i], dtype=np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        lab = rgb2lab(np.clip(rgb, 0, 1)).astype(np.float32)
        periods = scale_periods(native_size)
        for spec in STRUCTURES:
            d = _DISPATCH[spec.name](lab, periods[spec.name])["density"]
            acc[spec.name].append(float(np.percentile(d, 99.5)))
    return {k: float(np.mean(v)) for k, v in acc.items()}
