"""The Legend.

A map without a legend is a picture. This module is the machine-readable legend
for the generalization pipeline: every operator that touches the pixels declares,
in one place, *what it preserves* and *what it sacrifices*.

Nothing here computes anything. It exists so that `results/legend.json` can be
shipped alongside the benchmark, and so that a reviewer can audit the claim
"this 28x28 image is a faithful generalization" against a written contract
instead of against a plot.

Vocabulary follows the cartographic generalization literature
(Ratajski 1967; Brassel & Weibel 1988; McMaster & Shea 1992; Töpfer & Pillewizer
1966 for the radical law). The source-diagram idea is lifted directly from
nautical charting practice (zones-of-confidence / source diagrams, IHO S-57);
the critique that every generalization is a political choice about whose
features survive is Harley, "Deconstructing the Map" (1989).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class FidelityContract:
    """What an operator promises, and what it admits to destroying."""
    operator: str
    cartographic_analogue: str
    preserves: List[str]
    sacrifices: List[str]
    invariant: str            # the property a test can actually check
    parameters: Dict[str, str]
    citation: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# The six operators actually applied by cartomnist.generalize
# --------------------------------------------------------------------------
SELECTION = FidelityContract(
    operator="selection",
    cartographic_analogue=(
        "Deciding which classes of feature appear on the sheet at all. A "
        "1:1,000,000 chart carries motorways and drops farm tracks."
    ),
    preserves=[
        "the K structures with the highest diagnostic mutual information",
        "the full native-resolution evidence for every retained structure",
    ],
    sacrifices=[
        "all evidence for non-retained structures (they are absent, not faint)",
    ],
    invariant=(
        "the set of retained structures is fixed before seeing the test split "
        "and is recorded in results/selection.json"
    ),
    parameters={"K": "number of structure families retained"},
    citation="Töpfer & Pillewizer (1966), 'The principles of selection'",
)

SIMPLIFICATION = FidelityContract(
    operator="simplification",
    cartographic_analogue=(
        "Douglas-Peucker on a coastline: keep the shape, drop the vertices "
        "that the medium cannot resolve."
    ),
    preserves=[
        "orientation where the local structure tensor is coherent",
        "the zero/absent state where it is not",
    ],
    sacrifices=[
        "orientation in low-coherence regions, which is set to 0 rather than "
        "to a noisy estimate — an undrawn road, not a mis-placed one",
    ],
    invariant="orientation channels are exactly 0 wherever coherence < tau",
    parameters={"tau": "coherence floor in [0,1]"},
    citation="Douglas & Peucker (1973); McMaster & Shea (1992)",
)

AGGREGATION = FidelityContract(
    operator="aggregation",
    cartographic_analogue=(
        "Merging a cluster of buildings into a single built-up-area polygon."
    ),
    preserves=[
        "the areal integral of every response map: cell value == mean of the "
        "native-resolution response over that cell's footprint",
    ],
    sacrifices=[
        "the within-cell spatial arrangement of structure instances",
    ],
    invariant=(
        "sum(cell_value * cell_area) == sum(native_response) up to float error "
        "(checked by tests/test_operators.py::test_aggregation_conserves_mass)"
    ),
    parameters={"footprint": "native_size / target_size pixels per cell"},
    citation="Brassel & Weibel (1988)",
)

DISPLACEMENT = FidelityContract(
    operator="displacement",
    cartographic_analogue=(
        "Shifting a road away from the river it runs beside so both remain "
        "legible at scale, accepting that neither is now in its true place."
    ),
    preserves=[
        "the per-cell rank order of structure densities (which structure "
        "dominates a cell is never changed)",
    ],
    sacrifices=[
        "absolute density in congested cells, which is attenuated so that a "
        "co-located weaker structure survives above the noise floor",
    ],
    invariant=(
        "argsort of the symbol vector within each cell is unchanged; "
        "attenuation is the identity wherever the cell is not congested"
    ),
    parameters={"lambda": "congestion attenuation strength"},
    citation="Nickerson (1988); Mackaness (1994)",
)

EXAGGERATION = FidelityContract(
    operator="exaggeration",
    cartographic_analogue=(
        "A road 0.5 mm wide at scale is drawn 2 mm wide, because the map's "
        "purpose is navigation, not photometry."
    ),
    preserves=[
        "the ordering of densities (the map is strictly monotone in the "
        "underlying measurement)",
    ],
    sacrifices=[
        "amplitude linearity — a cell twice as dense is NOT twice as bright, "
        "so these channels must never be read as photometry",
    ],
    invariant="x -> x**gamma is strictly increasing on [0,1] for gamma > 0",
    parameters={"gamma": "exaggeration exponent, < 1 expands the faint end"},
    citation="Standard practice; see Robinson et al., 'Elements of Cartography'",
)

SYMBOLIZATION = FidelityContract(
    operator="symbolization",
    cartographic_analogue=(
        "The core move. A feature below the resolvable width of the medium is "
        "not dropped and not blurred: it is replaced by a SYMBOL that says "
        "'a feature of this class, this density, this bearing is here'."
    ),
    preserves=[
        "presence, density and bearing of each diagnostic structure family, "
        "measured at NATIVE resolution before any downsampling",
    ],
    sacrifices=[
        "the appearance of the structure — a symbolized pigment network does "
        "not look like a pigment network, and cannot be visually inspected "
        "for atypia by a human reader",
        "any structure not in the filter bank's vocabulary",
    ],
    invariant=(
        "every symbol channel is a deterministic function of the native-"
        "resolution image alone; no training signal enters symbolization"
    ),
    parameters={
        "structures": "pigment_network, dots_globules, streaks, vessels, veil",
        "measurement": "Gabor / scale-normalised LoG / oriented ridge / Sato",
    },
    citation="Argenziano et al. (1998) 7-point checklist; ISIC 2018 Task 2",
)

SOURCE_DIAGRAM = FidelityContract(
    operator="source_diagram",
    cartographic_analogue=(
        "The inset on a nautical chart showing which areas were surveyed, at "
        "what density, and in what year — so the navigator can see at a "
        "glance where the chart cannot be trusted."
    ),
    preserves=[
        "an explicit, per-cell record of whether the ACQUISITION could have "
        "detected each structure here, had it been present — noise floor and "
        "8-bit quantisation step against the structure's minimum contrast",
    ],
    sacrifices=[
        "nothing — this channel adds information rather than removing it. It "
        "is the honest accounting of what the other five operators cost.",
    ],
    invariant=(
        "certificate == min(noise_margin, quant_margin) over all structures, "
        "computed from the native image only, never from labels, and never "
        "from whether the structure is actually present — a chart is not "
        "untrustworthy merely because the sea floor is flat there"
    ),
    parameters={
        "noise_margin": "log2(structure contrast floor x sqrt(footprint) / "
                        "local noise floor)",
        "quant_margin": "log2(structure contrast floor / L* step of one 8-bit "
                        "code at this luminance), minus a clipping penalty",
    },
    citation="IHO S-57 CATZOC / S-4 source diagrams; Harley (1989) for the "
             "argument that omitting one is itself a rhetorical act",
)

OPERATORS: Sequence[FidelityContract] = (
    SELECTION,
    SIMPLIFICATION,
    AGGREGATION,
    DISPLACEMENT,
    EXAGGERATION,
    SYMBOLIZATION,
    SOURCE_DIAGRAM,
)


# --------------------------------------------------------------------------
# The naive baseline, written as a contract so it can be compared like-for-like
# --------------------------------------------------------------------------
NAIVE_RESIZE = FidelityContract(
    operator="naive_resize",
    cartographic_analogue=(
        "Photoreducing the 1:10,000 sheet on a copier and calling it a "
        "1:1,000,000 map. Everything narrower than the new pixel is gone, and "
        "nothing on the sheet says so."
    ),
    preserves=[
        "the low-pass content of the image below the new Nyquist limit",
        "mean colour",
    ],
    sacrifices=[
        "every diagnostic structure whose characteristic period is below "
        "2 * (native_size / target_size) native pixels — which, at 224->28, "
        "is all five of them",
        "the fact that this happened (no channel records the loss)",
    ],
    invariant="none declared — this is the point of the paper",
    parameters={"interpolation": "bilinear/area, anti-aliased"},
    citation="MedMNIST v2 (Yang et al., 2023), and essentially every "
             "downscaled medical benchmark",
)


SR_UPSAMPLE = FidelityContract(
    operator="sr_upsample",
    cartographic_analogue=(
        "Redrawing a photoreduced 1:1,000,000 sheet back up to 1:10,000 scale "
        "by interpolation, rather than resurveying the ground — inventing "
        "plausible detail instead of measuring it."
    ),
    preserves=[
        "the low-frequency content of the shipped 28x28 image",
        "edge contrast a plain interpolation kernel would smooth away",
    ],
    sacrifices=[
        "fidelity to the true native acquisition — a generic sharpening "
        "prior's added edges are not guaranteed to correspond to real "
        "anatomy, only to what a Lanczos-plus-unsharp-mask pipeline finds "
        "plausible",
        "any structure genuinely destroyed by the areal averaging that "
        "produced the 28x28 input in the first place",
    ],
    invariant=(
        "output is a deterministic function of the 28x28 naive-resized image "
        "alone — the native image and the labels are never read, so a "
        "measured recovery of the naive/native224 gap can only be credited "
        "to the upsampling prior, not to smuggled native-resolution access"
    ),
    parameters={"method": "Lanczos upsample + unsharp mask"},
    citation="Dong et al. (2014) SRCNN, for bicubic-upsampling-as-baseline "
             "convention in the super-resolution literature; Real-ESRGAN "
             "(Wang et al., 2021) was considered and excluded — see "
             "cartomnist/superres.py's module docstring",
)


LEARNED_POOL = FidelityContract(
    operator="learned_pool",
    cartographic_analogue=(
        "An adaptive survey grid that spends sampling density wherever the "
        "cartographer chooses, rather than a fixed one applied uniformly "
        "over every cell."
    ),
    preserves=[
        "whatever a gradient-trained pooling layer finds useful for the "
        "downstream classification loss — no fixed guarantee, by design",
    ],
    sacrifices=[
        "the areal-integral invariant area-mean pooling guarantees by "
        "construction (contracts.AGGREGATION) — this operator is now "
        "data- and task-dependent rather than a fixed, auditable function",
        "determinism: the exact pooling rule depends on the training run "
        "(seed, optimizer trajectory), not just on the pixels",
    ],
    invariant=(
        "mode='linear' initializes to exact area-mean pooling — the pooling "
        "weight equals 1/factor**2 before any gradient step — so a measured "
        "gap against the fixed `naive` baseline is attributable to what "
        "training changes, not to an initialization advantage "
        "(checked by tests/test_learned_pool.py)"
    ),
    parameters={"factor": "native_size / target_size",
               "mode": "linear (learned uniform weight) | attention "
                       "(per-pixel softmax weight within each block)"},
    citation="Lee et al. (2019), 'Set Transformer', for attention pooling; "
             "Sun et al. (2021), 'Learning to Downsample for Segmentation of "
             "Ultra-High-Resolution Images', for learned downsampling "
             "specifically",
)

PAN_SHARPEN = FidelityContract(
    operator="pan_sharpen",
    cartographic_analogue=(
        "This project's own symbolize-then-fuse pipeline — measure fine "
        "structure at native resolution, then inject it into the coarser "
        "product — is structurally the same move as remote-sensing pan-"
        "sharpening: fusing a high-resolution panchromatic band into "
        "coarser multispectral bands. This operator makes that comparison "
        "literal with an actual classical fusion algorithm in place of "
        "cartographic symbolization."
    ),
    preserves=[
        "high-frequency luminance detail from the native image, injected "
        "into the upsampled low-resolution color bands before the final "
        "area-mean pooling step",
    ],
    sacrifices=[
        "spectral fidelity — Brovey and IHS fusion are both known, in the "
        "remote-sensing literature, to distort color in saturated regions, "
        "since both assume a PAN-MS correlation a lesion photograph need "
        "not actually have",
    ],
    invariant=(
        "the fused image, before the final pooling step, is a deterministic "
        "function of the native image alone — no labels, no training signal "
        "— exactly like symbolization"
    ),
    parameters={"method": "brovey | ihs"},
    citation="Gillespie, Kahle & Walker (1987) for the Brovey transform; "
             "Chavez, Sides & Anderson (1991) for IHS fusion; Vivone et al. "
             "(2015), 'A Critical Comparison Among Pansharpening "
             "Algorithms', for the general framework this baseline borrows "
             "from",
)


def legend_dict() -> dict:
    return {
        "generalized": [c.as_dict() for c in OPERATORS],
        "baseline": [NAIVE_RESIZE.as_dict(), SR_UPSAMPLE.as_dict(),
                    LEARNED_POOL.as_dict(), PAN_SHARPEN.as_dict()],
    }
