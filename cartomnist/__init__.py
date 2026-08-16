"""cartomnist — cartographic generalization for low-resolution medical benchmarks.

    from cartomnist import Config, run_all
    out = run_all(Config().apply_fast())

The argument in one paragraph: a mapmaker reducing 1:10,000 to 1:1,000,000 faces
exactly the problem a 224 -> 28 medical benchmark faces, and did not solve it by
resizing. Cartography developed a vocabulary of named generalization operators,
each with an explicit contract about what is preserved and what is sacrificed;
it prints a scale bar in physical units; and nautical charts have carried a
source diagram — an inset showing where the survey is too sparse to trust —
since the nineteenth century. This package implements that discipline for
DermaMNIST and ships the audit.
"""
from .config import CFG, Config, Paths
from .contracts import OPERATORS, NAIVE_RESIZE, legend_dict
from .generalize import (channel_spec, generalize, generalize_batch,
                         generalize_multi, naive_resize)
from .sufficiency import nyquist_table, source_diagram

__version__ = "0.1.0"

__all__ = [
    "CFG", "Config", "Paths",
    "OPERATORS", "NAIVE_RESIZE", "legend_dict",
    "channel_spec", "generalize", "generalize_batch", "generalize_multi",
    "naive_resize", "source_diagram", "nyquist_table",
    "run_all",
]


def run_all(cfg=None, raw_data=None):
    """Lazy import so `import cartomnist` does not pull in torch."""
    from .pipeline import run_all as _run
    return _run(cfg, raw_data=raw_data)
