"""One shared parallel-map helper.

Two things here are load-bearing rather than stylistic:

1. **Each worker receives only its own slice of images**, not a closure over the
   whole array. A closure over a 1.5 GB image stack makes joblib memory-map it
   to a temp file; when that file is cleaned up under the worker, the process
   dies with SIGBUS and the traceback is truncated to "Fatal Python error:" with
   no frames — a genuinely miserable thing to debug.

2. **`max_nbytes=None`** disables that auto-memmapping outright, so the slices
   are pickled normally. Slices are small (a few tens of MB at most), so the
   serialisation cost is far below the filter-bank cost being parallelised.
"""
from __future__ import annotations

import os
from typing import Callable, List, Sequence

import numpy as np


def n_workers(n_jobs: int | None) -> int:
    if n_jobs in (0, None):
        return max(1, (os.cpu_count() or 2))
    return max(1, int(n_jobs))


def map_chunks(images, fn: Callable[[np.ndarray], object], n_jobs: int | None = 0,
               chunks_per_worker: int = 3) -> List[object]:
    """Apply `fn` to contiguous chunks of `images`, in parallel, in order.

    `fn` receives a real numpy array of shape (chunk, H, W, 3) and returns
    anything picklable. Results come back in input order.
    """
    n = len(images)
    nw = n_workers(n_jobs)
    if nw == 1 or n <= 1:
        return [fn(np.asarray(images))]

    bounds = [b for b in np.array_split(np.arange(n), min(n, nw * chunks_per_worker))
              if len(b)]
    try:
        from joblib import Parallel, delayed
    except ImportError:
        return [fn(np.asarray(images[b[0]:b[-1] + 1])) for b in bounds]

    return list(Parallel(n_jobs=nw, backend="loky", max_nbytes=None)(
        delayed(fn)(np.asarray(images[b[0]:b[-1] + 1])) for b in bounds))


def concat_arrays(parts: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(p) for p in parts], axis=0)


def concat_dicts(parts: Sequence[dict], keys: Sequence) -> dict:
    return {k: concat_arrays([p[k] for p in parts]) for k in keys}
