"""Learned generalization of area-mean pooling.

`operators.aggregate` is a fixed, auditable function: every cell is the
unweighted mean of its footprint, which is exactly what buys the areal-
integral invariant `contracts.AGGREGATION` states. The obvious question this
invites: is a FIXED pooling rule itself costing accuracy, independent of
symbolization? This module answers it with a pooling layer that is trained
end-to-end with the classifier instead of fixed in advance — see
`models.LearnedDownsampleCNN` for how it composes with `SmallCNN`, and
`contracts.LEARNED_POOL` for the fidelity contract.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedDownsample(nn.Module):
    """A pooling layer that starts at area-mean and is free to move away from it.

    `mode="linear"`: a per-channel (depthwise) convolution with
    kernel == stride == `factor`, weight initialized to `1 / factor**2` so the
    FIRST forward pass, before any gradient step, is bit-for-bit
    `operators.aggregate` (checked by `tests/test_learned_pool.py`). Any
    measured gap between this arm and the fixed `naive` baseline is then
    attributable to what training changes, not to an initialization
    advantage.

    `mode="attention"`: a strict generalization of the same idea. Instead of
    a fixed uniform weight over each `factor x factor` block, a small scoring
    convolution produces one score per pixel; softmax-normalising it within
    each block and taking the resulting weighted sum recovers area-mean
    pooling exactly when the score network is identically zero, and departs
    from it wherever the network learns a block is better summarised by a
    subset of its pixels.
    """

    def __init__(self, channels: int = 3, factor: int = 8, mode: str = "linear"):
        super().__init__()
        if mode not in ("linear", "attention"):
            raise ValueError(f"mode must be 'linear' or 'attention', got {mode!r}")
        self.channels, self.factor, self.mode = channels, factor, mode
        self.pool = nn.Conv2d(channels, channels, factor, stride=factor,
                              groups=channels, bias=False)
        with torch.no_grad():
            self.pool.weight.fill_(1.0 / (factor * factor))
        if mode == "attention":
            self.score = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "linear":
            return self.pool(x)

        B, C, H, W = x.shape
        f = self.factor
        t = H // f
        score = self.score(x)                                  # (B, 1, H, W)
        s_blocks = F.unfold(score, kernel_size=f, stride=f)      # (B, f*f, t*t)
        w = torch.softmax(s_blocks, dim=1)                       # normalise per block
        x_blocks = F.unfold(x, kernel_size=f, stride=f)          # (B, C*f*f, t*t)
        w_rep = w.repeat(1, C, 1)                                # (B, C*f*f, t*t)
        weighted = (x_blocks * w_rep).view(B, C, f * f, t * t).sum(dim=2)
        return weighted.view(B, C, t, t)
