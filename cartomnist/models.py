"""Networks. Deliberately small — the claim is about preprocessing, not capacity.

`SmallCNN` is the same architecture for every low-resolution regime; only
`in_channels` differs between naive (3) and generalized (18). Parameter counts
therefore differ by ~500 in the stem, which is reported in the results table so
nobody can attribute the effect to model size.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Three-stage CNN with GroupNorm, safe at grids from 7x7 to 56x56.

    GroupNorm rather than BatchNorm because the rare classes make small batches
    unavoidable in the stratified sweeps, and BatchNorm statistics on a batch
    dominated by `nv` were measurably unstable across seeds.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 7, width: int = 48,
                 p_drop: float = 0.15):
        super().__init__()
        w = width

        def block(cin, cout, pool: bool):
            layers = [
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.GroupNorm(min(8, cout), cout), nn.SiLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.GroupNorm(min(8, cout), cout), nn.SiLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, ceil_mode=True))
            return nn.Sequential(*layers)

        self.stem = block(in_channels, w, pool=True)
        self.mid = block(w, w * 2, pool=True)
        self.top = block(w * 2, w * 4, pool=False)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(p_drop), nn.Linear(w * 4, n_classes),
        )

    def forward(self, x):
        return self.head(self.top(self.mid(self.stem(x))))


def resnet18(in_channels: int = 3, n_classes: int = 7, pretrained: bool = False):
    """Torchvision ResNet-18 for the 224 arm.

    Pretrained weights are off by default: Kaggle notebooks frequently run with
    internet disabled, and more importantly an ImageNet-initialised 224 model
    versus a from-scratch 28 model would confound the comparison. Turn it on
    only for a clearly labelled supplementary row.
    """
    from torchvision.models import resnet18 as _r18
    m = _r18(weights="IMAGENET1K_V1" if pretrained else None)
    if in_channels != 3:
        old = m.conv1
        new = nn.Conv2d(in_channels, old.out_channels, old.kernel_size,
                        old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :3] = old.weight
        m.conv1 = new
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def build(regime: str, in_channels: int, n_classes: int, **kw) -> nn.Module:
    if regime == "native224":
        return resnet18(in_channels, n_classes, pretrained=kw.get("pretrained", False))
    return SmallCNN(in_channels, n_classes, width=kw.get("width", 48))


def count_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))
