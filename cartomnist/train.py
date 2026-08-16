"""Training and inference.

One subtlety that matters and is easy to get wrong: the symbol channels carry
*orientation*, so geometric augmentation is not channel-agnostic. Flipping an
image flips the bearing of every pigment-network segment in it. If you flip the
tensor without transforming cos2θ/sin2θ you are training the model on charts
whose compass rose disagrees with their roads, and the orientation channels
degrade to noise. `EquivariantAug` handles this; `test_train.py` asserts it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .models import build, count_params


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
class EquivariantAug:
    """Dihedral augmentation that transforms orientation channels correctly.

    Channels are stored as (cos2θ + 1)/2 and (sin2θ + 1)/2. Decoding to
    u = 2c - 1 and applying the double-angle rules:

        horizontal flip (θ -> π - θ):   cos2θ ->  cos2θ,  sin2θ -> -sin2θ
        vertical flip   (θ -> -θ):      cos2θ ->  cos2θ,  sin2θ -> -sin2θ
        rot90           (θ -> θ + π/2): cos2θ -> -cos2θ,  sin2θ -> -sin2θ

    (h-flip and v-flip act identically on a bearing; their composition is
    rot180, which is the identity on orientation — as it should be.)

    A channel that is exactly 0 means "no bearing drawn here" (the
    simplification contract). Encoded, that is 0.5, and every rule above maps
    0.5 to 0.5, so the undrawn state survives augmentation. That is why the
    encoding is centred rather than [0,1]-shifted.
    """

    def __init__(self, cos_idx: Sequence[int], sin_idx: Sequence[int],
                 flips: bool = True, rot90: bool = True):
        self.cos_idx = list(cos_idx)
        self.sin_idx = list(sin_idx)
        self.flips, self.rot90 = flips, rot90

    def __call__(self, x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        # x: (C, H, W)
        neg_cos = False
        neg_sin = False
        if self.flips:
            if rng.random() < 0.5:
                x = torch.flip(x, dims=[2])       # horizontal
                neg_sin = not neg_sin
            if rng.random() < 0.5:
                x = torch.flip(x, dims=[1])       # vertical
                neg_sin = not neg_sin
        if self.rot90:
            k = int(rng.integers(0, 4))
            if k:
                x = torch.rot90(x, k, dims=[1, 2])
                if k % 2 == 1:
                    neg_cos = not neg_cos
                    neg_sin = not neg_sin
        if neg_cos and self.cos_idx:
            x[self.cos_idx] = 1.0 - x[self.cos_idx]
        if neg_sin and self.sin_idx:
            x[self.sin_idx] = 1.0 - x[self.sin_idx]
        return x


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray,
                 aug: Optional[EquivariantAug] = None,
                 channels: Optional[Sequence[int]] = None,
                 zero_channels: Optional[Sequence[int]] = None,
                 seed: int = 0):
        self.x, self.y = x, np.asarray(y, dtype=np.int64)
        self.aug = aug
        self.channels = list(channels) if channels is not None else None
        self.zero = list(zero_channels) if zero_channels else None
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        # `copy=True` is required, not defensive: the cached tensors are opened
        # with mmap_mode="r", so a plain view is read-only and backed by the
        # cache file. Both the zero-channel ablation and the augmenter write in
        # place, and writing through a read-only mapping kills the process with
        # SIGBUS and a truncated traceback.
        a = np.array(self.x[i], dtype=np.float32, copy=True)
        if a.ndim == 3 and a.shape[-1] in (1, 3) and a.shape[0] not in (1, 3):
            a = np.transpose(a, (2, 0, 1))        # HWC -> CHW (raw 224 uint8)
        if a.max() > 1.5:
            a = a / 255.0
        t = torch.from_numpy(np.ascontiguousarray(a))
        if self.zero:
            t[self.zero] = 0.0
        if self.channels is not None:
            t = t[self.channels]
        if self.aug is not None:
            t = self.aug(t, self.rng)
        return t, int(self.y[i])


# --------------------------------------------------------------------------
@dataclass
class TrainResult:
    val_logits: np.ndarray
    test_logits: np.ndarray
    val_y: np.ndarray
    test_y: np.ndarray
    n_params: int
    epochs: int
    history: List[Dict[str, float]]


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def _infer(model, loader, device, amp: bool) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    outs, ys = [], []
    dtype = torch.float16 if (amp and device.type == "cuda") else torch.float32
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=(amp and device.type == "cuda")):
            out = model(xb)
        outs.append(out.float().cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(outs), np.concatenate(ys)


def train_eval(train_x, train_y, val_x, val_y, test_x, test_y, *,
               regime: str, n_classes: int, cfg,
               channels: Optional[Sequence[int]] = None,
               aug: Optional[EquivariantAug] = None,
               test_zero_channels: Optional[Sequence[int]] = None,
               seed: int = 0, epochs: Optional[int] = None,
               verbose: bool = True) -> TrainResult:
    """Train one model and return logits on val and test.

    `test_zero_channels` ablates channels at *test time only* — the experiment
    that checks whether the model learned the symbols as a shortcut rather than
    as evidence.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = pick_device(cfg.device)
    native = regime == "native224"
    epochs = epochs or (cfg.epochs_native if native else cfg.epochs_small)
    bs = cfg.batch_size_native if native else cfg.batch_size
    lr = cfg.lr_native if native else cfg.lr

    in_ch = (len(channels) if channels is not None
             else (np.asarray(train_x[0]).shape[0] if np.asarray(train_x[0]).ndim == 3
                   and np.asarray(train_x[0]).shape[0] <= 32 else 3))

    ds_tr = ArrayDataset(train_x, train_y, aug=aug, channels=channels, seed=seed)
    ds_va = ArrayDataset(val_x, val_y, channels=channels)
    ds_te = ArrayDataset(test_x, test_y, channels=channels,
                         zero_channels=test_zero_channels)

    nw = 2 if device.type == "cuda" else 0
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=nw,
                       drop_last=len(ds_tr) > bs, pin_memory=device.type == "cuda")
    dl_va = DataLoader(ds_va, batch_size=bs * 2, num_workers=nw)
    dl_te = DataLoader(ds_te, batch_size=bs * 2, num_workers=nw)

    model = build(regime, in_ch, n_classes,
                  pretrained=getattr(cfg, "pretrained_224", False)).to(device)
    crit = nn.CrossEntropyLoss(
        weight=_class_weights(np.asarray(train_y), n_classes).to(device)
        if cfg.class_weighted_loss else None,
        label_smoothing=cfg.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    steps = max(1, len(dl_tr)) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.25)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: List[Dict[str, float]] = []
    for ep in range(epochs):
        model.train()
        tot, nseen = 0.0, 0
        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach()) * len(yb)
            nseen += len(yb)
        history.append({"epoch": ep, "loss": tot / max(1, nseen),
                        "lr": sched.get_last_lr()[0]})
        if verbose and (ep == epochs - 1 or ep % max(1, epochs // 4) == 0):
            print(f"    [{regime}|seed{seed}] epoch {ep+1}/{epochs} "
                  f"loss {history[-1]['loss']:.4f}")

    vlog, vy = _infer(model, dl_va, device, use_amp)
    tlog, ty = _infer(model, dl_te, device, use_amp)
    n_params = count_params(model)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return TrainResult(vlog, tlog, vy, ty, n_params, epochs, history)


def train_eval_seeds(*args, seeds: Sequence[int] = (0,), **kw) -> List[TrainResult]:
    return [train_eval(*args, seed=s, **kw) for s in seeds]


def average_logits(results: Sequence[TrainResult]) -> Tuple[np.ndarray, np.ndarray]:
    """Mean of per-seed softmax, re-expressed as logits.

    Averaging probabilities rather than logits is the right ensemble for a
    calibration study: averaging logits sharpens confidence artificially and
    would flatter the ECE numbers.
    """
    from .metrics import softmax
    v = np.mean([softmax(r.val_logits) for r in results], axis=0)
    t = np.mean([softmax(r.test_logits) for r in results], axis=0)
    return np.log(np.clip(v, 1e-12, None)), np.log(np.clip(t, 1e-12, None))
