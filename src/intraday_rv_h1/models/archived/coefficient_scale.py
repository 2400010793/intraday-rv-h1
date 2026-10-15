"""Separate coefficient saturation bounds from sensitivity at initialization."""
from __future__ import annotations

import torch


def bounded_coefficient_delta(raw, cap, sensitivity=None):
    """cap*tanh(raw*sensitivity/cap); default reproduces cap*tanh(raw).

    d(delta)/d(raw) at zero equals sensitivity. Thus an experiment can enlarge
    the saturation bound while holding the initial gradient scale constant.
    These scales are caller-supplied parameters/buffers, never Test estimates.
    """
    sensitivity = cap if sensitivity is None else sensitivity
    for name, value in (("raw", raw), ("cap", cap), ("sensitivity", sensitivity)):
        if not torch.is_tensor(value) or not torch.isfinite(value).all():
            raise ValueError(f"nonfinite/non-tensor coefficient {name}")
    if (cap <= 0).any() or (sensitivity <= 0).any():
        raise ValueError("coefficient cap and sensitivity must be positive")
    if torch.broadcast_shapes(raw.shape, cap.shape, sensitivity.shape) != raw.shape:
        raise ValueError("coefficient scales must broadcast to raw coefficients")
    return cap * torch.tanh(raw * (sensitivity / cap))
