"""Slope-only attention and a separate causal-state Ridge initialization.

The attention intervention masks only delta[0]. It preserves the original
reference intercept, limits, penalties, random draws and convolution path.
The state initialization is a fixed Ridge fit of log-variance residuals, not
an attention pretraining step or a fit of Test labels.
"""
from __future__ import annotations

import numpy as np
import torch

from .compact_prefix_attention import CompactPrefixStateAttention, CompactTrainablePrefixAttention
from .directional_coefficient_attention import DirectionalAttentionConfig
from .original_prefix_ablation import feature_mask


def state_design(features, state, variant="drop_reconstructed"):
    """RV1-rank x every retained slope, plus own-rank x auxiliary values."""
    active = feature_mask(variant)
    h = np.where(active, np.asarray(features, np.float64), 0.)
    s = np.asarray(state, np.float64)
    if h.ndim != 2 or h.shape[1] != 16 or s.shape != (len(h), 8, 3):
        raise ValueError("state Ridge requires aligned [N,16] and [N,8,3]")
    rv = np.where(s[:, 6, 2] > 0, s[:, 6, 0], 0.)
    own = np.where((s[:, :6, 2] > 0) & active[10:], s[:, :6, 0], 0.)
    slopes = np.flatnonzero(active & (np.arange(16) != 0))
    auxiliary = np.flatnonzero(active[10:]) + 10
    x = np.column_stack((h[:, slopes]*rv[:, None], h[:, auxiliary]*own[:, auxiliary-10]))
    if not np.isfinite(h).all() or not np.isfinite(x).all():
        raise FloatingPointError("nonfinite active state Ridge input")
    return x, slopes, auxiliary


def solve_state_ridge(gram, rhs, rows, *, ridge=.01):
    """Mean-loss Ridge after RMS scaling, with no coefficient sign constraint."""
    g, b = np.asarray(gram, float), np.asarray(rhs, float)
    if (g.ndim != 2 or g.shape != (len(b), len(b)) or rows < 1 or ridge <= 0
        or not np.isfinite(ridge) or not np.isfinite(g).all() or not np.isfinite(b).all()):
        raise ValueError("invalid state Ridge sufficient statistics")
    # X / RMS: alpha*N in normalized coordinates equals alpha*diag(X'X)
    # in original coordinates. Zero-variance columns are explicitly inactive.
    scale2 = np.diag(g).copy()
    keep = scale2 > 1e-20
    solution = np.zeros_like(b)
    if keep.any():
        reduced = g[np.ix_(keep, keep)]
        penalty = np.diag(ridge*scale2[keep])
        solution[keep] = np.linalg.solve(reduced+penalty, b[keep])
    if not np.isfinite(solution).all():
        raise FloatingPointError("nonfinite state Ridge solution")
    return solution, dict(rows=int(rows), ridge_mean_loss=float(ridge),
        normalization="interaction-column RMS from fitting rows only; no centering",
        intercept_fitted=False, signed_coefficients=True, active_columns=int(keep.sum()),
        condition=float(np.linalg.cond(reduced+penalty)) if keep.any() else 0.)


class SlopeOnlyAttention(CompactPrefixStateAttention):
    def raw_delta_for_assets(self, old_state, state, context, asset_ids, scale_context=None):
        raw = super().raw_delta_for_assets(old_state, state, context, asset_ids, scale_context)
        # No in-place modification and no new parameter/RNG draw. The shared
        # expert output bias receives gradients from retained slopes only.
        return torch.cat((torch.zeros_like(raw[:, :1]), raw[:, 1:]), 1)


class SlopeStateAttention(CompactTrainablePrefixAttention):
    def __init__(self, config, seed, limits, descriptors, variant, reference_state,
                 rank_values, rank_counts, *, no_intercept, state_coefficients, train_assets):
        if isinstance(config, dict):
            config = DirectionalAttentionConfig(**config)
        super().__init__(config, seed, limits, descriptors, variant, reference_state,
                         rank_values, rank_counts, prefix_state="both")
        if no_intercept:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                self.attention = SlopeOnlyAttention(config, limits, descriptors, variant, "both")
        self.no_intercept = bool(no_intercept)
        self.has_state_initialization = state_coefficients is not None
        if self.has_state_initialization:
            weights = torch.as_tensor(state_coefficients, dtype=torch.float64)
            if weights.shape != (len(limits), 2, 16) or not torch.isfinite(weights).all():
                raise ValueError("state initialization must have finite [assets,2,16] coefficients")
            if torch.count_nonzero(weights[:, :, 0]) or torch.count_nonzero(weights[:, :, ~self.reference.active]):
                raise ValueError("state initialization cannot update intercept or deleted slopes")
            self.register_buffer("state_coefficients", weights.clone())
        allowed = set(train_assets)
        if not allowed or min(allowed) < 0 or max(allowed) >= len(limits):
            raise ValueError("invalid trainable asset set")
        for asset, local in enumerate(self.reference.locals):
            local.requires_grad_(asset in allowed)
            local.output.weight.requires_grad_(False)

    def initialized_reference(self, reference, current, ids):
        if not self.has_state_initialization:
            return reference
        weights = self.state_coefficients[ids]
        rv = torch.where(current[:, 6, 2] > 0, current[:, 6, 0], 0.)
        own = torch.where((current[:, :6, 2] > 0) & self.reference.active[10:],
                          current[:, :6, 0], 0.)
        delta = rv[:, None]*weights[:, 0]
        delta = delta + torch.cat((torch.zeros_like(delta[:, :10]), own*weights[:, 1, 10:]), 1)
        return torch.where(self.reference.active, reference+delta, 0.)

    def forward(self, core, auxiliary, old_state, state, context, ids, position, groups):
        h, reference = self.reference.components(core, auxiliary, old_state, position, ids)
        current = self.current_state(h, state, groups)
        reference = self.initialized_reference(reference, current, ids)
        prediction, beta = self.attention(h, reference, old_state, current, context, ids)
        return prediction, beta, h, reference


def build_slope_state_attention(*args, **kwargs):
    return SlopeStateAttention(*args, **kwargs)
