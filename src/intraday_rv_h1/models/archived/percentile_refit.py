"""Joint epoch-zero refits of explicit slopes and their own percentile links."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .original_prefix_ablation import feature_mask
from .slope_state_attention import solve_state_ridge
from .trainable_prefix_attention import TrainablePrefixReference

KNOTS = (.2, .6, .8)


def response_width(response):
    if response not in ('none', 'linear', 'piecewise'):
        raise ValueError(f'unknown percentile response: {response}')
    return {'none': 0, 'linear': 1, 'piecewise': 4}[response]


def percentile_basis(q, response):
    """Continuous responses centered at q=.5; no independent free constant."""
    width = response_width(response)
    if not width:
        return q[..., None][..., :0]
    values = [2*q-1]
    if response == 'piecewise':
        relu = torch.relu if isinstance(q, torch.Tensor) else lambda x: np.maximum(x, 0.)
        values.extend((relu(q-k)-max(.5-k, 0.))/(1-k) for k in KNOTS)
    return torch.stack(values, -1) if isinstance(q, torch.Tensor) else np.stack(values, -1)


def checked_columns(columns):
    cols = tuple(columns)
    active = feature_mask('drop_reconstructed')
    if (not cols or len(set(cols)) != len(cols)
            or any(not isinstance(j, (int, np.integer)) or j <= 0 or j >= 16 or not active[j] for j in cols)):
        raise ValueError('joint refit requires unique retained non-intercept columns')
    return cols


def refit_design(features, percentile, *, columns, response='linear', connection='own_slope'):
    """Base slope corrections, followed by feature-major percentile terms."""
    cols = checked_columns(columns)
    h = np.asarray(features, np.float64); q = np.asarray(percentile, np.float64)
    if h.ndim != 2 or h.shape[1] != 16 or not len(h) or q.shape != (len(h), len(cols)):
        raise ValueError('joint refit needs aligned [N,16] features and [N,J] own ranks')
    if connection not in ('own_slope', 'direct'):
        raise ValueError(f'unknown percentile connection: {connection}')
    value = h[:, cols]
    if (not np.isfinite(value).all() or not np.isfinite(q).all() or ((q < 0) | (q > 1)).any()):
        raise FloatingPointError('invalid available joint refit features/percentiles')
    basis = percentile_basis(q, response)
    terms = basis * value[..., None] if connection == 'own_slope' else basis
    return np.column_stack((value, terms.reshape(len(h), -1)))


def fit_percentile_refit(features, percentile, eta_reference, target, *, columns,
                         response='linear', connection='own_slope', ridge=.01, chunk_size=8192):
    """Fit shared signed increments around asset-specific original coefficients.

    Original intercepts, time terms and common-tanh response stay fixed. Only
    declared constant slopes and their percentile terms are jointly refitted.
    The selected base coefficients become beta_original + fitted correction.
    """
    cols = checked_columns(columns); width = len(cols)*(1+response_width(response))
    h = np.asarray(features); q = np.asarray(percentile); eta = np.asarray(eta_reference, float)
    y = np.asarray(target, float)
    if (h.ndim != 2 or h.shape[1] != 16 or not len(h) or q.shape != (len(h), len(cols))
            or eta.shape != (len(h),) or y.shape != eta.shape or chunk_size < 1):
        raise ValueError('misaligned joint percentile fitting rows')
    if not np.isfinite(eta).all() or not np.isfinite(y).all() or (y <= 0).any():
        raise FloatingPointError('invalid joint percentile fitting labels/reference')
    gram = np.zeros((width, width)); rhs = np.zeros(width)
    for low in range(0, len(h), chunk_size):
        sl = slice(low, low+chunk_size)
        x = refit_design(h[sl], q[sl], columns=cols, response=response, connection=connection)
        gram += x.T @ x; rhs += x.T @ (2*np.log(y[sl])-eta[sl])
    weights, audit = solve_state_ridge(gram, rhs, len(h), ridge=ridge)
    return weights, dict(audit, columns=cols, response=response, connection=connection,
                         epochs=0, gradient_updates=0, optimizer_steps=0,
                         target_space='log variance', base_slopes_refitted=True,
                         asset_reference_differences_preserved=True, knots=KNOTS if response == 'piecewise' else ())


def refitted_coefficients(reference, percentile, weights, *, columns, response, connection):
    """Torch replay of the same explicit coefficient fit, including direct heads."""
    cols = checked_columns(columns); width = response_width(response); n = len(reference)
    if reference.shape != (n, 16) or percentile.shape != (n, len(cols)):
        raise ValueError('misaligned joint coefficient replay')
    w = torch.as_tensor(weights, dtype=reference.dtype, device=reference.device)
    if w.ndim == 1:
        w = w.expand(n, -1)
    if w.shape != (n, len(cols)*(1+width)) or not torch.isfinite(w).all():
        raise ValueError('invalid joint percentile weights')
    if (not torch.isfinite(percentile).all() or ((percentile < 0) | (percentile > 1)).any()
            or connection not in ('own_slope', 'direct')):
        raise ValueError('invalid joint percentile replay state/connection')
    beta = reference.clone(); beta[:, cols] += w[:, :len(cols)]
    if width:
        delta = (percentile_basis(percentile, response)*w[:, len(cols):].reshape(n, len(cols), width)).sum(-1)
        if connection == 'own_slope':
            beta[:, cols] += delta
        else:
            beta[:, 0] += delta.sum(-1)
    return beta


class JointPercentileInitializedReference(nn.Module):
    """Frozen H80 model with separately stored asset/clock percentile registries."""
    def __init__(self, reference_state, weights, columns, response, connection,
                 rank_values, rank_counts, registry_assets):
        super().__init__(); self.columns = checked_columns(columns)
        self.response = response; self.connection = connection
        w = torch.as_tensor(weights, dtype=torch.float64)
        values = torch.as_tensor(rank_values, dtype=torch.float64)
        counts = torch.as_tensor(rank_counts, dtype=torch.int64)
        assets = torch.as_tensor(registry_assets, dtype=torch.int64)
        if (w.ndim != 2 or w.shape[1] != len(self.columns)*(1+response_width(response))
                or not torch.isfinite(w).all() or connection not in ('own_slope', 'direct')):
            raise ValueError('invalid asset joint coefficient weights')
        if (values.ndim != 3 or values.shape[1] != len(self.columns) or counts.shape != values.shape[:2]
                or assets.shape != (len(values),) or (counts <= 0).any() or (counts > values.shape[2]).any()
                or (assets < 0).any() or (assets >= len(w)).any()):
            raise ValueError('invalid joint percentile registry layout')
        self.reference = TrainablePrefixReference(len(w), 'drop_reconstructed')
        self.reference.load_state_dict(reference_state); self.reference.requires_grad_(False)
        self.register_buffer('weights', w.clone())
        for name, value in (('rank_values', values), ('rank_counts', counts), ('registry_assets', assets)):
            self.register_buffer(name, value.clone(), persistent=False)

    def forward(self, core, auxiliary, old_state, state, context, ids, position, groups):
        if groups.dtype != torch.int64 or groups.shape != ids.shape or (groups < 0).any() or (groups >= len(self.rank_counts)).any():
            raise ValueError('invalid joint percentile group IDs')
        if not torch.equal(self.registry_assets[groups], ids):
            raise ValueError('joint percentile group belongs to another asset')
        h, reference = self.reference.components(core, auxiliary, old_state, position, ids)
        value = h[:, self.columns, None].contiguous(); ordered = self.rank_values[groups].contiguous()
        left = torch.searchsorted(ordered, value, right=False).squeeze(-1)
        right = torch.searchsorted(ordered, value, right=True).squeeze(-1)
        q = (left.double()+.5*(right-left).double()+.5)/(self.rank_counts[groups]+1)
        beta = refitted_coefficients(reference, q, self.weights[ids], columns=self.columns,
                                     response=self.response, connection=self.connection)
        prediction = torch.exp((h*beta).sum(-1)/2)
        if not torch.isfinite(prediction).all() or (prediction <= 0).any():
            raise FloatingPointError('invalid joint percentile initialized prediction')
        return prediction, beta, h, reference
