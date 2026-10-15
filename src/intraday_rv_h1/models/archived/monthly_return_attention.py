"""Frozen signed-return Ridge coefficients followed by trainable attention.

The return adjustment belongs to the ONDS reference coefficient. It uses the
current convolutional representation, so gradients reach the original kernels.
No optimizer parameter, attention limit, intercept or residual head is added.
"""
from __future__ import annotations
import torch
from .return_percentile_attention import ReturnPercentileHARNet
from .return_onds_initialization import window_basis
from .percentile_refit import refitted_coefficients

SPEC = dict(windows=(3, 1), mode='rank_relu2', column=10, dimension=4)


class MonthlyReturnHARNet(ReturnPercentileHARNet):
    def __init__(self, *args, **kwargs):
        if kwargs.get('return_state_count') != 5 or kwargs.get('return_state_width') != 3:
            raise ValueError('monthly reference requires the archived five signed-return windows')
        super().__init__(*args, **kwargs)
        self.register_buffer('return_ridge_weights', torch.zeros(4, dtype=torch.float64))
        self.register_buffer('return_zero_ranks', torch.full((6, 5), .5, dtype=torch.float64))

    def set_return_initializer(self, weights, zero_ranks):
        weights = torch.as_tensor(weights, dtype=torch.float64)
        zero_ranks = torch.as_tensor(zero_ranks, dtype=torch.float64)
        if (weights.shape != (4,) or zero_ranks.shape != (6, 5)
                or not torch.isfinite(weights).all() or not torch.isfinite(zero_ranks).all()
                or (zero_ranks <= 0).any() or (zero_ranks >= 1).any()):
            raise ValueError('invalid fixed monthly return Ridge parameters')
        with torch.no_grad():
            self.return_ridge_weights.copy_(weights)
            self.return_zero_ranks.copy_(zero_ranks)

    def forward(self, core, auxiliary, old_state, state, context, ids, position,
                groups, own_groups, calendar, return_state=None):
        h, original = self.reference.components(core, auxiliary, old_state, position, ids)
        q = self.own_percentiles(h, ids, own_groups)
        reference = refitted_coefficients(original, q, self.initial_weights[ids],
            columns=(2, 8), response='linear', connection='own_slope')
        response = window_basis(return_state, self.return_zero_ranks[ids], SPEC)
        reference = reference.clone()
        reference[:, 10] += response @ self.return_ridge_weights
        current = self.current_state(h, state, groups)
        prediction, beta = self.attention(h, reference, old_state, current, context, ids,
            own_q=q, calendar=calendar, history=core, research=return_state)
        return prediction, beta, h, reference
