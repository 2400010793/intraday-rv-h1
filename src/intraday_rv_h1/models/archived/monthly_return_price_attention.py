"""Trainable HAR representation with frozen signed-return and price Ridge heads."""
import torch

from .monthly_return_attention import MonthlyReturnHARNet
from .percentile_refit import refitted_coefficients
from .price_state_initialization import response


class MonthlyReturnPriceHARNet(MonthlyReturnHARNet):
    def __init__(self, *args, price_column=1, **kwargs):
        if price_column not in (0, 1):
            raise ValueError('price initialization must adjust intercept or RV1')
        super().__init__(*args, **kwargs)
        self.price_column = price_column
        self.register_buffer('signed_return_weights', torch.zeros(4, dtype=torch.float64))
        self.register_buffer('price_weights', torch.zeros(2, dtype=torch.float64))

    def set_state_initializer(self, signed_returns, price):
        for name, value, width in (('signed_return_weights', signed_returns, 4),
                                   ('price_weights', price, 2)):
            weights = torch.as_tensor(value, dtype=torch.float64)
            if weights.shape != (width,) or not torch.isfinite(weights).all():
                raise ValueError(f'invalid frozen state initializer:{name}')
            getattr(self, name).copy_(weights)

    def forward(self, core, auxiliary, old_state, state, context, ids, position,
                groups, own_groups, calendar, return_state=None, coefficient_state=None):
        if coefficient_state is None or coefficient_state.shape != (len(core), 3, 3):
            raise ValueError('coefficient state requires raw overnight/opening15 and price20 triples')
        h, original = self.reference.components(core, auxiliary, old_state, position, ids)
        q = self.own_percentiles(h, ids, own_groups)
        reference = refitted_coefficients(original, q, self.initial_weights[ids],
            columns=(2, 8), response='linear', connection='own_slope').clone()
        reference[:, 10] += response(coefficient_state, (0, 1)) @ self.signed_return_weights
        reference[:, self.price_column] += response(coefficient_state, (2,)) @ self.price_weights
        current = self.current_state(h, state, groups)
        prediction, beta = self.attention(h, reference, old_state, current, context, ids,
            own_q=q, calendar=calendar, history=core, research=return_state)
        return prediction, beta, h, reference
