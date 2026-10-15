"""Explicit short-RV replacement in both Ridge and attention forward paths."""
import torch

from .monthly_return_price_attention import MonthlyReturnPriceHARNet
from .percentile_refit import refitted_coefficients
from .price_state_initialization import response


def replace_policy(auxiliary, old_state, state, policy):
    """policy[n,2,3] = normalized value, same-clock centered rank, availability."""
    if policy.shape != (len(auxiliary), 2, 3) or not torch.isfinite(policy).all():
        raise ValueError('demo policy requires finite [N,2,3] input')
    values, ranks, flags = policy.unbind(-1)
    if not ((flags == 0) | (flags == 1)).all() or (ranks.abs() > 1).any():
        raise ValueError('invalid demo availability or centered percentile')
    auxiliary, old_state, state = auxiliary.clone(), old_state.clone(), state.clone()
    auxiliary[:, :2] = values
    old_state[:, [3, 6]] = ranks
    # Same directional response as the original six micro-state tokens.
    state[:, :2, 0] = ranks
    state[:, :2, 1] = torch.relu(ranks - .6) / .4
    state[:, :2, 2] = flags
    old_state[:, 3:9] = state[:, :2].flatten(1)
    return auxiliary, old_state, state


class MonthlyDemoHARNet(MonthlyReturnPriceHARNet):
    def forward(self, core, auxiliary, old_state, state, context, ids, position,
                groups, own_groups, calendar, return_state=None, coefficient_state=None,
                demo_policy=None):
        if coefficient_state is None or coefficient_state.shape != (len(core), 3, 3):
            raise ValueError('coefficient state requires raw returns and price triples')
        auxiliary, old_state, state = replace_policy(auxiliary, old_state, state, demo_policy)
        h, original = self.reference.components(core, auxiliary, old_state, position, ids)
        q = self.own_percentiles(h, ids, own_groups)
        q = torch.stack((q[:, 0], torch.full_like(q[:, 1], .5)), 1)
        reference = refitted_coefficients(original, q, self.initial_weights[ids],
            columns=(2, 8), response='linear', connection='own_slope').clone()
        reference[:, 10] += response(coefficient_state, (0, 1)) @ self.signed_return_weights
        reference[:, 0] += response(coefficient_state, (2,)) @ self.price_weights
        current = self.current_state(h, state, groups)
        prediction, beta = self.attention(h, reference, old_state, current, context, ids,
            own_q=q, calendar=calendar, history=core, research=return_state)
        return prediction, beta, h, reference


class MonthlyPolicyHARNet(MonthlyReturnPriceHARNet):
    """Replace policy inputs without deleting either downside percentile."""
    def forward(self,core,auxiliary,old_state,state,context,ids,position,
                groups,own_groups,calendar,return_state=None,coefficient_state=None,demo_policy=None):
        auxiliary,old_state,state=replace_policy(auxiliary,old_state,state,demo_policy)
        return super().forward(core,auxiliary,old_state,state,context,ids,position,
                               groups,own_groups,calendar,return_state,coefficient_state)
