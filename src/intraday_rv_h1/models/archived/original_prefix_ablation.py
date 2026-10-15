"""Prefix deletion in the frozen original shared-identity three-expert family."""
from __future__ import annotations

import numpy as np
import torch

from .directional_coefficient_attention import DirectionalAttentionConfig, INITIAL_KNOTS
from .identity_token_attention import IdentityTokenCoefficientAttention
from .svj_coefficient_conditioning import reference_coefficients

VARIANTS = ("all3", "drop_reconstructed", "no_prefix")


def feature_mask(variant):
    if variant not in VARIANTS:
        raise ValueError(f"unknown original-prefix variant: {variant}")
    mask = np.ones(16, dtype=bool)
    if variant == "drop_reconstructed":
        mask[15] = False
    elif variant == "no_prefix":
        mask[13:16] = False
    return mask


def ablated_reference(features, clock, rank, position, head, six, variant):
    """Recompute the *original* common tanh, without fitting any parameters.

    A cached full reference cannot just be sliced: its common tanh denominator
    transmits deleted prefixes to the policy and 1H coefficients.
    """
    active = feature_mask(variant)
    h = np.where(active[None], features, 0.)
    r = np.where(active[None, 10:], rank, 0.)
    beta = reference_coefficients(h, clock, r, position, head, six)
    return np.where(active[None], beta, 0.)


class OriginalPrefixAttention(IdentityTokenCoefficientAttention):
    def __init__(self, config, limits, descriptors, variant):
        super().__init__(config, limits, descriptors, identity_mode="shared")
        self.variant = variant
        # No RNG draw or extra persisted key: archived checkpoints load strictly
        # and all variants retain exactly the original allocated parameters.
        self.register_buffer("prefix_active", torch.from_numpy(feature_mask(variant)), persistent=False)

    def state_tokens(self, old_state, state):
        if self.variant != "all3":
            old_state, state = old_state.clone(), state.clone()
            for column in torch.nonzero(~self.prefix_active).flatten().tolist():
                token = column - 10
                state[:, token] = 0.
                old_state[:, 3 + 3*token:6 + 3*token] = 0.
        return super().state_tokens(old_state, state)

    def forward(self, features, reference, old_state, state, context, asset_ids, scale_context=None):
        if self.variant == "all3":
            return super().forward(features, reference, old_state, state, context, asset_ids, scale_context)
        h = torch.where(self.prefix_active[None], features, torch.zeros_like(features))
        ref = torch.where(self.prefix_active[None], reference, torch.zeros_like(reference))
        _, beta = super().forward(h, ref, old_state, state, context, asset_ids, scale_context)
        beta = torch.where(self.prefix_active[None], beta, torch.zeros_like(beta))
        return torch.exp((h * beta).sum(-1)/2), beta

    def coefficient_regularization(self, features, delta):
        delta = torch.where(self.prefix_active[None], delta, torch.zeros_like(delta))
        return super().coefficient_regularization(features, delta)

    def regularization(self):
        if self.variant == "all3":
            return super().regularization()
        difference = self.knots() - self.low_logits.new_tensor(INITIAL_KNOTS)
        # Four micro rows correspond to 1H, fast, slow, reconstructed. Preserve
        # the original sum/weight on every retained knot, remove only dead rows.
        return self.config.knot_penalty * (difference[self.prefix_active[12:16]]**2).sum()


def build_original_prefix(config, seed, limits, descriptors, variant):
    if isinstance(config, dict):
        config = DirectionalAttentionConfig(**config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return OriginalPrefixAttention(config, limits, descriptors, variant)
