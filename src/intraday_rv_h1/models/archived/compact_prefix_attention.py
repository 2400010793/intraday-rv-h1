"""Attention-only prefix state deletion with the numerical reference retained."""
from __future__ import annotations

import torch

from .directional_coefficient_attention import DirectionalAttentionConfig, INITIAL_KNOTS
from .original_prefix_ablation import OriginalPrefixAttention
from .trainable_prefix_attention import TrainablePrefixAttention

PREFIX_STATES = ("both", "fast_only", "slow_only", "none")


class CompactPrefixStateAttention(OriginalPrefixAttention):
    def __init__(self, config, limits, descriptors, variant, prefix_state):
        if prefix_state not in PREFIX_STATES:
            raise ValueError(f"unknown prefix state: {prefix_state}")
        super().__init__(config, limits, descriptors, variant)
        if config.percentile_mode != "percentile" or config.tokens != 8:
            raise ValueError("compact prefix state requires the original eight percentile tokens")
        self.prefix_state = prefix_state
        self.removed_tokens = tuple(k for k, keep in ((3, prefix_state in ("both", "fast_only")),
                                                      (4, prefix_state in ("both", "slow_only"))) if not keep)

    def state_tokens(self, old_state, state):
        if not self.removed_tokens:
            return super().state_tokens(old_state, state)
        # Sanitize before inherited finite checks. This is the attention copy;
        # the reference keeps its separate original ranks and numerical inputs.
        old_state, state = old_state.clone(), state.clone()
        for token in self.removed_tokens:
            state[:, token] = 0.
            old_state[:, 3+3*token:6+3*token] = 0.
        tokens, padding = super().state_tokens(old_state, state)
        time_offset = int(self.config.time_mode != "none")
        retained = [k for k in range(tokens.shape[1]) if k-time_offset not in self.removed_tokens]
        return tokens[:, retained], padding[:, retained]

    def regularization(self):
        if not self.removed_tokens:
            return super().regularization()
        keep = self.prefix_active[12:16].clone()
        for token in self.removed_tokens:
            keep[token-2] = False
        difference = self.knots()-self.low_logits.new_tensor(INITIAL_KNOTS)
        return self.config.knot_penalty*difference[keep].square().sum()


def build_compact_prefix_attention(config, seed, limits, descriptors, variant, prefix_state):
    if isinstance(config, dict): config = DirectionalAttentionConfig(**config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return CompactPrefixStateAttention(config, limits, descriptors, variant, prefix_state)


class CompactTrainablePrefixAttention(TrainablePrefixAttention):
    def __init__(self, config, seed, limits, descriptors, variant, reference_state,
                 rank_values, rank_counts, *, prefix_state="both"):
        super().__init__(config, seed, limits, descriptors, variant, reference_state, rank_values, rank_counts)
        self.attention = build_compact_prefix_attention(config, seed, limits, descriptors, variant, prefix_state)


def build_compact_trainable_prefix(config, seed, limits, descriptors, variant, reference_state,
                                   rank_values, rank_counts, *, prefix_state="both", train_assets=None):
    model = CompactTrainablePrefixAttention(config, seed, limits, descriptors, variant,
                                            reference_state, rank_values, rank_counts, prefix_state=prefix_state)
    allowed = set(range(len(limits))) if train_assets is None else set(train_assets)
    if not allowed or min(allowed) < 0 or max(allowed) >= len(limits):
        raise ValueError("invalid trainable asset set")
    for asset, local in enumerate(model.reference.locals):
        for parameter in local.parameters():
            parameter.requires_grad_(asset in allowed)
        local.output.weight.requires_grad_(False)
    return model
