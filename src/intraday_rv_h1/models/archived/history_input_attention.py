"""Current values and causal H80 patches as inputs to coefficient attention."""
from __future__ import annotations

import torch
from torch import nn

from .compact_prefix_attention import CompactPrefixStateAttention, CompactTrainablePrefixAttention
from .directional_coefficient_attention import DirectionalAttentionConfig

VALUE_MODES = ("values", "constant_values", "values_amplitude")
INPUT_MODES = (*VALUE_MODES, "history", "interaction")


class HistoryInputAttention(CompactPrefixStateAttention):
    """Retain state encoders while giving K/V actual values and ordered history.

    Each history token retains five consecutive trading-session observations
    of all three standardized SVJ channels. All 80 observations are known at
    the forecast origin. The optional interaction block operates within one
    origin only, never across batch rows or symbols. Final expert projections
    remain zero initialized, and the original total coefficient cap is used.
    """

    def __init__(self, config, limits, descriptors, variant, *, mode, prefix_state="both"):
        if mode not in INPUT_MODES:
            raise ValueError(f"unknown history input mode: {mode}")
        super().__init__(config, limits, descriptors, variant, prefix_state)
        # The H3 state pilot can compare the historical full3 clock token with
        # a scalar position token or no time token.  History inputs themselves
        # still retain the original complete prefix state.
        if prefix_state != "both":
            raise ValueError("history pilot retains the complete prefix state")
        self.input_mode = mode
        active = torch.nonzero(self.prefix_active).flatten()
        active = active[active != 0]
        self.register_buffer("value_indices", active)
        self.value_encoder = nn.Sequential(nn.Linear(1, 32, dtype=torch.float64), nn.GELU())
        self.value_types = nn.Parameter(torch.randn(len(active), 32, dtype=torch.float64) * .02)
        if mode not in VALUE_MODES:
            self.history_encoder = nn.Sequential(nn.Linear(15, 32, dtype=torch.float64), nn.GELU())
            # Fixed relative lag coordinates distinguish oldest/newest patches;
            # these are session lags, independent of intraday clock encoding.
            ages = torch.stack((torch.linspace(-1, 0, 16, dtype=torch.float64),
                                torch.linspace(1, 0, 16, dtype=torch.float64).square()), -1)
            self.register_buffer("history_ages", ages)
            self.age_encoder = nn.Linear(2, 32, bias=False, dtype=torch.float64)
            self.history_type = nn.Parameter(torch.randn(1, 1, 32, dtype=torch.float64) * .02)
        if mode == "interaction":
            self.input_interaction = nn.TransformerEncoderLayer(
                32, 4, 64, dropout=0., activation="gelu", batch_first=True,
                norm_first=True, dtype=torch.float64)

    def value_tokens(self, features, state):
        if features.shape != (len(state), 16) or state.shape[1:] != (8, 3):
            raise ValueError("invalid value token schema")
        availability = torch.ones_like(features, dtype=torch.bool)
        availability[:, 10:16] = state[:, :6, 2] > 0
        selected = features.index_select(1, self.value_indices)
        available = availability.index_select(1, self.value_indices)
        safe = torch.where(available, selected, torch.zeros_like(selected))
        if self.input_mode == "constant_values":
            safe = torch.zeros_like(safe)
        if not torch.isfinite(safe).all():
            raise FloatingPointError("nonfinite available attention value")
        return self.value_encoder(safe[..., None]) + self.value_types, ~available

    def attention_kv(self, tokens):
        """Normalize keys; the amplitude arm retains numeric values before LN.

        LayerNorm is not an exact erasure of scalar information, but it can
        attenuate token magnitude. This arm changes only numeric V normalization
        and introduces no parameters or fitted/test-derived statistics.
        """
        keys = self.head.norm_tokens(tokens)
        if self.input_mode != "values_amplitude":
            return keys, keys
        offset = 1 + self.config.tokens  # original full3 clock + percentile tokens
        stop = offset + len(self.value_indices)
        values = torch.cat((keys[:, :offset], tokens[:, offset:stop], keys[:, stop:]), 1)
        return keys, values

    def history_tokens(self, history):
        if history is None or history.ndim != 3 or history.shape[1:] != (80, 3):
            raise ValueError("history attention requires origin-known [batch,80,3]")
        if not torch.isfinite(history).all():
            raise FloatingPointError("nonfinite causal H80 history")
        patches = history.to(torch.float64).reshape(len(history), 16, 15)
        return self.history_encoder(patches) + self.age_encoder(self.history_ages)[None] + self.history_type

    def input_tokens(self, features, old_state, state, asset_ids, history, *, blocked_tokens=None):
        states, padding = self.state_tokens(old_state, state)
        values, value_padding = self.value_tokens(features, state)
        identity = self.encode_identity(asset_ids)
        tokens = torch.cat((states, values, identity[:, None] + self.identity_token_type), 1)
        padding = torch.cat((padding, value_padding, padding.new_zeros((len(state), 1))), 1)
        if self.input_mode not in VALUE_MODES:
            patches = self.history_tokens(history)
            if len(patches) != len(features):
                raise ValueError("history batch differs from attention values")
            tokens = torch.cat((tokens, patches), 1)
            padding = torch.cat((padding, padding.new_zeros((len(state), 16))), 1)
        if blocked_tokens is not None:
            padding=padding.clone();padding[:,blocked_tokens]=True
            tokens=torch.where(padding[...,None],torch.zeros_like(tokens),tokens)
        if self.input_mode == "interaction":
            tokens = self.input_interaction(tokens, src_key_padding_mask=padding)
        return tokens, padding, identity

    def forward(self, features, reference, old_state, state, context, asset_ids,
                scale_context=None, *, history=None, blocked_tokens=None, token_features=None):
        if scale_context is not None:
            raise ValueError("history input attention has no scale-query context")
        if features.shape != (len(asset_ids), 16) or reference.shape != features.shape:
            raise ValueError("history attention requires aligned 16-column values/reference")
        if asset_ids.dtype != torch.int64 or (asset_ids < 0).any() or (asset_ids >= len(self.asset_limits)).any():
            raise ValueError("invalid history attention asset identities")
        h = torch.where(self.prefix_active[None], features, torch.zeros_like(features))
        ref = torch.where(self.prefix_active[None], reference, torch.zeros_like(reference))
        if not torch.isfinite(h).all() or not torch.isfinite(ref).all():
            raise FloatingPointError("nonfinite active history attention design/reference")
        token_h = h if token_features is None else token_features
        tokens, padding, identity = self.input_tokens(token_h, old_state, state, asset_ids, history,blocked_tokens=blocked_tokens)
        queries = self.head.norm_query(self.head.queries)[None].expand(len(h), -1, -1)
        keys, values = self.attention_kv(tokens)
        attended, _ = self.head.attention(queries, keys, values, key_padding_mask=padding, need_weights=False)
        decoder = torch.cat((attended, identity[:, None].expand(-1, 16, -1), queries), -1)
        raw = torch.stack([layer(decoder).squeeze(-1) for layer in self.expert_layers], -1).mean(-1)
        delta = self.asset_limits[asset_ids] * torch.tanh(raw)
        beta = torch.where(self.prefix_active[None], ref + delta, torch.zeros_like(ref))
        prediction = torch.exp((h * beta).sum(-1) / 2)
        if not torch.isfinite(prediction).all() or (prediction <= 0).any():
            raise FloatingPointError("invalid history attention forecast")
        return prediction, beta


class HistoryInputHARNet(CompactTrainablePrefixAttention):
    def __init__(self, config, seed, limits, descriptors, variant, reference_state,
                 rank_values, rank_counts, *, mode, train_assets):
        super().__init__(config, seed, limits, descriptors, variant, reference_state, rank_values, rank_counts)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.attention = HistoryInputAttention(config, limits, descriptors, variant, mode=mode)
        allowed = set(train_assets)
        if not allowed or min(allowed) < 0 or max(allowed) >= len(limits):
            raise ValueError("invalid history attention trainable assets")
        for asset, local in enumerate(self.reference.locals):
            for parameter in local.parameters():
                parameter.requires_grad_(asset in allowed)
            local.output.weight.requires_grad_(False)

    def forward(self, core, auxiliary, old_state, state, context, ids, position, groups):
        h, reference = self.reference.components(core, auxiliary, old_state, position, ids)
        current = self.current_state(h, state, groups)
        prediction, beta = self.attention(h, reference, old_state, current, context, ids, history=core)
        return prediction, beta, h, reference


def build_history_input_attention(config, seed, limits, descriptors, variant,
                                  reference_state, rank_values, rank_counts, *, mode, train_assets):
    if isinstance(config, dict):
        config = DirectionalAttentionConfig(**config)
    return HistoryInputHARNet(config, seed, limits, descriptors, variant, reference_state,
                             rank_values, rank_counts, mode=mode, train_assets=train_assets)
