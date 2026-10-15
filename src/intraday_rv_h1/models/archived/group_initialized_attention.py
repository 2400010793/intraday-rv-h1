"""Group-shared attention around a sealed own-percentile Ridge initialization."""
from __future__ import annotations

import torch
from torch import nn

from .compact_prefix_attention import CompactTrainablePrefixAttention
from .directional_coefficient_attention import DirectionalAttentionConfig
from .history_input_attention import HistoryInputAttention
from .percentile_refit import refitted_coefficients


class GroupStateAttention(HistoryInputAttention):
    def __init__(self, config, limits, descriptors, variant, *, numeric_mode,
                 calendar_enabled=False, downside_token=False):
        if numeric_mode not in ('none', 'values', 'values_amplitude'):
            raise ValueError(f'unknown group numeric mode: {numeric_mode}')
        super().__init__(config, limits, descriptors, variant,
                         mode='values' if numeric_mode == 'none' else numeric_mode)
        self.numeric_mode = numeric_mode
        self.calendar_enabled = bool(calendar_enabled)
        self.downside_token_enabled = bool(downside_token)
        if numeric_mode == 'none':
            del self.value_encoder, self.value_types
            self.value_indices = self.value_indices[:0]
        if calendar_enabled:
            # Clock3 + weekday/past-gap8 + their explicit 3x8 interactions.
            self.calendar_encoder = nn.Sequential(nn.Linear(35, 32, dtype=torch.float64),
                                                  nn.GELU(), nn.LayerNorm(32, dtype=torch.float64))
        if downside_token:
            self.downside_encoder = nn.Sequential(nn.Linear(4, 32, dtype=torch.float64),
                                                  nn.GELU(), nn.LayerNorm(32, dtype=torch.float64))

    def value_tokens(self, features, state):
        if self.numeric_mode == 'none':
            return features.new_empty((len(features), 0, 32)), state.new_empty((len(state), 0), dtype=torch.bool)
        return super().value_tokens(features, state)

    def conditioned_tokens(self, h, old_state, state, ids, own_q, calendar, history=None, research=None):
        tokens, padding, identity = self.input_tokens(h, old_state, state, ids, None)
        additions = []
        if self.calendar_enabled:
            clock = old_state[:, :3]
            interactions = (clock[:, :, None] * calendar[:, None, :]).flatten(1)
            additions.append(self.calendar_encoder(torch.cat((clock, calendar, interactions), 1)))
        if self.downside_token_enabled:
            q = own_q[:, 0]
            x = torch.stack((h[:, 2], 2*q-1, torch.relu(q-.8)/.2, torch.ones_like(q)), 1)
            additions.append(self.downside_encoder(x))
        if additions:
            extra = torch.stack(additions, 1)
            tokens = torch.cat((tokens, extra), 1)
            padding = torch.cat((padding, padding.new_zeros(extra.shape[:2])), 1)
        return tokens, padding, identity

    def coefficient_attention(self, queries, keys, values, padding):
        """Explicit extension point; the original single read is unchanged."""
        attended, _ = self.head.attention(queries, keys, values,
            key_padding_mask=padding, need_weights=False)
        return attended

    def decode_coefficients(self, decoder, asset_ids):
        """Shared decoder; task-specific heads extend this explicit boundary."""
        return torch.stack([layer(decoder).squeeze(-1) for layer in self.expert_layers], -1).mean(-1)

    def coefficient_delta(self, raw, asset_ids):
        """Original scale and saturation, with an explicit ablation boundary."""
        return self.asset_limits[asset_ids]*torch.tanh(raw)

    def forward(self, features, reference, old_state, state, context, asset_ids,
                *, own_q, calendar, history=None, research=None):
        if features.shape != (len(asset_ids), 16) or reference.shape != features.shape:
            raise ValueError('expected aligned sixteen-column group features/reference')
        if own_q.shape != (len(features), 2) or calendar.shape != (len(features), 8):
            raise ValueError('expected downside1/80 ranks and eight-column observed calendar')
        if not torch.isfinite(own_q).all() or ((own_q < 0) | (own_q > 1)).any():
            raise ValueError('invalid own-feature percentiles')
        h = torch.where(self.prefix_active[None], features, torch.zeros_like(features))
        ref = torch.where(self.prefix_active[None], reference, torch.zeros_like(reference))
        if not torch.isfinite(h).all() or not torch.isfinite(ref).all():
            raise FloatingPointError('nonfinite active group attention inputs')
        if self.calendar_enabled and not torch.isfinite(calendar).all():
            raise FloatingPointError('nonfinite visible calendar state')
        extra_state = {} if research is None else {'research': research}
        tokens, padding, identity = self.conditioned_tokens(h, old_state, state, asset_ids, own_q, calendar, history, **extra_state)
        queries = self.head.norm_query(self.head.queries)[None].expand(len(h), -1, -1)
        keys, values = self.attention_kv(tokens)
        attended = self.coefficient_attention(queries, keys, values, padding)
        decoder = torch.cat((attended, identity[:, None].expand(-1, 16, -1), queries), -1)
        raw = self.decode_coefficients(decoder, asset_ids)
        beta = torch.where(self.prefix_active[None], ref+self.coefficient_delta(raw, asset_ids), 0.)
        prediction = torch.exp((h*beta).sum(-1)/2)
        if not torch.isfinite(prediction).all() or (prediction <= 0).any():
            raise FloatingPointError('invalid initialized group attention prediction')
        return prediction, beta


class GroupInitializedHARNet(CompactTrainablePrefixAttention):
    def __init__(self, config, seed, limits, descriptors, variant, reference_state,
                 rank_values, rank_counts, *, own_values, own_counts, own_assets,
                 initial_weights, train_assets, numeric_mode, calendar_enabled=False,
                 downside_token=False, percentile_cross_columns=(2, 8)):
        if isinstance(config, dict):
            config = DirectionalAttentionConfig(**config)
        super().__init__(config, seed, limits, descriptors, variant, reference_state, rank_values, rank_counts)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.attention = GroupStateAttention(config, limits, descriptors, variant,
                numeric_mode=numeric_mode, calendar_enabled=calendar_enabled, downside_token=downside_token)
        values = torch.as_tensor(own_values, dtype=torch.float64)
        counts = torch.as_tensor(own_counts, dtype=torch.int64)
        assets = torch.as_tensor(own_assets, dtype=torch.int64)
        weights = torch.as_tensor(initial_weights, dtype=torch.float64)
        cross_columns = tuple(int(column) for column in percentile_cross_columns)
        if (values.ndim != 3 or values.shape[1] != 2 or counts.shape != values.shape[:2]
                or assets.shape != (len(values),) or (counts <= 0).any()
                or (counts > values.shape[2]).any() or weights.shape != (len(limits), 4)
                or (assets < 0).any() or (assets >= len(limits)).any() or not torch.isfinite(weights).all()):
            raise ValueError('incompatible sealed downside initialization registry/weights')
        if (not cross_columns or any(column not in (2, 8) for column in cross_columns)
                or len(set(cross_columns)) != len(cross_columns)):
            raise ValueError('percentile cross columns must be a non-empty subset of (2, 8)')
        self.percentile_cross_columns = cross_columns
        for name, value in (('own_values', values), ('own_counts', counts), ('own_assets', assets)):
            self.register_buffer(name, value, persistent=False)
        self.register_buffer('initial_weights', weights.clone())
        allowed = set(train_assets)
        if not allowed or min(allowed) < 0 or max(allowed) >= len(limits):
            raise ValueError('invalid trainable group assets')
        for asset, local in enumerate(self.reference.locals):
            local.requires_grad_(asset in allowed)
            local.output.weight.requires_grad_(False)

    def own_percentiles(self, h, ids, groups):
        if (groups.dtype != torch.int64 or groups.shape != ids.shape
                or (groups < 0).any() or (groups >= len(self.own_counts)).any()
                or not torch.equal(self.own_assets[groups], ids)):
            raise ValueError('own percentile registry belongs to another asset/clock')
        value = h[:, [2, 8], None].detach().contiguous()
        ordered = self.own_values[groups].contiguous()
        left = torch.searchsorted(ordered, value, right=False).squeeze(-1)
        right = torch.searchsorted(ordered, value, right=True).squeeze(-1)
        return (left.double()+.5*(right-left).double()+.5)/(self.own_counts[groups]+1)

    def forward(self, core, auxiliary, old_state, state, context, ids, position,
                groups, own_groups, calendar):
        h, original = self.reference.components(core, auxiliary, old_state, position, ids)
        q = self.own_percentiles(h, ids, own_groups)
        # Keep every numeric feature and its fitted constant correction.  A
        # dropped column is neutralized only for the q-dependent cross-term;
        # q=.5 makes the centered linear percentile response exactly zero.
        reference_q = q.clone()
        for index, column in enumerate((2, 8)):
            if column not in self.percentile_cross_columns:
                reference_q[:, index] = .5
        reference = refitted_coefficients(original, reference_q, self.initial_weights[ids],
            columns=(2, 8), response='linear', connection='own_slope')
        current = self.current_state(h, state, groups)
        attention_q = q.clone()
        for index, column in enumerate((2, 8)):
            if column not in self.percentile_cross_columns:
                attention_q[:, index] = .5
        prediction, beta = self.attention(h, reference, old_state, current, context, ids,
                                           own_q=attention_q, calendar=calendar)
        return prediction, beta, h, reference
