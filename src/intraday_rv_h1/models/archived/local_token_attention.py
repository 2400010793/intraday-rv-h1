"""Isolated own-state/value fusion, one recent patch, and separated time tokens."""
from __future__ import annotations

import torch
from torch import nn

from .group_initialized_attention import GroupInitializedHARNet, GroupStateAttention
from .percentile_refit import refitted_coefficients

# Feature-column -> existing percentile-token index; never substitute RV for SV.
OWN_STATE = {10: 0, 11: 1, 12: 2, 13: 3, 14: 4, 1: 6, 3: 7}
FEATURE_NAMES = ('intercept', 'RV1', 'negative1', 'RV5', 'negative5', 'RV20',
                 'negative20', 'RV80', 'negative80', 'signed1', 'ONDS', 'DS15',
                 'hour', 'fast', 'slow', 'reconstructed')


class OwnValueFusion(nn.Module):
    """One percentile query reads exactly [its percentile, its scalar value].

    Each sample/feature is an independent two-token sequence. The result then
    replaces that percentile token in the global coefficient attention. Global
    coefficient queries can still combine features; this is local input isolation.
    """
    def __init__(self, amplitude):
        super().__init__()
        self.amplitude = bool(amplitude)
        self.value = nn.Sequential(nn.Linear(1, 32, dtype=torch.float64), nn.GELU())
        self.value_type = nn.Parameter(torch.randn(1, 32, dtype=torch.float64)*.02)
        self.norm = nn.LayerNorm(32, dtype=torch.float64)
        self.attention = nn.MultiheadAttention(32, 4, dropout=0., batch_first=True, dtype=torch.float64)

    def forward(self, rank_token, value, available):
        if rank_token.shape != (len(value), 32) or available.shape != value.shape:
            raise ValueError('unaligned own-value fusion inputs')
        safe = torch.where(available, value, torch.zeros_like(value))
        if not torch.isfinite(safe).all():
            raise FloatingPointError('nonfinite available local numeric token')
        numeric = self.value(safe[:, None])+self.value_type
        pair = torch.stack((rank_token, numeric), 1)
        keys = self.norm(pair)
        values = torch.stack((keys[:, 0], numeric), 1) if self.amplitude else keys
        output, _ = self.attention(keys[:, :1], keys, values, need_weights=False)
        fused = rank_token+output[:, 0]
        return torch.where(available[:, None], fused, torch.zeros_like(fused))


class SinglePatchEncoder(nn.Module):
    """Last five observed same-clock sessions x three channels -> one token."""
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(15, 32, dtype=torch.float64)

    def forward(self, history):
        if history is None or history.ndim != 3 or history.shape[1:] != (80, 3):
            raise ValueError('single patch requires observed [batch,80,3] history')
        recent = history[:, -5:].to(torch.float64)
        if not torch.isfinite(recent).all():
            raise FloatingPointError('nonfinite observed recent five-session patch')
        # No statistics, lag coordinates, older patches, or internal attention.
        return self.projection(recent.reshape(len(recent), 15))[:, None]


class LocalTokenStateAttention(GroupStateAttention):
    def __init__(self, *args, feature_column=None, numeric=False, patch=False,
                 split_calendar=False, local_amplitude=False, **kwargs):
        super().__init__(*args, **kwargs)
        if feature_column is not None and feature_column not in range(1, 15):
            raise ValueError('local token must name one retained non-intercept feature')
        if numeric and feature_column is None:
            raise ValueError('numeric token requires its own feature column')
        if feature_column is not None and self.numeric_mode != 'none':
            raise ValueError('single-feature experiment requires no other numeric tokens')
        self.feature_column = feature_column
        self.local_numeric = bool(numeric)
        self.single_patch = bool(patch)
        self.split_calendar = bool(split_calendar)
        if feature_column is not None and feature_column not in OWN_STATE:
            self.own_rank_encoder = nn.Sequential(nn.Linear(3, 32, dtype=torch.float64), nn.GELU())
            self.own_rank_type = nn.Parameter(torch.randn(1, 32, dtype=torch.float64)*.02)
        if numeric:
            # The original group protocol preserves amplitude only for IF/IH.
            self.local_fusion = OwnValueFusion(local_amplitude)
        if patch:
            self.patch_encoder = SinglePatchEncoder()
        if split_calendar:
            self.inter_session_encoder = nn.Sequential(nn.Linear(8, 32, dtype=torch.float64),
                                                       nn.GELU(), nn.LayerNorm(32, dtype=torch.float64))

    def feature_state(self, h, old_state, state, ids, own_q):
        """Expose the isolated local result for intervention audits."""
        column = self.feature_column
        if column is None:
            raise ValueError('no local feature configured')
        if column in OWN_STATE:
            slot = OWN_STATE[column]
            # Existing encoders act independently on columns; no self-attention.
            tokens, padding = self.state_tokens(old_state, state)
            rank = tokens[:, 1+slot]
            available = ~padding[:, 1+slot]
        else:
            if own_q.shape != (len(h), 2) or not torch.isfinite(own_q).all():
                raise ValueError('expected [initializer rank, selected own rank]')
            q = own_q[:, 1]
            if ((q < 0) | (q > 1)).any():
                raise ValueError('selected own percentile outside [0,1]')
            rank = self.own_rank_encoder(torch.stack((2*q-1, torch.relu(q-.8)/.2,
                                                      torch.ones_like(q)), 1))+self.own_rank_type
            available = torch.ones(len(h), dtype=torch.bool, device=h.device)
        if self.local_numeric:
            rank = self.local_fusion(rank, h[:, column], available)
        return rank, available

    def conditioned_tokens(self, h, old_state, state, ids, own_q, calendar, history=None, research=None):
        tokens, padding, identity = super().conditioned_tokens(h, old_state, state, ids, own_q, calendar)
        if self.feature_column is not None:
            local, available = self.feature_state(h, old_state, state, ids, own_q)
            if self.feature_column in OWN_STATE:
                tokens = tokens.clone()
                tokens[:, 1+OWN_STATE[self.feature_column]] = local
            else:
                tokens = torch.cat((tokens, local[:, None]), 1)
                padding = torch.cat((padding, ~available[:, None]), 1)
        extra = []
        if self.single_patch:
            extra.append(self.patch_encoder(history))
        if self.split_calendar:
            if calendar.shape != (len(h), 8) or not torch.isfinite(calendar).all():
                raise ValueError('invalid separate past-session calendar token')
            extra.append(self.inter_session_encoder(calendar)[:, None])
        for item in extra:
            tokens = torch.cat((tokens, item), 1)
            padding = torch.cat((padding, padding.new_zeros(item.shape[:2])), 1)
        return tokens, padding, identity


class LocalTokenHARNet(GroupInitializedHARNet):
    def __init__(self, *args, feature_column=None, numeric=False, patch=False,
                 split_calendar=False, selected_values=None, selected_counts=None,
                 local_amplitude=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_column = feature_column
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(args[1]))
            self.attention = LocalTokenStateAttention(self.attention.config, args[2], args[3], args[4],
                numeric_mode=kwargs['numeric_mode'], feature_column=feature_column,
                numeric=numeric, patch=patch, split_calendar=split_calendar,
                local_amplitude=local_amplitude)
        if feature_column is not None and feature_column not in OWN_STATE:
            values = torch.as_tensor(selected_values, dtype=torch.float64)
            counts = torch.as_tensor(selected_counts, dtype=torch.int64)
            if values.ndim != 2 or counts.shape != (len(values),) or len(values) != len(self.own_assets) or (counts <= 0).any():
                raise ValueError('invalid selected asset-clock ECDF')
            self.register_buffer('selected_values', values, persistent=False)
            self.register_buffer('selected_counts', counts, persistent=False)

    def selected_percentile(self, h, groups):
        value = h[:, self.feature_column, None].detach().contiguous()
        ordered = self.selected_values[groups].contiguous()
        left = torch.searchsorted(ordered, value, right=False).squeeze(-1)
        right = torch.searchsorted(ordered, value, right=True).squeeze(-1)
        return (left.double()+.5*(right-left).double()+.5)/(self.selected_counts[groups]+1)

    def attention_inputs(self, h, ids, own_groups, initializer_q):
        if self.feature_column is None or self.feature_column in OWN_STATE:
            return initializer_q
        return torch.stack((initializer_q[:, 0], self.selected_percentile(h, own_groups)), 1)

    def forward(self, core, auxiliary, old_state, state, context, ids, position,
                groups, own_groups, calendar, research=None):
        h, original = self.reference.components(core, auxiliary, old_state, position, ids)
        q = self.own_percentiles(h, ids, own_groups)
        reference = refitted_coefficients(original, q, self.initial_weights[ids],
            columns=(2, 8), response='linear', connection='own_slope')
        current = self.current_state(h, state, groups)
        attention_q = self.attention_inputs(h, ids, own_groups, q)
        prediction, beta = self.attention(h, reference, old_state, current, context, ids,
                                         own_q=attention_q, calendar=calendar, history=core, research=research)
        return prediction, beta, h, reference
