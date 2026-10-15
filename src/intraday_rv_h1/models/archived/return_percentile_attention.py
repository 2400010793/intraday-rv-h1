"""One signed-return token: percentile alone or fused with the raw log return."""
from __future__ import annotations
import torch
from torch import nn
from .stacked_coefficient_attention import StackedHARNet, StackedStateAttention
from .local_regime_attention import LocalRegimeStateAttention


class ReturnPercentileStateAttention(StackedStateAttention):
    def __init__(self, *args, return_column=None, return_state_count=3,
                 return_state_width=2, return_input='rank', **kwargs):
        super().__init__(*args, **kwargs)
        if (not isinstance(return_state_count,int) or return_state_count<1
                or (return_column is not None and (not isinstance(return_column,int) or not 0<=return_column<return_state_count))):
            raise ValueError('unknown return percentile column')
        self.return_column = return_column
        self.return_state_count = return_state_count
        if return_input not in ('rank', 'rank_raw') or return_state_width not in (2, 3):
            raise ValueError('unknown return token layout or input')
        if return_input == 'rank_raw' and return_state_width != 3:
            raise ValueError('raw return fusion requires [percentile, availability, raw log return]')
        self.return_state_width = return_state_width
        self.return_input = return_input
        if return_column is not None:
            self.return_encoder = nn.Sequential(nn.Linear(1, 32, dtype=torch.float64), nn.GELU())
            self.return_type = nn.Parameter(torch.randn(1, 32, dtype=torch.float64)*.02)
            if return_input == 'rank_raw':
                # Preserve every common parameter and its RNG sequence. Adding
                # this projection is equivalent to a two-input linear encoder;
                # its raw input retains the production sign and units.
                self.return_raw_encoder = nn.Linear(1, 32, bias=False, dtype=torch.float64)

    def return_token(self, research):
        if research is None or research.ndim != 3 or research.shape[1:] != (self.return_state_count, self.return_state_width):
            raise ValueError(f'return state requires {self.return_state_count} [percentile, availability] pairs')
        q, flag = research[:, self.return_column, :2].unbind(-1)
        available = flag.eq(1)
        if not (flag.eq(0) | available).all():
            raise ValueError('return percentile availability must be binary')
        safe = torch.where(available, q, torch.full_like(q, .5))
        if not torch.isfinite(safe).all() or ((safe < 0) | (safe > 1)).any():
            raise ValueError('invalid available signed-return percentile')
        encoded = self.return_encoder[0]((2*safe-1)[:, None])
        if self.return_input == 'rank_raw':
            raw = research[:, self.return_column, 2]
            raw = torch.where(available, raw, torch.zeros_like(raw))
            if not torch.isfinite(raw).all():
                raise ValueError('invalid available raw signed return')
            encoded = encoded + self.return_raw_encoder(raw[:, None])
        token = self.return_encoder[1](encoded)+self.return_type
        return torch.where(available[:, None], token, torch.zeros_like(token)), available

    def conditioned_tokens(self, h, old_state, state, ids, own_q, calendar, history=None, research=None):
        # Bypass the parent's already-mixed result: append, then mix exactly once.
        tokens, padding, identity = LocalRegimeStateAttention.conditioned_tokens(
            self, h, old_state, state, ids, own_q, calendar, history, research=None)
        if self.return_column is not None:
            if research is None or len(research) != len(h):
                raise ValueError('misaligned return state')
            token, available = self.return_token(research)
            tokens = torch.cat((tokens, token[:, None]), 1)
            padding = torch.cat((padding, ~available[:, None]), 1)
        return self.mix_state(tokens, padding), padding, identity


class ReturnPercentileHARNet(StackedHARNet):
    def __init__(self, *args, return_column=None, return_state_count=3,
                 return_state_width=2, return_input='rank', **kwargs):
        kwargs.update(topology='state_cross', regime_mode=None)
        super().__init__(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(args[1]))
            self.attention = ReturnPercentileStateAttention(self.attention.config, args[2], args[3], args[4],
                numeric_mode=kwargs['numeric_mode'], split_calendar=kwargs.get('split_calendar', False),
                local_amplitude=kwargs.get('local_amplitude', False), topology='state_cross',
                regime_mode=None, return_column=return_column, return_state_count=return_state_count,
                return_state_width=return_state_width, return_input=return_input)

    def forward(self, core, auxiliary, old_state, state, context, ids, position,
                groups, own_groups, calendar, return_state=None):
        # Explicit return input; the inherited internal relay is named research.
        return super().forward(core, auxiliary, old_state, state, context, ids,
                               position, groups, own_groups, calendar, research=return_state)
