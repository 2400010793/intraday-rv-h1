"""Optional observed-regime token after isolated own-value input fusion."""
from __future__ import annotations

import torch
from torch import nn
from .local_token_attention import LocalTokenHARNet, LocalTokenStateAttention


class LocalRegimeStateAttention(LocalTokenStateAttention):
    def __init__(self, *args, regime_mode=None, **kwargs):
        super().__init__(*args, **kwargs)
        if regime_mode not in (None, 'current', 'filtered'):
            raise ValueError(f'unknown regime mode:{regime_mode}')
        self.regime_mode = regime_mode
        if regime_mode:
            self.regime_encoder = nn.Sequential(nn.Linear(5, 32, dtype=torch.float64), nn.GELU())
            self.regime_type = nn.Parameter(torch.randn(1, 32, dtype=torch.float64)*.02)

    def regime_token(self, research):
        if research is None or research.ndim != 2 or research.shape[1] != 10:
            raise ValueError('regime requires aligned current5+filtered5 state')
        start = 0 if self.regime_mode == 'current' else 5
        state = research[:, start:start+5]
        available = state[:, 4].eq(1)
        if not ((state[:, 4] == 0) | available).all():
            raise ValueError('regime availability must be binary')
        safe = torch.where(available[:, None], state, torch.zeros_like(state))
        if (not torch.isfinite(safe).all() or (safe[:, :4] < 0).any()
                or (safe[:, :4] > 1+1e-10).any()
                or not torch.allclose(safe[available, :3].sum(1), torch.ones_like(safe[available, 0]), atol=1e-8, rtol=0)):
            raise ValueError('invalid available regime probabilities/entropy')
        token = self.regime_encoder(safe)+self.regime_type
        return torch.where(available[:, None], token, torch.zeros_like(token)), available

    def conditioned_tokens(self, h, old_state, state, ids, own_q, calendar, history=None, research=None):
        tokens, padding, identity = super().conditioned_tokens(h, old_state, state, ids, own_q, calendar, history)
        if self.regime_mode:
            if research is None or len(research) != len(h):
                raise ValueError('missing/misaligned observed regime input')
            token, available = self.regime_token(research)
            tokens = torch.cat((tokens, token[:, None]), 1)
            padding = torch.cat((padding, ~available[:, None]), 1)
        return tokens, padding, identity


class LocalRegimeHARNet(LocalTokenHARNet):
    def __init__(self, *args, regime_mode=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Rebuild under exactly the existing seed; no pretraining or frozen head
        # transplant. Appending an encoder leaves original RNG draws unchanged.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(args[1]))
            self.attention = LocalRegimeStateAttention(self.attention.config, args[2], args[3], args[4],
                numeric_mode=kwargs['numeric_mode'], feature_column=kwargs.get('feature_column'),
                numeric=kwargs.get('numeric', False), split_calendar=kwargs.get('split_calendar', False),
                local_amplitude=kwargs.get('local_amplitude', False), regime_mode=regime_mode)
