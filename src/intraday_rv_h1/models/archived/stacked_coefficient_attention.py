"""Two explicit attention topologies and an exactly capacity-matched MLP.

All operations are within one origin. Existing own-value fusion remains local
before global attention. No token, history, output cap, or expert gate is added.
"""
from __future__ import annotations

import torch
from torch import nn
from .local_regime_attention import LocalRegimeHARNet, LocalRegimeStateAttention

TOPOLOGIES = ('cross2', 'state_cross', 'ffn_control')


class StackedStateAttention(LocalRegimeStateAttention):
    def __init__(self, *args, topology, **kwargs):
        super().__init__(*args, **kwargs)
        if topology not in TOPOLOGIES:
            raise ValueError(f'unknown stacked attention topology:{topology}')
        self.topology = topology
        self.depth_norm = nn.LayerNorm(32, dtype=torch.float64)
        if topology == 'ffn_control':
            self.depth_block = nn.Sequential(nn.Linear(32, 64, bias=False, dtype=torch.float64),
                nn.GELU(), nn.Linear(64, 32, bias=False, dtype=torch.float64))
        else:
            self.depth_block = nn.MultiheadAttention(32, 4, bias=False, dropout=0.,
                batch_first=True, dtype=torch.float64)
        # LN64 + MHA4096 or bias-free FFN4096, exactly4160 added parameters.
        self.added_parameters = sum(p.numel() for p in self.depth_norm.parameters()) + sum(p.numel() for p in self.depth_block.parameters())
        if self.added_parameters != 4160:
            raise AssertionError(f'depth capacity mismatch:{self.added_parameters}')

    def mix_state(self, tokens, padding):
        if tokens.ndim != 3 or tokens.shape[-1] != 32 or padding.shape != tokens.shape[:2]:
            raise ValueError('misaligned stacked state tokens/mask')
        if padding.dtype != torch.bool or padding.all(1).any():
            raise ValueError('stacked state requires at least one visible token')
        safe = torch.where(padding[..., None], torch.zeros_like(tokens), tokens)
        if not torch.isfinite(safe).all():
            raise FloatingPointError('nonfinite visible stacked state token')
        normalized = self.depth_norm(safe)
        # Preserve the existing amplitude-V policy only on its numeric slots.
        _, values = self.attention_kv(safe)
        mixed, _ = self.depth_block(normalized, normalized, values,
            key_padding_mask=padding, need_weights=False)
        return torch.where(padding[..., None], torch.zeros_like(safe), safe+mixed)

    def conditioned_tokens(self, *args, **kwargs):
        tokens, padding, identity = super().conditioned_tokens(*args, **kwargs)
        if self.topology == 'state_cross':
            tokens = self.mix_state(tokens, padding)
        return tokens, padding, identity

    def coefficient_attention(self, queries, keys, values, padding):
        first = super().coefficient_attention(queries, keys, values, padding)
        if self.topology == 'state_cross':
            return first
        updated_query = self.depth_norm(queries+first)
        if self.topology == 'cross2':
            second, _ = self.depth_block(updated_query, keys, values,
                key_padding_mask=padding, need_weights=False)
        else:
            second = self.depth_block(updated_query)
        return first+second


class StackedHARNet(LocalRegimeHARNet):
    def __init__(self, *args, topology, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(args[1]))
            self.attention = StackedStateAttention(self.attention.config, args[2], args[3], args[4],
                numeric_mode=kwargs['numeric_mode'], feature_column=kwargs.get('feature_column'),
                numeric=kwargs.get('numeric', False), split_calendar=kwargs.get('split_calendar', False),
                local_amplitude=kwargs.get('local_amplitude', False), regime_mode=kwargs.get('regime_mode'),
                topology=topology)
