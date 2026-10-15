"""Independent asset identity in attention values and the coefficient decoder."""
from __future__ import annotations

import torch
from torch import nn

from .directional_coefficient_attention import DirectionalAttentionConfig
from .joint_coefficient_attention import ReferenceJointDirectionalCoefficientAttention


IDENTITY_MODES = ("shared", "id", "descriptors", "id_descriptors")
IDENTITY_ROUTES = ("both", "token_only", "decoder_only")


class IdentityTokenCoefficientAttention(ReferenceJointDirectionalCoefficientAttention):
    """Same architecture in all controls; only identity inputs are ablated.

    Identity is a separate always-visible K/V token and a direct decoder input.
    The coefficient query is also a decoder input to retain coefficient identity.
    Only the last expert projections are zeroed: epoch zero equals the reference,
    while the encoder can receive gradients after the first output update.
    """

    def __init__(self, config, asset_limits, descriptors, *, identity_mode, identity_route="both"):
        super().__init__(config, asset_limits)
        if identity_mode not in IDENTITY_MODES or config.experts != "uniform3":
            raise ValueError("identity token requires a declared mode and uniform3 experts")
        profile = torch.as_tensor(descriptors, dtype=torch.float64)
        if profile.shape != (len(self.asset_limits), 4) or not torch.isfinite(profile).all():
            raise ValueError("descriptors must be finite [asset,4]")
        self.identity_mode = identity_mode
        if identity_route not in IDENTITY_ROUTES:
            raise ValueError(f"unknown identity route: {identity_route}")
        self.identity_route = identity_route
        self.register_buffer("identity_descriptors", profile.clone())
        self.identity_embedding = nn.Embedding(len(profile), 4, dtype=torch.float64)
        nn.init.normal_(self.identity_embedding.weight, std=.02)
        self.identity_encoder = nn.Sequential(
            nn.Linear(8, 32, dtype=torch.float64), nn.GELU(),
            nn.Linear(32, 32, dtype=torch.float64),
        )
        self.identity_token_type = nn.Parameter(torch.empty(1, 1, 32, dtype=torch.float64))
        nn.init.normal_(self.identity_token_type, std=.02)
        self.expert_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(96, 8, dtype=torch.float64), nn.GELU(),
                          nn.Linear(8, 1, dtype=torch.float64)) for _ in range(3)
        ])
        for expert in self.expert_layers:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def encode_identity(self, asset_ids, *, id_override=None, descriptor_override=None):
        """Overrides affect identity only, never reference/features/asset caps."""
        ids = asset_ids if id_override is None else id_override
        if ids.shape != asset_ids.shape or ids.dtype != torch.int64:
            raise ValueError("identity override must be aligned int64 IDs")
        if (ids < 0).any() or (ids >= len(self.asset_limits)).any():
            raise ValueError("unknown identity override")
        if self.identity_mode in ("shared", "descriptors"):
            ids = torch.zeros_like(ids)
        profile = (self.identity_descriptors.index_select(0, asset_ids)
                   if descriptor_override is None else descriptor_override)
        if profile.shape != (len(ids), 4) or not torch.isfinite(profile).all():
            raise ValueError("invalid identity descriptor override")
        if self.identity_mode in ("shared", "id"):
            profile = torch.zeros_like(profile)
        return self.identity_encoder(torch.cat((self.identity_embedding(ids), profile), dim=-1))

    def raw_delta_for_assets(self, old_state, state, context, asset_ids, scale_context=None):
        if scale_context is not None:
            raise ValueError("identity token has no legacy scale-query context")
        identity = self.encode_identity(asset_ids)
        return self.raw_delta_for_identity(old_state, state, context, identity)

    def raw_delta_for_identity(self, old_state, state, context, identity):
        """Controlled identity response with the dynamic states held fixed."""
        decoder = self.coefficient_context(old_state, state, context, identity)
        experts = torch.stack([layer(decoder).squeeze(-1) for layer in self.expert_layers], dim=-1)
        return experts.mean(-1)

    def coefficient_context(self, old_state, state, context, identity):
        """Expose the existing 96-column decoder input without another attention pass."""
        if identity.shape != (len(state), 32) or not torch.isfinite(identity).all():
            raise ValueError("expected finite aligned encoded identity [batch,32]")
        token_identity = decoder_identity = identity
        if self.identity_route != "both":
            # An inactive route receives the same learned common identity used
            # by the shared control. Keep token count, decoder size and all
            # parameters identical; change only asset-specific information.
            common_input = torch.cat((self.identity_embedding.weight[:1], identity.new_zeros((1, 4))), dim=-1)
            common = self.identity_encoder(common_input).expand(len(state), -1)
            if self.identity_route == "token_only":
                decoder_identity = common
            else:
                token_identity = common
        tokens, padding = self.state_tokens(old_state, state)
        if tokens is None:
            raise ValueError("identity experiment requires the original state tokens")
        tokens = torch.cat((tokens, token_identity[:, None] + self.identity_token_type), dim=1)
        padding = torch.cat((padding, padding.new_zeros((len(state), 1))), dim=1)
        queries = self.head.norm_query(self.head.queries)[None].expand(len(state), -1, -1)
        kv = self.head.norm_tokens(tokens)
        attended, _ = self.head.attention(queries, kv, kv, key_padding_mask=padding, need_weights=False)
        return torch.cat((attended, decoder_identity[:, None].expand(-1, 16, -1), queries), dim=-1)


def build_identity_token_attention(config, seed, asset_limits, descriptors, *, identity_mode, identity_route="both"):
    if isinstance(config, dict):
        config = DirectionalAttentionConfig(**config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return IdentityTokenCoefficientAttention(config, asset_limits, descriptors, identity_mode=identity_mode, identity_route=identity_route)
