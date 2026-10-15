"""Shared coefficient attention with explicit multi-asset initialization.

This module is deliberately independent from the historical reference-anchored
heads.  It reuses the directional state encoder, but the coefficient base is
either one pooled Train-only OLS vector or one Train-only OLS vector per asset.
An optional zero-initialized asset branch tests whether a static asset identity
helps a shared state-to-coefficient mapping.
"""
from __future__ import annotations

import copy
import torch
from torch import nn

from .directional_coefficient_attention import DirectionalAttentionConfig, DirectionalCoefficientAttention


class JointDirectionalCoefficientAttention(DirectionalCoefficientAttention):
    """No-reference directional attention for clock-major multi-asset rows."""

    def __init__(
        self,
        config: DirectionalAttentionConfig,
        *,
        asset_count: int,
        base_mode: str,
        asset_conditioning: bool = False,
        asset_embedding_dim: int = 4,
    ) -> None:
        if config.reference_mode != "none":
            raise ValueError("joint model requires reference_mode=none")
        if base_mode not in ("pooled_ols", "asset_ols"):
            raise ValueError("base_mode must be pooled_ols or asset_ols")
        if asset_count < 1 or asset_embedding_dim < 1:
            raise ValueError("asset_count and asset_embedding_dim must be positive")
        super().__init__(config)
        self.asset_count = int(asset_count)
        self.base_mode = base_mode
        self.asset_conditioning = bool(asset_conditioning)
        self.asset_embedding_dim = int(asset_embedding_dim)
        if base_mode == "asset_ols":
            self.base_coefficients = nn.Parameter(torch.zeros(asset_count, 16, dtype=torch.float64))
        if self.asset_conditioning:
            self.asset_embedding = nn.Embedding(asset_count, asset_embedding_dim).double()
            self.asset_output = nn.Linear(asset_embedding_dim, 16).double()
            # Epoch 0 must be exactly the selected OLS initialization.  The
            # embedding is random, but its coefficient projection is zero.
            nn.init.zeros_(self.asset_output.weight)
            nn.init.zeros_(self.asset_output.bias)

    def set_base_coefficients(self, coefficients) -> None:
        beta = torch.as_tensor(
            coefficients,
            dtype=self.base_coefficients.dtype,
            device=self.base_coefficients.device,
        )
        expected = (16,) if self.base_mode == "pooled_ols" else (self.asset_count, 16)
        if beta.shape != expected or not torch.isfinite(beta).all():
            raise ValueError(f"base coefficients must be finite with shape {expected}")
        with torch.no_grad():
            self.base_coefficients.copy_(beta)

    def _base_for_assets(self, asset_ids: torch.Tensor, rows: int) -> torch.Tensor:
        if asset_ids.ndim != 1 or asset_ids.shape[0] != rows or asset_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("asset_ids must be a one-dimensional integer tensor")
        if torch.any(asset_ids < 0) or torch.any(asset_ids >= self.asset_count):
            raise ValueError("asset_ids contain an unknown asset")
        if self.base_mode == "pooled_ols":
            return self.base_coefficients[None, :].expand(rows, -1)
        return self.base_coefficients.index_select(0, asset_ids.to(torch.int64))

    def forward(self, features, old_state, state, context, asset_ids):
        if features.ndim != 2 or features.shape[1] != 16:
            raise ValueError("expected [batch,16] features")
        base = self._base_for_assets(asset_ids, len(features))
        if not torch.isfinite(features).all() or not torch.isfinite(base).all():
            raise FloatingPointError("non-finite features or coefficient base")
        raw = self.raw_delta(old_state, state, context)
        if self.asset_conditioning:
            raw = raw + self.asset_output(self.asset_embedding(asset_ids.to(torch.int64)))
        coefficients = base + raw
        if not torch.isfinite(coefficients).all():
            raise FloatingPointError("non-finite joint coefficient output")
        prediction = torch.exp((features * coefficients).sum(-1) / 2.0)
        if not torch.isfinite(prediction).all() or (prediction <= 0).any():
            raise FloatingPointError("invalid joint coefficient forecast")
        return prediction, coefficients


def build_joint_directional_attention(
    config,
    seed: int,
    *,
    asset_count: int,
    base_mode: str,
    asset_conditioning: bool = False,
    asset_embedding_dim: int = 4,
) -> JointDirectionalCoefficientAttention:
    """Construct a deterministic joint model without changing global RNG state."""
    if isinstance(config, dict):
        config = DirectionalAttentionConfig(**config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return JointDirectionalCoefficientAttention(
            config,
            asset_count=asset_count,
            base_mode=base_mode,
            asset_conditioning=asset_conditioning,
            asset_embedding_dim=asset_embedding_dim,
        )


class ReferenceJointDirectionalCoefficientAttention(DirectionalCoefficientAttention):
    """Share the original state head while preserving each asset's reference/cap.

    In the control, asset IDs select immutable coefficient limits only. The
    optional learned asset query lets identity condition the state response.
    """

    def __init__(self, config: DirectionalAttentionConfig, asset_limits, *, asset_query: bool = False, embedding_dim: int = 4, asset_heads: bool = False, scale_encoding: str = "none", scale_axis=None, loss_weights=None) -> None:
        if config.reference_mode != "anchored" or config.coefficient_mode != "bounded":
            raise ValueError("reference joint model requires the original anchored bounded head")
        super().__init__(config)
        limits = torch.as_tensor(asset_limits, dtype=torch.float64).clone()
        if limits.ndim != 2 or limits.shape[1] != 16 or not torch.isfinite(limits).all() or (limits <= 0).any():
            raise ValueError("asset limits must be finite positive [asset,16] values")
        self.register_buffer("asset_limits", limits)
        self.asset_query = bool(asset_query)
        self.scale_encoding = scale_encoding
        allowed = ("none", "linear", "mlp", "learned_axis", "clock_linear", "level_current")
        if scale_encoding not in allowed or (asset_query and scale_encoding != "none"):
            raise ValueError("invalid or conflicting scale/asset query encoding")
        if self.asset_query:
            if config.experts != "uniform3" or embedding_dim < 1:
                raise ValueError("asset query conditioning requires uniform3 and a positive embedding dimension")
            # Introduce identity before attention; the zero projection preserves
            # the original query and epoch0 prediction exactly. Existing RNG
            # calls/weights occur before these new parameters are allocated.
            self.asset_embedding = nn.Embedding(len(limits), embedding_dim, dtype=torch.float64)
            self.asset_query_projection = nn.Linear(embedding_dim, 32, bias=False, dtype=torch.float64)
            nn.init.normal_(self.asset_embedding.weight, std=0.02)
            nn.init.zeros_(self.asset_query_projection.weight)
        if scale_encoding != "none":
            if config.experts != "uniform3":
                raise ValueError("scale conditioning requires uniform3")
            width = 2 if scale_encoding == "level_current" else 1
            if scale_encoding in ("mlp", "level_current"):
                self.scale_query_projection = nn.Sequential(nn.Linear(width, 8, dtype=torch.float64), nn.GELU(), nn.Linear(8, 32, bias=False, dtype=torch.float64))
                nn.init.zeros_(self.scale_query_projection[-1].weight)
            else:
                self.scale_query_projection = nn.Linear(width, 32, bias=False, dtype=torch.float64)
                nn.init.zeros_(self.scale_query_projection.weight)
            if scale_encoding == "learned_axis":
                axis = torch.as_tensor(scale_axis, dtype=torch.float64)
                if axis.shape != (len(limits),) or not torch.isfinite(axis).all():
                    raise ValueError("learned scale axis requires finite per-asset initialization")
                self.scale_axis = nn.Parameter(axis[:, None].clone())
        if loss_weights is not None:
            weights = torch.as_tensor(loss_weights, dtype=torch.float64)
            if weights.shape != (len(limits),) or not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("loss weights must be finite positive per-asset values")
            self.register_buffer("scale_loss_weights", weights.clone())
        self.asset_heads = bool(asset_heads)
        if self.asset_heads:
            if config.experts != "uniform3":
                raise ValueError("per-asset output heads require uniform3")
            # Clone the same freshly initialized experts, without extra RNG
            # calls. The shared encoder/query parameters retain their identity.
            self.asset_experts = nn.ModuleList([copy.deepcopy(self.expert_layers) for _ in range(len(limits))])
            self.expert_layers.requires_grad_(False)

    def scale_coordinates(self, asset_ids, scale_context):
        if self.scale_encoding == "learned_axis":
            return self.scale_axis.index_select(0, asset_ids)
        if scale_context is None or scale_context.shape != (len(asset_ids), 3) or not torch.isfinite(scale_context).all():
            raise ValueError("scale encoding requires finite aligned [batch,3] context")
        if self.scale_encoding == "clock_linear":
            return scale_context[:, 1:2]
        if self.scale_encoding == "level_current":
            return scale_context[:, [0, 2]]
        return scale_context[:, :1]

    def raw_delta_for_assets(self, old_state, state, context, asset_ids, scale_context=None):
        if not self.asset_query and not self.asset_heads and self.scale_encoding == "none":
            return self.raw_delta(old_state, state, context)
        tokens, padding = self.state_tokens(old_state, state)
        if tokens is None:
            raise ValueError("asset query requires visible state tokens")
        if self.asset_query:
            identity = self.asset_query_projection(self.asset_embedding(asset_ids))
            queries = self.head.norm_query(self.head.queries[None] + identity[:, None, :])
        elif self.scale_encoding != "none":
            identity = self.scale_query_projection(self.scale_coordinates(asset_ids, scale_context))
            queries = self.head.norm_query(self.head.queries[None] + identity[:, None, :])
        else:
            queries = self.head.norm_query(self.head.queries)[None].expand(len(state), -1, -1)
        kv = self.head.norm_tokens(tokens)
        attended, _ = self.head.attention(queries, kv, kv, key_padding_mask=padding, need_weights=False)
        # Uniform3 has no input-dependent gates: preserve its original average.
        safe = torch.where(state[:, :, 2:3] > 0, state, torch.zeros_like(state))
        gates = self.gates(safe, context)
        if not self.asset_heads:
            experts = torch.stack([layer(attended).squeeze(-1) for layer in self.expert_layers], dim=-1)
            return (experts * gates[:, None, :]).sum(-1)
        raw = attended.new_zeros((len(attended), 16))
        for asset, layers in enumerate(self.asset_experts):
            rows = torch.nonzero(asset_ids == asset, as_tuple=False).flatten()
            if len(rows):
                experts = torch.stack([layer(attended[rows]).squeeze(-1) for layer in layers], dim=-1)
                raw = raw.index_copy(0, rows, (experts * gates[rows, None, :]).sum(-1))
        return raw

    def forward(self, features, reference, old_state, state, context, asset_ids, scale_context=None):
        if features.ndim != 2 or features.shape[1] != 16 or reference.shape != features.shape:
            raise ValueError("expected aligned [batch,16] features and reference")
        if asset_ids.shape != (len(features),) or asset_ids.dtype != torch.int64:
            raise ValueError("asset_ids must be a one-dimensional int64 tensor")
        if (asset_ids < 0).any() or (asset_ids >= len(self.asset_limits)).any():
            raise ValueError("unknown asset id")
        if not torch.isfinite(features).all() or not torch.isfinite(reference).all():
            raise FloatingPointError("non-finite reference joint inputs")
        raw = self.raw_delta_for_assets(old_state, state, context, asset_ids, scale_context)
        delta = self.asset_limits.index_select(0, asset_ids) * torch.tanh(raw)
        coefficients = reference + delta
        prediction = torch.exp((features * coefficients).sum(-1) / 2.0)
        if not torch.isfinite(coefficients).all() or not torch.isfinite(prediction).all() or (prediction <= 0).any():
            raise FloatingPointError("invalid reference joint forecast")
        return prediction, coefficients


def build_reference_joint_directional_attention(config, seed: int, asset_limits, *, asset_query: bool = False, embedding_dim: int = 4, asset_heads: bool = False, scale_encoding: str = "none", scale_axis=None, loss_weights=None):
    if isinstance(config, dict):
        config = DirectionalAttentionConfig(**config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return ReferenceJointDirectionalCoefficientAttention(config, asset_limits, asset_query=asset_query, embedding_dim=embedding_dim, asset_heads=asset_heads, scale_encoding=scale_encoding, scale_axis=scale_axis, loss_weights=loss_weights)
