"""Original prefix attention with a differentiable, per-asset H80 SVJ path."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .same_clock import SameClockHARNet
from .original_prefix_ablation import build_original_prefix, feature_mask
from .svj_coefficient_conditioning import CAP, DECAY_ROWS


class TrainablePrefixReference(nn.Module):
    def __init__(self, assets, variant):
        super().__init__()
        self.variant = variant
        self.locals = nn.ModuleList([
            SameClockHARNet(lags=(1, 5, 20, 80), svj=True, filters=2)
            for _ in range(assets)
        ])
        for local in self.locals:
            with torch.no_grad():
                local.output.weight.zero_()
            local.output.weight.requires_grad_(False)
        self.heads = nn.Parameter(torch.zeros(assets, 4, 16, dtype=torch.float64))
        self.six = nn.Parameter(torch.zeros(assets, 6, dtype=torch.float64))
        self.register_buffer("active", torch.from_numpy(feature_mask(variant)))

    def features(self, core, auxiliary, ids):
        if core.shape != (len(ids), 80, 3) or auxiliary.shape != (len(ids), 6):
            raise ValueError("expected aligned H80 core and six auxiliary values")
        if ids.dtype != torch.int64 or (ids < 0).any() or (ids >= len(self.locals)).any():
            raise ValueError("invalid asset IDs")
        h = auxiliary.new_zeros((len(ids), 16))
        # Only six bounded partitions; never a Python loop over panel rows.
        for asset, local in enumerate(self.locals):
            rows = torch.nonzero(ids == asset).flatten()
            if len(rows):
                f = torch.cat((local.feature_values(core[rows]).double(), auxiliary[rows]), 1)
                h = h.index_copy(0, rows, f)
        return torch.where(self.active, h, 0.)

    def coefficients(self, h, clock, rank, position, ids):
        head = self.heads[ids] * self.active[None, None]
        base = head[:, 0] + torch.einsum("bi,bij->bj", clock, head[:, 1:])
        factors = (torch.where(self.active[10:], rank, 0.) * self.six[ids]
                   * torch.exp(-(position - 1.) / DECAY_ROWS)[:, None])
        z = (h[:, 10:] * factors).sum(-1) / CAP
        # Avoid evaluating 0/0 even in the inactive branch of torch.where.
        denominator = torch.where(z == 0., torch.ones_like(z), z)
        ratio = torch.where(z == 0., torch.ones_like(z), torch.tanh(z) / denominator)
        correction = torch.cat((torch.zeros_like(h[:, :10]), factors * ratio[:, None]), 1)
        return torch.where(self.active, base + correction, 0.)

    def components(self, core, auxiliary, old_state, position, ids):
        h = self.features(core, auxiliary, ids)
        beta = self.coefficients(h, old_state[:, :3], old_state[:, 3::3], position, ids)
        return h, beta


class TrainablePrefixAttention(nn.Module):
    def __init__(self, config, seed, limits, descriptors, variant, reference_state,
                 rank_values, rank_counts):
        super().__init__()
        # Original attention RNG sequence is independent of the convolution constructors.
        self.attention = build_original_prefix(config, seed, limits, descriptors, variant)
        self.reference = TrainablePrefixReference(len(limits), variant)
        self.reference.load_state_dict(reference_state)
        self.reference.heads.requires_grad_(False)
        self.reference.six.requires_grad_(False)
        # Frozen fit-scope CDFs live in a separately hash-verified registry.
        # Their values must not be duplicated in all 51 weight checkpoints.
        self.register_buffer("rank_values", torch.as_tensor(rank_values, dtype=torch.float64), persistent=False)
        self.register_buffer("rank_counts", torch.as_tensor(rank_counts, dtype=torch.int64), persistent=False)

    def current_state(self, h, state, groups):
        if state.shape[1:] != (8, 3):
            raise ValueError("this experiment requires six auxiliary plus RV1/RV5 tokens")
        values = h[:, [1, 3]].detach().contiguous()
        ordered = self.rank_values[groups].contiguous()
        left = torch.searchsorted(ordered, values[..., None], right=False).squeeze(-1)
        right = torch.searchsorted(ordered, values[..., None], right=True).squeeze(-1)
        q = (left + .5 * (right - left) + .5) / (self.rank_counts[groups] + 1)
        r = 2 * q - 1
        result = state.clone()
        result[:, 6:, 0] = r
        result[:, 6:, 1] = torch.relu(r - .6) / .4
        result[:, 6:, 2] = 1.
        return result

    def forward(self, core, auxiliary, old_state, state, context, ids, position, groups):
        h, reference = self.reference.components(core, auxiliary, old_state, position, ids)
        current = self.current_state(h, state, groups)
        prediction, beta = self.attention(h, reference, old_state, current, context, ids)
        return prediction, beta, h, reference

    def loss(self, inputs, target, scale):
        prediction, beta, h, reference = self(*inputs)
        value = ((prediction - target).square().mean() / scale
                 + self.attention.regularization()
                 + self.attention.coefficient_regularization(h, beta - reference))
        if not torch.isfinite(value):
            raise FloatingPointError("nonfinite trainable-convolution attention loss")
        return value


def convolution_digest(model):
    import hashlib
    digest = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if "average_layers" in name:
            digest.update(name.encode()); digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
