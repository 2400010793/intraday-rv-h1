"""Extensible model registry for same-clock intraday RV training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

ModelBuilder = Callable[[int, int, int, Mapping[str, Any]], nn.Module]


class ContinuousPiecewiseLinear(nn.Module):
    """Continuous identity-centered hinge activation.

    ``slopes`` are the slopes on successive intervals and ``knots`` are the
    strictly increasing breakpoints.  The default identity configuration is
    exactly the old linear activation, while fixed alternatives remain
    differentiable almost everywhere and do not introduce trainable state.
    """

    def __init__(self, knots: tuple[float, ...] = (), slopes: tuple[float, ...] = (1.0,)) -> None:
        super().__init__()
        if len(slopes) != len(knots) + 1:
            raise ValueError("piecewise slopes must have exactly one more value than knots")
        if tuple(sorted(knots)) != tuple(knots) or len(set(knots)) != len(knots):
            raise ValueError("piecewise knots must be strictly increasing")
        if not torch.isfinite(torch.tensor(slopes)).all():
            raise ValueError("piecewise slopes must be finite")
        self.register_buffer("knots", torch.tensor(knots, dtype=torch.float32))
        self.register_buffer("slopes", torch.tensor(slopes, dtype=torch.float32))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        result = self.slopes[0] * values
        for index, knot in enumerate(self.knots):
            result = result + (self.slopes[index + 1] - self.slopes[index]) * torch.relu(values - knot)
        return result


def _activation(name: str) -> nn.Module:
    key = str(name).lower().strip()
    if key in {"identity", "linear"}:
        return nn.Identity()
    group = None
    if key.startswith("group0_"):
        group, key = 0, key[len("group0_"):]
    elif key.startswith("group1_"):
        group, key = 1, key[len("group1_"):]
    slopes = {"102": 1.02, "105": 1.05, "110": 1.10}
    if key.startswith("hinge_tau0_slope"):
        slope = slopes.get(key.removeprefix("hinge_tau0_slope"))
        if slope is not None:
            base = ContinuousPiecewiseLinear((0.0,), (1.0, slope))
            return ChannelwiseActivation(base, group) if group is not None else base
    if key.startswith("hinge_tau05_slope"):
        slope = slopes.get(key.removeprefix("hinge_tau05_slope"))
        if slope is not None:
            base = ContinuousPiecewiseLinear((0.5,), (1.0, slope))
            return ChannelwiseActivation(base, group) if group is not None else base
    if key == "hinge_tau1_slope110":
        return ContinuousPiecewiseLinear((1.0,), (1.0, 1.1))
    if key == "piecewise3_fixed":
        return ContinuousPiecewiseLinear((-1.0, 0.0, 1.0), (1.0, 1.0, 1.05, 1.1))
    raise ValueError(f"unknown HARNet activation: {name}")


class ChannelwiseActivation(nn.Module):
    """Apply a candidate activation to one hidden SVJ channel only."""

    def __init__(self, activation: nn.Module, channel: int) -> None:
        super().__init__()
        if channel not in (0, 1):
            raise ValueError("channelwise HARNet activation requires channel 0 or 1")
        self.activation = activation
        self.channel = int(channel)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        result = values.clone()
        result[:, self.channel:self.channel + 1] = self.activation(
            values[:, self.channel:self.channel + 1]
        )
        return result

# Public intraday registry. The internal registry also retains the canonical
# pure-HARNet 1/5/20/40/80 name because historical weight artifacts record it.
INTRADAY_HARNET_MODELS = (
    "har",
    "log_har",
    "harnet_1_5_20",
    "harnet_1_5_20_80",
    "harnetsvj_author_1_5_20",
    "harnetsvj_author_1_5_20_80",
    "harnetsvj_author_1_5_20_40_80",
    "harnetsvj_gsphar_h80",
    "multitask_commonality_graph_harnet_h80",
)


@dataclass(frozen=True)
class SameClockModelSpec:
    name: str
    joint: bool
    channels: tuple[str, ...]
    builder: ModelBuilder
    graph_kind: str = "none"
    target_transform: str = "level"


class SameClockModelRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, SameClockModelSpec] = {}

    def register(self, spec: SameClockModelSpec, *, aliases: tuple[str, ...] = ()) -> None:
        names = (spec.name, *aliases)
        collision = sorted(set(names).intersection(self._specs))
        if collision:
            raise ValueError(f"same-clock model names already registered: {collision}")
        for name in names:
            self._specs[str(name)] = spec

    def get(self, name: str) -> SameClockModelSpec:
        try:
            return self._specs[str(name)]
        except KeyError as exc:
            raise ValueError(f"unknown same-clock model: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, spec in self._specs.items() if name == spec.name))


class SameClockHARNet(nn.Module):
    """PyTorch transfer of the released HARNet/HARNetSVJ causal stack."""

    def __init__(
        self,
        *,
        lags: tuple[int, ...],
        svj: bool,
        filters: int,
        use_bias: bool = False,
        activation: str = "identity",
    ) -> None:
        super().__init__()
        if tuple(sorted(set(lags))) != tuple(lags):
            raise ValueError("HARNet lags must be strictly increasing")
        if any(current % previous for previous, current in zip(lags[:-1], lags[1:])):
            raise ValueError("each HARNet lag must be a multiple of its predecessor")
        self.lags = tuple(map(int, lags))
        self.svj = bool(svj)
        self.filters = 2 if self.svj else int(filters)
        if self.svj and self.filters != 2:
            raise ValueError("HARNetSVJ requires two causal aggregation filters")
        activation_key = str(activation).lower()
        activation_cls = {"relu": nn.ReLU, "tanh": nn.Tanh}.get(activation_key)
        layers: list[nn.Conv1d] = []
        previous_channels = 2 if self.svj else 1
        for index in range(len(self.lags) - 1):
            kernel = self.lags[index + 1] // self.lags[index]
            dilation = self.lags[index]
            layer = nn.Conv1d(
                previous_channels,
                self.filters,
                kernel_size=kernel,
                dilation=dilation,
                padding=0,
                bias=bool(use_bias and (self.svj or index == 0)),
            )
            with torch.no_grad():
                layer.weight.zero_()
                if self.svj:
                    for channel in range(2):
                        layer.weight[channel, channel, :].fill_(1.0 / kernel)
                else:
                    layer.weight.fill_(1.0 / kernel)
                if layer.bias is not None:
                    layer.bias.zero_()
            layers.append(layer)
            previous_channels = self.filters
        self.average_layers = nn.ModuleList(layers)
        self.activation = _activation(activation_key) if activation_cls is None else activation_cls()
        # intercept + raw averaged channels at every scale + signed channel for SVJ
        # Non-SVJ emits ``filters`` channels at each causal scale, while the
        # original one-filter case has one output per scale.  SVJ keeps its
        # two signed variance channels plus the explicit signed component.
        width = (
            1 + len(self.lags) * 2 + 1
            if self.svj
            else 2 + (len(self.lags) - 1) * self.filters
        )
        self.output = nn.Linear(width, 1, bias=False)
        nn.init.xavier_uniform_(self.output.weight)

    @staticmethod
    def _causal_last(layer: nn.Conv1d, values: torch.Tensor) -> torch.Tensor:
        receptive = (layer.kernel_size[0] - 1) * layer.dilation[0]
        padded = torch.nn.functional.pad(values, (receptive, 0))
        return layer(padded)

    def _validate_observed(self, observed: torch.Tensor) -> None:
        if observed.ndim != 3:
            raise ValueError("HARNet observed input must be [batch, history, channel]")
        expected_channels = 3 if self.svj else 1
        if observed.shape[-1] != expected_channels:
            raise ValueError(f"HARNet expected {expected_channels} input channels")
        if observed.shape[1] < max(self.lags):
            raise ValueError("HARNet history is shorter than its maximum lag")

    def feature_values(self, observed: torch.Tensor) -> torch.Tensor:
        """Return deterministic causal HAR/SVJ regressors before the output head."""
        self._validate_observed(observed)
        signed = observed[:, -1, 2:3] if self.svj else None
        current = observed[:, :, :2] if self.svj else observed[:, :, :1]
        features = [torch.ones_like(current[:, -1, :1]), current[:, -1]]
        conv_input = current.transpose(1, 2)
        for layer in self.average_layers:
            conv_output = self.activation(self._causal_last(layer, conv_input))
            features.append(conv_output[:, :, -1])
            conv_input = conv_output
        if signed is not None:
            features.append(signed)
        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def initialize_output(self, observed: torch.Tensor, target: torch.Tensor, method: str = "ols") -> None:
        """Fit the scalar output head by deterministic OLS in fit space.

        The convolutional HAR filters are already fixed causal averages.  The
        output layer is then initialized from the training rows only.  The
        caller supplies either the standardized level/variance target or a
        standardized log target, so ``log_ols`` is coherent with the loss
        space used by the subsequent optimizer.
        """
        name = str(method).lower().replace("-", "_")
        if name not in {"ols", "har_ols", "log_ols"}:
            raise ValueError("HARNet output initialization must be ols, har_ols, or log_ols")
        features = self.feature_values(observed).reshape(-1, self.output.in_features).double()
        values = torch.as_tensor(target, device=features.device, dtype=torch.float64).reshape(-1, 1)
        if features.shape[0] != values.shape[0]:
            raise ValueError("HARNet initialization rows and targets are misaligned")
        solution = torch.linalg.lstsq(features, values).solution.T.to(dtype=self.output.weight.dtype)
        if solution.shape != self.output.weight.shape or not torch.isfinite(solution).all():
            raise FloatingPointError("non-finite HARNet OLS output initialization")
        self.output.weight.copy_(solution)

    def forward(self, observed: torch.Tensor, adjacency=None) -> torch.Tensor:
        return self.output(self.feature_values(observed))


class HARNetScaleEncoder(nn.Module):
    """Expose HARNet-SVJ as an explicit ``[scale, channel]`` tensor.

    The released HARNet-SVJ jointly filters the RV and negative-semivariance
    channels and only appends the current signed-semivariance value.  This
    adapter preserves that learned two-channel block exactly and adds an
    independent causal hierarchy for the signed channel.  Consequently a
    transferred model is prediction-identical while exposing all three
    channels at every configured scale.
    """

    CHANNELS = ("rv", "negative_semivariance", "signed_semivariance")

    def __init__(self, lags: tuple[int, ...] = (1, 5, 20, 80)) -> None:
        super().__init__()
        if tuple(sorted(set(lags))) != tuple(lags) or not lags or lags[0] != 1:
            raise ValueError("scale encoder lags must be unique, sorted, and start at one")
        if any(current % previous for previous, current in zip(lags[:-1], lags[1:])):
            raise ValueError("each scale encoder lag must be a multiple of its predecessor")
        self.lags = tuple(map(int, lags))
        legacy_layers = []
        signed_layers = []
        for previous, current in zip(self.lags[:-1], self.lags[1:]):
            kernel = current // previous
            legacy = nn.Conv1d(2, 2, kernel, dilation=previous, bias=False)
            signed = nn.Conv1d(1, 1, kernel, dilation=previous, bias=False)
            with torch.no_grad():
                legacy.weight.zero_()
                legacy.weight[0, 0].fill_(1.0 / kernel)
                legacy.weight[1, 1].fill_(1.0 / kernel)
                signed.weight.fill_(1.0 / kernel)
            legacy_layers.append(legacy)
            signed_layers.append(signed)
        self.legacy_layers = nn.ModuleList(legacy_layers)
        self.signed_layers = nn.ModuleList(signed_layers)
        # Keep the released 10-value accumulation path byte-for-byte intact.
        # Three extra weights add signed-semivariance scales 5/20/80.
        self.legacy_output = nn.Linear(1 + len(self.lags) * 2 + 1, 1, bias=False)
        self.signed_scale_output = nn.Linear(len(self.lags) - 1, 1, bias=False)
        nn.init.xavier_uniform_(self.legacy_output.weight)
        nn.init.zeros_(self.signed_scale_output.weight)

    @staticmethod
    def _causal_last(layer: nn.Conv1d, values: torch.Tensor) -> torch.Tensor:
        receptive = (layer.kernel_size[0] - 1) * layer.dilation[0]
        return layer(torch.nn.functional.pad(values, (receptive, 0)))

    def scale_features(self, observed: torch.Tensor) -> torch.Tensor:
        if observed.ndim != 3 or observed.shape[-1] != 3:
            raise ValueError("scale encoder input must be [batch, history, 3]")
        if observed.shape[1] < self.lags[-1]:
            raise ValueError("scale encoder history is shorter than its largest scale")
        legacy = observed[..., :2].transpose(1, 2)
        signed = observed[..., 2:3].transpose(1, 2)
        scales = [observed[:, -1, :]]
        for legacy_layer, signed_layer in zip(self.legacy_layers, self.signed_layers):
            legacy = self._causal_last(legacy_layer, legacy)
            signed = self._causal_last(signed_layer, signed)
            scales.append(torch.cat((legacy[:, :, -1], signed[:, :, -1]), dim=-1))
        return torch.stack(scales, dim=1)

    def local_features(self, observed: torch.Tensor) -> torch.Tensor:
        scale = self.scale_features(observed).flatten(start_dim=1)
        return torch.cat((torch.ones_like(scale[:, :1]), scale), dim=-1)

    def forward(self, observed: torch.Tensor) -> torch.Tensor:
        scale = self.scale_features(observed)
        legacy = torch.cat((
            torch.ones_like(scale[:, 0, :1]),
            scale[..., :2].flatten(start_dim=1),
            scale[:, 0, 2:3],
        ), dim=-1)
        return self.legacy_output(legacy) + self.signed_scale_output(scale[:, 1:, 2])

    @torch.no_grad()
    def initialize_from_harnet(self, source: SameClockHARNet) -> None:
        """Copy a released H80-SVJ model and zero only the new signed scales."""

        if not isinstance(source, SameClockHARNet) or not source.svj:
            raise TypeError("scale transfer requires a HARNet-SVJ source")
        if tuple(source.lags) != self.lags:
            raise ValueError("source HARNet lags do not match scale encoder lags")
        for target, original in zip(self.legacy_layers, source.average_layers):
            if target.weight.shape != original.weight.shape:
                raise ValueError("source HARNet convolution shape does not match")
            target.weight.copy_(original.weight)
        self.legacy_output.weight.copy_(source.output.weight)
        self.signed_scale_output.weight.zero_()


class MultiTaskScaleGraphHARNet(nn.Module):
    """Two-horizon Graph-HARNet over explicit 4 x 3 node features.

    Output order is always ``[..., (H1, H3)]``.  Graph propagation is shared
    across horizons while local encoders and residual heads remain separate.
    Calling :meth:`components` exposes the local/commonality/graph audit paths.
    """

    HORIZONS = ("H1", "H3")
    GRAPH_OPERATORS = ("gnnhar", "gcn", "mean", "gsphar", "gat")

    def __init__(
        self, assets: int, *, lags: tuple[int, ...] = (1, 5, 20, 80),
        hidden_size: int = 8, graph_operator: str = "gnnhar", q: float = 0.25,
        use_commonality: bool = True,
    ) -> None:
        super().__init__()
        if int(assets) < 2:
            raise ValueError("multitask scale Graph-HARNet requires at least two assets")
        operator = str(graph_operator).strip().lower()
        if operator not in self.GRAPH_OPERATORS:
            raise ValueError(f"graph_operator must be one of {self.GRAPH_OPERATORS}")
        self.assets = int(assets)
        self.lags = tuple(lags)
        self.scales = len(self.lags)
        self.graph_operator = operator
        self.q = float(q)
        self.use_commonality = bool(use_commonality)
        self.uses_internal_input_stats = True
        self.outputs_log_variance = True
        self.input_stats_frozen = False
        self.target_stats_frozen = False
        self.register_buffer(
            "input_mean", torch.zeros(len(self.HORIZONS), self.assets, 3),
        )
        self.register_buffer(
            "input_std", torch.ones(len(self.HORIZONS), self.assets, 3),
        )
        self.register_buffer(
            "target_mean", torch.zeros(len(self.HORIZONS), self.assets),
        )
        self.register_buffer(
            "target_std", torch.ones(len(self.HORIZONS), self.assets),
        )
        self.encoders = nn.ModuleList([
            HARNetScaleEncoder(self.lags)
            for _ in range(len(self.HORIZONS) * self.assets)
        ])
        self.graph_norm = nn.LayerNorm(3)
        self.graph_filter = nn.Sequential(
            nn.Linear(3, int(hidden_size), bias=False), nn.Tanh(),
            nn.Linear(int(hidden_size), 3, bias=False),
        )
        self.gcn_projection = nn.Linear(3, 3, bias=False)
        self.gat_query = nn.Linear(3, 1, bias=False)
        self.gat_key = nn.Linear(3, 1, bias=False)
        self.gsphar_coefficients = nn.Parameter(torch.zeros(self.scales, 2, 3))
        with torch.no_grad():
            self.gsphar_coefficients[:, 0].fill_(1.0)
            self.gsphar_coefficients[:, 1].fill_(0.05)
        self.commonality_heads = nn.ModuleList([
            nn.Linear(self.scales * 3, self.assets, bias=False)
            for _ in self.HORIZONS
        ])
        self.graph_heads = nn.ModuleList([
            nn.Linear(self.scales * 3, 1, bias=False)
            for _ in self.HORIZONS
        ])
        self.commonality_gate = nn.Parameter(torch.zeros(len(self.HORIZONS), self.assets))
        self.graph_gate = nn.Parameter(torch.zeros(len(self.HORIZONS), self.scales))
        # Heads must be non-zero so zero gates still receive a first-step gradient.
        for head in (*self.commonality_heads, *self.graph_heads):
            nn.init.xavier_uniform_(head.weight, gain=0.1)

    def _encoder(self, horizon: int, asset: int) -> HARNetScaleEncoder:
        return self.encoders[horizon * self.assets + asset]

    @torch.no_grad()
    def initialize_from_harnets(
        self, sources: Sequence[Sequence[SameClockHARNet]], *,
        input_mean: torch.Tensor | None = None,
        input_std: torch.Tensor | None = None,
        target_mean: torch.Tensor | None = None,
        target_std: torch.Tensor | None = None,
    ) -> None:
        if len(sources) != len(self.HORIZONS) or any(len(row) != self.assets for row in sources):
            raise ValueError("sources must contain one H1 and one H3 HARNet per asset")
        for horizon, row in enumerate(sources):
            for asset, source in enumerate(row):
                self._encoder(horizon, asset).initialize_from_harnet(source)
        self.commonality_gate.zero_()
        self.graph_gate.zero_()
        if input_mean is not None or input_std is not None:
            if input_mean is None or input_std is None:
                raise ValueError("input_mean and input_std must be supplied together")
            self.set_input_preprocessing(input_mean, input_std, frozen=True)
        if target_mean is not None or target_std is not None:
            if target_mean is None or target_std is None:
                raise ValueError("target_mean and target_std must be supplied together")
            self.set_target_preprocessing(target_mean, target_std, frozen=True)

    @torch.no_grad()
    def set_input_preprocessing(
        self, input_mean: torch.Tensor, input_std: torch.Tensor, *, frozen: bool = True,
    ) -> None:
        mean = torch.as_tensor(input_mean, dtype=self.input_mean.dtype, device=self.input_mean.device)
        std = torch.as_tensor(input_std, dtype=self.input_std.dtype, device=self.input_std.device)
        expected = (len(self.HORIZONS), self.assets, 3)
        if tuple(mean.shape) != expected or tuple(std.shape) != expected:
            raise ValueError("input preprocessing must be [horizon,asset,channel]")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("input preprocessing must be finite with positive std")
        self.input_mean.copy_(mean)
        self.input_std.copy_(std)
        self.input_stats_frozen = bool(frozen)

    @torch.no_grad()
    def set_target_preprocessing(
        self, target_mean: torch.Tensor, target_std: torch.Tensor, *, frozen: bool = True,
    ) -> None:
        mean = torch.as_tensor(target_mean, dtype=self.target_mean.dtype, device=self.target_mean.device)
        std = torch.as_tensor(target_std, dtype=self.target_std.dtype, device=self.target_std.device)
        expected = (len(self.HORIZONS), self.assets)
        if tuple(mean.shape) != expected or tuple(std.shape) != expected:
            raise ValueError("target preprocessing must be [horizon,asset]")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("target preprocessing must be finite with positive std")
        self.target_mean.copy_(mean)
        self.target_std.copy_(std)
        self.target_stats_frozen = bool(frozen)

    def _horizon_observed(self, observed: torch.Tensor, horizon: int, asset: int) -> torch.Tensor:
        return (
            observed[:, :, asset, :] - self.input_mean[horizon, asset]
        ) / self.input_std[horizon, asset]

    def scale_features(self, observed: torch.Tensor) -> torch.Tensor:
        if observed.ndim != 4 or observed.shape[2:] != (self.assets, 3):
            raise ValueError("observed must be [batch, history, asset, 3]")
        horizons = []
        for horizon in range(len(self.HORIZONS)):
            horizons.append(torch.stack([
                self._encoder(horizon, asset).scale_features(
                    self._horizon_observed(observed, horizon, asset)
                )
                for asset in range(self.assets)
            ], dim=1))
        return torch.stack(horizons, dim=1)  # [B,H,A,S,C]

    def _local_predictions(self, observed: torch.Tensor) -> torch.Tensor:
        values = []
        for horizon in range(len(self.HORIZONS)):
            values.append(torch.cat([
                self._encoder(horizon, asset)(
                    self._horizon_observed(observed, horizon, asset)
                )
                for asset in range(self.assets)
            ], dim=1))
        standardized = torch.stack(values, dim=-1)  # [B,A,H]
        mean = self.target_mean.transpose(0, 1)[None, :, :]
        std = self.target_std.transpose(0, 1)[None, :, :]
        return standardized * std + mean

    def _prepare_graph(self, adjacency: torch.Tensor, batch: int) -> torch.Tensor:
        graph = torch.as_tensor(adjacency, dtype=torch.float32, device=self.graph_gate.device)
        if graph.ndim == 3:  # [S,A,A]
            graph = graph.unsqueeze(0).expand(batch, -1, -1, -1)
        if graph.ndim != 4 or graph.shape != (batch, self.scales, self.assets, self.assets):
            raise ValueError("adjacency must be [scale,asset,asset] or [batch,scale,asset,asset]")
        graph = graph.clone()
        diagonal = torch.arange(self.assets, device=graph.device)
        graph[:, :, diagonal, diagonal] = 0.0
        return graph

    @staticmethod
    def _normalized(graph: torch.Tensor) -> torch.Tensor:
        degree = graph.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return graph / degree

    def _graph_messages(self, idiosyncratic: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        # idiosyncratic: [B,H,A,S,C], graph: [B,S,A,A]
        if self.graph_operator == "gat":
            node = idiosyncratic.permute(0, 1, 3, 2, 4)  # [B,H,S,A,C]
            query = self.gat_query(node).squeeze(-1).unsqueeze(-1)
            key = self.gat_key(node).squeeze(-1).unsqueeze(-2)
            edge_strength = graph.abs().clamp_min(1e-12).log()[:, None]
            score = query + key + edge_strength
            mask = graph[:, None] != 0
            score = score.masked_fill(~mask, -torch.inf)
            empty = ~mask.any(dim=-1, keepdim=True)
            score = torch.where(empty, torch.zeros_like(score), score)
            attention = torch.softmax(score, dim=-1) * mask
            message = torch.einsum("bhsaq,bhsqc->bhsac", attention, node)
            message = message.permute(0, 1, 3, 2, 4)
        elif self.graph_operator == "gsphar":
            matrix = graph
            symmetric = (matrix + matrix.transpose(-1, -2)) / 2.0
            degree = symmetric.abs().sum(dim=-1).clamp_min(1e-6)
            normalized = symmetric / torch.sqrt(degree.unsqueeze(-1) * degree.unsqueeze(-2))
            theta = 2.0 * torch.pi * self.q * (matrix - matrix.transpose(-1, -2))
            magnetic = normalized.to(torch.complex64) * torch.exp(1j * theta.to(torch.complex64))
            identity = torch.eye(self.assets, device=graph.device, dtype=torch.complex64)
            laplacian = identity - magnetic
            laplacian = (laplacian + laplacian.conj().transpose(-1, -2)) / 2.0
            eigenvalues, vectors = torch.linalg.eigh(laplacian)
            inverse = vectors.conj().transpose(-1, -2)
            node = idiosyncratic.permute(0, 3, 1, 2, 4).to(torch.complex64)
            spectral = torch.einsum("bsij,bshjc->bshic", inverse, node)
            response = (
                self.gsphar_coefficients[None, :, 0, None, :]
                + eigenvalues[..., None] * self.gsphar_coefficients[None, :, 1, None, :]
            )
            spectral = spectral * response[:, :, None].to(torch.complex64)
            message = torch.einsum("bsij,bshjc->bshic", vectors, spectral).real
            message = message.permute(0, 2, 3, 1, 4)
        else:
            normalized = self._normalized(graph)
            message = torch.einsum("bsij,bhjsc->bhisc", normalized, idiosyncratic)
        if self.graph_operator == "mean":
            filtered = message
        elif self.graph_operator == "gcn":
            filtered = torch.tanh(self.gcn_projection(self.graph_norm(message)))
        else:
            filtered = self.graph_filter(self.graph_norm(message))
        has_edges = graph.ne(0).any(dim=-1).any(dim=-1)
        return filtered * has_edges[:, None, None, :, None]

    def components(self, observed: torch.Tensor, adjacency: torch.Tensor) -> dict[str, torch.Tensor]:
        local = self._local_predictions(observed)
        scale = self.scale_features(observed)
        market = scale.mean(dim=2)  # [B,H,S,C]
        idiosyncratic = scale - market.unsqueeze(2)
        graph = self._prepare_graph(adjacency, observed.shape[0])
        messages = self._graph_messages(idiosyncratic, graph)
        gated = messages * torch.tanh(self.graph_gate)[None, :, None, :, None]
        graph_residual = torch.stack([
            self.graph_heads[horizon](gated[:, horizon].flatten(start_dim=2)).squeeze(-1)
            for horizon in range(len(self.HORIZONS))
        ], dim=-1)
        if self.use_commonality:
            commonality_residual = torch.stack([
                self.commonality_heads[horizon](market[:, horizon].flatten(start_dim=1))
                * torch.tanh(self.commonality_gate[horizon])[None, :]
                for horizon in range(len(self.HORIZONS))
            ], dim=-1)
        else:
            commonality_residual = torch.zeros_like(local)
        return {
            "prediction": local + commonality_residual + graph_residual,
            "local_prediction": local,
            "commonality_residual": commonality_residual,
            "graph_residual": graph_residual,
            "scale_features": scale,
            "market_features": market,
        }

    def forward(self, observed: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.components(observed, adjacency)["prediction"]

    def set_training_stage(self, stage: str) -> None:
        """Freeze local features or selectively unfreeze long scales."""

        normalized = str(stage).strip().lower()
        if normalized not in {"graph_only", "long_scales"}:
            raise ValueError("training stage must be graph_only or long_scales")
        for encoder in self.encoders:
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)
            if normalized == "long_scales":
                for layer in (*encoder.legacy_layers[1:], *encoder.signed_layers[1:]):
                    for parameter in layer.parameters():
                        parameter.requires_grad_(True)
                encoder.legacy_output.weight.requires_grad_(True)
                encoder.signed_scale_output.weight.requires_grad_(True)


class SingleHorizonScaleGraphHARNet(nn.Module):
    """H1-only view of the scale graph core with no H3 optimization path."""

    def __init__(
        self, assets: int, *, hidden_size: int = 8,
        graph_operator: str = "gnnhar", q: float = 0.25,
        use_commonality: bool = True,
    ) -> None:
        super().__init__()
        self.assets = int(assets)
        self.core = MultiTaskScaleGraphHARNet(
            assets, hidden_size=hidden_size, graph_operator=graph_operator, q=q,
            use_commonality=use_commonality,
        )

    @torch.no_grad()
    def initialize_from_harnets(
        self, sources: Sequence[SameClockHARNet], *,
        input_mean: torch.Tensor, input_std: torch.Tensor,
        target_mean: torch.Tensor, target_std: torch.Tensor,
    ) -> None:
        if len(sources) != self.assets:
            raise ValueError("one H1 HARNet source is required per asset")
        mean_x = torch.as_tensor(input_mean).reshape(self.assets, 3)
        std_x = torch.as_tensor(input_std).reshape(self.assets, 3)
        mean_y = torch.as_tensor(target_mean).reshape(self.assets)
        std_y = torch.as_tensor(target_std).reshape(self.assets)
        self.core.initialize_from_harnets(
            (tuple(sources), tuple(sources)),
            input_mean=torch.stack((mean_x, mean_x)),
            input_std=torch.stack((std_x, std_x)),
            target_mean=torch.stack((mean_y, mean_y)),
            target_std=torch.stack((std_y, std_y)),
        )

    def set_training_stage(self, stage: str) -> None:
        self.core.set_training_stage(stage)
        # The duplicate H3 compatibility encoders/heads are never part of an
        # H1 loss and are explicitly frozen to keep optimizer audits clear.
        for asset in range(self.assets):
            for parameter in self.core._encoder(1, asset).parameters():
                parameter.requires_grad_(False)
        for module in (self.core.commonality_heads[1], self.core.graph_heads[1]):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def components(self, observed: torch.Tensor, adjacency: torch.Tensor) -> dict[str, torch.Tensor]:
        values = self.core.components(observed, adjacency)
        return {
            "prediction": values["prediction"][..., 0],
            "local_prediction": values["local_prediction"][..., 0],
            "commonality_residual": values["commonality_residual"][..., 0],
            "graph_residual": values["graph_residual"][..., 0],
            "scale_features": values["scale_features"][:, 0],
            "market_features": values["market_features"][:, 0],
        }

    def forward(self, observed: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.components(observed, adjacency)["prediction"]


def harnet_svj_state_hash(model: SameClockHARNet) -> str:
    """Hash a HARNet-SVJ state dict deterministically for transfer audits."""
    if not isinstance(model, SameClockHARNet) or not model.svj:
        raise TypeError("harnet_svj_state_hash requires a SameClockHARNet-SVJ model")
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def save_harnet_svj_weights(
    model: SameClockHARNet,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    input_mean: Any | None = None,
    input_std: Any | None = None,
    target_mean: Any | None = None,
    target_std: Any | None = None,
    target_transform: str | None = None,
) -> str:
    """Save portable HARNet-SVJ weights and preprocessing provenance.

    The state dict alone is not sufficient for same-clock inference: the
    trainer standardizes inputs and transformed targets from the fit window.
    When supplied, those arrays are stored in the artifact so predictions can
    be reproduced without relying on an external result object.
    """
    if not isinstance(model, SameClockHARNet) or not model.svj:
        raise TypeError("save_harnet_svj_weights requires a SameClockHARNet-SVJ model")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    digest = harnet_svj_state_hash(model)
    payload = {
        "schema": "harnetsvj_weights_v1",
        "model": "SameClockHARNet-SVJ",
        "lags": list(model.lags),
        "state_hash": digest,
        "metadata": dict(metadata or {}),
        "state_dict": state,
    }
    preprocessing = {
        "input_mean": None if input_mean is None else torch.as_tensor(input_mean).detach().cpu(),
        "input_std": None if input_std is None else torch.as_tensor(input_std).detach().cpu(),
        "target_mean": None if target_mean is None else torch.as_tensor(target_mean).detach().cpu(),
        "target_std": None if target_std is None else torch.as_tensor(target_std).detach().cpu(),
        "target_transform": None if target_transform is None else str(target_transform),
    }
    payload["preprocessing"] = preprocessing
    torch.save(payload, destination)
    return digest


def load_harnet_svj_weights(
    model: SameClockHARNet,
    path: str | Path,
    *,
    strict: bool = True,
) -> str:
    """Load and hash-check a portable HARNet-SVJ weight artifact."""
    payload = load_harnet_svj_artifact(model, path, strict=strict)
    return str(payload["state_hash"])


def load_harnet_svj_artifact(
    model: SameClockHARNet,
    path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load weights plus the scaler metadata required by graph transfer."""

    if not isinstance(model, SameClockHARNet) or not model.svj:
        raise TypeError("load_harnet_svj_weights requires a SameClockHARNet-SVJ model")
    payload = torch.load(Path(path), map_location="cpu")
    if payload.get("schema") != "harnetsvj_weights_v1":
        raise ValueError("unsupported HARNet-SVJ weight artifact schema")
    if tuple(payload.get("lags", ())) != tuple(model.lags):
        raise ValueError("HARNet-SVJ weight lags do not match the target model")
    model.load_state_dict(payload["state_dict"], strict=bool(strict))
    digest = harnet_svj_state_hash(model)
    if digest != payload.get("state_hash"):
        raise ValueError("HARNet-SVJ weight artifact hash mismatch")
    preprocessing = payload.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise ValueError("HARNet-SVJ artifact is missing preprocessing metadata")
    return dict(payload)


def multitask_preprocessing_from_artifacts(
    artifacts: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack two horizons of per-asset scaler metadata for graph transfer."""

    if len(artifacts) != 2 or not artifacts[0] or len(artifacts[0]) != len(artifacts[1]):
        raise ValueError("artifacts must be a rectangular [2, asset] collection")
    input_means = []
    input_stds = []
    target_means = []
    target_stds = []
    for horizon in artifacts:
        horizon_input_mean = []
        horizon_input_std = []
        horizon_target_mean = []
        horizon_target_std = []
        for artifact in horizon:
            preprocessing = artifact.get("preprocessing")
            if not isinstance(preprocessing, Mapping):
                raise ValueError("artifact is missing preprocessing metadata")
            values = {
                name: preprocessing.get(name)
                for name in ("input_mean", "input_std", "target_mean", "target_std")
            }
            if any(value is None for value in values.values()):
                raise ValueError("artifact preprocessing is incomplete")
            input_mean = torch.as_tensor(values["input_mean"]).reshape(-1, 3)
            input_std = torch.as_tensor(values["input_std"]).reshape(-1, 3)
            target_mean = torch.as_tensor(values["target_mean"]).reshape(-1)
            target_std = torch.as_tensor(values["target_std"]).reshape(-1)
            if input_mean.shape[0] != 1 or input_std.shape[0] != 1:
                raise ValueError("per-asset artifact input scaler must contain one asset")
            if target_mean.numel() != 1 or target_std.numel() != 1:
                raise ValueError("per-asset artifact target scaler must be scalar")
            horizon_input_mean.append(input_mean[0])
            horizon_input_std.append(input_std[0])
            horizon_target_mean.append(target_mean[0])
            horizon_target_std.append(target_std[0])
        input_means.append(torch.stack(horizon_input_mean))
        input_stds.append(torch.stack(horizon_input_std))
        target_means.append(torch.stack(horizon_target_mean))
        target_stds.append(torch.stack(horizon_target_std))
    return (
        torch.stack(input_means), torch.stack(input_stds),
        torch.stack(target_means), torch.stack(target_stds),
    )


class SameClockHARNetDiagnostic4(nn.Module):
    """Validation-only four-channel challenger.

    The released HARNet-SVJ contract is intentionally unchanged.  This
    adapter appends causal 1/5/20/80-day summaries of one extra channel to
    the deterministic SVJ feature vector and is only used by validation
    fourth-channel ablations; it is not registered as a production model.
    """

    def __init__(self, *, lags: tuple[int, ...] = (1, 5, 20, 80)) -> None:
        super().__init__()
        self.local = SameClockHARNet(lags=lags, svj=True, filters=2)
        self.extra_lags = (1, 5, 20, 80)
        self.output = nn.Linear(
            self.local.output.in_features + len(self.extra_lags), 1, bias=False,
        )
        nn.init.xavier_uniform_(self.output.weight)

    def _features(self, observed: torch.Tensor) -> torch.Tensor:
        if observed.ndim != 3 or observed.shape[-1] != 4:
            raise ValueError("diagnostic HARNet observed input must be [batch, history, 4]")
        local = self.local.feature_values(observed[..., :3])
        extra = observed[..., 3]
        extra_features = [
            extra[:, -lag:].mean(dim=1, keepdim=True) for lag in self.extra_lags
        ]
        return torch.cat((local, *extra_features), dim=-1)

    @torch.no_grad()
    def initialize_output(self, observed: torch.Tensor, target: torch.Tensor, method: str = "ols") -> None:
        name = str(method).lower().replace("-", "_")
        if name not in {"ols", "har_ols", "log_ols"}:
            raise ValueError("diagnostic HARNet output initialization must be OLS-compatible")
        features = self._features(observed).reshape(-1, self.output.in_features).double()
        values = torch.as_tensor(target, device=features.device, dtype=torch.float64).reshape(-1, 1)
        solution = torch.linalg.lstsq(features, values).solution.T.to(dtype=self.output.weight.dtype)
        if solution.shape != self.output.weight.shape or not torch.isfinite(solution).all():
            raise FloatingPointError("non-finite diagnostic HARNet OLS initialization")
        self.output.weight.copy_(solution)

    def forward(self, observed: torch.Tensor, adjacency=None) -> torch.Tensor:
        return self.output(self._features(observed))


class SameClockHARNetFourthChannel(nn.Module):
    """HARNet-SVJ with an explicit fourth-channel connection ablation.

    ``connection='conv'`` uses causal 1/5/20/80 summaries of the fourth
    channel.  ``connection='direct'`` uses only its current-origin value.
    The feature itself is supplied by the dataset, so raw versus Train-only
    orthogonalized inputs remain an independent experimental factor.
    """

    CONNECTIONS = ("conv", "direct")

    def __init__(self, *, lags: tuple[int, ...] = (1, 5, 20, 80), connection: str = "conv") -> None:
        super().__init__()
        connection = str(connection).lower().strip()
        if connection not in self.CONNECTIONS:
            raise ValueError(f"connection must be one of {self.CONNECTIONS}")
        self.connection = connection
        self.local = SameClockHARNet(lags=lags, svj=True, filters=2)
        self.extra_lags = tuple(lags)
        extra_features = len(self.extra_lags) if connection == "conv" else 1
        self.output = nn.Linear(self.local.output.in_features + extra_features, 1, bias=False)
        nn.init.xavier_uniform_(self.output.weight)

    def _features(self, observed: torch.Tensor) -> torch.Tensor:
        if observed.ndim != 3 or observed.shape[-1] != 4:
            raise ValueError("fourth-channel HARNet input must be [batch, history, 4]")
        local = self.local.feature_values(observed[..., :3])
        extra = observed[..., 3]
        if self.connection == "direct":
            additional = extra[:, -1:].contiguous()
        else:
            additional = torch.cat([
                extra[:, -lag:].mean(dim=1, keepdim=True)
                for lag in self.extra_lags
            ], dim=-1)
        return torch.cat((local, additional), dim=-1)

    @torch.no_grad()
    def initialize_output(self, observed: torch.Tensor, target: torch.Tensor, method: str = "ols") -> None:
        if str(method).lower().replace("-", "_") not in {"ols", "har_ols", "log_ols"}:
            raise ValueError("fourth-channel output initialization must be OLS-compatible")
        features = self._features(observed).reshape(-1, self.output.in_features).double()
        values = torch.as_tensor(target, device=features.device, dtype=torch.float64).reshape(-1, 1)
        solution = torch.linalg.lstsq(features, values).solution.T.to(dtype=self.output.weight.dtype)
        if solution.shape != self.output.weight.shape or not torch.isfinite(solution).all():
            raise FloatingPointError("non-finite fourth-channel OLS initialization")
        self.output.weight.copy_(solution)

    def forward(self, observed: torch.Tensor, adjacency=None) -> torch.Tensor:
        return self.output(self._features(observed))


class SameClockHARNetFiveChannelDirect(nn.Module):
    """HARNet-SVJ with two current-origin channels connected linearly."""

    def __init__(self, *, lags: tuple[int, ...] = (1, 5, 20, 80)) -> None:
        super().__init__()
        self.local = SameClockHARNet(lags=lags, svj=True, filters=2)
        self.output = nn.Linear(self.local.output.in_features + 2, 1, bias=False)
        nn.init.xavier_uniform_(self.output.weight)

    def _features(self, observed: torch.Tensor) -> torch.Tensor:
        if observed.ndim != 3 or observed.shape[-1] != 5:
            raise ValueError("five-channel HARNet input must be [batch, history, 5]")
        return torch.cat((self.local.feature_values(observed[..., :3]), observed[:, -1, 3:5]), dim=-1)

    @torch.no_grad()
    def initialize_output(self, observed: torch.Tensor, target: torch.Tensor, method: str = "ols") -> None:
        if str(method).lower().replace("-", "_") not in {"ols", "har_ols", "log_ols"}:
            raise ValueError("five-channel output initialization must be OLS-compatible")
        features = self._features(observed).reshape(-1, self.output.in_features).double()
        values = torch.as_tensor(target, device=features.device, dtype=torch.float64).reshape(-1, 1)
        solution = torch.linalg.lstsq(features, values).solution.T.to(dtype=self.output.weight.dtype)
        if solution.shape != self.output.weight.shape or not torch.isfinite(solution).all():
            raise FloatingPointError("non-finite five-channel OLS initialization")
        self.output.weight.copy_(solution)

    def forward(self, observed: torch.Tensor, adjacency=None) -> torch.Tensor:
        return self.output(self._features(observed))


class SameClockHARNetDirectExtras(nn.Module):
    """HARNet-SVJ with current-origin scalar channels in the linear head."""

    def __init__(self, extra_channels: int, *, lags: tuple[int, ...] = (1, 5, 20, 80)) -> None:
        super().__init__()
        if int(extra_channels) < 1:
            raise ValueError("DirectExtras requires at least one extra channel")
        self.extra_channels = int(extra_channels)
        self.local = SameClockHARNet(lags=lags, svj=True, filters=2)
        self.output = nn.Linear(self.local.output.in_features + self.extra_channels, 1, bias=False)
        nn.init.xavier_uniform_(self.output.weight)

    def _features(self, observed: torch.Tensor) -> torch.Tensor:
        expected = 3 + self.extra_channels
        if observed.ndim != 3 or observed.shape[-1] != expected:
            raise ValueError(f"direct-extra HARNet input must be [batch, history, {expected}]")
        return torch.cat((self.local.feature_values(observed[..., :3]), observed[:, -1, 3:]), dim=-1)

    @torch.no_grad()
    def initialize_output(self, observed: torch.Tensor, target: torch.Tensor, method: str = "ols") -> None:
        if str(method).lower().replace("-", "_") not in {"ols", "har_ols", "log_ols"}:
            raise ValueError("direct-extra output initialization must be OLS-compatible")
        features = self._features(observed).reshape(-1, self.output.in_features).double()
        values = torch.as_tensor(target, device=features.device, dtype=torch.float64).reshape(-1, 1)
        solution = torch.linalg.lstsq(features, values).solution.T.to(dtype=self.output.weight.dtype)
        if solution.shape != self.output.weight.shape or not torch.isfinite(solution).all():
            raise FloatingPointError("non-finite direct-extra OLS initialization")
        self.output.weight.copy_(solution)

    def forward(self, observed: torch.Tensor, adjacency=None) -> torch.Tensor:
        return self.output(self._features(observed))


class SameClockHARNetRawDirectExtras(nn.Module):
    """One-channel raw-RVol HARNet with current-origin extra regressors.

    ``SameClockHARNetDirectExtras`` is intentionally SVJ-only.  This adapter
    keeps the original one-channel causal HARNet backbone and appends scalar
    features read at the current origin, which is the contract used by the
    raw-RVol state experiments.
    """

    def __init__(self, extra_channels: int, *, lags: tuple[int, ...] = (1, 5, 20, 80)) -> None:
        super().__init__()
        if int(extra_channels) < 1:
            raise ValueError("RawDirectExtras requires at least one extra channel")
        self.extra_channels = int(extra_channels)
        self.local = SameClockHARNet(lags=lags, svj=False, filters=1)
        self.output = nn.Linear(self.local.output.in_features + self.extra_channels, 1, bias=False)
        nn.init.xavier_uniform_(self.output.weight)

    def _features(self, observed: torch.Tensor) -> torch.Tensor:
        expected = 1 + self.extra_channels
        if observed.ndim != 3 or observed.shape[-1] != expected:
            raise ValueError(f"raw direct-extra HARNet input must be [batch, history, {expected}]")
        return torch.cat((self.local.feature_values(observed[..., :1]), observed[:, -1, 1:]), dim=-1)

    @torch.no_grad()
    def initialize_output(self, observed: torch.Tensor, target: torch.Tensor, method: str = "ols") -> None:
        if str(method).lower().replace("-", "_") not in {"ols", "har_ols", "log_ols"}:
            raise ValueError("raw direct-extra output initialization must be OLS-compatible")
        features = self._features(observed).reshape(-1, self.output.in_features).double()
        values = torch.as_tensor(target, device=features.device, dtype=torch.float64).reshape(-1, 1)
        solution = torch.linalg.lstsq(features, values).solution.T.to(dtype=self.output.weight.dtype)
        if solution.shape != self.output.weight.shape or not torch.isfinite(solution).all():
            raise FloatingPointError("non-finite raw direct-extra OLS initialization")
        self.output.weight.copy_(solution)

    def forward(self, observed: torch.Tensor, adjacency=None) -> torch.Tensor:
        return self.output(self._features(observed))


class SameClockHARNetLongLinearHead(nn.Module):
    """Adaptive HARNet-SVJ plus a current-origin long-RVOL linear branch.

    The first ``base_extra_channels`` remain the existing DirectExtras path.
    The final ``long_channels`` are deliberately read only at the current
    origin; they are not passed through the HARNet cross-day lag stack.
    """

    def __init__(
        self,
        base_extra_channels: int,
        long_channels: int,
        *,
        lags: tuple[int, ...] = (1, 5, 20, 80),
    ) -> None:
        super().__init__()
        if int(base_extra_channels) < 1 or int(long_channels) < 1:
            raise ValueError("long linear head requires positive base and long channel counts")
        self.base_extra_channels = int(base_extra_channels)
        self.long_channels = int(long_channels)
        self.base = SameClockHARNetDirectExtras(self.base_extra_channels, lags=lags)
        self.long_head = nn.Linear(self.long_channels + 1, 1, bias=False)
        nn.init.zeros_(self.long_head.weight)

    def _split(self, observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = 3 + self.base_extra_channels + self.long_channels
        if observed.ndim != 3 or observed.shape[-1] != expected:
            raise ValueError(f"long-linear HARNet input must be [batch, history, {expected}]")
        base_end = 3 + self.base_extra_channels
        base = observed[..., :base_end]
        long = observed[:, -1, base_end:]
        return base, long

    def forward(self, observed: torch.Tensor, adjacency=None) -> torch.Tensor:
        base, long = self._split(observed)
        design = torch.cat((torch.ones_like(long[:, :1]), long), dim=-1)
        return self.base(base) + self.long_head(design)

    @torch.no_grad()
    def initialize_output(self, observed: torch.Tensor, target: torch.Tensor, method: str = "ols") -> None:
        """Initialize the base head, then fit the long branch to Train residuals."""

        name = str(method).lower().replace("-", "_")
        if name not in {"ols", "har_ols", "log_ols"}:
            raise ValueError("long-linear initialization must be OLS-compatible")
        base, long = self._split(observed)
        self.base.initialize_output(base, target, name)
        base_prediction = self.base(base).reshape(-1, 1).double()
        values = torch.as_tensor(target, device=base_prediction.device, dtype=torch.float64).reshape(-1, 1)
        design = torch.cat((torch.ones_like(long[:, :1]), long), dim=-1).reshape(-1, self.long_channels + 1).double()
        residual = values - base_prediction
        solution = torch.linalg.lstsq(design, residual).solution.T.to(dtype=self.long_head.weight.dtype)
        if solution.shape != self.long_head.weight.shape or not torch.isfinite(solution).all():
            raise FloatingPointError("non-finite long-linear OLS initialization")
        self.long_head.weight.copy_(solution)


class SameClockHARNetChannelSpecific(nn.Module):
    """Causal linear HAR head with an independent lag ladder per channel."""

    def __init__(self, channel_lags: tuple[tuple[int, ...], ...]) -> None:
        super().__init__()
        if not channel_lags:
            raise ValueError("channel-specific HARNet requires at least one channel")
        normalized = []
        for lags in channel_lags:
            values = tuple(int(lag) for lag in lags)
            if not values or values[0] != 1 or tuple(sorted(set(values))) != values or any(lag < 1 for lag in values):
                raise ValueError("each channel lag ladder must be sorted, unique, positive, and start at one")
            normalized.append(values)
        self.channel_lags = tuple(normalized)
        self.output = nn.Linear(1 + sum(len(lags) for lags in self.channel_lags), 1, bias=False)
        nn.init.xavier_uniform_(self.output.weight)

    def _features(self, observed: torch.Tensor) -> torch.Tensor:
        if observed.ndim != 3 or observed.shape[-1] != len(self.channel_lags):
            raise ValueError(f"channel-specific HARNet input must be [batch, history, {len(self.channel_lags)}]")
        features = [torch.ones_like(observed[:, -1, :1])]
        for channel, lags in enumerate(self.channel_lags):
            values = observed[:, :, channel:channel + 1]
            features.extend(values[:, -lag:].mean(dim=1) for lag in lags)
        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def initialize_output(self, observed: torch.Tensor, target: torch.Tensor, method: str = "ols") -> None:
        if str(method).lower().replace("-", "_") not in {"ols", "har_ols", "log_ols"}:
            raise ValueError("channel-specific output initialization must be OLS-compatible")
        features = self._features(observed).reshape(-1, self.output.in_features).double()
        values = torch.as_tensor(target, device=features.device, dtype=torch.float64).reshape(-1, 1)
        solution = torch.linalg.lstsq(features, values).solution.T.to(dtype=self.output.weight.dtype)
        if solution.shape != self.output.weight.shape or not torch.isfinite(solution).all():
            raise FloatingPointError("non-finite channel-specific OLS initialization")
        self.output.weight.copy_(solution)

    def forward(self, observed: torch.Tensor, adjacency=None) -> torch.Tensor:
        return self.output(self._features(observed))


class SameClockHARNetShortLinearHead(nn.Module):
    """HARNet-SVJ plus a current-origin-only short-RV linear head."""

    def __init__(self, *, lags: tuple[int, ...] = (1, 5, 20, 80)) -> None:
        super().__init__()
        self.local = SameClockHARNet(lags=lags, svj=True, filters=2)
        self.output = nn.Linear(self.local.output.in_features + 1, 1, bias=False)
        nn.init.xavier_uniform_(self.output.weight)

    def _features(self, observed: torch.Tensor) -> torch.Tensor:
        if observed.ndim != 3 or observed.shape[-1] != 4:
            raise ValueError("short-linear HARNet observed input must be [batch, history, 4]")
        return torch.cat((self.local.feature_values(observed[..., :3]), observed[:, -1, 3:4]), dim=-1)

    @torch.no_grad()
    def initialize_output(self, observed: torch.Tensor, target: torch.Tensor, method: str = "ols") -> None:
        if str(method).lower().replace("-", "_") not in {"ols", "har_ols", "log_ols"}:
            raise ValueError("short-linear output initialization must be OLS-compatible")
        features = self._features(observed).reshape(-1, self.output.in_features).double()
        values = torch.as_tensor(target, device=features.device, dtype=torch.float64).reshape(-1, 1)
        solution = torch.linalg.lstsq(features, values).solution.T.to(dtype=self.output.weight.dtype)
        if solution.shape != self.output.weight.shape or not torch.isfinite(solution).all():
            raise FloatingPointError("non-finite short-linear OLS output initialization")
        self.output.weight.copy_(solution)

    def forward(self, observed: torch.Tensor, adjacency=None) -> torch.Tensor:
        return self.output(self._features(observed))


class SameClockHARNetONDSProbabilityHead(nn.Module):
    """HARNet-SVJ plus ONDS lag summaries and a current causal probability."""

    def __init__(self, *, lags: tuple[int, ...] = (1, 5, 20, 80)) -> None:
        super().__init__(); self.local=SameClockHARNet(lags=lags,svj=True,filters=2); self.extra_lags=tuple(lags); self.output=nn.Linear(self.local.output.in_features+len(lags)+1,1,bias=False); nn.init.xavier_uniform_(self.output.weight)

    def _features(self, observed: torch.Tensor) -> torch.Tensor:
        if observed.ndim!=3 or observed.shape[-1]!=5: raise ValueError("ONDS-probability HARNet input must be [batch, history, 5]")
        local=self.local.feature_values(observed[...,:3]); onds=observed[...,3]; summaries=[onds[:,-lag:].mean(dim=1,keepdim=True) for lag in self.extra_lags]
        return torch.cat((local,*summaries,observed[:,-1,4:5]),dim=-1)

    @torch.no_grad()
    def initialize_output(self, observed: torch.Tensor, target: torch.Tensor, method: str = "ols") -> None:
        if str(method).lower().replace("-","_") not in {"ols","har_ols","log_ols"}: raise ValueError("ONDS-probability initialization must be OLS-compatible")
        features=self._features(observed).reshape(-1,self.output.in_features).double(); values=torch.as_tensor(target,device=features.device,dtype=torch.float64).reshape(-1,1); solution=torch.linalg.lstsq(features,values).solution.T.to(dtype=self.output.weight.dtype)
        if solution.shape!=self.output.weight.shape or not torch.isfinite(solution).all(): raise FloatingPointError("non-finite ONDS-probability OLS initialization")
        self.output.weight.copy_(solution)

    def forward(self, observed: torch.Tensor, adjacency=None) -> torch.Tensor: return self.output(self._features(observed))


class OvernightResidualHead(nn.Module):
    """Small causal correction head for a frozen H80 prediction.

    The head never changes the three-channel HARNet contract.  ``state`` is a
    row of origin-known overnight/current-RV features and ``base_prediction``
    is the frozen H80 prediction.  A sigmoid scale keeps the correction from
    becoming an unconstrained replacement forecaster.
    """

    def __init__(self, state_size: int, *, hidden_size: int = 0) -> None:
        super().__init__()
        if int(state_size) < 1:
            raise ValueError("OvernightResidualHead requires at least one state feature")
        if int(hidden_size) < 0:
            raise ValueError("OvernightResidualHead hidden_size must be non-negative")
        self.state_size = int(state_size)
        self.hidden_size = int(hidden_size)
        if hidden_size:
            self.state = nn.Sequential(
                nn.Linear(self.state_size, int(hidden_size)), nn.Tanh(),
                nn.Linear(int(hidden_size), 1),
            )
        else:
            self.state = nn.Linear(self.state_size, 1)
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, base_prediction: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if base_prediction.ndim not in {1, 2}:
            raise ValueError("base_prediction must be [batch] or [batch, 1]")
        if state.ndim != 2 or state.shape[0] != base_prediction.shape[0] or state.shape[1] != self.state_size:
            raise ValueError("state must be [batch, state_size] aligned to base_prediction")
        base = base_prediction.reshape(-1, 1)
        correction = self.state(state)
        scale = torch.sigmoid(self.residual_scale)
        return base + scale * correction


class HARNetSVJGraph(nn.Module):
    """H80-SVJ local features with a causal GSPHAR-style graph residual.

    This is a transfer model, not an exact paper reproduction.  The local
    encoder is the released three-channel H80 HARNet-SVJ; graph propagation is
    performed across assets with a magnetic Laplacian basis supplied per
    origin.  ``graph_scale`` starts at zero so identity ablation is exactly the
    local HARNet path at initialization.
    """

    def __init__(
        self, assets: int, *, lags: tuple[int, ...] = (1, 5, 20, 80),
        hidden_size: int = 8, q: float = 0.35,
    ) -> None:
        super().__init__()
        if int(assets) < 2:
            raise ValueError("HARNetSVJGraph requires at least two assets")
        self.assets = int(assets)
        self.q = float(q)
        self.local = SameClockHARNet(lags=tuple(lags), svj=True, filters=2)
        self.feature_size = int(self.local.output.in_features)
        # Independent same-clock H80 fits can learn different causal
        # aggregation weights.  Keep one local encoder per asset so transfer
        # from those fits remains exact under the identity graph ablation.
        self.asset_locals = nn.ModuleList(
            [SameClockHARNet(lags=tuple(lags), svj=True, filters=2) for _ in range(self.assets)]
        )
        for encoder in self.asset_locals:
            encoder.load_state_dict(self.local.state_dict())
        # Formal H80 fits have one output head per asset.  Keeping those heads
        # separate is required for identity graph to be an exact local-H80
        # control; a single shared scalar head is not an equivalent baseline.
        self.asset_output = nn.Linear(self.feature_size, self.assets, bias=False)
        with torch.no_grad():
            self.asset_output.weight.copy_(self.local.output.weight.repeat(self.assets, 1))
        # The graph path uses per-asset ``asset_output`` heads.  The local
        # scalar head remains in the state dict for compatibility with the
        # standalone HARNet artifact, but is intentionally not optimized here.
        self.local.output.weight.requires_grad_(False)
        for encoder in self.asset_locals:
            encoder.output.weight.requires_grad_(False)
        self.graph_filter = nn.Sequential(
            nn.Linear(2 * self.feature_size, 2 * int(hidden_size)), nn.Tanh(),
            nn.Linear(2 * int(hidden_size), 2 * self.feature_size),
        )
        self.graph_head = nn.Linear(self.feature_size, 1, bias=False)
        self.graph_scale = nn.Parameter(torch.tensor(0.0))

    @torch.no_grad()
    def initialize_from_harnets(self, sources: Sequence[SameClockHARNet]) -> None:
        """Transfer six independent H80 heads into the joint graph model.

        The causal convolutional aggregation layers are deterministic and
        must agree across sources.  Learned output heads are copied per asset,
        then the graph branch is made an exact no-op for the identity control.
        """
        if len(sources) != self.assets:
            raise ValueError("one H80 source is required for every graph asset")
        if any(not isinstance(source, SameClockHARNet) or not source.svj for source in sources):
            raise TypeError("all graph transfer sources must be HARNet-SVJ models")
        if any(tuple(source.lags) != tuple(self.local.lags) for source in sources):
            raise ValueError("graph transfer H80 lags do not match")
        for encoder, source in zip(self.asset_locals, sources):
            encoder.load_state_dict(source.state_dict())
        self.local.load_state_dict(sources[0].state_dict())
        output = torch.cat([source.output.weight.detach().cpu() for source in sources], dim=0)
        self.asset_output.weight.copy_(output.to(self.asset_output.weight.device))
        self.local.output.weight.copy_(sources[0].output.weight.detach().to(self.local.output.weight.device))
        # Keep the graph head trainable so graph_scale receives a gradient;
        # zeroing both graph_head and graph_scale creates a dead branch.
        self.graph_scale.zero_()

    def _basis(self, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0)
        if adjacency.ndim != 3 or adjacency.shape[1:] != (self.assets, self.assets):
            raise ValueError("adjacency must be [assets, assets] or [batch, assets, assets]")
        matrix = adjacency.to(dtype=torch.float32)
        symmetric = (matrix + matrix.transpose(-1, -2)) / 2.0
        degree = symmetric.abs().sum(dim=-1).clamp_min(1e-6)
        normalized = symmetric / torch.sqrt(degree.unsqueeze(-1) * degree.unsqueeze(-2))
        theta = 2.0 * torch.pi * self.q * (matrix - matrix.transpose(-1, -2))
        magnetic = normalized.to(torch.complex64) * torch.exp(1j * theta.to(torch.complex64))
        identity = torch.eye(self.assets, device=matrix.device, dtype=torch.complex64).expand(matrix.shape[0], -1, -1)
        laplacian = (identity - magnetic)
        laplacian = (laplacian + laplacian.conj().transpose(-1, -2)) / 2.0
        _, vectors = torch.linalg.eigh(laplacian)
        return vectors, vectors.conj().transpose(-1, -2)

    def forward(self, observed: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if observed.ndim != 4 or observed.shape[2] != self.assets or observed.shape[3] != 3:
            raise ValueError("observed must be [batch, history, asset, 3]")
        batch, history, _, _ = observed.shape
        per_asset = observed.permute(0, 2, 1, 3).unbind(dim=1)
        local_features = torch.stack(
            [encoder.feature_values(values) for encoder, values in zip(self.asset_locals, per_asset)],
            dim=1,
        )
        # Per-asset F.linear preserves the standalone HARNet accumulation
        # order, which is required by the exact transfer/identity contract.
        local_prediction = torch.stack([
            F.linear(
                local_features[:, asset],
                self.asset_output.weight[asset : asset + 1],
            ).squeeze(-1)
            for asset in range(self.assets)
        ], dim=1)
        graph = adjacency
        if graph.ndim == 4:
            if graph.shape[1] != 1:
                raise ValueError("HARNetSVJGraph accepts one graph per origin")
            graph = graph[:, 0]
        if graph.ndim == 2:
            graph = graph.unsqueeze(0).expand(batch, -1, -1)
        if graph.ndim != 3:
            raise ValueError("adjacency must be [batch, asset, asset]")
        if graph.shape[0] != batch:
            raise ValueError("dynamic adjacency batch does not match observed")
        vectors, inverse = self._basis(graph)
        spectral = torch.bmm(inverse, local_features.to(torch.complex64))
        spectral_real_imag = torch.cat((spectral.real, spectral.imag), dim=-1)
        filtered = self.graph_filter(spectral_real_imag)
        half = self.feature_size
        filtered_complex = torch.complex(filtered[..., :half], filtered[..., half:])
        graph_features = torch.bmm(vectors, filtered_complex).real
        graph_prediction = self.graph_head(graph_features).squeeze(-1)
        return local_prediction + torch.tanh(self.graph_scale) * graph_prediction


class SameClockHARBaseline(nn.Module):
    """Small same-clock HAR/log-HAR baseline for the unified final protocol."""

    def __init__(self, history: int, *, log_input: bool = False) -> None:
        super().__init__()
        self.history = int(history)
        self.log_input = bool(log_input)
        self.register_buffer("input_mean", torch.zeros(1), persistent=False)
        self.register_buffer("input_std", torch.ones(1), persistent=False)
        self.lags = tuple(lag for lag in (1, 5, 22) if lag <= self.history)
        if not self.lags:
            raise ValueError("HAR baseline requires at least one lag")
        self.linear = nn.Linear(len(self.lags), 1)

    def set_input_stats(self, mean, std) -> None:
        self.input_mean = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)[0]
        self.input_std = torch.as_tensor(std, dtype=torch.float32).reshape(-1)[0]

    def forward(self, observed: torch.Tensor, adjacency=None) -> torch.Tensor:
        if observed.ndim != 3 or observed.shape[-1] != 1:
            raise ValueError("HAR baseline observed input must be [batch, history, 1]")
        values = observed[..., 0]
        if self.log_input:
            values = values * self.input_std.to(values.device) + self.input_mean.to(values.device)
        if self.log_input:
            values = torch.log(torch.clamp(values, min=1e-8))
        features = [values[:, -lag:].mean(dim=1) for lag in self.lags]
        output = self.linear(torch.stack(features, dim=-1))
        return torch.exp(output) if self.log_input else output


def _harnet(lags: tuple[int, ...], svj: bool) -> ModelBuilder:
    return lambda history, channels, assets, params: SameClockHARNet(
        lags=lags,
        svj=svj,
        filters=2 if svj else int(params.get("filters", 1)),
        use_bias=bool(params.get("use_bias", False)),
        activation=str(params.get("activation", "identity")),
    )


def _harnet_graph(history: int, channels: int, assets: int, params: Mapping[str, Any]) -> nn.Module:
    if int(channels) != 3:
        raise ValueError("harnetsvj_gsphar_h80 requires the three-channel SVJ contract")
    return HARNetSVJGraph(
        int(assets),
        lags=(1, 5, 20, 80),
        hidden_size=int(params.get("hidden_size", 8)),
        q=float(params.get("q", 0.35)),
    )


def _multitask_scale_graph(
    history: int, channels: int, assets: int, params: Mapping[str, Any],
) -> nn.Module:
    if int(history) != 80 or int(channels) != 3:
        raise ValueError("multitask scale Graph-HARNet requires history=80 and three SVJ channels")
    return MultiTaskScaleGraphHARNet(
        int(assets), lags=(1, 5, 20, 80),
        hidden_size=int(params.get("hidden_size", 8)),
        graph_operator=str(params.get("graph_operator", "gnnhar")),
        q=float(params.get("q", 0.25)),
    )


def _har(history: int, channels: int, assets: int, params: Mapping[str, Any]) -> nn.Module:
    return SameClockHARBaseline(history, log_input=False)


def _log_har(history: int, channels: int, assets: int, params: Mapping[str, Any]) -> nn.Module:
    return SameClockHARBaseline(history, log_input=True)


SAME_CLOCK_MODELS = SameClockModelRegistry()
SAME_CLOCK_MODELS.register(SameClockModelSpec("har", False, ("rv",), _har))
SAME_CLOCK_MODELS.register(SameClockModelSpec("log_har", False, ("rv",), _log_har))
SAME_CLOCK_MODELS.register(SameClockModelSpec(
    "harnet_1_5_20", False, ("svj_rv_variance",), _harnet((1, 5, 20), False),
    target_transform="variance",
))
SAME_CLOCK_MODELS.register(SameClockModelSpec(
    "harnet_1_5_20_40_80", False, ("svj_rv_variance",),
    _harnet((1, 5, 20, 40, 80), False), target_transform="variance",
), aliases=("harnet_1_5_20_80",))
SAME_CLOCK_MODELS.register(SameClockModelSpec(
    "harnetsvj_author_1_5_20", False,
    ("svj_rv_variance", "svj_negative_variance", "svj_signed_variance"),
    _harnet((1, 5, 20), True), target_transform="variance",
))
SAME_CLOCK_MODELS.register(SameClockModelSpec(
    "harnetsvj_author_1_5_20_80", False,
    ("svj_rv_variance", "svj_negative_variance", "svj_signed_variance"),
    _harnet((1, 5, 20, 80), True), target_transform="variance",
))
SAME_CLOCK_MODELS.register(SameClockModelSpec(
    "harnetsvj_author_1_5_20_40_80", False,
    ("svj_rv_variance", "svj_negative_variance", "svj_signed_variance"),
    _harnet((1, 5, 20, 40, 80), True), target_transform="variance",
))
SAME_CLOCK_MODELS.register(SameClockModelSpec(
    "harnetsvj_gsphar_h80", True,
    ("svj_rv_variance", "svj_negative_variance", "svj_signed_variance"),
    _harnet_graph, graph_kind="gsphar_dynamic", target_transform="variance",
))
SAME_CLOCK_MODELS.register(SameClockModelSpec(
    "multitask_commonality_graph_harnet_h80", True,
    ("svj_rv_variance", "svj_negative_variance", "svj_signed_variance"),
    _multitask_scale_graph, graph_kind="scale_graph", target_transform="log_variance",
))


__all__ = [
    "HARNetScaleEncoder", "HARNetSVJGraph", "INTRADAY_HARNET_MODELS",
    "MultiTaskScaleGraphHARNet", "SingleHorizonScaleGraphHARNet",
    "OvernightResidualHead", "SAME_CLOCK_MODELS",
    "SameClockHARBaseline", "SameClockHARNet",
    "SameClockHARNetDiagnostic4", "SameClockHARNetFourthChannel",
    "SameClockHARNetFiveChannelDirect", "SameClockHARNetDirectExtras",
    "SameClockHARNetRawDirectExtras",
    "SameClockHARNetLongLinearHead",
    "SameClockHARNetChannelSpecific", "SameClockHARNetShortLinearHead",
    "SameClockHARNetONDSProbabilityHead", "SameClockModelRegistry", "SameClockModelSpec",
    "harnet_svj_state_hash", "save_harnet_svj_weights", "load_harnet_svj_weights",
    "load_harnet_svj_artifact", "multitask_preprocessing_from_artifacts",
]
