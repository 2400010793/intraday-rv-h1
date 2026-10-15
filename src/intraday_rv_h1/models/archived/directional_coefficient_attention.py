"""Fresh HARNet coefficient attention with directional percentile and routed experts.

The default ``anchored`` mode preserves the historical Notebook 10 contract.
The explicit ``none`` reference mode is intentionally a different model
contract: its forward pass does not accept reference coefficients, and its
coefficient base is initialized independently with no coefficient cap,
coefficient penalty, or optimizer gradient cap.
"""
from dataclasses import asdict, dataclass
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .svj_coefficient_conditioning import SVJCoefficientHead

MICRO = (2, 3, 4, 5)
INITIAL_KNOTS = (.2, .6, .8, .9)
EXPERTS = {'none': 0, 'volatility': 3, 'time': 4, 'cross': 12,
           'uniform3': 3, 'uniform4': 4, 'uniform12': 12}


@dataclass(frozen=True)
class DirectionalAttentionConfig:
    tokens: int = 6
    encoding: str = 'legacy'
    temperature: float = .05
    learn_knots: bool = False
    knot_penalty: float = 0.
    heads: int = 4
    experts: str = 'none'
    coefficient_mode: str = 'bounded'
    reference_mode: str = 'anchored'
    initialization: str = 'zero_delta'
    time_mode: str = 'full3'
    percentile_mode: str = 'percentile'
    availability_mode: str = 'both'
    delta_penalty: float = .05

    def __post_init__(self):
        if self.tokens not in (6, 7, 8, 9) or self.heads not in (4, 8):
            raise ValueError('tokens must be 6/7/8/9 and heads must be 4/8')
        if self.reference_mode == 'anchored' and self.coefficient_mode not in ('bounded', 'free', 'free_normalized', 'eta_trust_region'):
            raise ValueError('coefficient mode must be bounded, free, free_normalized or eta_trust_region')
        if self.reference_mode not in ('anchored', 'none'):
            raise ValueError('reference mode must be anchored or none')
        if self.initialization not in ('zero_delta', 'random_delta', 'random_beta', 'train_only_ols_beta'):
            raise ValueError('unknown coefficient initialization')
        if self.reference_mode == 'anchored' and self.initialization in ('random_beta', 'train_only_ols_beta'):
            raise ValueError('random_beta and train_only_ols_beta require reference_mode=none')
        if self.time_mode not in ('full3', 'position', 'open_close', 'none'):
            raise ValueError('time mode must be full3, position, open_close or none')
        if self.percentile_mode not in ('percentile', 'none'):
            raise ValueError('percentile mode must be percentile or none')
        if self.availability_mode not in ('both', 'mask_only', 'feature_only', 'none'):
            raise ValueError('availability mode must be both, mask_only, feature_only or none')
        if not np.isfinite(self.delta_penalty):
            raise ValueError('delta penalty must be finite')
        if self.encoding not in ('legacy', 'relu', 'softplus', 'sigmoid'):
            raise ValueError('unknown percentile encoder')
        if self.experts not in EXPERTS or self.temperature <= 0 or self.knot_penalty < 0:
            raise ValueError('invalid experts, temperature or penalty')
        if self.learn_knots and self.encoding == 'legacy':
            raise ValueError('legacy encoder has no learnable knots')
        if self.experts in ('volatility', 'cross') and self.percentile_mode == 'none':
            raise ValueError('volatility expert routing requires a visible 1H percentile state')
        if self.time_mode == 'none' and self.percentile_mode == 'none' and self.experts != 'none':
            raise ValueError('constant-state ablation supports experts=none only')

    def payload(self):
        # Keep this JSON payload constructor-compatible.  Effective behavior
        # is recorded by the model/run audit rather than mixed into the config
        # that is later passed back to ``DirectionalAttentionConfig``.
        return asdict(self)


def half_day_context(meta):
    """Clock-only signed physical distance from 10:00/10:01 or 14:30/14:31.

    Infer the physical order from declared clock_position, preserving repeated
    timestamps with distinct slots. Never interpolate across the lunch break.
    """
    required = {'clock_slot_id', 'clock_position', 'datetime'}
    if not required.issubset(meta.columns):
        raise ValueError(f'missing half-day context fields: {sorted(required-set(meta))}')
    table = meta[['clock_slot_id', 'clock_position', 'datetime']].copy()
    table['minute'] = table.datetime.dt.hour * 60 + table.datetime.dt.minute
    for col in ('clock_position', 'minute'):
        if table.groupby('clock_slot_id')[col].nunique().max() != 1:
            raise ValueError(f'physical clock maps to multiple {col} values')
    unique = table.drop_duplicates('clock_slot_id').sort_values('clock_position')
    if unique.clock_position.duplicated().any():
        raise ValueError('duplicate physical positions')
    mapping = {}
    for pm, boundary in ((False, 600), (True, 870)):
        half = unique[unique.minute.ge(720).eq(pm)].reset_index(drop=True)
        left = np.flatnonzero(half.minute.to_numpy() <= boundary)
        right = np.flatnonzero(half.minute.to_numpy() > boundary)
        if not len(left) or not len(right):
            raise ValueError('context requires full physical-clock coverage on both sides of boundary')
        midpoint = (left[-1] + right[0]) / 2
        for position, key in enumerate(half.clock_slot_id):
            mapping[key] = (float(pm), position-midpoint)
    return np.asarray([mapping[k] for k in unique.clock_slot_id], dtype=np.float64)[
        unique.reset_index(drop=True).reset_index().set_index('clock_slot_id')['index'].reindex(meta.clock_slot_id).to_numpy()]


class DirectionalCoefficientAttention(nn.Module):
    def __init__(self, config: DirectionalAttentionConfig):
        super().__init__()
        self.config = config
        # Same random calls and initial weights as the original six-token head.
        self.head = SVJCoefficientHead(
            'local_attention',
            coefficient_limit=config.reference_mode == 'anchored',
        )
        if config.reference_mode == 'none':
            # This is a trainable, model-owned coefficient base. It is not a
            # copy of the HARNet reference and is never populated from it.
            self.base_coefficients = nn.Parameter(torch.zeros(16, dtype=torch.float64))
        included = (1 if config.time_mode != 'none' else 0) + (config.tokens if config.percentile_mode == 'percentile' else 0)
        effective_tokens = max(included, 1)
        # Preserve the legacy parameter layout only for the exact full3+percentile
        # six-token configuration. Other ablations get an explicit compact layout.
        if not (config.time_mode == 'full3' and config.percentile_mode == 'percentile' and config.tokens == 6):
            self.head.token_type = nn.Parameter(torch.zeros(effective_tokens, 32, dtype=torch.float64))
        if config.time_mode != 'full3' and config.time_mode != 'none':
            width = 1 if config.time_mode == 'position' else 2
            self.time_encoder = nn.Linear(width, 32).double()
            self.head.time.requires_grad_(False)
        elif config.time_mode == 'none':
            self.head.time.requires_grad_(False)
        if config.percentile_mode == 'none':
            self.head.percentile.requires_grad_(False)
        if config.encoding != 'legacy' and config.percentile_mode == 'percentile':
            self.micro_encoder = nn.Linear(6, 32).double()
            # Preserve the legacy r / q=.8 / availability projection exactly.
            with torch.no_grad():
                self.micro_encoder.weight.zero_()
                self.micro_encoder.weight[:, [0, 3, 5]] = self.head.percentile.weight
                self.micro_encoder.bias.copy_(self.head.percentile.bias)
        if config.time_mode == 'none' and config.percentile_mode == 'none':
            self.constant_delta = nn.Parameter(torch.zeros(16, dtype=torch.float64))
        if config.heads == 8:
            weights = self.head.attention.state_dict()
            with torch.random.fork_rng(devices=[]):
                self.head.attention = nn.MultiheadAttention(32, 8, batch_first=True, dropout=0.).double()
            self.head.attention.load_state_dict(weights)
        # Low threshold .05..40; ordered high thresholds .50..95 with .05 gaps.
        low_fraction = (.2-.05)/.35
        self.low_logits = nn.Parameter(torch.full((4,), float(np.log(low_fraction/(1-low_fraction))), dtype=torch.float64), requires_grad=config.learn_knots and config.percentile_mode == 'percentile')
        high_gaps = np.array([.1, .2, .1, .05]) - np.array([0., .05, .05, 0.])
        self.high_logits = nn.Parameter(torch.tensor(np.tile(np.log(high_gaps/.35), (4, 1)), dtype=torch.float64), requires_grad=config.learn_knots and config.percentile_mode == 'percentile')
        count = EXPERTS[config.experts]
        if count:
            self.expert_layers = nn.ModuleList([
                nn.Sequential(nn.Linear(32, 8), nn.GELU(), nn.Linear(8, 1)).double()
                for _ in range(count)])
            for expert in self.expert_layers:
                nn.init.zeros_(expert[-1].weight); nn.init.zeros_(expert[-1].bias)
            self.head.output.requires_grad_(False)
            vol = config.experts in ('volatility', 'cross')
            timing = config.experts in ('time', 'cross')
            fractions = np.array([(.2-.1)/.25, (.8-.65)/.25, (.05-.02)/.08])
            self.vol_logits = nn.Parameter(torch.tensor(np.log(fractions/(1-fractions)), dtype=torch.float64), requires_grad=vol)
            self.time_offsets = nn.Parameter(torch.zeros(2, dtype=torch.float64), requires_grad=timing)
        self._initialize_coefficients()

    @property
    def uses_reference(self):
        """Whether the model contract requires frozen reference coefficients."""
        return self.config.reference_mode == 'anchored'

    @property
    def effective_coefficient_mode(self):
        """Return the coefficient transform that actually reaches forward."""
        return self.config.coefficient_mode if self.uses_reference else 'free'

    @property
    def effective_delta_penalty(self):
        """Return the coefficient penalty that actually reaches the loss."""
        return self.config.delta_penalty if self.uses_reference else 0.

    def _initialize_coefficients(self):
        """Apply the declared coefficient initialization after construction."""
        mode = self.config.initialization
        if mode == 'random_beta':
            if self.uses_reference:
                raise ValueError('random_beta cannot initialize an anchored model')
            with torch.no_grad():
                self.base_coefficients.normal_(mean=0., std=.1)
            return
        if mode != 'random_delta':
            # SVJCoefficientHead and expert output projections are already
            # zero-initialized. This is the identity-preserving path.
            return
        with torch.no_grad():
            if EXPERTS[self.config.experts]:
                for expert in self.expert_layers:
                    expert[-1].weight.normal_(mean=0., std=.02)
                    expert[-1].bias.normal_(mean=0., std=.02)
            else:
                self.head.output.weight.normal_(mean=0., std=.02)
                self.head.output.bias.normal_(mean=0., std=.02)

    def set_base_coefficients(self, coefficients):
        """Set a no-reference base, typically from Train-only OLS."""
        if self.uses_reference:
            raise ValueError('anchored models do not own base coefficients')
        beta = torch.as_tensor(coefficients, dtype=self.base_coefficients.dtype, device=self.base_coefficients.device)
        if beta.shape != (16,) or not torch.isfinite(beta).all():
            raise ValueError('base coefficients must be finite with shape (16,)')
        with torch.no_grad():
            self.base_coefficients.copy_(beta)

    def state_hash(self):
        """Hash all parameters and buffers for initialization provenance."""
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            digest.update(name.encode('utf-8'))
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def knots(self):
        low = .05 + .35 * self.low_logits.sigmoid()
        minimum = self.high_logits.new_tensor([0., .05, .05, 0.])
        high = .5 + (minimum + .35*self.high_logits.softmax(-1)).cumsum(-1)[:, :3]
        return torch.cat((low[:, None], high), dim=-1)

    def responses(self, q):
        """Input B×4 percentile values; output B×4×4 directional responses."""
        k = self.knots()[None]
        q = q[:, :, None]
        direction = q.new_tensor([-1., 1., 1., 1.])
        x = direction*(q-k)
        lo = torch.where(direction > 0, -k, k-1)
        hi = torch.where(direction > 0, 1-k, k)
        if self.config.encoding == 'relu':
            fn = torch.relu
        elif self.config.encoding == 'softplus':
            tau = self.config.temperature
            fn = lambda value: tau*F.softplus(value/tau)
        elif self.config.encoding == 'sigmoid':
            fn = lambda value: torch.sigmoid(value/self.config.temperature)
        else:
            raise ValueError('legacy encoding does not use directional responses')
        return (fn(x)-fn(lo))/(fn(hi)-fn(lo))

    def gate_parameters(self):
        if not EXPERTS[self.config.experts]:
            return {}
        v = self.vol_logits.sigmoid()
        return dict(volatility_thresholds=torch.stack((.1+.25*v[0], .65+.25*v[1])),
                    volatility_temperature=.02+.08*v[2], time_offsets=10*self.time_offsets.tanh())

    def gates(self, state, context):
        count = EXPERTS[self.config.experts]
        if not count:
            return state.new_ones(len(state), 1)
        if self.config.experts.startswith('uniform'):
            return state.new_full((len(state), count), 1/count)
        parameters = self.gate_parameters()
        q = (state[:, 2, 0]+1)/2
        a,b = parameters['volatility_thresholds']
        tau = parameters['volatility_temperature']
        sa, sb = torch.sigmoid((q-a)/tau), torch.sigmoid((q-b)/tau)
        vol = torch.stack((1-sa, (sa-sb).clamp_min(0), sb), dim=-1)
        missing = state[:, 2, 2] <= 0
        vol = torch.where(missing[:,None], vol.new_tensor([0.,1.,0.]), vol)
        if context.shape != (len(state), 2) or not torch.isfinite(context).all():
            raise ValueError('expected finite [is_pm, physical_boundary_distance]')
        pm = context[:, 0]
        if not torch.all((pm==0)|(pm==1)):
            raise ValueError('half-day indicator must be binary')
        offsets = parameters['time_offsets']
        opening = torch.sigmoid((offsets[0]-context[:,1])/5)
        afternoon = torch.sigmoid((offsets[1]-context[:,1])/5)
        timing = torch.stack(((1-pm)*opening, (1-pm)*(1-opening), pm*afternoon, pm*(1-afternoon)), -1)
        if self.config.experts == 'volatility': return vol
        if self.config.experts == 'time': return timing
        return (vol[:,:,None]*timing[:,None,:]).reshape(len(state),12)

    def state_tokens(self, old_state, state):
        if old_state.shape != (len(state),21) or state.shape[1:] != (self.config.tokens,3):
            raise ValueError('invalid original/token state layout')
        if not torch.isfinite(old_state).all() or not torch.isfinite(state).all():
            raise FloatingPointError('nonfinite state')
        pieces = []
        masks = []
        if self.config.time_mode != 'none':
            if self.config.time_mode == 'full3':
                time = self.head.time(old_state[:, :3])
            elif self.config.time_mode == 'position':
                time = self.time_encoder(old_state[:, 2:3])
            else:
                time = self.time_encoder(old_state[:, :2])
            pieces.append(time[:, None, :])
            masks.append(torch.zeros(len(state), 1, dtype=torch.bool, device=state.device))
        if self.config.percentile_mode == 'percentile':
            mask = state[:, :, 2] > 0
            values = torch.where(mask[:, :, None], state[:, :, :2], torch.zeros_like(state[:, :, :2]))
            if self.config.availability_mode == 'both':
                # Preserve the original contract: unavailable tokens are zeroed
                # and removed from the attention key set.
                safe = torch.cat((values, state[:, :, 2:3]), dim=-1)
                padding = ~mask
            elif self.config.availability_mode == 'mask_only':
                # Test whether the explicit availability scalar is redundant
                # once the key-padding mask already identifies missing tokens.
                safe = torch.cat((values, torch.zeros_like(state[:, :, 2:3])), dim=-1)
                padding = ~mask
            elif self.config.availability_mode == 'feature_only':
                # Keep the availability scalar, but expose every token to
                # attention. Invalid numeric values remain zeroed.
                safe = torch.cat((values, state[:, :, 2:3]), dim=-1)
                padding = torch.zeros_like(mask)
            else:
                # Remove both uses of the flag. A missing token is represented
                # only by its zeroed numeric inputs and remains visible.
                safe = torch.cat((values, torch.zeros_like(state[:, :, 2:3])), dim=-1)
                padding = torch.zeros_like(mask)
            embedded = self.head.percentile(safe)
            if self.config.encoding != 'legacy':
                micro = safe[:, MICRO]
                q = (micro[:, :, 0] + 1) / 2
                expanded = torch.cat((micro[:, :, :1], self.responses(q), micro[:, :, 2:3]), -1)
                # Missing numeric inputs are zero placeholders, not q=0.5.
                # Keep the availability scalar only in modes that expose it.
                numeric = expanded[:, :, :-1] * mask[:, MICRO, None]
                expanded = torch.cat((numeric, expanded[:, :, -1:]), dim=-1)
                replacement = self.micro_encoder(expanded)
                embedded = torch.cat((embedded[:, :2], replacement, embedded[:, 6:]), dim=1)
            pieces.append(embedded)
            masks.append(padding)
        if not pieces:
            return None, None
        tokens = torch.cat(pieces, dim=1) + self.head.token_type
        padding = torch.cat(masks, dim=1)
        return tokens, padding

    def raw_delta(self, old_state, state, context):
        tokens, padding = self.state_tokens(old_state, state)
        if tokens is None:
            return self.constant_delta[None, :].expand(len(state), -1)
        safe = torch.where(state[:, :, 2:3] > 0, state, torch.zeros_like(state))
        queries = self.head.norm_query(self.head.queries)[None].expand(len(state),-1,-1)
        kv = self.head.norm_tokens(tokens)
        attended,_ = self.head.attention(queries,kv,kv,key_padding_mask=padding,need_weights=False)
        if not EXPERTS[self.config.experts]:
            raw = self.head.output(attended).squeeze(-1)
        else:
            experts = torch.stack([layer(attended).squeeze(-1) for layer in self.expert_layers], -1)
            raw = (experts*self.gates(safe,context)[:,None,:]).sum(-1)
        return raw

    def _bounded_delta(self, raw):
        if not self.uses_reference:
            raise RuntimeError('coefficient bounds are unavailable in reference_mode=none')
        return self.head.limit * torch.tanh(raw)

    def delta(self, old_state, state, context):
        raw = self.raw_delta(old_state, state, context)
        if self.effective_coefficient_mode == 'bounded':
            return self._bounded_delta(raw)
        return raw

    def coefficient_regularization(self, features, delta):
        if not self.uses_reference:
            return delta.new_zeros(())
        weight = self.effective_delta_penalty
        if weight == 0:
            return delta.new_zeros(())
        if self.effective_coefficient_mode == 'free_normalized':
            scaled = delta / self.head.limit
            return weight * scaled.square().sum(dim=1).mean()
        if self.effective_coefficient_mode == 'eta_trust_region':
            eta_delta = (features * delta).sum(dim=1)
            return weight * eta_delta.square().mean()
        return weight * delta.square().sum(dim=1).mean()

    def regularization(self):
        # Independent coefficient models have no parameter regularization. The
        # ordered knot parameterization itself remains a validity contract for
        # the percentile encoder, but its optional pullback penalty is off.
        if not self.uses_reference:
            return self.low_logits.new_zeros(())
        return self.config.knot_penalty*(self.knots()-self.low_logits.new_tensor(INITIAL_KNOTS)).square().sum()

    def forward(self, features, *inputs):
        """Forecast with either the anchored or independent input contract.

        Anchored: ``(features, reference, old_state, state, context)``.
        Independent: ``(features, old_state, state, context)``.
        Passing the reference tensor to an independent model is rejected so a
        supposedly free experiment cannot accidentally consume the old anchor.
        """
        if features.ndim != 2 or features.shape[1] != 16:
            raise ValueError('expected 16 named features')
        if self.uses_reference:
            if len(inputs) != 4:
                raise TypeError('anchored model requires reference, old_state, state and context')
            reference, old_state, state, context = inputs
            if reference.shape != features.shape:
                raise ValueError('reference coefficient shape mismatch')
            base = reference
        else:
            if len(inputs) != 3:
                raise TypeError('reference_mode=none requires old_state, state and context; reference is forbidden')
            old_state, state, context = inputs
            base = self.base_coefficients[None, :].expand(len(features), -1)
        if not torch.isfinite(features).all() or not torch.isfinite(base).all():
            raise FloatingPointError('non-finite features or coefficient base')
        raw = self.raw_delta(old_state, state, context)
        if self.uses_reference:
            delta = self._bounded_delta(raw) if self.effective_coefficient_mode == 'bounded' else raw
            coefficients = base+delta
        else:
            # No-reference mode is deliberately unconstrained.  The coefficient
            # values are not reference-relative and receive no cap, penalty,
            # normalization, or trust-region transform.
            coefficients = base+raw
        if not torch.isfinite(coefficients).all():
            raise FloatingPointError('non-finite coefficient output')
        prediction = torch.exp((features*coefficients).sum(-1)/2)
        if not torch.isfinite(prediction).all() or (prediction<=0).any():
            raise FloatingPointError('invalid coefficient forecast')
        return prediction, coefficients


def build_directional_attention(config, seed):
    if isinstance(config,dict):config=DirectionalAttentionConfig(**config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return DirectionalCoefficientAttention(config)
