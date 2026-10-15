"""Single-factor ablations of the archived HARNet return-token architecture."""
from __future__ import annotations

import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

from .coefficient_scale import bounded_coefficient_delta
from .local_token_attention import FEATURE_NAMES
from .return_percentile_attention import ReturnPercentileStateAttention

ABLATIONS=('single_decoder','split_intercept','no_intercept','frozen_convolution',
    'no_state_mix','cap_half_matched','cap_double_matched','L_double',
    'learned_L','learned_cap_matched')
LEARNED_SCALES=('learned_L','learned_cap_matched')
LINEAR_ABLATIONS=('linear_L','linear_unscaled','softsign_L',
    'linear_decoder_tanh','linear_decoder_L','linear_decoder_unscaled')
ZERO_L_ABLATION='zero_L_random_f'
EXPERT_COUNT_ABLATIONS=('split_three_linear_tanh','split_single_mlp_tanh')
LINEAR_SHARING_ABLATIONS=('split_linear_decoder_tanh','coefficient_linear_decoder_tanh')
GROUPED_LINEAR_ABLATIONS=('split_linear_decoder_L','three_group_linear_decoder_tanh',
    'three_group_linear_decoder_L','four_group_linear_decoder_tanh','four_group_linear_decoder_L')
ALL_LINEAR_SHARING_ABLATIONS=LINEAR_SHARING_ABLATIONS+GROUPED_LINEAR_ABLATIONS
ALL_ABLATIONS=(*ABLATIONS,*LINEAR_ABLATIONS,*ALL_LINEAR_SHARING_ABLATIONS,*EXPERT_COUNT_ABLATIONS,ZERO_L_ABLATION)
LINEAR_SCALES=('linear_L','linear_unscaled','linear_decoder_L','linear_decoder_unscaled',
    'split_linear_decoder_L','three_group_linear_decoder_L','four_group_linear_decoder_L')
LINEAR_DECODERS=('linear_decoder_tanh','linear_decoder_L','linear_decoder_unscaled')


def linear_head_groups(mode):
    """Fixed financial partitions of the fifteen live coefficient positions."""
    if mode in ('split_linear_decoder_tanh','split_linear_decoder_L'):
        return {'intercept':(0,), 'slopes':tuple(range(1,15))}
    if mode.startswith('three_group_linear_decoder_'):
        return {'intercept':(0,), 'har_svj':tuple(range(1,10)), 'intraday_policy':tuple(range(10,15))}
    if mode.startswith('four_group_linear_decoder_'):
        return {'intercept':(0,), 'har_rv':(1,3,5,7), 'svj':(2,4,6,8,9),
            'intraday_policy':tuple(range(10,15))}
    if mode=='coefficient_linear_decoder_tanh':return {str(j):(j,) for j in range(15)}
    if mode in LINEAR_DECODERS:return {'shared':tuple(range(15))}
    raise ValueError(f'no linear head layout:{mode}')


def decoder_head_groups(attention):
    """Map actual module names to the exposures whose gradients they receive."""
    mode=attention.ablation
    if mode=='split_three_linear_tanh':
        return {f'{bank}.{j}':columns for bank,columns in
                (('intercept_linear_experts',(0,)),('slopes_linear_experts',tuple(range(1,15)))) for j in range(3)}
    if mode=='split_single_mlp_tanh':return {'intercept_single':(0,), 'expert_layers.0':tuple(range(1,15))}
    if mode in ('split_linear_decoder_tanh','split_linear_decoder_L'):
        return {'intercept_linear':(0,), 'slopes_linear':tuple(range(1,15))}
    if mode in GROUPED_LINEAR_ABLATIONS:
        return {f'group_linears.{name}':columns for name,columns in linear_head_groups(mode).items()}
    if mode=='coefficient_linear_decoder_tanh':
        return {f'coefficient_linears.{j}':(int(j),) for j in attention.coefficient_linears}
    raise ValueError(f'independent decoder gradient audit:{mode}')


class StructureStateAttention(ReturnPercentileStateAttention):
    def __init__(self,*args,ablation,**kwargs):
        super().__init__(*args,**kwargs)
        if ablation not in ('control',*ALL_ABLATIONS):raise ValueError(f'unknown structural ablation:{ablation}')
        self.ablation=ablation
        if ablation in LEARNED_SCALES:
            self.L_logits=nn.Parameter(torch.zeros_like(self.asset_limits))
        if ablation==ZERO_L_ABLATION:
            # Signed coefficient units, exactly zero, with no softplus or ceiling.
            self.L_values=nn.Parameter(torch.zeros_like(self.asset_limits))

    def effective_cap(self,asset_ids):
        original=self.asset_limits[asset_ids]
        if self.ablation==ZERO_L_ABLATION:
            values=self.L_values[asset_ids]
            if not torch.isfinite(values).all():raise FloatingPointError('nonfinite zero-start learned L')
            return values.abs()
        if self.ablation in LEARNED_SCALES:
            # Positive and initially exactly L0; no clipping or artificial ceiling.
            cap=original*(F.softplus(self.L_logits[asset_ids])/math.log(2.))
        else:
            factor=.5 if self.ablation=='cap_half_matched' else 2. if self.ablation in ('cap_double_matched','L_double') else 1.
            cap=factor*original
        if not torch.isfinite(cap).all() or (cap<=0).any():raise FloatingPointError('nonfinite/nonpositive learned coefficient cap')
        return cap

    def initial_sensitivity(self,asset_ids):
        if self.ablation==ZERO_L_ABLATION:return torch.zeros_like(self.asset_limits[asset_ids])
        if self.ablation=='learned_L':return self.effective_cap(asset_ids)
        if self.ablation in ('linear_unscaled','linear_decoder_unscaled'):return torch.ones_like(self.asset_limits[asset_ids])
        return self.asset_limits[asset_ids]*(2 if self.ablation=='L_double' else 1)

    def configure_decoder(self):
        # Copy after loading the archived initialization, not after training.
        if self.ablation in LINEAR_DECODERS:
            with torch.random.fork_rng(devices=[]):
                self.linear_decoder=nn.Linear(96,1,dtype=torch.float64)
            nn.init.zeros_(self.linear_decoder.weight);nn.init.zeros_(self.linear_decoder.bias)
            self.expert_layers.requires_grad_(False)
        if self.ablation in ALL_LINEAR_SHARING_ABLATIONS:
            with torch.random.fork_rng(devices=[]):
                if self.ablation in ('split_linear_decoder_tanh','split_linear_decoder_L'):
                    self.intercept_linear=nn.Linear(96,1,dtype=torch.float64)
                    self.slopes_linear=nn.Linear(96,1,dtype=torch.float64)
                    layers=(self.intercept_linear,self.slopes_linear)
                elif self.ablation=='coefficient_linear_decoder_tanh':
                    # No unused reconstructed head: exactly15 active connections.
                    self.coefficient_linears=nn.ModuleDict({str(j):nn.Linear(96,1,dtype=torch.float64)
                        for j in torch.nonzero(self.prefix_active).flatten().tolist()})
                    layers=self.coefficient_linears.values()
                else:
                    self.group_linears=nn.ModuleDict({name:nn.Linear(96,1,dtype=torch.float64)
                        for name in linear_head_groups(self.ablation)})
                    layers=self.group_linears.values()
            for layer in layers:
                nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
            self.expert_layers.requires_grad_(False)
        if self.ablation=='split_intercept':self.intercept_experts=copy.deepcopy(self.expert_layers)
        if self.ablation=='split_three_linear_tanh':
            with torch.random.fork_rng(devices=[]):
                self.intercept_linear_experts=nn.ModuleList([nn.Linear(96,1,dtype=torch.float64) for _ in range(3)])
                self.slopes_linear_experts=nn.ModuleList([nn.Linear(96,1,dtype=torch.float64) for _ in range(3)])
            for layer in (*self.intercept_linear_experts,*self.slopes_linear_experts):
                nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
            self.expert_layers.requires_grad_(False)
        if self.ablation=='split_single_mlp_tanh':
            self.intercept_single=copy.deepcopy(self.expert_layers[0])
            for layer in self.expert_layers[1:]:layer.requires_grad_(False)
        if self.ablation=='single_decoder':
            for layer in self.expert_layers[1:]:layer.requires_grad_(False)
        if self.ablation=='no_state_mix':
            self.depth_norm.requires_grad_(False);self.depth_block.requires_grad_(False)

    def mix_state(self,tokens,padding):
        if self.ablation!='no_state_mix':return super().mix_state(tokens,padding)
        return torch.where(padding[...,None],torch.zeros_like(tokens),tokens)

    def decode_coefficients(self,decoder,asset_ids):
        if self.ablation=='split_three_linear_tanh':
            intercept=torch.stack([layer(decoder[:,:1]).squeeze(-1) for layer in self.intercept_linear_experts],-1).mean(-1)
            slopes=torch.stack([layer(decoder[:,1:]).squeeze(-1) for layer in self.slopes_linear_experts],-1).mean(-1)
            return torch.cat((intercept,slopes),1)
        if self.ablation=='split_single_mlp_tanh':
            return torch.cat((self.intercept_single(decoder[:,:1]).squeeze(-1),self.expert_layers[0](decoder[:,1:]).squeeze(-1)),1)
        if self.ablation in LINEAR_DECODERS:return self.linear_decoder(decoder).squeeze(-1)
        if self.ablation in ('split_linear_decoder_tanh','split_linear_decoder_L'):
            return torch.cat((self.intercept_linear(decoder[:,:1]).squeeze(-1),
                self.slopes_linear(decoder[:,1:]).squeeze(-1)),1)
        if self.ablation in GROUPED_LINEAR_ABLATIONS:
            # One batched linear call per bank; no Python loop over observations.
            columns=[decoder.new_zeros(len(decoder)) for _ in range(16)]
            for name,indices in linear_head_groups(self.ablation).items():
                values=self.group_linears[name](decoder[:,indices]).squeeze(-1)
                for index,value in zip(indices,values.unbind(1)):columns[index]=value
            return torch.stack(columns,1)
        if self.ablation=='coefficient_linear_decoder_tanh':
            return torch.cat([self.coefficient_linears[str(j)](decoder[:,j:j+1]).squeeze(-1)
                if str(j) in self.coefficient_linears else decoder.new_zeros((len(decoder),1))
                for j in range(16)],1)
        if self.ablation=='single_decoder':return self.expert_layers[0](decoder).squeeze(-1)
        if self.ablation=='split_intercept':
            intercept=torch.stack([layer(decoder[:,:1]).squeeze(-1) for layer in self.intercept_experts],-1).mean(-1)
            slopes=torch.stack([layer(decoder[:,1:]).squeeze(-1) for layer in self.expert_layers],-1).mean(-1)
            return torch.cat((intercept,slopes),1)
        return super().decode_coefficients(decoder,asset_ids)

    def coefficient_delta(self,raw,asset_ids):
        scale=self.asset_limits[asset_ids]
        if self.ablation=='cap_half_matched':delta=bounded_coefficient_delta(raw,.5*scale,scale)
        elif self.ablation=='cap_double_matched':delta=bounded_coefficient_delta(raw,2*scale,scale)
        elif self.ablation=='L_double':delta=2*scale*torch.tanh(raw)
        elif self.ablation=='learned_L':delta=self.effective_cap(asset_ids)*torch.tanh(raw)
        elif self.ablation=='learned_cap_matched':delta=bounded_coefficient_delta(raw,self.effective_cap(asset_ids),scale)
        elif self.ablation==ZERO_L_ABLATION:
            self.effective_cap(asset_ids)  # Reject nonfinite amplitudes before prediction.
            delta=self.L_values[asset_ids]*torch.tanh(raw)
        elif self.ablation in ('linear_L','linear_decoder_L','split_linear_decoder_L',
                'three_group_linear_decoder_L','four_group_linear_decoder_L'):delta=scale*raw
        elif self.ablation in ('linear_unscaled','linear_decoder_unscaled'):delta=raw
        elif self.ablation=='softsign_L':delta=scale*F.softsign(raw)
        else:delta=super().coefficient_delta(raw,asset_ids)
        if self.ablation=='no_intercept':delta=torch.cat((torch.zeros_like(delta[:,:1]),delta[:,1:]),1)
        return delta


def apply_ablation(model,ablation,*,output_seed=0):
    """Factory extension; no forward monkey patch or transplant of trained heads."""
    if ablation not in ('control',*ALL_ABLATIONS):raise ValueError(f'unknown structural ablation:{ablation}')
    old=model.attention
    with torch.random.fork_rng(devices=[]):
        replacement=StructureStateAttention(old.config,old.asset_limits,old.identity_descriptors,old.variant,
            numeric_mode=old.numeric_mode,split_calendar=old.split_calendar,
            topology='state_cross',regime_mode=None,
            return_column=old.return_column,return_state_count=old.return_state_count,
            return_state_width=old.return_state_width,return_input=old.return_input,ablation=ablation)
    state=dict(old.state_dict())
    if ablation in LEARNED_SCALES:state['L_logits']=replacement.L_logits.detach().clone()
    if ablation==ZERO_L_ABLATION:state['L_values']=replacement.L_values.detach().clone()
    replacement.load_state_dict(state,strict=True);replacement.configure_decoder()
    if ablation==ZERO_L_ABLATION:
        # Only the three final 8->1 projections change. All upstream weights and
        # the caller's RNG state remain identical to the original initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(output_seed))
            for expert in replacement.expert_layers:expert[-1].reset_parameters()
    # Preserve all previously frozen parameters, including the original dead head.
    old_trainable={n:p.requires_grad for n,p in old.named_parameters()}
    for name,p in replacement.named_parameters():
        if not old_trainable.get(name,True):p.requires_grad_(False)
    model.attention=replacement
    if ablation=='frozen_convolution':
        for local in model.reference.locals:local.requires_grad_(False)
    return model


def coefficient_diagnostics(model,output,asset_ids,epoch):
    _,beta,h,reference=output;delta=beta-reference;limits=model.attention.asset_limits[asset_ids]
    caps=model.attention.effective_cap(asset_ids);sensitivity=model.attention.initial_sensitivity(asset_ids)
    records=[]
    for asset in torch.unique(asset_ids).tolist():
        take=asset_ids.eq(asset)
        for j,name in enumerate(FEATURE_NAMES):
            if not model.attention.prefix_active[j]:continue
            d=delta[take,j];limit=limits[take,j];cap=caps[take,j]
            if (cap==0).any() and torch.count_nonzero(d):raise AssertionError('nonzero delta at zero L')
            ratio=d.abs()/torch.where(cap>0,cap,torch.ones_like(cap))
            bounded=model.attention.ablation not in LINEAR_SCALES
            records.append(dict(epoch=epoch,asset_id=asset,feature=name,column=j,sample_rows=int(take.sum()),
                original_L=float(limit[0]),cap=float(cap[0]) if bounded else None,
                cap_over_original_L=float(cap[0]/limit[0]) if bounded else None,
                signed_learned_L=float(model.attention.L_values[asset,j]) if model.attention.ablation==ZERO_L_ABLATION else None,
                bounded=bounded,
                initial_sensitivity=float(sensitivity[take,j][0]),
                mean_reference=float(reference[take,j].mean()),mean_coefficient=float(beta[take,j].mean()),
                negative_fraction=float((beta[take,j]<0).double().mean()),mean_abs_delta=float(d.abs().mean()),
                max_abs_delta=float(d.abs().max()),max_abs_delta_over_original_L=float((d.abs()/limit).max()),
                max_abs_delta_over_cap=float(ratio.max()) if bounded else None,
                fraction_over_90pct_cap=float((ratio>.9).double().mean()) if bounded else None,
                mean_abs_eta_contribution=float((h[take,j]*d).abs().mean()),
                q95_abs_eta_contribution=float(torch.quantile((h[take,j]*d).abs(),.95))))
    return records


def zero_L_gradient_record(model,step):
    """Audit the actual loss graph, including the initially blocked branch."""
    attention=model.attention
    if attention.ablation!=ZERO_L_ABLATION:raise ValueError('zero-L gradient audit requires zero_L_random_f')
    gradient=attention.L_values.grad
    internal=[p.grad for n,p in attention.named_parameters() if n!='L_values' and p.requires_grad]
    maximum=max((float(g.abs().max()) for g in internal if g is not None),default=0.)
    record=dict(step=step,L_max_abs=float(attention.L_values.detach().abs().max()),
        L_gradient_max_abs=0. if gradient is None else float(gradient.abs().max()),
        internal_gradient_max_abs=maximum,
        return_encoder_gradient_max_abs=0. if attention.return_encoder[0].weight.grad is None else float(attention.return_encoder[0].weight.grad.abs().max()))
    if not all(math.isfinite(v) for k,v in record.items() if k!='step'):raise FloatingPointError(f'nonfinite zero-L gradient:{record}')
    if step==0 and (record['L_max_abs']!=0 or record['L_gradient_max_abs']<=0 or maximum!=0):
        raise AssertionError(f'zero-L first-step gradient contract:{record}')
    if step==1 and (record['L_max_abs']<=0 or maximum<=0):
        raise AssertionError(f'zero-L second-step gradient contract:{record}')
    return record


def decoder_parameter_audit(model):
    """Count live decoder weights, separately from common attention parameters."""
    a=model.attention;names=('expert_layers','linear_decoder','intercept_experts',
        'intercept_linear','slopes_linear','coefficient_linears','group_linears',
        'intercept_linear_experts','slopes_linear_experts','intercept_single')
    parameters={n:p for n,p in a.named_parameters() if p.requires_grad and n.split('.')[0] in names}
    count=sum(p.numel() for p in parameters.values());mode=getattr(a,'ablation','control')
    if mode=='split_three_linear_tanh':expected=582
    elif mode=='split_single_mlp_tanh':expected=1570
    elif mode in ALL_LINEAR_SHARING_ABLATIONS:expected=len(linear_head_groups(mode))*97
    elif mode in LINEAR_DECODERS:expected=97
    elif mode=='split_intercept':expected=4710
    elif mode=='single_decoder':expected=785
    else:expected=2355
    if count!=expected:raise AssertionError(f'decoder parameter count:{mode}:{count}!={expected}')
    return dict(mode=mode,active_coefficients=int(a.prefix_active.sum()),
        decoder_trainable_parameters=count,attention_trainable_parameters=sum(p.numel() for p in a.parameters() if p.requires_grad),
        model_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        decoder_parameter_shapes={n:list(p.shape) for n,p in parameters.items()})


def record_decoder_gradients(model, records):
    """Keep maxima over actual optimizer steps, including masked final batches."""
    for name,p in model.attention.named_parameters():
        if p.requires_grad and name.split('.')[0] in ('intercept_linear','slopes_linear','coefficient_linears','group_linears',
                'intercept_linear_experts','slopes_linear_experts','intercept_single','expert_layers'):
            value=0. if p.grad is None else float(p.grad.abs().max())
            if not math.isfinite(value):raise FloatingPointError(f'nonfinite decoder gradient:{name}')
            records[name]=max(records.get(name,0.),value)


def decoder_gradient_audit(model, records, features):
    """Require a gradient for exposed heads; disclose inactive smoke channels."""
    a=model.attention
    groups={name:features[:,indices] for name,indices in decoder_head_groups(a).items()}
    result={}
    for prefix,h in groups.items():
        exposed=bool(torch.count_nonzero(h));maximum=max((v for k,v in records.items() if k.startswith(prefix+'.')),default=0.)
        if exposed and maximum<=0:raise AssertionError(f'exposed independent decoder missing gradient:{prefix}')
        result[prefix]=dict(probe_exposed=exposed,max_abs_gradient=maximum,
            status='gradient_observed' if maximum>0 else 'no_exposure_in_probe')
    return result


def connectivity_audit(model,args):
    """Observe actual masks and module shapes without changing weights or RNG."""
    a=model.attention;observed={}
    def capture(label):
        def hook(module,inputs,kwargs):
            mask=kwargs['key_padding_mask'];visible=(~mask).sum(1)
            observed[label]=dict(queries=int(inputs[0].shape[1]),tokens=int(inputs[1].shape[1]),
                heads=module.num_heads,width=module.embed_dim,
                visible_min=int(visible.min()),visible_max=int(visible.max()),
                attention_trainable_parameters=sum(p.numel() for p in module.parameters() if p.requires_grad))
        return hook
    hooks=[a.depth_block.register_forward_pre_hook(capture('state_self_attention'),with_kwargs=True),
           a.head.attention.register_forward_pre_hook(capture('coefficient_cross_attention'),with_kwargs=True)]
    try:
        with torch.no_grad():model(*args)
    finally:
        for hook in hooks:hook.remove()
    cross=observed['coefficient_cross_attention'];active=int(a.prefix_active.sum());tokens=cross['tokens'];heads=cross['heads']
    return {**decoder_parameter_audit(model),'observed_modules':observed,'sampled_rows':len(args[0]),
        'allocated_tokens':tokens,'active_coefficients':active,'allocated_coefficients':cross['queries'],'decoder_input_width':96,
        'per_coefficient_attention_scores_allocated':heads*tokens,
        'per_coefficient_attention_scores_visible_sample_range':[heads*cross['visible_min'],heads*cross['visible_max']],
        'active_coefficient_attention_scores_allocated':active*heads*tokens,
        'state_attention_scores_allocated':heads*tokens*tokens,
        'attention_scores_are_trainable_parameters':False,
        'original_shared_mlp_weighted_connections_per_coefficient':3*(96*8+8),
        'original_shared_mlp_unique_decoder_parameters':2355,
        'own_linear_weighted_connections_per_coefficient':96,
        'note':'visible range is a sampled origin mask diagnostic, not full-panel coverage'}
