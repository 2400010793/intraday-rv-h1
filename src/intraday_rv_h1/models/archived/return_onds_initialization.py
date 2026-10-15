"""Signed return states condition only the frozen ONDS coefficient at epoch0."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

WINDOWS={'overnight':(0,), 'opening15':(1,), 'overnight_opening15':(0,1)}
MODES=('rank_linear','rank_relu2','raw_linear','raw_relu2')
ARMS={f'{window}__{mode}':dict(windows=indices,mode=mode,column=10,
    dimension=len(indices)*(2 if mode.endswith('relu2') else 1))
    for window,indices in WINDOWS.items() for mode in MODES}


def zero_rank(ordered):
    ordered=np.asarray(ordered,float)
    if ordered.ndim!=1 or not len(ordered) or not np.isfinite(ordered).all() or (np.diff(ordered)<0).any():
        raise ValueError('invalid fit-only sorted return registry')
    left=np.searchsorted(ordered,0.,side='left');right=np.searchsorted(ordered,0.,side='right')
    return (left+.5*(right-left)+.5)/(len(ordered)+1)


def basis(state,zero_q,spec):
    """State [N,2,3]=[overnight,opening15] x [q,availability,raw]."""
    if state.ndim!=3 or state.shape[1:]!=(2,3) or tuple(spec['windows']) not in WINDOWS.values():
        raise ValueError('wrong ONDS return state/spec schema')
    return window_basis(state,zero_q,spec)


def window_basis(state,zero_q,spec):
    """General signed window layout [N,W,3], preserving the legacy formula."""
    tensor=isinstance(state,torch.Tensor);s=state if tensor else np.asarray(state,float)
    windows=tuple(spec['windows'])
    if (s.ndim!=3 or s.shape[2]!=3 or spec['mode'] not in MODES or not windows
            or len(set(windows))!=len(windows) or min(windows)<0 or max(windows)>=s.shape[1]
            or spec['dimension']!=len(windows)*(2 if spec['mode'].endswith('relu2') else 1)):
        raise ValueError('wrong ONDS return state/spec schema')
    if spec['column']!=10:raise ValueError('return initialization changes ONDS only')
    z=torch.as_tensor(zero_q,dtype=s.dtype,device=s.device) if tensor else np.asarray(zero_q,float)
    if z.shape==(s.shape[1],):z=z[None,:]
    if z.shape not in ((1,s.shape[1]),(len(s),s.shape[1])):raise ValueError('unaligned zero-return percentile')
    terms=[]
    for j in spec['windows']:
        available=s[:,j,1]==1
        valid=(s[:,j,1]==0)|available
        if not valid.all():raise ValueError('nonbinary return availability')
        if spec['mode'].startswith('rank'):
            v=s[:,j,0];q0=z[:,j]
            finite=torch.isfinite if tensor else np.isfinite
            if not finite(q0).all() or (q0<=0).any() or (q0>=1).any():raise ValueError('invalid zero-return rank')
            if not finite(v[available]).all() or (v[available]<0).any() or (v[available]>1).any():raise ValueError('invalid visible return percentile')
            v=v-q0
        else:v=s[:,j,2]
        v=torch.where(available,v,0.) if tensor else np.where(available,v,0.)
        finite=torch.isfinite if tensor else np.isfinite
        if not finite(v).all():raise FloatingPointError('nonfinite visible return response')
        if spec['mode'].endswith('relu2'):
            relu=torch.relu if tensor else lambda x:np.maximum(x,0.)
            terms.extend((relu(v),relu(-v)))
        else:terms.append(v)
    return torch.stack(terms,1) if tensor else np.column_stack(terms)


def design(h,state,zero_q,spec):
    h=np.asarray(h,float)
    if h.shape!=(len(state),16) or not np.isfinite(h[:,10]).all():raise ValueError('invalid ONDS input column')
    return window_basis(state,zero_q,spec)*h[:,10,None]


def sufficient_statistics(h,state,zero_q,eta,y,spec,*,chunk_size=8192):
    y=np.asarray(y,float);eta=np.asarray(eta,float)
    if y.shape!=(len(h),) or eta.shape!=y.shape or not len(y) or not np.isfinite(eta).all() or not np.isfinite(y).all() or (y<=0).any():raise ValueError('invalid ONDS fitting rows/labels')
    gram=np.zeros((spec['dimension'],spec['dimension']));rhs=np.zeros(spec['dimension'])
    for low in range(0,len(y),chunk_size):
        sl=slice(low,low+chunk_size);q0=zero_q if np.ndim(zero_q)==1 else zero_q[sl]
        x=design(h[sl],state[sl],q0,spec);gram+=x.T@x;rhs+=x.T@(2*np.log(y[sl])-eta[sl])
    return gram,rhs,len(y)


def predict(eta,h,state,zero_q,weights,spec):
    result=np.empty(len(h))
    for low in range(0,len(h),8192):
        sl=slice(low,low+8192);q0=zero_q if np.ndim(zero_q)==1 else zero_q[sl]
        result[sl]=np.exp((eta[sl]+design(h[sl],state[sl],q0,spec)@weights)/2)
    if not np.isfinite(result).all() or (result<=0).any():raise FloatingPointError('invalid ONDS initialized forecast')
    return result


class ReturnONDSReference(nn.Module):
    def __init__(self,base,weights,spec,*,frozen_terms=()):
        super().__init__();self.base=base;self.base.requires_grad_(False);self.spec=dict(spec)
        self.register_buffer('weights',torch.as_tensor(weights,dtype=torch.float64).clone())
        self.frozen_specs=[]
        for j,(parent_spec,parent_weights) in enumerate(frozen_terms):
            self.frozen_specs.append(dict(parent_spec))
            self.register_buffer(f'frozen_weights_{j}',torch.as_tensor(parent_weights,dtype=torch.float64).clone())
    def forward(self,args,state,zero_q):
        _,beta,h,_=self.base(*args);delta=window_basis(state,zero_q,self.spec)@self.weights
        for j,spec in enumerate(self.frozen_specs):
            delta=delta+window_basis(state,zero_q,spec)@getattr(self,f'frozen_weights_{j}')
        beta=beta.clone();beta[:,10]+=delta
        return torch.exp((h*beta).sum(1)/2),beta
