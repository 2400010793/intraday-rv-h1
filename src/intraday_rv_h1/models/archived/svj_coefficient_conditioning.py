"""Named HARNet-SVJ coefficient conditioning on an immutable trained reference."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

WIDTH=16
AUX_OFFSET=10
CAP=.2
DECAY_ROWS=60.


def reference_coefficients(h,clock,rank,position,head,six):
    """Algebraically distribute CAP*tanh(sum_j h_j*r_j*g*w_j / CAP) over six coefficients."""
    h=np.asarray(h,np.float64);clock=np.asarray(clock,np.float64)
    rank=np.asarray(rank,np.float64);position=np.asarray(position,np.float64)
    beta=np.asarray(head,np.float64)
    w=np.asarray(six,np.float64)
    if h.ndim!=2 or h.shape[1]!=WIDTH or clock.shape!=(len(h),3) or rank.shape!=(len(h),6) or beta.shape!=(4,16) or w.shape!=(6,):
        raise ValueError('wrong named reference layout')
    if position.shape!=(len(h),) or np.any(position<1) or not all(np.isfinite(array).all() for array in (h,clock,rank,position,beta,w)):
        raise ValueError('invalid reference inputs')
    base=beta[0]+clock@beta[1:]
    gate=np.exp(-(position-1.)/DECAY_ROWS)
    factors=rank*w[None,:]*gate[:,None]
    raw=np.sum(h[:,AUX_OFFSET:]*factors,axis=1)
    z=raw/CAP
    ratio=np.ones_like(z)
    nz=z!=0.
    ratio[nz]=np.tanh(z[nz])/z[nz]
    base[:,AUX_OFFSET:]+=factors*ratio[:,None]
    if not np.isfinite(base).all():raise FloatingPointError('invalid effective reference coefficients')
    return base


class SVJCoefficientHead(nn.Module):
    """Per-feature coefficient queries into six own-percentile tokens and one time token.

    ``coefficient_limit=False`` is used by independent coefficient models.  It
    removes the legacy reference-relative cap from the module entirely rather
    than leaving an unused limit buffer in the model state.
    """
    def __init__(self,arm='mlp_time_percentile',width=32,coefficient_limit=True):
        super().__init__()
        if arm not in ('mlp_time','mlp_time_percentile','local_attention'):
            raise ValueError('unknown coefficient arm')
        self.arm=arm
        self.coefficient_limit=bool(coefficient_limit)
        if self.coefficient_limit:
            self.register_buffer('limit', torch.ones(WIDTH,dtype=torch.float64)*.25)
        if arm.startswith('mlp'):
            n=3 if arm=='mlp_time' else 21
            self.encoder=nn.Sequential(nn.Linear(n,width),nn.Tanh(),nn.Linear(width,WIDTH))
            nn.init.zeros_(self.encoder[-1].weight);nn.init.zeros_(self.encoder[-1].bias)
        else:
            self.time=nn.Linear(3,width)
            self.percentile=nn.Linear(3,width)
            self.token_type=nn.Parameter(torch.zeros(7,width))
            self.queries=nn.Parameter(torch.randn(16,width)*.02)
            self.norm_query=nn.LayerNorm(width)
            self.norm_tokens=nn.LayerNorm(width)
            self.attention=nn.MultiheadAttention(width,4,batch_first=True,dropout=0.)
            self.output=nn.Linear(width,1)
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.double()

    def delta(self,state):
        if state.ndim!=2 or state.shape[1]!=21 or not torch.isfinite(state).all():
            raise ValueError('expected finite 3+6x3 state')
        if self.arm.startswith('mlp'):
            raw=self.encoder(state[:,:3] if self.arm=='mlp_time' else state)
        else:
            batch=len(state)
            tokens=torch.cat((self.time(state[:,:3]).unsqueeze(1),
                              self.percentile(state[:,3:].reshape(batch,6,3))),dim=1)
            tokens=tokens+self.token_type
            queries=self.norm_query(self.queries).unsqueeze(0).expand(batch,-1,-1)
            # Mask only absent percentile channels; the clock token remains visible.
            missing=state[:,3:].reshape(batch,6,3)[:,:,2]<=0
            padding=torch.cat((torch.zeros(batch,1,device=state.device,dtype=torch.bool),missing),dim=1)
            attended,_=self.attention(queries,self.norm_tokens(tokens),self.norm_tokens(tokens),key_padding_mask=padding,need_weights=False)
            raw=self.output(attended).squeeze(-1)
        if not self.coefficient_limit:
            return raw
        return self.limit*torch.tanh(raw)

    def forward(self,features,reference,state):
        if features.ndim!=2 or features.shape[1]!=WIDTH or reference.shape!=features.shape:
            raise ValueError('named feature/reference shape mismatch')
        coefficients=reference+self.delta(state)
        eta=(features*coefficients).sum(dim=-1)
        forecast=torch.exp(eta/2.)
        if not torch.isfinite(forecast).all() or torch.any(forecast<=0):
            raise FloatingPointError('invalid conditioned SVJ prediction')
        return forecast,coefficients
