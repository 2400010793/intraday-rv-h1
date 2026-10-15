"""Pure causal native features. No target, database, or weight access."""
import numpy as np
import pandas as pd
from .prefix import _attach_unanchored_prefix, UNANCHORED_COLUMNS


def three_origin(r,t):
    r=pd.Series(np.asarray(r,float));t=pd.Series(np.asarray(t,float))
    dt=t.diff().where(lambda v:v>0);branches=[]
    for w in (15,30):
        duration=dt.rolling(w,min_periods=w).sum()
        sigma=np.sqrt(260*r.rolling(w,min_periods=w).sum().pow(2)/duration).clip(.08,1.)
        branches.append(np.sqrt((sigma.pow(2)*duration).rolling(3,min_periods=3).sum()/duration.rolling(3,min_periods=3).sum()).clip(.08,1.))
    a,b=branches;weight=((a+.05)/(a+b+.1))**2
    return np.log((weight*a+(1-weight)*b)**2).to_numpy()


def ds15(r,t,dates):
    r=np.array(r,float,copy=True);t=pd.Series(np.asarray(t,float));r[np.r_[True,dates[1:]!=dates[:-1]]]=0.
    branches=[]
    for frequency in (15,60):
        dt=t.diff(frequency);dt=dt.where(dt>0)
        sampled=pd.Series(r).rolling(frequency,min_periods=frequency).sum()
        sigma=np.sqrt(260*sampled.pow(2)/dt).clip(.05,1.)
        branches.append(np.sqrt((sigma.pow(2)*dt).rolling(15,min_periods=15).sum()/dt.rolling(15,min_periods=15).sum()).clip(.08,1.))
    a,b=branches;w=((a+.05)/(a+b+.1))**2
    return np.log((w*a+(1-w)*b)**2).to_numpy()


def raw_features(raw,symbol):
    """Vectorized bootstrap, also usable on bounded past-only step buffers."""
    f=raw.copy().reset_index(drop=True)
    f['datetime']=pd.to_datetime(f.datetime);f['date']=pd.to_datetime(f.trade_date).dt.normalize();f['symbol']=symbol
    f['return']=f.ret.astype(float)
    if 'clock_position' not in f:f['clock_position']=f.groupby('date',sort=False).cumcount()
    minute=f.datetime.dt.strftime('%H:%M');occurrence=f.groupby(['date',minute],sort=False).cumcount()
    f['clock_slot_id']=minute+'__'+occurrence.astype(str)
    interior=f.clock_position.gt(0);p=f.loc[interior].copy();r=p.ret.to_numpy(float)
    width=242 if symbol in ('IF','IH','IC','IM') else 239
    negative=pd.Series(260*np.where(r<0,r*r,0.),index=p.index)
    positive=pd.Series(260*np.where(r>0,r*r,0.),index=p.index)
    neg=negative.rolling(width,min_periods=width).sum();pos=positive.rolling(width,min_periods=width).sum()
    close=p.datetime.dt.strftime('%H:%M').eq('15:00')
    neg.loc[close]=negative.groupby(p.date,sort=False).cumsum().loc[close]
    pos.loc[close]=positive.groupby(p.date,sort=False).cumsum().loc[close]
    rv=p.rvol1D.pow(2);arg=1+(pos-neg)/rv.where(rv>0)
    with np.errstate(divide='ignore',invalid='ignore'):
        p['core0']=np.log(rv.where(rv>0));p['core1']=np.log(neg.where(neg>0));p['core2']=np.log(arg.where(arg>0))
    p['log_argument']=arg;p['negative_variance']=neg
    for c in ('core0','core1','core2','log_argument','negative_variance'):f[c]=p[c].reindex(f.index)
    # Prefix includes the auction row as the zero-return anchor, matching the archived source cache.
    pref=_attach_unanchored_prefix(f)
    for c in UNANCHORED_COLUMNS:f[c]=f.datetime.map(pref.set_index('datetime')[c])
    f['policy0']=three_origin(f.ret,f.ttime);f['policy1']=ds15(f.ret,f.ttime,f.date.to_numpy())
    with np.errstate(divide='ignore',invalid='ignore'):
        f['aux2']=2*np.log(np.maximum(f.rvol1hour,1e-12))
        for j,c in enumerate(UNANCHORED_COLUMNS,3):f[f'aux{j}']=np.log(np.maximum(f[c],1e-12))
    return f


def midrank(ordered,x):
    a=np.asarray(ordered,float);lo=np.searchsorted(a,x,'left');hi=np.searchsorted(a,x,'right')
    return float((lo+.5*(hi-lo)+.5)/(len(a)+1))


def extended_log(a,epsilon,mode):
    if np.isfinite(a) and a>=epsilon:return float(np.log(a))
    if not np.isfinite(a):return float('nan')
    d=(a-epsilon)/epsilon
    if mode=='tangent':return float(np.log(epsilon)+d)
    if mode=='asinh':return float(np.log(epsilon)+np.arcsinh(d))
    return float('nan')
