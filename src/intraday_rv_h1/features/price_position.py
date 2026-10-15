"""Causal price position and trend against completed prior-session closes."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .seasonal_energy import KEY, _observations

BASIS = {s:'production_return' for s in ('IF','IH','IC','IM')}
BASIS.update({s:'spot_ref' for s in ('159915','588000')})


def causal_price_path(observations, *, basis_by_symbol):
    """Futures use production returns; spot levels anchor each finite segment.

    A missing return breaks the reconstructed path: subsequent levels restart
    at their first observed reference price and never compare across that break.
    ETF reference prices are read directly, with missing prices retained.
    """
    if 'spot_ref' not in observations:raise ValueError('price position requires spot_ref')
    f=_observations(observations)
    prices=observations[[*KEY,'spot_ref']]
    f=f.merge(prices,on=KEY,how='left',validate='one_to_one',sort=False)
    spot=pd.to_numeric(f.spot_ref,errors='raise').to_numpy(float)
    if np.isinf(spot).any() or (spot[np.isfinite(spot)]<=0).any():
        raise ValueError('price reference must be finite positive or explicitly missing')
    out=f[[*KEY,'datetime','return','spot_ref']].copy()
    out['log_price']=np.nan;out['price_available']=0.;out['price_segment']=-1
    out['price_basis']=''
    for symbol,indices in f.groupby('symbol',sort=False).indices.items():
        mode=basis_by_symbol.get(symbol)
        if mode not in ('spot_ref','production_return'):raise ValueError(f'unknown declared price basis:{symbol}/{mode}')
        ix=np.asarray(indices);r=f['return'].iloc[ix].to_numpy(float);p=spot[ix]
        valid=np.isfinite(p)
        if mode=='production_return':valid&=np.isfinite(r)
        log_spot=np.log(np.where(valid,p,1.))
        segment=np.cumsum(~valid).astype(np.int64)
        if mode=='production_return':
            cumulative=pd.Series(np.where(valid,r,0.)).groupby(segment,sort=False).cumsum()
            anchor_cumulative=cumulative.where(valid).groupby(segment,sort=False).transform('first')
            anchor_price=pd.Series(log_spot).where(valid).groupby(segment,sort=False).transform('first')
            log_price=anchor_price.to_numpy()+cumulative.to_numpy()-anchor_cumulative.to_numpy()
        else:
            log_price=log_spot
            # ETF levels are observed directly, so a missing minute does not
            # force artificial rebasing of later known prices.
            segment=np.zeros(len(ix),dtype=np.int64)
        out.loc[ix,'log_price']=np.where(valid,log_price,np.nan)
        out.loc[ix,'price_available']=valid.astype(float)
        out.loc[ix,'price_segment']=np.where(valid,segment,-1)
        out.loc[ix,'price_basis']=mode
    return out


def completed_price_closes(path):
    """Do not replace a missing/unobserved close with an earlier minute."""
    daily=path.groupby(['symbol','date'],sort=False).tail(1).copy()
    actual_close=daily.datetime.dt.hour.eq(15)&daily.datetime.dt.minute.eq(0)&daily.price_available.eq(1)
    daily['close_available']=actual_close.astype(float)
    daily.loc[~actual_close,'log_price']=np.nan
    daily.loc[~actual_close,'price_segment']=-1
    return daily.reset_index(drop=True)


def causal_price_position(observations, *, basis_by_symbol, windows=(20,80), chunk_size=8192):
    if (not windows or len(set(windows))!=len(windows) or any(not isinstance(w,int) or w<2 for w in windows)
            or not isinstance(chunk_size,int) or chunk_size<1):
        raise ValueError('invalid price history windows/chunk size')
    path=causal_price_path(observations,basis_by_symbol=basis_by_symbol)
    daily=completed_price_closes(path);out=path.copy()
    for window in windows:
        out[f'price_q{window}']=.5;out[f'price_state_available{window}']=0.
        for name in ('momentum','drawdown','breakout'):out[f'price_log_{name}{window}']=0.
    for symbol,indices in path.groupby('symbol',sort=False).indices.items():
        ix=np.asarray(indices);f=path.iloc[ix];d=daily.loc[daily.symbol.eq(symbol)]
        dates=pd.Index(d.date);day=dates.get_indexer(f.date)
        if (day<0).any():raise ValueError('price history missing origin session')
        closes=d.log_price.to_numpy();segments=d.price_segment.to_numpy()
        current=f.log_price.to_numpy();current_segments=f.price_segment.to_numpy()
        for window in windows:
            # The row for day i ends with day i-1's close. Today's complete
            # close is never part of today's reference distribution.
            history=np.lib.stride_tricks.sliding_window_view(np.r_[np.full(window,np.nan),closes],window)[:len(d)]
            old_segments=np.lib.stride_tricks.sliding_window_view(np.r_[np.full(window,-1),segments],window)[:len(d)]
            for low in range(0,len(ix),chunk_size):
                chosen=slice(low,min(low+chunk_size,len(ix)));hist=history[day[chosen]]
                seg=old_segments[day[chosen]];value=current[chosen]
                available=np.isfinite(value)&np.isfinite(hist).all(1)&(seg==current_segments[chosen,None]).all(1)
                q=((hist<value[:,None]).sum(1)+.5*(hist==value[:,None]).sum(1)+.5)/(window+1)
                safe=np.where(np.isfinite(hist),hist,0.)
                high=safe.max(1);momentum=value-hist[:,0];breakout=value-high
                rows=ix[chosen]
                out.loc[rows,f'price_q{window}']=np.where(available,q,.5)
                out.loc[rows,f'price_state_available{window}']=available.astype(float)
                out.loc[rows,f'price_log_momentum{window}']=np.where(available,momentum,0.)
                out.loc[rows,f'price_log_drawdown{window}']=np.where(available,np.maximum(-breakout,0.),0.)
                out.loc[rows,f'price_log_breakout{window}']=np.where(available,breakout,0.)
    return out
