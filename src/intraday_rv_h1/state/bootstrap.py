"""Vectorized history warmup; current-day observations must not be present."""
import numpy as np
import pandas as pd
from intraday_rv_h1.features.native import raw_features
from intraday_rv_h1.features.return_percentile import causal_net_returns
from intraday_rv_h1.features.price_position import causal_price_path,BASIS,completed_price_closes
from intraday_rv_h1.data.schema import COLUMNS


def bootstrap(raw,symbol,registry,*,before):
    before=pd.Timestamp(before)
    f=raw.loc[:,list(COLUMNS)].copy();f['datetime']=pd.to_datetime(f.datetime);f['trade_date']=pd.to_datetime(f.trade_date)
    if f.empty or f.datetime.ge(before).any():raise ValueError('history must be strictly before replay boundary')
    f=f.loc[f.trade_date.ne(pd.Timestamp('2025-02-04'))].sort_values('datetime').reset_index(drop=True)
    if f.datetime.duplicated().any() or not f.datetime.dt.normalize().eq(f.trade_date).all():raise ValueError('invalid history keys')
    if not f.usymbol.astype(str).eq(symbol).all():raise ValueError('history includes other symbols')
    v=raw_features(f,symbol)
    histories={};last_aux={};eps={}
    for clock,indices in v.groupby('clock_slot_id',sort=False).indices.items():
        if clock not in registry:continue
        enabled=registry[clock]['aux_enabled'];g=v.iloc[indices]
        need=['core0','core1','core2']+[f'aux{j}' for j in range(2,6) if enabled[j]]
        valid=np.isfinite(g[need].to_numpy(float)).all(1)
        history=g.loc[valid,['core0','core1','core2']].to_numpy(np.float32)[-80:]
        if len(history)<80:raise ValueError(f'insufficient valid H80 history for {clock}')
        histories[clock]=history.tolist()
        last_aux[clock]=[float(g[f'aux{j}'].dropna().iloc[-1]) if g[f'aux{j}'].notna().any() else None for j in range(2,6)]
    if set(histories)!=set(registry):raise ValueError('history is missing registered clocks')
    # Thresholds use only source observations before the specified fit limit.
    for col in ('log_argument','negative_variance'):
        z=v.loc[v.datetime.le('2026-08-03 09:30:10'),col].to_numpy(float);z=z[np.isfinite(z)&(z>0)]
        if not len(z):raise ValueError(f'no positive extension threshold support for {col}')
        eps[col]=float(np.quantile(z,.001))
    obs=v.rename(columns={'spotRef':'spot_ref'})
    path=causal_price_path(obs,basis_by_symbol=BASIS);daily=completed_price_closes(path)
    returns=causal_net_returns(obs)
    over=obs['return'].where(obs.clock_position.eq(0)).groupby(obs.date,sort=False).first()
    opening=returns.groupby('date',sort=False).opening15.max() # unavailable values are zero; select via flags below
    opening=returns.loc[returns.opening15_available.eq(1)].groupby('date',sort=False).opening15.first()
    final_date=v.date.iloc[-1]
    if v.datetime.iloc[-1].strftime('%H:%M')!='15:00':raise ValueError('history warmup must end at an actual session close')
    price=path.iloc[-1]
    if not np.isfinite(price.log_price):raise ValueError('missing terminal price state')
    # Include only closes in the same reconstructed price segment.
    closes=daily.loc[daily.price_segment.eq(price.price_segment)&daily.close_available.eq(1),'log_price'].tail(80).tolist()
    if len(closes)<20:raise ValueError('insufficient completed price closes')
    tail=f.loc[f.trade_date.isin(f.trade_date.drop_duplicates().tail(2))].copy()
    # Preserve physical positions when a bounded raw buffer begins inside history.
    tail['clock_position']=v.loc[tail.index,'clock_position'].to_numpy()
    raw_rows=tail.astype(object).where(pd.notna(tail),None).to_dict('records')
    for r in raw_rows:r['datetime']=pd.Timestamp(r['datetime']).isoformat();r['trade_date']=pd.Timestamp(r['trade_date']).isoformat()
    return {'schema':'h1-state-v1','symbol':symbol,'history_before':before.isoformat(),
        'last_datetime':v.datetime.iloc[-1].isoformat(),'date':final_date.isoformat(),'histories':histories,
        'last_aux':last_aux,'raw_rows':raw_rows,'opening_history':sorted(opening.dropna().astype(float).tolist()),
        'overnight_history':sorted(over.dropna().astype(float).tolist()),'price_closes':closes,
        'log_price':float(price.log_price),'price_segment_valid':True,'thresholds':eps,
        'today_opening':None,'today_overnight':None,'previous_session':final_date.isoformat(),
        'last_predictions':{},'retries':{},'observed_today':[],'extension_mode':'carry',
        'extension_selection':'historical carry default; alternative modes require audited pre-cutoff evaluation'}
