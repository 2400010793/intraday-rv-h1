"""Offline reuse of archived original tensors, plus causal auction inputs.
No alternate feature or network formula. Original regular rows are read unchanged.
"""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from intraday_rv_h1.data.schema import COLUMNS, validate_row
from intraday_rv_h1.features.native import raw_features
from intraday_rv_h1.features.return_percentile import causal_net_returns
from intraday_rv_h1.features.price_position import causal_price_path, completed_price_closes, BASIS
from intraday_rv_h1.inference.engine import Predictor

PARENT='notebook10-onds-three-origin-monthly-adam-shampoo-e6-20261014-v1'
CONTROL='notebook10-monthly-linear-head-sharing-e8-20261009-v1'
ARM='onds15_30_three__drop_negative80'

class OriginalInputs:
    def __init__(self,source,symbol,month):
        root=Path(source)/'results/intraday_rv';self.root=root/PARENT;self.control=root/CONTROL
        self.cache=self.root/'cache'/symbol/month;self.symbol=symbol;self.month=month
        self.audit=json.loads((self.cache/'audit.json').read_text());self.offset=self.audit['local_fit_offset']
        self.arrays={s:[np.load(self.control/'source_cache'/symbol/s/f'arg{j}.npy',mmap_mode='r') for j in range(10)]+[np.load(self.control/'states'/symbol/f'{s}_{name}.npy',mmap_mode='r') for name in ('return','coefficient')] for s in ('fit','test')}
        self.targets={s:np.load(self.control/'source_cache'/symbol/s/'targets.npy',mmap_mode='r') for s in ('fit','test')}
        self.rows={s:np.load(self.cache/f'{s}_rows.npy',mmap_mode='r') for s in ('fit','score')}
        self.policy={s:np.load(self.cache/f'{ARM}_{s}_policy.npy',mmap_mode='r') for s in ('fit','score')}
        self.metadata=pd.concat([pd.read_parquet(self.control/'source_cache'/symbol/s/'meta.parquet') for s in ('fit','test')],ignore_index=True)
        self.frame=pd.read_parquet(self.cache/'keys.parquet')
    def batch(self,split,positions):
        ix=self.rows[split][positions];a=ix<self.offset;out=[]
        for j in range(12):
            value=np.empty((len(ix),*self.arrays['fit'][j].shape[1:]),dtype=self.arrays['fit'][j].dtype)
            value[a]=self.arrays['fit'][j][ix[a]];value[~a]=self.arrays['test'][j][ix[~a]-self.offset];out.append(torch.from_numpy(value))
        return out+[torch.from_numpy(self.policy[split][positions].copy())]
    def target(self,positions):
        ix=self.rows['fit'][positions];a=ix<self.offset;out=np.empty(len(ix));out[a]=self.targets['fit'][ix[a]];out[~a]=self.targets['test'][ix[~a]-self.offset];return out


def opening_inputs(raw,symbol,bundle,month,wanted,progress=None):
    """Construct each auction from strictly prior history with the accepted engine.
    Bounded per-auction work, never a loop over all >100k production rows.
    """
    raw=raw.copy();raw['usymbol']=symbol
    raw=raw.loc[pd.to_datetime(raw.trade_date).ne(pd.Timestamp('2025-02-04'))].sort_values('datetime').reset_index(drop=True)
    raw['trade_date']=pd.to_datetime(raw.trade_date);raw['datetime']=pd.to_datetime(raw.datetime)
    features=raw_features(raw,symbol)
    base=Path(bundle)/symbol;folder=base/month
    engine=Predictor.__new__(Predictor);engine.symbol=symbol
    engine.registry=json.loads((base/'clock_registry.json').read_text());engine.transform=json.loads((base/'transform.json').read_text())
    engine.normalization=json.loads((folder/'policy_normalization.json').read_text())
    with np.load(base/'aux_ecdf.npz') as z:engine.aux_ecdf={k:z[k] for k in z.files}
    with np.load(folder/'policy_ecdf.npz') as z:engine.policy_ecdf={k:z[k] for k in z.files}
    engine.asset=['IF','IH','IC','IM','159915','588000'].index(symbol)
    first_mapped=min(engine.registry,key=lambda c:engine.registry[c]['position'])
    perclock={}
    for clock,group in features.groupby('clock_slot_id',sort=False):
        if clock not in engine.registry:continue
        enabled=engine.registry[clock]['aux_enabled']
        need=['core0','core1','core2']+[f'aux{j}' for j in range(2,6) if enabled[j]]
        good=group.loc[np.isfinite(group[need]).all(1)]
        perclock[clock]=(group,good.datetime.to_numpy(),good[['core0','core1','core2']].to_numpy(np.float32))
    obs=features.rename(columns={'spotRef':'spot_ref'})
    returns=causal_net_returns(obs)
    opens=returns.loc[returns.opening15_available.eq(1)].groupby('date',sort=False).head(1)
    auction=features.loc[features.clock_position.eq(0)]
    path=causal_price_path(obs,basis_by_symbol=BASIS);closes=completed_price_closes(path)
    wanted=pd.DatetimeIndex(wanted);indices=pd.Index(raw.datetime).get_indexer(wanted)
    if (indices<0).any():raise ValueError('missing auction origin')
    sessions=raw.trade_date.drop_duplicates().to_numpy();session_starts=np.flatnonzero(~raw.trade_date.duplicated().to_numpy());collected=[[] for _ in range(13)];flags=[]
    for count,i in enumerate(indices):
        row=raw.iloc[i];dt=row.datetime;date=row.trade_date
        isopening=features.clock_position.iloc[i]==0
        mapped=first_mapped if isopening else features.clock_slot_id.iloc[i]
        if mapped not in perclock:raise ValueError(f'unregistered original clock {mapped}')
        regular,gt,gh=perclock[mapped]
        n=np.searchsorted(gt,np.datetime64(dt));history=gh[max(0,n-80):n]
        if len(history)!=80:raise ValueError(f'insufficient auction H80 {dt}')
        earlier=regular.loc[regular.datetime.lt(dt)]
        aux=[float(earlier[f'aux{j}'].dropna().iloc[-1]) if earlier[f'aux{j}'].notna().any() else None for j in range(2,6)]
        d=int(np.searchsorted(sessions,np.datetime64(date)));past=raw.iloc[session_starts[max(0,d-2)]:i][list(COLUMNS)].copy()
        if past.empty:raise ValueError('missing previous sessions')
        past['clock_position']=features.loc[past.index,'clock_position'];past=past.astype(object).where(pd.notna(past),None)
        rr=past.to_dict('records')
        for r in rr:r['datetime']=pd.Timestamp(r['datetime']).isoformat();r['trade_date']=pd.Timestamp(r['trade_date']).isoformat()
        previous=closes.loc[closes.date.lt(date)&closes.close_available.eq(1)]
        terminal=previous.iloc[-1];same=previous.loc[previous.price_segment.eq(terminal.price_segment)]
        if len(same)<20:raise ValueError('auction price history unavailable')
        state={'symbol':symbol,'date':pd.Timestamp(sessions[d-1] if isopening else date).isoformat(),'previous_session':pd.Timestamp(sessions[d-1]).isoformat(),'today_opening':None if isopening or i==0 or returns.opening15_available.iloc[i-1]!=1 else float(returns.opening15.iloc[i-1]),'today_overnight':None if isopening else float(raw.ret.iloc[session_starts[d]]),'observed_today':[dict(r) for r in rr if pd.Timestamp(r['trade_date'])==date],'last_predictions':{},'raw_rows':rr,'histories':{mapped:history.tolist()},'last_aux':{mapped:aux},'opening_history':sorted(opens.loc[opens.date.lt(date),'opening15'].dropna().tolist()),'overnight_history':sorted(auction.loc[auction.date.lt(date),'ret'].dropna().tolist()),'price_closes':same.log_price.tail(80).tolist(),'log_price':float(path.log_price.iloc[i-1]),'price_segment_valid':bool(np.isfinite(path.log_price.iloc[i-1])),'last_datetime':pd.Timestamp(raw.datetime.iloc[i-1]).isoformat(),'extension_mode':'carry','thresholds':{}}
        args,ff,clock=engine._features(state,validate_row(row[list(COLUMNS)].to_dict(),symbol))
        for j,value in enumerate(args):collected[j].append(value)
        flags.append({'datetime':str(dt),'flags':ff})
        if progress is not None and (count==0 or (count+1)%100==0):progress(count+1,len(indices))
    return [np.concatenate(v,axis=0) for v in collected],flags
