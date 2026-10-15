"""Offline metrics; future labels are only accepted in this module."""
import numpy as np
import pandas as pd


def evaluate(predictions,targets,threshold,*,prediction_columns):
    keys=['symbol','datetime']
    for f in (predictions,targets):
        if f.empty or f.duplicated(keys).any():raise ValueError('empty/duplicate evaluation keys')
    f=predictions.merge(targets,on=keys,validate='one_to_one',how='inner')
    if len(f)!=len(predictions):raise ValueError('missing exact targets; report coverage separately before scoring')
    f=f.sort_values(keys).reset_index(drop=True);times=pd.to_datetime(f.datetime)
    group=[f.symbol,times.dt.normalize()];gaps=times.groupby(group).diff().dt.total_seconds().to_numpy()
    links=np.isfinite(gaps)&(gaps>0)&(gaps<=90);segments=np.cumsum(~links);seconds=times.astype('int64').to_numpy()/1e9
    y=f.y_true.to_numpy(float);current=f.current_Rvol1D.to_numpy(float)
    if not np.isfinite(y).all() or (y<=0).any():raise ValueError('invalid evaluation target')
    rows=[]
    for name in prediction_columns:
        p=f[name].to_numpy(float)
        if not np.isfinite(p).all() or (p<=0).any():raise ValueError(f'invalid prediction: {name}')
        delta=p-np.roll(p,1);level=np.maximum(.5*(current+np.roll(current,1)),1e-8)
        limits=np.array([threshold[str(s)] for s in f.symbol]) if isinstance(threshold,dict) else float(threshold)
        jumps=links&(np.abs(delta)/level>limits);reversal=np.zeros(len(f),bool)
        for offset in range(1,min(32,len(f))):
            allowed=(segments[:-offset]==segments[offset:])&(seconds[offset:]-seconds[:-offset]<=600)
            reversal[:-offset]|=allowed&((p[:-offset]-p[offset:])*np.sign(delta[:-offset])>=.5*np.abs(delta[:-offset]))
        ratio=(y/p)**2;den=int(links.sum())
        rows.append({'model':name,'rows':len(f),'mse':float(np.mean((p-y)**2)),'mae':float(np.mean(np.abs(p-y))),
            'qlike':float(np.mean(ratio-np.log(ratio)-1)),'eligible_links':den,'jumps':int(jumps.sum()),
            'needle_spikes':int((jumps&reversal).sum()),'jump_rate':float(jumps.sum()/den) if den else None,
            'needle_rate':float((jumps&reversal).sum()/den) if den else None})
    return pd.DataFrame(rows)
