"""Explicit user-authorized small Train-only Ridge screen, separate from deployment.
Never loads production weights or modifies bundles. Test is not used for fitting.
"""
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import pandas as pd
from intraday_rv_h1.features.native import raw_features

SYMBOLS=['IF','IH','IC','IM','159915','588000']


def metrics(y,p):
    if not np.isfinite(p).all() or (p<=0).any():raise ValueError('invalid positive prediction')
    r=(y/p)**2
    return {'rows':len(y),'mse':float(np.mean((p-y)**2)),'mae':float(np.mean(abs(p-y))),'qlike':float(np.mean(r-np.log(r)-1))}


def fit(x,y,mask,mean,scale):
    z=(x-mean)/scale;d=np.column_stack([np.ones(len(x)),z]);a=d[mask];target=np.log(y[mask])
    penalty=np.eye(d.shape[1])*mask.sum()*.001;penalty[0,0]=0
    beta=np.linalg.solve(a.T@a+penalty,a.T@target)
    return np.exp(d@beta),beta


def build(raw,symbol):
    raw=raw.loc[pd.to_datetime(raw.trade_date).ne(pd.Timestamp('2025-02-04'))].sort_values('datetime').reset_index(drop=True)
    raw['clock_position']=raw.groupby('trade_date',sort=False).cumcount()
    targets=raw[['trade_date','datetime','clock_position','rvol1D']].copy()
    targets['y']=targets.groupby('clock_position',sort=False).rvol1D.shift(-1)
    targets['target_datetime']=targets.groupby('clock_position',sort=False).datetime.shift(-1)
    # Do not jump across missing sessions: verify actual next observed session.
    sessions=raw.trade_date.drop_duplicates().sort_values();nxt=dict(zip(sessions.iloc[:-1],sessions.iloc[1:]));wanted=raw.trade_date.map(nxt)
    targets.loc[pd.to_datetime(targets.target_datetime).dt.normalize().ne(wanted),['y','target_datetime']]=[np.nan,pd.NaT]
    f=raw_features(raw,symbol);rv=np.log(np.maximum(raw.rvol1D.to_numpy(float),1e-12)**2)
    rv=pd.Series(rv).where(raw.rvol1D.gt(0));base=[rv]
    for width in [5,20,80]:base.append(rv.groupby(raw.clock_position,sort=False).transform(lambda s:s.rolling(width,min_periods=width).mean()))
    base += [np.log(raw.rvol1hour.where(raw.rvol1hour.gt(0))**2),f.core1.ffill(),f.core2.ffill(),f.policy0,f.policy1,np.arcsinh(raw.ret/.01)]
    overnight=raw.ret.where(raw.clock_position.eq(0)).groupby(raw.trade_date,sort=False).transform('first')
    base.append(np.arcsinh(overnight/.01))
    names=['log_rv','mean5','mean20','mean80','log_1hour','log_negative','log_signed','three_origin','ds15','return_asinh','overnight_asinh']
    x=np.column_stack(base);missing=~np.isfinite(x);x=np.where(missing,0,x);x=np.column_stack([x,missing.astype(float)])
    names += [n+'_missing' for n in names]
    opening=raw.clock_position.eq(0).to_numpy();x=np.column_stack([x,opening]);names+=['opening']
    return raw,targets,x,opening,names


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();records=[];audits=[]
    manifest={'status':'running','protocol_id':'INTRADAY-SSC-20260829','model':'small standalone log-vol Ridge; NOT attention retraining','ridge_mean_penalty':.001,'fit_target_end_exclusive':'2024-03-01','validation_target_start':'2024-03-01','validation_target_end_exclusive':'2025-03-01','purge':'fit max target must precede first Validation origin','symbols':a.symbols.split(','),'workers':1,'threads':1,'test_used_for_selection':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'inputs':{}}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    try:
      for symbol in a.symbols.split(','):
        tick=time.monotonic();path=a.snapshot/f'{symbol}.parquet';raw=pd.read_parquet(path,filters=[('datetime','<',pd.Timestamp('2025-03-01'))]);raw,t,x,opening,names=build(raw,symbol)
        valid=t.y.gt(0).to_numpy()&np.isfinite(t.y)&raw.rvol1D.gt(0).to_numpy()
        train=(pd.to_datetime(t.target_datetime)<pd.Timestamp('2024-03-01')).to_numpy()&valid
        val=(pd.to_datetime(t.target_datetime)>=pd.Timestamp('2024-03-01')).to_numpy()&(pd.to_datetime(t.target_datetime)<pd.Timestamp('2025-03-01')).to_numpy()&valid
        first=pd.to_datetime(raw.loc[val,'datetime']).min();train &= pd.to_datetime(t.target_datetime).lt(first).to_numpy()
        # Warmup excludes short same-clock H80 support; no future imputation.
        mature=~(x[:,3+11].astype(bool)) # mean80 missing mask
        train &= mature;val &= mature
        # Fixed checkpoints and dense first 15 regular positions; bounded regression fit.
        sampled=(raw.clock_position.le(16)|raw.clock_position.mod(15).eq(0)).to_numpy()
        regular=train&sampled&~opening;with_open=train&sampled
        mean=x[regular].mean(0);scale=x[regular].std(0);scale[scale<1e-10]=1
        y=t.y.to_numpy(float)
        # Fit rows only influence coefficients and transforms. Other y values never enter fitting.
        base,_=fit(x,y,regular,mean,scale)
        both,beta=fit(x,y,with_open,mean,scale)
        z=(x[:,:11]-mean[:11])/scale[:11];interactions=z*opening[:,None]
        xx=np.column_stack([x,interactions]);mm=np.r_[mean,np.zeros(11)];ss=np.r_[scale,np.ones(11)]
        crossed,crossbeta=fit(xx,y,with_open,mm,ss)
        predictions={'regular_only':base,'include_opening':both,'opening_interactions':crossed,'current_rv':raw.rvol1D.to_numpy(float)}
        replica=json.loads((a.benchmark/'validation/all_rvol__level__common__ridge0/model.json').read_text())
        predictions['benchmark_replica_train']=raw[replica['columns']].to_numpy(float)@np.array(replica['raw_slopes'])+replica['raw_intercepts'][0]
        predictions['benchmark_database']=raw.predRVol1D.to_numpy(float)
        common=val.copy()
        for p in predictions.values():common &= np.isfinite(p)&(p>0)
        for group,select in [('opening',opening),('first15_regular',(raw.clock_position.gt(0)&raw.clock_position.le(16)).to_numpy()),('remaining',raw.clock_position.gt(16).to_numpy()),('all',np.ones(len(raw),bool))]:
          mask=common&select
          if not mask.any():raise ValueError(f'empty validation {symbol}/{group}')
          for name,pred in predictions.items():records.append({'symbol':symbol,'group':group,'model':name,**metrics(y[mask],pred[mask])})
        fitmax=str(pd.to_datetime(t.loc[with_open,'target_datetime']).max());assert pd.Timestamp(fitmax)<first
        audit={'symbol':symbol,'fit_regular':int(regular.sum()),'fit_with_open':int(with_open.sum()),'validation_common':int(common.sum()),'validation_before_common':int(val.sum()),'max_fit_target':fitmax,'first_validation_origin':str(first),'seconds':time.monotonic()-tick,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'snapshot_sha256':hashlib.sha256(path.read_bytes()).hexdigest()};audits.append(audit)
        np.savez(a.output/f'{symbol}_screen_parameters.npz',mean=mean,scale=scale,beta=beta,cross_beta=crossbeta,feature_names=np.array(names))
        # Restrict saved observations to Validation; raw data remains local.
        pd.DataFrame({'datetime':raw.loc[common,'datetime'],'target_datetime':t.loc[common,'target_datetime'],'y':y[common],'opening':opening[common],**{name:pred[common] for name,pred in predictions.items()}}).to_parquet(a.output/f'{symbol}_validation.parquet',index=False)
        print(json.dumps(audit),flush=True)
      pd.DataFrame(records).to_csv(a.output/'metrics.csv',index=False);manifest.update(status='complete',seconds=time.monotonic()-start,audits=audits);(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    except Exception as exc:
      manifest.update(status='failed',error_type=type(exc).__name__,error=str(exc));(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--symbols',default=','.join(SYMBOLS));main(p.parse_args())
