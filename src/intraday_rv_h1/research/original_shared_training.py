"""Small paired trial of the ORIGINAL archived HARNet graph on every clock.
Explicit offline source dependency for original Ridge fitting; never inference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
import torch
from intraday_rv_h1.models.loader import build_model
from intraday_rv_h1.models.archived.monthly_demo_attention import replace_policy
from .original_shared_inputs import OriginalInputs, opening_inputs
from .metrics import metric
from volatility_forecast.rv_predict.models.harnet import percentile_initialization_ablation as ridge


def run(a):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    dest=a.output/a.symbol;dest.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    status={'status':'running','pid':os.getpid(),'symbol':a.symbol,'model':'ORIGINAL MonthlyDemoHARNet','stage':'load'}
    def write():(dest/'status.json').write_text(json.dumps(status,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    write()
    try:
        data=OriginalInputs(a.source,a.symbol,a.month);bundle=a.bundle
        factory=torch.load(bundle/a.symbol/'factory.pt',map_location='cpu',weights_only=True)
        status['source_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        status['factory_sha256']=hashlib.sha256((bundle/a.symbol/'factory.pt').read_bytes()).hexdigest()
        cp=data.root/'tasks/adam'/a.month/f'{a.symbol}__seed7'/'checkpoint_00.pt'
        payload=torch.load(cp,map_location='cpu',weights_only=False)
        model=build_model(factory,payload['model'],7,training=True)
        model.eval();ref=np.load(data.cache/'onds15_30_three__drop_negative80_reference.npy')
        def normal_score(m):
            with torch.no_grad():return np.concatenate([m(*data.batch('score',np.arange(i,min(i+512,len(data.frame)))))[0].numpy() for i in range(0,len(data.frame),512)])
        archived=normal_score(model);error=float(abs(archived-ref).max())
        if error>1e-8:raise ValueError(f'original epoch0 closure {error}')
        raw=pd.read_parquet(a.source/'data/snapshots/notebook0-rvol-prod-intraday-multiscale-20260918-v1'/f'{a.symbol}.parquet')
        raw=raw.loc[pd.to_datetime(raw.trade_date).ne(pd.Timestamp('2025-02-04'))].sort_values('datetime').reset_index(drop=True)
        raw['_position']=raw.groupby('trade_date',sort=False).cumcount()
        first=raw.groupby('trade_date',sort=False).head(1).copy()
        first['y']=first.rvol1D.shift(-1);first['target_datetime']=first.datetime.shift(-1)
        fitmeta=data.metadata.iloc[data.rows['fit']];cutoff=pd.Timestamp(data.audit['plan']['max_fit_target_datetime'])
        earliest=fitmeta.datetime.min()
        fit=first.datetime.ge(earliest)&first.target_datetime.le(cutoff)&first.y.gt(0)
        score=first.datetime.gt(cutoff)&first.datetime.ge(data.frame.datetime.min().normalize())&first.datetime.le(data.frame.datetime.max())&first.y.gt(0)
        if a.smoke:
            eligible=first.index[fit].to_numpy();picked=np.unique(np.r_[eligible[:2],eligible[-2:]])
            fit &= first.index.isin(picked)
            eligible_score=first.index[score].to_numpy();score &= first.index.isin(eligible_score[:3])
        chosen=first.loc[fit|score].copy();chosen['_fit']=fit.loc[chosen.index];chosen['_score']=score.loc[chosen.index];chosen['opening']=True;nopen=int(fit.sum())
        if not a.smoke:
            begin=first.loc[score,'datetime'].min().normalize();end=data.frame.datetime.max()
            clocks=raw.datetime.dt.strftime('%H:%M:%S');all_y=raw.groupby(clocks,sort=False).rvol1D.shift(-1);all_td=raw.groupby(clocks,sort=False).datetime.shift(-1)
            missing=raw.datetime.ge(begin)&raw.datetime.le(end)&~raw.datetime.isin(data.frame.datetime)&~raw.index.isin(first.index)
            extra=raw.loc[missing].copy();extra['y']=all_y[missing];extra['target_datetime']=all_td[missing];extra['_fit']=False;extra['_score']=True;extra['opening']=False
            if not extra.empty and (extra.y.isna().any() or extra.target_datetime.isna().any()):raise ValueError('missing extra regular target')
            chosen=pd.concat([chosen,extra]).sort_values('datetime')
        if not nopen or not score.any():raise ValueError('opening fit/score empty')
        if first.loc[score,'datetime'].min()<=cutoff:raise ValueError('opening predictions predate training labels')
        status.update(stage='opening_inputs',regular_rows=len(data.rows['fit']),opening_rows=nopen,epoch0_error=error);write()
        def progress(n,total):
            status.update(opening_built=n,opening_total=total,seconds=time.monotonic()-start);write();print(json.dumps(status),flush=True)
        oa,flags=opening_inputs(raw,a.symbol,bundle,a.month,chosen.datetime,progress)
        trainopen=chosen['_fit'].to_numpy(bool);scoreopen=chosen['_score'].to_numpy(bool)
        ofit=[torch.from_numpy(v[trainopen]) for v in oa];oscore=[torch.from_numpy(v[scoreopen]) for v in oa]
        oy=chosen.loc[trainopen,'y'].to_numpy(float);sy=chosen.loc[scoreopen,'y'].to_numpy(float)
        n=len(data.rows['fit']);targets=np.r_[data.target(np.arange(n)),oy]
        def batch(ix,include):
            normal=ix<n;parts=[]
            if normal.any():normalargs=data.batch('fit',ix[normal])
            for j in range(13):
                array=np.empty((len(ix),*ofit[j].shape[1:]),dtype=ofit[j].numpy().dtype)
                if normal.any():array[normal]=normalargs[j].numpy()
                if (~normal).any():array[~normal]=ofit[j][ix[~normal]-n].numpy()
                parts.append(torch.from_numpy(array))
            return parts
        # Reuse original reference fitting code exactly. Extract original h, clock,
        # rank and downside-percentile basis, including causally mapped auctions.
        values={k:[] for k in ['h','clock','rank','position','q']};states=[]
        with torch.no_grad():
            for low in range(0,n+nopen,512):
                args=batch(np.arange(low,min(low+512,n+nopen)),True)
                _,_,h,_=model(*args)
                _,_,effective=replace_policy(args[1],args[2],args[3],args[12])
                q=model.own_percentiles(h,args[5],args[8]);q[:,1]=.5
                for k,v in [('h',h),('clock',args[2][:,:3]),('rank',effective[:,:6,0]),('position',args[6]),('q',q)]:values[k].append(v.numpy())
                states.append(args[11].numpy())
        design={k:np.concatenate(v) for k,v in values.items()};design['y']=targets
        parameters,fit_audit=ridge.fit(design,np.concatenate(states),'drop_negative80')
        np.savez(dest/'shared_initializer.npz',**parameters)
        if getattr(a,'prepare_only',False):
            for split,args in [('fit',ofit),('score',oscore)]:
                for j,value in enumerate(args):np.save(dest/f'{split}_extra_arg{j}.npy',value.numpy())
            np.save(dest/'fit_extra_targets.npy',oy)
            chosen.loc[scoreopen,['datetime','target_datetime','y','opening']].to_parquet(dest/'extra_score_keys.parquet',index=False)
            (dest/'opening_flags.json').write_text(json.dumps(flags,indent=2))
            status.update(status='complete',stage='cache_complete',seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,regular_fit_rows=n,opening_fit_rows=nopen,max_fit_target=str(cutoff),score_begin=str(first.loc[score,'datetime'].min()),regular_epoch0_error=error,layout='drop_negative80',monthly_frozen_transforms_reused=True)
            write()
            files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file()}
            (dest/'complete.json').write_text(json.dumps(files,indent=2))
            return
        fit_scale=json.loads((data.root/'manifest.json').read_text())['loss_scale']
        summaries=[]
        for arm in ['original_samples','include_opening_shared']:
            m=build_model(factory,payload['model'],7,training=True);asset=factory['asset']
            if arm=='include_opening_shared':
                with torch.no_grad():
                    m.reference.heads[asset].copy_(torch.from_numpy(parameters['head']));m.reference.six[asset].copy_(torch.from_numpy(parameters['six']));m.initial_weights[asset].copy_(torch.from_numpy(parameters['downside']));m.set_state_initializer(parameters['raw'],parameters['price'])
            if arm=='include_opening_shared':
                probe=np.r_[np.arange(min(256,n)),np.arange(n,n+min(64,nopen))]
                with torch.no_grad():actual=m(*batch(probe,True))[0].numpy()
                expected=ridge.predict({k:v[probe] for k,v in design.items()},np.concatenate(states)[probe],parameters,'drop_negative80')
                initializer_error=float(abs(actual-expected).max())
                if initializer_error>1e-8:raise ValueError(f'candidate original Ridge closure {initializer_error}')
                status['candidate_initializer_error']=initializer_error;write()
            opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=1e-5,foreach=False)
            folder=dest/arm;folder.mkdir();trainrows=n if arm=='original_samples' else n+nopen
            trainable={name:p.numel() for name,p in m.named_parameters() if p.requires_grad}
            (folder/'trainable.json').write_text(json.dumps(trainable,indent=2))
            # Original L and all forward formulas stay unchanged.
            torch.save({'model':m.state_dict(),'epoch':0,'factory_sha256':hashlib.sha256((bundle/a.symbol/'factory.pt').read_bytes()).hexdigest()},folder/'checkpoint_00.pt')
            for epoch in range(a.epochs+1):
                epoch_tick=time.monotonic()
                if epoch:
                    torch.manual_seed(7)
                    order=np.random.RandomState(7).permutation(trainrows);m.train()
                    if a.smoke:
                        order=order[:128]
                        if arm!='original_samples':order[-1]=n
                    for low in range(0,len(order),128):
                        ix=order[low:low+128];opt.zero_grad(set_to_none=True)
                        loss=m.loss(batch(ix,arm!='original_samples'),torch.tensor(targets[ix],dtype=torch.float64),fit_scale);loss.backward()
                        for p in m.parameters():
                            if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('nonfinite original gradient')
                        opt.step()
                        if low==0:(folder/'first_update.json').write_text(json.dumps({'status':'training_started','loss':float(loss.detach()),'optimizer_state_parameters':len(opt.state),'pid':os.getpid()}))
                        if low%6400==0:status.update(stage='train',arm=arm,epoch=epoch,updates=low//128+1,seconds=time.monotonic()-start);write()
                    torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch},folder/f'checkpoint_{epoch:02d}.pt')
                m.eval();normal=normal_score(m)
                with torch.no_grad():op=m(*oscore)[0].numpy()
                frame=data.frame[['datetime','target_datetime','y_true']].copy();frame['prediction']=normal;frame['opening']=False;frame=frame.loc[frame.datetime.ge(chosen.loc[scoreopen,'datetime'].min().normalize())]
                extra=chosen.loc[scoreopen,['datetime','target_datetime','y']].rename(columns={'y':'y_true'}).copy();extra['prediction']=op;extra['opening']=chosen.loc[scoreopen,'opening'].to_numpy()
                out=pd.concat([frame,extra]).sort_values('datetime')
                if out.datetime.duplicated().any():raise ValueError('duplicate scored clock')
                expected=raw.loc[raw.datetime.ge(out.datetime.min().normalize())&raw.datetime.le(data.frame.datetime.max())]
                if not a.smoke and len(out)!=len(expected):raise ValueError(f'incomplete physical-clock scoring {len(out)} vs {len(expected)}')
                replica=json.loads((a.source/'results/intraday_rv/production-benchmark-reproduction-20261008-v3/fits/direct_fullfit/all_rvol__level__common__ridge0/model.json').read_text())
                index=pd.Index(raw.datetime).get_indexer(out.datetime)
                if (index<0).any():raise ValueError('missing benchmark evaluation keys')
                out['clock_position']=raw.iloc[index]['_position'].to_numpy();out['physical_clock']=pd.to_datetime(out.datetime).dt.strftime('%H:%M:%S')
                out['benchmark_replica']=raw.iloc[index][replica['columns']].to_numpy(float)@np.array(replica['raw_slopes'])+replica['raw_intercepts'][0]
                out.to_parquet(folder/f'predictions_{epoch:02d}.parquet',index=False)
                if not np.isfinite(out.prediction).all() or not out.prediction.gt(0).all():raise ValueError('invalid original prediction')
                for group,mask in [('all',np.ones(len(out),bool)),('opening',out.opening.to_numpy()),('regular',~out.opening.to_numpy()),('first15_regular',out.clock_position.between(1,16).to_numpy()),('remaining',out.clock_position.gt(16).to_numpy())]:summaries.append({'arm':arm,'epoch':epoch,'group':group,**metric(out.y_true.to_numpy()[mask],out.prediction.to_numpy()[mask])})
                (folder/f'epoch_{epoch:02d}_timing.json').write_text(json.dumps({'seconds':time.monotonic()-epoch_tick,'training_rows':trainrows,'updates':0 if epoch==0 else (1 if a.smoke else (trainrows+127)//128)}))
                print(a.symbol,arm,epoch,'completed',flush=True)
        pd.DataFrame(summaries).to_csv(dest/'metrics.csv',index=False)
        (dest/'opening_flags.json').write_text(json.dumps(flags,indent=2))
        status.update(status='complete',stage='complete',seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,regular_fit_rows=n,opening_fit_rows=nopen,original_epoch0_error=error,max_fit_target=str(cutoff),first_score_origin=str(first.loc[score,'datetime'].min()),initializer='original Ridge code, same layout and frozen transforms; candidate includes opening',fit_audit=fit_audit)
        status['input_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,data.root/'manifest.json',data.control/'manifest.json',data.cache/'complete.json',bundle/'manifest.json',data.cache/'audit.json',data.cache/'fit_rows.npy',data.cache/'score_rows.npy',data.cache/'keys.parquet',a.source/'data/snapshots/notebook0-rvol-prod-intraday-multiscale-20260918-v1'/f'{a.symbol}.parquet']}
        write()
    except BaseException as exc:
        status.update(status='failed',error_type=type(exc).__name__,error=str(exc));write();raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--symbol',required=True);p.add_argument('--month',default='2026-08');p.add_argument('--epochs',type=int,default=1);p.add_argument('--smoke',action='store_true');p.add_argument('--prepare-only',action='store_true');run(p.parse_args())
