"""Original HARNet monthly shared-clock cache and full-epoch optimizer workers."""
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from intraday_rv_h1.models.loader import build_model
from .original_shared_inputs import OriginalInputs
from .original_shared_training import run as prepare_inputs
from .metrics import metric
from volatility_forecast.rv_predict.training.matrix_optimization import make_matrix_optimizer, matrix_partition
from volatility_forecast.rv_predict.experiments.notebook10_attention.epoch_ordering import shuffled_rows
from volatility_forecast.rv_predict.models.harnet import percentile_initialization_ablation as ridge


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def write(p,v):
    Path(p).write_text(json.dumps(v,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))

def threads():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)


def prepare_task(run,symbol,month):
    threads();run=Path(run);m=json.loads((run/'manifest.json').read_text());root=run/'cache'/month;dest=root/symbol
    if (dest/'complete.json').exists():
        for name,digest in json.loads((dest/'complete.json').read_text()).items():
            if sha(dest/name)!=digest:raise ValueError(f'cache hash mismatch {dest/name}')
        return json.loads((dest/'status.json').read_text())
    prepare_inputs(SimpleNamespace(source=Path(m['source']),bundle=Path(m['bundle']),output=root,symbol=symbol,month=month,epochs=0,smoke=False,prepare_only=True))
    return json.loads((dest/'status.json').read_text())


class Joined:
    def __init__(self,run,symbol,month):
        self.run=Path(run);self.manifest=json.loads((self.run/'manifest.json').read_text());self.symbol=symbol;self.month=month
        self.cache=self.run/'cache'/month/symbol
        for file,digest in json.loads((self.cache/'complete.json').read_text()).items():
            if sha(self.cache/file)!=digest:raise ValueError('cache changed')
        self.audit=json.loads((self.cache/'status.json').read_text())
        self.regular=OriginalInputs(self.manifest['source'],symbol,month)
        self.extras={s:[np.load(self.cache/f'{s}_extra_arg{j}.npy',mmap_mode='r') for j in range(13)] for s in ['fit','score']}
        self.n=len(self.regular.rows['fit']);self.nextra=len(self.extras['fit'][0]);self.nscore=len(self.regular.frame)
        self.target_values=np.r_[self.regular.target(np.arange(self.n)),np.load(self.cache/'fit_extra_targets.npy')]
        self.extra_frame=pd.read_parquet(self.cache/'extra_score_keys.parquet').rename(columns={'y':'y_true'})
    def batch(self,split,ix):
        if split=='score':return self.regular.batch('score',ix)
        normal=ix<self.n
        if normal.any():na=self.regular.batch('fit',ix[normal])
        result=[]
        for j in range(13):
            v=np.empty((len(ix),*self.extras['fit'][j].shape[1:]),dtype=self.extras['fit'][j].dtype)
            if normal.any():v[normal]=na[j].numpy()
            if (~normal).any():v[~normal]=self.extras['fit'][j][ix[~normal]-self.n]
            result.append(torch.from_numpy(v))
        return result
    def score(self,model):
        model.eval()
        with torch.no_grad():
            regular=np.concatenate([model(*self.batch('score',np.arange(i,min(i+512,self.nscore))))[0].numpy() for i in range(0,self.nscore,512)])
            extra=np.concatenate([model(*[torch.from_numpy(np.array(v[i:i+512],copy=True)) for v in self.extras['score']])[0].numpy() for i in range(0,len(self.extra_frame),512)])
        frame=self.regular.frame[['datetime','target_datetime','y_true','clock_position']].copy();frame['prediction']=regular;frame['opening']=False
        frame=frame.loc[frame.datetime.ge(pd.Timestamp(self.audit['score_begin']).normalize())]
        ex=self.extra_frame.copy();ex['prediction']=extra
        rawkeys=pd.read_parquet(Path(self.manifest['source'])/'data/snapshots/notebook0-rvol-prod-intraday-multiscale-20260918-v1'/f'{self.symbol}.parquet',columns=['datetime','trade_date'])
        rawkeys['clock_position']=rawkeys.groupby('trade_date',sort=False).cumcount()
        ix=pd.Index(rawkeys.datetime).get_indexer(ex.datetime);ex['clock_position']=rawkeys.clock_position.iloc[ix].to_numpy()
        out=pd.concat([frame,ex],ignore_index=True).sort_values('datetime').reset_index(drop=True)
        if out.datetime.duplicated().any() or not np.isfinite(out.prediction).all() or not out.prediction.gt(0).all():raise ValueError('invalid prediction keys/values')
        counts=out.groupby(out.datetime.dt.normalize()).size()
        expected=243 if self.symbol in ['IF','IH','IC','IM'] else 240
        if not counts.eq(expected).all():raise ValueError(f'incomplete clock coverage: {counts.to_dict()}')
        return out
    def model(self,seed):
        bundle=Path(self.manifest['bundle']);factory=torch.load(bundle/self.symbol/'factory.pt',map_location='cpu',weights_only=True)
        cp=self.regular.root/'tasks/adam'/self.month/f'{self.symbol}__seed{seed}'/'checkpoint_00.pt'
        payload=torch.load(cp,map_location='cpu',weights_only=False);model=build_model(factory,payload['model'],seed,training=True)
        with np.load(self.cache/'shared_initializer.npz') as p,torch.no_grad():
            asset=factory['asset'];model.reference.heads[asset].copy_(torch.from_numpy(p['head']));model.reference.six[asset].copy_(torch.from_numpy(p['six']));model.initial_weights[asset].copy_(torch.from_numpy(p['downside']));model.set_state_initializer(p['raw'],p['price'])
        return model,factory


def train_task(run,symbol,month,seed,optimizer,smoke_updates=None):
    threads();run=Path(run);folder=run/('preflight' if smoke_updates else 'tasks')/optimizer/month/f'{symbol}__seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();status={'status':'running','stage':'load','pid':os.getpid(),'symbol':symbol,'month':month,'seed':seed,'optimizer':optimizer};write(folder/'status.json',status)
    try:
        data=Joined(run,symbol,month);model,factory=data.model(seed)
        opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-5,foreach=False) if optimizer=='adam' else make_matrix_optimizer(model,dict(optimizer=optimizer,scope='attention',learning_rate=1e-5))
        write(folder/'optimizer_partition.json',matrix_partition(model,'attention')[2])
        # Refit algebra closure on original h, not another network or surrogate.
        probe=np.r_[np.arange(32),np.arange(data.n,data.n+min(16,data.nextra))];args=data.batch('fit',probe);model.eval()
        with torch.no_grad():
            actual,_,h,_=model(*args);q=model.own_percentiles(h,args[5],args[8]);q[:,1]=.5
            from intraday_rv_h1.models.archived.monthly_demo_attention import replace_policy
            _,_,effective=replace_policy(args[1],args[2],args[3],args[12])
            design={'h':h.numpy(),'clock':args[2][:,:3].numpy(),'rank':effective[:,:6,0].numpy(),'position':args[6].numpy(),'q':q.numpy()}
            with np.load(data.cache/'shared_initializer.npz') as p:expected=ridge.predict(design,args[11].numpy(),{k:p[k] for k in p.files},'drop_negative80')
        closure=float(abs(actual.numpy()-expected).max())
        if closure>1e-8:raise ValueError(f'original epoch0 closure {closure}')
        rows=len(data.target_values);epochs=1 if smoke_updates else 6;metrics=[];train_seconds=0.;updates=0
        scale=json.loads((data.regular.root/'manifest.json').read_text())['loss_scale']
        for epoch in range(epochs+1):
            if epoch:
                model.train();torch.manual_seed(seed+epoch)
                order=shuffled_rows(np.arange(rows),seed,factory['asset'],epoch)
                if smoke_updates:order=order[:smoke_updates*128];order[-1]=data.n
                tick=time.monotonic()
                for low in range(0,len(order),128):
                    ix=order[low:low+128];opt.zero_grad(set_to_none=True)
                    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad} if updates==0 else None
                    loss=model.loss(data.batch('fit',ix),torch.tensor(data.target_values[ix],dtype=torch.float64),scale);loss.backward()
                    for p in model.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('nonfinite gradient')
                    opt.step();updates+=1
                    if before is not None:
                        delta=max(float((p.detach()-before[n]).abs().max()) for n,p in model.named_parameters() if n in before)
                        if delta<=0:raise ValueError('optimizer did not change parameters')
                        write(folder/'first_update.json',{'status':'training_started','pid':os.getpid(),'loss':float(loss.detach()),'max_parameter_change':delta,'optimizer_state_parameters':len(opt.state)})
                    if updates%50==0 or updates==1:status.update(stage='train',epoch=epoch,updates=updates,seconds=time.monotonic()-start);write(folder/'status.json',status)
                train_seconds+=time.monotonic()-tick
            prediction=data.score(model);prediction.to_parquet(folder/f'predictions_{epoch:02d}.parquet',index=False)
            for group,mask in [('all',np.ones(len(prediction),bool)),('opening',prediction.opening.to_numpy()),('first15_regular',prediction.clock_position.between(1,16).to_numpy()),('remaining',prediction.clock_position.gt(16).to_numpy())]:metrics.append({'epoch':epoch,'group':group,**metric(prediction.y_true.to_numpy()[mask],prediction.prediction.to_numpy()[mask])})
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'updates':updates,'symbol':symbol,'month':month,'seed':seed,'optimizer_name':optimizer,'cache_sha256':sha(data.cache/'complete.json')},folder/f'checkpoint_{epoch:02d}.pt')
            pd.DataFrame(metrics).to_csv(folder/'metrics.csv',index=False)
            print(symbol,month,seed,optimizer,epoch,'complete',flush=True)
        # Strict state_dict reload must reproduce final forecast exactly.
        replay,_=data.model(seed);payload=torch.load(folder/f'checkpoint_{epochs:02d}.pt',map_location='cpu',weights_only=False);replay.load_state_dict(payload['model'],strict=True)
        rerun=data.score(replay);np.testing.assert_array_equal(rerun.prediction,prediction.prediction)
        status.update(status='complete',stage='complete',seconds=time.monotonic()-start,train_seconds=train_seconds,updates=updates,fit_rows=rows,score_rows=len(prediction),epochs=epochs,epoch0_error=closure,reload_max_error=0.,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(folder/'status.json',status);write(folder/'complete.json',{p.name:sha(p) for p in folder.iterdir() if p.is_file()})
        return status
    except BaseException as exc:
        status.update(status='failed',error_type=type(exc).__name__,error=str(exc));write(folder/'status.json',status);raise
