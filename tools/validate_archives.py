"""Offline all-month, all-seed archival closure; not part of live inference.
Uses original tensor inputs as a differential fixture, never as deployment data.
"""
import argparse,gc,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from intraday_rv_h1.models.loader import build_model

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--symbol',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--months',default='');p.add_argument('--limit',type=int,default=0);a=p.parse_args();torch.set_num_threads(1)
 sys.path.insert(0,str(a.source));from volatility_forecast.rv_predict.experiments.notebook10_attention.monthly_demo_training import Inputs
 run=a.source/'results/intraday_rv/notebook10-onds-three-origin-monthly-adam-shampoo-e6-20261014-v1';top=json.loads((a.bundle/'manifest.json').read_text());factory=torch.load(a.bundle/a.symbol/'factory.pt',map_location='cpu',weights_only=True)
 a.output.mkdir(parents=True,exist_ok=True);tick=time.monotonic();entries=[]
 months=a.months.split(',') if a.months else top['months']
 statuspath=a.output/f'{a.symbol}_status.json'
 try:
  for month in months:
   data=Inputs(run,a.symbol,month,'onds15_30_three__drop_negative80');manifest=json.loads((a.bundle/a.symbol/month/'manifest.json').read_text())
   assert pd.Timestamp(manifest['max_fit_target_datetime'])<=pd.to_datetime(data.frame.datetime).min()
   count=min(a.limit,len(data.frame)) if a.limit else len(data.frame)
   for entry in manifest['weights']:
    weight=a.bundle/a.symbol/month/entry['path'];assert hashlib.sha256(weight.read_bytes()).hexdigest()==entry['sha256']
    model=build_model(factory,torch.load(weight,map_location='cpu',weights_only=True),entry['seed'])
    archive=run/'tasks'/entry['optimizer']/month/f"{a.symbol}__seed{entry['seed']}"/'predictions.npy';expected=np.load(archive,mmap_mode='r')[entry['epoch']]
    assert len(expected)==len(data.frame);maximum=0.
    with torch.no_grad():
     for low in range(0,count,512):
      hi=min(low+512,count);got=model(*data.batch('score',np.arange(low,hi)))[0].numpy();maximum=max(maximum,float(np.max(np.abs(got-expected[low:hi]))))
    if not np.isfinite(maximum) or maximum>1e-8:raise AssertionError(f'{month} {entry["path"]} archive difference {maximum}')
    entries.append({'month':month,'optimizer':entry['optimizer'],'epoch':entry['epoch'],'seed':entry['seed'],'rows':count,'maximum_error':maximum})
    del model;gc.collect()
   statuspath.write_text(json.dumps({'status':'running','symbol':a.symbol,'completed_weights':len(entries),'seconds':time.monotonic()-tick}))
  result={'status':'passed','symbol':a.symbol,'weights':len(entries),'full_rows':not bool(a.limit),'prediction_values':sum(e['rows'] for e in entries),'maximum_error':max(e['maximum_error'] for e in entries),'seconds':time.monotonic()-tick,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'entries':entries}
  statuspath.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='entries'}),flush=True)
 except Exception as exc:
  statuspath.write_text(json.dumps({'status':'failed','symbol':a.symbol,'error_type':type(exc).__name__,'error':str(exc),'completed_weights':len(entries)}));raise
