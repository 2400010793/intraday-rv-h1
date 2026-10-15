"""Local research-side differential audit; never imported by live inference."""
import argparse,copy,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from intraday_rv_h1.state.bootstrap import bootstrap
from intraday_rv_h1.inference.engine import load_bundle
from intraday_rv_h1.state.storage import save_state
from intraday_rv_h1.data.schema import COLUMNS

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--symbol',default='IF');p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=0);a=p.parse_args();torch.set_num_threads(1)
 sys.path.insert(0,str(a.source));from volatility_forecast.rv_predict.experiments.notebook10_attention import monthly_demo_training as legacy
 from volatility_forecast.rv_predict.features.intraday.onds_three_origin import policy_frame,policy_values
 from volatility_forecast.rv_predict.features.intraday.short_policy_estimators import normalize_policy
 symbol=a.symbol;a.output.mkdir(parents=True,exist_ok=True)
 (a.output/f'{symbol}_status.json').write_text(json.dumps({'status':'running'}))
 raw=pd.read_parquet(a.source/'data/snapshots/notebook0-rvol-prod-intraday-multiscale-20260918-v1'/f'{symbol}.parquet');raw=raw.loc[raw.trade_date.ne(pd.Timestamp('2025-02-04'))].sort_values('datetime').reset_index(drop=True)
 raw['usymbol']=symbol;registry=json.loads((a.bundle/symbol/'clock_registry.json').read_text());tick=time.monotonic()
 state=bootstrap(raw.loc[raw.datetime.lt('2026-09-16')],symbol,registry,before='2026-09-16');save_state(a.output/f'{symbol}_history.json',state)
 engine=load_bundle(a.bundle,symbol,'2026-08',state)
 folder=legacy.CONTROL/'source_cache'/symbol/'test';meta=pd.read_parquet(folder/'meta.parquet');arrays=[np.load(folder/f'arg{j}.npy',mmap_mode='r') for j in range(10)]
 states=[np.load(legacy.CONTROL/'states'/symbol/f'test_{name}.npy',mmap_mode='r') for name in ['return','coefficient']]
 index=pd.Index(meta.datetime);day=raw.loc[raw.trade_date.eq(pd.Timestamp('2026-09-16')),list(COLUMNS)]
 day=day.iloc[:a.limit] if a.limit else day
 records=[];diffs=[];prediction_diffs=[]
 policy_all,policy_flags=policy_values(policy_frame(raw),'onds15_30_three')
 source_index=pd.Index(raw.datetime)
 for k,row in enumerate(day.to_dict('records')):
  result=engine.predict_step(row);records.append(result);i=index.get_indexer([pd.Timestamp(row['datetime'])])[0]
  if i>=0:
   for j in range(12):
    ref=arrays[j][i] if j<10 else states[j-10][i]
    got=engine.last_args[j][0]
    if j==1:got,ref=got[2:],ref[2:]
    if j==2:got,ref=got[9:],ref[9:]
    if j==3:got,ref=got[2:6],ref[2:6]
    error=float(np.max(np.abs(got-ref)))
    if j==10:
     slot=1 if symbol in ['IF','IH','IC','IM'] else 3;error=float(np.max(np.abs(got[slot]-ref[slot])))
    diffs.append({'datetime':str(row['datetime']),'arg':j,'error':error})
   args=[torch.from_numpy(np.array((arrays[j][i] if j<10 else states[j-10][i])[None],copy=True,order='C')) for j in range(12)]
   pi=source_index.get_loc(pd.Timestamp(row['datetime']));pol=np.zeros((1,2,3))
   for j in range(2):
    if policy_flags[pi,j]:
     value=(policy_all[pi,j]-engine.normalization[j]['mean'])/engine.normalization[j]['std']
     ordered=engine.policy_ecdf[f'{j}__{meta.clock_slot_id.iloc[i]}'];lo=np.searchsorted(ordered,value,'left');hi=np.searchsorted(ordered,value,'right');rank=2*((lo+.5*(hi-lo)+.5)/(len(ordered)+1))-1
     pol[0,j]=[value,rank,1.]
   args.append(torch.from_numpy(pol))
   with torch.no_grad():
    for opt,models in engine.models.items():
     expected=float(np.mean([model(*args)[0].item() for model in models]));name='shampoo_e5' if opt=='shampoo_adam_graft' else 'adam_e6'
     prediction_diffs.append({'datetime':str(row['datetime']),'model':name,'absolute_error':abs(expected-result[name])})
  if k in [0,1,20,120]:print(symbol,k,'flags',result['flags'],'pred',result['shampoo_e5'],flush=True)
 pd.DataFrame(records).to_parquet(a.output/f'{symbol}_predictions.parquet',index=False)
 d=pd.DataFrame(diffs);d.to_csv(a.output/f'{symbol}_input_diff.csv',index=False);print(d.groupby('arg').error.agg(['max','mean']).to_string(),flush=True)
 save_state(a.output/f'{symbol}_end.json',engine.state)
 pd.DataFrame(prediction_diffs).to_csv(a.output/f'{symbol}_prediction_diff.csv',index=False)
 maximum=max(x['absolute_error'] for x in prediction_diffs)
 status={'status':'passed' if maximum<=1e-8 else 'failed','symbol':symbol,'rows':len(records),'expected_rows':243 if symbol in ['IF','IH','IC','IM'] else 240,'max_original_prediction_error':maximum,'seconds':time.monotonic()-tick,'flagged_rows':sum(bool(x['flags']) for x in records)}
 (a.output/f'{symbol}_status.json').write_text(json.dumps(status,indent=2));print(status,flush=True)
 if maximum>1e-8:raise AssertionError(f'original prediction closure failed: {maximum}')
