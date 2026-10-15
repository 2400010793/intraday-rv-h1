"""Paired target-session block intervals; exploratory Validation diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();frames=[]
 for source in sorted(a.run.glob('*_validation.parquet')):
  f=pd.read_parquet(source);f=f.loc[f.opening].copy();f['symbol']=source.name.split('_')[0];frames.append(f)
 f=pd.concat(frames,ignore_index=True);y=f.y.to_numpy();base=f.regular_only.to_numpy();rng=np.random.default_rng(20261015);summaries=[]
 for name in ['include_opening','opening_interactions','benchmark_replica_train']:
  pred=f[name].to_numpy();delta=(pred-y)**2-(base-y)**2
  daily=pd.Series(delta).groupby(pd.to_datetime(f.target_datetime).dt.normalize()).mean().to_numpy();bootstrap=daily[rng.integers(0,len(daily),size=(2000,len(daily)))].mean(1)
  summaries.append({'model':name,'opening_rows':len(f),'target_session_blocks':len(daily),'mse':float(np.mean((pred-y)**2)),'mse_change_vs_regular_only':float(np.mean(delta)),'mse_relative_improvement':float(-np.mean(delta)/np.mean((base-y)**2)),'paired_date_bootstrap_ci95':np.quantile(bootstrap,[.025,.975]).tolist()})
 (a.run/'opening_paired_diagnostics.json').write_text(json.dumps({'role':'exploratory Validation; shared calendar-date blocks preserve cross-asset dependence; no Test selection','bootstrap_replicates':2000,'results':summaries},indent=2));print(json.dumps(summaries,indent=2))
