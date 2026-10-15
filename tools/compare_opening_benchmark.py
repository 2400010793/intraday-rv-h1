"""Retrospective same-key report; no fitting or model selection on Test."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from opening_training_screen import metrics

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 orig=a.source/'results/intraday_rv';snap=a.source/'data/snapshots/notebook0-rvol-prod-intraday-multiscale-20260918-v1'
 coeff=json.loads((orig/'production-benchmark-reproduction-20261008-v3/fits/direct_fullfit/all_rvol__level__common__ridge0/model.json').read_text())
 archive=pd.read_parquet(orig/'notebook10-three-origin-optimizer-old-blend-20261014-v2/predictions.parquet');records=[];precision=[]
 for symbol in ['IF','IH','IC','IM','159915','588000']:
  source=pd.read_parquet(snap/f'{symbol}.parquet',columns=['datetime','trade_date','rvol1D','predRVol1D']+list(dict.fromkeys(c for c in coeff['columns'] if c!='rvol1D')));source['symbol']=symbol
  source['benchmark_replica']=source[coeff['columns']].to_numpy(float)@np.array(coeff['raw_slopes'])+coeff['raw_intercepts'][0]
  stored=archive.loc[archive.symbol.eq(symbol)].copy()
  merged=stored.merge(source[['datetime','benchmark_replica','predRVol1D']],on='datetime',validate='one_to_one')
  assert len(merged)==len(stored);np.testing.assert_allclose(merged.benchmark_prediction,merged.predRVol1D,rtol=0,atol=1e-10)
  for group,mask in [('first15_regular',merged.clock_position.between(1,16)),('remaining',merged.clock_position.gt(16)),('all_regular',np.ones(len(merged),bool))]:
   frame=merged.loc[mask]
   for model in ['shampoo_e5','adam_e6','benchmark_replica','benchmark_prediction']:
    records.append({'symbol':symbol,'scope':'archived_monthly_test','group':group,'model':model,**metrics(frame.y_true.to_numpy(),frame[model].to_numpy())})
  # Frozen-August, entire day comparison including new first clock.
  day=source.loc[source.trade_date.eq(pd.Timestamp('2026-09-16'))].sort_values('datetime').copy();target=source.loc[source.trade_date.eq(pd.Timestamp('2026-09-17'))].sort_values('datetime').copy()
  assert len(day)==len(target);day['clock_position']=np.arange(len(day));target['clock_position']=np.arange(len(target))
  day=day.merge(target[['datetime','clock_position','rvol1D']].rename(columns={'datetime':'target_datetime','rvol1D':'y_true'}),on='clock_position',validate='one_to_one')
  pred=pd.read_parquet(a.replay/f'{symbol}_predictions.parquet');pred.datetime=pd.to_datetime(pred.datetime);day=day.merge(pred,on=['symbol','datetime'],validate='one_to_one');assert len(day)==(243 if symbol in ['IF','IH','IC','IM'] else 240)
  for group,mask in [('opening',day.clock_position.eq(0)),('first15_regular',day.clock_position.between(1,16)),('remaining',day.clock_position.gt(16)),('all',np.ones(len(day),bool))]:
   f=day.loc[mask]
   for model in ['shampoo_e5','adam_e6','benchmark_replica','predRVol1D']:
    records.append({'symbol':symbol,'scope':'frozen_august_2026_09_16','group':group,'model':model,**metrics(f.y_true.to_numpy(),f[model].to_numpy())})
  precision.append({'symbol':symbol,'archive_rows':len(merged),'archive_database_rounding_max_difference':float(np.max(abs(merged.benchmark_prediction-merged.predRVol1D))),'database_vs_replica_max_difference':float(np.max(abs(merged.predRVol1D-merged.benchmark_replica)))})
 f=pd.DataFrame(records);f.to_csv(a.output/'by_symbol.csv',index=False)
 pooled=[]
 for (scope,group,model),g in f.groupby(['scope','group','model']):
  pooled.append({'scope':scope,'group':group,'model':model,'rows':int(g.rows.sum()),**{m:float(np.average(g[m],weights=g.rows)) for m in ['mse','mae','qlike']}})
 pd.DataFrame(pooled).to_csv(a.output/'pooled.csv',index=False)
 (a.output/'audit.json').write_text(json.dumps({'status':'passed','test_role':'retrospective diagnostic only; no epoch/feature/seed selection','benchmark':'fixed reproduced coefficients direct_fullfit/all_rvol__level__common__ridge0; database separate','precision':precision},indent=2))
 print(pd.DataFrame(pooled).to_string(index=False))
