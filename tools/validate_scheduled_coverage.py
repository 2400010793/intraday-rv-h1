"""Six-asset actual-event + one-minute scheduler closure against audited forecasts."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from intraday_rv_h1 import load_bundle
from intraday_rv_h1.inference.schedule import scheduled_replay


def run(root,output,symbol):
    tick=time.monotonic();assets=Path(os.environ.get('H1_ASSETS',str(root/'artifacts')))
    rows=pd.read_parquet(assets/f'test_input/{symbol}.parquet').to_dict('records')
    engine=load_bundle(assets/'models',symbol,'2026-08',assets/f'full-replay-v4/{symbol}_history.json')
    outputs=list(scheduled_replay(engine,rows,start='2026-09-16 09:25:00',end='2026-09-16 15:00:00'))
    f=pd.DataFrame(outputs);f['datetime']=pd.to_datetime(f.datetime);reference=pd.read_parquet(assets/f'full-replay-v4/{symbol}_predictions.parquet');reference.datetime=pd.to_datetime(reference.datetime)
    joined=f.merge(reference,on=['symbol','datetime'],suffixes=('_new','_reference'),validate='one_to_one')
    assert len(joined)==len(rows)
    maximum=0.
    for column in ['shampoo_e5','adam_e6','shampoo_e5_half0.5','adam_e6_half0.5']:
      assert np.isfinite(f[column]).all() and f[column].gt(0).all()
      maximum=max(maximum,float(np.max(abs(joined[column+'_new']-joined[column+'_reference']))))
    assert maximum<=1e-8
    assert not f.datetime.duplicated().any() and set(pd.date_range('2026-09-16 09:25','2026-09-16 15:00',freq='min')).issubset(set(f.datetime))
    assert len(engine.state['observed_today'])==len(rows)
    f.to_parquet(output/f'{symbol}_scheduled.parquet',index=False)
    return {'symbol':symbol,'status':'passed','requested_minute_ticks':336,'real_production_rows':len(rows),'total_unique_outputs':len(f),'fallback_outputs':int(f.prediction_mode.eq('fallback').sum()),'model_outputs':int(f.prediction_mode.eq('model').sum()),'maximum_original_prediction_error':maximum,'seconds':time.monotonic()-tick}

if __name__=='__main__':
 root=Path(__file__).resolve().parents[1];out=Path(os.environ['H1_COVERAGE_OUTPUT']);out.mkdir(parents=True,exist_ok=False)
 (out/'status.json').write_text(json.dumps({'status':'running','workers':6,'symbols':['IF','IH','IC','IM','159915','588000'],'target':'H1 frozen August; no fitting','history':'through 2026-09-15; inputs 2026-09-16 only','runtime_expected_seconds':120,'timeout_per_symbol':300,'cancellation':'tmux kill-session -t h1-scheduled-coverage-v1; check descendants'}))
 def child(symbol):
  # Run separate Python processes, not six competing in-process Torch calls.
  with (out/f'{symbol}.log').open('w') as log:
   p=subprocess.run([sys.executable,'-c','from pathlib import Path;import json;from tools.validate_scheduled_coverage import run;r=run(Path("."),Path('+repr(str(out))+'),'+repr(symbol)+');Path('+repr(str(out/f'{symbol}_status.json'))+').write_text(json.dumps(r,indent=2))'],stdout=log,stderr=subprocess.STDOUT,timeout=300)
  if p.returncode:raise RuntimeError(f'{symbol}: exit {p.returncode}')
  return json.loads((out/f'{symbol}_status.json').read_text())
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:results=list(ex.map(child,['IF','IH','IC','IM','159915','588000']))
  (out/'status.json').write_text(json.dumps({'status':'passed','results':results},indent=2))
 except Exception as exc:
  (out/'status.json').write_text(json.dumps({'status':'failed','error':str(exc)}));raise
