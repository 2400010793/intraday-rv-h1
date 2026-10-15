"""Driver sends production rows through a pipe into a filesystem-isolated process.
The bwrap child cannot access this driver, research checkout or input-day tables.
"""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
import pandas as pd

CHILD='''import json,sys
from pathlib import Path
from intraday_rv_h1 import load_bundle
assert not Path('/public_space_2/chenxy/volatility').exists()
assert not Path('/repo').exists()
e=load_bundle('/bundle',sys.argv[1],'2026-08','/history.json')
for line in sys.stdin:
 print(json.dumps(e.predict_step(json.loads(line)),allow_nan=False),flush=True)
'''

def run(root,symbol,output):
 tick=time.monotonic()
 cmd=['bwrap','--die-with-parent','--unshare-net','--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64','--ro-bind',str(root/'.venv'),'/env','--ro-bind',str(Path.home()/'.local/lib/python3.9/site-packages'),'/deps','--ro-bind',str(root/'artifacts/models'),'/bundle','--ro-bind',str(root/f"{os.environ.get('H1_REFERENCE', 'artifacts/full-replay-v1')}/{symbol}_history.json"),'/history.json','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--chdir','/tmp','--setenv','PYTHONPATH','/env/lib/python3.9/site-packages:/deps','/env/bin/python','-c',CHILD,symbol]
 frame=pd.read_parquet(root/f'artifacts/test_input/{symbol}.parquet');records=[]
 for r in frame.to_dict('records'):
  for k in ['datetime','trade_date']:r[k]=pd.Timestamp(r[k]).isoformat()
  records.append(json.dumps(r,allow_nan=False))
 proc=subprocess.run(cmd,input='\n'.join(records)+'\n',text=True,capture_output=True,timeout=600,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1'))
 (output/f'{symbol}.stderr.log').write_text(proc.stderr)
 if proc.returncode:raise RuntimeError(f'{symbol}: isolation exit {proc.returncode}: {proc.stderr[-1000:]}')
 actual=[json.loads(x) for x in proc.stdout.splitlines()];ref=pd.read_parquet(root/f"{os.environ.get('H1_REFERENCE', 'artifacts/full-replay-v1')}/{symbol}_predictions.parquet")
 assert len(actual)==len(frame)
 for model in ['shampoo_e5','adam_e6','shampoo_e5_half0.5','adam_e6_half0.5']:
  assert list(ref[model])==[r[model] for r in actual],model
 return {'symbol':symbol,'status':'passed','rows':len(actual),'max_error':0,'seconds':time.monotonic()-tick,'isolation':'bwrap: research checkout and full input tables not mounted; network disabled'}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[1];a.output.mkdir(parents=True,exist_ok=True)
 (a.output/'status.json').write_text(json.dumps({'status':'running','workers':2,'timeout_seconds_per_symbol':600}))
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
   results=list(ex.map(lambda s:run(root,s,a.output),['IF','IH','IC','IM','159915','588000']))
  (a.output/'status.json').write_text(json.dumps({'status':'passed','results':results},indent=2))
 except Exception as exc:
  (a.output/'status.json').write_text(json.dumps({'status':'failed','error':str(exc)}));raise
