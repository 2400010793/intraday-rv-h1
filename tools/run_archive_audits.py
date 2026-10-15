"""Six bounded audit processes; model inference only, no training."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'artifacts/archive-full-v1';out.mkdir(parents=True,exist_ok=False)
(out/'status.json').write_text(json.dumps({'status':'running','workers':6,'timeout_per_symbol_seconds':900,'expected_seconds':300,'source':os.environ['H1_LEGACY_SOURCE'],'model_bundle':os.environ['H1_BUNDLE'],'fit_scope':'original frozen monthly Train+Validation with maturity gate','target':'H1 same physical clock','no_training':True,'seed_selection':False,'cancellation':'tmux kill-session -t h1-archive-full-v1 (also check descendants)'}))
def run(symbol):
 with (out/f'{symbol}.log').open('w') as log:
  result=subprocess.run([sys.executable,str(root/'tools/validate_archives.py'),'--source',os.environ['H1_LEGACY_SOURCE'],'--bundle',os.environ['H1_BUNDLE'],'--symbol',symbol,'--output',str(out)],stdout=log,stderr=subprocess.STDOUT,timeout=900)
 return {'symbol':symbol,'exit_code':result.returncode}
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:tasks=list(ex.map(run,['IF','IH','IC','IM','159915','588000']))
 (out/'status.json').write_text(json.dumps({'status':'passed' if all(t['exit_code']==0 for t in tasks) else 'failed','tasks':tasks},indent=2))
except Exception as exc:
 (out/'status.json').write_text(json.dumps({'status':'failed','error_type':type(exc).__name__,'error':str(exc)}));raise
