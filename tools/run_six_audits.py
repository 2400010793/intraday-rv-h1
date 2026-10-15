"""Bounded two-process migration validation; logs/status persist on failure."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];output=root/'artifacts/full-replay-v1';output.mkdir(parents=True,exist_ok=False)
# Each child reads one symbol; no shared full-panel copies.
symbols=('IF','IH','IC','IM','159915','588000')
source=os.environ['H1_LEGACY_SOURCE'];bundle=os.environ['H1_BUNDLE']
(output/'status.json').write_text(json.dumps({'status':'running','workers':2,'stage':'six_asset_replay','started':time.time()}))
def run(s):
 with open(output/f'{s}.log','w') as log:
  r=subprocess.run([sys.executable,str(root/'tools/validate_migration.py'),'--source',source,'--bundle',bundle,'--output',str(output),'--symbol',s],stdout=log,stderr=subprocess.STDOUT,timeout=600)
 return {'symbol':s,'exit_code':r.returncode}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=list(ex.map(run,symbols))
(output/'status.json').write_text(json.dumps({'status':'passed' if all(x['exit_code']==0 for x in results) else 'failed','tasks':results},indent=2))
