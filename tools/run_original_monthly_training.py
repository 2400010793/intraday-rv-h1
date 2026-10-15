"""Thin immutable pipeline adapter with descendant-aware wall-clock timeout."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run.resolve()
    manifest=json.loads((root/'manifest.json').read_text());env=os.environ.copy();capture=root/'source_snapshot'
    env['PYTHONPATH']=str(capture/'src')+os.pathsep+str(capture/'legacy')
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','TORCH_NUM_THREADS']:env[key]='1'
    child=subprocess.Popen([sys.executable,'-m','intraday_rv_h1.research.original_monthly_pipeline','run','--run',str(root)],env=env,start_new_session=True)
    (root/'supervisor.json').write_text(json.dumps({'pid':os.getpid(),'pipeline_pid':child.pid,'hard_timeout_seconds':manifest['hard_timeout_seconds']},indent=2))
    try:
        code=child.wait(timeout=manifest['hard_timeout_seconds'])
    except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        m=json.loads((root/'manifest.json').read_text());m.update(status='aborted',error_type=type(exc).__name__,error='supervisor timeout or user cancellation; process group terminated');(root/'manifest.json').write_text(json.dumps(m,indent=2));raise
    raise SystemExit(code)
