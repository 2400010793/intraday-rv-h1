"""Atomic, versioned local state. State files contain restricted historical data."""
import json,os
from pathlib import Path


def save_state(path,state):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with open(tmp,'w') as f:json.dump(state,f,allow_nan=False,separators=(',',':'));f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)


def load_state(path):
    d=json.loads(Path(path).read_text())
    if d.get('schema')!='h1-state-v1':raise ValueError('incompatible state schema')
    return d
