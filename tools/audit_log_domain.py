"""Retrospective domain coverage; does not select a continuation using Test."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from intraday_rv_h1.features.native import raw_features
from intraday_rv_h1.data.schema import COLUMNS

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=[]
 for symbol in ['IF','IH','IC','IM','159915','588000']:
  raw=pd.read_parquet(a.input/f'{symbol}.parquet',columns=list(COLUMNS));raw=raw.loc[pd.to_datetime(raw.datetime).lt('2026-09-01')].sort_values('datetime');f=raw_features(raw,symbol)
  hist=f.datetime.le('2026-08-03 09:30:10');audit=f.datetime.gt('2026-08-03 09:30:10')&f.clock_position.gt(0)
  row={'symbol':symbol,'evaluation_rows':int(audit.sum())}
  for col in ['negative_variance','log_argument']:
   z=f.loc[hist,col].to_numpy(float);z=z[np.isfinite(z)&(z>0)];eps=float(np.quantile(z,.001));x=f.loc[audit,col].to_numpy(float)
   row[col]={'epsilon':eps,'nonpositive':int((x<=0).sum()),'below_epsilon':int((np.isfinite(x)&(x<eps)).sum()),'nonfinite':int((~np.isfinite(x)).sum())}
  out.append(row);print(json.dumps(row),flush=True)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'scope':'2026-08-03 09:30:10 < origin < 2026-09-01; physical opening excluded','rows':out,'selection':'carry retained; coverage is not an MSE/QLIKE/needle noninferiority test; alternative modes remain experimental and are not deployed'},indent=2))
