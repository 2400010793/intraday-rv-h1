"""An actual ETF current-price perturbation must reach its price-state feature."""
import copy,os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from intraday_rv_h1 import load_bundle,load_state


def test_etf_price_perturbation_updates_rank():
 base=Path(os.environ.get('H1_ASSETS',str(Path(__file__).resolve().parents[1]/'artifacts')))
 history=base/'full-replay-v1/159915_history.json';data=base/'test_input/159915.parquet'
 if not history.exists() or not data.exists():pytest.skip('local authorized ETF fixtures missing')
 state=load_state(history);row=pd.read_parquet(data).iloc[0].to_dict();engine=load_bundle(base/'models','159915','2026-08',state)
 engine.predict_step(row);before=engine.last_args[11][0,2].copy()
 engine.state=copy.deepcopy(state)
 # Extreme synthetic price forces a different reference rank without using a future row.
 price=float(row['spotRef']);engine.predict_step(dict(row,spotRef=price*100))
 after=engine.last_args[11][0,2].copy()
 if np.array_equal(before,after):
  engine.state=copy.deepcopy(state);engine.predict_step(dict(row,spotRef=price/100));after=engine.last_args[11][0,2].copy()
 assert not np.array_equal(before,after)
