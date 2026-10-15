"""Local restricted-data integration; portable source-only checkouts skip explicitly."""
import builtins,copy,json,os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from intraday_rv_h1 import load_bundle,load_state

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def assets():
    base=Path(os.environ.get('H1_ASSETS',str(ROOT/'artifacts')))
    state=base/'probe-v3/IF_history.json'
    if not state.exists():pytest.skip('local H1 state asset missing; set H1_ASSETS')
    s=load_state(state);e=load_bundle(base/'models','IF','2026-08',s)
    path=base/'test_input/IF.parquet'
    if not path.exists():pytest.skip('local IF production-row fixture missing')
    rows=pd.read_parquet(path).to_dict('records')
    return e,s,rows

@pytest.fixture
def stream(assets):
    e,s,rows=assets;e.state=copy.deepcopy(s);e.state['weight_version']=e.version
    return e,rows


def test_prefix_no_io_and_labels_ignored(stream,monkeypatch):
    e,rows=stream;s=copy.deepcopy(e.state);first=[]
    # No file may be opened during a step, after loading weights/state.
    with monkeypatch.context() as m:
        m.setattr(builtins,'open',lambda *a,**k:(_ for _ in ()).throw(AssertionError('step attempted file IO')))
        for row in rows[:3]:first.append(e.predict_step(row))
    e.state=s
    second=[]
    for row in rows[:3]:second.append(e.predict_step(dict(row,y_true=12345,predRVol1D=.00001)))
    assert first==second


def test_retry_conflict_and_out_of_order(stream):
    e,rows=stream;one=e.predict_step(rows[0]);before=copy.deepcopy(e.state)
    assert e.predict_step(rows[0])==one and e.state==before
    changed=dict(rows[0],ret=float(rows[0]['ret'])+.01)
    with pytest.raises(ValueError,match='different production row'):e.predict_step(changed)
    bad=dict(rows[0],datetime='2026-09-15 09:29:00',trade_date='2026-09-15')
    with pytest.raises(ValueError,match='out-of-order'):e.predict_step(bad)


def test_state_resume_exact(stream,tmp_path):
    e,rows=stream;e.predict_step(rows[0]);e.save_state(tmp_path/'state.json');s=load_state(tmp_path/'state.json')
    expected=e.predict_step(rows[1]);e.state=s
    assert e.predict_step(rows[1])==expected


def test_current_return_updates_features(stream):
    e,rows=stream;e.predict_step(rows[0]);s=copy.deepcopy(e.state)
    e.predict_step(rows[1]);original=e.last_args[12].copy();e.state=s
    e.predict_step(dict(rows[1],ret=float(rows[1]['ret'])+.01))
    assert not np.array_equal(e.last_args[12],original)


def test_missing_values_are_not_dropped(stream):
    e,rows=stream;e.predict_step(rows[0]);out=e.predict_step(dict(rows[1],ret=None,rvol1D=None,rvol1hour=None,spotRef=None))
    assert np.isfinite(out['shampoo_e5']) and out['shampoo_e5']>0
    assert out['flags']


def test_extension_failure_rolls_back_state(stream):
    e,rows=stream;s=copy.deepcopy(e.state)
    with pytest.raises(ValueError):e.predict_step(dict(rows[0],ttime=None))
    assert e.state==s
