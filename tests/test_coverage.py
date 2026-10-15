import copy
import pandas as pd
from test_real_stream import assets,stream


def test_missing_ttime_and_missing_column_return_fallback(stream):
    e,rows=stream;s=copy.deepcopy(e.state)
    result=e.predict_step({k:v for k,v in rows[0].items() if k not in ('ttime','ret')})
    assert result['prediction_mode']=='fallback' and result['shampoo_e5']>0
    assert 'missing_ttime_fallback' in result['flags']
    assert e.state['raw_rows']==s['raw_rows'] and e.state['histories']==s['histories']
    assert result['max_observed_datetime']==s['last_datetime']


def test_new_clock_has_output_and_no_synthetic_feature_row(stream):
    e,rows=stream;e.predict_step(rows[0]);s=copy.deepcopy(e.state)
    result=e.predict_at('2026-09-16 09:29:30')
    assert result['prediction_mode']=='fallback'
    assert result['shampoo_e5']==s['last_predictions']['shampoo_e5']
    assert result['stale_seconds']==30
    assert len(e.state['observed_today'])==len(s['observed_today'])
    assert e.predict_at('2026-09-16 09:29:30')==result
    normal=e.predict_step(rows[1])
    assert normal['prediction_mode']=='model'
    assert e.state['observed_today'][-1]['clock_position']==1


def test_invalid_rv_numeric_is_explicit_missing(stream):
    e,rows=stream;out=e.predict_step(dict(rows[0],rvol1D=0,spotRef=float('inf')))
    assert out['shampoo_e5']>0 and 'invalid_rvol1D' in out['flags']
    assert 'invalid_spotRef' in out['flags']


def test_scheduled_all_minutes_coverage_and_state_resume(stream,tmp_path):
    e,rows=stream
    stamps=pd.date_range('2026-09-16 09:25:00','2026-09-16 15:00:00',freq='min')
    # Deliberately includes pre-open, lunch and closing-auction minutes.
    outputs=[e.predict_at(t) for t in stamps[:150]]
    e.save_state(tmp_path/'state.json');state=copy.deepcopy(e.state)
    expected=e.predict_at(stamps[150]);e.state=state
    assert e.predict_at(stamps[150])==expected
    outputs.extend([expected]+[e.predict_at(t) for t in stamps[151:]])
    assert len(outputs)==len(stamps) and all(x['shampoo_e5']>0 for x in outputs)
    assert not e.state['observed_today']


def test_numeric_feature_failure_returns_marked_fallback(stream,monkeypatch):
    e,rows=stream;s=copy.deepcopy(e.state)
    def fail(*args):raise FloatingPointError('simulated feature overflow')
    monkeypatch.setattr(e,'_features',fail)
    output=e.predict_step(rows[0])
    assert 'numeric_failure_fallback' in output['flags']
    assert output['prediction_mode']=='fallback' and e.state['raw_rows']==s['raw_rows']
    assert e.predict_step(rows[0])==output


def test_schedule_real_event_wins_and_updates_features(stream):
    from intraday_rv_h1.inference.schedule import scheduled_replay
    e,rows=stream
    out=list(scheduled_replay(e,rows[:2],start='2026-09-16 09:28',end='2026-09-16 09:31'))
    assert len(out)==5
    bytime={x['datetime']:x for x in out}
    assert bytime['2026-09-16T09:29:00']['prediction_mode']=='model'
    assert bytime['2026-09-16T09:30:10']['prediction_mode']=='model'
    assert bytime['2026-09-16T09:30:00']['prediction_mode']=='fallback'
    assert len(e.state['observed_today'])==2


def test_scheduled_ticks_do_not_bridge_lunch_smoothing(stream):
    e,rows=stream
    e.predict_step(rows[0])
    # Even a tick immediately before afternoon reopening cannot make the
    # carried morning output a fresh observation for smoothing.
    e.predict_at('2026-09-16 12:59:30')
    after=next(r for r in rows if pd.Timestamp(r['datetime']).strftime('%H:%M')=='13:00')
    out=e.predict_step(after)
    assert out['shampoo_e5_half0.5']==out['shampoo_e5']
