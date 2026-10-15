"""Real-model boundary checks, using the same local fixture as streaming tests."""
import copy
import pandas as pd
import pytest
from test_real_stream import assets,stream


def test_lunch_and_day_reset(stream):
    e,rows=stream
    # A sparse received feed is explicit: gaps reset smoothing and are flagged.
    e.predict_step(rows[0])
    noon=next(r for r in rows if pd.Timestamp(r['datetime']).strftime('%H:%M')=='13:00')
    out=e.predict_step(noon)
    assert out['shampoo_e5_half0.5']==out['shampoo_e5']
    assert 'physical_clock_gap' in out['flags']
    tomorrow=dict(rows[0],datetime='2026-09-17 09:29:00',trade_date='2026-09-17')
    out=e.predict_step(tomorrow)
    assert out['shampoo_e5_half0.5']==out['shampoo_e5']
    # Yesterday's acknowledged row is still idempotent after a day transition.
    saved=e.state.copy()
    assert e.predict_step(rows[0])['datetime']==pd.Timestamp(rows[0]['datetime']).isoformat()
    assert e.state==saved


def test_real_illegal_log_is_explicitly_extended(stream):
    e,rows=stream;e.predict_step(rows[0])
    # Force invalid RV log on a non-opening row without dropping its clock.
    out=e.predict_step(dict(rows[1],rvol1D=None))
    assert 'core0_historical_extension' in out['flags']
    assert out['max_consumed_datetime']==pd.Timestamp(rows[1]['datetime']).isoformat()
