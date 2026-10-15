"""Caller-owned chronological schedule driver; no market-feature generation."""
import pandas as pd


def scheduled_replay(engine,rows,*,start,end,frequency='1min'):
    """Yield every observed event and requested tick. A real event wins a tie.

    This is an offline driver and can hold input rows; the Predictor receives
    only the current row or timestamp. Scheduled gaps never create fake returns.
    """
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    if start.tz is not None or end.tz is not None or start>end:raise ValueError('invalid naive schedule bounds')
    events={}
    for row in rows:
        stamp=pd.Timestamp(row['datetime'])
        if stamp in events:raise ValueError('duplicate driver input timestamp')
        if start<=stamp<=end:events[stamp]=row
    ticks=pd.date_range(start,end,freq=frequency)
    if not len(ticks):raise ValueError('empty schedule')
    for stamp in sorted(set(ticks)|set(events)):
        yield engine.predict_step(events[stamp]) if stamp in events else engine.predict_at(stamp)
