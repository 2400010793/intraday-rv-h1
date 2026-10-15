"""Causal signed return windows used for vectorized history warmup."""
import numpy as np
import pandas as pd
from .seasonal_energy import KEY,_observations
NAMES = ('previous_day', 'opening15', 'rolling15')
EXPECTED_ROWS={"IF":243,"IH":243,"IC":243,"IM":243,"159915":240,"588000":240}

def causal_net_returns(observations, *, expected_rows=None):
    """Exclude overnight; opening15 starts AFTER the first regular observation.

    Rolling15 is an elapsed-time window, reset at lunch/missing-minute gaps.
    Daily return includes observed lunch/closing-auction intervals. Only a
    complete previous observed session is eligible; no older-day substitution.
    """
    f = _observations(observations)
    expected = EXPECTED_ROWS if expected_rows is None else expected_rows
    if not set(f.symbol).issubset(expected):
        raise ValueError('missing expected physical session row counts')
    day = ['symbol', 'date']
    intraday = f['return'].where(~f.opening)
    daily = f.assign(intraday=intraday).groupby(day, sort=False).agg(
        value=('intraday', 'sum'), finite=('intraday', 'count'),
        rows=('clock_position', 'size'), last=('datetime', 'max'))
    need = daily.index.get_level_values('symbol').map(expected).to_numpy()
    close = daily['last'].dt.hour.eq(15) & daily['last'].dt.minute.eq(0)
    complete = daily.rows.eq(need) & daily.finite.eq(need-1) & close
    daily['value'] = daily.value.where(complete)
    daily['previous_day'] = daily.groupby(level=0, sort=False).value.shift()
    out = f[[*KEY, 'datetime']].merge(daily[['previous_day']],
        left_on=day, right_index=True, how='left', validate='many_to_one', sort=False)

    # A gap starts a fresh support. First regular observation is an anchor,
    # even for ETFs where its return spans 09:25 -> 09:30 and is not one minute.
    boundary = ~f.continuous | f.clock_position.eq(1)
    segment = boundary.groupby([f.symbol, f.date], sort=False).cumsum()
    f['segment'] = segment
    support = [f.symbol, f.date, segment]
    anchor = f.datetime.groupby(support, sort=False).transform('first')
    usable = ~boundary & f['return'].notna()
    x = f.assign(window_return=f['return'].where(usable, 0.))
    roll = x.groupby([*day, 'segment'], sort=False).rolling(
        '15min', on='datetime', closed='right').window_return.sum()
    # Time-based grouped rolling indexes by datetime, not original integer row.
    if not np.array_equal(roll.index.get_level_values(-1), f.datetime):
        raise ValueError('grouped return window changed physical timestamp order')
    roll = pd.Series(roll.to_numpy(), index=f.index)
    missing = x.assign(missing=(~usable & ~boundary).astype(float)).groupby(
        [*day, 'segment'], sort=False).rolling('15min', on='datetime', closed='right').missing.sum()
    if not np.array_equal(missing.index.get_level_values(-1), f.datetime):
        raise ValueError('grouped missing-return window changed timestamp order')
    missing = pd.Series(missing.to_numpy(), index=f.index)
    valid = f.datetime.sub(anchor).ge(pd.Timedelta(minutes=15)) & missing.eq(0)
    out['rolling15'] = roll.where(valid).to_numpy()

    first = f.clock_position.eq(1)
    opening_anchor = f.datetime.where(first).groupby([f.symbol, f.date], sort=False).transform('first')
    opening_segment = segment.where(first).groupby([f.symbol, f.date], sort=False).transform('first')
    opening_endpoint = segment.eq(opening_segment) & f.datetime.ge(opening_anchor+pd.Timedelta(minutes=15))
    ends = f.loc[opening_endpoint, [*day, 'datetime']].groupby(day, sort=False).head(1)
    # Never replace an incomplete opening window with a later rolling window.
    ends = ends.loc[valid.loc[ends.index]]
    ends['opening15'] = roll.loc[ends.index].to_numpy()
    ends = ends.rename(columns={'datetime': 'opening_available_at'})
    o = out.merge(ends, on=day, how='left', validate='many_to_one', sort=False)
    out['opening15'] = o.opening15.where(o.datetime.ge(o.opening_available_at)).to_numpy()
    for name in NAMES:
        out[name+'_available'] = out[name].notna().astype(float)
        out[name] = out[name].fillna(0.)
    if not np.isfinite(out[[*NAMES, *(n+'_available' for n in NAMES)]]).all().all():
        raise FloatingPointError('nonfinite causal return window')
    return out
