"""Production-row boundary; future labels and predictions are not inputs."""
import math
import pandas as pd

SYMBOLS=('IF','IH','IC','IM','159915','588000')
COLUMNS=('usymbol','trade_date','datetime','ret','ttime','spotRef','rvol1D','rvol1hour')


def validate_row(row,symbol):
    missing=set(COLUMNS)-set(row)
    if missing:raise ValueError(f'missing production columns: {sorted(missing)}')
    supplied={k:row[k] for k in COLUMNS} # extra evaluation fields are never consumed
    if str(supplied['usymbol']).upper()!=symbol:raise ValueError('wrong symbol')
    dt=pd.Timestamp(supplied['datetime']);date=pd.Timestamp(supplied['trade_date'])
    if dt.tz is not None or date.tz is not None or date!=date.normalize() or dt.normalize()!=date:raise ValueError('require exchange-local naive datetime and matching trade_date')
    if date==pd.Timestamp('2025-02-04'):raise ValueError('excluded market date')
    supplied.update(datetime=dt.isoformat(),trade_date=date.isoformat(),usymbol=symbol)
    for k in COLUMNS[3:]:
        v=supplied[k];v=float('nan') if v is None else float(v)
        if math.isinf(v):raise ValueError(f'infinite {k}')
        if k=='ttime' and not math.isfinite(v):raise ValueError('finite ttime required')
        if k in ('spotRef','rvol1D','rvol1hour') and math.isfinite(v) and v<=0:raise ValueError(f'positive or missing {k} required')
        supplied[k]=None if math.isnan(v) else v
    return supplied
