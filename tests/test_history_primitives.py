import numpy as np
import pandas as pd
from intraday_rv_h1.features.return_percentile import causal_net_returns


def test_opening15_only_released_after_window():
    stamps=pd.DatetimeIndex(['2026-08-03 09:29:00']+list(pd.date_range('2026-08-03 09:30:00',periods=17,freq='min').astype(str)))
    f=pd.DataFrame({'symbol':'IF','date':pd.Timestamp('2026-08-03'),'clock_position':np.arange(len(stamps)),'datetime':stamps,'return':.001})
    out=causal_net_returns(f)
    assert not out.loc[:15,'opening15_available'].any()
    assert out.loc[16,'opening15_available']==1
    assert abs(out.loc[16,'opening15']-.015)<1e-12
