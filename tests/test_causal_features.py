import numpy as np
import pandas as pd
import pytest
from intraday_rv_h1.features.native import raw_features,extended_log
from intraday_rv_h1.data.schema import validate_row


def synthetic():
    rng=np.random.default_rng(19);dates=pd.bdate_range('2026-08-03',periods=4);frames=[]
    for i,date in enumerate(dates):
        times=[date+pd.Timedelta(hours=9,minutes=29)]+list(pd.date_range(date+pd.Timedelta(hours=9,minutes=30,seconds=10),periods=100,freq='min'))
        r=rng.normal(0,.0004,len(times));n=len(times)
        frames.append(pd.DataFrame({'usymbol':'IF','trade_date':date,'datetime':times,'ret':r,'ttime':i*2+np.arange(n)/243,'spotRef':4000*np.exp(r.cumsum()),'rvol1D':.2,'rvol1hour':.15}))
    return pd.concat(frames,ignore_index=True)


def test_native_prefix_and_future_mutation():
    f=synthetic();full=raw_features(f,'IF');end=len(f)-35
    short=raw_features(f.iloc[:end+1],'IF')
    mutated=f.copy();mutated.loc[end+1:,'ret']=.9;mutated.loc[end+1:,'rvol1D']=99
    changed=raw_features(mutated,'IF')
    cols=['core0','core1','core2','policy0','policy1','aux2','aux3','aux4','aux5']
    np.testing.assert_allclose(full.loc[end,cols].to_numpy(float),short.loc[end,cols].to_numpy(float),rtol=0,atol=1e-12,equal_nan=True)
    np.testing.assert_array_equal(full.loc[:end,cols],changed.loc[:end,cols])


def test_extension_is_continuous_and_finite_for_negative_input():
    eps=.02
    for mode in ['tangent','asinh']:
        assert extended_log(eps,eps,mode)==np.log(eps)
        assert np.isfinite(extended_log(-1.,eps,mode))
        h=1e-8
        assert abs((extended_log(eps+h,eps,mode)-extended_log(eps-h,eps,mode))/(2*h)-1/eps)<1e-3
    assert abs(extended_log(-1,eps,'asinh'))<abs(extended_log(-1,eps,'tangent'))


def test_schema_does_not_consume_labels_or_benchmark():
    row=synthetic().iloc[0].to_dict();one=validate_row(row,'IF')
    row.update(y_true=99,predRVol1D=999,target_datetime='2099-01-01')
    assert validate_row(row,'IF')==one
    row['datetime']='2026-08-04 09:29:00'
    with pytest.raises(ValueError,match='matching trade_date'):validate_row(row,'IF')
