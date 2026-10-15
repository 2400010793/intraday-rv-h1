import numpy as np
import pandas as pd
import pytest
from intraday_rv_h1.evaluation.metrics import evaluate


def test_evaluation_exact_keys_and_positive_predictions():
 p=pd.DataFrame({'symbol':['IF','IF'],'datetime':pd.to_datetime(['2026-08-03 10:00','2026-08-03 10:01']),'p':[.1,.2]})
 t=p[['symbol','datetime']].assign(y_true=[.1,.1],current_Rvol1D=.1)
 r=evaluate(p,t,.1,prediction_columns=['p']).iloc[0]
 assert abs(r.mse-.005)<1e-15 and r.eligible_links==1
 with pytest.raises(ValueError,match='missing exact targets'):evaluate(p,t.iloc[:1],.1,prediction_columns=['p'])
 p.loc[0,'p']=np.nan
 with pytest.raises(ValueError,match='invalid prediction'):evaluate(p,t,.1,prediction_columns=['p'])
