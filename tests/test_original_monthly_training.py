"""Frozen-cache integrity and separate monthly optimizer protocol gates."""
import hashlib
import json

import numpy as np
import pytest

from intraday_rv_h1.research.metrics import metric
from intraday_rv_h1.research.original_monthly_tasks import prepare_task
from intraday_rv_h1.research.original_monthly_pipeline import SYMBOLS, SEEDS, OPTIMIZERS


def test_verified_cache_resume_rejects_changed_bytes(tmp_path):
    (tmp_path/'manifest.json').write_text('{}')
    cache=tmp_path/'cache'/'2026-08'/'IF'
    cache.mkdir(parents=True)
    payload=cache/'input.npy'
    payload.write_bytes(b'original input')
    status={'status':'complete','symbol':'IF'}
    (cache/'status.json').write_text(json.dumps(status))
    (cache/'complete.json').write_text(json.dumps({'input.npy':hashlib.sha256(payload.read_bytes()).hexdigest()}))
    assert prepare_task(tmp_path,'IF','2026-08')==status
    payload.write_bytes(b'changed input')
    with pytest.raises(ValueError,match='cache hash mismatch'):
        prepare_task(tmp_path,'IF','2026-08')


def test_fixed_protocol_and_metric_units():
    assert len(SYMBOLS)==6 and len(SEEDS)==5
    assert OPTIMIZERS==['adam','shampoo_adam_graft']
    y=np.array([1.,2.]);p=np.array([2.,1.])
    result=metric(y,p)
    assert result['mse']==1. and result['mae']==1.
    assert result['qlike']==pytest.approx(np.mean((y/p)**2-np.log((y/p)**2)-1))
