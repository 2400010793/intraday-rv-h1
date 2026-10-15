"""Offline loss summaries for fixed research predictions."""
import numpy as np


def metric(y, p):
    r = (y/p)**2
    return dict(rows=len(y), mse=float(np.mean((y-p)**2)), mae=float(np.mean(abs(y-p))), qlike=float(np.mean(r-np.log(r)-1)))
