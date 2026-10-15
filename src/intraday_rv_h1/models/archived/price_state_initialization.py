"""Masked, asymmetric state interactions for deterministic coefficient fits."""
import numpy as np
import torch

from .slope_state_attention import solve_state_ridge


def response(state, features):
    """[N,F,(value, availability, neutral/zero value)] -> +/- hinges."""
    tensor = isinstance(state, torch.Tensor)
    s = state if tensor else np.asarray(state, float)
    columns = tuple(features)
    if (s.ndim != 3 or s.shape[2] != 3 or not columns
            or len(set(columns)) != len(columns)
            or min(columns) < 0 or max(columns) >= s.shape[1]):
        raise ValueError('invalid coefficient initialization state layout')
    value, mask, center = (s[:, columns, j] for j in range(3))
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError('nonbinary coefficient state availability')
    where = torch.where if tensor else np.where
    delta = where(mask == 1, value-center, 0.)
    finite = torch.isfinite if tensor else np.isfinite
    if not finite(delta).all():
        raise FloatingPointError('nonfinite visible coefficient response')
    relu = torch.relu if tensor else lambda x: np.maximum(x, 0.)
    stack = torch.stack if tensor else np.stack
    return stack((relu(delta), relu(-delta)), -1).reshape(len(s), -1)


def design(h, state, features, column):
    if not isinstance(column, int) or not 0 <= column < 16 or h.shape != (len(state), 16):
        raise ValueError('invalid explicit HAR coefficient column')
    finite = torch.isfinite if isinstance(h, torch.Tensor) else np.isfinite
    if not finite(h[:, column]).all():
        raise FloatingPointError('nonfinite explicit HAR coefficient input')
    return response(state, features)*h[:, column, None]


def fit(h, state, eta, y, features, column, *, chunk_size=8192):
    y, eta = np.asarray(y, float), np.asarray(eta, float)
    if (y.shape != (len(h),) or eta.shape != y.shape or not len(y)
            or not np.isfinite(y).all() or (y <= 0).any() or not np.isfinite(eta).all()):
        raise ValueError('invalid coefficient initialization fit labels')
    width = 2*len(features)
    gram, rhs = np.zeros((width, width)), np.zeros(width)
    for low in range(0, len(h), chunk_size):
        sl = slice(low, low+chunk_size)
        x = design(h[sl], state[sl], features, column)
        gram += x.T@x
        rhs += x.T@(2*np.log(y[sl])-eta[sl])
    weights, audit = solve_state_ridge(gram, rhs, len(y), ridge=.01)
    return weights, dict(audit, features=list(features), column=column), gram, rhs


def predict(h, state, eta, weights, features, column, *, chunk_size=8192):
    result = np.empty(len(h))
    for low in range(0, len(h), chunk_size):
        sl = slice(low, low+chunk_size)
        result[sl] = np.exp((eta[sl]+design(h[sl], state[sl], features, column)@weights)/2)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise FloatingPointError('invalid initialized coefficient forecast')
    return result
