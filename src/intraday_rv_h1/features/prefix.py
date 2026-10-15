"""Causal 5/15 opening prefix, migrated unchanged from the research implementation."""
import numpy as np
import pandas as pd
UNANCHORED_COLUMNS=("pfx_5_15_fast_variance","pfx_5_15_slow_variance","pfx_5_15_reconstructed_variance")
def _prefix_branch(values: np.ndarray, clock: np.ndarray, dates: np.ndarray, frequency: int) -> np.ndarray:
    """Compute an unanchored opening-prefix branch with the production clips."""
    n = len(values)
    finite_input = np.isfinite(values) & np.isfinite(clock)
    safe_values = np.where(finite_input, values, 0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(safe_values)))
    invalid_cumulative = np.concatenate(([0], np.cumsum((~finite_input).astype(np.int64))))
    sampled = np.full(n, np.nan)
    dt = np.full(n, np.nan)
    if n > frequency:
        endpoints = np.arange(frequency, n)
        sampled[endpoints] = cumulative[endpoints + 1] - cumulative[endpoints - frequency + 1]
        dt[endpoints] = clock[endpoints] - clock[endpoints - frequency]
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.sqrt(260.0 * np.square(sampled) / dt)
    invalid_windows = np.zeros(n, dtype=bool)
    if n > frequency:
        invalid_windows[endpoints] = (
            invalid_cumulative[endpoints + 1] - invalid_cumulative[endpoints - frequency + 1]
        ) > 0
    valid = np.isfinite(raw) & np.isfinite(dt) & (dt > 0) & ~invalid_windows
    raw[~valid] = np.nan
    clipped = np.clip(raw, 0.05, 1.0)
    energy = np.where(valid, np.square(clipped) * dt, 0.0)
    dt_energy = np.where(valid, dt, 0.0)
    valid_count = valid.astype(np.int64)
    cumulative_energy = np.concatenate(([0.0], np.cumsum(energy)))
    cumulative_dt = np.concatenate(([0.0], np.cumsum(dt_energy)))
    cumulative_valid = np.concatenate(([0], np.cumsum(valid_count)))
    result = np.full(n, np.nan)
    for _, index in pd.Series(np.arange(n)).groupby(dates, sort=False):
        positions = index.to_numpy(dtype=int)
        if len(positions) < 2:
            continue
        endpoints = positions[1:]
        start = int(positions[1])
        count = np.arange(1, len(positions), dtype=np.int64)
        numerator = cumulative_energy[endpoints + 1] - cumulative_energy[start]
        denominator = cumulative_dt[endpoints + 1] - cumulative_dt[start]
        observed = cumulative_valid[endpoints + 1] - cumulative_valid[start]
        ok = (observed == count) & np.isfinite(denominator) & (denominator > 0)
        vol = np.full(len(endpoints), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            vol[ok] = np.sqrt(numerator[ok] / denominator[ok])
        result[endpoints] = np.square(np.clip(vol, 0.08, 1.0))
    return result

def _attach_unanchored_prefix(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reset_index(drop=True).copy()
    for column in UNANCHORED_COLUMNS:
        result[column] = np.nan
    values = pd.to_numeric(result["return"], errors="coerce").to_numpy(float)
    clock = pd.to_numeric(result["ttime"], errors="coerce").to_numpy(float)
    dates = result["date"].to_numpy()
    for _, index in result.groupby("symbol", sort=False).groups.items():
        positions = np.asarray(index, dtype=int)
        order = positions[np.argsort(result.iloc[positions]["datetime"].to_numpy(), kind="mergesort")]
        local_values = values[order].copy()
        local_clock = clock[order]
        local_dates = dates[order]
        first = np.r_[True, local_dates[1:] != local_dates[:-1]]
        local_values[first] = 0.0
        # rvol-demo defines the intraday bucket as fast=5, slow=15.  The
        # 15/60 operator in the source panel is a separate multi-day/H1
        # auxiliary and must not be used as the rvol1hour reconstruction.
        fast = _prefix_branch(local_values, local_clock, local_dates, 5)
        slow = _prefix_branch(local_values, local_clock, local_dates, 15)
        with np.errstate(invalid="ignore"):
            fast_vol, slow_vol = np.sqrt(fast), np.sqrt(slow)
            weight = np.square((fast_vol + 0.05) / (fast_vol + slow_vol + 0.1))
            reconstructed = np.square(weight * fast_vol + (1.0 - weight) * slow_vol)
        result.loc[order, "pfx_5_15_fast_variance"] = fast
        result.loc[order, "pfx_5_15_slow_variance"] = slow
        result.loc[order, "pfx_5_15_reconstructed_variance"] = reconstructed
    return result
