"""Physical observation schema shared by causal return/price history."""
import numpy as np
import pandas as pd
KEY=["symbol","date","clock_position"]

def _observations(frame: pd.DataFrame) -> pd.DataFrame:
    required = {*KEY, "datetime", "return"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"seasonal energy fields missing: {missing}")
    f = frame[[*KEY, "datetime", "return"]].copy()
    f["datetime"] = pd.to_datetime(f.datetime, errors="raise")
    f["date"] = pd.to_datetime(f.date, errors="raise")
    if f.empty or f.datetime.isna().any() or f.date.isna().any():
        raise ValueError("empty or missing seasonal energy datetimes")
    if f.datetime.dt.tz is not None or f.date.dt.tz is not None:
        raise ValueError("seasonal energy requires naive exchange-local timestamps")
    if not f.datetime.dt.normalize().eq(f.date).all():
        raise ValueError("seasonal energy date does not match physical datetime")
    if f.symbol.isna().any() or f.clock_position.isna().any():
        raise ValueError("missing seasonal energy physical key")
    pos = f.clock_position.to_numpy(float)
    if not np.isfinite(pos).all() or (pos < 0).any() or (pos != np.floor(pos)).any():
        raise ValueError("invalid physical clock positions")
    if f.duplicated(KEY).any() or f.duplicated(["symbol", "datetime"]).any():
        raise ValueError("duplicate seasonal energy physical keys")
    if f.date.eq(pd.Timestamp("2025-02-04")).any():
        raise ValueError("excluded 2025-02-04 must be removed before seasonal energy")
    if np.isinf(f["return"].to_numpy(float)).any():
        raise FloatingPointError("infinite canonical return in seasonal energy")
    f = f.sort_values(["symbol", "datetime"], kind="mergesort").reset_index(drop=True)
    previous = f.groupby(["symbol", "date"], sort=False).datetime.shift()
    gap = (f.datetime - previous).dt.total_seconds()
    f["opening"] = previous.isna()
    f["lunch"] = gap.ge(3600)
    f["continuous"] = gap.gt(0) & gap.le(90)
    # A sparse requested slice cannot redefine physical position 10 as opening.
    if not f.loc[f.opening, "clock_position"].eq(0).all():
        raise ValueError("seasonal energy requires each session's physical opening row")
    order = f.groupby(["symbol", "date"], sort=False).clock_position.diff()
    if order.dropna().le(0).any():
        raise ValueError("physical position order differs from datetime order")
    return f
