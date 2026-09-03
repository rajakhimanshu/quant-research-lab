from __future__ import annotations

import numpy as np
import pandas as pd

from ats.timeutil import pip_size


def clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out = out.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    ohlc = ["open", "high", "low", "close"]
    out[ohlc] = out[ohlc].replace(0, np.nan)
    out = out.dropna(subset=ohlc)
    bad = (
        (out["high"] < out["low"])
        | (out["high"] < out["open"])
        | (out["high"] < out["close"])
        | (out["low"] > out["open"])
        | (out["low"] > out["close"])
    )
    out = out.loc[~bad].reset_index(drop=True)
    out["range"] = out["high"] - out["low"]
    out["body"] = (out["close"] - out["open"]).abs()
    out["direction"] = np.where(out["close"] > out["open"], 1, np.where(out["close"] < out["open"], -1, 0))
    if "spread" not in out.columns:
        symbol = str(out["requested_symbol"].iloc[0]) if "requested_symbol" in out.columns else "EURUSD"
        out["spread"] = np.nan
        out.attrs["pip_size"] = pip_size(symbol)
    return out
