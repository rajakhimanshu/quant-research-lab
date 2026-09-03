from __future__ import annotations

import numpy as np
import pandas as pd


def add_candle_features(df: pd.DataFrame, atr_period: int, avg_body_period: int) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            (out["high"] - out["low"]).abs(),
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.rolling(atr_period, min_periods=atr_period).mean()
    out["avg_body"] = out["body"].rolling(avg_body_period, min_periods=avg_body_period).mean()
    out["avg_range"] = out["range"].rolling(avg_body_period, min_periods=avg_body_period).mean()
    upper = out["high"] - np.maximum(out["open"], out["close"])
    lower = np.minimum(out["open"], out["close"]) - out["low"]
    out["upper_wick"] = upper
    out["lower_wick"] = lower
    body = out["body"].replace(0, np.nan)
    out["upper_wick_ratio"] = upper / body
    out["lower_wick_ratio"] = lower / body
    out["atr_multiple"] = out["range"] / out["atr"]
    return out


def classify_size(atr_multiple: pd.Series, large: float, extreme: float) -> pd.Series:
    labels = pd.Series("normal", index=atr_multiple.index)
    labels = labels.mask(atr_multiple >= large, "large")
    labels = labels.mask(atr_multiple >= extreme, "extreme")
    return labels
