from __future__ import annotations

import numpy as np
import pandas as pd


def add_regimes(df: pd.DataFrame, compression: float, expansion: float, lookback: int = 20) -> pd.DataFrame:
    out = df.copy()
    avg = out["atr"].rolling(lookback, min_periods=lookback).mean()
    ratio = out["atr"] / avg
    out["atr_regime_ratio"] = ratio
    out["regime"] = "normal"
    out.loc[ratio < compression, "regime"] = "compression"
    out.loc[ratio > expansion, "regime"] = "expansion"
    slope = out["close"].diff().rolling(lookback, min_periods=lookback).mean()
    out["trend_regime"] = np.where(slope.abs() > out["atr"] * 0.05, "trending", "ranging")
    return out
