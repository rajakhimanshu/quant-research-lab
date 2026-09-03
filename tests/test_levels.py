import pandas as pd

from ats.features.levels import swing_points
from tests.helpers import synthetic_ohlc


def test_swing_high_at_clear_peak():
    df = synthetic_ohlc(80, seed=2)
    peak = 40
    df.loc[peak, "high"] = df["high"].max() + 0.01
    df.loc[peak, "low"] = df.loc[peak, "high"] - 0.0003
    df.loc[peak, "close"] = df.loc[peak, "high"] - 0.0001
    df["atr"] = 0.001
    out = swing_points(df, n=3)
    assert bool(out.loc[peak, "swing_high"])
