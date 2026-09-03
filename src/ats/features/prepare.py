from __future__ import annotations

import pandas as pd

from ats.config import load_settings
from ats.data.clean import clean_ohlc
from ats.features.candles import add_candle_features, classify_size
from ats.features.levels import swing_points
from ats.features.sessions import add_sessions
from ats.features.volatility import add_regimes


def prepare_frame(raw: pd.DataFrame, settings: dict | None = None) -> pd.DataFrame:
    settings = settings or load_settings()
    feat = settings["features"]
    df = clean_ohlc(raw)
    df = add_candle_features(df, feat["atr_period"], feat["avg_body_period"])
    df = add_sessions(df, settings["sessions_ist"])
    df = add_regimes(df, feat["compression_atr_ratio"], feat["expansion_atr_ratio"])
    df = swing_points(df, feat["swing_n"])
    df["size_class"] = classify_size(
        df["atr_multiple"], feat["large_candle_atr"], feat["extreme_candle_atr"]
    )
    return df
