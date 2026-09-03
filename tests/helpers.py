from __future__ import annotations

import numpy as np
import pandas as pd

from ats.config import load_settings
from ats.data.clean import clean_ohlc
from ats.features.prepare import prepare_frame


def synthetic_ohlc(n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2023-01-02", periods=n, freq="h", tz="UTC")
    close = 1.0800 + np.cumsum(rng.normal(0, 0.00025, n))
    noise = rng.uniform(0.00005, 0.00035, n)
    high = close + noise
    low = close - noise
    open_ = np.r_[close[0], close[:-1]]
    df = pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": np.maximum(high, np.maximum(open_, close)),
            "low": np.minimum(low, np.minimum(open_, close)),
            "close": close,
            "volume": 100,
            "spread": 12,
            "symbol": "EURUSD",
            "requested_symbol": "EURUSD",
            "timeframe": "H1",
        }
    )
    return clean_ohlc(df)


def prepared(n: int = 400, seed: int = 1) -> pd.DataFrame:
    return prepare_frame(synthetic_ohlc(n, seed), load_settings())
