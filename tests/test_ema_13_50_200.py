import numpy as np
import pandas as pd

from ats.hypotheses.ema_13_50_200 import ema_13_50_200_events


def test_ema_13_50_200_runs():
    n = 800
    times = pd.date_range("2022-06-10", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(7)
    close = 1800 + np.cumsum(rng.normal(0.05, 0.6, n))
    df = pd.DataFrame(
        {
            "time": times,
            "open": np.r_[close[0], close[:-1]],
            "high": close + 1.2,
            "low": close - 1.2,
            "close": close,
            "atr": np.full(n, 1.5),
        }
    )
    ev = ema_13_50_200_events(df, "XAUUSD", {"horizon_bars": 4}, 240, 50)
    assert ev is not None
    if not ev.empty:
        assert "treatment" in ev.columns
        assert set(ev["side"]).issubset({"long", "short"})
