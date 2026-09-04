import numpy as np
import pandas as pd

from ats.hypotheses.ema200_drd_entry import ema200_drd_entry_events


def test_ema200_drd_runs():
    n = 2500
    times = pd.date_range("2025-04-07", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(6)
    close = 1800 + np.cumsum(rng.normal(0.02, 0.5, n))
    df = pd.DataFrame(
        {
            "time": times,
            "open": np.r_[close[0], close[:-1]],
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        }
    )
    ev = ema200_drd_entry_events(df, "XAUUSD", {"ema_period": 50, "horizon_bars": 6}, 240, 50)
    assert ev is not None
    if not ev.empty:
        assert "treatment" in ev.columns
        assert set(ev["side"]).issubset({"long", "short"})
