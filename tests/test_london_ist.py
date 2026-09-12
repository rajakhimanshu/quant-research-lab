import numpy as np
import pandas as pd

from ats.hypotheses.london_ist import london_fib_bounce_events, london_ist_break_events


def _m15(n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-06-03 00:00", periods=n, freq="15min", tz="UTC")
    close = 1.08 + np.cumsum(rng.normal(0, 0.0004, n))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": np.maximum(close, open_) + 0.0006,
            "low": np.minimum(close, open_) - 0.0006,
            "close": close,
        }
    )


def test_london_ist_handlers_run():
    df = _m15()
    params = {"horizon_bars": 8, "rr": 1.0, "impulse_bars": 8}
    for fn in london_ist_break_events, london_fib_bounce_events:
        ev = fn(df, "EURUSD", params, 0.8, 0.2)
        assert ev is not None
        if not ev.empty:
            assert "treatment" in ev.columns
            assert set(ev["side"]).issubset({"long", "short"})
