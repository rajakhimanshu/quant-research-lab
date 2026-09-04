import numpy as np
import pandas as pd

from ats.hypotheses.ny_ema_drd import ny_ema_drd_events, wilder_di_adx


def test_drd_adx_has_plus_minus():
    n = 80
    times = pd.date_range("2024-01-02", periods=n, freq="5min", tz="UTC")
    close = 1800 + np.linspace(0, 40, n)
    df = pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
        }
    )
    out = wilder_di_adx(df)
    assert out["plus_di"].notna().sum() > 10
    assert out["adx"].notna().sum() > 5
    # strong uptrend should eventually show +DI > -DI
    tail = out.dropna(subset=["plus_di", "minus_di"]).iloc[-10:]
    assert float(tail["plus_di"].mean()) >= float(tail["minus_di"].mean())


def test_ny_ema_drd_emits_rows():
    n = 3000
    times = pd.date_range("2024-03-04 08:00", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(2)
    close = 1800 + np.cumsum(rng.normal(0, 0.4, n))
    df = pd.DataFrame(
        {
            "time": times,
            "open": np.r_[close[0], close[:-1]],
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "atr": np.full(n, 1.2),
        }
    )
    ev = ny_ema_drd_events(df, "XAUUSD", {"horizon_bars": 4}, 240, 50)
    assert ev is not None
    if not ev.empty:
        assert "treatment" in ev.columns
        assert "r_mult" in ev.columns
        assert set(ev["side"]).issubset({"long", "short"})
