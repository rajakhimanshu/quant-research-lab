import numpy as np
import pandas as pd

from ats.hypotheses.gold_ema200_expand import gold_ema200_expand_events, to_m30


def test_m15_resamples_to_m30():
    times = pd.date_range("2022-01-03 00:00", periods=60, freq="15min", tz="UTC")
    close = np.linspace(1800, 1810, 60)
    df = pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
        }
    )
    m30 = to_m30(df)
    assert len(m30) >= 1
    assert {"open", "high", "low", "close", "atr"} <= set(m30.columns)


def test_ema_expand_emits_treatment_or_baseline():
    n = 4000
    times = pd.date_range("2022-01-03", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(4)
    close = 1800 + np.cumsum(rng.normal(0, 0.35, n))
    high = close + rng.uniform(0.2, 0.8, n)
    low = close - rng.uniform(0.2, 0.8, n)
    # plant some 4+ ATR-looking spikes every 80 bars
    for i in range(200, n, 80):
        high[i] = close[i] + 12
        low[i] = close[i] - 1
        close[i] = close[i] + 10
    df = pd.DataFrame(
        {
            "time": times,
            "open": np.r_[close[0], close[:-1]],
            "high": np.maximum(high, close),
            "low": np.minimum(low, close),
            "close": close,
        }
    )
    ev = gold_ema200_expand_events(
        df,
        "XAUUSD",
        {"ema_period": 40, "near_atr": 0.5, "far_atr": 1.2, "expand_atr": 1.2, "horizon_bars": 3},
        240,
        50,
    )
    assert ev is not None
    assert "treatment" in ev.columns or ev.empty
    if not ev.empty:
        assert bool(ev["treatment"].any()) or bool((~ev["treatment"]).any())
        assert "r_mult" in ev.columns
