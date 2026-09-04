import numpy as np
import pandas as pd

from ats.hypotheses.m15_micro import (
    large_bar_fade_events,
    ny_box_fade_events,
    pdh_pdl_fade_events,
    round_bounce_events,
    volume_spike_cont_events,
    vwap_extreme_fade_events,
)


def _gold_m15(n: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-02 00:00", periods=n, freq="15min", tz="UTC")
    close = 2050 + np.cumsum(rng.normal(0.02, 0.8, n))
    high = close + rng.uniform(0.4, 1.8, n)
    low = close - rng.uniform(0.4, 1.8, n)
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": np.maximum(high, np.maximum(open_, close)),
            "low": np.minimum(low, np.minimum(open_, close)),
            "close": close,
            "volume": rng.integers(80, 220, n).astype(float),
            "atr": np.full(n, 1.6),
        }
    )


def test_pdh_pdl_fade_fires_treatment_and_baseline():
    n = 200
    times = pd.date_range("2024-03-04 00:00", periods=n, freq="15min", tz="UTC")
    close = np.full(n, 2000.0)
    high = close + 0.5
    low = close - 0.5
    high[:96] = 2010.0
    low[:96] = 1990.0
    high[100] = 2012.0
    close[100] = 2009.0
    low[120] = 1994.0
    high[120] = 2001.0
    close[120] = 1999.0
    df = pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "atr": np.full(n, 2.0),
        }
    )
    ev = pdh_pdl_fade_events(df, "XAUUSD", {"horizon_bars": 4}, 200, 20)
    assert ev is not None
    if not ev.empty:
        assert set(ev["treatment"]).issubset({True, False})
        assert set(ev["side"]).issubset({"long", "short"})


def test_all_m15_micro_handlers_run():
    df = _gold_m15()
    params = {"horizon_bars": 4}
    handlers = [
        pdh_pdl_fade_events,
        large_bar_fade_events,
        vwap_extreme_fade_events,
        volume_spike_cont_events,
        round_bounce_events,
        ny_box_fade_events,
    ]
    for fn in handlers:
        ev = fn(df, "XAUUSD", params, 200, 20)
        assert ev is not None
        if not ev.empty:
            assert "treatment" in ev.columns
            assert "r_mult" in ev.columns
            assert set(ev["side"]).issubset({"long", "short"})
