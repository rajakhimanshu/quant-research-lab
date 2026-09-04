import numpy as np
import pandas as pd

from ats.hypotheses.fx_session import (
    compression_expand_events,
    friday_flatten_events,
    h1_tsmom_events,
    inside_bar_break_events,
    london_asia_break_events,
    long_foreign_side,
    month_end_usd_events,
    ny_close_asia_fade_events,
    overlap_continuation_events,
    postfix_usd_fade_events,
    pre_ecb_usd_events,
    prior_day_range_fade_events,
    ranaldo_local_hours_events,
    short_foreign_side,
    tokyo_close_flatten_events,
    tokyo_lunch_fade_events,
    weekend_gap_fx_events,
    xs_momentum_events,
)


def test_quote_convention():
    assert long_foreign_side("EURUSD") == "long"
    assert long_foreign_side("GBPUSD") == "long"
    assert long_foreign_side("USDJPY") == "short"
    assert short_foreign_side("EURUSD") == "short"


def _eurusd_h1(n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-02 00:00", periods=n, freq="h", tz="UTC")
    close = 1.08 + np.cumsum(rng.normal(0, 0.0003, n))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": np.maximum(close, open_) + 0.0004,
            "low": np.minimum(close, open_) - 0.0004,
            "close": close,
            "atr": np.full(n, 0.0012),
        }
    )


def test_fx_session_handlers_run():
    h1 = _eurusd_h1()
    m5 = _eurusd_h1(n=600, seed=2)
    m5["time"] = pd.date_range("2024-01-02 00:00", periods=len(m5), freq="5min", tz="UTC")
    params = {"horizon_bars": 4, "lookback_bars": 20}
    for fn, df in (
        (postfix_usd_fade_events, h1),
        (pre_ecb_usd_events, h1),
        (london_asia_break_events, m5),
        (h1_tsmom_events, h1),
        (ranaldo_local_hours_events, h1),
        (inside_bar_break_events, h1),
        (compression_expand_events, h1),
        (friday_flatten_events, h1),
        (overlap_continuation_events, h1),
        (weekend_gap_fx_events, h1),
        (tokyo_lunch_fade_events, h1),
        (ny_close_asia_fade_events, h1),
        (prior_day_range_fade_events, h1),
        (tokyo_close_flatten_events, h1),
        (month_end_usd_events, h1),
    ):
        ev = fn(df, "EURUSD", params, 0.8, 0.2)
        assert ev is not None
        if not ev.empty:
            assert "treatment" in ev.columns
            assert set(ev["side"]).issubset({"long", "short"})


def test_xs_momentum_ranks_two_pairs():
    a = _eurusd_h1(n=200, seed=3)
    b = _eurusd_h1(n=200, seed=4)
    b["close"] = b["close"] + 0.02
    settings = {
        "costs": {
            "spread_pips": {"EURUSD": 0.8, "GBPUSD": 1.0},
            "slippage_pips": 0.2,
        }
    }
    ev = xs_momentum_events(
        {"EURUSD": a, "GBPUSD": b},
        {"lookback_bars": 20, "horizon_bars": 4},
        settings,
    )
    if not ev.empty:
        assert "treatment" in ev.columns
        assert set(ev["symbol"]).issubset({"EURUSD", "GBPUSD"})
