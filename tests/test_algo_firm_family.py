from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ats.hypotheses.fx_session import (
    carry_roll_events,
    tokyo_fix_follow_events,
    traditional_carry_side,
    nyse_open_fx_events,
)
from ats.hypotheses.ny_open_sweep import ny_sweep_follow_events

NY = ZoneInfo("America/New_York")
TOKYO = ZoneInfo("Asia/Tokyo")
LONDON = ZoneInfo("Europe/London")


def test_traditional_carry_side_is_frozen():
    assert traditional_carry_side("AUDUSD") == "long"
    assert traditional_carry_side("NZDUSD") == "long"
    assert traditional_carry_side("USDJPY") == "long"
    assert traditional_carry_side("USDCHF") == "long"


def test_ny_sweep_follow_takes_unreclaimed_raid_not_reclaim():
    times = pd.date_range("2024-01-03 03:00", periods=48, freq="15min", tz=NY)
    n = len(times)
    close = np.full(n, 2000.0)
    high = np.full(n, 2001.0)
    low = np.full(n, 1999.0)
    # 09:00 NY: sweep overnight high and stay outside (follow, not reclaim).
    i9 = 24
    assert times[i9].hour == 9
    high[i9] = 2008.0
    close[i9] = 2006.0
    open_ = np.r_[close[0], close[:-1]]
    open_[i9] = 2000.5
    df = pd.DataFrame(
        {
            "time": times.tz_convert("UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "atr": np.full(n, 2.0),
        }
    )
    ev = ny_sweep_follow_events(df, "XAUUSD", {"horizon_bars": 8, "sweep_atr": 0.10}, 200, 20)
    assert not ev.empty
    treat = ev.loc[ev["treatment"]]
    assert not treat.empty
    assert set(treat["side"]) == {"long"}


def test_carry_roll_and_tokyo_fix_handlers_run():
    h1_times = pd.date_range("2024-01-02 00:00", periods=120, freq="h", tz="UTC")
    close = 0.67 + np.cumsum(np.random.default_rng(3).normal(0, 0.0004, len(h1_times)))
    open_ = np.r_[close[0], close[:-1]]
    h1 = pd.DataFrame(
        {
            "time": h1_times,
            "open": open_,
            "high": np.maximum(close, open_) + 0.0005,
            "low": np.minimum(close, open_) - 0.0005,
            "close": close,
            "atr": np.full(len(h1_times), 0.002),
        }
    )
    carry = carry_roll_events(h1, "AUDUSD", {"horizon_bars": 4}, 1.1, 0.2)
    assert carry is not None
    if not carry.empty:
        assert set(carry["side"]) == {"long"}

    jst = pd.date_range("2024-01-04 09:00", periods=80, freq="15min", tz=TOKYO)
    jpy = 148 + np.cumsum(np.random.default_rng(4).normal(0, 0.02, len(jst)))
    j_open = np.r_[jpy[0], jpy[:-1]]
    m15 = pd.DataFrame(
        {
            "time": jst.tz_convert("UTC"),
            "open": j_open,
            "high": np.maximum(jpy, j_open) + 0.04,
            "low": np.minimum(jpy, j_open) - 0.04,
            "close": jpy,
            "atr": np.full(len(jst), 0.08),
        }
    )
    fix = tokyo_fix_follow_events(m15, "USDJPY", {"horizon_bars": 4, "min_prior_atr": 0.01}, 1.0, 0.2)
    assert fix is not None
    if not fix.empty:
        assert "treatment" in fix.columns
        assert set(fix["side"]).issubset({"long", "short"})


def test_nyse_open_fx_handler_runs():
    times = pd.date_range("2024-01-02 00:00", periods=200, freq="15min", tz="UTC")
    close = 1.08 + np.cumsum(np.random.default_rng(5).normal(0, 0.0003, len(times)))
    open_ = np.r_[close[0], close[:-1]]
    df = pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": np.maximum(close, open_) + 0.0004,
            "low": np.minimum(close, open_) - 0.0004,
            "close": close,
            "atr": np.full(len(times), 0.001),
        }
    )
    ev = nyse_open_fx_events(df, "EURUSD", {"horizon_bars": 4}, 0.8, 0.2)
    assert ev is not None
    if not ev.empty:
        assert set(ev["side"]) == {"long"}
