from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ats.config import load_settings
from ats.hypotheses.fx_session import weekend_gap_fx_events
from ats.paper.h86_book import replay_oos_equity, scan_symbol
from ats.research.h86_deep import _first_bar, _walk_bars


def _weekend_h1(gap: float = 0.01, atr: float = 0.002, extra_after: int = 12, spread_pts: float = 10.0):
    """Thu+Fri hourly, then Sunday 21:00 UTC reopen. Gap hours ~46."""
    t1 = pd.date_range("2024-01-04 00:00", periods=48, freq="h", tz="UTC")
    t2 = pd.date_range("2024-01-07 21:00", periods=extra_after, freq="h", tz="UTC")
    times = t1.append(t2)
    n = len(times)
    close = np.full(n, 1.2700)
    close[48:] = 1.2700 + gap
    open_ = close.copy()
    open_[48] = 1.2700 + gap
    return pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": np.maximum(open_, close) + 0.0002,
            "low": np.minimum(open_, close) - 0.0002,
            "close": close,
            "atr": np.full(n, atr),
            "spread": np.full(n, spread_pts),
        }
    )


def test_h86_oos_replay_grows_on_positive_r():
    oos = pd.DataFrame({"r_mult": [1.0, 1.0, -1.0]})
    book = replay_oos_equity(oos, starting=2000.0, risk_pct=0.01)
    assert book["n"] == 3
    assert book["ending"] > 2000.0
    assert book["mean_r"] == 1.0 / 3.0


def test_h86_scan_waits_on_saturday():
    df = _weekend_h1()
    sat = datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc)
    row = scan_symbol(df, "GBPUSD", load_settings(), now=sat)
    assert row["action"] == "WAIT_HALT"


def test_h86_scan_enters_on_reopen_bar():
    df = _weekend_h1(extra_after=1)
    now = datetime(2024, 1, 7, 21, 20, tzinfo=timezone.utc)
    row = scan_symbol(df, "GBPUSD", load_settings(), now=now)
    assert row["action"] == "ENTER"
    assert row["side"] == "short"


def test_h86_scan_skips_tiny_gap():
    df = _weekend_h1(gap=0.0001, atr=0.002, extra_after=1)
    now = datetime(2024, 1, 7, 21, 20, tzinfo=timezone.utc)
    row = scan_symbol(df, "GBPUSD", load_settings(), now=now)
    assert row["action"] == "SKIP"


def test_h86_scan_hold_then_flat():
    hold_df = _weekend_h1(extra_after=5)
    now_hold = datetime(2024, 1, 8, 1, 20, tzinfo=timezone.utc)
    assert scan_symbol(hold_df, "GBPUSD", load_settings(), now=now_hold)["action"] == "HOLD"
    flat_df = _weekend_h1(extra_after=12)
    now_flat = datetime(2024, 1, 8, 9, 20, tzinfo=timezone.utc)
    assert scan_symbol(flat_df, "GBPUSD", load_settings(), now=now_flat)["action"] == "FLAT"


def test_weekend_gap_fades_up_gap_and_follow_inverts():
    df = _weekend_h1()
    params = {
        "min_gap_hours": 36,
        "min_gap_atr": 0.15,
        "horizon_bars": 8,
        "stop_atr": 1.0,
        "target_atr": 1.0,
        "base_hour_london": 8,
        "prior_hour_london": 16,
    }
    fade = weekend_gap_fx_events(df, "GBPUSD", params, 1.0, 0.2)
    treat = fade.loc[fade["treatment"]]
    assert not treat.empty
    assert set(treat["side"]) == {"short"}
    follow = weekend_gap_fx_events(df, "GBPUSD", {**params, "follow_gap": True}, 1.0, 0.2)
    ft = follow.loc[follow["treatment"]]
    assert set(ft["side"]) == {"long"}


def test_bar_spread_costs_more_on_wide_sunday_print():
    df = _weekend_h1(spread_pts=10.0)
    df.loc[48, "spread"] = 40.0
    params = {
        "min_gap_hours": 36,
        "min_gap_atr": 0.15,
        "horizon_bars": 8,
        "stop_atr": 1.0,
        "target_atr": 1.0,
        "use_bar_spread": True,
    }
    ev = weekend_gap_fx_events(df, "GBPUSD", params, 1.0, 0.2)
    treat = ev.loc[ev["treatment"]].iloc[0]
    lab = weekend_gap_fx_events(df, "GBPUSD", {**params, "use_bar_spread": False}, 1.0, 0.2)
    lab_t = lab.loc[lab["treatment"]].iloc[0]
    assert treat["entry"] < lab_t["entry"]


def test_first_bar_both_hits_counts_stop_first():
    assert _first_bar("short", high=1.29, low=1.25, stop=1.28, target=1.26) == "both_stop_first"


def test_m5_walk_hits_target():
    bars = pd.DataFrame(
        {
            "high": [1.271, 1.269],
            "low": [1.269, 1.258],
            "close": [1.270, 1.259],
        }
    )
    out = _walk_bars(bars, "short", entry=1.270, stop=1.272, target=1.268)
    assert out["end"] == "target"
    assert out["r_mult"] == 1.0
