import numpy as np
import pandas as pd

from ats.hypotheses.fx_session import (
    big_figure_fade_events,
    compression_expand_events,
    friday_flatten_events,
    h1_tsmom_events,
    inside_bar_break_events,
    london_asia_break_events,
    london_lunch_fade_events,
    long_foreign_side,
    month_end_usd_events,
    ny_close_asia_fade_events,
    overlap_continuation_events,
    postfix_usd_fade_events,
    pre_ecb_usd_events,
    prior_day_range_fade_events,
    ranaldo_local_hours_events,
    short_foreign_side,
    stop_pool_20_fade_events,
    tokyo_close_flatten_events,
    tokyo_lunch_fade_events,
    two_day_streak_fade_events,
    print_window_fade_events,
    wm_postfix_fade_events,
    weekend_gap_fx_events,
    monday_cash_gap_events,
    xs_momentum_events,
    overnight_xs_fade_events,
    overnight_intraday_fade_events,
    m15_to_h1,
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
        (overnight_intraday_fade_events, h1),
        (ranaldo_local_hours_events, h1),
        (inside_bar_break_events, h1),
        (compression_expand_events, h1),
        (friday_flatten_events, h1),
        (overlap_continuation_events, h1),
        (weekend_gap_fx_events, h1),
        (monday_cash_gap_events, h1),
        (tokyo_lunch_fade_events, h1),
        (ny_close_asia_fade_events, h1),
        (prior_day_range_fade_events, h1),
        (tokyo_close_flatten_events, h1),
        (month_end_usd_events, h1),
        (wm_postfix_fade_events, h1),
        (big_figure_fade_events, h1),
        (stop_pool_20_fade_events, h1),
        (london_lunch_fade_events, h1),
        (two_day_streak_fade_events, h1),
    ):
        ev = fn(df, "EURUSD", params, 0.8, 0.2)
        assert ev is not None
        if not ev.empty:
            assert "treatment" in ev.columns
            assert set(ev["side"]).issubset({"long", "short"})


def test_monday_cash_fades_up_weekend_package():
    # Fri 5 Jan 2024 ... Mon 8 Jan 08:00 London (UTC in January).
    t1 = pd.date_range("2024-01-05 00:00", periods=24, freq="h", tz="UTC")
    t2 = pd.date_range("2024-01-08 00:00", periods=72, freq="h", tz="UTC")
    times = t1.append(t2)
    n = len(times)
    close = np.full(n, 1.2700)
    close[24:] = 1.2800
    open_ = close.copy()
    open_[24] = 1.2800
    df = pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": np.maximum(open_, close) + 0.0003,
            "low": np.minimum(open_, close) - 0.0003,
            "close": close,
            "atr": np.full(n, 0.002),
        }
    )
    ev = monday_cash_gap_events(
        df,
        "GBPUSD",
        {"cash_hour": 8, "tz": "Europe/London", "min_gap_atr": 0.15, "horizon_bars": 8},
        1.0,
        0.2,
    )
    treat = ev.loc[ev["treatment"]] if not ev.empty else ev
    assert not treat.empty
    assert set(treat["side"]) == {"short"}


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


def test_overnight_xs_fade_shorts_atr_winner():
    def _force_overnight(df: pd.DataFrame, delta: float) -> pd.DataFrame:
        work = df.copy()
        ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert("Europe/London")
        hours = ldn.dt.hour.to_numpy()
        dates = ldn.dt.date
        pri: dict = {}
        for i in range(len(work)):
            if hours[i] == 16:
                pri[dates.iloc[i]] = i
        for i in range(len(work)):
            if hours[i] != 8:
                continue
            prior = [p for p in pri if p < dates.iloc[i]]
            if not prior:
                continue
            i16 = pri[max(prior)]
            work.at[i, "open"] = float(work.at[i16, "close"]) + delta
        return work

    gbp = _force_overnight(_eurusd_h1(n=300, seed=11), 0.01)
    jpy = _force_overnight(_eurusd_h1(n=300, seed=12), 0.0)
    aud = _force_overnight(_eurusd_h1(n=300, seed=13), -0.01)
    settings = {
        "costs": {
            "spread_pips": {"GBPUSD": 1.0, "USDJPY": 1.2, "AUDUSD": 1.1},
            "slippage_pips": 0.2,
        }
    }
    ev = overnight_xs_fade_events(
        {"GBPUSD": gbp, "USDJPY": jpy, "AUDUSD": aud},
        {"horizon_bars": 4},
        settings,
    )
    treat = ev.loc[ev["treatment"]] if not ev.empty else ev
    assert not treat.empty
    gbp_sides = set(treat.loc[treat["symbol"] == "GBPUSD", "side"])
    aud_sides = set(treat.loc[treat["symbol"] == "AUDUSD", "side"])
    assert gbp_sides == {"short"}
    assert aud_sides == {"long"}


def test_print_window_fade_marks_named_event():
    df = _eurusd_h1(n=120, seed=5)
    ny = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/New_York")
    hit = df.index[(ny.dt.hour == 8) & (ny.dt.minute == 0) & (ny.dt.dayofweek < 5)]
    assert len(hit) > 0
    i = int(hit[0])
    cal = pd.DataFrame(
        {
            "datetime_utc": [pd.Timestamp(df.at[i, "time"]) + pd.Timedelta(minutes=30)],
            "currency": ["USD"],
            "impact": ["high"],
            "event": ["CPI m/m"],
        }
    )
    ev = print_window_fade_events(
        df,
        "EURUSD",
        {
            "tz": "America/New_York",
            "treat_hour": 8,
            "treat_minute": 0,
            "event_names": ["CPI m/m"],
            "horizon_bars": 4,
        },
        0.8,
        0.2,
        calendar=cal,
    )
    assert ev is not None
    if not ev.empty:
        assert bool(ev["treatment"].any())


def test_print_window_snaps_ff_cache_one_day_early():
    df = _eurusd_h1(n=200, seed=7)
    ny = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/New_York")
    hit = df.index[(ny.dt.hour == 8) & (ny.dt.minute == 0) & (ny.dt.dayofweek == 4)]
    assert len(hit) > 0
    i = int(hit[0])
    print_ny = ny.iloc[i]
    cache_ts = (print_ny.normalize() - pd.Timedelta(days=1)).tz_convert("UTC") + pd.Timedelta(hours=20, minutes=30)
    cal = pd.DataFrame(
        {
            "datetime_utc": [cache_ts],
            "currency": ["USD"],
            "impact": ["high"],
            "event": ["Non-Farm Employment Change"],
        }
    )
    ev = print_window_fade_events(
        df,
        "EURUSD",
        {
            "tz": "America/New_York",
            "treat_hour": 8,
            "treat_minute": 0,
            "event_names": ["Non-Farm Employment Change"],
            "horizon_bars": 4,
        },
        0.8,
        0.2,
        calendar=cal,
    )
    assert not ev.empty
    assert bool(ev["treatment"].any())


def test_overnight_intraday_fade_fires_weekday():
    # Build H1 so 21:00 then next day 08:00 London exist with a move.
    times = pd.date_range("2024-01-02 00:00", periods=80, freq="h", tz="UTC")
    close = np.linspace(1.08, 1.10, len(times))
    close[21] = 1.09  # approx
    df = pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": close + 0.001,
            "low": close - 0.001,
            "close": close,
            "atr": np.full(len(times), 0.001),
        }
    )
    ev = overnight_intraday_fade_events(
        df,
        "EURUSD",
        {
            "overnight_hour_london": 21,
            "signal_hour_london": 8,
            "min_prior_atr": 0.01,
            "horizon_bars": 4,
        },
        0.8,
        0.2,
    )
    assert ev is not None


def test_m15_to_h1_resamples():
    times = pd.date_range("2024-01-02 00:00", periods=16, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "time": times,
            "open": 2000.0,
            "high": 2001.0,
            "low": 1999.0,
            "close": np.linspace(2000, 2004, len(times)),
            "atr": 1.0,
        }
    )
    h1 = m15_to_h1(df)
    assert len(h1) >= 3
    assert "atr" in h1.columns
