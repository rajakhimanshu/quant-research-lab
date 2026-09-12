import numpy as np
import pandas as pd

from ats.hypotheses.m15_micro import (
    asia_range_london_fade_events,
    large_bar_fade_events,
    ny_box_fade_events,
    pdh_pdl_fade_events,
    prior_hour_extreme_fade_events,
    hour_close_flatten_events,
    hour_open_gap_fade_events,
    prior_close_tag_fade_events,
    round_bounce_events,
    session_box_fade_events,
    session_range_later_fade_events,
    silver_bullet_fvg_events,
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
        asia_range_london_fade_events,
        large_bar_fade_events,
        vwap_extreme_fade_events,
        volume_spike_cont_events,
        round_bounce_events,
        ny_box_fade_events,
        silver_bullet_fvg_events,
        prior_hour_extreme_fade_events,
        hour_close_flatten_events,
        hour_open_gap_fade_events,
        prior_close_tag_fade_events,
        session_range_later_fade_events,
    ]
    for fn in handlers:
        ev = fn(df, "XAUUSD", params, 200, 20)
        assert ev is not None
        if not ev.empty:
            assert "treatment" in ev.columns
            assert "r_mult" in ev.columns
            assert set(ev["side"]).issubset({"long", "short"})


def test_silver_bullet_follows_bullish_fvg_in_ny_am():
    n = 48
    times = pd.date_range("2024-01-08 13:00", periods=n, freq="15min", tz="UTC")
    close = np.full(n, 1.0800)
    high = close + 0.0004
    low = close - 0.0004
    open_ = close.copy()
    # 15:00 UTC = 10:00 NY in January. FVG completes there; tag at 15:15.
    i_fvg = int(np.flatnonzero(times.hour == 15)[0])
    high[i_fvg - 2] = 1.0800
    low[i_fvg - 2] = 1.0796
    high[i_fvg] = 1.0812
    low[i_fvg] = 1.0808
    close[i_fvg] = 1.0810
    high[i_fvg + 1] = 1.0809
    low[i_fvg + 1] = 1.0801
    close[i_fvg + 1] = 1.0804
    df = pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "atr": np.full(n, 0.0008),
        }
    )
    ev = silver_bullet_fvg_events(
        df,
        "EURUSD",
        {"treat_hour": 10, "base_hour": 11, "min_gap_atr": 0.15, "horizon_bars": 4},
        0.8,
        0.2,
    )
    treat = ev.loc[ev["treatment"]] if not ev.empty else ev
    assert not treat.empty
    assert set(treat["side"]) == {"long"}


def test_london_orb_fades_first_break():
    n = 48
    times = pd.date_range("2024-01-08 07:00", periods=n, freq="15min", tz="Europe/London")
    times = times.tz_convert("UTC")
    close = np.full(n, 1.0800)
    high = close + 0.0004
    low = close - 0.0004
    ldn = pd.to_datetime(times, utc=True).tz_convert("Europe/London")
    box = (ldn.hour == 8) & (ldn.minute < 30)
    brk = (ldn.hour == 8) & (ldn.minute == 30)
    high[box] = 1.0804
    low[box] = 1.0796
    close[box] = 1.0800
    high[brk] = 1.0812
    low[brk] = 1.0802
    close[brk] = 1.0810
    df = pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "atr": np.full(n, 0.0008),
        }
    )
    ev = session_box_fade_events(
        df,
        "EURUSD",
        {
            "tz": "Europe/London",
            "box_start_hour": 8,
            "box_end_hour": 8,
            "box_end_minute": 30,
            "hunt_end_hour": 12,
            "box_bars": 2,
            "horizon_bars": 4,
        },
        0.8,
        0.2,
    )
    treat = ev.loc[ev["treatment"]] if not ev.empty else ev
    assert not treat.empty
    assert set(treat["side"]) == {"short"}


def test_prior_hour_fade_london_vs_ny():
    # Monday 2024-01-08. London=UTC, NY=UTC-5.
    times = pd.date_range("2024-01-08 06:00", periods=48, freq="15min", tz="UTC")
    close = np.full(48, 1.0800)
    high = close + 0.0002
    low = close - 0.0002
    utc_h = times.hour
    utc_m = times.minute
    # London 07:00 hour high, tagged at 08:00.
    prior_ldn = utc_h == 7
    tag_ldn = (utc_h == 8) & (utc_m == 0)
    high[prior_ldn] = 1.0810
    low[prior_ldn] = 1.0790
    high[tag_ldn] = 1.0816
    close[tag_ldn] = 1.0812
    # NY 07:00 = 12:00 UTC hour high, tagged at 13:00 UTC = 08:00 NY.
    prior_ny = utc_h == 12
    tag_ny = (utc_h == 13) & (utc_m == 0)
    high[prior_ny] = 1.0820
    low[prior_ny] = 1.0790
    high[tag_ny] = 1.0826
    close[tag_ny] = 1.0822
    df = pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": np.maximum(high, close),
            "low": low,
            "close": close,
            "atr": np.full(48, 0.0008),
        }
    )
    ev = prior_hour_extreme_fade_events(
        df,
        "EURUSD",
        {
            "treat_tz": "Europe/London",
            "base_tz": "America/New_York",
            "window_start_hour": 8,
            "window_end_hour": 12,
            "horizon_bars": 4,
        },
        0.8,
        0.2,
    )
    assert not ev.empty
    treat = ev.loc[ev["treatment"]].sort_values("time")
    base = ev.loc[~ev["treatment"]].sort_values("time")
    assert not treat.empty
    assert not base.empty
    assert treat.iloc[0]["side"] == "short"
    assert base.iloc[0]["side"] == "short"


def test_hour_close_flatten_fades_up_hour():
    times = pd.date_range("2024-01-08 07:00", periods=48, freq="15min", tz="UTC")
    close = np.full(48, 1.0800)
    open_ = np.full(48, 1.0800)
    high = close + 0.0003
    low = close - 0.0003
    utc_h = times.hour
    utc_m = times.minute
    ldn_45 = (utc_h == 8) & (utc_m == 45)
    ny_45 = (utc_h == 13) & (utc_m == 45)
    close[ldn_45] = 1.0815
    high[ldn_45] = 1.0817
    close[ny_45] = 1.0815
    high[ny_45] = 1.0817
    df = pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "atr": np.full(48, 0.0008),
        }
    )
    ev = hour_close_flatten_events(
        df,
        "EURUSD",
        {
            "treat_tz": "Europe/London",
            "base_tz": "America/New_York",
            "window_start_hour": 8,
            "window_end_hour": 12,
            "fire_minute": 45,
            "min_prior_atr": 0.25,
            "horizon_bars": 4,
        },
        0.8,
        0.2,
    )
    assert not ev.empty
    treat = ev.loc[ev["treatment"]].sort_values("time")
    base = ev.loc[~ev["treatment"]].sort_values("time")
    assert not treat.empty
    assert not base.empty
    assert treat.iloc[0]["side"] == "short"
    assert base.iloc[0]["side"] == "short"


def test_hour_open_gap_fades_up_gap():
    times = pd.date_range("2024-01-08 07:00", periods=48, freq="15min", tz="UTC")
    close = np.full(48, 1.0800)
    open_ = np.full(48, 1.0800)
    high = close + 0.0003
    low = close - 0.0003
    utc_h = times.hour
    utc_m = times.minute
    ldn_open = (utc_h == 8) & (utc_m == 0)
    ny_open = (utc_h == 13) & (utc_m == 0)
    open_[ldn_open] = 1.0815
    high[ldn_open] = 1.0817
    open_[ny_open] = 1.0815
    high[ny_open] = 1.0817
    df = pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": np.maximum(high, open_),
            "low": low,
            "close": close,
            "atr": np.full(48, 0.0008),
        }
    )
    ev = hour_open_gap_fade_events(
        df,
        "EURUSD",
        {
            "treat_tz": "Europe/London",
            "base_tz": "America/New_York",
            "window_start_hour": 8,
            "window_end_hour": 12,
            "min_prior_atr": 0.25,
            "horizon_bars": 4,
        },
        0.8,
        0.2,
    )
    assert not ev.empty
    treat = ev.loc[ev["treatment"]].sort_values("time")
    base = ev.loc[~ev["treatment"]].sort_values("time")
    assert not treat.empty
    assert not base.empty
    assert treat.iloc[0]["side"] == "short"
    assert base.iloc[0]["side"] == "short"


def test_prior_close_tag_fades_in_london_and_ny():
    times = pd.date_range("2024-01-08 00:00", periods=192, freq="15min", tz="UTC")
    close = np.full(192, 1.0800)
    high = close + 0.0002
    low = close - 0.0002
    d = times.day
    utc_h = times.hour
    utc_m = times.minute
    # London 8 Jan last close sits above the 9 Jan default range.
    close[(d == 8) & (utc_h == 23) & (utc_m == 45)] = 1.0820
    # NY 8 Jan last bar = 04:45 UTC on 9 Jan.
    close[(d == 9) & (utc_h == 4) & (utc_m == 45)] = 1.0830
    tag_ldn = (d == 9) & (utc_h == 8) & (utc_m == 0)
    high[tag_ldn] = 1.0826
    low[tag_ldn] = 1.0816
    close[tag_ldn] = 1.0824
    tag_ny = (d == 9) & (utc_h == 13) & (utc_m == 0)
    high[tag_ny] = 1.0836
    low[tag_ny] = 1.0826
    close[tag_ny] = 1.0834
    df = pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": np.maximum(high, close),
            "low": np.minimum(low, close),
            "close": close,
            "atr": np.full(192, 0.0008),
        }
    )
    ev = prior_close_tag_fade_events(
        df,
        "EURUSD",
        {
            "treat_tz": "Europe/London",
            "base_tz": "America/New_York",
            "window_start_hour": 8,
            "window_end_hour": 12,
            "horizon_bars": 4,
        },
        0.8,
        0.2,
    )
    assert not ev.empty
    treat = ev.loc[ev["treatment"]].sort_values("time")
    base = ev.loc[~ev["treatment"]].sort_values("time")
    assert not treat.empty
    assert not base.empty
    assert treat.iloc[0]["side"] == "short"
    assert base.iloc[0]["side"] == "short"


def test_session_range_later_fades_london_high_in_ny():
    times = pd.date_range("2024-01-08 08:00", periods=48, freq="15min", tz="UTC")
    n = len(times)
    close = np.full(n, 1.0950)
    high = close + 0.0004
    low = close - 0.0004
    ldn = (times.hour >= 8) & (times.hour < 12)
    high[ldn] = 1.1000
    low[ldn] = 1.0900
    close[ldn] = 1.0950
    ny_hi = int(np.flatnonzero(times.hour == 13)[0])
    high[ny_hi] = 1.1008
    low[ny_hi] = 1.0960
    close[ny_hi] = 1.0990
    ny_mid = int(np.flatnonzero(times.hour == 14)[0])
    high[ny_mid] = 1.0954
    low[ny_mid] = 1.0946
    close[ny_mid] = 1.0952
    df = pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": np.maximum(high, close),
            "low": np.minimum(low, close),
            "close": close,
            "atr": np.full(n, 0.0008),
        }
    )
    ev = session_range_later_fade_events(
        df,
        "EURUSD",
        {
            "source_tz": "Europe/London",
            "source_start_hour": 8,
            "source_end_hour": 12,
            "hunt_tz": "America/New_York",
            "hunt_start_hour": 8,
            "hunt_end_hour": 12,
            "horizon_bars": 4,
        },
        0.8,
        0.2,
    )
    assert not ev.empty
    treat = ev.loc[ev["treatment"]]
    base = ev.loc[~ev["treatment"]]
    assert not treat.empty
    assert treat.iloc[0]["side"] == "short"
    assert not base.empty


def test_session_range_later_fades_ny_am_high_in_ny_afternoon():
    times = pd.date_range("2024-01-08 13:00", periods=36, freq="15min", tz="UTC")
    n = len(times)
    close = np.full(n, 1.0950)
    high = close + 0.0004
    low = close - 0.0004
    ny_am = (times.hour >= 13) & (times.hour < 17)
    high[ny_am] = 1.1000
    low[ny_am] = 1.0900
    close[ny_am] = 1.0950
    pm_hi = int(np.flatnonzero(times.hour == 18)[0])
    high[pm_hi] = 1.1008
    low[pm_hi] = 1.0960
    close[pm_hi] = 1.0990
    pm_mid = int(np.flatnonzero(times.hour == 19)[0])
    high[pm_mid] = 1.0954
    low[pm_mid] = 1.0946
    close[pm_mid] = 1.0952
    df = pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": np.maximum(high, close),
            "low": np.minimum(low, close),
            "close": close,
            "atr": np.full(n, 0.0008),
        }
    )
    ev = session_range_later_fade_events(
        df,
        "EURUSD",
        {
            "source_tz": "America/New_York",
            "source_start_hour": 8,
            "source_end_hour": 12,
            "hunt_tz": "America/New_York",
            "hunt_start_hour": 13,
            "hunt_end_hour": 16,
            "horizon_bars": 4,
        },
        0.8,
        0.2,
    )
    assert not ev.empty
    treat = ev.loc[ev["treatment"]]
    base = ev.loc[~ev["treatment"]]
    assert not treat.empty
    assert treat.iloc[0]["side"] == "short"
    assert not base.empty
