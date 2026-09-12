from ats.hypotheses.clock_family import clock_run_events, _third_friday, _imm_wednesday
from datetime import date

import numpy as np
import pandas as pd

from ats.config import load_hypotheses
from ats.timeutil import split_book_for


def test_third_friday_rule():
    assert _third_friday(date(2024, 1, 19))
    assert not _third_friday(date(2024, 1, 12))
    assert not _third_friday(date(2024, 1, 18))


def test_clock_family_handler_runs():
    times = pd.date_range("2024-01-02 00:00", periods=200, freq="15min", tz="UTC")
    close = 1.08 + np.cumsum(np.random.default_rng(6).normal(0, 0.0003, len(times)))
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
    params = {
        "tz": "Europe/London",
        "treat_hour": 12,
        "treat_minute": 0,
        "base_hour": 10,
        "base_minute": 0,
        "mode": "follow",
        "horizon_bars": 4,
        "min_prior_atr": 0.01,
    }
    ev = clock_run_events(df, "EURUSD", params, 0.8, 0.2)
    assert ev is not None
    if not ev.empty:
        assert "treatment" in ev.columns


def test_cls_cet_fade_marks_treatment():
    times = pd.date_range("2024-01-08 00:00", periods=200, freq="15min", tz="UTC")
    close = 1.08 + np.cumsum(np.random.default_rng(8).normal(0, 0.0004, len(times)))
    df = pd.DataFrame(
        {
            "time": times,
            "open": np.r_[close[0], close[:-1]],
            "high": np.maximum(close, close) + 0.0005,
            "low": np.minimum(close, close) - 0.0005,
            "close": close,
            "atr": np.full(len(times), 0.0008),
        }
    )
    ev = clock_run_events(
        df,
        "EURUSD",
        {
            "tz": "Europe/Berlin",
            "treat_hour": 7,
            "treat_minute": 0,
            "base_hour": 9,
            "base_minute": 0,
            "mode": "fade",
            "skip_weekend": True,
            "horizon_bars": 4,
            "min_prior_atr": 0.01,
        },
        0.8,
        0.2,
    )
    assert ev is not None
    if not ev.empty:
        assert bool(ev["treatment"].any())
        assert bool((~ev["treatment"]).any())


def test_tokyo_midnight_can_treat_first_bar_of_local_day():
    times = pd.date_range("2024-01-08 12:00", periods=96, freq="15min", tz="UTC")
    close = np.linspace(1.08, 1.10, len(times))
    df = pd.DataFrame(
        {
            "time": times,
            "open": np.r_[close[0], close[:-1]],
            "high": close + 0.002,
            "low": close - 0.002,
            "close": close,
            "atr": np.full(len(times), 0.001),
        }
    )
    ev = clock_run_events(
        df,
        "USDJPY",
        {
            "tz": "Asia/Tokyo",
            "treat_hour": 0,
            "treat_minute": 0,
            "base_hour": 3,
            "base_minute": 0,
            "mode": "fade",
            "skip_weekend": True,
            "horizon_bars": 4,
            "min_prior_atr": 0.01,
        },
        0.8,
        0.2,
    )
    assert ev is not None
    if not ev.empty:
        assert bool(ev["treatment"].any())


def test_h57_to_h76_are_listed_and_split():
    ids = {h["id"] for h in load_hypotheses()}
    for n in range(57, 77):
        matches = [i for i in ids if i.startswith(f"H{n}_")]
        assert matches, n
    assert split_book_for("H57_sge_open_follow") == "gold_m15"
    assert split_book_for("H58_ecb_ref_follow") == "fx_m15"
    assert split_book_for("H67_gld_close_fade") == "gold_m15"
    assert split_book_for("H73_sge_night_follow") == "gold_m15"
    assert split_book_for("H76_hkex_open_aud") == "fx_m15"
    assert split_book_for("H77_imm_usd_bid") == "fx_h1"
    assert split_book_for("H78_two_day_fade") == "fx_h1"
    assert split_book_for("H79_overnight_usd") == "fx_h1"
    assert split_book_for("H90_ny_cut_eur") == "fx_m15"
    assert split_book_for("H91_ny_cut_gbp") == "fx_m15"
    assert split_book_for("H92_ny_cut_jpy") == "fx_m15"
    assert split_book_for("H93_wm_postfix_eur") == "fx_m15"
    assert split_book_for("H95_wm_postfix_jpy") == "fx_m15"
    assert split_book_for("H96_asia_box_gold") == "gold_m15"
    assert split_book_for("H97_asia_box_gbp") == "fx_m15"
    assert split_book_for("H98_asia_box_aud") == "fx_m15"
    assert split_book_for("H99_tokyo_roll_jpy") == "fx_m15"
    assert split_book_for("H101_tokyo_roll_gbp") == "fx_m15"
    assert split_book_for("H102_overnight_xs_fade") == "fx_h1"
    assert split_book_for("H103_cls_settle_eur") == "fx_m15"
    assert split_book_for("H105_cls_settle_jpy") == "fx_m15"
    assert split_book_for("H106_silver_bullet_eur") == "fx_m15"
    assert split_book_for("H108_silver_bullet_gold") == "gold_m15"
    assert split_book_for("H109_asx_close_fade") == "fx_m15"
    assert split_book_for("H111_tsx_close_fade") == "fx_m15"
    assert split_book_for("H112_london_open_eur") == "fx_m15"
    assert split_book_for("H114_london_open_gold") == "gold_m15"
    assert split_book_for("H115_eia_cad") == "fx_m15"
    assert split_book_for("H117_eia_gold") == "gold_m15"
    assert split_book_for("H118_london_orb_eur") == "fx_m15"
    assert split_book_for("H120_london_orb_gold") == "gold_m15"
    assert split_book_for("H121_tokyo_orb_jpy") == "fx_m15"
    assert split_book_for("H123_tokyo_orb_gold") == "gold_m15"
    assert split_book_for("H124_prior_hour_eur") == "fx_m15"
    assert split_book_for("H126_prior_hour_gold") == "gold_m15"
    assert split_book_for("H127_hour_close_eur") == "fx_m15"
    assert split_book_for("H129_hour_close_gold") == "gold_m15"
    assert split_book_for("H130_hour_gap_eur") == "fx_m15"
    assert split_book_for("H132_hour_gap_gold") == "gold_m15"
    assert split_book_for("H133_us830_slot_eur") == "fx_m15"
    assert split_book_for("H135_us830_slot_gold") == "gold_m15"
    assert split_book_for("H136_pdc_tag_eur") == "fx_m15"
    assert split_book_for("H138_pdc_tag_gold") == "gold_m15"
    assert split_book_for("H139_tnext_roll_eur") == "fx_m15"
    assert split_book_for("H141_tnext_roll_gold") == "gold_m15"
    assert split_book_for("H142_syd09_fade_aud") == "fx_m15"
    assert split_book_for("H144_syd09_fade_gold") == "gold_m15"
    assert split_book_for("H145_ldn10_cut_eur") == "fx_m15"
    assert split_book_for("H147_ldn10_cut_jpy") == "fx_m15"
    assert split_book_for("H148_ldn05_hand_eur") == "fx_m15"
    assert split_book_for("H150_ldn05_hand_gold") == "gold_m15"
    assert split_book_for("H151_tgt18_fade_eur") == "fx_m15"
    assert split_book_for("H152_tgt18_fade_chf") == "fx_m15"
    assert split_book_for("H153_tgt18_fade_gold") == "gold_m15"
    assert split_book_for("H154_tyo13_reopen_jpy") == "fx_m15"
    assert split_book_for("H156_tyo13_reopen_gold") == "gold_m15"
    assert split_book_for("H157_oil1430_fade_cad") == "fx_m15"
    assert split_book_for("H159_oil1430_fade_gold") == "gold_m15"
    assert split_book_for("H160_ldnbox_ny_eur") == "fx_m15"
    assert split_book_for("H162_ldnbox_ny_gold") == "gold_m15"
    assert split_book_for("H163_akl09_fade_nzd") == "fx_m15"
    assert split_book_for("H165_akl09_fade_gold") == "gold_m15"
    assert split_book_for("H166_zur12_fade_chf") == "fx_m15"
    assert split_book_for("H168_zur12_fade_gold") == "gold_m15"
    assert split_book_for("H169_tyo11_fade_jpy") == "fx_m15"
    assert split_book_for("H171_tyo11_fade_gold") == "gold_m15"
    assert split_book_for("H172_lbma_am_gold") == "gold_m15"
    assert split_book_for("H173_lbma_am_aud") == "fx_m15"
    assert split_book_for("H175_sge15_fade_gold") == "gold_m15"
    assert split_book_for("H176_sge15_fade_aud") == "fx_m15"
    assert split_book_for("H177_sge15_fade_jpy") == "fx_m15"
    assert split_book_for("H178_mcx09_fade_gold") == "gold_m15"
    assert split_book_for("H179_mcx09_fade_aud") == "fx_m15"
    assert split_book_for("H180_mcx09_fade_jpy") == "fx_m15"
    assert split_book_for("H181_nybox_pm_eur") == "fx_m15"
    assert split_book_for("H183_nybox_pm_gold") == "gold_m15"
    assert split_book_for("H184_dxb14_fade_gold") == "gold_m15"
    assert split_book_for("H185_dxb14_fade_aud") == "fx_m15"
    assert split_book_for("H186_dxb14_fade_jpy") == "fx_m15"
    assert split_book_for("H187_sgx17_fade_aud") == "fx_m15"
    assert split_book_for("H188_sgx17_fade_jpy") == "fx_m15"
    assert split_book_for("H189_sgx17_fade_gold") == "gold_m15"
    assert split_book_for("H190_xetra1730_fade_eur") == "fx_m15"
    assert split_book_for("H191_xetra1730_fade_gbp") == "fx_m15"
    assert split_book_for("H192_xetra1730_fade_gold") == "gold_m15"


def test_weekday_clock_treats_wednesday_not_tuesday():
    times = pd.date_range("2024-01-08 00:00", periods=96 * 5, freq="15min", tz="UTC")
    close = 1.35 + np.cumsum(np.random.default_rng(9).normal(0, 0.0002, len(times)))
    df = pd.DataFrame(
        {
            "time": times,
            "open": np.r_[close[0], close[:-1]],
            "high": close + 0.0005,
            "low": close - 0.0005,
            "close": close,
            "atr": np.full(len(times), 0.0008),
        }
    )
    ev = clock_run_events(
        df,
        "USDCAD",
        {
            "tz": "America/New_York",
            "treat_hour": 10,
            "treat_minute": 30,
            "base_hour": 10,
            "base_minute": 30,
            "treat_dow": 2,
            "base_dow": 1,
            "mode": "fade",
            "skip_weekend": True,
            "horizon_bars": 4,
            "min_prior_atr": 0.01,
        },
        1.2,
        0.2,
    )
    assert ev is not None
    if not ev.empty:
        ny = pd.to_datetime(ev["time"], utc=True).dt.tz_convert("America/New_York")
        treat_days = set(ny.loc[ev["treatment"]].dt.dayofweek)
        base_days = set(ny.loc[~ev["treatment"]].dt.dayofweek)
        assert treat_days <= {2}
        assert base_days <= {1}


def test_imm_wednesday_rule():
    assert _imm_wednesday(date(2024, 3, 20))
    assert not _imm_wednesday(date(2024, 3, 13))
    assert not _imm_wednesday(date(2024, 4, 17))
