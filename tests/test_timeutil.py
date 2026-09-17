from datetime import time

from ats.timeutil import hyp_key, locked_calendar_split, parse_hhmm, split_book_for, time_in_window, time_splits
import pandas as pd


def test_ny_window_wraps_midnight():
    start = parse_hhmm("18:30")
    end = parse_hhmm("03:00")
    assert time_in_window(time(22, 0), start, end)
    assert time_in_window(time(1, 0), start, end)
    assert not time_in_window(time(12, 0), start, end)


def test_time_splits_are_ordered():
    idx = pd.date_range("2023-01-01", periods=100, freq="D", tz="UTC")
    split = time_splits(idx, 0.6, 0.2)
    assert split.train_end < split.val_end
    assert split.val_end <= split.oos_start


def test_locked_calendar_split_respects_settings():
    settings = {
        "splits": {
            "locked": True,
            "fx_h1": {"train_end": "2025-06-22", "val_end": "2026-01-27"},
        }
    }
    split = locked_calendar_split(settings, "fx_h1")
    assert split is not None
    assert str(split.train_end.date()) == "2025-06-22"
    assert str(split.val_end.date()) == "2026-01-27"
    assert locked_calendar_split({"splits": {"locked": False, "fx_h1": {}}}, "fx_h1") is None


def test_split_book_for_h8_fx_h9_gold():
    assert split_book_for("H8_atr_momentum") == "fx_h1"
    assert split_book_for("H9_cot_spec_fade") == "gold_m15"
    assert split_book_for("H10_rapid_bullet") == "fx_h1"
    assert split_book_for("H11_gold_dxy_relink") == "gold_m15"
    assert split_book_for("H13_gold_ema200_expand") == "gold_m15"
    assert split_book_for("H14_ny_ema_drd") == "gold_m5"
    assert split_book_for("H15_ema200_drd_entry") == "gold_m5"
    assert split_book_for("H16_ema_13_50_200") == "gold_m15"
    assert split_book_for("H17_pdh_pdl_fade") == "gold_m15"
    assert split_book_for("H18_large_bar_fade") == "gold_m15"
    assert split_book_for("H19_session_vwap_extreme") == "gold_m15"
    assert split_book_for("H20_volume_spike_cont") == "gold_m15"
    assert split_book_for("H21_round_number_bounce") == "gold_m15"
    assert split_book_for("H22_ny_box_fade_vs_break") == "gold_m5"
    assert split_book_for("H23_lbma_pm_run") == "gold_m15"
    assert split_book_for("H24_month_end_rebalance") == "gold_m15"
    assert split_book_for("H27_wm_fix_follow") == "fx_h1"
    assert split_book_for("H28_postfix_usd_fade") == "fx_h1"
    assert split_book_for("H30_london_asia_break") == "fx_m5"
    assert split_book_for("H31_h1_tsmom") == "fx_h1"
    assert split_book_for("H48_london_ist_break") == "fx_m15"
    assert split_book_for("H49_london_ist_break_gold") == "gold_m15"
    assert split_book_for("H50_london_fib_bounce") == "fx_m15"
    assert split_book_for("H51_ny_sweep_follow") == "gold_m15"
    assert split_book_for("H52_carry_roll") == "fx_h1"
    assert split_book_for("H53_tokyo_fix_follow") == "fx_m15"
    assert split_book_for("H54_comex_open_follow") == "gold_m15"
    assert split_book_for("H55_nyse_open_eur") == "fx_m15"
    assert split_book_for("H56_comex_close_fade") == "gold_m15"
    assert split_book_for("H193_ovn_intraday_fade_eur") == "fx_h1"
    assert split_book_for("H194_ovn_intraday_fade_gold") == "gold_m15"
    assert split_book_for("H195_gold_h1_tsmom") == "gold_m15"
    assert split_book_for("H196_fedwire18_fade_eur") == "fx_m15"
    assert split_book_for("H198_fedwire18_fade_gold") == "gold_m15"
    assert split_book_for("H207_hk12_fade_gold") == "gold_m15"
    assert split_book_for("H208_tokyo08_follow_jpy") == "fx_m15"
    assert split_book_for("H213_ny19_fade_gold") == "gold_m15"


def test_hyp_key_does_not_confuse_h17_with_h1():
    assert "H17_pdh_pdl_fade".startswith("H1")
    assert hyp_key("H17_pdh_pdl_fade") == "H17"
    assert hyp_key("H1_event_reversal") == "H1"
    assert hyp_key("H19_session_vwap_extreme") == "H19"
