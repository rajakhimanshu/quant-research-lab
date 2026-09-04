import yaml

from ats.config import CONFIG_DIR
from ats.config import load_hypotheses


def test_every_hypothesis_has_a_why():
    items = load_hypotheses()
    assert items
    for item in items:
        assert item["why"].strip()


def test_hypotheses_file_lists_h1_and_h2():
    raw = yaml.safe_load((CONFIG_DIR / "hypotheses.yaml").read_text(encoding="utf-8"))
    ids = {h["id"] for h in raw["hypotheses"]}
    assert "H1_event_reversal" in ids
    assert "H2_tap_breakout" in ids
    assert "H8_atr_momentum" in ids
    assert "H10_rapid_bullet" in ids
    assert "H9_cot_spec_fade" in ids
    assert "H11_gold_dxy_relink" in ids
    assert "H12_london_close_fade" in ids
    assert "H13_gold_ema200_expand" in ids
    assert "H14_ny_ema_drd" in ids
    assert "H15_ema200_drd_entry" in ids
    assert "H16_ema_13_50_200" in ids
    assert "H17_pdh_pdl_fade" in ids
    assert "H18_large_bar_fade" in ids
    assert "H19_session_vwap_extreme" in ids
    assert "H20_volume_spike_cont" in ids
    assert "H21_round_number_bounce" in ids
    assert "H22_ny_box_fade_vs_break" in ids
    assert "H23_lbma_pm_run" in ids
    assert "H24_month_end_rebalance" in ids
    assert "H25_streak_inventory_fade" in ids
    assert "H26_weekend_gap_fade" in ids
    assert "H27_wm_fix_follow" in ids
    assert "H28_postfix_usd_fade" in ids
    assert "H29_pre_ecb_usd" in ids
    assert "H30_london_asia_break" in ids
    assert "H31_h1_tsmom" in ids
    assert "H32_ranaldo_local_hours" in ids
    assert "H33_inside_bar_break" in ids
    assert "H34_compression_expand" in ids
    assert "H35_friday_flatten" in ids
    assert "H36_overlap_continuation" in ids
    assert "H37_weekend_gap_fx" in ids
    assert "H38_xs_momentum" in ids
    assert "H39_tokyo_lunch_jpy" in ids
    assert "H40_ny_close_asia_fade" in ids
    assert "H41_prior_day_range_fade" in ids
    assert "H42_tokyo_close_flatten" in ids
    assert "H43_month_end_usd" in ids
