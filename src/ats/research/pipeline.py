from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ats.config import DATA_DIR, load_hypotheses, load_settings
from ats.data.calendar import load_calendar
from ats.data.mt5_client import load_raw
from ats.features.prepare import prepare_frame
from ats.hypotheses.atr_momentum import atr_momentum_events
from ats.hypotheses.cot_spec_fade import cot_spec_fade_events
from ats.hypotheses.gold_dxy_relink import gold_dxy_relink_events
from ats.hypotheses.gold_ema200_expand import gold_ema200_expand_events
from ats.hypotheses.london_close_fade import london_close_fade_events
from ats.hypotheses.clock_family import clock_run_events
from ats.hypotheses.london_ist import london_fib_bounce_events, london_ist_break_events
from ats.hypotheses.rapid_bullet import rapid_bullet_events
from ats.hypotheses.equity_rsi2 import equity_rsi2_events
from ats.hypotheses.event_reversal import event_reversal_events
from ats.hypotheses.ema_13_50_200 import ema_13_50_200_events
from ats.hypotheses.london_close import london_close_events
from ats.hypotheses.bb_squeeze import bb_squeeze_events
from ats.hypotheses.london_breakout import london_breakout_events
from ats.hypotheses.london_breakout_fade import london_breakout_fade_events
from ats.hypotheses.tuesday_turnaround import tuesday_turnaround_events
from ats.hypotheses.rsi_extreme_fade import rsi_extreme_fade_events
from ats.hypotheses.trend_pullback import trend_pullback_events
from ats.hypotheses.bb_exhaustion import bb_exhaustion_events
from ats.hypotheses.engulfing_pullback import engulfing_pullback_events
from ats.hypotheses.mtf_pullback import mtf_pullback_events
from ats.hypotheses.pure_momentum import pure_momentum_events
from ats.hypotheses.daily_engulfing import daily_engulfing_events
from ats.hypotheses.pdhl_sweep import pdhl_sweep_events
from ats.hypotheses.rsi_divergence import rsi_divergence_events
from ats.hypotheses.trend_continuation import trend_continuation_events
from ats.hypotheses.strong_trend import strong_trend_events
from ats.hypotheses.rsi_extreme import rsi_extreme_events
from ats.hypotheses.ema200_drd_entry import ema200_drd_entry_events
from ats.hypotheses.ny_ema_drd import ny_ema_drd_events
from ats.hypotheses.fx_session import (
    big_figure_fade_events,
    compression_expand_events,
    friday_flatten_events,
    h1_tsmom_events,
    inside_bar_break_events,
    london_asia_break_events,
    london_lunch_fade_events,
    month_end_usd_events,
    ny_close_asia_fade_events,
    overlap_continuation_events,
    postfix_usd_fade_events,
    pre_ecb_usd_events,
    prior_day_range_fade_events,
    ranaldo_local_hours_events,
    stop_pool_20_fade_events,
    tokyo_close_flatten_events,
    tokyo_lunch_fade_events,
    weekend_gap_fx_events,
    monday_cash_gap_events,
    wm_postfix_fade_events,
    xs_momentum_events,
    overnight_xs_fade_events,
    overnight_intraday_fade_events,
    m15_to_h1,
    carry_roll_events,
    tokyo_fix_follow_events,
    nyse_open_fx_events,
    two_day_streak_fade_events,
    print_window_fade_events,
)
from ats.hypotheses.forced_flow import (
    lbma_pm_run_events,
    month_end_rebalance_events,
    streak_inventory_fade_events,
    weekend_gap_fade_events,
    wm_fix_follow_events,
    comex_session_run_events,
)
from ats.hypotheses.m15_micro import (
    asia_range_london_fade_events,
    large_bar_fade_events,
    ny_box_fade_events,
    pdh_pdl_fade_events,
    round_bounce_events,
    hour_close_flatten_events,
    hour_open_gap_fade_events,
    prior_close_tag_fade_events,
    prior_hour_extreme_fade_events,
    session_box_fade_events,
    session_range_later_fade_events,
    silver_bullet_fvg_events,
    volume_spike_cont_events,
    vwap_extreme_fade_events,
)
from ats.hypotheses.gulf_dual_liq import gulf_dual_liq_fade_events
from ats.hypotheses.ny_open_sweep import ny_open_sweep_events, ny_sweep_follow_events
from ats.hypotheses.session_orb import session_orb_events
from ats.hypotheses.tap_breakout import tap_events
from ats.hypotheses.cross_pair_lag import cross_pair_lag_events
from ats.hypotheses.three_bar_exhaustion import three_bar_exhaustion_events
from ats.hypotheses.stop_cascade import stop_cascade_absorption_events
from ats.hypotheses.post_ny_drift import post_ny_drift_reversion_events
from ats.hypotheses.eur_gbp_divergence import eur_gbp_divergence_events
from ats.hypotheses.div_filtered_london_asia import div_filtered_london_asia_events
from ats.hypotheses.div_filtered_rsi import div_filtered_rsi_events
from ats.hypotheses.h1_vol_exhaustion import h1_vol_exhaustion_events
from ats.hypotheses.asia_mean_reversion import asia_mean_reversion_events
from ats.hypotheses.h1_weekly_reversal import h1_weekly_reversal_events
from ats.research.stats import HypothesisReport, rates
from ats.timeutil import hyp_key, time_splits, locked_calendar_split, split_book_for

LOCKED = "LOCKED — re-run with --unlock-oos after you pick a survivor. Do not peek."


def load_frames(symbols: list[str], timeframe: str, settings: dict) -> dict[str, pd.DataFrame]:
    frames = {}
    for symbol in symbols:
        raw = load_raw(symbol, timeframe)
        frames[symbol] = prepare_frame(raw, settings)
    return frames


def _events_for(hyp: dict, frames: dict[str, pd.DataFrame], settings: dict) -> pd.DataFrame:
    calendar = load_calendar()
    costs = settings["costs"]
    chunks = []
    merged_params = {**settings["features"], **(hyp.get("params") or {})}
    hid = hyp["id"]
    key = hyp_key(hid)
    if hyp.get("template"):
        from ats.lab.templates import template_events

        ctx = {
            "timeframe": merged_params.get("timeframe"),
            "load": lambda sym, tf: load_frames([sym], tf, settings)[sym],
        }
        for symbol, df in frames.items():
            spread = float(costs["spread_pips"].get(symbol, 1.5))
            slip = float((costs.get("slippage_by_symbol") or {}).get(symbol, costs["slippage_pips"]))
            print(f"  scoring {hid} [{hyp['template']}] on {symbol} ({len(df)} bars)")
            ev = template_events(hyp["template"], df, symbol, merged_params, spread, slip, ctx)
            if ev is not None and not ev.empty:
                chunks.append(ev)
        return pd.concat(chunks, ignore_index=True).sort_values("time") if chunks else pd.DataFrame()
    if key == "H38":
        print(f"  scoring {hid} cross-section ({len(frames)} pairs)")
        return xs_momentum_events(frames, merged_params, settings)
    if key == "H102":
        print(f"  scoring {hid} overnight XS ({len(frames)} pairs)")
        return overnight_xs_fade_events(frames, merged_params, settings)
    for symbol, df in frames.items():
        spread = float(costs["spread_pips"].get(symbol, 1.5))
        slip = float((costs.get("slippage_by_symbol") or {}).get(symbol, costs["slippage_pips"]))
        print(f"  scoring {hid} on {symbol} ({len(df)} bars)")
        if key == "H1":
            ev = event_reversal_events(df, symbol, merged_params, calendar, spread, slip)
        elif key == "H2":
            ev = tap_events(df, symbol, merged_params)
            if not ev.empty:
                ev = ev.loc[ev["treatment"] | ev["tap_no"].isin([1, 2])].copy()
        elif key in {"H3", "H6"}:
            ev = ny_open_sweep_events(df, symbol, merged_params, spread, slip)
        elif key in {"H4", "H7"}:
            ev = session_orb_events(df, symbol, merged_params, spread, slip)
        elif key == "H8":
            ev = atr_momentum_events(df, symbol, merged_params, spread, slip)
        elif key == "H9":
            ev = cot_spec_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H10":
            ev = rapid_bullet_events(df, symbol, merged_params, spread, slip)
        elif key == "H11":
            ev = gold_dxy_relink_events(df, symbol, merged_params, spread, slip)
        elif key == "H12":
            ev = london_close_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H13":
            ev = gold_ema200_expand_events(df, symbol, merged_params, spread, slip)
        elif key == "H14":
            ev = ny_ema_drd_events(df, symbol, merged_params, spread, slip)
        elif key == "H15":
            ev = ema200_drd_entry_events(df, symbol, merged_params, spread, slip)
        elif key == "H16":
            ev = ema_13_50_200_events(df, symbol, merged_params, spread, slip)
        elif key == "H17":
            ev = pdh_pdl_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H18":
            ev = large_bar_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H19":
            ev = vwap_extreme_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H20":
            ev = volume_spike_cont_events(df, symbol, merged_params, spread, slip)
        elif key == "H21":
            ev = round_bounce_events(df, symbol, merged_params, spread, slip)
        elif key == "H22":
            ev = ny_box_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H23":
            ev = lbma_pm_run_events(df, symbol, merged_params, spread, slip)
        elif key == "H24":
            ev = month_end_rebalance_events(df, symbol, merged_params, spread, slip)
        elif key == "H25":
            ev = streak_inventory_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H26":
            ev = weekend_gap_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H27":
            ev = wm_fix_follow_events(df, symbol, merged_params, spread, slip)
        elif key == "H28":
            ev = postfix_usd_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H29":
            ev = pre_ecb_usd_events(df, symbol, merged_params, spread, slip)
        elif key == "H30":
            ev = london_asia_break_events(df, symbol, merged_params, spread, slip)
        elif key == "H31":
            ev = h1_tsmom_events(df, symbol, merged_params, spread, slip)
        elif key == "H32":
            ev = ranaldo_local_hours_events(df, symbol, merged_params, spread, slip)
        elif key == "H33":
            ev = inside_bar_break_events(df, symbol, merged_params, spread, slip)
        elif key == "H34":
            ev = compression_expand_events(df, symbol, merged_params, spread, slip)
        elif key == "H35":
            ev = friday_flatten_events(df, symbol, merged_params, spread, slip)
        elif key == "H36":
            ev = overlap_continuation_events(df, symbol, merged_params, spread, slip)
        elif key == "H37":
            ev = weekend_gap_fx_events(df, symbol, merged_params, spread, slip)
        elif key == "H39":
            ev = tokyo_lunch_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H40":
            ev = ny_close_asia_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H41":
            ev = prior_day_range_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H42":
            ev = tokyo_close_flatten_events(df, symbol, merged_params, spread, slip)
        elif key == "H43":
            ev = month_end_usd_events(df, symbol, merged_params, spread, slip)
        elif key == "H44":
            ev = wm_postfix_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H45":
            ev = big_figure_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H46":
            ev = stop_pool_20_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H47":
            ev = london_lunch_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H48":
            ev = london_ist_break_events(df, symbol, merged_params, spread, slip)
        elif key == "H49":
            ev = london_ist_break_events(df, symbol, merged_params, spread, slip)
        elif key == "H50":
            ev = london_fib_bounce_events(df, symbol, merged_params, spread, slip)
        elif key == "H51":
            ev = ny_sweep_follow_events(df, symbol, merged_params, spread, slip)
        elif key == "H52":
            ev = carry_roll_events(df, symbol, merged_params, spread, slip)
        elif key == "H53":
            ev = tokyo_fix_follow_events(df, symbol, merged_params, spread, slip)
        elif key == "H54":
            ev = comex_session_run_events(df, symbol, merged_params, spread, slip)
        elif key == "H55":
            ev = nyse_open_fx_events(df, symbol, merged_params, spread, slip)
        elif key == "H56":
            ev = comex_session_run_events(df, symbol, merged_params, spread, slip)
        elif key in {f"H{i}" for i in range(57, 77)}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H77", "H79"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key == "H78":
            ev = two_day_streak_fade_events(df, symbol, merged_params, spread, slip)
        elif key in {"H80", "H81"}:
            ev = print_window_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H82":
            ev = month_end_usd_events(df, symbol, merged_params, spread, slip)
        elif key in {"H83", "H84", "H85"}:
            ev = cot_spec_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H86":
            ev = weekend_gap_fx_events(df, symbol, merged_params, spread, slip)
        elif key in {"H87", "H88", "H89"}:
            ev = monday_cash_gap_events(df, symbol, merged_params, spread, slip)
        elif key in {"H90", "H91", "H92", "H93", "H94", "H95"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H96", "H97", "H98"}:
            ev = asia_range_london_fade_events(df, symbol, merged_params, spread, slip)
        elif key in {"H99", "H100", "H101"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H103", "H104", "H105"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H106", "H107", "H108"}:
            ev = silver_bullet_fvg_events(df, symbol, merged_params, spread, slip)
        elif key in {"H109", "H110", "H111"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H112", "H113", "H114"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H115", "H116", "H117"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H118", "H119", "H120"}:
            ev = session_box_fade_events(df, symbol, merged_params, spread, slip)
        elif key in {"H121", "H122", "H123"}:
            ev = session_box_fade_events(df, symbol, merged_params, spread, slip)
        elif key in {"H124", "H125", "H126"}:
            ev = prior_hour_extreme_fade_events(df, symbol, merged_params, spread, slip)
        elif key in {"H127", "H128", "H129"}:
            ev = hour_close_flatten_events(df, symbol, merged_params, spread, slip)
        elif key in {"H130", "H131", "H132"}:
            ev = hour_open_gap_fade_events(df, symbol, merged_params, spread, slip)
        elif key in {"H133", "H134", "H135"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H136", "H137", "H138"}:
            ev = prior_close_tag_fade_events(df, symbol, merged_params, spread, slip)
        elif key in {"H139", "H140", "H141"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H142", "H143", "H144"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H145", "H146", "H147"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H148", "H149", "H150"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H151", "H152", "H153"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H154", "H155", "H156"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H157", "H158", "H159"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H160", "H161", "H162"}:
            ev = session_range_later_fade_events(df, symbol, merged_params, spread, slip)
        elif key in {"H163", "H164", "H165"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H166", "H167", "H168"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H169", "H170", "H171"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H172", "H173", "H174"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H175", "H176", "H177"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H178", "H179", "H180"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H181", "H182", "H183"}:
            ev = session_range_later_fade_events(df, symbol, merged_params, spread, slip)
        elif key in {"H184", "H185", "H186"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H187", "H188", "H189"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H190", "H191", "H192"}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key in {"H193", "H194"}:
            ev = overnight_intraday_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H195":
            # Gold has M15 history only in raw/; resample to H1 for Moskowitz-style TSMOM.
            h1 = m15_to_h1(df)
            ev = h1_tsmom_events(h1, symbol, merged_params, spread, slip)
        elif key in {f"H{i}" for i in range(196, 214)}:
            ev = clock_run_events(df, symbol, merged_params, spread, slip)
        elif key == "H214":
            ev = gulf_dual_liq_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H219":
            ev = london_close_events(df, symbol, merged_params, spread, slip)
        elif key == "H220":
            ev = bb_squeeze_events(df, symbol, merged_params, spread, slip)
        elif key == "H221":
            ev = london_breakout_events(df, symbol, merged_params, spread, slip)
        elif key == "H222":
            ev = london_breakout_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H223":
            ev = tuesday_turnaround_events(df, symbol, merged_params, spread, slip)
        elif key == "H224":
            ev = rsi_extreme_fade_events(df, symbol, merged_params, spread, slip)
        elif key == "H225":
            ev = trend_pullback_events(df, symbol, merged_params, spread, slip)
        elif key == "H226":
            ev = bb_exhaustion_events(df, symbol, merged_params, spread, slip)
        elif key == "H227":
            ev = engulfing_pullback_events(df, symbol, merged_params, spread, slip)
        elif key == "H228":
            ev = mtf_pullback_events(df, symbol, merged_params, spread, slip)
        elif key == "H229":
            ev = pure_momentum_events(df, symbol, merged_params, spread, slip)
        elif key == "H230":
            ev = london_close_events(df, symbol, merged_params, spread, slip)
        elif key == "H231":
            ev = daily_engulfing_events(df, symbol, merged_params, spread, slip)
        elif key == "H232":
            ev = pdhl_sweep_events(df, symbol, merged_params, spread, slip)
        elif key == "H233":
            ev = rsi_divergence_events(df, symbol, merged_params, spread, slip)
        elif key == "H234":
            ev = trend_continuation_events(df, symbol, merged_params, spread, slip)
        elif key == "H235":
            ev = strong_trend_events(df, symbol, merged_params, spread, slip)
        elif key == "H236":
            ev = rsi_extreme_events(df, symbol, merged_params, spread, slip)
        elif key in {"H237", "H238"}:
            ev = cross_pair_lag_events(df, symbol, merged_params, spread, slip)
        elif key == "H239":
            ev = three_bar_exhaustion_events(df, symbol, merged_params, spread, slip)
        elif key == "H240":
            ev = stop_cascade_absorption_events(df, symbol, merged_params, spread, slip)
        elif key == "H241":
            ev = post_ny_drift_reversion_events(df, symbol, merged_params, spread, slip)
        elif key in {"H242", "H245"}:
            ev = eur_gbp_divergence_events(df, symbol, merged_params, spread, slip)
        elif key == "H243":
            ev = div_filtered_london_asia_events(df, symbol, merged_params, spread, slip)
        elif key == "H244":
            ev = div_filtered_rsi_events(df, symbol, merged_params, spread, slip)
        elif key == "H246":
            ev = h1_vol_exhaustion_events(df, symbol, merged_params, spread, slip)
        elif key == "H247":
            ev = asia_mean_reversion_events(df, symbol, merged_params, spread, slip)
        elif key == "H248":
            ev = h1_weekly_reversal_events(df, symbol, merged_params, spread, slip)
        else:
            raise ValueError(f"Unknown hypothesis {hid}")
        if ev is not None and not ev.empty:
            chunks.append(ev)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).sort_values("time")


def _mean_r(events: pd.DataFrame, mask: pd.Series) -> float | None:
    if "r_mult" not in events.columns:
        return None
    part = events.loc[mask & events["treatment"]]
    if part.empty:
        return None
    return float(part["r_mult"].mean())


def mirage_reason(train_r: float | None, val_r: float | None) -> str | None:
    """A hit-rate win with mean R <= 0 after costs is not an edge."""
    if train_r is None:
        return None
    if train_r <= 0 or val_r is None or val_r <= 0:
        shown = None if val_r is None else round(val_r, 3)
        return f"hit-rate mirage: mean R train={train_r:.3f} val={shown} not > 0 after costs"
    return None


def _part_rates(events: pd.DataFrame, mask: pd.Series, success_col: str = "success") -> dict:
    sub = events.loc[mask]
    if sub.empty:
        return rates(pd.Series(dtype=bool), pd.Series(dtype=bool)).__dict__
    return rates(sub[success_col], sub["treatment"]).__dict__


def decide(
    train: dict,
    validation: dict,
    settings: dict,
    n_tests: int,
    notes: list[str],
) -> tuple[str, str | None]:
    v = settings["validation"]
    min_n = int(v["min_trades"])
    max_gap = float(v["max_train_val_gap"])
    alpha = float(v["significance_alpha"]) / max(n_tests, 1)

    if train["n"] < min_n:
        return "NEEDS_MORE_DATA", f"train treatment n={train['n']} < {min_n}"
    if not (train["rate"] == train["rate"]):  # NaN
        return "REJECT", "no treatment events"
    if not train.get("baseline_n"):
        return "REJECT", "no baseline arm (treatment-only design cannot be tested)"
    if train["rate"] <= train["baseline_rate"]:
        return "REJECT", "train success is not above baseline"
    if train["p_value"] is None or train["p_value"] > alpha:
        return "REJECT", f"train p={train['p_value']} not < {alpha:.4f} (multiple-testing adjusted)"
    if validation["n"] < max(30, min_n // 4):
        return "NEEDS_MORE_DATA", f"validation n={validation['n']} too small"
    gap = abs(train["rate"] - validation["rate"])
    if gap > max_gap:
        return "REJECT", f"train-val gap {gap:.3f} > {max_gap} (overfit signature)"
    if validation["rate"] <= validation["baseline_rate"]:
        return "REJECT", "validation success is not above baseline"
    return "CANDIDATE", None


def evaluate(
    hyp: dict,
    frames: dict[str, pd.DataFrame],
    settings: dict,
    n_tests: int,
    unlock_oos: bool,
) -> HypothesisReport:
    notes: list[str] = []
    hid = hyp["id"]
    key = hyp_key(hid)
    if key == "H5":
        print("  scoring H5_equity_rsi2 via Yahoo daily")
        events = equity_rsi2_events(hyp.get("params") or {})
        frames = frames or {}
    elif key == "H9":
        from ats.ideas.cot import combined_path

        if not combined_path().exists():
            return HypothesisReport(
                hypothesis=hid,
                name=hyp["name"],
                why=hyp["why"].strip(),
                sample_size=0,
                success_rate=None,
                baseline_rate=None,
                p_value=None,
                train={},
                validation={},
                oos={"status": "skipped"},
                regime_breakdown={},
                cost_adjusted_success_rate=None,
                train_val_gap=None,
                decision="NEEDS_DATA",
                reject_reason="No data/cot/combined.csv — run python -m ats ideas cot",
                notes=["H9 uses official CFTC weekly files, not a scrape."],
            )
    elif key == "H1" and load_calendar().empty:
        return HypothesisReport(
            hypothesis=hid,
            name=hyp["name"],
            why=hyp["why"].strip(),
            sample_size=0,
            success_rate=None,
            baseline_rate=None,
            p_value=None,
            train={},
            validation={},
            oos={"status": "skipped"},
            regime_breakdown={},
            cost_adjusted_success_rate=None,
            train_val_gap=None,
            decision="NEEDS_DATA",
            reject_reason="No data/calendar/high_impact.csv — H1 cannot run",
            notes=["Export a Forex Factory calendar to data/calendar/high_impact.csv"],
        )

    if key != "H5":
        events = _events_for(hyp, frames, settings)
    if events.empty:
        return HypothesisReport(
            hypothesis=hid,
            name=hyp["name"],
            why=hyp["why"].strip(),
            sample_size=0,
            success_rate=None,
            baseline_rate=None,
            p_value=None,
            train={},
            validation={},
            oos={"status": "empty"},
            regime_breakdown={},
            cost_adjusted_success_rate=None,
            train_val_gap=None,
            decision="NEEDS_MORE_DATA",
            reject_reason="zero events produced",
            notes=notes,
        )

    v = settings["validation"]
    times = pd.to_datetime(events["time"], utc=True)
    idx = pd.DatetimeIndex(times)
    book = hyp.get("split_book") or split_book_for(hid)
    split = locked_calendar_split(settings, book)
    if split is None:
        split = time_splits(idx, float(v["train"]), float(v["validation"]))
    notes.append(f"split {book} train_end={split.train_end} val_end={split.val_end}")
    train_m = times <= split.train_end
    val_m = (times > split.train_end) & (times <= split.val_end)
    oos_m = times > split.val_end

    train = _part_rates(events, train_m)
    validation = _part_rates(events, val_m)
    if unlock_oos:
        oos = _part_rates(events, oos_m)
    else:
        oos = {"status": LOCKED, "n": int(oos_m.sum())}

    seen = events if unlock_oos else events.loc[~oos_m]
    treated = seen.loc[seen["treatment"]]
    overall = rates(seen["success"], seen["treatment"])
    cost_rate = None
    if "success_cost_adj" in events.columns and not treated.empty:
        cost_rate = float(treated["success_cost_adj"].mean())

    regime = {}
    if "trend_regime" in events.columns and not treated.empty:
        for name, g in treated.groupby("trend_regime"):
            regime[str(name)] = float(g["success"].mean())

    decision, reason = decide(train, validation, settings, n_tests, notes)
    gap = None
    if train.get("n") and validation.get("n"):
        gap = abs(train["rate"] - validation["rate"])

    if "r_mult" in events.columns:
        for label, mask in ("train", train_m), ("validation", val_m), ("oos", oos_m):
            part = events.loc[mask & events["treatment"]]
            if part.empty or not unlock_oos and label == "oos":
                continue
            notes.append(
                f"{label} treatment mean R={float(part['r_mult'].mean()):.4f} "
                f"n={len(part)} total R={float(part['r_mult'].sum()):.1f}"
            )

    if decision == "CANDIDATE":
        mirage = mirage_reason(_mean_r(events, train_m), _mean_r(events, val_m))
        if mirage:
            decision, reason = "REJECT", mirage

    if decision == "CANDIDATE" and not unlock_oos:
        notes.append("OOS is locked until you explicitly unlock it.")
    elif decision == "CANDIDATE" and unlock_oos:
        min_oos = max(30, int(v["min_trades"]) // 4)
        oos_base = oos.get("baseline_rate")
        oos_r = _mean_r(events, oos_m)
        if oos.get("n", 0) < min_oos:
            decision, reason = "NEEDS_MORE_DATA", f"oos n={oos.get('n')} too small"
        elif oos.get("rate") is None or oos_base is None or oos_base != oos_base or oos["rate"] <= oos_base:
            decision, reason = "REJECT", "oos success is not above baseline"
        elif oos_r is not None and oos_r <= 0:
            decision, reason = "REJECT", f"oos mean R={oos_r:.3f} not > 0 after costs"
        else:
            gap_oos = abs(train["rate"] - oos["rate"])
            if gap_oos > float(v["max_train_val_gap"]):
                decision, reason = "REJECT", f"train-oos gap {gap_oos:.3f} > {v['max_train_val_gap']}"
            else:
                decision = "OOS_PASS"
                notes.append("OOS unlocked after survivor pick. Not a live go.")

    return HypothesisReport(
        hypothesis=hid,
        name=hyp["name"],
        why=hyp["why"].strip(),
        sample_size=int(overall.n),
        success_rate=overall.rate if overall.n else None,
        baseline_rate=overall.baseline_rate if overall.baseline_n else None,
        p_value=overall.p_value,
        train=train,
        validation=validation,
        oos=oos,
        regime_breakdown=regime,
        cost_adjusted_success_rate=cost_rate,
        train_val_gap=gap,
        decision=decision,
        reject_reason=reason,
        notes=notes,
    )


def save_report(report: HypothesisReport) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = DATA_DIR / "results" / f"{report.hypothesis}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


def print_report(report: HypothesisReport) -> None:
    print("=" * 64)
    print(f"Hypothesis: {report.hypothesis} — {report.name}")
    print(f"Why: {report.why}")
    print(f"Sample size (treatment): {report.sample_size}")
    print(f"Success rate: {report.success_rate}")
    print(f"Baseline/comparison rate: {report.baseline_rate}")
    print(f"Statistical significance (p, one-sided): {report.p_value}")
    print(f"Train: {report.train}")
    print(f"Validation: {report.validation}")
    print(f"Out-of-sample: {report.oos}")
    print(f"Regime breakdown: {report.regime_breakdown}")
    print(f"Cost-adjusted success: {report.cost_adjusted_success_rate}")
    print(f"Train-val gap: {report.train_val_gap}")
    print(f"DECISION: {report.decision}")
    if report.reject_reason:
        print(f"Reason: {report.reject_reason}")
    for note in report.notes:
        print(f"Note: {note}")
    print("=" * 64)


def run_hypotheses(
    ids: list[str] | None, unlock_oos: bool, n_tests: int | None = None
) -> list[HypothesisReport]:
    settings = load_settings()
    hyps = load_hypotheses()
    if ids:
        want = set(ids)
        hyps = [h for h in hyps if h["id"] in want]
        missing = want - {h["id"] for h in hyps}
        if missing:
            raise SystemExit(f"Unknown hypothesis ids: {missing}")
    else:
        hyps = [h for h in hyps if h.get("enabled", True)]
    n_tests = max(n_tests or 0, len(hyps))
    reports = []
    for hyp in hyps:
        params = hyp.get("params") or {}
        if hyp_key(hyp["id"]) == "H5":
            frames = {}
        else:
            symbols = params.get("symbols") or settings["universe"]["symbols"]
            tf = params.get("timeframe") or settings["universe"]["timeframe"]
            frames = load_frames(symbols, tf, settings)
        report = evaluate(hyp, frames, settings, n_tests, unlock_oos)
        path = save_report(report)
        print_report(report)
        print(f"Wrote {path}")
        reports.append(report)
    return reports
