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
from ats.hypotheses.rapid_bullet import rapid_bullet_events
from ats.hypotheses.equity_rsi2 import equity_rsi2_events
from ats.hypotheses.event_reversal import event_reversal_events
from ats.hypotheses.ema_13_50_200 import ema_13_50_200_events
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
    wm_postfix_fade_events,
    xs_momentum_events,
)
from ats.hypotheses.forced_flow import (
    lbma_pm_run_events,
    month_end_rebalance_events,
    streak_inventory_fade_events,
    weekend_gap_fade_events,
    wm_fix_follow_events,
)
from ats.hypotheses.m15_micro import (
    large_bar_fade_events,
    ny_box_fade_events,
    pdh_pdl_fade_events,
    round_bounce_events,
    volume_spike_cont_events,
    vwap_extreme_fade_events,
)
from ats.hypotheses.ny_open_sweep import ny_open_sweep_events
from ats.hypotheses.session_orb import session_orb_events
from ats.hypotheses.tap_breakout import tap_events
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
    if key == "H38":
        print(f"  scoring {hid} cross-section ({len(frames)} pairs)")
        return xs_momentum_events(frames, merged_params, settings)
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
        else:
            raise ValueError(f"Unknown hypothesis {hid}")
        if ev is not None and not ev.empty:
            chunks.append(ev)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).sort_values("time")


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
    split = locked_calendar_split(settings, split_book_for(hid))
    if split is None:
        split = time_splits(idx, float(v["train"]), float(v["validation"]))
    notes.append(f"split {split_book_for(hid)} train_end={split.train_end} val_end={split.val_end}")
    train_m = times <= split.train_end
    val_m = (times > split.train_end) & (times <= split.val_end)
    oos_m = times > split.val_end

    train = _part_rates(events, train_m)
    validation = _part_rates(events, val_m)
    if unlock_oos:
        oos = _part_rates(events, oos_m)
    else:
        oos = {"status": LOCKED, "n": int(oos_m.sum())}

    treated = events.loc[events["treatment"]]
    overall = rates(events["success"], events["treatment"])
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

    if decision == "CANDIDATE" and not unlock_oos:
        notes.append("OOS is locked until you explicitly unlock it.")
    elif decision == "CANDIDATE" and unlock_oos:
        min_oos = max(30, int(v["min_trades"]) // 4)
        if oos.get("n", 0) < min_oos:
            decision, reason = "NEEDS_MORE_DATA", f"oos n={oos.get('n')} too small"
        elif oos.get("rate") is None or oos.get("rate") <= oos.get("baseline_rate", 1):
            decision, reason = "REJECT", "oos success is not above baseline"
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


def run_hypotheses(ids: list[str] | None, unlock_oos: bool) -> list[HypothesisReport]:
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
    n_tests = len(hyps)
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
