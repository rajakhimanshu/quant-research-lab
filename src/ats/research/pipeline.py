from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ats.config import DATA_DIR, load_hypotheses, load_settings
from ats.data.calendar import load_calendar
from ats.data.mt5_client import load_raw
from ats.features.prepare import prepare_frame
from ats.hypotheses.event_reversal import event_reversal_events
from ats.hypotheses.tap_breakout import tap_events
from ats.research.stats import HypothesisReport, rates
from ats.timeutil import time_splits

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
    for symbol, df in frames.items():
        spread = float(costs["spread_pips"].get(symbol, 1.5))
        slip = float(costs["slippage_pips"])
        hid = hyp["id"]
        print(f"  scoring {hid} on {symbol} ({len(df)} bars)")
        if hid.startswith("H1"):
            ev = event_reversal_events(df, symbol, merged_params, calendar, spread, slip)
        elif hid.startswith("H2"):
            ev = tap_events(df, symbol, merged_params)
            if not ev.empty:
                ev = ev.loc[ev["treatment"] | ev["tap_no"].isin([1, 2])].copy()
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
    notes.append("OOS is locked until you explicitly unlock it.")
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
    if hid.startswith("H1") and load_calendar().empty:
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
    idx = pd.DatetimeIndex(pd.to_datetime(events["time"], utc=True))
    split = time_splits(idx, float(v["train"]), float(v["validation"]))
    times = pd.to_datetime(events["time"], utc=True)
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
    hyps = [h for h in load_hypotheses() if h.get("enabled", True)]
    if ids:
        want = set(ids)
        hyps = [h for h in hyps if h["id"] in want]
        missing = want - {h["id"] for h in hyps}
        if missing:
            raise SystemExit(f"Unknown hypothesis ids: {missing}")
    universe = settings["universe"]
    frames = load_frames(universe["symbols"], universe["timeframe"], settings)
    n_tests = len(hyps)
    reports = []
    for hyp in hyps:
        report = evaluate(hyp, frames, settings, n_tests, unlock_oos)
        path = save_report(report)
        print_report(report)
        print(f"Wrote {path}")
        reports.append(report)
    return reports
