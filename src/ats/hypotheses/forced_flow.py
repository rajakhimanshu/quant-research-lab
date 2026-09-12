"""Forced-flow tests: auctions, month-end, inventory streaks, weekend gaps, WM fix.

Not EMA stacks. Not a reopen of H12/H17-H22.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ats.hypotheses.m15_micro import _trade
from ats.timeutil import cost_price

LONDON = ZoneInfo("Europe/London")
NY = ZoneInfo("America/New_York")


def _params(params: dict, symbol: str, spread_pips: float, slippage_pips: float) -> tuple:
    horizon = int(params.get("horizon_bars", 8))
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    return horizon, stop_atr, target_atr, cost


def lbma_pm_run_events(df, symbol, params, spread_pips=200, slippage_pips=20) -> pd.DataFrame:
    """Follow the 30 min into LBMA PM (15:00 London) vs the same 30 min at 11:30 London."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    min_prior = float(params.get("min_prior_atr", 0.25))
    treat_h, treat_m = int(params.get("treat_hour", 14)), int(params.get("treat_minute", 30))
    base_h, base_m = int(params.get("base_hour", 11)), int(params.get("base_minute", 30))
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["lh"] = ldn.dt.hour
    work["lm"] = ldn.dt.minute
    work["ldate"] = ldn.dt.date
    rows = []
    for _, g in work.groupby("ldate", sort=True):
        for treat, hour, minute in (True, treat_h, treat_m), (False, base_h, base_m):
            hit = g[(g["lh"] == hour) & (g["lm"] == minute)]
            prev = g[(g["lh"] == hour) & (g["lm"] == minute - 15)]
            if hit.empty or prev.empty:
                continue
            i = int(hit.index[-1])
            j = int(prev.index[-1])
            if j >= i:
                continue
            atr = float(work.at[i, "atr"])
            if not np.isfinite(atr) or atr <= 0:
                continue
            prior = float(work.at[i, "close"]) - float(work.at[j, "open"])
            if abs(prior) < min_prior * atr:
                continue
            side = "long" if prior > 0 else "short"
            row = _trade(work, i, side, treat, symbol, cost, horizon, stop_atr, target_atr)
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def month_end_rebalance_events(df, symbol, params, spread_pips=200, slippage_pips=20) -> pd.DataFrame:
    """Follow month-to-date gold on the last N NY sessions vs mid-month sessions."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    last_n = int(params.get("last_n_sessions", 5))
    mid_lo = int(params.get("mid_dom_lo", 10))
    mid_hi = int(params.get("mid_dom_hi", 14))
    entry_hour = int(params.get("entry_hour_ny", 9))
    ny = pd.to_datetime(work["time"], utc=True).dt.tz_convert(NY)
    work["ny_date"] = ny.dt.date
    work["ny_hour"] = ny.dt.hour
    work["ym"] = ny.dt.strftime("%Y-%m")
    work["dom"] = ny.dt.day
    rows = []
    for _, month in work.groupby("ym", sort=True):
        days = sorted(month["ny_date"].unique())
        if len(days) < last_n + mid_hi:
            continue
        first_open = float(month.iloc[0]["open"])
        last_set = set(days[-last_n:])
        for day in days:
            g = month.loc[month["ny_date"] == day]
            prev_days = [d for d in days if d < day]
            if not prev_days:
                continue
            prev = month.loc[month["ny_date"] == prev_days[-1]]
            mtd = float(prev.iloc[-1]["close"]) - first_open
            if mtd == 0:
                continue
            treat = day in last_set
            baseline = (not treat) and (mid_lo <= int(pd.Timestamp(day).day) <= mid_hi)
            if not treat and not baseline:
                continue
            entry_bars = g[g["ny_hour"] == entry_hour]
            if entry_bars.empty:
                continue
            i = int(entry_bars.index[0])
            if i < 1:
                continue
            side = "long" if mtd > 0 else "short"
            row = _trade(work, i - 1, side, treat, symbol, cost, horizon, stop_atr, target_atr)
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def streak_inventory_fade_events(df, symbol, params, spread_pips=200, slippage_pips=20) -> pd.DataFrame:
    """Fade a 5-bar same-direction streak vs fade a 2-bar streak (inventory binds later)."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    long_n = int(params.get("long_streak", 5))
    short_n = int(params.get("short_streak", 2))
    direction = np.sign(work["close"] - work["open"]).replace(0, np.nan)
    rows = []
    n = len(work)
    for i in range(long_n - 1, n):
        long_w = direction.iloc[i - long_n + 1 : i + 1]
        short_w = direction.iloc[i - short_n + 1 : i + 1]
        if long_w.isna().any() or short_w.isna().any():
            continue
        last = float(long_w.iloc[-1])
        treat = bool(np.all(long_w.to_numpy() == last))
        base = (not treat) and bool(np.all(short_w.to_numpy() == last))
        if not treat and not base:
            continue
        run = long_n if treat else short_n
        before = i - run
        if before >= 0 and direction.iloc[before] == last:
            continue
        side = "short" if last > 0 else "long"
        row = _trade(work, i, side, treat, symbol, cost, horizon, stop_atr, target_atr)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def weekend_gap_fade_events(df, symbol, params, spread_pips=200, slippage_pips=20) -> pd.DataFrame:
    """Fade the weekend gap vs fade a London-open jump of similar size (not a halt)."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    min_atr = float(params.get("min_gap_atr", 0.30))
    gap_hours = float(params.get("weekend_gap_hours", 24))
    london_hour = int(params.get("london_open_hour", 8))
    times = pd.to_datetime(work["time"], utc=True)
    ldn = times.dt.tz_convert(LONDON)
    work["lh"] = ldn.dt.hour
    delta_h = times.diff().dt.total_seconds() / 3600.0
    rows = []
    n = len(work)
    for i in range(1, n):
        atr = float(work.at[i, "atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue
        gap = float(work.at[i, "open"]) - float(work.at[i - 1, "close"])
        if abs(gap) < min_atr * atr:
            continue
        is_weekend = float(delta_h.iloc[i]) >= gap_hours
        is_london = (not is_weekend) and int(work.at[i, "lh"]) == london_hour
        if not is_weekend and not is_london:
            continue
        side = "short" if gap > 0 else "long"
        row = _trade(work, i - 1, side, is_weekend, symbol, cost, horizon, stop_atr, target_atr)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def wm_fix_follow_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Follow the 14:00 London H1 bar into the WM 4pm window vs follow the 10:00 bar."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    min_prior = float(params.get("min_prior_atr", 0.25))
    treat_hour = int(params.get("treat_hour_london", 14))
    base_hour = int(params.get("base_hour_london", 10))
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["lh"] = ldn.dt.hour
    rows = []
    n = len(work)
    for i in range(n):
        hour = int(work.at[i, "lh"])
        if hour == treat_hour:
            treat = True
        elif hour == base_hour:
            treat = False
        else:
            continue
        atr = float(work.at[i, "atr"])
        body = float(work.at[i, "close"]) - float(work.at[i, "open"])
        if not np.isfinite(atr) or atr <= 0 or abs(body) < min_prior * atr:
            continue
        side = "long" if body > 0 else "short"
        row = _trade(work, i, side, treat, symbol, cost, horizon, stop_atr, target_atr)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def comex_session_run_events(df, symbol, params, spread_pips=200, slippage_pips=20) -> pd.DataFrame:
    """Signed M15 run at a NY COMEX clock vs the same run at a control hour.

    fade=False follows into the print (H54 8:20 open). fade=True fades into it (H56 13:30 close).
    """
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    min_prior = float(params.get("min_prior_atr", 0.25))
    fade = bool(params.get("fade", False))
    treat_h, treat_m = int(params.get("treat_hour", 8)), int(params.get("treat_minute", 0))
    base_h, base_m = int(params.get("base_hour", 6)), int(params.get("base_minute", 0))
    ny = pd.to_datetime(work["time"], utc=True).dt.tz_convert(NY)
    work["nh"] = ny.dt.hour
    work["nm"] = ny.dt.minute
    work["ndate"] = ny.dt.date
    rows = []
    for _, g in work.groupby("ndate", sort=True):
        for treat, hour, minute in (True, treat_h, treat_m), (False, base_h, base_m):
            hit = g[(g["nh"] == hour) & (g["nm"] == minute)]
            if hit.empty:
                continue
            i = int(hit.index[-1])
            earlier = g[g.index < i]
            if earlier.empty:
                continue
            j = int(earlier.index[-1])
            atr = float(work.at[i, "atr"])
            if not np.isfinite(atr) or atr <= 0:
                continue
            prior = float(work.at[i, "close"]) - float(work.at[j, "open"])
            if abs(prior) < min_prior * atr:
                continue
            follow = "long" if prior > 0 else "short"
            side = ("short" if follow == "long" else "long") if fade else follow
            row = _trade(work, i, side, treat, symbol, cost, horizon, stop_atr, target_atr)
            if row:
                rows.append(row)
    return pd.DataFrame(rows)
