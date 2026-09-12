"""London 12:30 IST M15 opening-range break and Fibonacci bounce. Range 1:1, not ATR."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price

IST = ZoneInfo("Asia/Kolkata")


def _range_row(
    work: pd.DataFrame,
    signal_i: int,
    side: str,
    treat: bool,
    symbol: str,
    entry: float,
    stop: float,
    rr: float,
    horizon: int,
) -> dict | None:
    risk = abs(entry - stop)
    if risk <= 0 or not np.isfinite(risk):
        return None
    if side == "long":
        if stop >= entry:
            return None
        target = entry + rr * risk
    else:
        if stop <= entry:
            return None
        target = entry - rr * risk
    hit, r_mult = _forward(work, signal_i, side, entry, target, stop, horizon)
    return {
        "time": work.at[signal_i + 1, "time"],
        "symbol": symbol,
        "treatment": treat,
        "side": side,
        "success": bool(hit),
        "success_cost_adj": bool(r_mult > 0),
        "r_mult": float(r_mult),
    }


def london_ist_break_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Follow a second-M15 close through the 12:30 IST candle vs fade that close. 1:1 of the range."""
    work = df.dropna(subset=["open", "high", "low", "close"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon = int(params.get("horizon_bars", 24))
    rr = float(params.get("rr", 1.0))
    mark_h = int(params.get("mark_hour_ist", 12))
    mark_m = int(params.get("mark_minute_ist", 30))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    ist = pd.to_datetime(work["time"], utc=True).dt.tz_convert(IST)
    work["ih"] = ist.dt.hour
    work["im"] = ist.dt.minute
    work["idate"] = ist.dt.date
    rows = []
    n = len(work)
    for _, g in work.groupby("idate", sort=True):
        first = g[(g["ih"] == mark_h) & (g["im"] == mark_m)]
        if first.empty:
            continue
        i0 = int(first.index[0])
        i1 = i0 + 1
        if i1 + 1 + horizon >= n:
            continue
        rh = float(work.at[i0, "high"])
        rl = float(work.at[i0, "low"])
        rng = rh - rl
        if rng <= cost * 2:
            continue
        close1 = float(work.at[i1, "close"])
        if close1 > rh:
            follow = "long"
        elif close1 < rl:
            follow = "short"
        else:
            continue
        raw_open = float(work.at[i1 + 1, "open"])
        fade = "short" if follow == "long" else "long"
        follow_stop = rl if follow == "long" else rh
        fade_stop = rh if follow == "long" else rl
        for treat, side, stop_px in (True, follow, follow_stop), (False, fade, fade_stop):
            entry = raw_open + cost if side == "long" else raw_open - cost
            row = _range_row(work, i1, side, treat, symbol, entry, stop_px, rr, horizon)
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def london_fib_bounce_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Bounce the 61.8% retrace of the 12:30-14:30 IST impulse vs bounce at 50%."""
    work = df.dropna(subset=["open", "high", "low", "close"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon = int(params.get("horizon_bars", 16))
    rr = float(params.get("rr", 1.0))
    start_h = int(params.get("impulse_start_hour_ist", 12))
    start_m = int(params.get("impulse_start_minute_ist", 30))
    n_impulse = int(params.get("impulse_bars", 8))
    treat_fib = float(params.get("treat_fib", 0.618))
    base_fib = float(params.get("base_fib", 0.50))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    ist = pd.to_datetime(work["time"], utc=True).dt.tz_convert(IST)
    work["ih"] = ist.dt.hour
    work["im"] = ist.dt.minute
    work["idate"] = ist.dt.date
    rows = []
    n = len(work)
    for _, g in work.groupby("idate", sort=True):
        first = g[(g["ih"] == start_h) & (g["im"] == start_m)]
        if first.empty:
            continue
        i0 = int(first.index[0])
        i_end = i0 + n_impulse - 1
        if i_end + 2 + horizon >= n:
            continue
        window = work.loc[i0:i_end]
        hi = float(window["high"].max())
        lo = float(window["low"].min())
        rng = hi - lo
        if rng <= cost * 2:
            continue
        last_close = float(work.at[i_end, "close"])
        first_open = float(work.at[i0, "open"])
        if last_close == first_open:
            continue
        impulse_up = last_close > first_open
        levels = ((True, treat_fib), (False, base_fib))
        fired = {True: False, False: False}
        for i in range(i_end + 1, min(i_end + 1 + 16, n - 1 - horizon)):
            bar_h = float(work.at[i, "high"])
            bar_l = float(work.at[i, "low"])
            for treat, fib in levels:
                if fired[treat]:
                    continue
                if impulse_up:
                    level = hi - fib * rng
                    if bar_l <= level <= bar_h:
                        side = "long"
                        entry = float(work.at[i + 1, "open"]) + cost
                        stop = lo
                        row = _range_row(work, i, side, treat, symbol, entry, stop, rr, horizon)
                        if row:
                            rows.append(row)
                        fired[treat] = True
                else:
                    level = lo + fib * rng
                    if bar_l <= level <= bar_h:
                        side = "short"
                        entry = float(work.at[i + 1, "open"]) - cost
                        stop = hi
                        row = _range_row(work, i, side, treat, symbol, entry, stop, rr, horizon)
                        if row:
                            rows.append(row)
                        fired[treat] = True
            if fired[True] and fired[False]:
                break
    return pd.DataFrame(rows)
