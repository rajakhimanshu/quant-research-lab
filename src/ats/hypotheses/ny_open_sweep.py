from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ats.hypotheses.m15_micro import _trade
from ats.timeutil import cost_price

NY = ZoneInfo("America/New_York")


def _forward_path(
    work: pd.DataFrame, i: int, side: str, target: float, stop: float, horizon: int
) -> bool:
    """Target before the sweep extreme is taken back. Same-bar: stop wins."""
    future = work.iloc[i + 1 : i + 1 + horizon]
    if future.empty:
        return False
    for _, bar in future.iterrows():
        if side == "short":
            stopped = float(bar["high"]) >= stop
            won = float(bar["low"]) <= target
        else:
            stopped = float(bar["low"]) <= stop
            won = float(bar["high"]) >= target
        if stopped:
            return False
        if won:
            return True
    return False


def ny_open_sweep_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 240,
    slippage_pips: float = 50,
) -> pd.DataFrame:
    """Overnight-range raid + reclaim. treatment = 9-10am NY, baseline = later NY hours."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    ny = pd.to_datetime(work["time"], utc=True).dt.tz_convert(NY)
    work["ny_date"] = ny.dt.date
    work["ny_hour"] = ny.dt.hour
    sweep_atr = float(params.get("sweep_atr", 0.10))
    target_atr = float(params.get("target_atr", 0.50))
    horizon = int(params.get("horizon_bars", 8))
    ny_start = int(params.get("ny_start", 9))
    ny_end = int(params.get("ny_end", 10))
    c_start = int(params.get("control_start", 11))
    c_end = int(params.get("control_end", 16))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    rows = []

    pool_s = params.get("pool_ny_start")
    pool_e = params.get("pool_ny_end")
    for day, g in work.groupby("ny_date", sort=True):
        if pool_s is None:
            pool = g[g["ny_hour"] < ny_start]
        else:
            pool = g[(g["ny_hour"] >= int(pool_s)) & (g["ny_hour"] < int(pool_e if pool_e is not None else ny_start))]
        if len(pool) < 8:
            continue
        oh = float(pool["high"].max())
        ol = float(pool["low"].min())
        windows = (
            (g[(g["ny_hour"] >= ny_start) & (g["ny_hour"] < ny_end)], True),
            (g[(g["ny_hour"] >= c_start) & (g["ny_hour"] < c_end)], False),
        )
        for subset, treat in windows:
            if subset.empty:
                continue
            swept_high = swept_low = False
            for i in subset.index:
                if i + horizon >= len(work):
                    break
                atr = float(work.at[i, "atr"])
                if not np.isfinite(atr) or atr <= 0:
                    continue
                buf = sweep_atr * atr
                high = float(work.at[i, "high"])
                low = float(work.at[i, "low"])
                close = float(work.at[i, "close"])
                if high > oh + buf:
                    swept_high = True
                if low < ol - buf:
                    swept_low = True
                side = None
                sweep_ext = None
                if swept_high and close < oh:
                    side = "short"
                    sweep_ext = high
                elif swept_low and close > ol:
                    side = "long"
                    sweep_ext = low
                if side is None:
                    continue
                target = close - target_atr * atr if side == "short" else close + target_atr * atr
                target_cost = (
                    close - (target_atr * atr + cost) if side == "short" else close + (target_atr * atr + cost)
                )
                hit = _forward_path(work, int(i), side, target, sweep_ext, horizon)
                hit_cost = _forward_path(work, int(i), side, target_cost, sweep_ext, horizon)
                rows.append(
                    {
                        "time": work.at[i, "time"],
                        "symbol": symbol,
                        "treatment": treat,
                        "side": side,
                        "success": hit,
                        "success_cost_adj": hit_cost,
                        "regime": work.at[i, "regime"] if "regime" in work.columns else "",
                        "trend_regime": work.at[i, "trend_regime"] if "trend_regime" in work.columns else "",
                        "ny_date": str(day),
                    }
                )
                break
    return pd.DataFrame(rows)


def ny_sweep_follow_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 240,
    slippage_pips: float = 50,
) -> pd.DataFrame:
    """Overnight-range raid that does not reclaim: follow continuation.

    treatment = 9-10am NY (cash-open stop run). baseline = 11-16 NY.
    Distinct from ny_open_sweep_events, which requires a reclaim close.
    """
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    ny = pd.to_datetime(work["time"], utc=True).dt.tz_convert(NY)
    work["ny_date"] = ny.dt.date
    work["ny_hour"] = ny.dt.hour
    sweep_atr = float(params.get("sweep_atr", 0.10))
    horizon = int(params.get("horizon_bars", 8))
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    ny_start = int(params.get("ny_start", 9))
    ny_end = int(params.get("ny_end", 10))
    c_start = int(params.get("control_start", 11))
    c_end = int(params.get("control_end", 16))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    rows = []

    pool_s = params.get("pool_ny_start")
    pool_e = params.get("pool_ny_end")
    for _day, g in work.groupby("ny_date", sort=True):
        if pool_s is None:
            pool = g[g["ny_hour"] < ny_start]
        else:
            pool = g[(g["ny_hour"] >= int(pool_s)) & (g["ny_hour"] < int(pool_e if pool_e is not None else ny_start))]
        if len(pool) < 8:
            continue
        oh = float(pool["high"].max())
        ol = float(pool["low"].min())
        windows = (
            (g[(g["ny_hour"] >= ny_start) & (g["ny_hour"] < ny_end)], True),
            (g[(g["ny_hour"] >= c_start) & (g["ny_hour"] < c_end)], False),
        )
        for subset, treat in windows:
            if subset.empty:
                continue
            for i in subset.index:
                atr = float(work.at[i, "atr"])
                if not np.isfinite(atr) or atr <= 0:
                    continue
                buf = sweep_atr * atr
                high = float(work.at[i, "high"])
                low = float(work.at[i, "low"])
                close = float(work.at[i, "close"])
                side = None
                if high > oh + buf and close > oh:
                    side = "long"
                elif low < ol - buf and close < ol:
                    side = "short"
                if side is None:
                    continue
                row = _trade(work, int(i), side, treat, symbol, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
                break
    return pd.DataFrame(rows)
