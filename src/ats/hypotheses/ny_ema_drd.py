"""H14: M5 EMA 4-9-18 stack + Dr.D (ADX-colored) follow in NY vs the same follow in London."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price

NY = ZoneInfo("America/New_York")


def wilder_di_adx(df: pd.DataFrame, di_len: int = 14, dx_smooth: int = 14) -> pd.DataFrame:
    """Pine/Wilder +DI -DI ADX. Dr.D color is ADX > lim and DI sign, not a separate oscillator."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    alpha = 1.0 / di_len
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=di_len).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=di_len).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=di_len).mean() / atr.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / dx_smooth, adjust=False, min_periods=dx_smooth).mean()
    out = df.copy()
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di
    out["adx"] = adx
    out["atr"] = atr
    return out


def ny_ema_drd_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 240,
    slippage_pips: float = 50,
) -> pd.DataFrame:
    """Follow a new EMA4>9>18 + Dr.D lime (and the bearish opposite). treatment=NY, baseline=London hours."""
    work = df.dropna(subset=["high", "low", "close"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    e4 = int(params.get("ema_fast", 4))
    e9 = int(params.get("ema_mid", 9))
    e18 = int(params.get("ema_slow", 18))
    di_len = int(params.get("di_length", 14))
    dx_n = int(params.get("dx_smooth", 14))
    lim = float(params.get("adx_active", 18))
    horizon = int(params.get("horizon_bars", 6))
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    ny_s = int(params.get("ny_start", 9))
    ny_e = int(params.get("ny_end", 12))
    ldn_s = int(params.get("london_start_ny", 3))
    ldn_e = int(params.get("london_end_ny", 8))
    cost = cost_price(symbol, spread_pips, slippage_pips)

    work = wilder_di_adx(work, di_len, dx_n)
    c = work["close"].astype(float)
    work["ema4"] = c.ewm(span=e4, adjust=False, min_periods=e4).mean()
    work["ema9"] = c.ewm(span=e9, adjust=False, min_periods=e9).mean()
    work["ema18"] = c.ewm(span=e18, adjust=False, min_periods=e18).mean()
    lime = (work["adx"] > lim) & (work["plus_di"] > work["minus_di"])
    red = (work["adx"] > lim) & (work["plus_di"] < work["minus_di"])
    bull = (work["ema4"] > work["ema9"]) & (work["ema9"] > work["ema18"]) & lime
    bear = (work["ema4"] < work["ema9"]) & (work["ema9"] < work["ema18"]) & red
    aligned = bull | bear
    start = aligned & ~aligned.shift(1, fill_value=False)

    ny = pd.to_datetime(work["time"], utc=True).dt.tz_convert(NY)
    hour = ny.dt.hour
    in_ny = (hour >= ny_s) & (hour < ny_e)
    in_ldn = (hour >= ldn_s) & (hour < ldn_e)

    rows = []
    n = len(work)
    idx = np.flatnonzero(start.to_numpy())
    for i in idx:
        if i + 1 + horizon >= n:
            continue
        if in_ny.iloc[i]:
            treat = True
        elif in_ldn.iloc[i]:
            treat = False
        else:
            continue
        atr = float(work.at[i, "atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue
        side = "long" if bool(bull.iloc[i]) else "short"
        raw_open = float(work.at[i + 1, "open"])
        if side == "long":
            entry = raw_open + cost
            stop = entry - stop_atr * atr
            target = entry + target_atr * atr
        else:
            entry = raw_open - cost
            stop = entry + stop_atr * atr
            target = entry - target_atr * atr
        hit, r_mult = _forward(work, i, side, entry, target, stop, horizon)
        rows.append(
            {
                "time": work.at[i + 1, "time"],
                "symbol": symbol,
                "treatment": treat,
                "side": side,
                "success": bool(hit),
                "success_cost_adj": bool(r_mult > 0),
                "r_mult": float(r_mult),
            }
        )
    return pd.DataFrame(rows)
