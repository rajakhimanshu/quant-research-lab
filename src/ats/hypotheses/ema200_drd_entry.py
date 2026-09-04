"""H15: EMA200 trend filter, Dr.D color as entry — with-trend vs against-trend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ats.hypotheses.cot_spec_fade import _forward
from ats.hypotheses.ny_ema_drd import wilder_di_adx
from ats.timeutil import cost_price


def ema200_drd_entry_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 240,
    slippage_pips: float = 50,
) -> pd.DataFrame:
    """treatment = new Dr.D lime/red in the EMA200 direction; baseline = same Dr.D against EMA200."""
    work = df.dropna(subset=["high", "low", "close"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    ema_n = int(params.get("ema_period", 200))
    di_len = int(params.get("di_length", 14))
    dx_n = int(params.get("dx_smooth", 14))
    lim = float(params.get("adx_active", 18))
    horizon = int(params.get("horizon_bars", 12))
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    cost = cost_price(symbol, spread_pips, slippage_pips)

    work = wilder_di_adx(work, di_len, dx_n)
    close = work["close"].astype(float)
    work["ema200"] = close.ewm(span=ema_n, adjust=False, min_periods=ema_n).mean()
    lime = (work["adx"] > lim) & (work["plus_di"] > work["minus_di"])
    red = (work["adx"] > lim) & (work["plus_di"] < work["minus_di"])
    lime_on = lime & ~lime.shift(1, fill_value=False)
    red_on = red & ~red.shift(1, fill_value=False)
    above = close > work["ema200"]
    with_t = (lime_on & above) | (red_on & ~above)
    against = (lime_on & ~above) | (red_on & above)

    rows = []
    n = len(work)
    idx = np.flatnonzero((with_t | against).to_numpy())
    for i in idx:
        if i + 1 + horizon >= n:
            continue
        if not np.isfinite(float(work.at[i, "ema200"])):
            continue
        atr = float(work.at[i, "atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue
        if bool(lime_on.iloc[i]):
            side = "long"
        else:
            side = "short"
        treat = bool(with_t.iloc[i])
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
