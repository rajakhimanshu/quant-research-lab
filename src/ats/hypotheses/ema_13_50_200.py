"""H16: M15 EMA 13>50>200 stack vs the same 13>50 fighting the 200."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price


def ema_13_50_200_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 240,
    slippage_pips: float = 50,
) -> pd.DataFrame:
    """treatment = new 13/50/200 agreement; baseline = new 13/50 against the 200."""
    work = df.dropna(subset=["close"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    n13 = int(params.get("ema_fast", 13))
    n50 = int(params.get("ema_mid", 50))
    n200 = int(params.get("ema_slow", 200))
    horizon = int(params.get("horizon_bars", 8))
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    cost = cost_price(symbol, spread_pips, slippage_pips)

    close = work["close"].astype(float)
    work["ema13"] = close.ewm(span=n13, adjust=False, min_periods=n13).mean()
    work["ema50"] = close.ewm(span=n50, adjust=False, min_periods=n50).mean()
    work["ema200"] = close.ewm(span=n200, adjust=False, min_periods=n200).mean()
    if "atr" not in work.columns or work["atr"].isna().all():
        prev = close.shift(1)
        tr = pd.concat(
            [
                work["high"].astype(float) - work["low"].astype(float),
                (work["high"].astype(float) - prev).abs(),
                (work["low"].astype(float) - prev).abs(),
            ],
            axis=1,
        ).max(axis=1)
        work["atr"] = tr.rolling(14).mean()

    bull_fast = work["ema13"] > work["ema50"]
    bear_fast = work["ema13"] < work["ema50"]
    bull_full = bull_fast & (work["ema50"] > work["ema200"])
    bear_full = bear_fast & (work["ema50"] < work["ema200"])
    bull_vs = bull_fast & (work["ema50"] < work["ema200"])
    bear_vs = bear_fast & (work["ema50"] > work["ema200"])
    full = bull_full | bear_full
    against = bull_vs | bear_vs
    full_on = full & ~full.shift(1, fill_value=False)
    against_on = against & ~against.shift(1, fill_value=False)

    rows = []
    n = len(work)
    idx = np.flatnonzero((full_on | against_on).to_numpy())
    for i in idx:
        if i + 1 + horizon >= n:
            continue
        if not np.isfinite(float(work.at[i, "ema200"])):
            continue
        atr = float(work.at[i, "atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue
        if bool(bull_full.iloc[i] or bull_vs.iloc[i]):
            side = "long"
        else:
            side = "short"
        treat = bool(full_on.iloc[i])
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
