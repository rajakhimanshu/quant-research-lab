"""H13: gold M30 expansion off EMA200 vs the same expansion far from that mean."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price


def to_m30(df: pd.DataFrame) -> pd.DataFrame:
    w = df.copy()
    w["time"] = pd.to_datetime(w["time"], utc=True)
    w = w.set_index("time").sort_index()
    d = w.resample("30min").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    d = d.dropna(how="any")
    prev = d["close"].shift(1)
    tr = pd.concat(
        [d["high"] - d["low"], (d["high"] - prev).abs(), (d["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    return d.dropna(subset=["atr"])


def gold_ema200_expand_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 240,
    slippage_pips: float = 50,
) -> pd.DataFrame:
    """treatment = rapid M30 bar leaving a prior EMA200 touch; baseline = same-size bar far from EMA."""
    if df.empty:
        return pd.DataFrame()
    ema_n = int(params.get("ema_period", 200))
    near_atr = float(params.get("near_atr", 0.35))
    far_atr = float(params.get("far_atr", 1.50))
    expand_atr = float(params.get("expand_atr", 1.50))
    horizon = int(params.get("horizon_bars", 4))
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    cost = cost_price(symbol, spread_pips, slippage_pips)

    work = to_m30(df)
    if work.empty:
        return pd.DataFrame()
    work["ema"] = work["close"].ewm(span=ema_n, adjust=False, min_periods=ema_n).mean()
    work["atr_prior"] = work["atr"].shift(1)
    work["rng"] = work["high"] - work["low"]
    prev_low = work["low"].shift(1)
    prev_high = work["high"].shift(1)
    prev_close = work["close"].shift(1)
    prev_ema = work["ema"].shift(1)
    touched = (prev_low <= prev_ema) & (prev_high >= prev_ema)
    near = touched | ((prev_close - prev_ema).abs() <= near_atr * work["atr_prior"])
    far = ((prev_close - prev_ema).abs() >= far_atr * work["atr_prior"]) & ~touched
    large = work["rng"] > expand_atr * work["atr_prior"]
    treat_m = large & near & work["ema"].notna() & (work["close"] != work["ema"])
    base_m = large & far & work["ema"].notna() & (work["close"] != work["open"])

    rows = []
    n = len(work)
    for i in range(n - horizon - 1):
        is_t = bool(treat_m.iloc[i])
        is_b = bool(base_m.iloc[i])
        if not is_t and not is_b:
            continue
        atr = float(work["atr_prior"].iloc[i])
        if not np.isfinite(atr) or atr <= 0:
            continue
        if is_t:
            side = "long" if float(work["close"].iloc[i]) > float(work["ema"].iloc[i]) else "short"
            treat = True
        else:
            side = "long" if float(work["close"].iloc[i]) > float(work["open"].iloc[i]) else "short"
            treat = False
        loc = i + 1
        raw_open = float(work["open"].iloc[loc])
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
                "time": work.index[loc],
                "symbol": symbol,
                "treatment": treat,
                "side": side,
                "success": bool(hit),
                "success_cost_adj": bool(r_mult > 0),
                "r_mult": float(r_mult),
            }
        )
    return pd.DataFrame(rows)
