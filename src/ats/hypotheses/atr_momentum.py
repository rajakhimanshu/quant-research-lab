from __future__ import annotations

import numpy as np
import pandas as pd

from ats.timeutil import cost_price, pip_size


def _first_hit(
    high: np.ndarray,
    low: np.ndarray,
    start: int,
    sl: float,
    tp: float,
    side: str,
) -> str | None:
    """Walk forward from the fill bar. Same bar: stop wins. None if unresolved."""
    n = len(high)
    if side == "long":
        for j in range(start, n):
            if low[j] <= sl:
                return "sl"
            if high[j] >= tp:
                return "tp"
    else:
        for j in range(start, n):
            if high[j] >= sl:
                return "sl"
            if low[j] <= tp:
                return "tp"
    return None


def atr_momentum_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 1.2,
    slippage_pips: float = 0.5,
) -> pd.DataFrame:
    """ATR expansion candle that closes near the extreme.

    treatment: range > prior ATR × multiplier AND close in the outer close_percent.
    baseline: equally large directional candles that close mid-range (EA would skip).
    Entry at next bar open. SL/TP frozen from the EA. Same-bar stop wins.
    Daily-loss latch and trailing are execution overlays — not scored here.
    """
    work = df.dropna(subset=["atr"]).reset_index(drop=True)
    if len(work) < 3:
        return pd.DataFrame()

    high = work["high"].to_numpy(dtype=float)
    low = work["low"].to_numpy(dtype=float)
    open_ = work["open"].to_numpy(dtype=float)
    close = work["close"].to_numpy(dtype=float)
    atr = work["atr"].to_numpy(dtype=float)
    times = work["time"].to_numpy()
    n = len(work)

    atr_prior = np.empty(n, dtype=float)
    atr_prior[0] = np.nan
    atr_prior[1:] = atr[:-1]

    rng = high - low
    mult = float(params.get("atr_multiplier", 1.5))
    close_frac = float(params.get("close_percent", 20.0)) / 100.0
    mode = str(params.get("sltp_mode", "percent")).lower()
    sl_pct = float(params.get("stop_loss_pct", 1.0)) / 100.0
    tp_pct = float(params.get("take_profit_pct", 2.0)) / 100.0
    sl_atr = float(params.get("atr_sl_multiplier", 1.5))
    tp_atr = float(params.get("atr_tp_multiplier", 3.0))
    max_spread_pct = float(params.get("max_spread_atr_pct", 15.0))

    floor = pip_size(symbol)  # analog of Point()*10 on 5-digit FX
    valid = np.isfinite(atr_prior) & (atr_prior > floor) & (rng > 0.0)
    large = valid & (rng > atr_prior * mult)

    is_bull = close > open_
    is_bear = close < open_
    edge_buy = is_bull & (close >= high - rng * close_frac)
    edge_sell = is_bear & (close <= low + rng * close_frac)

    cost = cost_price(symbol, spread_pips, slippage_pips)
    spread_ok = np.ones(n, dtype=bool)
    if max_spread_pct > 0.0:
        spread_ok = atr_prior * (max_spread_pct / 100.0) >= cost

    treatment = large & spread_ok & (edge_buy | edge_sell)
    baseline = large & spread_ok & (is_bull | is_bear) & ~treatment

    rows: list[dict] = []
    for i in np.flatnonzero(treatment | baseline):
        i = int(i)
        entry_i = i + 1
        if entry_i >= n:
            continue
        entry = float(open_[entry_i])
        if entry <= 0.0:
            continue
        side = "long" if is_bull[i] else "short"
        prior = float(atr_prior[i])
        if mode == "atr":
            sl_off = prior * sl_atr
            tp_off = prior * tp_atr
        else:
            sl_off = entry * sl_pct
            tp_off = entry * tp_pct
        if sl_off <= 0.0 or tp_off <= 0.0:
            continue
        if side == "long":
            sl = entry - sl_off
            tp = entry + tp_off
            tp_cost = entry + tp_off + cost
        else:
            sl = entry + sl_off
            tp = entry - tp_off
            tp_cost = entry - tp_off - cost

        raw = _first_hit(high, low, entry_i, sl, tp, side)
        if raw is None:
            continue
        cost_hit = _first_hit(high, low, entry_i, sl, tp_cost, side)
        win = raw == "tp"
        rows.append(
            {
                "time": times[i],
                "symbol": symbol,
                "treatment": bool(treatment[i]),
                "side": side,
                "success": win,
                "success_cost_adj": cost_hit == "tp",
                "r_mult": (tp_off / sl_off) if win else -1.0,
                "atr_multiple": float(rng[i] / prior),
                "regime": work.at[i, "regime"] if "regime" in work.columns else "",
                "trend_regime": work.at[i, "trend_regime"] if "trend_regime" in work.columns else "",
            }
        )
    return pd.DataFrame(rows)
