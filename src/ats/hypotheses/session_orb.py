from __future__ import annotations

import pandas as pd

from ats.timeutil import cost_price


def _session_mask(hours: pd.Series, start: int, end: int) -> pd.Series:
    if start < end:
        return (hours >= start) & (hours < end)
    return (hours >= start) | (hours < end)


def session_orb_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 240,
    slippage_pips: float = 50,
) -> pd.DataFrame:
    """Opening-range breakout: first M15 of the window is the range. treatment=Asia, baseline=NY."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    utc = pd.to_datetime(work["time"], utc=True)
    work["utc_hour"] = utc.dt.hour
    work["utc_date"] = utc.dt.date
    asia_s = int(params.get("asia_start_hour_utc", 0))
    asia_e = int(params.get("asia_end_hour_utc", 7))
    ny_s = int(params.get("ny_start_hour_utc", 13))
    ny_e = int(params.get("ny_end_hour_utc", 17))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    rows = []

    for day, g in work.groupby("utc_date", sort=True):
        blocks = (
            (g[_session_mask(g["utc_hour"], asia_s, asia_e)], True),
            (g[_session_mask(g["utc_hour"], ny_s, ny_e)], False),
        )
        for subset, treat in blocks:
            if len(subset) < 3:
                continue
            first_i = int(subset.index[0])
            last_i = int(subset.index[-1])
            if last_i <= first_i:
                continue
            rh = float(work.at[first_i, "high"])
            rl = float(work.at[first_i, "low"])
            rng = rh - rl
            if rng <= cost * 2:
                continue
            side = None
            entry_i = None
            for i in subset.index[1:]:
                close = float(work.at[i, "close"])
                if close > rh:
                    side, entry_i = "long", int(i)
                    break
                if close < rl:
                    side, entry_i = "short", int(i)
                    break
            if side is None or entry_i is None:
                continue
            rest = work.iloc[entry_i + 1 : last_i + 1]
            if rest.empty:
                continue
            fade = str(params.get("mode", "breakout")).lower() == "fade"
            extra = cost
            if fade:
                # Fade the first break: opposite side, SL through 1R extension, TP other side of range.
                if side == "long":
                    hit_sl = bool((rest["high"] >= rh + rng).any())
                    hit_tp = bool((rest["low"] <= rl).any())
                    hit_tp_c = bool((rest["low"] <= rl - extra).any())
                    side = "short"
                else:
                    hit_sl = bool((rest["low"] <= rl - rng).any())
                    hit_tp = bool((rest["high"] >= rh).any())
                    hit_tp_c = bool((rest["high"] >= rh + extra).any())
                    side = "long"
            elif side == "long":
                hit_sl = bool((rest["low"] <= rl).any())
                hit_tp = bool((rest["high"] >= rh + rng).any())
                hit_tp_c = bool((rest["high"] >= rh + rng + extra).any())
            else:
                hit_sl = bool((rest["high"] >= rh).any())
                hit_tp = bool((rest["low"] <= rl - rng).any())
                hit_tp_c = bool((rest["low"] <= rl - rng - extra).any())
            # Conservative: if both in the window, SL wins.
            success = hit_tp and not hit_sl
            success_cost = hit_tp_c and not hit_sl
            rows.append(
                {
                    "time": work.at[entry_i, "time"],
                    "symbol": symbol,
                    "treatment": treat,
                    "side": side,
                    "success": success,
                    "success_cost_adj": success_cost,
                    "regime": work.at[entry_i, "regime"] if "regime" in work.columns else "",
                    "trend_regime": work.at[entry_i, "trend_regime"] if "trend_regime" in work.columns else "",
                    "utc_date": str(day),
                }
            )
    return pd.DataFrame(rows)
