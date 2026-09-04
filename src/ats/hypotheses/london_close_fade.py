"""H12: fade the prior H1 run into London close vs the same fade at London open."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price

LONDON = ZoneInfo("Europe/London")


def london_close_fade_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 1.5,
    slippage_pips: float = 0.5,
) -> pd.DataFrame:
    """treatment = 15:00 London flatten; baseline = 08:00 London. Same fade of prior 7 H1 bars."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    london = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["london_hour"] = london.dt.hour
    lookback = int(params.get("lookback_bars", 7))
    horizon = int(params.get("horizon_bars", 2))
    treat_hour = int(params.get("flatten_hour_london", 15))
    base_hour = int(params.get("open_hour_london", 8))
    min_prior = float(params.get("min_prior_atr", 0.25))
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    rows = []
    n = len(work)
    for i in range(lookback, n - horizon):
        hour = int(work.at[i, "london_hour"])
        if hour == treat_hour:
            treat = True
        elif hour == base_hour:
            treat = False
        else:
            continue
        atr = float(work.at[i, "atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue
        prior = float(work.at[i - 1, "close"]) - float(work.at[i - lookback, "close"])
        if abs(prior) < min_prior * atr:
            continue
        side = "short" if prior > 0 else "long"
        raw_open = float(work.at[i, "open"])
        if side == "long":
            entry = raw_open + cost
            stop = entry - stop_atr * atr
            target = entry + target_atr * atr
        else:
            entry = raw_open - cost
            stop = entry + stop_atr * atr
            target = entry - target_atr * atr
        # Enter at this bar's open: path includes the signal bar.
        hit, r_mult = _forward(work, i - 1, side, entry, target, stop, horizon)
        rows.append(
            {
                "time": work.at[i, "time"],
                "symbol": symbol,
                "treatment": treat,
                "side": side,
                "success": bool(hit),
                "success_cost_adj": bool(r_mult > 0),
                "r_mult": float(r_mult),
                "london_hour": hour,
            }
        )
    return pd.DataFrame(rows)
