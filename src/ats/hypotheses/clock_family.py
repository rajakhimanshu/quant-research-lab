"""Named-session clocks. One handler, frozen per-hypothesis tz/hour/mode.

Not a scan of every hour. Not a reopen of H12/H23/H40/H47/H48/H53–H56.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ats.hypotheses.m15_micro import _trade
from ats.timeutil import cost_price


def _third_friday(d) -> bool:
    return d.weekday() == 4 and 15 <= d.day <= 21


def _imm_wednesday(d) -> bool:
    return d.month in {3, 6, 9, 12} and d.weekday() == 2 and 15 <= d.day <= 21


def clock_run_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Signed M15 run or fixed-side hold at treat clock vs control clock."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon = int(params.get("horizon_bars", 8))
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    min_prior = float(params.get("min_prior_atr", 0.25))
    mode = str(params.get("mode", "follow"))
    tz = ZoneInfo(str(params["tz"]))
    treat_h = int(params["treat_hour"])
    treat_m = int(params.get("treat_minute", 0))
    base_h = int(params["base_hour"])
    base_m = int(params.get("base_minute", 0))
    opex = bool(params.get("opex_friday", False))
    imm = bool(params.get("imm_wednesday", False))
    skip_friday = bool(params.get("skip_friday", False))
    skip_weekend = bool(params.get("skip_weekend", False))
    treat_dow = params.get("treat_dow")
    base_dow = params.get("base_dow")
    local = pd.to_datetime(work["time"], utc=True).dt.tz_convert(tz)
    work["h"] = local.dt.hour
    work["m"] = local.dt.minute
    work["d"] = local.dt.date
    work["dow"] = local.dt.dayofweek
    rows = []

    if mode in {"long", "short"}:
        for i in range(len(work)):
            dow = int(work.at[i, "dow"])
            if dow > 4 or (skip_friday and dow == 4) or (skip_weekend and dow > 4):
                continue
            hour, minute = int(work.at[i, "h"]), int(work.at[i, "m"])
            if hour == treat_h and minute == treat_m:
                treat = True
            elif hour == base_h and minute == base_m:
                treat = False
            else:
                continue
            if imm:
                if dow != 2:
                    continue
                treat = _imm_wednesday(work.at[i, "d"])
            row = _trade(work, i, mode, treat, symbol, cost, horizon, stop_atr, target_atr)
            if row:
                rows.append(row)
        return pd.DataFrame(rows)

    for day, g in work.groupby("d", sort=True):
        if skip_friday and day.weekday() == 4:
            continue
        if skip_weekend and day.weekday() > 4:
            continue
        if opex:
            if day.weekday() != 4:
                continue
            windows = ((g[(g["h"] == treat_h) & (g["m"] == treat_m)], _third_friday(day)),)
        elif imm:
            if day.weekday() != 2:
                continue
            windows = ((g[(g["h"] == treat_h) & (g["m"] == treat_m)], _imm_wednesday(day)),)
        elif treat_dow is not None:
            wd = day.weekday()
            if wd == int(treat_dow):
                windows = ((g[(g["h"] == treat_h) & (g["m"] == treat_m)], True),)
            elif base_dow is not None and wd == int(base_dow):
                windows = ((g[(g["h"] == base_h) & (g["m"] == base_m)], False),)
            else:
                continue
        else:
            windows = (
                (g[(g["h"] == treat_h) & (g["m"] == treat_m)], True),
                (g[(g["h"] == base_h) & (g["m"] == base_m)], False),
            )
        for subset, treat in windows:
            if subset.empty:
                continue
            i = int(subset.index[-1])
            earlier = g[g.index < i]
            if earlier.empty:
                prev = work.loc[work["d"] < day]
                if prev.empty:
                    continue
                j = int(prev.index[-1])
            else:
                j = int(earlier.index[-1])
            atr = float(work.at[i, "atr"])
            if not np.isfinite(atr) or atr <= 0:
                continue
            prior = float(work.at[i, "close"]) - float(work.at[j, "open"])
            if abs(prior) < min_prior * atr:
                continue
            follow = "long" if prior > 0 else "short"
            side = ("short" if follow == "long" else "long") if mode == "fade" else follow
            row = _trade(work, i, side, treat, symbol, cost, horizon, stop_atr, target_atr)
            if row:
                rows.append(row)
    return pd.DataFrame(rows)
