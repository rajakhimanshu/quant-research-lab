from __future__ import annotations

import pandas as pd

import numpy as np

from ats.data.calendar import load_calendar, pair_currencies
from ats.timeutil import cost_price


def _retrace_after_large(df: pd.DataFrame, i: int, horizon: int) -> float:
    row = df.iloc[i]
    rng = float(row["range"])
    if rng <= 0:
        return 0.0
    future = df.iloc[i + 1 : i + 1 + horizon]
    if future.empty:
        return float("nan")
    if row["direction"] >= 0:
        retrace = (float(row["high"]) - float(future["low"].min())) / rng
    else:
        retrace = (float(future["high"].max()) - float(row["low"])) / rng
    return float(max(0.0, retrace))


def event_reversal_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    calendar: pd.DataFrame | None = None,
    spread_pips: float = 1.2,
    slippage_pips: float = 0.5,
) -> pd.DataFrame:
    """Each row is a large/extreme candle. treatment=near high-impact news."""
    calendar = calendar if calendar is not None else load_calendar()
    horizon = int(params["horizon_bars"])
    window = pd.Timedelta(minutes=int(params["event_window_minutes"]))
    need = float(params["retrace_pct"])
    min_atr = float(params["large_atr_mult"])
    impact = str(params.get("min_impact", "high")).lower()

    work = df.dropna(subset=["atr"]).reset_index(drop=True)
    large_idx = work.index[work["atr_multiple"] >= min_atr]
    if len(work) == 0:
        return pd.DataFrame()

    ccy = pair_currencies(symbol)
    if calendar.empty:
        news_times = pd.DatetimeIndex([], tz="UTC")
    else:
        news = calendar[
            (calendar["currency"].isin(ccy)) & (calendar["impact"] == impact)
        ]
        news_times = pd.DatetimeIndex(pd.to_datetime(news["datetime_utc"], utc=True))

    rows = []
    cost = cost_price(symbol, spread_pips, slippage_pips)
    for i in large_idx:
        i = int(i)
        if i + horizon >= len(work):
            continue
        ts = pd.Timestamp(work.at[i, "time"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        near = False
        if len(news_times):
            near = bool((np.abs(news_times - ts) <= window).any())
        retrace = _retrace_after_large(work, i, horizon)
        rng = float(work.at[i, "range"])
        cost_frac = cost / rng if rng else 1.0
        rows.append(
            {
                "time": ts,
                "symbol": symbol,
                "treatment": near,
                "retrace": retrace,
                "success": retrace >= need,
                "success_cost_adj": retrace >= (need + cost_frac),
                "atr_multiple": float(work.at[i, "atr_multiple"]),
                "regime": work.at[i, "regime"],
                "trend_regime": work.at[i, "trend_regime"],
                "session_peak_ny": bool(work.at[i, "session_peak_ny"])
                if "session_peak_ny" in work.columns
                else False,
                "direction": int(work.at[i, "direction"]),
            }
        )
    return pd.DataFrame(rows)
