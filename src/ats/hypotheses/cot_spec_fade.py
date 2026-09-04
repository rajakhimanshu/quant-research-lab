"""H9: fade crowded CFTC non-commercial gold positioning vs a mid-percentile fade."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ats.ideas.cot import gold_series, load_combined
from ats.timeutil import cost_price


def _to_daily(df: pd.DataFrame) -> pd.DataFrame:
    w = df.copy()
    w["time"] = pd.to_datetime(w["time"], utc=True)
    w = w.set_index("time").sort_index()
    d = w.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    d = d.dropna(how="any")
    prev = d["close"].shift(1)
    tr = pd.concat(
        [d["high"] - d["low"], (d["high"] - prev).abs(), (d["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    return d.dropna(subset=["atr"])


def rolling_last_pct(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).apply(
        lambda x: float((x <= x[-1]).mean()),
        raw=True,
    )


def _forward(
    daily: pd.DataFrame,
    i: int,
    side: str,
    entry: float,
    target: float,
    stop: float,
    horizon: int,
) -> tuple[bool, float]:
    risk = abs(entry - stop)
    if risk <= 0 or not np.isfinite(risk):
        return False, 0.0
    future = daily.iloc[i + 1 : i + 1 + horizon]
    if future.empty:
        return False, 0.0
    for _, bar in future.iterrows():
        hi = float(bar["high"])
        lo = float(bar["low"])
        if side == "long":
            if lo <= stop:
                return False, (stop - entry) / risk
            if hi >= target:
                return True, (target - entry) / risk
        else:
            if hi >= stop:
                return False, (entry - stop) / risk
            if lo <= target:
                return True, (entry - target) / risk
    last = float(future.iloc[-1]["close"])
    r_mult = (last - entry) / risk if side == "long" else (entry - last) / risk
    return r_mult > 0, r_mult


def cot_spec_fade_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 240,
    slippage_pips: float = 50,
    cot_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Weekly fade of spec net. treatment = extreme percentile, baseline = mid band."""
    if cot_df is None:
        combined = load_combined()
        if combined is None or combined.empty:
            return pd.DataFrame()
        cot_df = gold_series(combined)
    if cot_df is None or cot_df.empty or df.empty:
        return pd.DataFrame()

    window = int(params.get("percentile_window", 52))
    hi = float(params.get("extreme_hi", 0.80))
    lo = float(params.get("extreme_lo", 0.20))
    mid_lo = float(params.get("mid_lo", 0.40))
    mid_hi = float(params.get("mid_hi", 0.60))
    lag = int(params.get("publish_lag_days", 6))
    horizon = int(params.get("horizon_days", 5))
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    cost = cost_price(symbol, spread_pips, slippage_pips)

    g = cot_df.copy()
    g["asof"] = pd.to_datetime(g["asof"], utc=True)
    g = g.sort_values("asof").drop_duplicates("asof")
    g["pct"] = rolling_last_pct(g["nc_net_oi"].astype(float), window)
    g = g.dropna(subset=["pct", "nc_net"])

    daily = _to_daily(df)
    if daily.empty:
        return pd.DataFrame()
    idx = daily.index

    rows = []
    for _, row in g.iterrows():
        pct = float(row["pct"])
        extreme = pct >= hi or pct <= lo
        mid = mid_lo <= pct <= mid_hi
        if not extreme and not mid:
            continue
        trade_ts = pd.Timestamp(row["asof"]) + pd.Timedelta(days=lag)
        loc = int(idx.searchsorted(trade_ts))
        if loc >= len(daily) - horizon - 1:
            continue
        bar = daily.iloc[loc]
        atr = float(bar["atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue
        net = float(row["nc_net"])
        side = "short" if net > 0 else "long"
        raw_open = float(bar["open"])
        if side == "long":
            entry = raw_open + cost
            stop = entry - stop_atr * atr
            target = entry + target_atr * atr
        else:
            entry = raw_open - cost
            stop = entry + stop_atr * atr
            target = entry - target_atr * atr
        hit, r_mult = _forward(daily, loc, side, entry, target, stop, horizon)
        hit_cost = r_mult > 0
        rows.append(
            {
                "time": idx[loc],
                "symbol": symbol,
                "treatment": bool(extreme),
                "side": side,
                "success": bool(hit),
                "success_cost_adj": bool(hit_cost),
                "r_mult": float(r_mult),
                "pct": pct,
                "nc_net_oi": float(row["nc_net_oi"]),
            }
        )
    return pd.DataFrame(rows)
