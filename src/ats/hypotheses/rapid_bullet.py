from __future__ import annotations

import numpy as np
import pandas as pd

from ats.features.levels import swing_points
from ats.timeutil import cost_price, pip_size


def _intact_level(
    px: np.ndarray,
    is_swing: np.ndarray,
    t: int,
    bars_n: int,
    lookback: int,
    want_high: bool,
) -> float | None:
    """Most recent confirmed swing that later bars have not taken out (v1.30 findHigh/findLow)."""
    run = -np.inf if want_high else np.inf
    last = min(lookback, t + 1)
    for i in range(last):
        idx = t - i
        val = px[idx]
        if i > bars_n and is_swing[idx]:
            if want_high and val > run:
                return float(val)
            if (not want_high) and val < run:
                return float(val)
        if want_high:
            run = val if not np.isfinite(run) else max(run, val)
        else:
            run = val if not np.isfinite(run) else min(run, val)
    return None


def _path_hit(
    high: np.ndarray,
    low: np.ndarray,
    start: int,
    sl: float,
    tp: float,
    side: str,
) -> str | None:
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


def _fill_then_path(
    high: np.ndarray,
    low: np.ndarray,
    start: int,
    expire: int,
    entry: float,
    sl: float,
    tp: float,
    tp_cost: float,
    side: str,
) -> tuple[str | None, str | None, int]:
    """Returns (raw hit, cost hit, last bar touched). last is expire-1 if no fill."""
    n = len(high)
    end = min(expire, n)
    fill = None
    for j in range(start, end):
        if side == "long" and high[j] >= entry:
            fill = j
            break
        if side == "short" and low[j] <= entry:
            fill = j
            break
    if fill is None:
        return None, None, end - 1
    raw = _path_hit(high, low, fill, sl, tp, side)
    cost_hit = _path_hit(high, low, fill, sl, tp_cost, side)
    last = fill
    if raw is not None:
        # last bar is fill or later; occupancy ends when SL/TP prints
        if side == "long":
            for j in range(fill, n):
                if low[j] <= sl or high[j] >= tp:
                    last = j
                    break
        else:
            for j in range(fill, n):
                if high[j] >= sl or low[j] <= tp:
                    last = j
                    break
    else:
        last = n - 1
    return raw, cost_hit, last


def rapid_bullet_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 1.2,
    slippage_pips: float = 0.5,
) -> pd.DataFrame:
    """Pending stop at the nearest intact swing (Rapid Bullet v1.30).

    treatment: buy/sell stop at the most recent unbroken BarsN swing.
    baseline: same stop mechanics at a Donchian extreme (no pause / no pooled stops).
    Trailing and session windows are overlays — not scored.
    """
    work = df.dropna(subset=["atr"]).reset_index(drop=True)
    n = len(work)
    bars_n = int(params.get("bars_n", 5))
    lookback = int(params.get("lookback_bars", 200))
    expire_bars = int(params.get("expiration_bars", 100))
    donchian_n = int(params.get("donchian_n", 20))
    if n < lookback + expire_bars + bars_n + 3:
        return pd.DataFrame()
    work = swing_points(work, bars_n)

    pip = pip_size(symbol)
    sl_off = float(params.get("sl_pips", 20.0)) * pip
    tp_off = float(params.get("tp_pips", 20.0)) * pip
    min_dist = float(params.get("order_distance_pips", 10.0)) * pip
    cost = cost_price(symbol, spread_pips, slippage_pips)

    high = work["high"].to_numpy(dtype=float)
    low = work["low"].to_numpy(dtype=float)
    close = work["close"].to_numpy(dtype=float)
    sh = work["swing_high"].to_numpy(dtype=bool)
    slg = work["swing_low"].to_numpy(dtype=bool)
    times = work["time"].to_numpy()

    d_high = pd.Series(high).rolling(donchian_n, min_periods=donchian_n).max().to_numpy()
    d_low = pd.Series(low).rolling(donchian_n, min_periods=donchian_n).min().to_numpy()

    rows: list[dict] = []
    busy = {"treat_long": -1, "treat_short": -1, "base_long": -1, "base_short": -1}

    def _try(
        t: int,
        entry: float | None,
        side: str,
        treat: bool,
        key: str,
    ) -> None:
        if entry is None or not np.isfinite(entry):
            return
        if t <= busy[key]:
            return
        mid = float(close[t])
        if side == "long":
            if mid > entry - min_dist:
                return
            sl = entry - sl_off
            tp = entry + tp_off
            tp_c = entry + tp_off + cost
        else:
            if mid < entry + min_dist:
                return
            sl = entry + sl_off
            tp = entry - tp_off
            tp_c = entry - tp_off - cost
        if sl_off <= 0.0 or tp_off <= 0.0:
            return
        raw, cost_hit, last = _fill_then_path(
            high, low, t + 1, t + 1 + expire_bars, entry, sl, tp, tp_c, side
        )
        busy[key] = last
        if raw is None:
            return
        win = raw == "tp"
        rows.append(
            {
                "time": times[t],
                "symbol": symbol,
                "treatment": treat,
                "side": side,
                "success": win,
                "success_cost_adj": cost_hit == "tp",
                "r_mult": 1.0 if win else -1.0,
                "regime": work.at[t, "regime"] if "regime" in work.columns else "",
                "trend_regime": work.at[t, "trend_regime"] if "trend_regime" in work.columns else "",
            }
        )

    start = max(lookback, donchian_n, bars_n * 2 + 1)
    for t in range(start, n - 2):
        _try(
            t,
            _intact_level(high, sh, t, bars_n, lookback, True),
            "long",
            True,
            "treat_long",
        )
        _try(
            t,
            _intact_level(low, slg, t, bars_n, lookback, False),
            "short",
            True,
            "treat_short",
        )
        _try(t, float(d_high[t]) if np.isfinite(d_high[t]) else None, "long", False, "base_long")
        _try(t, float(d_low[t]) if np.isfinite(d_low[t]) else None, "short", False, "base_short")

    return pd.DataFrame(rows)
