from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class LiveLevel:
    level_id: int
    kind: str
    price: float
    born: int
    in_zone: bool = False
    taps: list = field(default_factory=list)


def _incoming_range(kind: str, high: np.ndarray, low: np.ndarray, t: int, lookback: int) -> float:
    start = max(0, t - lookback)
    if kind == "resistance":
        return float(high[t] - low[start : t + 1].min())
    return float(high[start : t + 1].max() - low[t])


def _classify_outcome(
    kind: str,
    t: int,
    price: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    params: dict,
) -> tuple[str, float]:
    horizon = int(params["tap_window_bars"])
    sustain = int(params["sustain_bars"])
    hold_need = float(params["hold_retrace"])
    fail_need = float(params["fail_retrace"])
    brk_atr = float(params["breakout_atr"])
    end = min(len(close) - 1, t + horizon)
    if end <= t:
        return "incomplete", float("nan")

    incoming = _incoming_range(kind, high, low, t, lookback=20)
    incoming = incoming if incoming > 0 else float(atr[t] or 1e-9)
    fut_high = high[t + 1 : end + 1]
    fut_low = low[t + 1 : end + 1]
    if kind == "resistance":
        retrace = float(price - fut_low.min()) / incoming
        penetrate = close[t + 1 : end + 1] > (price + brk_atr * atr[t])
    else:
        retrace = float(fut_high.max() - price) / incoming
        penetrate = close[t + 1 : end + 1] < (price - brk_atr * atr[t])

    if penetrate.any():
        first = int(np.argmax(penetrate))
        abs_i = t + 1 + first
        last = min(len(close) - 1, abs_i + sustain)
        if last - abs_i >= sustain:
            held = close[abs_i : last + 1]
            if kind == "resistance" and (held > price).mean() >= 0.8:
                return "break", retrace
            if kind == "support" and (held < price).mean() >= 0.8:
                return "break", retrace

    if retrace >= hold_need:
        return "hold", retrace
    if retrace < fail_need:
        return "fail", retrace
    return "neutral", retrace


def tap_events(df: pd.DataFrame, symbol: str, params: dict) -> pd.DataFrame:
    """Causal taps: a swing becomes a level only after it is confirmed."""
    n = int(params["swing_n"])
    tol_mult = float(params["level_tolerance_atr"])
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float)
    times = pd.to_datetime(df["time"], utc=True)
    swing_high = df["swing_high"].to_numpy(dtype=bool)
    swing_low = df["swing_low"].to_numpy(dtype=bool)
    levels: list[LiveLevel] = []
    rows = []
    next_id = 1

    def upsert(kind: str, price: float, born: int, local_atr: float) -> None:
        nonlocal next_id
        if not np.isfinite(local_atr) or local_atr <= 0:
            return
        tol = local_atr * tol_mult
        for lv in levels:
            if lv.kind == kind and abs(lv.price - price) <= tol:
                lv.price = (lv.price + price) / 2.0
                return
        levels.append(LiveLevel(level_id=next_id, kind=kind, price=price, born=born))
        next_id += 1

    for t in range(len(df)):
        if t >= n:
            pivot = t - n
            a = atr[pivot]
            if swing_high[pivot]:
                upsert("resistance", high[pivot], t, a)
            if swing_low[pivot]:
                upsert("support", low[pivot], t, a)

        if not np.isfinite(atr[t]) or atr[t] <= 0:
            continue
        if t + int(params["tap_window_bars"]) + int(params["sustain_bars"]) >= len(df):
            continue

        for lv in levels:
            if lv.born >= t:
                continue
            tol = atr[t] * float(params["level_tolerance_atr"])
            if lv.kind == "resistance":
                touching = (high[t] >= lv.price - tol) and (low[t] <= lv.price + tol) and (close[t] <= lv.price + tol)
            else:
                touching = (low[t] <= lv.price + tol) and (high[t] >= lv.price - tol) and (close[t] >= lv.price - tol)

            if touching and not lv.in_zone:
                outcome, retrace = _classify_outcome(lv.kind, t, lv.price, high, low, close, atr, params)
                if outcome == "incomplete":
                    continue
                tap_no = len(lv.taps) + 1
                rec = {
                    "time": times.iloc[t],
                    "symbol": symbol,
                    "kind": lv.kind,
                    "level_id": lv.level_id,
                    "level": lv.price,
                    "tap_no": tap_no,
                    "outcome": outcome,
                    "retrace": retrace,
                    "regime": df.iloc[t]["regime"],
                    "trend_regime": df.iloc[t]["trend_regime"],
                }
                lv.taps.append(rec)
                rows.append(rec)
                lv.in_zone = True
            elif not touching:
                lv.in_zone = False

    events = pd.DataFrame(rows)
    if events.empty:
        return events

    # H2 treatment: 3rd tap whose previous two were hold then fail, on the same level.
    events["treatment"] = False
    events["success"] = events["outcome"] == "break"
    events["success_cost_adj"] = events["success"]
    grouped = events.groupby(["symbol", "kind", "level_id"], sort=False)
    treat_index = []
    for _, g in grouped:
        g = g.sort_values("time")
        outs = g["outcome"].tolist()
        idxs = list(g.index)
        for k in range(2, len(outs)):
            if outs[k - 2] == "hold" and outs[k - 1] == "fail":
                treat_index.append(idxs[k])
    events.loc[treat_index, "treatment"] = True
    # Baseline = all 1st and 2nd taps (break rate we compare against).
    events["baseline_flag"] = events["tap_no"].isin([1, 2])
    return events
