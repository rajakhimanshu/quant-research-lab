"""Gulf UTC+4 dual-hour liquidity fade: 09:00 sweeps max(05,08) H / min(05,08) L."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price

DUBAI = ZoneInfo("Asia/Dubai")  # UTC+4, no DST


def _hour_hl(g: pd.DataFrame, hour: int) -> tuple[float, float] | None:
    sub = g[g["gulf_hour"] == hour]
    if sub.empty:
        return None
    return float(sub["high"].max()), float(sub["low"].min())


def _rr_trade(
    work: pd.DataFrame,
    i: int,
    side: str,
    treat: bool,
    symbol: str,
    cost: float,
    sweep_ext: float,
    horizon: int,
    rr: float,
    sweep_buf: float,
) -> dict | None:
    n = len(work)
    if i + 1 + horizon >= n:
        return None
    raw_open = float(work.at[i + 1, "open"])
    if side == "short":
        entry = raw_open - cost
        stop = sweep_ext + sweep_buf
        if stop <= entry:
            return None
        risk = stop - entry
        target = entry - rr * risk
    else:
        entry = raw_open + cost
        stop = sweep_ext - sweep_buf
        if stop >= entry:
            return None
        risk = entry - stop
        target = entry + rr * risk
    if risk <= 0:
        return None
    hit, r_mult = _forward(work, i, side, entry, target, stop, horizon)
    return {
        "time": work.at[i + 1, "time"],
        "symbol": symbol,
        "treatment": treat,
        "side": side,
        "success": bool(hit),
        "success_cost_adj": bool(r_mult > 0),
        "r_mult": float(r_mult),
        "entry": float(entry),
        "stop": float(stop),
        "target": float(target),
        "atr": float(work.at[i, "atr"]) if np.isfinite(work.at[i, "atr"]) else float("nan"),
        "raw_open": float(raw_open),
        "sweep_ext": float(sweep_ext),
    }


def gulf_dual_liq_fade_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 200,
    slippage_pips: float = 20,
) -> pd.DataFrame:
    """Fade first 09:00 Gulf take of both 05:00 and 08:00 hour extremes vs 11:00.

    Liquidity high = max(H_05, H_08); liquidity low = min(L_05, L_08).
    High taken → short; low taken → long. SL beyond sweep extreme; TP = rr * risk.
    """
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()

    gulf = pd.to_datetime(work["time"], utc=True).dt.tz_convert(DUBAI)
    work["gulf_date"] = gulf.dt.date
    work["gulf_hour"] = gulf.dt.hour
    work["gulf_dow"] = gulf.dt.dayofweek

    h5 = int(params.get("ref_hour_a", 5))
    h8 = int(params.get("ref_hour_b", 8))
    treat_h = int(params.get("treat_hour", 9))
    base_h = int(params.get("base_hour", 11))
    horizon = int(params.get("horizon_bars", 36))
    rr = float(params.get("rr", 2.0))
    sweep_atr = float(params.get("sweep_atr", 0.05))
    skip_weekend = bool(params.get("skip_weekend", True))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    rows: list[dict] = []

    for day, g in work.groupby("gulf_date", sort=True):
        if skip_weekend and int(g["gulf_dow"].iloc[0]) >= 5:
            continue
        a = _hour_hl(g, h5)
        b = _hour_hl(g, h8)
        if a is None or b is None:
            continue
        liq_high = max(a[0], b[0])
        liq_low = min(a[1], b[1])
        if not (np.isfinite(liq_high) and np.isfinite(liq_low)) or liq_high <= liq_low:
            continue

        windows = (
            (g[g["gulf_hour"] == treat_h], True),
            (g[g["gulf_hour"] == base_h], False),
        )
        for subset, treat in windows:
            if subset.empty:
                continue
            for i in subset.index:
                atr = float(work.at[i, "atr"])
                if not np.isfinite(atr) or atr <= 0:
                    continue
                buf = sweep_atr * atr
                high = float(work.at[i, "high"])
                low = float(work.at[i, "low"])
                side = None
                sweep_ext = None
                if high > liq_high + buf:
                    side = "short"
                    sweep_ext = high
                elif low < liq_low - buf:
                    side = "long"
                    sweep_ext = low
                if side is None or sweep_ext is None:
                    continue
                row = _rr_trade(
                    work, int(i), side, treat, symbol, cost, sweep_ext, horizon, rr, buf
                )
                if row:
                    rows.append(row)
                break
    return pd.DataFrame(rows)
