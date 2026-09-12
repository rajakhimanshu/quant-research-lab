"""H86 paper scan. Frozen weekend-gap fade. Not live money. Not an EA."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ats.config import load_hypotheses, load_settings
from ats.hypotheses.fx_session import weekend_gap_fx_events
from ats.timeutil import cost_price, locked_calendar_split, pip_size

LONDON = ZoneInfo("Europe/London")
H86_ID = "H86_weekend_gap_g10"
STARTING_EQUITY = 2000.0
RISK_PCT = 0.01
MIN_LOT = 0.01


def h86_hyp() -> dict:
    return next(h for h in load_hypotheses() if h["id"] == H86_ID)


def h86_params() -> dict:
    return dict(h86_hyp()["params"])


def pair_cost(symbol: str, settings: dict) -> tuple[float, float]:
    costs = settings["costs"]
    spread = float(costs["spread_pips"].get(symbol, 1.5))
    slip = float((costs.get("slippage_by_symbol") or {}).get(symbol, costs["slippage_pips"]))
    return spread, slip


def oos_treatment(frames: dict[str, pd.DataFrame], settings: dict) -> pd.DataFrame:
    params = h86_params()
    split = locked_calendar_split(settings, "fx_h1")
    chunks = []
    for symbol, df in frames.items():
        spread, slip = pair_cost(symbol, settings)
        ev = weekend_gap_fx_events(df, symbol, params, spread, slip)
        if ev.empty:
            continue
        ev = ev.loc[ev["treatment"]].copy()
        ev["time"] = pd.to_datetime(ev["time"], utc=True)
        ev = ev.loc[ev["time"] > split.val_end]
        chunks.append(ev)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).sort_values("time")


def replay_oos_equity(oos: pd.DataFrame, starting: float = STARTING_EQUITY, risk_pct: float = RISK_PCT) -> dict:
    """Stack independent 1% R hits. Three pairs can fire the same Sunday."""
    eq = float(starting)
    peak = eq
    max_dd = 0.0
    rs = []
    if oos.empty:
        return {
            "starting": starting,
            "ending": starting,
            "return_pct": 0.0,
            "max_dd_pct": 0.0,
            "n": 0,
            "mean_r": float("nan"),
            "total_r": 0.0,
        }
    for r in oos["r_mult"].astype(float):
        eq = eq + eq * risk_pct * r
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak if peak else 0.0)
        rs.append(float(r))
    return {
        "starting": starting,
        "ending": round(eq, 2),
        "return_pct": round(100 * (eq / starting - 1), 2),
        "max_dd_pct": round(100 * max_dd, 2),
        "n": len(rs),
        "mean_r": float(np.mean(rs)),
        "total_r": float(np.sum(rs)),
    }


def _last_halt_gap(df: pd.DataFrame) -> tuple[int, float] | None:
    times = pd.to_datetime(df["time"], utc=True)
    gaps = times.diff().dt.total_seconds() / 3600.0
    for i in range(len(df) - 1, 0, -1):
        g = float(gaps.iloc[i])
        if np.isfinite(g) and g >= 36:
            return i, g
    return None


def scan_symbol(df: pd.DataFrame, symbol: str, settings: dict, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    params = h86_params()
    horizon = int(params.get("horizon_bars", 8))
    min_atr = float(params.get("min_gap_atr", 0.15))
    empty = {
        "symbol": symbol,
        "action": "FLAT",
        "side": "",
        "note": "no bars",
        "gap_hours": None,
        "entry": None,
        "stop": None,
        "target": None,
        "asof": now.isoformat(),
    }
    if work.empty:
        return empty
    times = pd.to_datetime(work["time"], utc=True)
    last_i = len(work) - 1
    last_t = times.iloc[last_i]
    hours_since = (pd.Timestamp(now) - last_t).total_seconds() / 3600.0
    now_ldn = pd.Timestamp(now).tz_convert(LONDON)
    pre_reopen = now_ldn.dayofweek == 5 or (now_ldn.dayofweek == 6 and now_ldn.hour < 21)
    if pre_reopen or hours_since >= 36:
        return {
            **empty,
            "action": "WAIT_HALT",
            "note": (
                f"Weekend halt. Fade the Sunday reopen (~21:00/22:00 London) if the "
                f"gap is >=36h and >=0.15 ATR. Exness demo 0.01 lot. Not live. Not an EA."
            ),
            "gap_hours": round(hours_since, 1),
            "asof": last_t.isoformat(),
        }
    hit = _last_halt_gap(work)
    if hit is None:
        return {**empty, "action": "WAIT_HALT", "note": "no >=36h gap on this file", "asof": last_t.isoformat()}
    i, gap_h = hit
    atr = float(work.at[i, "atr"])
    move = float(work.at[i, "open"]) - float(work.at[i - 1, "close"])
    gap_atr = abs(move) / atr if atr > 0 else 0.0
    bars_held = last_i - i
    spread, slip = pair_cost(symbol, settings)
    cost = cost_price(symbol, spread, slip)
    side = "short" if move > 0 else "long"
    raw = float(work.at[i, "open"])
    if side == "long":
        entry, stop, target = raw + cost, raw + cost - atr, raw + cost + atr
    else:
        entry, stop, target = raw - cost, raw - cost + atr, raw - cost - atr
    pip = pip_size(symbol)
    stop_pips = abs(entry - stop) / pip if pip else None
    if gap_atr < min_atr:
        return {
            **empty,
            "action": "SKIP",
            "note": f"reopen gap {gap_atr:.2f} ATR < {min_atr} (frozen). Do not loosen.",
            "gap_hours": gap_h,
            "asof": times.iloc[i].isoformat(),
        }
    if bars_held == 0:
        action, note = "ENTER", f"fade {side} the reopen now. 1:1 ATR, {horizon} H1, demo 0.01 lot."
    elif bars_held < horizon:
        action, note = "HOLD", f"in trade, bar {bars_held}/{horizon}. Manage 1:1 ATR."
    else:
        action, note = "FLAT", f"last reopen is {bars_held} bars old. Wait next halt."
    return {
        "symbol": symbol,
        "action": action,
        "side": side if action in {"ENTER", "HOLD"} else "",
        "note": note,
        "gap_hours": round(gap_h, 1),
        "gap_atr": round(gap_atr, 3),
        "entry": round(entry, 5),
        "stop": round(stop, 5),
        "target": round(target, 5),
        "stop_pips": round(stop_pips, 1) if stop_pips is not None else None,
        "bars_held": int(bars_held),
        "asof": times.iloc[i].isoformat(),
    }


def scan_book(frames: dict[str, pd.DataFrame], settings: dict | None = None, now: datetime | None = None) -> pd.DataFrame:
    settings = settings or load_settings()
    rows = [scan_symbol(df, symbol, settings, now) for symbol, df in frames.items()]
    return pd.DataFrame(rows)
