"""Paper book for H5 — frozen live-config risk, not a new strategy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ats.hypotheses.equity_rsi2 import atr, rsi_wilder
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")


def drop_incomplete_us_session(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Decisions use the last completed US cash session, not a partial today bar."""
    if df.empty:
        return df
    now = now or datetime.now(timezone.utc)
    now_ny = now.astimezone(NY)
    closed = now_ny.weekday() < 5 and (now_ny.hour > 16 or (now_ny.hour == 16 and now_ny.minute >= 5))
    last = pd.Timestamp(df.index.max())
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    last_ny = last.tz_convert(NY)
    last_utc_date = last.astimezone(timezone.utc).date()
    session_date = now_ny.date()
    is_today_bar = last_ny.date() == session_date or last_utc_date == session_date
    if is_today_bar and not closed:
        trimmed = df.iloc[:-1]
        return trimmed if len(trimmed) else df
    return df

STARTING_EQUITY = 10_000.0
RISK_PCT_PER_TRADE = 0.005
MAX_OPEN_POSITIONS = 4
MAX_DRAWDOWN_PCT = 0.12


def shares_for_risk(equity: float, risk_pct: float, entry_price: float, stop_price: float) -> int:
    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0 or entry_price <= 0 or equity <= 0:
        return 0
    shares = int((equity * risk_pct) // risk_per_share)
    max_affordable = int(equity // entry_price)
    return max(0, min(shares, max_affordable))


def prepare_book_frame(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out["rsi2"] = rsi_wilder(out["Close"], 2)
    out["sma_trend"] = out["Close"].rolling(int(params.get("trend_sma", 200))).mean()
    out["sma_exit"] = out["Close"].rolling(int(params.get("exit_sma", 5))).mean()
    out["atr"] = atr(out, 20)
    return out.sort_index()


@dataclass
class BookResult:
    start: str
    starting_equity: float
    ending_equity: float
    peak_equity: float
    max_dd_pct: float
    halted: bool
    n_trades: int
    win_rate: float | None
    mean_r: float | None
    total_r: float | None
    total_pnl: float
    open_positions: dict
    trades: pd.DataFrame
    equity_curve: list[dict] = field(default_factory=list)


def replay_paper_book(
    frames: dict[str, pd.DataFrame],
    params: dict,
    start: pd.Timestamp,
    starting_equity: float = STARTING_EQUITY,
) -> BookResult:
    rsi_in = float(params.get("rsi_entry", 10))
    rsi_out = float(params.get("rsi_exit", 65))
    max_hold = int(params.get("max_hold", 10))
    atr_stop = float(params.get("atr_stop", 2.0))
    cost = float(params.get("cost_one_way", 0.00015))
    prepared = {s: prepare_book_frame(df, params) for s, df in frames.items()}
    start = pd.Timestamp(start)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")

    dates = sorted({d for df in prepared.values() for d in df.index if d >= start})
    positions: dict[str, dict] = {}
    equity = float(starting_equity)
    peak = float(starting_equity)
    max_dd = 0.0
    halted = False
    trades: list[dict] = []
    curve: list[dict] = []

    loc = {s: {ts: i for i, ts in enumerate(df.index)} for s, df in prepared.items()}

    def dd_pct() -> float:
        return (peak - equity) / peak if peak > 0 else 0.0

    for d in dates:
        # Exits
        for sym in list(positions):
            df = prepared[sym]
            i = loc[sym].get(d)
            if i is None or i < 1:
                continue
            pos = positions[sym]
            prev = df.iloc[i - 1]
            row = df.iloc[i]
            hold = i - pos["entry_i"]
            hit_stop = float(row["Low"]) <= pos["stop"]
            exit_sma = float(prev["Close"]) > float(prev["sma_exit"])
            exit_rsi = float(prev["rsi2"]) > rsi_out
            exit_time = hold >= max_hold
            if not (hit_stop or exit_sma or exit_rsi or exit_time):
                continue
            if hit_stop:
                fill = pos["stop"] * (1 - cost)
                reason = "hard_stop"
            else:
                fill = float(row["Open"]) * (1 - cost)
                reason = "sma5" if exit_sma else ("rsi_exit" if exit_rsi else "time_stop")
            pnl = (fill - pos["entry"]) * pos["shares"]
            risk = pos["risk_ps"]
            r_mult = (fill - pos["entry"]) / risk if risk > 0 else 0.0
            equity += pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, dd_pct())
            trades.append(
                {
                    "symbol": sym,
                    "entry_time": pos["entry_time"],
                    "exit_time": d,
                    "reason": reason,
                    "r_mult": r_mult,
                    "pnl_usd": pnl,
                    "equity_after": equity,
                    "shares": pos["shares"],
                }
            )
            del positions[sym]

        halted = dd_pct() >= MAX_DRAWDOWN_PCT
        slots = MAX_OPEN_POSITIONS - len(positions)
        if halted or slots <= 0:
            curve.append({"time": d, "equity": equity, "open": len(positions), "halted": halted})
            continue

        cands = []
        for sym, df in prepared.items():
            if sym in positions:
                continue
            i = loc[sym].get(d)
            if i is None or i < 1:
                continue
            prev = df.iloc[i - 1]
            row = df.iloc[i]
            rsi = float(prev["rsi2"])
            trend = float(prev["sma_trend"])
            atr_i = float(prev["atr"])
            close = float(prev["Close"])
            if not (np.isfinite(rsi) and np.isfinite(trend) and np.isfinite(atr_i) and atr_i > 0):
                continue
            if rsi >= rsi_in or close <= trend:
                continue
            entry = float(row["Open"]) * (1 + cost)
            stop = entry - atr_stop * atr_i
            cands.append((rsi, sym, entry, stop, i, d))
        cands.sort(key=lambda x: x[0])
        for rsi, sym, entry, stop, i, et in cands:
            if len(positions) >= MAX_OPEN_POSITIONS:
                break
            sh = shares_for_risk(equity, RISK_PCT_PER_TRADE, entry, stop)
            if sh <= 0:
                continue
            positions[sym] = {
                "shares": sh,
                "entry": entry,
                "stop": stop,
                "entry_i": i,
                "entry_time": et,
                "risk_ps": entry - stop,
            }

        curve.append({"time": d, "equity": equity, "open": len(positions), "halted": halted})

    mtm = equity
    for sym, pos in positions.items():
        last = float(prepared[sym]["Close"].iloc[-1])
        mtm += (last * (1 - cost) - pos["entry"]) * pos["shares"]

    tdf = pd.DataFrame(trades)
    return BookResult(
        start=str(start),
        starting_equity=starting_equity,
        ending_equity=mtm,
        peak_equity=peak,
        max_dd_pct=max_dd,
        halted=halted,
        n_trades=int(len(tdf)),
        win_rate=float((tdf["r_mult"] > 0).mean()) if len(tdf) else None,
        mean_r=float(tdf["r_mult"].mean()) if len(tdf) else None,
        total_r=float(tdf["r_mult"].sum()) if len(tdf) else None,
        total_pnl=float(mtm - starting_equity),
        open_positions={
            s: {"shares": p["shares"], "entry": p["entry"], "stop": p["stop"], "entry_time": str(p["entry_time"])}
            for s, p in positions.items()
        },
        trades=tdf,
        equity_curve=curve[-8:] if curve else [],
    )


def scan_signals(frames: dict[str, pd.DataFrame], params: dict, open_symbols: set[str] | None = None) -> pd.DataFrame:
    rsi_in = float(params.get("rsi_entry", 10))
    rsi_out = float(params.get("rsi_exit", 65))
    open_symbols = open_symbols or set()
    rows = []
    for sym, raw in frames.items():
        df = drop_incomplete_us_session(prepare_book_frame(raw, params))
        if len(df) < 5:
            continue
        last = df.iloc[-1]
        rsi = float(last["rsi2"]) if np.isfinite(last["rsi2"]) else np.nan
        action = "HOLD_CASH"
        reason = ""
        if sym in open_symbols:
            action = "HOLD"
            reason = "open_in_paper_book"
            if np.isfinite(last["sma_exit"]) and float(last["Close"]) > float(last["sma_exit"]):
                action, reason = "EXIT", "sma5"
            elif np.isfinite(rsi) and rsi > rsi_out:
                action, reason = "EXIT", "rsi_exit"
        elif (
            np.isfinite(rsi)
            and np.isfinite(last["sma_trend"])
            and np.isfinite(last["atr"])
            and rsi < rsi_in
            and float(last["Close"]) > float(last["sma_trend"])
        ):
            action = "ENTER"
            reason = f"rsi2={rsi:.1f}<{rsi_in:g}"
        rows.append(
            {
                "symbol": sym,
                "asof": str(df.index[-1].date()),
                "close": float(last["Close"]),
                "rsi2": rsi,
                "sma200": float(last["sma_trend"]) if np.isfinite(last["sma_trend"]) else np.nan,
                "atr": float(last["atr"]) if np.isfinite(last["atr"]) else np.nan,
                "action": action,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows).sort_values(["action", "rsi2"]).reset_index(drop=True)
