"""Forward paper log for H5. Separate from the OOS replay. Not live money."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from ats.config import DATA_DIR, ROOT
from ats.paper.h5_book import (
    MAX_DRAWDOWN_PCT,
    MAX_OPEN_POSITIONS,
    RISK_PCT_PER_TRADE,
    STARTING_EQUITY,
    shares_for_risk,
)

PAPER = DATA_DIR / "paper"
STATE_PATH = PAPER / "h5_forward_state.json"
LOG_PATH = PAPER / "h5_forward_log.csv"
JOURNAL_PATH = ROOT / "research" / "journal.md"
TARGET_FILLS = 30
TARGET_MEAN_R = 0.10

COLS = [
    "event_id",
    "timestamp",
    "event",
    "symbol",
    "shares",
    "price",
    "stop",
    "reason",
    "r",
    "pnl_usd",
    "equity_after",
]


def _empty_log() -> pd.DataFrame:
    return pd.DataFrame(columns=COLS)


def load_state() -> dict:
    PAPER.mkdir(parents=True, exist_ok=True)
    if not STATE_PATH.exists():
        return {
            "equity": STARTING_EQUITY,
            "peak_equity": STARTING_EQUITY,
            "open": {},
            "halt_new_entries": False,
            "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def load_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return _empty_log()
    df = pd.read_csv(LOG_PATH)
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    return df[COLS]


def save_log(df: pd.DataFrame) -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_PATH, index=False)


def closed_stats(log: pd.DataFrame) -> dict:
    exits = log.loc[log["event"] == "EXIT"].copy() if not log.empty else _empty_log()
    n = int(len(exits))
    if n == 0:
        return {
            "n": 0,
            "mean_r": None,
            "target_fills": TARGET_FILLS,
            "remaining": TARGET_FILLS,
            "gate": f"0/{TARGET_FILLS} — wait for RSI(2)<10 above SMA200, then log fills",
        }
    r = pd.to_numeric(exits["r"], errors="coerce")
    mean_r = float(r.mean())
    remaining = max(0, TARGET_FILLS - n)
    if n < TARGET_FILLS:
        gate = f"{n}/{TARGET_FILLS} closed, mean R={mean_r:.3f} vs ~{TARGET_MEAN_R:.2f}R — keep papering"
    elif mean_r >= TARGET_MEAN_R * 0.5:
        gate = f"{n} fills, mean R={mean_r:.3f} vs ~{TARGET_MEAN_R:.2f}R — still paper, not live"
    else:
        gate = f"{n} fills, mean R={mean_r:.3f} below ~{TARGET_MEAN_R:.2f}R — stop, do not go live"
    return {
        "n": n,
        "mean_r": mean_r,
        "win_rate": float((r > 0).mean()),
        "target_fills": TARGET_FILLS,
        "remaining": remaining,
        "gate": gate,
    }


def _next_id(log: pd.DataFrame) -> int:
    if log.empty:
        return 1
    return int(pd.to_numeric(log["event_id"], errors="coerce").max()) + 1


def log_entry(symbol: str, price: float, stop: float, reason: str = "rsi2_entry") -> dict:
    state = load_state()
    log = load_log()
    symbol = symbol.upper()
    if symbol in state.get("open", {}):
        raise ValueError(f"{symbol} already open")
    if state.get("halt_new_entries"):
        raise ValueError("halt_new_entries is on")
    if len(state.get("open", {})) >= MAX_OPEN_POSITIONS:
        raise ValueError("max open positions")
    sh = shares_for_risk(float(state["equity"]), RISK_PCT_PER_TRADE, price, stop)
    if sh <= 0:
        raise ValueError("size_zero")
    state.setdefault("open", {})[symbol] = {
        "shares": sh,
        "entry": float(price),
        "stop": float(stop),
        "risk_per_share": float(price - stop),
    }
    row = {
        "event_id": _next_id(log),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": "ENTRY",
        "symbol": symbol,
        "shares": sh,
        "price": round(price, 4),
        "stop": round(stop, 4),
        "reason": reason,
        "r": None,
        "pnl_usd": None,
        "equity_after": round(float(state["equity"]), 2),
    }
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    save_state(state)
    save_log(log)
    return row


def log_exit(symbol: str, price: float, reason: str) -> dict:
    state = load_state()
    log = load_log()
    symbol = symbol.upper()
    open_ = state.get("open", {})
    if symbol not in open_:
        raise ValueError(f"{symbol} not open in forward book")
    pos = open_.pop(symbol)
    shares = int(pos["shares"])
    entry = float(pos["entry"])
    risk = float(pos.get("risk_per_share") or (entry - float(pos["stop"])))
    pnl = (price - entry) * shares
    r = (price - entry) / risk if risk > 0 else 0.0
    state["equity"] = float(state["equity"]) + pnl
    state["peak_equity"] = max(float(state.get("peak_equity", STARTING_EQUITY)), state["equity"])
    peak = float(state["peak_equity"])
    dd = (peak - state["equity"]) / peak if peak else 0.0
    state["halt_new_entries"] = dd >= MAX_DRAWDOWN_PCT
    row = {
        "event_id": _next_id(log),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": "EXIT",
        "symbol": symbol,
        "shares": shares,
        "price": round(price, 4),
        "stop": round(float(pos["stop"]), 4),
        "reason": reason,
        "r": round(r, 4),
        "pnl_usd": round(pnl, 2),
        "equity_after": round(state["equity"], 2),
    }
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    save_state(state)
    save_log(log)
    return row


def append_research_journal(line: str) -> None:
    """One row per (date, id). Re-running the same day updates the line."""
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = "# Research journal — one line per test. Accept / reject with the reason. Do not retune after the fact.\n\n| Date | ID | What | Result | Why accept/reject |\n|---|---|---|---|---|\n"
    if not JOURNAL_PATH.exists():
        JOURNAL_PATH.write_text(header, encoding="utf-8")
    text = JOURNAL_PATH.read_text(encoding="utf-8")
    row = line.rstrip()
    cells = [p.strip() for p in row.strip().strip("|").split("|")]
    date = cells[0] if cells else ""
    hid = cells[1] if len(cells) > 1 else ""
    out: list[str] = []
    replaced = False
    for ln in text.splitlines():
        existing = [p.strip() for p in ln.strip().strip("|").split("|")]
        if ln.startswith("| 20") and len(existing) > 1 and existing[0] == date and existing[1] == hid:
            out.append(row)
            replaced = True
        else:
            out.append(ln)
    if not replaced:
        out.append(row)
    JOURNAL_PATH.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
