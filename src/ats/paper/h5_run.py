"""Run the frozen H5 paper book on OOS and emit the current scan."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from ats.config import DATA_DIR, load_hypotheses
from ats.hypotheses.equity_rsi2 import load_equity_frames
from ats.paper.forward import TARGET_FILLS, TARGET_MEAN_R, append_research_journal, closed_stats, load_log, load_state
from ats.paper.h5_book import (
    MAX_DRAWDOWN_PCT,
    MAX_OPEN_POSITIONS,
    RISK_PCT_PER_TRADE,
    STARTING_EQUITY,
    replay_paper_book,
    scan_signals,
)

PAPER = DATA_DIR / "paper"
# Frozen from the published H5 audit. Do not recompute 60/20/20 after new bars.
FROZEN_OOS_START = pd.Timestamp("2023-03-17", tz="UTC")


def _params() -> dict:
    return dict(next(h for h in load_hypotheses() if h["id"] == "H5_equity_rsi2")["params"])


def oos_start() -> pd.Timestamp:
    return FROZEN_OOS_START


def run_paper(refresh: bool = True) -> dict:
    params = _params()
    try:
        frames = load_equity_frames(params, refresh=refresh)
    except Exception:
        frames = load_equity_frames(params, refresh=False)
    start = oos_start()
    book = replay_paper_book(frames, params, start=start)
    scan = scan_signals(frames, params, open_symbols=set(book.open_positions))
    PAPER.mkdir(parents=True, exist_ok=True)
    if not book.trades.empty:
        book.trades.to_csv(PAPER / "h5_oos_trades.csv", index=False)
    scan.to_csv(PAPER / "h5_signals.csv", index=False)
    fwd_state = load_state()
    fwd_open = set(fwd_state.get("open", {}))
    fwd_scan = scan_signals(frames, params, open_symbols=fwd_open)
    fwd_scan.to_csv(PAPER / "h5_forward_signals.csv", index=False)
    fwd = closed_stats(load_log())
    asof = str(fwd_scan["asof"].iloc[0]) if not fwd_scan.empty else ""
    actions = ", ".join(
        f"{r.action} {r.symbol}" for r in fwd_scan.itertuples() if r.action in ("ENTER", "EXIT")
    ) or "no ENTER/EXIT"
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    append_research_journal(
        f"| {day} | H5 paper | Forward scan asof {asof}: {actions}. {fwd['gate']} | "
        f"{fwd['n']}/{TARGET_FILLS} fills | Not live. Target ~{TARGET_MEAN_R:.2f}R. |"
    )
    payload = {
        "hypothesis": "H5_equity_rsi2",
        "verdict": "PAPER_CANDIDATE",
        "not_live": True,
        "paper_start": book.start,
        "starting_equity": book.starting_equity,
        "ending_equity": round(book.ending_equity, 2),
        "return_pct": round(100 * (book.ending_equity / book.starting_equity - 1), 2),
        "peak_equity": round(book.peak_equity, 2),
        "max_dd_pct": round(100 * book.max_dd_pct, 2),
        "halted_now": book.halted,
        "n_closed": book.n_trades,
        "win_rate": book.win_rate,
        "mean_r": book.mean_r,
        "total_r": book.total_r,
        "total_pnl": round(book.total_pnl, 2),
        "open": book.open_positions,
        "rules": {
            "risk_pct": RISK_PCT_PER_TRADE,
            "max_open": MAX_OPEN_POSITIONS,
            "dd_halt": MAX_DRAWDOWN_PCT,
            "starting_equity": STARTING_EQUITY,
        },
        "scan": scan.to_dict(orient="records"),
        "forward": {
            **fwd,
            "open": fwd_state.get("open", {}),
            "equity": fwd_state.get("equity"),
            "scan": fwd_scan.to_dict(orient="records"),
        },
        "asof_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = PAPER / "h5_paper_final.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    payload["path"] = str(path)
    payload["_scan_df"] = scan
    payload["_fwd_scan"] = fwd_scan
    payload["_book"] = book
    return payload


def print_paper(report: dict) -> None:
    print("=" * 64)
    print("H5 paper book — OOS window, frozen live-config risk")
    print(f"Start {report['paper_start']}")
    print(
        f"  ${report['starting_equity']:,.0f} -> ${report['ending_equity']:,.2f}  "
        f"({report['return_pct']:+.1f}%)  maxDD {report['max_dd_pct']:.1f}%"
    )
    print(
        f"  Closed {report['n_closed']}  WR={None if report['win_rate'] is None else round(100*report['win_rate'],1)}%  "
        f"meanR={report['mean_r']}  halt={report['halted_now']}"
    )
    print(f"  Open now: {list(report['open']) or 'none'}")
    print("OOS-sim scan (research replay, not your forward book):")
    scan = report["_scan_df"]
    for _, r in scan.iterrows():
        extra = f" rsi2={r['rsi2']:.1f}" if pd.notna(r["rsi2"]) else ""
        print(f"  {r['action']:10} {r['symbol']:4}  {r['asof']}  {r['close']:.2f}{extra}  {r['reason']}")
    fwd = report.get("forward") or {}
    print(f"FORWARD paper clock: {fwd.get('gate')}")
    print(f"  Equity ${float(fwd.get('equity') or STARTING_EQUITY):,.2f}  open={list((fwd.get('open') or {})) or 'none'}")
    print("Forward scan (paper-trade at NEXT open — not live):")
    for _, r in report["_fwd_scan"].iterrows():
        extra = f" rsi2={r['rsi2']:.1f}" if pd.notna(r["rsi2"]) else ""
        print(f"  {r['action']:10} {r['symbol']:4}  {r['asof']}  {r['close']:.2f}{extra}  {r['reason']}")
        if r["action"] == "ENTER" and pd.notna(r["atr"]):
            stop = float(r["close"]) - 2.0 * float(r["atr"])
            print(f"             fill: python -m ats paper-h5 --fill-entry {r['symbol']} <OPEN> {stop:.2f}")
        if r["action"] == "EXIT":
            print(f"             fill: python -m ats paper-h5 --fill-exit {r['symbol']} <OPEN> {r['reason']}")
    print("VERDICT: PAPER_CANDIDATE — not live. 30 forward fills vs ~0.10R before any live size.")
    print(f"Wrote {report['path']}")
    print("=" * 64)
