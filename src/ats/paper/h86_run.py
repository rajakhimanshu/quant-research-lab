"""Run H86 paper scan. OOS already unlocked. Not live. Not an EA."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from ats.config import DATA_DIR, load_settings
from ats.data.mt5_client import load_raw
from ats.features.prepare import prepare_frame
from ats.paper.h86_book import (
    H86_ID,
    MIN_LOT,
    STARTING_EQUITY,
    h86_params,
    oos_treatment,
    replay_oos_equity,
    scan_book,
)

PAPER = DATA_DIR / "paper"


def load_h86_frames(settings: dict | None = None) -> dict:
    settings = settings or load_settings()
    params = h86_params()
    tf = params.get("timeframe", "H1")
    frames = {}
    for symbol in params["symbols"]:
        frames[symbol] = prepare_frame(load_raw(symbol, tf), settings)
    return frames


def run_paper() -> dict:
    settings = load_settings()
    frames = load_h86_frames(settings)
    oos = oos_treatment(frames, settings)
    book = replay_oos_equity(oos)
    scan = scan_book(frames, settings)
    PAPER.mkdir(parents=True, exist_ok=True)
    if not oos.empty:
        oos.to_csv(PAPER / "h86_oos_trades.csv", index=False)
    scan.to_csv(PAPER / "h86_signals.csv", index=False)
    payload = {
        "hypothesis": H86_ID,
        "verdict": "PAPER_CANDIDATE",
        "not_live": True,
        "not_ea": True,
        "demo_lot": MIN_LOT,
        "oos_n": book["n"],
        "oos_mean_r": book["mean_r"],
        "oos_total_r": book["total_r"],
        "paper_start_usd": book["starting"],
        "paper_end_usd": book["ending"],
        "paper_return_pct": book["return_pct"],
        "paper_max_dd_pct": book["max_dd_pct"],
        "scan": scan.to_dict(orient="records"),
        "asof_utc": datetime.now(timezone.utc).isoformat(),
        "rule": "Fade GBPUSD/USDJPY/AUDUSD Sunday reopen if gap >=36h and >=0.15 ATR. 1:1 ATR, 8 H1.",
    }
    (PAPER / "h86_paper.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def print_paper(report: dict) -> None:
    print("H86 paper — weekend-gap fade. NOT live. NOT an EA.")
    print("Exness demo: 0.01 lot when WAIT_HALT becomes ENTER. $100-200 cannot 1%-risk a 15-pip stop.")
    print(
        f"OOS replay (hypothetical ${report['paper_start_usd']:.0f} at 1%/R, not the demo lot): "
        f"n={report['oos_n']} mean R={report['oos_mean_r']:.3f} "
        f"end ${report['paper_end_usd']:.0f} ({report['paper_return_pct']:+.1f}%) "
        f"max DD {report['paper_max_dd_pct']:.1f}%"
    )
    print("Scan:")
    for row in report["scan"]:
        side = f" {row['side']}" if row.get("side") else ""
        print(f"  {row['action']:10} {row['symbol']}{side}  {row['note']}")
    print("Do not add NZD/CAD/CHF. Do not loosen 36h. Do not port to MQL5 until this paper book has fills.")
