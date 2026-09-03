from __future__ import annotations

import argparse
import sys
from ats.config import DATA_DIR, load_settings
from ats.data.calendar import calendar_path
from ats.data.mt5_client import (
    account_snapshot,
    connect,
    copy_ohlc,
    discover_terminals,
    save_raw,
    shutdown,
    terminal_path,
)


def cmd_doctor(_: argparse.Namespace) -> int:
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    if sys.version_info[:2] != (3, 12):
        print("WARNING: MetaTrader5 wheels target Python 3.12. Recreate venv with: py -3.12 -m venv .venv")
    print(f"Configured terminal: {terminal_path() or '(auto-detect)'}")
    print(f"Detected installs: {discover_terminals()}")
    cal = calendar_path()
    print(f"Calendar file: {cal} exists={cal.exists()}")
    try:
        connect()
    except Exception as exc:
        print(f"MT5 connect FAILED: {exc}")
        return 1
    try:
        snap = account_snapshot()
        print("MT5 account:", snap)
        settings = load_settings()
        from ats.data.mt5_client import resolve_symbol

        print("Resolved symbols:")
        for sym in settings["universe"]["symbols"]:
            try:
                print(f"  {sym} -> {resolve_symbol(sym)}")
            except Exception as exc:
                print(f"  {sym} -> ERROR {exc}")
    finally:
        shutdown()
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    settings = load_settings()
    symbols = args.symbols or settings["universe"]["symbols"]
    timeframe = args.timeframe or settings["universe"]["timeframe"]
    years = args.years or settings["universe"]["lookback_years"]
    connect()
    try:
        for symbol in symbols:
            df = copy_ohlc(symbol, timeframe, int(years))
            path = save_raw(df, symbol, timeframe)
            print(f"{symbol} {timeframe}: {len(df)} bars -> {path}")
    finally:
        shutdown()
    return 0


def cmd_calendar(args: argparse.Namespace) -> int:
    from ats.data.calendar import import_public_calendar, load_calendar

    path = import_public_calendar(years=args.years or 4)
    df = load_calendar(path)
    print(f"High-impact events: {len(df)} -> {path}")
    if not df.empty:
        print(f"Range: {df['datetime_utc'].min()} .. {df['datetime_utc'].max()}")
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    from ats.research.pipeline import run_hypotheses

    ids = [args.id] if args.id else None
    run_hypotheses(ids, unlock_oos=args.unlock_oos)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ats",
        description="Hypothesis research lab. You propose. This tests. It does not invent edges.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="Check Python 3.12 + MT5 connection")
    d.set_defaults(func=cmd_doctor)

    pull = sub.add_parser("pull", help="Download OHLC from MT5 into data/raw")
    pull.add_argument("--symbols", nargs="*")
    pull.add_argument("--timeframe")
    pull.add_argument("--years", type=int)
    pull.set_defaults(func=cmd_pull)

    cal = sub.add_parser("calendar", help="Download public high-impact news calendar")
    cal.add_argument("--years", type=int, default=4)
    cal.set_defaults(func=cmd_calendar)

    t = sub.add_parser("test", help="Run enabled hypotheses on pulled data")
    t.add_argument("--id", help="Single hypothesis id, e.g. H1_event_reversal")
    t.add_argument(
        "--unlock-oos",
        action="store_true",
        help="Peek at locked out-of-sample. Use only after choosing a survivor.",
    )
    t.set_defaults(func=cmd_test)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raise SystemExit(args.func(args))
