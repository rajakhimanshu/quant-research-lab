from __future__ import annotations

import argparse
import sys
from ats.config import DATA_DIR, load_settings
from ats.data.calendar import calendar_path
from ats.data.mt5_client import (
    _mt5,
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
        mt5 = _mt5()
        term = mt5.terminal_info()
        maxbars = getattr(term, "maxbars", None)
        print(f"Terminal path: {getattr(term, 'path', None)}")
        print(f"Terminal company: {getattr(term, 'company', None)}")
        print(f"Max bars in chart: {maxbars}")
        if maxbars is not None and int(maxbars) <= 100000:
            print(
                "WARNING: MT5 is capped at 100000 bars (Tools > Options > Charts > "
                "Max bars in chart). Set Unlimited, restart MT5, open the symbol "
                "chart, scroll left, then python -m ats pull. History is not unlimited."
            )
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

    ids = args.id if args.id else None
    run_hypotheses(ids, unlock_oos=args.unlock_oos)
    return 0


def cmd_ledger(_: argparse.Namespace) -> int:
    from ats.research.ledger import print_ledger

    print_ledger()
    return 0


def cmd_audit_h86(_: argparse.Namespace) -> int:
    from ats.research.h86_audit import print_audit, run_audit

    print_audit(run_audit())
    return 0


def cmd_deep_h86(_: argparse.Namespace) -> int:
    from ats.research.h86_deep import print_deep, run_deep

    print_deep(run_deep())
    return 0


def cmd_audit_h5(_: argparse.Namespace) -> int:
    from ats.research.h5_audit import print_audit, run_audit

    print_audit(run_audit())
    return 0


def cmd_paper_h86(_: argparse.Namespace) -> int:
    from ats.paper.h86_run import print_paper, run_paper

    print_paper(run_paper())
    return 0


def cmd_paper_h5(args: argparse.Namespace) -> int:
    from ats.paper.forward import log_entry, log_exit
    from ats.paper.h5_run import print_paper, run_paper

    if args.fill_entry:
        sym, price, stop = args.fill_entry[0], float(args.fill_entry[1]), float(args.fill_entry[2])
        row = log_entry(sym, price, stop)
        print(f"FORWARD ENTRY {row['symbol']} {row['shares']} sh @ {row['price']} stop={row['stop']}")
    if args.fill_exit:
        sym, price, reason = args.fill_exit[0], float(args.fill_exit[1]), args.fill_exit[2]
        row = log_exit(sym, price, reason)
        print(f"FORWARD EXIT {row['symbol']} @ {row['price']} R={row['r']} pnl=${row['pnl_usd']}")
    if args.fill_entry or args.fill_exit:
        print_paper(run_paper(refresh=False))
        return 0
    print_paper(run_paper(refresh=not args.no_refresh))
    return 0


DEFAULT_ARXIV = (
    'all:forex OR all:"foreign exchange" OR all:"currency carry" '
    'OR all:"FX momentum" OR all:"currency mean reversion"'
)


def cmd_ideas(args: argparse.Namespace) -> int:
    cmd = args.ideas_cmd or "list"
    if cmd == "list":
        from ats.ideas.inbox import print_inbox

        print_inbox()
        return 0
    if cmd == "arxiv":
        from ats.ideas.arxiv import print_arxiv, save_arxiv_leads, search_arxiv

        query = args.query or DEFAULT_ARXIV
        rows = search_arxiv(query, max_results=args.max_results)
        print_arxiv(rows, query)
        if args.save:
            n = save_arxiv_leads(rows)
            print(f"Saved {n} leads into config/ideas.yaml (still not hypotheses).")
        return 0
    if cmd == "blogs":
        from ats.ideas.blogs import print_blogs

        print_blogs()
        return 0
    if cmd == "cot":
        from datetime import datetime, timezone

        from ats.ideas.cot import gold_series, pull_cot

        years = None
        if args.years:
            now_y = datetime.now(timezone.utc).year
            years = list(range(now_y - args.years + 1, now_y + 1))
        cot = pull_cot(years)
        gold = gold_series(cot)
        print(f"COT markets: {cot['market'].nunique()}  rows={len(cot)}")
        if gold.empty:
            print("No COMEX gold rows parsed. Inspect data/cot/combined.csv")
            return 1
        print(f"Gold COMEX: {len(gold)} weeks  {gold['asof'].min().date()} .. {gold['asof'].max().date()}")
        tail = gold.tail(4)
        for _, row in tail.iterrows():
            print(
                f"  {row['asof'].date()} oi={row['oi']:.0f} nc_net={row['nc_net']:.0f} "
                f"nc_net/oi={float(row['nc_net_oi']):.3f}"
            )
        print("Positioning is a lead. Frozen test: python -m ats test --id H9_cot_spec_fade")
        print("OOS stays locked. This command does not find an edge.")
        return 0
    if cmd == "cross":
        from ats.ideas.cross_market import snapshot

        out = snapshot()
        for key, val in out.items():
            print(f"{key}: {val}")
        return 0
    if cmd == "sentiment":
        from pathlib import Path

        from ats.ideas.sentiment import print_sentiment

        path = Path(args.csv) if args.csv else None
        print_sentiment(path)
        return 0
    raise SystemExit(f"Unknown ideas command {cmd}")


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
    t.add_argument(
        "--id",
        action="append",
        help="Hypothesis id (repeatable). A family of ids uses Bonferroni n_tests.",
    )
    t.add_argument(
        "--unlock-oos",
        action="store_true",
        help="Peek at locked out-of-sample. Use only after choosing a survivor.",
    )
    t.set_defaults(func=cmd_test)

    lg = sub.add_parser("ledger", help="Print the closed book (rejects, why, do-not-retune)")
    lg.set_defaults(func=cmd_ledger)

    ah = sub.add_parser(
        "audit-h5",
        help="Frozen H5 equity battery: OOS, per-ticker, walk-forward, cost stress, max-4 book",
    )
    ah.set_defaults(func=cmd_audit_h5)

    a86 = sub.add_parser(
        "audit-h86",
        help="H86 frozen battery: splits, walk-forward, cost, delay, invert, bootstrap",
    )
    a86.set_defaults(func=cmd_audit_h86)

    d86 = sub.add_parser(
        "deep-h86",
        help="H86 last-level: bar spread, M5 path, LOO, weekend cluster. Not an EA.",
    )
    d86.set_defaults(func=cmd_deep_h86)

    ph = sub.add_parser(
        "paper-h5",
        help="H5 paper book on OOS with frozen 0.5% risk / max-4 / 12% DD halt, plus current scan",
    )
    ph.add_argument("--no-refresh", action="store_true", help="Use cached ETF CSVs only")
    ph.add_argument(
        "--fill-entry",
        nargs=3,
        metavar=("SYMBOL", "PRICE", "STOP"),
        help="Log a next-open paper BUY into the forward book (not the OOS replay)",
    )
    ph.add_argument(
        "--fill-exit",
        nargs=3,
        metavar=("SYMBOL", "PRICE", "REASON"),
        help="Log a next-open paper SELL (reason: sma5|rsi_exit|time_stop|hard_stop)",
    )
    ph.set_defaults(func=cmd_paper_h5)

    p86 = sub.add_parser(
        "paper-h86",
        help="H86 weekend-gap paper scan (GBP/JPY/AUD). Not live. Not an EA.",
    )
    p86.set_defaults(func=cmd_paper_h86)

    ideas = sub.add_parser(
        "ideas",
        help="Idea intake (papers, COT, dislocations, sentiment). Not an edge finder.",
    )
    ideas.set_defaults(func=cmd_ideas, ideas_cmd="list")
    isub = ideas.add_subparsers(dest="ideas_cmd")

    isub.add_parser("list", help="Show the idea inbox")

    arx = isub.add_parser("arxiv", help="Search arXiv q-fin for paper leads")
    arx.add_argument("--query", help="arXiv search_query (default: FX/currency q-fin)")
    arx.add_argument("--max-results", type=int, default=8)
    arx.add_argument("--save", action="store_true", help="Append paper leads to config/ideas.yaml")

    isub.add_parser("blogs", help="Methodology blogs (not signals)")

    cotp = isub.add_parser("cot", help="Pull official CFTC Commitment of Traders files")
    cotp.add_argument("--years", type=int, help="How many calendar years back, including this year")

    isub.add_parser("cross", help="Gold–DXY (and yields–JPY) correlation snapshot")

    sent = isub.add_parser("sentiment", help="Read a dropped retail-positioning CSV")
    sent.add_argument("--csv", help="Path to CSV (default: data/sentiment/*.csv)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raise SystemExit(args.func(args))
