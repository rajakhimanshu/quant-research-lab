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


def cmd_scoreboard(_: argparse.Namespace) -> int:
    from ats.research.scoreboard import write_scoreboard

    print(f"Wrote {write_scoreboard()}")
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


# ── Intake pipeline commands ─────────────────────────────────────────────────

def cmd_intake(args: argparse.Namespace) -> int:  # noqa: C901
    """Dispatcher for `python -m ats intake <subcommand>`."""
    cmd = args.intake_cmd or "list"

    # ── list ──────────────────────────────────────────────────────────────────
    if cmd == "list":
        from ats.ideas.intake_store import list_rows, print_intake
        rows = list_rows(status_filter=args.status or None)
        total = list_rows()  # unfiltered count for context
        label = f"(filtered: status={args.status})" if args.status else ""
        print(f"Intake backlog: {len(rows)} records {label}  |  total in file: {len(total)}")
        print_intake(rows)
        return 0

    # ── review ────────────────────────────────────────────────────────────────
    if cmd == "review":
        from datetime import date
        from ats.ideas.intake_store import update_status
        extra: dict = {"reviewed": str(date.today())}
        if args.tier:
            extra["timeframe_tier"] = args.tier
        if args.category:
            extra["causal_category"] = args.category
        if args.causal_actor:
            extra["causal_actor"] = args.causal_actor
        if args.trades:
            extra["expected_trades_month"] = args.trades
        if args.bars:
            extra["expected_holding_bars"] = args.bars
        ok = update_status(args.intake_id, "reviewed", **extra)
        if ok:
            print(f"[OK] {args.intake_id} -> status=reviewed")
            for k, v in extra.items():
                print(f"     {k}: {v}")
            print(f"\nNext: python -m ats intake promote {args.intake_id}")
        else:
            print(f"[ERROR] {args.intake_id} not found in config/intake.yaml")
        return 0 if ok else 1

    # ── promote ───────────────────────────────────────────────────────────────
    if cmd == "promote":
        from ats.ideas.promoter import promote
        promote(
            args.intake_id,
            dry_run=not args.confirm,
            override_dedup=args.override_dedup,
        )
        return 0

    # ── reject ────────────────────────────────────────────────────────────────
    if cmd == "reject":
        from ats.ideas.intake_store import update_status
        ok = update_status(
            args.intake_id,
            "rejected_at_intake",
            rejection_reason=args.reason or "",
        )
        if ok:
            print(f"[OK] {args.intake_id} -> status=rejected_at_intake")
            print(f"     reason: {args.reason or '(none given)'}")
        else:
            print(f"[ERROR] {args.intake_id} not found in config/intake.yaml")
        return 0 if ok else 1

    # ── paper-monitor ─────────────────────────────────────────────────────────
    if cmd == "paper-monitor":
        from ats.ideas.paper_monitor import run_paper_monitor
        run_paper_monitor(
            max_per_query=args.max_per_query or 6,
            save=args.save,
            ssrn_url=args.ssrn or None,
        )
        return 0

    # ── groq-brainstorm ───────────────────────────────────────────────────────
    if cmd == "groq-brainstorm":
        from ats.ideas.groq_brainstorm import run_groq_brainstorm, run_groq_brainstorm_all
        n = args.n or 5
        if args.all:
            run_groq_brainstorm_all(n=n)
        else:
            if not args.category:
                print("[ERROR] Provide --category forced_flow|inventory_pressure|structural_liquidity")
                print("        or use --all to run all three categories.")
                return 1
            run_groq_brainstorm(args.category, n=n)
        return 0

    # ── brainstorm-prompt (copy-paste fallback, no API key needed) ────────────
    if cmd == "brainstorm-prompt":
        from ats.ideas.prompts import CATEGORIES, _PROMPTS
        cat = args.category
        n = args.n or 5
        if cat not in CATEGORIES:
            print(f"[ERROR] Unknown category: {cat!r}. Choose from: {CATEGORIES}")
            return 1
        prompt = _PROMPTS[cat].format(n=n)
        print("=" * 80)
        print(f"BRAINSTORM PROMPT — category={cat}, n={n}")
        print("Copy the block below into any LLM, then import the JSON response with:")
        print("  python -m ats intake brainstorm-import --file response.json")
        print("=" * 80)
        print(prompt)
        return 0

    # ── brainstorm-import (JSON response from any LLM) ────────────────────────
    if cmd == "brainstorm-import":
        import json
        from pathlib import Path
        from ats.ideas.dedup import check_dedup
        from ats.ideas.intake_store import add_idea
        json_path = Path(args.file)
        if not json_path.exists():
            print(f"[ERROR] File not found: {json_path}")
            return 1
        with open(json_path, encoding="utf-8") as f:
            ideas = json.load(f)
        if not isinstance(ideas, list):
            print("[ERROR] Expected a JSON array of idea objects.")
            return 1
        saved = discarded = 0
        for idea in ideas:
            actor = (idea.get("causal_actor") or "").strip()
            if not actor:
                print(f"  [DISCARD] Missing causal_actor: {idea.get('name', '?')[:60]}")
                discarded += 1
                continue
            dedup = check_dedup(actor, idea.get("name", ""))
            iid = add_idea({
                "source": "llm_brainstorm",
                "source_ref": str(json_path.name),
                "name": (idea.get("name") or "")[:80],
                "causal_category": idea.get("causal_category", ""),
                "causal_actor": actor,
                "timeframe_tier": "",
                "instrument": idea.get("instrument", ""),
                "timeframe": idea.get("timeframe", ""),
                "expected_holding_bars": idea.get("expected_holding_bars"),
                "expected_trades_month": idea.get("expected_trades_month"),
                "dedup_check": dedup,
                "notes": (idea.get("notes") or "")[:300],
                "status": "raw",
            })
            print(f"  Saved {iid}: {idea.get('name', '')[:60]}")
            saved += 1
        print(f"\nImported: {saved}  Discarded (no causal_actor): {discarded}")
        return 0

    # ── log-trade (K-A04) ─────────────────────────────────────────────────────
    if cmd == "log-trade":
        from ats.ideas.ka04_log import log_trade
        log_trade(
            instrument=args.instrument,
            direction=args.direction,
            timeframe=args.timeframe,
            reason=args.reason,
            causal_actor=args.causal_actor,
            outcome=args.outcome or "",
        )
        return 0

    # ── library ───────────────────────────────────────────────────────────────
    if cmd == "library":
        from ats.ideas.mechanism_library import print_library
        print_library(tier_filter=args.tier or None)
        return 0

    # ── dedup (standalone check) ──────────────────────────────────────────────
    if cmd == "dedup":
        from ats.ideas.dedup import print_dedup_report
        print_dedup_report(args.causal_actor, args.name or "")
        return 0

    raise SystemExit(f"Unknown intake subcommand: {cmd!r}")


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

    sb = sub.add_parser(
        "scoreboard",
        help="Rebuild research/HYPOTHESIS_SCOREBOARD.md from the ledger",
    )
    sb.set_defaults(func=cmd_scoreboard)

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
    # ── intake subcommand group ───────────────────────────────────────────────
    intake = sub.add_parser(
        "intake",
        help="Hypothesis intake pipeline — upstream of the lab. Generates, screens, and queues ideas.",
    )
    intake.set_defaults(func=cmd_intake, intake_cmd="list")
    isub2 = intake.add_subparsers(dest="intake_cmd")

    # list
    il = isub2.add_parser("list", help="Show the intake backlog")
    il.add_argument(
        "--status",
        choices=["raw", "reviewed", "promoted", "rejected_at_intake"],
        help="Filter by status",
    )

    # review
    irv = isub2.add_parser("review", help="Mark an intake record as reviewed and fill fields")
    irv.add_argument("intake_id", help="INTAKE_xxx id")
    irv.add_argument("--tier", choices=["retail_feasible", "infra_gated"], help="Assign timeframe tier")
    irv.add_argument(
        "--category",
        choices=["forced_flow", "inventory_pressure", "structural_liquidity"],
        help="Assign causal category",
    )
    irv.add_argument("--causal-actor", dest="causal_actor", help="Override/set causal_actor text")
    irv.add_argument("--trades", type=int, dest="trades", help="Set expected_trades_month")
    irv.add_argument("--bars", type=int, dest="bars", help="Set expected_holding_bars")

    # promote
    ipr = isub2.add_parser("promote", help="Run promotion gate + print hypotheses.yaml block")
    ipr.add_argument("intake_id", help="INTAKE_xxx id")
    ipr.add_argument("--confirm", action="store_true", help="Mark as promoted in intake.yaml")
    ipr.add_argument(
        "--override-dedup",
        action="store_true",
        dest="override_dedup",
        help="Skip dedup gate (use after manual review confirms genuine novelty)",
    )

    # reject
    irj = isub2.add_parser("reject", help="Mark as rejected at intake (never reaches the lab)")
    irj.add_argument("intake_id", help="INTAKE_xxx id")
    irj.add_argument("--reason", required=True, help="Why this idea is rejected at intake")

    # paper-monitor
    ipm = isub2.add_parser("paper-monitor", help="Weekly arXiv multi-query M5–M30 paper run")
    ipm.add_argument("--save", action="store_true", help="Save results to config/intake.yaml")
    ipm.add_argument("--max-per-query", type=int, default=6, dest="max_per_query")
    ipm.add_argument("--ssrn", help="Manually drop a single SSRN paper URL instead of running the full sweep")

    # groq-brainstorm
    ibr = isub2.add_parser("groq-brainstorm", help="Fully automated Groq LLM brainstorm (requires GROQ_API_KEY)")
    ibr.add_argument(
        "--category",
        choices=["forced_flow", "inventory_pressure", "structural_liquidity"],
        help="Single category to brainstorm",
    )
    ibr.add_argument("--all", action="store_true", help="Run all three categories")
    ibr.add_argument("--n", type=int, default=5, help="Number of ideas to request per category")

    # brainstorm-prompt (copy-paste fallback, no API key needed)
    ibp = isub2.add_parser("brainstorm-prompt", help="Print a brainstorm prompt for manual copy-paste into any LLM")
    ibp.add_argument(
        "category",
        choices=["forced_flow", "inventory_pressure", "structural_liquidity"],
    )
    ibp.add_argument("--n", type=int, default=5)

    # brainstorm-import (import JSON response from any LLM)
    ibi = isub2.add_parser("brainstorm-import", help="Import LLM response JSON file into intake backlog")
    ibi.add_argument("--file", required=True, help="Path to JSON response file")

    # log-trade (K-A04)
    ilt = isub2.add_parser("log-trade", help="K-A04: log a discretionary trade to extract its causal mechanism")
    ilt.add_argument("--instrument", required=True, help="e.g. XAUUSD")
    ilt.add_argument("--direction", required=True, choices=["long", "short"])
    ilt.add_argument("--timeframe", required=True, help="e.g. M5")
    ilt.add_argument("--reason", required=True, help="Setup description")
    ilt.add_argument(
        "--causal-actor",
        required=True,
        dest="causal_actor",
        help="One sentence: who pays you and why",
    )
    ilt.add_argument("--outcome", help="e.g. 'win +1.2R'")

    # library
    ilib = isub2.add_parser("library", help="View the static mechanism library")
    ilib.add_argument("--tier", choices=["retail_feasible", "infra_gated"])

    # dedup (standalone check)
    idd = isub2.add_parser("dedup", help="Check a causal_actor string against the ledger for near-duplicates")
    idd.add_argument("causal_actor", help="The causal_actor text to check")
    idd.add_argument("--name", default="", help="Optional mechanism name for extra signal")

    # ── automated research lab ────────────────────────────────────────────────
    lab = sub.add_parser("lab", help="Automated research loop: leads -> template specs -> family test -> book")
    lab.set_defaults(func=cmd_lab, lab_cmd="status")
    lsub = lab.add_subparsers(dest="lab_cmd")
    lsub.add_parser("status", help="Queue, open candidates, batches run")
    lsub.add_parser("templates", help="Print the template catalogue (what the lab can test)")
    lrun = lsub.add_parser("run", help="One research cycle (stops at CANDIDATE)")
    lrun.add_argument("--no-sweep", action="store_true", help="Skip paper/feed sweep")
    lrun.add_argument("--no-propose", action="store_true", help="Skip LLM proposals")
    lrun.add_argument("--no-test", action="store_true", help="Stop before freezing/testing")
    lrun.add_argument("--max-batch", type=int, help="Override lab.max_batch")
    lrun.add_argument("--force", action="store_true", help="Ignore the batch-frequency budget")
    lrun.add_argument("--dry-run", action="store_true", help="Show what would be frozen; write nothing")
    lpr = lsub.add_parser("propose", help="Queue a spec by hand (YAML) or ask the LLM about one intake lead")
    lpr.add_argument("--file", help="YAML file with one spec or a list of specs")
    lpr.add_argument("--lead", help="INTAKE_xxx or MECH_xxx id to translate with the LLM")
    lun = lsub.add_parser("unlock", help="Human gate: spend the OOS segment of a CANDIDATE (once)")
    lun.add_argument("--id", required=True)
    lun.add_argument("--confirm", action="store_true", help="Required: OOS can only be spent once")
    lde = lsub.add_parser("decide", help="Human gate: PAPER_CANDIDATE (after OOS_PASS) or REJECT")
    lde.add_argument("--id", required=True)
    g = lde.add_mutually_exclusive_group(required=True)
    g.add_argument("--paper", action="store_true")
    g.add_argument("--reject", action="store_true")
    lde.add_argument("--note", required=True, help="Why you decided")

    return p


def cmd_lab(args: argparse.Namespace) -> int:
    import json

    import yaml

    from ats.lab import loop

    cmd = args.lab_cmd or "status"
    if cmd == "status":
        loop.print_status()
        return 0
    if cmd == "templates":
        from ats.lab.templates import catalogue

        print(json.dumps(catalogue(), indent=2))
        return 0
    if cmd == "run":
        loop.run_cycle(sweep=not args.no_sweep, propose=not args.no_propose, test=not args.no_test,
                       force=args.force, dry_run=args.dry_run, max_batch=args.max_batch)
        return 0
    if cmd == "propose":
        from ats.lab.proposer import load_queue, propose_from_lead, record_spec, save_queue

        rows = load_queue()
        if args.file:
            with open(args.file, encoding="utf-8") as f:
                blob = yaml.safe_load(f)
            specs = blob if isinstance(blob, list) else [blob]
            for s in specs:
                rec = record_spec(s, source=str(s.get("source") or args.file), rows=rows)
                print(f"{rec['id']} {rec['status']}: {'; '.join(rec.get('errors') or []) or s.get('name')}")
        elif args.lead:
            lead = None
            if args.lead.startswith("INTAKE_"):
                from ats.ideas.intake_store import get_idea

                lead = get_idea(args.lead)
            else:
                from ats.ideas.mechanism_library import load_library

                lead = next((m for m in load_library() if m.get("id") == args.lead), None)
            if lead is None:
                raise SystemExit(f"{args.lead} not found")
            rec = propose_from_lead({**lead, "source_ref": lead.get("source_ref") or args.lead}, rows=rows)
            print(f"{rec['id']} {rec['status']}: {'; '.join(rec.get('errors') or []) or rec['spec']['name']}")
        else:
            raise SystemExit("Give --file or --lead")
        save_queue(rows)
        return 0
    if cmd == "unlock":
        if not args.confirm:
            raise SystemExit("OOS can be spent only once. Re-run with --confirm.")
        print(f"{args.id}: {loop.unlock(args.id)}")
        return 0
    if cmd == "decide":
        print(f"{args.id}: {loop.decide(args.id, paper=args.paper, note=args.note)}")
        return 0
    raise SystemExit(f"Unknown lab command {cmd}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raise SystemExit(args.func(args))
