"""Continuous lab runner. No Cursor, no chat tokens. One instance at a time.

Every hour: sweep papers (every 6h), Groq turns paper / library leads into specs.
Once a day after TEST_HOUR: pull data, freeze up to 20 queued specs, test them
as one Bonferroni family (budget in config/settings.yaml lab block).

Groq-brainstormed ideas are NOT tested until a human approves them:
    python -m ats intake list --status raw
    python -m ats intake review INTAKE_xxx            (approve)
    python -m ats intake reject INTAKE_xxx --reason   (reject)

research/lab/STOP pauses the loop. It is written automatically when anything is
CANDIDATE or OOS_PASS. OOS unlock and the paper decision stay human.
"""

from __future__ import annotations

import msvcrt
import subprocess
import sys
import time
from datetime import datetime, timedelta

from ats.config import ROOT, load_settings
from ats.lab import loop

LAB = ROOT / "research" / "lab"
STOP = LAB / "STOP"
LOCK = LAB / "auto_lab.lock"
PULL_TIMEFRAMES = ("M5", "M15", "H1")
PULL_YEARS = 5
TEST_HOUR = 20
CYCLE_SECONDS = 3600
SWEEP_EVERY = timedelta(hours=6)
MAX_PENDING_BRAINSTORM = 20


def log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M} {msg}", flush=True)


def pull_best_effort() -> None:
    from ats.data.mt5_client import connect, copy_ohlc, save_raw, shutdown

    symbols = list(dict.fromkeys([*load_settings()["universe"]["symbols"], "XAUUSD"]))
    try:
        connect()
    except Exception as exc:
        log(f"pull skipped, using existing data (MT5 not available): {str(exc).splitlines()[0]}")
        return
    try:
        for tf in PULL_TIMEFRAMES:
            for sym in symbols:
                try:
                    save_raw(copy_ohlc(sym, tf, PULL_YEARS), sym, tf)
                except Exception as exc:
                    log(f"  {sym} {tf}: pull failed: {exc}")
    finally:
        shutdown()


def notify(text: str) -> None:
    try:
        subprocess.run(["msg", "*", text], check=False, timeout=15)
    except Exception:
        pass


def brainstorm_for_approval() -> None:
    """Top up the human approval list. These never reach a test unreviewed."""
    from ats.ideas.groq_brainstorm import run_groq_brainstorm_all
    from ats.ideas.intake_store import list_rows

    pending = [r for r in list_rows("raw") if str(r.get("source_ref") or "").startswith("groq_")]
    if len(pending) >= MAX_PENDING_BRAINSTORM:
        log(f"{len(pending)} brainstormed ideas await your approval; not adding more")
        return
    log("Groq brainstorm -> intake (status raw, needs approval)")
    run_groq_brainstorm_all(n=3)


def check_waiting() -> bool:
    waiting = loop.status()["awaiting_human"]
    if not waiting:
        return False
    ids = ", ".join(f"{hid} ({decision})" for hid, decision in waiting)
    text = (
        f"Proofbook auto lab paused {datetime.now():%Y-%m-%d %H:%M}: awaiting human -> {ids}\n"
        "This is not a paper candidate yet. Next:\n"
        "  python -m ats lab status\n"
        "  python -m ats lab unlock --id <ID> --confirm   (spends OOS once)\n"
        "  python -m ats lab decide --id <ID> --paper|--reject --note \"why\"\n"
        "Delete research/lab/STOP to resume.\n"
    )
    STOP.write_text(text, encoding="utf-8")
    log(text)
    notify(f"Proofbook: {ids} awaiting your decision. See research/lab/STOP")
    return True


def run_forever() -> None:
    last_sweep: datetime | None = None
    last_brainstorm_day = None
    while True:
        if STOP.exists():
            log("STOP present; paused. Delete research/lab/STOP to resume.")
            time.sleep(600)
            continue

        now = datetime.now()
        sweep = last_sweep is None or now - last_sweep >= SWEEP_EVERY
        log(f"cycle: sweep={sweep} propose=True")
        try:
            loop.run_cycle(sweep=sweep, propose=True, test=False)
            if sweep:
                last_sweep = now
            if last_brainstorm_day != now.date():
                brainstorm_for_approval()
                last_brainstorm_day = now.date()
            if now.hour >= TEST_HOUR:
                cfg = loop.lab_settings()
                if loop.budget_block(loop.book.load_runs(), int(cfg["min_days_between_batches"])) is None:
                    log("daily test window: pull data, then test one family")
                    pull_best_effort()
                    loop.run_cycle(sweep=False, propose=False, test=True)
        except Exception as exc:
            log(f"cycle failed: {type(exc).__name__}: {exc}")

        if check_waiting():
            continue
        time.sleep(CYCLE_SECONDS)


def main() -> int:
    LAB.mkdir(parents=True, exist_ok=True)
    lock = LOCK.open("a+")
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        log("another auto_lab is already running; exiting")
        return 0
    log(f"auto lab started (python {sys.version.split()[0]})")
    run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
