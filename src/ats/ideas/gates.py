"""Automatic gates for LLM-brainstormed ideas, applied before a human sees them.

fact_check:   the idea must name one event in config/event_calendar.yaml that covers its
              instrument. The clock time comes from the calendar, never from the LLM.
novelty_check: the event + currency must not already be a tested family in the ledger
              (research/ledger.yaml + lab book) or the do-not-reopen list.

Both return a rejection reason, or None. False positives are preferred over letting a
retune through: the rejected idea stays in intake.yaml with its reason and can be revived.
"""

from __future__ import annotations

from functools import lru_cache

import yaml

from ats.config import CONFIG_DIR

CALENDAR_PATH = CONFIG_DIR / "event_calendar.yaml"

ALIASES = {
    "EURUSD": ("eurusd", "eur"), "GBPUSD": ("gbpusd", "gbp", "cable"), "USDJPY": ("usdjpy", "jpy", "yen"),
    "AUDUSD": ("audusd", "aud"), "USDCAD": ("usdcad", "cad"), "USDCHF": ("usdchf", "chf", "franc"),
    "NZDUSD": ("nzdusd", "nzd"), "XAUUSD": ("xauusd", "gold"),
}


@lru_cache(maxsize=1)
def load_calendar() -> dict[str, dict]:
    with CALENDAR_PATH.open(encoding="utf-8") as f:
        return {e["id"]: e for e in (yaml.safe_load(f) or {}).get("events") or []}


def calendar_prompt() -> str:
    """Calendar lines with only the instruments whose event family is still untested."""
    ledger, dnr = _ledger_text(), _do_not_reopen()
    lines = []
    for e in load_calendar().values():
        open_syms = [s for s in e["instruments"]
                     if novelty_check({"event_id": e["id"], "instrument": s}, ledger, dnr) is None]
        if open_syms:
            lines.append(f"- {e['id']}: {e['name']}, {e['time']} {e['tz']} ({e['days']}); {', '.join(open_syms)}")
    return "\n".join(lines)


def fact_check(idea: dict) -> str | None:
    event_id = str(idea.get("event_id") or "").strip()
    event = load_calendar().get(event_id)
    if not event:
        return f"fact-check: event_id {event_id or '(none)'} is not a scheduled event in config/event_calendar.yaml"
    symbol = str(idea.get("instrument") or "").upper().strip()
    if symbol not in event["instruments"]:
        return f"fact-check: {symbol or '(no instrument)'} is not traded around {event_id}"
    return None


def _ledger_text() -> list[tuple[str, str]]:
    from ats.research.ledger import ledger_rows

    return [(str(t.get("id")), " ".join(str(t.get(k) or "") for k in ("id", "mechanism", "why_closed", "baseline")).lower())
            for t in ledger_rows()]


def _do_not_reopen() -> list[str]:
    from ats.lab.proposer import do_not_reopen

    return [s.lower() for s in do_not_reopen()]


def novelty_check(idea: dict, ledger: list[tuple[str, str]] | None = None,
                  dnr: list[str] | None = None) -> str | None:
    event = load_calendar().get(str(idea.get("event_id") or ""))
    if not event:
        return None
    ledger = _ledger_text() if ledger is None else ledger
    dnr = _do_not_reopen() if dnr is None else dnr
    keys = [k.lower() for k in event.get("keywords") or []]
    names = ALIASES.get(str(idea.get("instrument") or "").upper(), ())
    hits = [hid for hid, text in ledger if any(k in text for k in keys) and any(n in text for n in names)]
    if hits:
        return f"novelty: {event['id']} on {idea.get('instrument')} already tested ({', '.join(hits[:5])})"
    closed = [s for s in dnr if any(k in s for k in keys)]
    if closed:
        return f"novelty: {event['id']} is on the do-not-reopen list ({closed[0][:80]})"
    return None
