"""Canonical closed book. A REJECT is closed. Do not retune it."""

from __future__ import annotations

import yaml

from ats.config import ROOT

LEDGER_PATH = ROOT / "research" / "ledger.yaml"


def load_ledger() -> dict:
    with LEDGER_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ledger_rows() -> list[dict]:
    return list(load_ledger().get("tests") or [])


def print_ledger() -> None:
    blob = load_ledger()
    rows = ledger_rows()
    counts: dict[str, int] = {}
    for r in rows:
        d = str(r.get("decision") or "?")
        counts[d] = counts.get(d, 0) + 1
    print("Closed book — python -m ats ledger")
    print("An edge is who pays you, after costs, vs a baseline. OOS stays locked.")
    print("Counts:", "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print()
    print(f"{'id':24} {'decision':18} {'book':14} why closed")
    print("-" * 100)
    for r in rows:
        hid = str(r.get("id", ""))[:24]
        dec = str(r.get("decision", ""))[:18]
        book = str(r.get("book", ""))[:14]
        why = " ".join(str(r.get("why_closed") or "").split())
        print(f"{hid:24} {dec:18} {book:14} {why[:70]}")
    print()
    print("Do not reopen:")
    for item in blob.get("do_not_reopen") or []:
        print(f"  - {item}")
    print()
    print("Still open:")
    for item in blob.get("open") or []:
        print(f"  - {item}")
