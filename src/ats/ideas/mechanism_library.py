"""Mechanism library loader and printer.

Usage:
    python -m ats intake library
    python -m ats intake library --tier retail_feasible
"""
from __future__ import annotations

import yaml

from ats.config import CONFIG_DIR

LIBRARY_PATH = CONFIG_DIR / "mechanism_library.yaml"


def load_library() -> list[dict]:
    """Load all mechanisms from mechanism_library.yaml."""
    if not LIBRARY_PATH.exists():
        return []
    with LIBRARY_PATH.open(encoding="utf-8") as f:
        blob = yaml.safe_load(f) or {}
    return blob.get("mechanisms", [])


def _safe(text: str, width: int = 110) -> str:
    """Truncate and strip characters that break Windows cp1252 terminal."""
    return text[:width].encode("ascii", errors="replace").decode("ascii")


def print_library(tier_filter: str | None = None) -> None:
    """Print the mechanism library to stdout, optionally filtered by tier."""
    import sys
    # Reconfigure stdout to UTF-8 on Windows to handle special characters
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    mechs = load_library()
    if tier_filter:
        mechs = [m for m in mechs if m.get("tier") == tier_filter]

    if not mechs:
        print(f"No mechanisms found{f' for tier={tier_filter}' if tier_filter else ''}.")
        print(f"Library path: {LIBRARY_PATH}")
        return

    retail = [m for m in mechs if m.get("tier") == "retail_feasible"]
    infra = [m for m in mechs if m.get("tier") == "infra_gated"]

    print(f"Mechanism Library  ({len(mechs)} total | {len(retail)} retail_feasible | {len(infra)} infra_gated)")
    if tier_filter:
        print(f"Filter: tier={tier_filter}")
    print()
    print(f"{'id':12}  {'tier':18}  {'category':24}  name")
    print("-" * 90)
    for m in mechs:
        name = _safe(m.get("name", ""), 60)
        print(
            f"{m.get('id', '?'):12}  "
            f"{m.get('tier', '?'):18}  "
            f"{m.get('causal_category', '?'):24}  "
            f"{name}"
        )
        actor = (m.get("causal_actor") or "").strip().replace("\n", " ")
        if actor:
            print(f"             actor: {_safe(actor)}")
        ref = (m.get("reference") or "").strip().replace("\n", " ")
        if ref:
            print(f"             ref:   {_safe(ref)}")
        notes = (m.get("notes") or "").strip().replace("\n", " ")
        if notes:
            print(f"             notes: {_safe(notes)}")
        print()
