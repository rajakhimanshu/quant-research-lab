"""Intake backlog CRUD — the CRM layer sitting upstream of hypotheses.yaml.

All reads/writes to config/intake.yaml go through this module.
Never writes to hypotheses.yaml or ledger.yaml.

Usage (via CLI):
    python -m ats intake list [--status raw]
    python -m ats intake review INTAKE_001 --tier retail_feasible --category forced_flow
"""
from __future__ import annotations

import re
from datetime import date

import yaml

from ats.config import CONFIG_DIR

INTAKE_PATH = CONFIG_DIR / "intake.yaml"


# ── I/O ──────────────────────────────────────────────────────────────────────

def _load() -> dict:
    if not INTAKE_PATH.exists():
        return {"intake": []}
    with INTAKE_PATH.open(encoding="utf-8") as f:
        blob = yaml.safe_load(f) or {}
    # Normalize: the YAML may have `intake: null` on first load
    if blob.get("intake") is None:
        blob["intake"] = []
    return blob


def _save(blob: dict) -> None:
    with INTAKE_PATH.open("w", encoding="utf-8") as f:
        yaml.dump(blob, f, sort_keys=False, allow_unicode=True, default_flow_style=False)


# ── ID management ─────────────────────────────────────────────────────────────

def _next_id(rows: list[dict]) -> str:
    nums: list[int] = []
    for r in rows:
        m = re.match(r"INTAKE_(\d+)", r.get("id", ""))
        if m:
            nums.append(int(m.group(1)))
    n = max(nums, default=0) + 1
    return f"INTAKE_{n:03d}"


# ── Public API ────────────────────────────────────────────────────────────────

def add_idea(idea: dict) -> str:
    """Add a new idea to the intake backlog.

    Deduplicates by causal_actor text (case-insensitive, exact match).
    Returns the INTAKE_xxx id (new or existing).
    """
    blob = _load()
    rows: list[dict] = blob.get("intake") or []

    ref = (idea.get("source_ref") or "").strip()
    if ref.startswith("http"):
        for r in rows:
            if (r.get("source_ref") or "").strip() == ref:
                return r["id"]

    # Soft dedup: skip only if causal_actor is *identical* to an existing
    # non-rejected record. Near-duplicates are flagged by dedup.py separately.
    new_actor = (idea.get("causal_actor") or "").strip().lower()
    if new_actor:
        for r in rows:
            if (
                (r.get("causal_actor") or "").strip().lower() == new_actor
                and r.get("status") != "rejected_at_intake"
            ):
                return r["id"]  # already present — return existing id

    iid = _next_id(rows)
    idea = dict(idea)  # don't mutate caller's dict
    idea["id"] = iid
    idea.setdefault("created", str(date.today()))
    idea.setdefault("status", "raw")
    rows.append(idea)
    blob["intake"] = rows
    _save(blob)
    return iid


def update_status(intake_id: str, status: str, **kwargs) -> bool:
    """Update the status (and any extra fields) of an existing record.

    Returns True if found and updated, False if not found.
    """
    blob = _load()
    rows: list[dict] = blob.get("intake") or []
    for r in rows:
        if r["id"] == intake_id:
            r["status"] = status
            r.update(kwargs)
            blob["intake"] = rows
            _save(blob)
            return True
    return False


def get_idea(intake_id: str) -> dict | None:
    """Fetch a single record by id. Returns None if not found."""
    rows = list_rows()
    return next((r for r in rows if r["id"] == intake_id), None)


def list_rows(status_filter: str | None = None) -> list[dict]:
    """Return all intake records, optionally filtered by status."""
    rows: list[dict] = _load().get("intake") or []
    if status_filter:
        rows = [r for r in rows if r.get("status") == status_filter]
    return rows


def print_intake(rows: list[dict]) -> None:
    """Pretty-print a list of intake records."""
    if not rows:
        print("No records found.")
        return
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    def _safe(text: str) -> str:
        return text.encode("ascii", errors="replace").decode("ascii")

    print(f"{'id':14}  {'status':22}  {'tier':18}  {'source':18}  name")
    print("-" * 100)
    for r in rows:
        name = _safe((r.get("name") or "")[:40])
        print(
            f"{r['id']:14}  "
            f"{r.get('status', '?'):22}  "
            f"{r.get('timeframe_tier', '?'):18}  "
            f"{r.get('source', '?'):18}  "
            f"{name}"
        )
        actor = _safe((r.get("causal_actor") or "").strip().replace("\n", " "))
        if actor:
            print(f"               actor: {actor[:100]}")
        dedup = r.get("dedup_check") or []
        if dedup:
            print(f"               dedup: {dedup[:5]}")
