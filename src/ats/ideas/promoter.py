"""Promotion gate — validates a reviewed intake record and prints the hypotheses.yaml block.

Fixes applied (per spec v2):
  FIX #1: dedup is a FLAG, not a hard-block. Use --override-dedup after manual review.
  FIX #2: MIN_TRADES_MONTH = 8 (~100/year, the lab min_trades on a 12-month train).
          A floor of 30 rejected every clock event (max ~22 sessions/month), which is
          the forced-flow class the lab exists to test (INTAKE_009, INTAKE_011).
  FIX #3: Prints an explicit STUB DEFAULTS warning when stop_atr/target_atr are left
          at the 1.0/1.0 defaults — prevents a lazy default from becoming a frozen spec.

Usage:
    python -m ats intake promote INTAKE_001           # dry run, prints YAML, does not update intake.yaml
    python -m ats intake promote INTAKE_001 --confirm  # marks as promoted in intake.yaml
    python -m ats intake promote INTAKE_001 --override-dedup --confirm  # skip dedup flag
"""
from __future__ import annotations

import re

import yaml

from ats.config import CONFIG_DIR
from ats.ideas.intake_store import get_idea, update_status

HYPOTHESES_PATH = CONFIG_DIR / "hypotheses.yaml"

MIN_TRADES_MONTH = 8


# ── H-number detection ────────────────────────────────────────────────────────

def _ids_in(path, key: str) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        blob = yaml.safe_load(f) or {}
    return [str(r.get("id", "")) for r in blob.get(key) or []]


def _next_h_number() -> int:
    """Next H-number after every id in hypotheses.yaml AND the ledger (voids included)."""
    from ats.research.ledger import LEDGER_PATH

    ids = _ids_in(HYPOTHESES_PATH, "hypotheses") + _ids_in(LEDGER_PATH, "tests")
    nums = [int(n) for i in ids for n in re.findall(r"H(\d+)", i)]
    return max(nums, default=214) + 1


# ── Validation gate ───────────────────────────────────────────────────────────

def _validate(record: dict, override_dedup: bool = False) -> list[str]:
    """Return list of gate failure messages. Empty list = all gates passed.

    FIX #1: dedup is a flag, not a hard-block. --override-dedup bypasses the flag.
    FIX #2: MIN_TRADES_MONTH = 8.
    """
    failures: list[str] = []

    # Gate 1: causal_actor must be non-empty
    if not (record.get("causal_actor") or "").strip():
        failures.append("causal_actor is empty — this is a mandatory field with no exceptions")

    # Gate 2: tier must be retail_feasible
    tier = record.get("timeframe_tier", "")
    if tier != "retail_feasible":
        failures.append(
            f"timeframe_tier='{tier}' — must be 'retail_feasible'. "
            "Infra-gated ideas stay in the backlog until infra situation changes."
        )

    # Gate 3: expected_trades_month >= MIN_TRADES_MONTH (FIX #2)
    trades = record.get("expected_trades_month")
    try:
        trades_int = int(trades) if trades is not None else 0
    except (TypeError, ValueError):
        trades_int = 0
    if trades_int < MIN_TRADES_MONTH:
        failures.append(
            f"expected_trades_month={trades} — below the floor of {MIN_TRADES_MONTH}. "
            f"Below {MIN_TRADES_MONTH}/month the train window cannot reach min_trades=100. "
            "Update the estimate or reconsider the hypothesis design."
        )

    # Gate 4: dedup flag (FIX #1 — flag, not hard-block)
    dedup = record.get("dedup_check") or []
    if dedup and not override_dedup:
        failures.append(
            f"Possible near-duplicates in ledger: {dedup[:5]}. "
            "Review these rejected mechanisms, confirm this idea is genuinely distinct, "
            "then re-run with --override-dedup."
        )

    # Gate 5: must be reviewed by a human
    if record.get("status") != "reviewed":
        failures.append(
            f"status='{record.get('status')}' — must be 'reviewed' before promoting. "
            "Run: python -m ats intake review INTAKE_xxx --tier retail_feasible --category ..."
        )

    return failures


# ── Promotion output ──────────────────────────────────────────────────────────

def _build_hypothesis_block(record: dict, h_num: int) -> dict:
    """Build the hypotheses.yaml-compatible dict from the intake record."""
    slug = re.sub(r"[^a-z0-9]+", "_", (record.get("name") or "idea").lower())[:30].strip("_")
    h_id = f"H{h_num}_{slug}"

    params: dict = {
        "timeframe": record.get("timeframe") or "M15",
        "horizon_bars": record.get("expected_holding_bars") or 8,
        "stop_atr": 1.0,    # STUB — must be frozen before adding to hypotheses.yaml
        "target_atr": 1.0,  # STUB — must be frozen before adding to hypotheses.yaml
    }
    if record.get("instrument"):
        params["symbols"] = [record["instrument"]]

    return {
        "id": h_id,
        "enabled": False,
        "name": record.get("name", ""),
        "why": (record.get("causal_actor") or "").strip(),
        "params": params,
    }


def promote(intake_id: str, dry_run: bool = True, override_dedup: bool = False) -> None:
    """Validate an intake record and print the hypotheses.yaml YAML block.

    Args:
        intake_id:      The INTAKE_xxx id to promote.
        dry_run:        If True (default), only prints — does not update intake.yaml.
        override_dedup: If True, suppress the dedup gate failure (use after manual review).
    """
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    record = get_idea(intake_id)
    if record is None:
        print(f"[ERROR] {intake_id} not found in config/intake.yaml")
        return

    failures = _validate(record, override_dedup=override_dedup)
    if failures:
        print(f"\n{'=' * 72}")
        print(f"PROMOTION GATE FAILED for {intake_id}")
        print(f"{'=' * 72}")
        for f in failures:
            print(f"  ✗ {f}")
        print(f"\nFix these issues, then re-run: python -m ats intake promote {intake_id}")
        return

    h_num = _next_h_number()
    block = _build_hypothesis_block(record, h_num)
    h_id = block["id"]

    # FIX #3: explicit STUB DEFAULTS warning
    stub_warning = (
        "\n[WARNING] STUB DEFAULTS DETECTED:\n"
        "   stop_atr=1.0 and target_atr=1.0 are placeholder values.\n"
        "   You MUST freeze the actual entry rule, stop anchor, and target\n"
        "   before adding this block to hypotheses.yaml.\n"
        "   A stub default must NEVER become a frozen spec.\n"
        "   Freeze params -> add to hypotheses.yaml -> then python -m ats test --id " + h_id
    )

    print(f"\n{'=' * 72}")
    print(f"PROMOTION OUTPUT  {intake_id}  ->  {h_id}")
    print(f"{'=' * 72}")
    print()
    print("Add the block below to config/hypotheses.yaml AFTER freezing real params:")
    print()
    print(yaml.dump([block], sort_keys=False, allow_unicode=True, default_flow_style=False))
    print(stub_warning)
    print(f"\n{'=' * 72}")

    if not dry_run:
        ok = update_status(intake_id, "promoted", promoted_to=h_id)
        if ok:
            print(f"\n[OK] {intake_id} marked as promoted -> {h_id} in intake.yaml.")
            print("Core lab files (hypotheses.yaml, ledger.yaml) were NOT modified.")
            print("Next steps:")
            print("  1. Freeze real entry, stop, and target params in hypotheses.yaml")
            print(f"  2. python -m ats test --id {h_id}")
        else:
            print(f"[WARN] Could not update intake.yaml — {intake_id} not found.")
    else:
        print("\n[DRY RUN] intake.yaml not updated. Re-run with --confirm to mark as promoted.")
