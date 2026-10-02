"""K-A04 discretionary trade reverse-engineering.

Structured log of your own discretionary trades to extract the causal mechanism
behind trades that worked, formalized into the intake schema.

Usage:
    python -m ats intake log-trade \
        --instrument XAUUSD \
        --direction long \
        --timeframe M5 \
        --reason "Gold swept the overnight low at NY open, then reclaimed within 3 bars" \
        --causal-actor "NY market-makers absorb retail stop orders below overnight low, then offload inventory into the reclaim" \
        --outcome "win +1.2R"
"""
from __future__ import annotations

from datetime import date

from ats.ideas.dedup import check_dedup
from ats.ideas.intake_store import add_idea


def log_trade(
    instrument: str,
    direction: str,
    timeframe: str,
    reason: str,
    causal_actor: str,
    outcome: str = "",
) -> str:
    """Log a discretionary trade and extract its causal mechanism as an intake record.

    Args:
        instrument:   e.g. "XAUUSD"
        direction:    "long" or "short"
        timeframe:    e.g. "M5"
        reason:       Setup description — what you saw on the chart.
        causal_actor: REQUIRED — one sentence: who pays you and why.
        outcome:      Optional trade result, e.g. "win +1.2R" or "loss -1R".

    Returns:
        The INTAKE_xxx id of the created record.

    Raises:
        ValueError: If causal_actor is empty.
    """
    actor = causal_actor.strip()
    if not actor:
        raise ValueError(
            "causal_actor is required for K-A04 log entries — no exceptions.\n"
            "Write one sentence: who is forced to trade and why.\n"
            "Example: 'NY market-makers absorb retail stop orders then offload inventory into the reclaim.'"
        )

    dedup = check_dedup(actor, reason)
    if dedup:
        print(f"[INFO] Dedup check found potential near-duplicates: {dedup[:5]}")
        print("  Review these before assuming this mechanism is new.")

    iid = add_idea({
        "source": "discretionary",
        "source_ref": f"K-A04 {date.today()}",
        "name": reason[:80],
        "causal_category": "",          # owner assigns at review
        "causal_actor": actor,
        "timeframe_tier": "",           # owner assigns at review
        "instrument": instrument,
        "timeframe": timeframe,
        "direction": direction,
        "outcome": outcome,
        "expected_holding_bars": None,  # owner fills at review
        "expected_trades_month": None,  # owner fills at review
        "dedup_check": dedup,
        "status": "raw",
    })

    print(f"K-A04 trade logged as {iid}.")
    print(f"  reason:  {reason[:80]}")
    print(f"  actor:   {actor[:80]}")
    if outcome:
        print(f"  outcome: {outcome}")
    print(f"\nNext: python -m ats intake review {iid} --tier retail_feasible --category <category>")
    return iid
