"""Dedup checker — cross-references new ideas against research/ledger.yaml.

Verified field names in ledger.yaml (inspected 2026-09-26):
  tests[].id             e.g. "H3_ny_open_sweep"
  tests[].mechanism      e.g. "Overnight-range stops get run at NY cash open, then reclaim."
  tests[].why_closed     e.g. "Hit-rate mirage. Do not retune stop/target or RR."
  tests[].baseline       e.g. "Same sweep rule 11-16 NY."
  tests[].decision       e.g. "REJECT"

The dedup check tokenizes the new idea's causal_actor + name against the combined
text of mechanism + why_closed + baseline for each ledger entry, and returns ids
where 3+ significant tokens overlap. This is a FLAG, not a hard-block — the
promotion gate uses --override-dedup to let you proceed after review.

Important: this intentionally returns false-positives rather than false-negatives.
A spurious hit is a minor annoyance; a missed near-duplicate wastes a test slot.
"""
from __future__ import annotations

import yaml

from ats.config import CONFIG_DIR

LEDGER_PATH = CONFIG_DIR.parent / "research" / "ledger.yaml"

# Tokens shorter than this are ignored (noise words)
_MIN_TOKEN_LEN = 4

# Minimum shared token count to flag a hit
_OVERLAP_THRESHOLD = 3

# Hard stop-words that appear in almost every hypothesis and add no signal
_STOP_WORDS = {
    "trade", "trades", "trading", "bars", "same", "then", "with",
    "that", "this", "from", "into", "after", "before", "when",
    "versus", "baseline", "does", "not", "more", "than", "the",
    "and", "for", "are", "have", "will", "been", "such", "only",
    "first", "each", "also", "both", "their", "which", "should",
    "versus", "using", "used",
}


def _tokenize(text: str) -> set[str]:
    """Split text into lowercase tokens, filtering noise."""
    return {
        w.lower().strip(".,;:()[]'\"")
        for w in text.split()
        if len(w) >= _MIN_TOKEN_LEN and w.lower() not in _STOP_WORDS
    }


def _load_ledger_tests() -> list[dict]:
    """Load ledger.yaml tests plus the lab book. Returns [] if the ledger is missing."""
    if not LEDGER_PATH.exists():
        return []
    from ats.research.ledger import ledger_rows

    return ledger_rows()


def check_dedup(causal_actor: str, mechanism_name: str = "") -> list[str]:
    """Return list of ledger ids that look similar to the new idea.

    Args:
        causal_actor:   The one-sentence causal actor field of the new idea.
        mechanism_name: The name/title of the new idea (optional, adds signal).

    Returns:
        List of ledger H-ids with ≥ OVERLAP_THRESHOLD token overlap.
        Empty list = no near-duplicates found (but not a guarantee of novelty).
    """
    tests = _load_ledger_tests()
    if not tests:
        return []

    query_tokens = _tokenize(causal_actor + " " + mechanism_name)
    if not query_tokens:
        return []

    hits: list[str] = []
    for t in tests:
        # Combine the three text fields that describe what was actually tested
        corpus_text = " ".join([
            t.get("mechanism") or "",
            t.get("why_closed") or "",
            t.get("baseline") or "",
        ])
        corpus_tokens = _tokenize(corpus_text)
        overlap = query_tokens & corpus_tokens
        if len(overlap) >= _OVERLAP_THRESHOLD:
            hits.append(t["id"])

    # Cap at 10 to avoid noise overwhelming the display
    return hits[:10]


def print_dedup_report(causal_actor: str, mechanism_name: str = "") -> None:
    """Print a human-readable dedup report for manual inspection."""
    hits = check_dedup(causal_actor, mechanism_name)
    if not hits:
        print("Dedup: no near-duplicates found in ledger.yaml.")
        print("  (This does not guarantee novelty — review the ledger manually for related families.)")
    else:
        print(f"Dedup: {len(hits)} potential near-duplicate(s) in ledger.yaml:")
        for h in hits:
            print(f"  - {h}")
        print("  Review these before promoting. Use --override-dedup if genuinely distinct.")
