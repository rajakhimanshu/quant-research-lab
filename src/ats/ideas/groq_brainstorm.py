"""Fully automated Groq brainstorm — calls Groq LLM API directly.

Requires: GROQ_API_KEY in .env (or environment).
Requires: pip install openai  (Groq uses the OpenAI-compatible client)

Usage:
    python -m ats intake groq-brainstorm --category forced_flow --n 5
    python -m ats intake groq-brainstorm --all --n 5  (runs all 3 categories)

The brainstorm prompt explicitly lists ~80 already-rejected mechanisms to avoid
generating duplicates. Ideas with an empty causal_actor are discarded automatically.
All others land in intake.yaml with status=raw for human review before promotion.

Model: llama-3.3-70b-versatile (verify current model at console.groq.com/docs/models)
Free tier: ~14,400 requests/day — far more than weekly use requires.
"""
from __future__ import annotations

import json
import os

from ats.ideas.dedup import check_dedup
from ats.ideas.intake_store import add_idea
from ats.ideas.prompts import CATEGORIES, _PROMPTS

# ── Groq client setup ─────────────────────────────────────────────────────────

def _get_client():
    """Build the Groq-compatible OpenAI client. Raises if key is missing."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package not installed. Run:\n"
            "  .venv\\Scripts\\pip install openai"
        ) from None

    # Load .env if present (python-dotenv is already in requirements.txt)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv optional here; key may be set in env directly

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found in environment or .env file.\n"
            "Setup:\n"
            "  1. Sign up free at console.groq.com\n"
            "  2. Create an API key (no card required)\n"
            "  3. Add to .env:  GROQ_API_KEY=gsk_..."
        )
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


# Groq model — check https://console.groq.com/docs/models for the current list
GROQ_MODEL = "openai/gpt-oss-120b"


# ── Core brainstorm function ───────────────────────────────────────────────────

def run_groq_brainstorm(category: str, n: int = 5) -> int:
    """Run a single-category brainstorm via Groq and save ideas to intake.yaml.

    Args:
        category: One of 'forced_flow', 'inventory_pressure', 'structural_liquidity'.
        n:        Number of ideas to request from the LLM.

    Returns:
        Number of ideas saved (discards any with empty causal_actor).
    """
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category!r}. Choose from: {CATEGORIES}")

    client = _get_client()

    prompt = _PROMPTS[category].format(n=n)
    system_msg = (
        "You are a precise market microstructure researcher. "
        "Respond ONLY with a valid JSON array. No markdown, no preamble, no explanation. "
        "Every object in the array must have a non-empty 'causal_actor' field."
    )

    print(f"[groq-brainstorm] category={category}, n={n}, model={GROQ_MODEL}")
    print("  Calling Groq API...")

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=2000,
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"Groq API call failed: {exc}") from exc

    raw = resp.choices[0].message.content.strip()

    # Strip markdown code fences if the model ignored the instruction
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    # Parse JSON
    try:
        ideas: list[dict] = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"  [ERROR] Could not parse Groq response as JSON: {exc}")
        print(f"  Raw response:\n{raw[:500]}")
        raise

    if not isinstance(ideas, list):
        raise ValueError(f"Expected a JSON array, got {type(ideas).__name__}")

    saved = 0
    discarded = 0
    for idea in ideas:
        actor = (idea.get("causal_actor") or "").strip()
        if not actor:
            name = idea.get("name", "<no name>")
            print(f"  [DISCARD] Missing causal_actor: {name[:60]}")
            discarded += 1
            continue

        dedup = check_dedup(actor, idea.get("name", ""))
        iid = add_idea({
            "source": "llm_brainstorm",
            "source_ref": f"groq_{category}",
            "name": (idea.get("name") or "")[:80],
            "causal_category": category,
            "causal_actor": actor,
            "timeframe_tier": "",           # owner assigns at review
            "instrument": idea.get("instrument", ""),
            "timeframe": idea.get("timeframe", ""),
            "expected_holding_bars": idea.get("expected_holding_bars"),
            "expected_trades_month": idea.get("expected_trades_month"),
            "dedup_check": dedup,
            "notes": (idea.get("notes") or "")[:300],
            "status": "raw",
        })
        dedup_note = f" [dedup: {dedup[:2]}]" if dedup else ""
        safe_name = idea.get("name", "")[:60].encode("ascii", errors="replace").decode("ascii")
        print(f"  Saved {iid}{dedup_note}: {safe_name}")
        saved += 1

    print(f"\n[{category}] Saved: {saved}  Discarded (no causal_actor): {discarded}")
    return saved


def run_groq_brainstorm_all(n: int = 5) -> dict[str, int]:
    """Run brainstorm for all three causal categories.

    Returns a dict mapping category → number of ideas saved.
    """
    results: dict[str, int] = {}
    for cat in CATEGORIES:
        print(f"\n{'=' * 60}")
        results[cat] = run_groq_brainstorm(cat, n=n)
    total = sum(results.values())
    print(f"\n{'=' * 60}")
    print(f"All categories done. Total saved: {total}")
    for cat, count in results.items():
        print(f"  {cat}: {count}")
    return results
