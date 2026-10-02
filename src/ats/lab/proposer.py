"""Turn research leads into template specs (or a stated rejection).

The LLM is a translator, not an optimiser: it may only map a documented
mechanism onto one template and copy clock times / thresholds the mechanism
itself implies. It never sees results. Every proposal, accepted or not, is
kept in research/lab/proposals.yaml.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import date

import yaml

from ats.config import ROOT
from ats.lab.spec import CATEGORIES, known_signatures, signature, validate_spec
from ats.lab.templates import TIMEFRAMES, UNIVERSE, catalogue

LAB_DIR = ROOT / "research" / "lab"
PROPOSALS_PATH = LAB_DIR / "proposals.yaml"

SYSTEM = (
    "You are a market-microstructure research assistant for an FX and gold research lab. "
    "You translate ONE documented mechanism into ONE frozen test spec, or reject it. "
    "You never optimise parameters and never see results. Respond with JSON only."
)

RULES = """Rules:
1. Universe: {universe}. Timeframes: {timeframes}. Nothing else.
2. The spec must name a payer: a participant who is FORCED to trade (fix, auction, settlement,
   rebalancing, hedging, inventory limits, stop orders). No payer -> reject.
3. Indicator rules (RSI, moving averages, Bollinger, Fibonacci, MACD, patterns) are not a why -> reject.
4. Use exactly one template from the catalogue and only its params. If no template fits -> reject.
5. Take clock times and thresholds from the mechanism itself (e.g. a fix at 16:00 London).
   Where the mechanism is silent, leave the param out so the template default applies.
6. Do not propose anything on the do-not-reopen list.
7. Output JSON: {{"decision": "spec" | "reject", "reason": "...", "spec": {{"name": "...",
   "why": "...(>=80 chars, the mechanism)...", "payer": "...", "causal_category": one of {categories},
   "template": "...", "symbols": ["..."], "timeframe": "...", "params": {{...}}}}}}
"""


def _load_queue() -> list[dict]:
    if not PROPOSALS_PATH.exists():
        return []
    with PROPOSALS_PATH.open(encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("proposals") or []


def save_queue(rows: list[dict]) -> None:
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    with PROPOSALS_PATH.open("w", encoding="utf-8") as f:
        f.write("# Every lab proposal, including rejected ones. Written by `ats lab`.\n")
        yaml.safe_dump({"proposals": rows}, f, sort_keys=False, allow_unicode=False, width=100)


def load_queue() -> list[dict]:
    return _load_queue()


def _next_pid(rows: list[dict]) -> str:
    nums = [int(m.group(1)) for r in rows if (m := re.match(r"P(\d+)", str(r.get("id", ""))))]
    return f"P{max(nums, default=0) + 1:04d}"


def do_not_reopen() -> list[str]:
    path = ROOT / "research" / "ledger.yaml"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [str(x) for x in (yaml.safe_load(f) or {}).get("do_not_reopen") or []]


def build_prompt(lead: dict) -> list[dict]:
    rules = RULES.format(universe=", ".join(UNIVERSE), timeframes=", ".join(TIMEFRAMES),
                         categories=list(CATEGORIES))
    user = (
        f"{rules}\nTemplate catalogue:\n{json.dumps(catalogue(), indent=1)}\n\n"
        f"Do-not-reopen list:\n- " + "\n- ".join(do_not_reopen()[:150]) + "\n\n"
        f"Lead:\n{json.dumps(lead, indent=1, default=str)}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def groq_complete(messages: list[dict]) -> str:
    from ats.ideas.groq_brainstorm import GROQ_MODEL, _get_client

    resp = _get_client().chat.completions.create(
        model=GROQ_MODEL, max_tokens=1500, temperature=0.0, messages=messages,
    )
    return resp.choices[0].message.content or ""


def _parse(raw: str) -> dict:
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start:end + 1])


def record_spec(spec: dict, source: str, rows: list[dict] | None = None, origin: str = "manual") -> dict:
    """Validate a spec, dedup it by signature, append it to the queue. Returns the record."""
    rows = _load_queue() if rows is None else rows
    clean, errors = validate_spec({**spec, "source": spec.get("source") or source})
    rec: dict = {"id": _next_pid(rows), "created": date.today().isoformat(), "origin": origin,
                 "source": source}
    if errors:
        rec.update(status="invalid", errors=errors, spec=spec)
    else:
        sig = signature(clean)
        queued = {r.get("signature") for r in rows if r.get("status") in {"queued", "frozen"}}
        if sig in known_signatures() or sig in queued:
            rec.update(status="duplicate", signature=sig, spec=clean,
                       errors=["same template/symbols/timeframe/params already tested or queued"])
        else:
            rec.update(status="queued", signature=sig, spec=clean)
    rows.append(rec)
    return rec


def propose_from_lead(lead: dict, complete: Callable[[list[dict]], str] = groq_complete,
                      rows: list[dict] | None = None) -> dict:
    rows = _load_queue() if rows is None else rows
    source = str(lead.get("source_ref") or lead.get("id") or lead.get("name") or "lead")
    try:
        out = _parse(complete(build_prompt(lead)))
    except Exception as exc:
        rec = {"id": _next_pid(rows), "created": date.today().isoformat(), "origin": "llm",
               "source": source, "status": "error", "errors": [str(exc)[:200]]}
        rows.append(rec)
        return rec
    if out.get("decision") != "spec" or not isinstance(out.get("spec"), dict):
        rec = {"id": _next_pid(rows), "created": date.today().isoformat(), "origin": "llm",
               "source": source, "status": "rejected_by_model",
               "errors": [str(out.get("reason") or "no spec")[:300]]}
        rows.append(rec)
        return rec
    return record_spec(out["spec"], source, rows, origin="llm")
