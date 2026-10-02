"""Validate, fingerprint and freeze template specs.

A spec is the only thing the automated lab can test. It must name who is
forced to trade (the payer), pick a template, and stay inside that
template's parameter schema. Specs are frozen into config/hypotheses.yaml
before any result exists.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date

import yaml

from ats.config import CONFIG_DIR
from ats.lab.templates import TEMPLATES, TIMEFRAMES, UNIVERSE

HYPOTHESES_PATH = CONFIG_DIR / "hypotheses.yaml"

CATEGORIES = ("forced_flow", "inventory_pressure", "structural_liquidity")
RISK_KEYS = {"horizon_bars", "stop_atr", "target_atr"}
MIN_WHY_CHARS = 80
MIN_PAYER_CHARS = 15

# Indicator folklore is not a causal why (protocol).
BANNED_TERMS = (
    "rsi", "ema", "sma", "macd", "bollinger", "fibonacci", "fib ", "moving average",
    "stochastic", "ichimoku", "supertrend", "support and resistance", "candlestick pattern",
    "golden cross", "death cross", "martingale", "grid",
)


def _has_banned(text: str) -> str | None:
    low = f" {text.lower()} "
    for term in BANNED_TERMS:
        if re.search(rf"(?<![a-z]){re.escape(term.strip())}(?![a-z])", low):
            return term.strip()
    return None


def validate_spec(spec: dict) -> tuple[dict, list[str]]:
    """Return (clean spec with defaults filled, errors). Empty errors = testable."""
    errors: list[str] = []
    clean: dict = {}

    name = str(spec.get("name") or "").strip()
    why = " ".join(str(spec.get("why") or "").split())
    payer = " ".join(str(spec.get("payer") or "").split())
    if not name:
        errors.append("name is empty")
    if len(why) < MIN_WHY_CHARS:
        errors.append(f"why must explain the market mechanism (>= {MIN_WHY_CHARS} chars)")
    if len(payer) < MIN_PAYER_CHARS:
        errors.append("payer must name who is forced to trade and why")
    banned = _has_banned(f"{name} {why} {payer}")
    if banned:
        errors.append(f"indicator folklore is not a causal why ({banned!r})")
    category = spec.get("causal_category")
    if category not in CATEGORIES:
        errors.append(f"causal_category must be one of {list(CATEGORIES)}")

    tname = spec.get("template")
    template = TEMPLATES.get(str(tname))
    if template is None:
        errors.append(f"template {tname!r} unknown; choose from {sorted(TEMPLATES)}")
        return clean, errors

    tf = str(spec.get("timeframe") or "")
    if tf not in TIMEFRAMES or tf not in template.timeframes:
        errors.append(f"timeframe {tf!r} not allowed for {tname} {list(template.timeframes)}")
    symbols = spec.get("symbols") or []
    if isinstance(symbols, str):
        symbols = [symbols]
    symbols = [str(s).upper() for s in symbols]
    if not symbols or len(symbols) > 3 or len(set(symbols)) != len(symbols):
        errors.append("symbols must be 1-3 distinct instruments")
    bad = [s for s in symbols if s not in UNIVERSE]
    if bad:
        errors.append(f"symbols {bad} outside the universe (FX majors + XAUUSD)")

    raw = dict(spec.get("params") or {})
    unknown = sorted(set(raw) - set(template.params))
    if unknown:
        errors.append(f"unknown params for {tname}: {unknown}")
    params: dict = {}
    for key, p in template.params.items():
        if key in raw and raw[key] is not None:
            value, err = p.check(key, raw[key])
            if err:
                errors.append(err)
            params[key] = value
        elif p.required:
            errors.append(f"missing required param {key}")
        elif p.default is not None:
            params[key] = p.default
    if not errors:
        for rule in template.rules:
            msg = rule(params)
            if msg:
                errors.append(msg)
    if template.needs_leader and params.get("leader") in symbols:
        errors.append("leader must differ from the traded symbols")

    clean = {
        "name": name[:90],
        "why": why,
        "payer": payer,
        "causal_category": category,
        "template": template.name,
        "baseline": template.baseline,
        "symbols": symbols,
        "timeframe": tf,
        "params": params,
        "source": str(spec.get("source") or "").strip(),
    }
    return clean, errors


def signature(spec: dict) -> str:
    """Fingerprint ignoring risk knobs: a new stop/target/horizon is a retune, not a new test."""
    core = {k: v for k, v in sorted((spec.get("params") or {}).items()) if k not in RISK_KEYS}
    blob = json.dumps(
        {"t": spec["template"], "s": sorted(spec["symbols"]), "tf": spec["timeframe"], "p": core},
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def split_book(symbols: list[str], timeframe: str) -> str:
    gold = "XAUUSD" in symbols
    if timeframe == "M5":
        return "gold_m5" if gold else "fx_m5"
    if timeframe == "M15":
        return "gold_m15" if gold else "fx_m15"
    return "fx_h1"


def _load(path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def known_signatures() -> set[str]:
    from ats.research.ledger import ledger_rows

    sigs = {str(h["signature"]) for h in _load(HYPOTHESES_PATH).get("hypotheses") or [] if h.get("signature")}
    sigs |= {str(t["signature"]) for t in ledger_rows() if t.get("signature")}
    return sigs


def next_h_number() -> int:
    from ats.research.ledger import ledger_rows

    ids = [str(h.get("id", "")) for h in _load(HYPOTHESES_PATH).get("hypotheses") or []]
    ids += [str(t.get("id", "")) for t in ledger_rows()]
    nums = [int(n) for i in ids for n in re.findall(r"H(\d+)", i)]
    return max(nums, default=0) + 1


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:32].strip("_") or "spec"


def freeze(specs: list[dict], batch_id: str, path=HYPOTHESES_PATH) -> list[dict]:
    """Assign H ids and append frozen entries to hypotheses.yaml. Returns the entries."""
    start = next_h_number()
    today = date.today().isoformat()
    entries = []
    for offset, spec in enumerate(specs):
        hid = f"H{start + offset}_{_slug(spec['name'])}"
        entries.append({
            "id": hid,
            "enabled": False,
            "name": spec["name"],
            "why": spec["why"],
            "payer": spec["payer"],
            "causal_category": spec["causal_category"],
            "template": spec["template"],
            "baseline": spec["baseline"],
            "split_book": split_book(spec["symbols"], spec["timeframe"]),
            "signature": signature(spec),
            "family": batch_id,
            "frozen": today,
            "source": spec.get("source", ""),
            "params": {"symbols": spec["symbols"], "timeframe": spec["timeframe"], **spec["params"]},
        })
    block = yaml.safe_dump(entries, sort_keys=False, allow_unicode=False, width=100)
    text = "".join(f"  {ln}\n" if ln else "\n" for ln in block.splitlines())
    header = f"\n  # --- Lab batch {batch_id}: frozen {today} by `ats lab run` before any result ---\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(header + text)
    return entries
