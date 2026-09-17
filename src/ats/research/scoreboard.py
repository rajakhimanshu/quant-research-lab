"""Build research/HYPOTHESIS_SCOREBOARD.md from research/ledger.yaml."""

from __future__ import annotations

from collections import Counter
from datetime import date

from ats.config import ROOT
from ats.research.ledger import load_ledger

OUT = ROOT / "research" / "HYPOTHESIS_SCOREBOARD.md"


def _one_line(s: object, n: int) -> str:
    t = str(s or "").replace("|", "/").replace("\n", " ").strip()
    return t if len(t) <= n else t[: n - 3] + "..."


def build_scoreboard_markdown() -> str:
    led = load_ledger()
    tests = list(led.get("tests") or [])
    counts = Counter(str(t.get("decision") or "?") for t in tests)
    today = date.today().isoformat()
    n = len(tests)
    last_id = str(tests[-1]["id"]) if tests else "—"

    lines: list[str] = [
        "# Proofbook — Hypothesis Scoreboard",
        "",
        f"**As of:** {today}  ",
        "**Lab:** Proofbook — individual FX majors + XAUUSD research (Python). Not live. Not an EA.  ",
        "**Rule:** Each `H` is a frozen causal hypothesis tested after spread+slippage on locked "
        "train/validation. OOS unlocked only for survivors. REJECT is closed — no retune.",
        "",
        "## Summary",
        "",
        "| Decision | Count | Meaning |",
        "|---|---:|---|",
        f"| **REJECT** | {counts.get('REJECT', 0)} | Failed train/val gates, negative R, or OOS fail. Closed. |",
        f"| **NEEDS_MORE_DATA** | {counts.get('NEEDS_MORE_DATA', 0)} | Spec frozen; sample too thin. Do not loosen. |",
        f"| **PAPER_CANDIDATE** | {counts.get('PAPER_CANDIDATE', 0)} | Lab label only — see status notes. |",
        f"| **Total tested** | {n} | Through `{last_id}` |",
        "",
        "### Current status (honest)",
        "",
        "- **No FX/gold live survivor.** Do not start an MT5 EA.",
        "- **H5** PAPER_CANDIDATE on equities only — archived / out of FX+gold universe.",
        "- **H86** weekend G10 gap: REJECT after last-level FAIL on Sunday fill spread.",
        "- **H192** Xetra 17:30 gold: OOS REJECT after lab clear.",
        "- Still thin (NEEDS_MORE_DATA): H37, H68, H77, H80, H81, H85.",
        "",
        "## Protocol",
        "",
        _one_line(led.get("protocol"), 2000),
        "",
        "## Full list",
        "",
        "| # | ID | Decision | Book | Mechanism | Key result |",
        "|---:|---|---|---|---|---|",
    ]

    for i, t in enumerate(tests, 1):
        book = t.get("book") or ""
        fam = t.get("family") or ""
        bf = book if not fam else (fam if not book else (book if book == fam else f"{book}/{fam}"))
        mech = _one_line(t.get("mechanism") or t.get("name"), 110)
        train = _one_line(t.get("train"), 90)
        oos = _one_line(t.get("oos"), 70)
        why = _one_line(t.get("why_closed"), 90)
        if oos:
            result = f"{train} | OOS: {oos}" if train else f"OOS: {oos}"
        elif train and why:
            result = f"{train} — {why}"
        else:
            result = train or why
        result = _one_line(result, 150)
        lines.append(
            f"| {i} | `{t.get('id')}` | **{t.get('decision')}** | {bf} | {mech} | {result} |"
        )

    lines += [
        "",
        "## Source of truth",
        "",
        "- Ledger: `research/ledger.yaml`",
        "- Dated log: `research/journal.md`",
        "- Reprint: `python -m ats ledger` · `python -m ats scoreboard`",
        "",
    ]
    return "\n".join(lines)


def write_scoreboard() -> str:
    text = build_scoreboard_markdown()
    OUT.write_text(text, encoding="utf-8")
    return str(OUT)


def main() -> None:
    path = write_scoreboard()
    led = load_ledger()
    n = len(led.get("tests") or [])
    print(f"Wrote {path} ({n} hypotheses)")


if __name__ == "__main__":
    main()
