"""Machine-written closed book for lab tests (research/lab/book.yaml) and run log."""

from __future__ import annotations

import re
from datetime import date

import yaml

from ats.config import ROOT
from ats.research.ledger import LAB_BOOK_PATH

RUNS_PATH = ROOT / "research" / "lab" / "runs.yaml"
JOURNAL_PATH = ROOT / "research" / "journal.md"

HEADER = (
    "# Lab closed book. Written by `ats lab`; humans only change rows via\n"
    "# `ats lab unlock` / `ats lab decide`. A REJECT is closed: no retune.\n"
)


def _read(path, key: str) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list((yaml.safe_load(f) or {}).get(key) or [])


def _write(path, key: str, rows: list[dict], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump({key: rows}, f, sort_keys=False, allow_unicode=False, width=100)


def load_book() -> list[dict]:
    return _read(LAB_BOOK_PATH, "tests")


def save_book(rows: list[dict]) -> None:
    _write(LAB_BOOK_PATH, "tests", rows, HEADER)


def load_runs() -> list[dict]:
    return _read(RUNS_PATH, "runs")


def save_runs(rows: list[dict]) -> None:
    _write(RUNS_PATH, "runs", rows, "# One row per `ats lab run` test batch (one Bonferroni family).\n")


def mean_r(report, part: str) -> float | None:
    for note in report.notes:
        m = re.match(rf"{part} treatment mean R=(-?[\d.]+)", note)
        if m:
            return float(m.group(1))
    return None


def result_line(report) -> str:
    tr, va = report.train or {}, report.validation or {}

    def pct(x):
        return "n/a" if x is None or x != x else f"{x * 100:.1f}%"

    def r(x):
        return "n/a" if x is None else f"{x:+.2f}"

    return (f"{pct(tr.get('rate'))} vs {pct(tr.get('baseline_rate'))} n={tr.get('n', 0)}, "
            f"R {r(mean_r(report, 'train'))}; val {pct(va.get('rate'))} vs "
            f"{pct(va.get('baseline_rate'))} n={va.get('n', 0)}, R {r(mean_r(report, 'validation'))}")


def row_from_report(entry: dict, report, batch_id: str, n_tests: int, alpha: float) -> dict:
    decision = report.decision
    why = report.reject_reason or ""
    if decision == "CANDIDATE":
        why = f"Cleared train+val in a family of {n_tests} (alpha {alpha:.4f}). Awaiting human: ats lab unlock --id {entry['id']}"
    elif decision == "NEEDS_MORE_DATA":
        why = f"{why}. Spec frozen; do not loosen to raise n."
    else:
        why = f"{why}. Closed; do not retune."
    return {
        "id": entry["id"],
        "date": date.today().isoformat(),
        "book": entry["split_book"],
        "family": batch_id,
        "decision": decision,
        "template": entry["template"],
        "signature": entry["signature"],
        "mechanism": entry["why"],
        "payer": entry["payer"],
        "baseline": entry["baseline"],
        "source": entry.get("source", ""),
        "train": result_line(report),
        "why_closed": why,
    }


def append_journal(cells: list[str]) -> None:
    line = "| " + " | ".join(c.replace("|", "/").replace("\n", " ") for c in cells) + " |\n"
    text = JOURNAL_PATH.read_text(encoding="utf-8") if JOURNAL_PATH.exists() else ""
    if text and not text.endswith("\n"):
        text += "\n"
    JOURNAL_PATH.write_text(text + line, encoding="utf-8")
