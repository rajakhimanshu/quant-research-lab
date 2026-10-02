"""The automated research cycle and the human gates after it.

    sweep -> propose -> validate/dedup -> budget -> freeze -> test as ONE family -> record

The machine stops at CANDIDATE. Unlocking out-of-sample data and deciding a
paper candidate are human commands (`ats lab unlock`, `ats lab decide`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import date, datetime, timezone

import yaml

from ats.config import DATA_DIR, ROOT, load_settings
from ats.lab import book
from ats.lab.proposer import groq_complete, load_queue, propose_from_lead, save_queue
from ats.lab.spec import freeze

REPORTS_DIR = ROOT / "research" / "lab" / "reports"
DEFAULTS = {"max_batch": 8, "min_days_between_batches": 7, "max_leads_per_cycle": 10}


def lab_settings() -> dict:
    return {**DEFAULTS, **(load_settings().get("lab") or {})}


def _has_data(spec: dict) -> list[str]:
    tf = spec["timeframe"]
    syms = list(spec["symbols"]) + ([spec["params"]["leader"]] if spec["params"].get("leader") else [])
    return [s for s in syms if not (DATA_DIR / "raw" / f"{s}_{tf}.parquet").exists()]


def budget_block(runs: list[dict], min_days: int, today: date | None = None) -> str | None:
    """Return a reason to stop, or None. One family per `min_days` limits snooping by volume."""
    if not runs:
        return None
    today = today or date.today()
    last = date.fromisoformat(str(runs[-1]["date"]))
    gap = (today - last).days
    if gap < min_days:
        return f"last batch {runs[-1]['batch']} was {gap} day(s) ago; budget allows one per {min_days} days"
    return None


def _intake_leads(limit: int) -> list[dict]:
    from ats.ideas.intake_store import list_rows

    leads = []
    for r in list_rows():
        if r.get("lab_proposal") or r.get("status") in {"rejected_at_intake", "promoted"}:
            continue
        leads.append({k: r.get(k) for k in ("id", "name", "title", "summary_excerpt", "causal_actor",
                                             "causal_category", "instrument", "timeframe", "source_ref")})
        if len(leads) >= limit:
            break
    return leads


def _library_leads(limit: int, used: set[str]) -> list[dict]:
    from ats.ideas.mechanism_library import load_library

    out = []
    for m in load_library():
        if m.get("tier") != "retail_feasible" or m.get("id") in used:
            continue
        out.append({"id": m["id"], "name": m.get("name"), "causal_actor": m.get("causal_actor"),
                    "causal_category": m.get("causal_category"), "source_ref": m["id"],
                    "reference": m.get("reference"), "notes": m.get("notes")})
        if len(out) >= limit:
            break
    return out


def _mark_intake(lead_id: str, pid: str) -> None:
    from ats.ideas import intake_store

    blob = intake_store._load()
    for r in blob.get("intake") or []:
        if r.get("id") == lead_id:
            r["lab_proposal"] = pid
    intake_store._save(blob)


def propose_cycle(max_leads: int, complete: Callable[[list[dict]], str] = groq_complete) -> list[dict]:
    rows = load_queue()
    used = {str(r.get("source")) for r in rows}
    leads = [lead for lead in _intake_leads(max_leads) if str(lead.get("source_ref") or lead["id"]) not in used]
    leads += _library_leads(max(0, max_leads - len(leads)), used)
    made = []
    for lead in leads:
        rec = propose_from_lead(lead, complete, rows)
        if str(lead.get("id", "")).startswith("INTAKE_"):
            _mark_intake(lead["id"], rec["id"])
        made.append(rec)
        print(f"  {rec['id']} {rec['status']:18} <- {str(lead.get('name') or lead['id'])[:60]}")
    save_queue(rows)
    return made


def test_batch(max_batch: int, dry_run: bool = False) -> dict | None:
    """Freeze up to `max_batch` queued specs and test them as one family."""
    from ats.research.pipeline import run_hypotheses
    from ats.research.scoreboard import write_scoreboard

    rows = load_queue()
    picked = []
    for r in rows:
        if r.get("status") != "queued":
            continue
        missing = _has_data(r["spec"])
        if missing:
            r["status"], r["errors"] = "no_data", [f"missing data/raw for {missing} {r['spec']['timeframe']}"]
            continue
        picked.append(r)
        if len(picked) >= max_batch:
            break
    if not picked:
        save_queue(rows)
        print("No testable proposals in the queue.")
        return None

    batch_id = "lab_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    print(f"Batch {batch_id}: {len(picked)} spec(s) -> one Bonferroni family")
    if dry_run:
        for r in picked:
            print(f"  would freeze {r['id']}: {r['spec']['name']} [{r['spec']['template']}]")
        return None

    entries = freeze([r["spec"] for r in picked], batch_id)
    for r, e in zip(picked, entries):
        r["status"], r["frozen_as"] = "frozen", e["id"]
    save_queue(rows)

    n = len(entries)
    alpha = float(load_settings()["validation"]["significance_alpha"]) / n
    reports = run_hypotheses([e["id"] for e in entries], unlock_oos=False, n_tests=n)

    lab = book.load_book()
    for e, rep in zip(entries, reports):
        lab.append(book.row_from_report(e, rep, batch_id, n, alpha))
    book.save_book(lab)

    counts = Counter(rep.decision for rep in reports)
    runs = book.load_runs()
    runs.append({"batch": batch_id, "date": date.today().isoformat(), "n_tests": n, "alpha": round(alpha, 6),
                 "ids": [e["id"] for e in entries], "decisions": dict(counts)})
    book.save_runs(runs)

    summary = ", ".join(f"{k} x{v}" for k, v in sorted(counts.items()))
    book.append_journal([date.today().isoformat(), batch_id,
                         f"Automated lab family of {n} (alpha {alpha:.4f}); OOS locked. "
                         + "; ".join(f"{e['id']} [{e['template']}]" for e in entries),
                         summary, "Machine stops at CANDIDATE; humans unlock OOS."])
    write_scoreboard()
    path = write_report(batch_id, entries, reports, n, alpha)
    print(f"Report: {path}")
    return {"batch": batch_id, "ids": [e["id"] for e in entries], "decisions": dict(counts)}


def write_report(batch_id: str, entries: list[dict], reports: list, n: int, alpha: float):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"# Lab batch {batch_id}", "",
             f"- Family size: **{n}** · Bonferroni alpha: **{alpha:.4f}** · OOS: **locked**",
             "- Specs were frozen in `config/hypotheses.yaml` before this run.", "",
             "| ID | Template | Market | Decision | Train vs baseline / val | Reason |",
             "|---|---|---|---|---|---|"]
    for e, rep in zip(entries, reports):
        mk = f"{'+'.join(e['params']['symbols'])} {e['params']['timeframe']}"
        reason = (rep.reject_reason or "").replace("|", "/")
        lines.append(f"| `{e['id']}` | {e['template']} | {mk} | **{rep.decision}** | {book.result_line(rep)} | {reason} |")
    lines += ["", "## Mechanisms", ""]
    for e in entries:
        lines += [f"**{e['id']}** — {e['name']}", "", f"- Why: {e['why']}", f"- Payer: {e['payer']}",
                  f"- Baseline: {e['baseline']}", f"- Source: {e.get('source') or 'n/a'}", ""]
    path = REPORTS_DIR / f"{batch_id}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_cycle(sweep: bool = True, propose: bool = True, test: bool = True, force: bool = False,
              dry_run: bool = False, max_batch: int | None = None,
              complete: Callable[[list[dict]], str] = groq_complete) -> dict | None:
    cfg = lab_settings()
    if sweep:
        from ats.ideas.paper_monitor import run_paper_monitor

        print("[1/4] Sweep research sources")
        run_paper_monitor(save=not dry_run)
    if propose:
        print("[2/4] Propose template specs from leads")
        try:
            propose_cycle(int(cfg["max_leads_per_cycle"]), complete)
        except EnvironmentError as exc:
            print(f"  skipped (no LLM key): {str(exc).splitlines()[0]}")
            print("  Add specs by hand: python -m ats lab propose --file my_spec.yaml")
    if not test:
        return None
    print("[3/4] Budget check")
    block = budget_block(book.load_runs(), int(cfg["min_days_between_batches"]))
    if block and not force:
        print(f"  stop: {block}")
        return None
    print("[4/4] Freeze and test one family")
    return test_batch(int(max_batch or cfg["max_batch"]), dry_run=dry_run)


def _find(rows: list[dict], hid: str) -> dict:
    for r in rows:
        if r["id"] == hid:
            return r
    raise SystemExit(f"{hid} is not in the lab book")


def unlock(hid: str) -> str:
    """Human gate 1: spend the OOS segment of a CANDIDATE, once."""
    from ats.research.pipeline import run_hypotheses
    from ats.research.scoreboard import write_scoreboard

    rows = book.load_book()
    row = _find(rows, hid)
    if row["decision"] != "CANDIDATE":
        raise SystemExit(f"{hid} is {row['decision']}; only a CANDIDATE can be unlocked")
    family = next((r for r in book.load_runs() if r["batch"] == row["family"]), {"n_tests": 1})
    rep = run_hypotheses([hid], unlock_oos=True, n_tests=int(family["n_tests"]))[0]
    oos = rep.oos or {}
    row["oos_unlocked"] = date.today().isoformat()
    row["oos"] = (f"{(oos.get('rate') or 0) * 100:.1f}% vs {(oos.get('baseline_rate') or 0) * 100:.1f}% "
                  f"n={oos.get('n', 0)}, R {book.mean_r(rep, 'oos')}")
    row["decision"] = rep.decision if rep.decision in {"OOS_PASS", "REJECT", "NEEDS_MORE_DATA"} else "REJECT"
    row["why_closed"] = ("OOS cleared. Awaiting human: ats lab decide --id " + hid + " --paper|--reject"
                         if row["decision"] == "OOS_PASS" else f"OOS: {rep.reject_reason}. OOS spent; closed.")
    book.save_book(rows)
    book.append_journal([date.today().isoformat(), hid, "Human OOS unlock (one-time).", row["decision"], row["oos"]])
    write_scoreboard()
    return row["decision"]


def decide(hid: str, paper: bool, note: str) -> str:
    """Human gate 2: label an OOS_PASS a paper candidate, or reject anything open."""
    from ats.research.scoreboard import write_scoreboard

    if not note.strip():
        raise SystemExit("--note is required (why you decided)")
    rows = book.load_book()
    row = _find(rows, hid)
    if paper and row["decision"] != "OOS_PASS":
        raise SystemExit(f"{hid} is {row['decision']}; PAPER_CANDIDATE needs OOS_PASS first")
    if row["decision"] in {"REJECT", "PAPER_CANDIDATE"}:
        raise SystemExit(f"{hid} is already {row['decision']}")
    row["decision"] = "PAPER_CANDIDATE" if paper else "REJECT"
    row["human_decision"] = {"date": date.today().isoformat(), "note": note.strip()}
    row["why_closed"] = note.strip()
    book.save_book(rows)
    book.append_journal([date.today().isoformat(), hid, "Human decision.", row["decision"], note.strip()])
    write_scoreboard()
    return row["decision"]


def status() -> dict:
    from ats.research.ledger import ledger_rows

    queue = Counter(str(r.get("status")) for r in load_queue())
    lab = book.load_book()
    runs = book.load_runs()
    open_rows = [r for r in lab if r["decision"] in {"CANDIDATE", "OOS_PASS"}]
    return {
        "queue": dict(queue),
        "lab_tests": len(lab),
        "lab_batches": len(runs),
        "last_batch": runs[-1]["batch"] if runs else None,
        "lifetime_tests": len(ledger_rows()),
        "awaiting_human": [(r["id"], r["decision"]) for r in open_rows],
    }


def print_status() -> None:
    s = status()
    print(yaml.safe_dump(s, sort_keys=False))
    print("Note: every test ever run raises the chance that one 'pass' is luck. "
          "Treat an OOS_PASS as a reason to paper trade, never as proof.")
