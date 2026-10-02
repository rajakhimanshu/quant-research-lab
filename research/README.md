# Research records (Proofbook)

Every hypothesis we test is recorded. Nothing is “try and forget.”

## Where each test lives

| File | Role | Update when |
|---|---|---|
| `config/hypotheses.yaml` | Frozen spec (why, params) **before** the run | New H created |
| `research/ledger.yaml` | Canonical closed book (one row per H, decision, why_closed) | After every decision |
| `research/journal.md` | Dated log (what ran, numbers, next constraint) | After every decision |
| `research/HYPOTHESIS_SCOREBOARD.md` | Human/shareable table of all Hs | After ledger changes (`python -m ats scoreboard`) |
| `data/results/H*_*.json` | Raw machine report from that run (gitignored) | Auto on `ats test` |
| `research/HOW_WE_TEST.md` | Gates, costs, reject rules | When protocol changes |
| `research/lab/book.yaml` | Automated lab rows (merged into the ledger view) | Written by `ats lab run` / `unlock` / `decide` |
| `research/lab/runs.yaml` | One entry per lab family (n, alpha, ids) | Written by `ats lab run` |
| `research/lab/reports/*.md` | Per-batch report | Written by `ats lab run` |

## Commands

```powershell
python -m ats test --id H196_example
python -m ats ledger
python -m ats scoreboard
```

## After every test

Follow `AFTER_EACH_TEST.md` (checklist). Commit the ledger/journal/scoreboard update in **separate tiny commits** when possible.
