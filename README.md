# GrowEdge — Algo Trading System

Retail **FX majors + XAUUSD** research lab. You propose a causal hypothesis. This repo freezes it, costs it, and scores it. It does **not** invent profitable EAs.

**Status (honest):** **195 hypotheses (H1–H195). No FX/gold survivor. No MT5 EA yet.**

## Records (every test is logged)

| Record | Path |
|---|---|
| Frozen specs | `config/hypotheses.yaml` |
| Closed book | `research/ledger.yaml` |
| Dated log | `research/journal.md` |
| Shareable table | `research/HYPOTHESIS_SCOREBOARD.md` |
| How we test / reject | `research/HOW_WE_TEST.md` |
| After-each-test checklist | `research/AFTER_EACH_TEST.md` |
| External AI brief | `research/PERPLEXITY_RESEARCH_BRIEF.md` |

```powershell
python -m ats ledger
python -m ats scoreboard
```

## You vs the lab

**You:** bring a `why` → accept/reject from the report → paper only after a survivor.  
**Lab:** pull MT5 data → code the frozen H → train/val (OOS locked) → print decision.

If you ask “find an edge” or “EMA 13/50/200 on M15,” the answer is **no** without a new causal why (and EMA stacks are already REJECT).

## Setup (Python 3.12 only)

```powershell
cd "W:\Currently Working\Algo Trading System"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
```

Open one MT5 terminal, then:

```powershell
python -m ats doctor
python -m ats pull
python -m ats test --id H9_cot_spec_fade
```

Do **not** pass `--unlock-oos` until train+val survivor + you choose to unlock.

## Retail path (locked)

Python lab → paper on this broker → several **$100–$200** accounts (execution) → **$1,000–$2,000** (~1% risk possible) → EA.  
Exness min lot 0.01. No grid / martingale. One symbol until a survivor.

## Goals

- Find a cost-adjusted edge on FX or gold with a clear payer.  
- Aspiration ~5%/month at ~1% risk **when account size allows** — not a curve-fit target.  
- Keep a permanent reject book so we never retest noise.

## Docs map

- Protocol rule: `.cursor/rules/research-protocol.mdc`  
- Research index: `research/README.md`  
- Ideas inbox: `config/ideas.yaml` (`python -m ats ideas …`)

## GitHub

Remote: `https://github.com/rajakhimanshu/algo-trading-system`  
Update the ledger/journal/scoreboard after **every** test and commit in small slices (see `research/AFTER_EACH_TEST.md`).
