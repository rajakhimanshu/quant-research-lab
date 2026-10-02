# Proofbook

**Individual FX & gold hypothesis lab.**  
Not a company. Not a signal service. Not a quick-money project.

You bring a causal **why**. Proofbook freezes it as `H###`, runs it after spread + slippage on locked train/validation, and writes the verdict into a permanent **closed book**. Rejects stay closed. No EA until a real survivor exists.

| | |
|---|---|
| **Name** | Proofbook |
| **CLI** | `ats` → `python -m ats …` |
| **Universe** | FX majors + **XAUUSD** only (M5 / M15 / H1) |
| **Stack** | Python **3.12** + MetaTrader 5 data |
| **Repo** | https://github.com/rajakhimanshu/algo-trading-system |

## Status (honest)

- **250** ledger rows (**H1–H255**)
- **233 REJECT** · **13 NEEDS_MORE_DATA** · **3 VOID** · **1 PAPER label (H5 equity archived only)**
- **No FX/gold survivor** → **no MT5 EA**
- H5 equity RSI is **archived** (out of universe)
- H86 weekend gap: **REJECT** after last-level FAIL (Sunday fill spread)
- H192 Xetra gold fade: **OOS REJECT**
- H214 Gulf 05+08→09 liq fade: **REJECT** (below 11:00 baseline, R−)
- H219–H248 September batch: **audited and re-run as one family of 29** — 0 candidates (see below)
- Still thin: H37, H68, H77, H80, H81, H85 (+ seven sept_batch tests below `min_trades`)

## Audit case study: the September batch (H215–H255)

A fast batch of 41 tests produced several "winners". The audit found the wins were artefacts:

| Problem | Effect | Fix |
|---|---|---|
| Fill at signal-bar close, no spread | Inflated R in 15 modules | Next-bar open ± cost, net R (`hypotheses/execution.py`) |
| No baseline arm | Nothing to beat | Mirror/control baseline required; pipeline rejects treatment-only |
| Hit rate up, mean R ≤ 0 | "Hit-rate mirage" | Pipeline gate: CANDIDATE and OOS_PASS need mean R > 0 |
| Overlapping 48-bar holds | n inflated ~6×, fake p-values | One position at a time |
| α = 0.05 per test, reruns | Multiple-testing leak | One Bonferroni family of 29 (α = 0.0017) |
| Spec chosen on full sample | OOS snooped (H255) | VOID, not a result |

After the fixes, the one pre-fix CANDIDATE (H234, gold SMA20 pullback) fell from n=2,766 to n=199 and p=0.15 — its edge was overlapping entries plus long gold beta. Full rows in [`research/ledger.yaml`](research/ledger.yaml).

## Research intake (where ideas come from)

Ideas enter `config/intake.yaml` as raw leads, are deduplicated against the closed book, and need a human-assigned causal payer before the promotion gate prints a frozen-spec stub. Sources:

- **Papers:** arXiv q-fin queries plus RSS from **BIS**, **Federal Reserve (FEDS, IFDP)**, **ECB** and **Bank of England** working papers, filtered for FX/gold/dealer/liquidity terms. SSRN/NBER by manual drop.
- **Mechanism library:** `config/mechanism_library.yaml` — documented payers (fixes, auctions, inventory, settlement).
- **LLM brainstorm (Groq):** constrained by category and the ledger's do-not-reopen list.
- **Discretionary trade log:** manual trades that suggest a mechanism.

```powershell
python -m ats intake paper-monitor          # dry run
python -m ats intake list --status raw
python -m ats intake promote INTAKE_xxx     # gate: payer, retail tier, >= 8 trades/month, dedup, reviewed
```

Design notes: [`research/INTAKE_PIPELINE.md`](research/INTAKE_PIPELINE.md) · [`research/OPERATING_LOOP.md`](research/OPERATING_LOOP.md)

Full table: [`research/HYPOTHESIS_SCOREBOARD.md`](research/HYPOTHESIS_SCOREBOARD.md)

## What Proofbook is for

1. Force every idea through the same gates (costs, baseline, Bonferroni, train/val gap).  
2. Keep memory so we never retest noise.  
3. Stay retail-realistic: one symbol, simple rules, paper → small live → EA only after proof.

Aspiration (~5%/month, ~1% risk) is a **goal**, not something we curve-fit in a backtest.

## How a test works

1. Write the **why** (who pays you).  
2. Freeze rules + baseline in `config/hypotheses.yaml` **before** results.  
3. `python -m ats test --id Hxxx`  
4. Decide from the report (you decide; the lab does not declare “profitable”).  
5. Update ledger → journal → scoreboard (see checklist below).  
6. OOS stays **locked** until an explicit survivor unlock.

Details: [`research/HOW_WE_TEST.md`](research/HOW_WE_TEST.md)

**Refused without a new causal why:** indicator stacks (EMA 13/50/200, S/R folklore), multi-TF fishing, “find me an edge,” reopening a REJECT.

## Records (every H is logged)

| Record | Path |
|---|---|
| Frozen specs | [`config/hypotheses.yaml`](config/hypotheses.yaml) |
| Closed book | [`research/ledger.yaml`](research/ledger.yaml) |
| Dated log | [`research/journal.md`](research/journal.md) |
| Shareable scoreboard | [`research/HYPOTHESIS_SCOREBOARD.md`](research/HYPOTHESIS_SCOREBOARD.md) |
| After each test | [`research/AFTER_EACH_TEST.md`](research/AFTER_EACH_TEST.md) |
| External AI brief | [`research/PERPLEXITY_RESEARCH_BRIEF.md`](research/PERPLEXITY_RESEARCH_BRIEF.md) |
| Name note | [`docs/PROJECT_NAME.md`](docs/PROJECT_NAME.md) |

```powershell
python -m ats ledger
python -m ats scoreboard
```

## Setup (once)

```powershell
cd "W:\Currently Working\Algo Trading System"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
```

Open **one** MT5 terminal (logged in), then:

```powershell
python -m ats doctor
python -m ats pull
python -m ats test --id H9_cot_spec_fade
```

Do **not** pass `--unlock-oos` until train+val clears and you choose to unlock.

## Retail path (locked)

Python lab → paper on this broker → several **$100–$200** accounts (execution / fills) → **$1,000–$2,000** (where ~1% risk on a ~15–20 pip stop is realistic with 0.01 lot) → EA.

No grid. No martingale. One symbol until a survivor.

## Useful commands

| Command | Purpose |
|---|---|
| `python -m ats doctor` | Python 3.12 + MT5 check |
| `python -m ats pull` | OHLC into `data/raw` |
| `python -m ats test --id H…` | Costed train/val (OOS locked) |
| `python -m ats ledger` | Print closed book |
| `python -m ats scoreboard` | Rebuild scoreboard markdown |
| `python -m ats ideas …` | Idea inbox (not an edge finder) |
| `python -m ats intake …` | Research intake: paper monitor, review, promotion gate |

## Docs map

- Research index: [`research/README.md`](research/README.md)  
- Repo layout: [`docs/REPO_LAYOUT.md`](docs/REPO_LAYOUT.md)  
- Contributing / tiny commits: [`CONTRIBUTING.md`](CONTRIBUTING.md)  
- Agent protocol: [`.cursor/rules/research-protocol.mdc`](.cursor/rules/research-protocol.mdc)  

## Git hygiene

After **every** test: update ledger + journal + scoreboard, prefer **small commits**, push.  
No Cursor co-author trailers on purpose — commits should read as the owner’s lab history.
