# Proofbook

**An automated quant research lab for FX majors and gold that is built to reject ideas honestly.**

Proofbook takes research leads (papers, a mechanism library, an LLM, or your own notes), turns each one into a frozen hypothesis with a causal *why*, tests it after spread and slippage against a baseline on locked train/validation data, and writes every verdict into a permanent public book. The machine stops at `CANDIDATE`. Unlocking out-of-sample data and choosing a paper candidate are human decisions.

> **Not financial advice. No strategy here is profitable or recommended.** See [Disclaimer](#disclaimer).

| | |
|---|---|
| **CLI** | `python -m ats …` |
| **Universe** | 7 FX majors + XAUUSD, M5 / M15 / H1 |
| **Data** | MetaTrader 5 (required, Windows) |
| **Stack** | Python **3.12** (the MetaTrader5 package has no newer wheel) |
| **Record** | 250 frozen tests, 0 survivors in FX/gold |

## Status (honest)

- **250** ledger rows (**H1–H255**): **233 REJECT** · **13 NEEDS_MORE_DATA** · **3 VOID** · **1 ARCHIVED**
- **No FX/gold hypothesis has survived.** Nothing here is a strategy to trade.
- H5 (textbook RSI(2) equity dip-buy) is **archived**: out of universe, never live, and today's validator would refuse it.
- Full table: [`research/HYPOTHESIS_SCOREBOARD.md`](research/HYPOTHESIS_SCOREBOARD.md)

The value of the repo is the process and the closed book, not a result.

## How the lab works

```
sources ─► leads ─► template spec ─► validate + dedup ─► budget ─► freeze ─► test as one family ─► book
                                                                                   │
                                        CANDIDATE ─► human: unlock OOS once ─► OOS_PASS ─► human: paper or reject
```

- **Templates, not generated code.** An LLM (optional) may only fill the parameters of six fixed event templates (`clock_run`, `session_box`, `sweep_reclaim`, `lead_lag`, `weekend_gap`, `month_end`), each with a closed, bounded schema.
- **A payer or nothing.** Every spec names who is forced to trade and why. Indicator terms (RSI, EMA, MACD, Bollinger, Fibonacci, support/resistance…) are refused.
- **No retuning.** A signature excludes stop/target/horizon, so a retune of a closed idea is caught as a duplicate.
- **Frozen before results**, tested as **one Bonferroni family**, at most one family per week by default.
- **Costs and a baseline on every trade**: next-bar fill with spread + slippage, treatment must beat a control arm, and mean R must be positive after costs.
- **OOS locked** until a human runs `ats lab unlock`, once.

Architecture, gates and limits: [`docs/LAB.md`](docs/LAB.md) · Test protocol: [`research/HOW_WE_TEST.md`](research/HOW_WE_TEST.md)

## Quickstart

Requirements: Windows, Python 3.12, a MetaTrader 5 terminal logged in to any broker (a demo account is fine).

```powershell
git clone https://github.com/rajakhimanshu/algo-trading-system.git proofbook
cd proofbook
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env          # optional: MT5 terminal path, GROQ_API_KEY for the proposer
pytest -q                       # synthetic-data tests; no MT5 needed
```

Open MT5, log in, then:

```powershell
python -m ats doctor            # Python 3.12 + MT5 connection check
python -m ats pull              # OHLC into data/raw (gitignored)
python -m ats lab templates     # what the lab can test
python -m ats lab propose --file examples/proposal.yaml
python -m ats lab run --dry-run
python -m ats lab run --no-sweep
python -m ats lab status
```

Before trusting any result, replace the example cost defaults in [`config/settings.yaml`](config/settings.yaml) with your own broker's median spreads.

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

After the fixes, the one pre-fix CANDIDATE (H234, gold SMA20 pullback) fell from n=2,766 to n=199 and p=0.15 — its edge was overlapping entries plus long gold beta. These failures are what the automated lab's gates now enforce by construction.

## Research intake (where leads come from)

- **Papers:** arXiv q-fin queries plus RSS from BIS, Federal Reserve (FEDS, IFDP), ECB and Bank of England working papers, filtered for FX/gold/dealer/liquidity terms.
- **Mechanism library:** [`config/mechanism_library.yaml`](config/mechanism_library.yaml) — documented payers (fixes, auctions, inventory, settlement).
- **LLM (Groq, optional):** constrained by the template catalogue and the ledger's do-not-reopen list.
- **Manual:** hand-written specs or intake leads.

Details: [`research/INTAKE_PIPELINE.md`](research/INTAKE_PIPELINE.md)

## Records

| Record | Path |
|---|---|
| Frozen specs | [`config/hypotheses.yaml`](config/hypotheses.yaml) |
| Closed book (hand-curated) | [`research/ledger.yaml`](research/ledger.yaml) |
| Closed book (lab-written) | `research/lab/book.yaml` |
| Dated log | [`research/journal.md`](research/journal.md) |
| Scoreboard | [`research/HYPOTHESIS_SCOREBOARD.md`](research/HYPOTHESIS_SCOREBOARD.md) |
| Lab batch reports | `research/lab/reports/` |

## Commands

| Command | Purpose |
|---|---|
| `python -m ats doctor` | Python 3.12 + MT5 check |
| `python -m ats pull` | OHLC into `data/raw` |
| `python -m ats lab …` | Automated loop: `templates`, `propose`, `run`, `status`, `unlock`, `decide` |
| `python -m ats test --id H…` | Run one frozen hypothesis (OOS locked) |
| `python -m ats ledger` | Print the closed book |
| `python -m ats scoreboard` | Rebuild the scoreboard |
| `python -m ats intake …` | Paper monitor, intake review, promotion gate |

## Repo layout

See [`docs/REPO_LAYOUT.md`](docs/REPO_LAYOUT.md). Contributions: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Disclaimer

This is a research and education project. It is **not financial advice**, not a signal service, and not an offer to manage money. Every hypothesis in the book is closed or unproven, and none should be traded. Backtests, including honest ones, do not predict future results. Trading leveraged FX and gold can lose more than your deposit. Cost defaults are examples from one retail account; your broker will differ. You are solely responsible for anything you do with this code.

## License

[MIT](LICENSE)
