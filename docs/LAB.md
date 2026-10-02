# The automated research lab (`ats lab`)

The lab turns research leads into frozen, costed hypothesis tests and records
every result, including the failures. It is built to **reject** ideas cheaply
and honestly. It never declares anything profitable, and it stops before the
two decisions that need a human.

```
 sources ──► leads ──► template spec ──► validate + dedup ──► budget ──► freeze ──► test (one family) ──► book
 (papers,    (intake    (LLM or hand,     (payer, why,         (one       (hypotheses   (train/val, OOS       (ledger view,
  library,    queue)     closed schema)    banned terms,        family     .yaml, before  locked, costs,        journal,
  manual)                                  universe, sig)       per week)  results)       Bonferroni)           scoreboard, report)
                                                                                                 │
                                                                     CANDIDATE ──► human: ats lab unlock ──► OOS_PASS ──► human: ats lab decide
```

## Why it is built this way

Most "automated strategy finders" fail in the same ways. Each design choice
below closes one of them.

| Failure mode | What the lab does |
|---|---|
| An LLM writes trading code and overfits it | The LLM only fills parameters of a **fixed template**. It never writes code. Every template has a closed parameter schema with bounds. |
| Indicator soup with no reason to work | A spec needs a **payer** (who is forced to trade) and a causal `why` of at least 80 characters. Indicator words (RSI, EMA, MACD, Bollinger, Fibonacci, support/resistance, …) are refused outright. |
| The same idea retested with new knobs until it passes | A **signature** hash over template, symbols, timeframe and mechanism parameters. Risk knobs (`stop_atr`, `target_atr`, `horizon_bars`) are excluded, so a retune counts as a duplicate. |
| Rules picked after seeing results | Specs are **frozen** into `config/hypotheses.yaml` (with batch id and signature) before the run. |
| Many tests, one lucky pass | Each batch is **one Bonferroni family** (α = 0.05 / n). A **budget** allows one family per `min_days_between_batches`. |
| Peeking at out-of-sample data | OOS stays **locked**. Only `ats lab unlock` spends it, once, using the original family size for α. Headline stats exclude OOS while locked. |
| Ignoring costs | Every trade fills at the next bar open with spread + slippage from `config/settings.yaml`. Success and R are net of cost. |
| "Beats nothing" | Every template has a **baseline arm** (a control clock, the opposite side on the same bar, or a control day). Treatment must beat it. |
| High hit rate, negative expectancy | CANDIDATE and OOS_PASS also need **mean R > 0** after costs. |
| Overlapping trades inflating n | One position at a time per template. |

## Templates

`python -m ats lab templates` prints the full catalogue with parameter bounds.

| Template | Mechanism it can express | Baseline |
|---|---|---|
| `clock_run` | A named clock (fix, auction, session open/close, settlement) forces flow; fade or follow the bar into it. | Same rule at a control clock on the same days. |
| `session_box` | Stops cluster around a session's opening range; the first break is a stop run (fade) or initiative flow (follow). | Opposite side on the same break bar. |
| `sweep_reclaim` | Dealers absorb stops just beyond a recent swing; a shallow sweep that closes back inside reverts. | Opposite side on the same sweep bar. |
| `lead_lag` | Price discovery happens first in the more liquid pair; the follower reprices with a lag. | Opposite side on the same follower bar. |
| `weekend_gap` | Weekend news reprices with no liquidity; the reopen gap overshoots and is faded. | Fade of a same-size London-open jump on a normal day. |
| `month_end` | Month-end rebalancing forces flow in the month-to-date direction over the last sessions. | Same follow on mid-month sessions. |

Universe: the seven FX majors plus XAUUSD, on M5, M15 or H1. Nothing else validates.

## Statuses

| Status | Set by | Meaning |
|---|---|---|
| `queued` / `invalid` / `duplicate` / `rejected_by_model` / `error` / `no_data` / `frozen` | proposer, loop | Proposal queue states (`research/lab/proposals.yaml`). |
| `REJECT` | machine | Failed a gate. Closed forever. Do not retune or flip. |
| `NEEDS_MORE_DATA` | machine | Too few trades. Do not loosen the spec to get more. |
| `CANDIDATE` | machine | Cleared train and validation. **The machine stops here.** |
| `OOS_PASS` | `ats lab unlock` | Cleared the one-time OOS check. Still not proof. |
| `PAPER_CANDIDATE` | `ats lab decide --paper` | A human chose to paper trade it. Needs a written note. |

## Commands

```powershell
python -m ats lab templates                      # what the lab can test
python -m ats lab propose --file examples/proposal.yaml   # queue a hand-written spec
python -m ats lab propose --lead INTAKE_xxx      # ask the LLM to translate one lead (needs GROQ_API_KEY)
python -m ats lab run --dry-run                  # show what would be frozen
python -m ats lab run                            # sweep -> propose -> budget -> freeze -> test
python -m ats lab status                         # queue, open candidates, lifetime test count
python -m ats lab unlock --id H260_x --confirm   # human: spend OOS once
python -m ats lab decide --id H260_x --paper --note "why"   # human: label after OOS_PASS
python -m ats lab decide --id H260_x --reject --note "why"
```

`ats lab run` options: `--no-sweep`, `--no-propose`, `--no-test`, `--max-batch N`,
`--force` (ignore the budget; the override is visible in `runs.yaml` dates), `--dry-run`.

Without a `GROQ_API_KEY`, the propose step is skipped and you add specs by hand.

## Writing a spec by hand

See [`examples/proposal.yaml`](../examples/proposal.yaml). Required fields:
`name`, `why` (≥ 80 chars, no indicator terms), `payer`, `causal_category`
(`forced_flow`, `inventory_pressure` or `structural_liquidity`), `template`,
`symbols`, `timeframe`, `params`. Missing parameters take template defaults;
unknown or out-of-range ones are refused.

## Where results go

| File | Content |
|---|---|
| `config/hypotheses.yaml` | Frozen specs (`template`, `signature`, `family`, `frozen`, `payer`). |
| `research/lab/book.yaml` | One row per lab test. Merged into `ats ledger` and the scoreboard. |
| `research/lab/runs.yaml` | One entry per family: date, n, α, ids, decisions. |
| `research/lab/reports/<batch>.md` | Human-readable batch report. |
| `research/journal.md` | One dated row per batch, unlock and decision. |
| `research/HYPOTHESIS_SCOREBOARD.md` | Rebuilt after every lab write. |

The hand-curated `research/ledger.yaml` is never rewritten by the machine.

## Limits you should know

- Templates bound what can be tested. A mechanism that does not fit one needs
  a new template (code review), not a looser schema.
- The lifetime test count keeps growing. Even with per-family Bonferroni, one
  OOS_PASS among hundreds of tests can be luck. Paper trading exists for this.
- Cost defaults are one retail account's medians. Use your own broker's.
- MT5 data depth is limited (about 100k bars per symbol/timeframe on many brokers).
