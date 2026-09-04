# Algo Trading System

Python research lab for **one retail forex system**. You propose hypotheses. This repo tests them. It will not invent a profitable EA.

MT5 is the data source (and later, execution). Python is the research stack. Those are parallel tracks, not a rewrite of your C++/DSA interview prep.

## What you do vs what Cursor does

You:
1. Bring a hypothesis with a reason (`why` this should exist in the market).
2. Read the report and decide: reject, needs more data, or candidate.
3. Paper trade survivors yourself. Do not skip that.

Cursor / this repo:
1. Pull and clean MT5 data.
2. Turn your hypothesis into a test.
3. Run train / validation (out-of-sample stays locked until you unlock it).
4. Print reject / candidate with the numbers in front of you.

If you type "find me an edge" or "make a profitable strategy", the correct response is: **no**. Give a testable statement instead.

## GitHub repo to create

Name: **`algo-trading-system`**  
Visibility: **private**  
Do **not** tick "Add a README" (this folder already has the files).

Then connect it (after this repo has its first commit):

```powershell
gh auth login
gh repo create algo-trading-system --private --source=. --remote=origin --push
```

If you create the empty repo in the GitHub UI instead:

```powershell
git remote add origin https://github.com/YOUR_USER/algo-trading-system.git
git push -u origin main
```

## Setup (once)

Use **Python 3.12**. The MetaTrader5 package has no 3.14 wheel. You already have 3.12 installed.

```powershell
cd "W:\Currently Working\Algo Trading System"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
```

Open **one** MT5 terminal, log in, enable AutoTrading / Algo Trading.

```powershell
python -m ats doctor
python -m ats pull
```

H1 needs a calendar CSV at `data/calendar/high_impact.csv` (see that folder's README). H2 does not.

```powershell
python -m ats test --id H1_event_reversal
python -m ats test --id H2_tap_breakout
```

Do not pass `--unlock-oos` until a hypothesis is a survivor on train+validation.

## Current hypotheses

Defined in `config/hypotheses.yaml`:

- **H1** — large candle inside a high-impact news window retraces more than equally large candles with no news (overreaction / inventory unwind).
- **H2** — third tap of a level, after hold then fail, breaks more often than tap 1 or 2 (order absorption).

Thresholds live in `config/settings.yaml` and `config/hypotheses.yaml`. Change numbers there, not by sprinkling magic constants into detectors.

## Idea intake (not an edge finder)

Sources feed an inbox. You still write the `why` and freeze a spec before `python -m ats test`. The lab will not auto-code papers into EAs or scrape broker logins.

```powershell
python -m ats ideas list
python -m ats ideas arxiv
python -m ats ideas blogs
python -m ats ideas cot
python -m ats ideas cross
python -m ats ideas sentiment
python -m ats test --id H9_cot_spec_fade
```

| Command | What it is | What it is not |
|---|---|---|
| `ideas arxiv` | arXiv q-fin paper leads | A trade rule |
| `ideas blogs` | Methodology reading list | Signals |
| `ideas cot` | Official CFTC weekly files | A buy/sell |
| `ideas cross` | Gold–DXY (and yields–JPY) correlation snapshot | A mean-reversion EA |
| `ideas sentiment` | CSV you drop under `data/sentiment/` | An IG scrape |

Inbox: `config/ideas.yaml`. Same discipline as H1/H2: precise definition, causal why, frozen params, costed train/val, OOS locked.

## Honest expectations

Most hypotheses die in the quick-reject step. That is the system working. A candidate is not a live strategy. Paper trade comes after a survivor, then tiny size.

There is no MQL5 EA in this repo yet on purpose. Automation of a rejected idea is how the last EA lost money.
