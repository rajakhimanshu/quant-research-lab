# Hypothesis Intake Pipeline — How Ideas Enter the Lab

**This pipeline sits UPSTREAM of the lab.**
It generates, deduplicates, and queues causal hypothesis candidates.
Nothing here touches the lab's Bonferroni / OOS / frozen-spec machinery.
`ledger.yaml` and `hypotheses.yaml` are read-only from the pipeline's perspective.

---

## Feasibility Tiers

Every idea is tagged with one of two tiers before it can be reviewed:

| Tier | Mechanism examples | Lab action |
|---|---|---|
| **`retail_feasible`** | Stop-cascade absorption, cross-pair lead-lag, post-news drift (min-scale), micro-session vacuums, inventory skew after one-sided flow | → Promote to queue → test |
| **`infra_gated`** | Tick-level OFI, sub-second adverse selection, any co-location-grade strategy | → Store in backlog with tag, **deprioritize permanently** |

> [!IMPORTANT]
> M1–M3 causal mechanisms (tick-level OFI, sub-second adverse selection) are **real** but require HFT-grade infrastructure. MT5 retail execution cannot compete here. Tag `infra_gated` and never promote — this is Graviton-territory, not retail lab territory.

---

## Pipeline Architecture

```
Mechanism Library (static seed, monthly review)
  ↓
Paper Monitor (arXiv + BIS/Fed/ECB/BoE)    ──→  config/intake.yaml  (status=raw)
Groq Brainstorm (weekly, fully automated)  ──→        ↓
K-A04 Trade Log (manual, ongoing)          ──→    dedup vs ledger.yaml (flag, not block)
                                                       ↓ you review + assign tier/category
                                                  Promotion Gate (validates all gates)
                                                       ↓ prints YAML block to stdout
                                                  config/hypotheses.yaml (you paste, freeze params, run)
```

---

## Causal Categories

The pipeline uses three categories from the existing lab:

| Category | Who pays you | M5–M30 examples |
|---|---|---|
| **`forced_flow`** | Someone who MUST trade (fix, auction, settlement) | Post-news secondary drift, LBMA post-auction drift, option-expiry pre-hedging |
| **`inventory_pressure`** | Dealer offloading accumulated position | Cross-pair lead-lag, post-sweep normalization, one-sided retail flow reversal |
| **`structural_liquidity`** | Predictable book depth drop at a clock point | Overlap open thin-book, pre-open thin periods, session-boundary pauses |

---

## Cadence

| Frequency | Command | What it does |
|---|---|---|
| **Weekly (automated)** | `python -m ats intake paper-monitor --save` | 8 arXiv queries + 6 central-bank research feeds (BIS working papers, BIS research hub, Fed FEDS, Fed IFDP, ECB WP, Bank of England), FX/gold relevance filter, URL dedup → raw leads |
| **As found (manual)** | `python -m ats intake paper-monitor --ssrn <URL>` | One SSRN/NBER/bank-research paper (NBER blocks feed reads) |
| **Weekly (automated)** | `python -m ats intake groq-brainstorm --all --n 5` | 3 × 5 = 15 LLM-generated ideas → intake.yaml |
| **Ongoing (manual)** | `python -m ats intake log-trade ...` | K-A04 discretionary trade → intake.yaml |
| **Monthly (manual)** | Review `config/mechanism_library.yaml` | Add new mechanisms you've read, prune still-infeasible infra-gated entries |
| **Monthly (manual)** | `python -m ats intake list --status raw` | Clear stale raw entries, reject or review |

---

## Promotion Gate

An intake record must pass **all five gates** before the YAML block is printed:

| Gate | Requirement |
|---|---|
| **causal_actor** | Non-empty, one sentence, names who is forced to trade and why |
| **timeframe_tier** | Must be `retail_feasible` (not `infra_gated`) |
| **expected_trades_month** | ≥ 8/month floor (~100/year so train can reach min_trades; 30 blocked every clock event) |
| **dedup_check** | No near-duplicates in ledger, OR `--override-dedup` after manual review |
| **status** | Must be `reviewed` (human reviewed and assigned tier/category) |

> [!WARNING]
> The promoter prints a **STUB DEFAULTS WARNING** when `stop_atr=1.0 / target_atr=1.0` are left as placeholders. You **must** freeze real entry rules, stop anchor, and target before adding to `hypotheses.yaml`. A stub default must never become a frozen spec.

---

## Workflow Step-by-Step

### 1. Weekly automated run (set up in Task Scheduler)
```bat
cd /d W:\Currently Working\Algo Trading System
.venv\Scripts\python -m ats intake paper-monitor --save
.venv\Scripts\python -m ats intake groq-brainstorm --all --n 5
```

### 2. Review your backlog (you do this, not the machine)
```powershell
# See what's new
python -m ats intake list --status raw

# Review a specific record and assign tier + category
python -m ats intake review INTAKE_001 --tier retail_feasible --category forced_flow --trades 80 --bars 8

# Check dedup explicitly before promoting
python -m ats intake dedup "NY market-makers absorb retail stops then offload inventory into the reclaim"

# Reject at intake (bad causal_actor, infra-gated, or duplicate mechanism)
python -m ats intake reject INTAKE_002 --reason "infra_gated: requires tick-level book data"
```

### 3. Promote a reviewed winner
```powershell
# Dry run first (default) — prints YAML, does NOT update intake.yaml
python -m ats intake promote INTAKE_001

# If dedup flagged near-duplicates but you've confirmed it's distinct:
python -m ats intake promote INTAKE_001 --override-dedup

# Finalize (marks promoted in intake.yaml, still prints YAML)
python -m ats intake promote INTAKE_001 --confirm
```

### 4. Freeze params and add to hypotheses.yaml
1. Copy the printed YAML block
2. Open `config/hypotheses.yaml`
3. **Replace the stub `stop_atr: 1.0 / target_atr: 1.0` with real values** — freeze entry rule, stop anchor, target
4. Set `enabled: true` only when ready to run
5. `python -m ats test --id H215_<slug>`

### 5. K-A04 trade log (after a discretionary trade)
```powershell
python -m ats intake log-trade `
  --instrument XAUUSD `
  --direction long `
  --timeframe M5 `
  --reason "Gold swept overnight low at NY open then reclaimed within 3 bars" `
  --causal-actor "NY market-makers absorb retail stop orders below overnight low then offload inventory into the reclaim" `
  --outcome "win +1.2R"
```

---

## Success Metric

Not hypothesis count — **a backlog of 15–20 well-formed, retail-feasible, causal-actor-stated candidates available at any time**, so you are never blocked on "what do I test next," and H215 onward has a queue instead of being decided one idea at a time.

> [!TIP]
> If the backlog drops below 10 reviewed candidates: run `groq-brainstorm --all`, spend 30 minutes reviewing, promote 3–5, reject the rest. One weekly session keeps the queue full.

---

## Command Reference

```
python -m ats intake list [--status raw|reviewed|promoted|rejected_at_intake]
python -m ats intake review INTAKE_xxx [--tier ...] [--category ...] [--trades N] [--bars N]
python -m ats intake promote INTAKE_xxx [--confirm] [--override-dedup]
python -m ats intake reject INTAKE_xxx --reason "..."
python -m ats intake paper-monitor [--save] [--ssrn URL]
python -m ats intake groq-brainstorm [--category ...] [--all] [--n 5]
python -m ats intake brainstorm-prompt forced_flow [--n 5]   # copy-paste fallback
python -m ats intake brainstorm-import --file response.json
python -m ats intake log-trade --instrument ... --direction ... --timeframe ... --reason ... --causal-actor ...
python -m ats intake library [--tier retail_feasible|infra_gated]
python -m ats intake dedup "causal actor text" [--name "mechanism name"]
```

*Config: `config/intake.yaml` (backlog), `config/mechanism_library.yaml` (static seed). Groq key: `GROQ_API_KEY` in `.env`.*
