# How We Test — Parameters, Gates, and Why We Reject

**Lab:** **Proofbook** (`ats`, Python 3.12) — individual FX majors + XAUUSD research.  
**We do not test “strategies to make profitable.”** We test **frozen hypotheses** with a causal `why`.

---

## 1. What we are testing

| Term | Meaning |
|---|---|
| **Hypothesis (`H###`)** | One frozen idea: who pays you, exact rules, one symbol, one timeframe, written **before** results. |
| **Treatment** | Trades that use the claimed edge. |
| **Baseline** | Same mechanics **without** the claimed edge (other clock, opposite side, mid-percentile, etc.). |
| **Success** | Path hits target before stop (or defined success column), **after** spread + slippage. |
| **R (r_mult)** | PnL in multiples of stop risk (ATR-based). Hit-rate without +R is a **mirage**. |

If there is **no causal why** (who is forced to trade / inventory / documented payer), we **do not code it**. Indicator stacks (EMA 13/50/200, S/R folklore) are refused or already closed (e.g. H13–H16).

---

## 2. Pipeline (correct order)

1. **Why** — economic / structural / behavioral payer.  
2. **Freeze** — write rules + baseline + metrics in `config/hypotheses.yaml` (no peeking first).  
3. **Code** — simplest version; **no optimization** while creating.  
4. **Run** — `python -m ats test --id Hxxx` (family of ids → Bonferroni).  
5. **Decide** — train/val gates (below). OOS stays **LOCKED**.  
6. **Last-level** (rare) — fill/spread/path stress if lab CANDIDATE.  
7. **OOS unlock** — only after survivor pick + pre-committed gates; one-shot.  
8. **Paper → small live execution check → EA** — only after a human decides; no EA with no survivor.

---

## 3. Fixed lab parameters (`config/settings.yaml`)

### Universe
- Symbols: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD (+ **XAUUSD** per hypothesis).  
- Timeframes used in research: **M5 / M15 / H1** (not a free scan of every TF to pick a winner).  
- No equities / crypto / indices (H5 equity archived).

### Costs (example retail defaults, applied every trade — replace with your broker's medians)
| Symbol | Spread (lab pips) | Slippage |
|---|---:|---:|
| EURUSD | 0.8 | 0.2 |
| GBPUSD | 1.0 | 0.2 |
| USDJPY | 1.0 | 0.2 |
| AUDUSD | 1.1 | 0.2 |
| USDCAD | 1.5 | 0.2 |
| USDCHF | 1.2 | 0.2 |
| NZDUSD | 1.7 | 0.2 |
| XAUUSD | 200 | 20 |

Gold lab unit = 0.001 (MT5 point). These are **not** optional “optimistic” costs.

### Features (shared defaults; hyp may freeze its own)
- ATR period **14**  
- Typical trade shape in many Hs: **stop_atr = 1.0**, **target_atr = 1.0** (1:1), horizon often **8** bars (or as frozen)  
- Other feature knobs (swing_n, large_candle_atr, etc.) live in settings but **must not be tuned after seeing results**

### Validation / statistics
| Parameter | Value | Role |
|---|---:|---|
| Train / val / OOS share (when not calendar-locked) | **60% / 20% / 20%** | Split design |
| **min_trades** | **100** | Train treatment n below this → NEEDS_MORE_DATA |
| **max_train_val_gap** | **0.20** (20 percentage points) | Larger gap → REJECT (overfit signature) |
| **significance_alpha** | **0.05** | Base α |
| **multiple_testing** | **Bonferroni** | Effective α = `0.05 / n_tests` in that run |
| Val n floor | `max(30, min_trades // 4)` | Else NEEDS_MORE_DATA |

### Locked calendar splits (do not move after seeing results)
Examples:
- **fx_h1:** train_end `2025-06-22`, val_end `2026-01-27`  
- **gold_m15:** train_end `2024-12-23`, val_end `2025-10-29`  
- **fx_m15 / gold_m5 / fx_m5:** frozen dates in settings  

OOS = everything after `val_end`. Default: **do not peek** (`--unlock-oos` only after a survivor).

### Retail path (not a fund)
- Min lot **0.01**: a small account **cannot** hold 1% risk on a 15-pip stop, so a first live step checks fills and spread, not returns  
- No return target is ever fitted in a backtest  

---

## 4. Decision gates (why the machine says REJECT / etc.)

Implemented in `src/ats/research/pipeline.py` → `decide()`:

### Automatic **REJECT** if
1. Train success rate **≤** baseline rate  
2. Train **p-value** not &lt; Bonferroni α (`0.05 / n_tests`)  
3. Train–val **gap** &gt; **20%**  
4. Validation success rate **≤** validation baseline  
5. (After unlock) OOS rate ≤ OOS baseline, or train–OOS gap &gt; 20%, or OOS n too small  

### Automatic **NEEDS_MORE_DATA** if
1. Train treatment **n &lt; 100**  
2. Validation **n** too small (`&lt; max(30, 25)`)  

### **CANDIDATE** (lab only) if
- Passes train + val gates above  
- Still **not** live; OOS locked; then last-level / owner decision  

### Human / protocol **REJECT** even if hit-rate looks good
- **Mean R ≤ 0** (you win often but lose money after costs) — “hit-rate mirage”  
- Last-level fail (e.g. Sunday fill spread kills H86)  
- OOS fail after unlock (e.g. H192)  
- No causal why / indicator fishing / reopen of a closed H  

### Permanent close rules
- Do **not** retune knobs after results  
- Do **not** flip long/short to “save” a test  
- Do **not** loosen filters to inflate n  
- Do **not** reopen REJECT because spreads got cheaper  

---

## 5. What we measure (report fields)

- Treatment **n**, success **rate** vs **baseline rate**  
- One-sided **p-value**  
- **Train / validation / OOS** blocks  
- **Train–val gap**  
- Notes: **mean R**, total R (when `r_mult` present)  
- Cost-adjusted success when available  

**Pass narrative we care about:** rate beats baseline **and** mean R &gt; 0 **and** stats clear **and** val holds **and** (later) OOS holds.

---

## 6. Why we reject “EMA 13/50/200 + S/R on M15” style ideas *before* coding

1. **No payer** — indicators are descriptions of price, not who must trade.  
2. **Already tested** — H13–H16 EMA family **REJECT**.  
3. **HARKing risk** — multi-condition stacks + TF choice after looking = overfitting.  
4. **Costs** — M15 FX/gold + many signals → spread bleeds expectancy.  

---

## 7. After a rare CANDIDATE

1. Last-level script (fill spread / path)  
2. Pre-commit OOS gates in writing  
3. `python -m ats test --id H… --unlock-oos`  
4. Fail → REJECT forever  
5. Pass → paper book only — still not “proven live profit”  

---

## 8. One-line summary

**We freeze a causal hypothesis, cost it, compare to a baseline on locked train/val with Bonferroni and a 20% gap rule, require enough trades and positive economics (R), keep OOS locked, and permanently close fails — including indicator folklore we already killed.**

*Config source: `config/settings.yaml`. Automated loop: `docs/LAB.md`. Book: `research/ledger.yaml`.*
