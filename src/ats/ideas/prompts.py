"""Shared LLM brainstorm prompt templates — three causal categories.

These are imported by groq_brainstorm.py (live Groq calls) and by the
`brainstorm-prompt` CLI command (copy-paste fallback).
"""
from __future__ import annotations

_PROMPTS: dict[str, str] = {
    "forced_flow": """\
You are a market microstructure researcher helping design causal trading hypotheses
for FX (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD) and XAUUSD (gold).

Target: M5–M30 timeframes, short holding periods (5–30 bars), minimum 100 trades/month.
Feasibility: RETAIL MT5 only — no co-location, no tick-level L2 order book, no sub-second fills.
Lab discipline: every idea MUST state one sentence: "WHO is forced to trade and EXACTLY WHY."

Category: FORCED FLOW
Forced-flow mechanisms arise when a participant MUST trade regardless of price:
  - Fix benchmarks (WM 16:00 London, LBMA AM/PM, Tokyo 9:55)
  - Settlement / roll mechanics (tom-next, month-end rebalancing)
  - Post-news repositioning by slower institutional desks (5-30 min after a print)
  - Auction-adjacent drift (participants unable to participate in the auction trade after it)

Already tested and permanently REJECTED (do NOT suggest these):
  - WM 16:00 London follow (H27), Krohn postfix long (H28), pre-ECB short (H29)
  - LBMA PM 15:00 follow (H23), month-end MTD follow (H24), tom-next fade (H139-H141)
  - EIA Wednesday fade (H115-H117), NFP 8:00 fade (H80), FOMC 14:00 fade (H81)
  - Any EMA stack, VWAP, RSI, round-number, or volume-spike idea (all REJECT)
  - Any sub-second OFI, tick-level adverse selection, co-location strategy

Generate {n} ideas in this EXACT JSON array format (no extra text, no markdown, only JSON):
[
  {{
    "name": "Short descriptive hypothesis name",
    "causal_actor": "One sentence: who is forced to trade and exactly why they have no choice.",
    "instrument": "EURUSD",
    "timeframe": "M5",
    "expected_holding_bars": 12,
    "expected_trades_month": 80,
    "notes": "How this differs from already-tested ideas listed above."
  }}
]
""",

    "inventory_pressure": """\
You are a market microstructure researcher helping design causal trading hypotheses
for FX (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD) and XAUUSD (gold).

Target: M5–M30 timeframes, short holding periods (5–30 bars), minimum 100 trades/month.
Feasibility: RETAIL MT5 only — no co-location, no tick-level L2 order book, no sub-second fills.
Lab discipline: every idea MUST state one sentence: "WHO accumulates the inventory and HOW/WHEN they offload it."

Category: INVENTORY PRESSURE
Inventory-pressure mechanisms arise when a dealer or intermediary accumulates a directional
position they need to offload:
  - Cross-pair lead-lag (high-volume pair leads lower-volume pair by 1-3 M5 bars)
  - Post-sweep normalization (after a stop cascade, MM inventory reverts)
  - Consecutive one-sided retail-flow signal (tick volume as inventory proxy)
  - Cross-asset lead-lag (commodity → correlated currency within M5 bars)
  - Session-boundary inventory reduction before end-of-day risk limits

Already tested and permanently REJECTED (do NOT suggest these):
  - 5-bar streak fade (H25), 20-H1 stop-pool fade (H46), prior-day H/L fade (H17)
  - EURUSD 5-day TSMOM (H31), cross-sectional FX momentum (H38)
  - Any EMA stack, VWAP, RSI, round-number, or volume-spike idea (all REJECT)
  - Any sub-second OFI, tick-level adverse selection, co-location strategy

Generate {n} ideas in this EXACT JSON array format (no extra text, no markdown, only JSON):
[
  {{
    "name": "Short descriptive hypothesis name",
    "causal_actor": "One sentence: who holds the inventory and how/when they offload it.",
    "instrument": "GBPUSD",
    "timeframe": "M15",
    "expected_holding_bars": 8,
    "expected_trades_month": 120,
    "notes": "How this differs from already-tested ideas listed above."
  }}
]
""",

    "structural_liquidity": """\
You are a market microstructure researcher helping design causal trading hypotheses
for FX (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD) and XAUUSD (gold).

Target: M5–M30 timeframes, short holding periods (5–30 bars), minimum 100 trades/month.
Feasibility: RETAIL MT5 only — no co-location, no tick-level L2 order book, no sub-second fills.
Lab discipline: every idea MUST state one sentence: "WHAT structural clock event creates the liquidity vacuum and why."

Category: STRUCTURAL LIQUIDITY / CLOCK EFFECTS
These mechanisms arise from predictable, recurring points in the trading day when book depth
drops structurally (not because of news):
  - Session transitions (Asia → London, London → NY overlap)
  - Lunch breaks with thin books (Tokyo lunch 12:00-13:00 JST)
  - Pre-open thin periods before a major exchange cash open
  - Mid-session structural pauses that reduce active market-maker count

Already tested and permanently REJECTED (do NOT suggest these):
  - Tokyo lunch fade (H39), London lunch fade (H47), NY-close Asia fade (H40)
  - London 08:00 cash-open follow (H112-H114), all session-clock follow ideas (H54-H82)
  - Asia vs NY opening range breakout/fade (H4, H7)
  - Any EMA stack, VWAP, RSI, round-number, or volume-spike idea (all REJECT)
  - Any sub-second OFI, tick-level adverse selection, co-location strategy

Generate {n} ideas in this EXACT JSON array format (no extra text, no markdown, only JSON):
[
  {{
    "name": "Short descriptive hypothesis name",
    "causal_actor": "One sentence: what structural clock event causes the liquidity change and why.",
    "instrument": "USDJPY",
    "timeframe": "M5",
    "expected_holding_bars": 6,
    "expected_trades_month": 150,
    "notes": "How this differs from already-tested ideas listed above."
  }}
]
""",
}

CATEGORIES = list(_PROMPTS.keys())
