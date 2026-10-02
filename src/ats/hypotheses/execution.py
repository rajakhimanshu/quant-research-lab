"""Shared fill model for intraday event modules.

Signal on bar i close, fill at bar i+1 open plus spread+slippage, ATR stop and
target from that fill, fixed horizon, R net of cost (via m15_micro._trade).
"""

from __future__ import annotations

import pandas as pd

from ats.hypotheses.m15_micro import _trade


def mirror_trades(
    work: pd.DataFrame,
    i: int,
    side: str,
    symbol: str,
    cost: float,
    horizon: int,
    stop_atr: float,
    target_atr: float,
) -> list[dict]:
    """Treatment trade on `side`; baseline is the same bar traded the other way."""
    other = "short" if side == "long" else "long"
    treat = _trade(work, i, side, True, symbol, cost, horizon, stop_atr, target_atr)
    base = _trade(work, i, other, False, symbol, cost, horizon, stop_atr, target_atr)
    if treat is None or base is None:
        return []
    return [treat, base]
