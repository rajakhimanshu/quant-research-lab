from __future__ import annotations

import numpy as np
import pandas as pd
from ats.hypotheses.execution import mirror_trades
from ats.timeutil import cost_price

def three_bar_exhaustion_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 1.2,
    slippage_pips: float = 0.5,
) -> pd.DataFrame:
    """
    H239: Three-Bar Exhaustion Pullback
    Treatment: Fade (mean-revert) after 3 consecutive M5 bars in the same direction 
               where each body > body_atr_mult * ATR.
    """
    work = df.dropna(subset=["close"]).copy()
    if work.empty:
        return pd.DataFrame()
        
    work = work.reset_index(drop=True)
    
    if "atr" not in work.columns:
        # crude atr if missing
        tr1 = work["high"] - work["low"]
        tr2 = (work["high"] - work["close"].shift(1)).abs()
        tr3 = (work["low"] - work["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        work["atr"] = tr.rolling(14).mean()
        
    close = work["close"].to_numpy()
    open_px = work["open"].to_numpy()
    high = work["high"].to_numpy()
    low = work["low"].to_numpy()
    atr = work["atr"].to_numpy()
    times = work["time"].to_numpy()
    
    body = close - open_px
    body_abs = np.abs(body)
    
    body_mult = float(params.get("body_atr_mult", 0.8))
    
    # conditions
    bull_bar = (body > body_mult * atr)
    bear_bar = (body < -body_mult * atr)
    
    # 3 consecutive
    bull_3 = bull_bar & pd.Series(bull_bar).shift(1).fillna(False) & pd.Series(bull_bar).shift(2).fillna(False)
    bear_3 = bear_bar & pd.Series(bear_bar).shift(1).fillna(False) & pd.Series(bear_bar).shift(2).fillna(False)
    
    bull_mask = bull_3.to_numpy()
    bear_mask = bear_3.to_numpy()
    
    stop_mult = float(params.get("stop_atr", 1.0))
    tp_mult = float(params.get("target_atr", 1.0))
    hb = int(params.get("horizon_bars", 3))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    
    rows = []
    n = len(work)
    
    busy_until = -1
    for t in range(14, n - hb - 1):
        if t <= busy_until:
            continue
        is_bull = bull_mask[t]
        is_bear = bear_mask[t]
        if not is_bull and not is_bear:
            continue
            
        entry = close[t]
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
            
        # We fade the move. If bull_3, we short. If bear_3, we long.
        direction = -1.0 if is_bull else 1.0
        
        side = "long" if direction == 1.0 else "short"
        rows.extend(mirror_trades(work, t, side, symbol, cost, hb, stop_mult, tp_mult))
        busy_until = t + hb
        
    return pd.DataFrame(rows)
