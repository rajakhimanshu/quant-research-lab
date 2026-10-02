from __future__ import annotations

import numpy as np
import pandas as pd
from ats.hypotheses.execution import mirror_trades
from ats.timeutil import cost_price

def stop_cascade_absorption_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 1.2,
    slippage_pips: float = 0.5,
) -> pd.DataFrame:
    """
    H240: Stop Cascade Absorption (MECH_001)
    Treatment: Fade when price breaks a prominent swing high/low but immediately 
               reverses (closes back below/above the swept level).
    """
    work = df.dropna(subset=["close"]).copy()
    if work.empty:
        return pd.DataFrame()
        
    work = work.reset_index(drop=True)
    
    if "atr" not in work.columns:
        tr1 = work["high"] - work["low"]
        tr2 = (work["high"] - work["close"].shift(1)).abs()
        tr3 = (work["low"] - work["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        work["atr"] = tr.rolling(14).mean()
        
    close = work["close"].to_numpy()
    high = work["high"].to_numpy()
    low = work["low"].to_numpy()
    atr = work["atr"].to_numpy()
    times = work["time"].to_numpy()
    
    lookback = int(params.get("swing_lookback", 20))
    min_bars = int(params.get("min_bars_since", 5))
    
    # prominent high/low over lookback
    roll_high = work["high"].rolling(lookback, min_periods=lookback).max().shift(1).to_numpy()
    roll_low = work["low"].rolling(lookback, min_periods=lookback).min().shift(1).to_numpy()
    
    # We trigger if the CURRENT bar's high > roll_high, but CLOSE < roll_high (a sweep and fail)
    # AND the sweep was not too large (e.g. < 1.0 ATR) so it's a stop hunt, not a real breakout.
    # We also check that the swept level was formed at least min_bars ago, so it's a real swing.
    
    stop_mult = float(params.get("stop_atr", 1.0))
    tp_mult = float(params.get("target_atr", 1.0))
    hb = int(params.get("horizon_bars", 5))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    
    rows = []
    n = len(work)
    
    busy_until = -1
    for t in range(lookback, n - hb - 1):
        if t <= busy_until:
            continue
        h = high[t]
        l = low[t]
        c = close[t]
        rh = roll_high[t]
        rl = roll_low[t]
        a = atr[t]
        
        if not np.isfinite(rh) or not np.isfinite(a) or a <= 0:
            continue
            
        is_bull_sweep = (h > rh) and (c < rh) and ((h - rh) < 1.0 * a) and ((h - rh) > 0.1 * a)
        is_bear_sweep = (l < rl) and (c > rl) and ((rl - l) < 1.0 * a) and ((rl - l) > 0.1 * a)
        
        if not is_bull_sweep and not is_bear_sweep:
            continue
            
        # Optional: verify that the high/low was from > min_bars ago (not just sweeping yesterday's bar).
        # We can approximate by checking if the previous min_bars were all below rh.
        if is_bull_sweep:
            recent_highs = high[t-min_bars:t]
            if np.any(recent_highs >= rh):
                continue
                
        if is_bear_sweep:
            recent_lows = low[t-min_bars:t]
            if np.any(recent_lows <= rl):
                continue
                
        entry = c
        direction = -1.0 if is_bull_sweep else 1.0
        
        side = "long" if direction == 1.0 else "short"
        rows.extend(mirror_trades(work, t, side, symbol, cost, hb, stop_mult, tp_mult))
        busy_until = t + hb
        
    return pd.DataFrame(rows)
