from __future__ import annotations

import numpy as np
import pandas as pd
from ats.hypotheses.execution import mirror_trades
from ats.timeutil import cost_price

def asia_mean_reversion_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 1.2,
    slippage_pips: float = 0.5,
) -> pd.DataFrame:
    """
    H247: Asia Session Mean Reversion
    Treatment: Fade Bollinger Band breakouts during the low-liquidity Asia session (22:00 to 05:00 UTC).
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
    times = pd.Series(work["time"])
    
    # Calculate BB
    bb_window = int(params.get("bb_window", 20))
    bb_std = float(params.get("bb_std", 2.0))
    
    rolling_mean = work["close"].rolling(bb_window).mean()
    rolling_std = work["close"].rolling(bb_window).std()
    
    bb_upper = rolling_mean + (rolling_std * bb_std)
    bb_lower = rolling_mean - (rolling_std * bb_std)
    
    bbu = bb_upper.to_numpy()
    bbl = bb_lower.to_numpy()
    
    # Session filter
    asia_start = int(params.get("asia_start", 22))
    asia_end = int(params.get("asia_end", 5))
    
    hr = times.dt.hour
    is_asia = ((hr >= asia_start) | (hr <= asia_end)).to_numpy()
    
    stop_mult = float(params.get("stop_atr", 1.0))
    tp_mult = float(params.get("target_atr", 1.0))
    hb = int(params.get("horizon_bars", 12))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    
    rows = []
    n = len(work)
    
    busy_until = -1
    for t in range(max(bb_window, 14), n - hb - 1):
        if t <= busy_until:
            continue
        if not is_asia[t]:
            continue
            
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
            
        c = close[t]
        
        if not np.isfinite(bbu[t]):
            continue
            
        extreme_bull = (c < bbl[t])
        extreme_bear = (c > bbu[t])
        
        if not extreme_bull and not extreme_bear:
            continue
            
        entry = c
        direction = -1.0 if extreme_bear else 1.0
        
        side = "long" if direction == 1.0 else "short"
        rows.extend(mirror_trades(work, t, side, symbol, cost, hb, stop_mult, tp_mult))
        busy_until = t + hb
        
    return pd.DataFrame(rows)
