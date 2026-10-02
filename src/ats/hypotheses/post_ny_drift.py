from __future__ import annotations

import numpy as np
import pandas as pd
from ats.hypotheses.execution import mirror_trades
from ats.timeutil import cost_price

def post_ny_drift_reversion_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 1.2,
    slippage_pips: float = 0.5,
) -> pd.DataFrame:
    """
    H241: Post-NY Drift Reversion (Asian Open Re-centering)
    Treatment: If price moves > 0.5 ATR between 20:00 UTC and 23:55 UTC, fade it at 00:00 UTC.
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
    
    # Identify 00:00 UTC bars (or close to it if using M5)
    # Actually, we can just find the 00:00 bar, then look back to the 20:00 bar.
    # On M15, 20:00 to 00:00 is 16 bars. On M5, it's 48 bars.
    
    # We expect timeframe to be M15, so 16 bars lookback.
    is_midnight = (times.dt.hour == 0) & (times.dt.minute == 0)
    
    drift_atr_mult = float(params.get("drift_atr_mult", 0.5))
    stop_mult = float(params.get("stop_atr", 1.0))
    tp_mult = float(params.get("target_atr", 1.0))
    hb = int(params.get("horizon_bars", 12)) # 3 hours on M15
    cost = cost_price(symbol, spread_pips, slippage_pips)
    
    lookback = 16 # 4 hours on M15
    
    rows = []
    n = len(work)
    
    # Boolean array for midnight
    mid_arr = is_midnight.to_numpy()
    
    busy_until = -1
    for t in range(lookback, n - hb - 1):
        if t <= busy_until:
            continue
        if not mid_arr[t]:
            continue
            
        entry = close[t]
        start_px = close[t - lookback]
        drift = entry - start_px
        
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
            
        if abs(drift) < drift_atr_mult * a:
            continue
            
        # Fade the drift
        direction = -1.0 if drift > 0 else 1.0
        
        side = "long" if direction == 1.0 else "short"
        rows.extend(mirror_trades(work, t, side, symbol, cost, hb, stop_mult, tp_mult))
        busy_until = t + hb
        
    return pd.DataFrame(rows)
