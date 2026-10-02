from __future__ import annotations

import numpy as np
import pandas as pd
from ats.hypotheses.execution import mirror_trades
from ats.timeutil import cost_price
from ats.config import DATA_DIR

def div_filtered_rsi_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 1.2,
    slippage_pips: float = 0.5,
) -> pd.DataFrame:
    """
    H244: Divergence Filtered RSI Extreme
    Takes a classic RSI extreme (RSI > 70 or < 30) and applies the H242 
    cross-pair divergence filter to significantly boost win rate.
    """
    if symbol not in ["EURUSD", "GBPUSD"]:
        return pd.DataFrame()
        
    work = df.dropna(subset=["close"]).copy()
    if work.empty:
        return pd.DataFrame()
        
    # Load the OTHER pair for divergence filter
    other_sym = "GBPUSD" if symbol == "EURUSD" else "EURUSD"
    other_path = DATA_DIR / "raw" / f"{other_sym}_{params.get('timeframe', 'M5')}.parquet"
    if not other_path.exists():
        return pd.DataFrame()
        
    other = pd.read_parquet(other_path).set_index("time").dropna(subset=["close"])
    work = work.set_index("time")
    
    # 3-bar move
    move_main = work["close"] - work["close"].shift(3)
    move_other = other["close"] - other["close"].shift(3)
    
    # RSI on main pair
    delta = work["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Join
    joined = pd.DataFrame({"move_main": move_main, "move_other": move_other, "rsi": rsi})
    work = work.join(joined, rsuffix='_j')
    
    work = work.reset_index()
    if "atr" not in work.columns:
        tr1 = work["high"] - work["low"]
        tr2 = (work["high"] - work["close"].shift(1)).abs()
        tr3 = (work["low"] - work["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        work["atr"] = tr.rolling(14).mean()
        
    div_atr_mult = float(params.get("div_atr_mult", 0.5))
    
    close = work["close"].to_numpy()
    high = work["high"].to_numpy()
    low = work["low"].to_numpy()
    atr = work["atr"].to_numpy()
    times = pd.Series(work["time"])
    
    m_main = work["move_main"].to_numpy()
    m_other = work["move_other"].to_numpy()
    rsi_arr = work["rsi"].to_numpy()
    
    stop_mult = float(params.get("stop_atr", 1.0))
    tp_mult = float(params.get("target_atr", 1.0))
    hb = int(params.get("horizon_bars", 6))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    
    rsi_upper = float(params.get("rsi_upper", 70))
    rsi_lower = float(params.get("rsi_lower", 30))
    
    rows = []
    n = len(work)
    
    busy_until = -1
    for t in range(14, n - hb - 1):
        if t <= busy_until:
            continue
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
            
        r = rsi_arr[t]
        if not np.isfinite(r):
            continue
            
        # Is there an RSI extreme?
        extreme_bull = (r < rsi_lower)
        extreme_bear = (r > rsi_upper)
        
        if not extreme_bull and not extreme_bear:
            continue
            
        # Filter: Divergence condition
        mm = m_main[t]
        mo = m_other[t]
        
        if not np.isfinite(mm) or not np.isfinite(mo):
            continue
            
        # If extreme_bear (RSI > 70), we want to FADE it (short).
        # Divergence filter: Main pair MUST be up, and Other pair MUST be down.
        if extreme_bear:
            div_bear = (mm > div_atr_mult * a) and (mo < -div_atr_mult * a)
            if not div_bear:
                continue
            direction = -1.0 # fade
            
        # If extreme_bull (RSI < 30), we want to FADE it (long).
        # Divergence filter: Main pair MUST be down, and Other pair MUST be up.
        elif extreme_bull:
            div_bull = (mm < -div_atr_mult * a) and (mo > div_atr_mult * a)
            if not div_bull:
                continue
            direction = 1.0 # fade
            
        entry = close[t]
        
        side = "long" if direction == 1.0 else "short"
        rows.extend(mirror_trades(work, t, side, symbol, cost, hb, stop_mult, tp_mult))
        busy_until = t + hb
        
    return pd.DataFrame(rows)
