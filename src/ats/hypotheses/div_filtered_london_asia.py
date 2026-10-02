from __future__ import annotations

import numpy as np
import pandas as pd
from ats.hypotheses.execution import mirror_trades
from ats.timeutil import cost_price
from ats.config import DATA_DIR
import os

def div_filtered_london_asia_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 1.2,
    slippage_pips: float = 0.5,
) -> pd.DataFrame:
    """
    H243: Divergence Filtered London/Asia Breakout Fade
    Takes the H30 (London break of Asia range) setup and applies the H242 
    cross-pair divergence filter. If EURUSD breaks the Asia High but GBPUSD is down, 
    we fade the EURUSD breakout (and vice versa).
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
    
    # 3-bar (15m on M5) move
    move_main = work["close"] - work["close"].shift(3)
    move_other = other["close"] - other["close"].shift(3)
    
    # Join
    joined = pd.DataFrame({"move_main": move_main, "move_other": move_other})
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
    
    # Asia Range logic
    asia_start = int(params.get("asia_start", 0))
    asia_end = int(params.get("asia_end", 7))
    london_start = int(params.get("london_start", 8))
    london_end = int(params.get("london_end", 11))
    
    hr = times.dt.hour
    
    # Track daily Asia high/low
    asia_highs = pd.Series(np.nan, index=work.index)
    asia_lows = pd.Series(np.nan, index=work.index)
    
    is_asia = (hr >= asia_start) & (hr <= asia_end)
    # Get daily max/min during asia session
    asia_h = work["high"].where(is_asia).groupby(times.dt.date).transform('max')
    asia_l = work["low"].where(is_asia).groupby(times.dt.date).transform('min')
    
    stop_mult = float(params.get("stop_atr", 1.0))
    tp_mult = float(params.get("target_atr", 1.0))
    hb = int(params.get("horizon_bars", 12))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    
    ah_arr = asia_h.to_numpy()
    al_arr = asia_l.to_numpy()
    is_london = ((hr >= london_start) & (hr <= london_end)).to_numpy()
    
    rows = []
    n = len(work)
    
    # We only take the FIRST breakout per day.
    last_trade_day = None
    
    busy_until = -1
    for t in range(14, n - hb - 1):
        if t <= busy_until:
            continue
        if not is_london[t]:
            continue
            
        cur_day = times.iloc[t].date()
        if cur_day == last_trade_day:
            continue
            
        ah = ah_arr[t]
        al = al_arr[t]
        if pd.isna(ah) or pd.isna(al):
            continue
            
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
            
        c = close[t]
        h = high[t]
        l = low[t]
        
        # Breakout condition (Price touches beyond Asia High/Low)
        break_up = h > ah
        break_dn = l < al
        
        if not break_up and not break_dn:
            continue
            
        # Filter: Divergence condition
        mm = m_main[t]
        mo = m_other[t]
        
        if not np.isfinite(mm) or not np.isfinite(mo):
            continue
            
        # If breaking UP, we want to FADE it (short). 
        # Divergence filter: Main pair MUST be up, and Other pair MUST be down.
        if break_up:
            div_bull = (mm > div_atr_mult * a) and (mo < -div_atr_mult * a)
            if not div_bull:
                continue
            direction = -1.0 # fade
            
        # If breaking DOWN, we want to FADE it (long).
        # Divergence filter: Main pair MUST be down, and Other pair MUST be up.
        elif break_dn:
            div_bear = (mm < -div_atr_mult * a) and (mo > div_atr_mult * a)
            if not div_bear:
                continue
            direction = 1.0 # fade
            
        last_trade_day = cur_day
        entry = close[t]
        
        side = "long" if direction == 1.0 else "short"
        rows.extend(mirror_trades(work, t, side, symbol, cost, hb, stop_mult, tp_mult))
        busy_until = t + hb
        
    return pd.DataFrame(rows)
