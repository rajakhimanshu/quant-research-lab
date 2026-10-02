from __future__ import annotations

import numpy as np
import pandas as pd
from ats.hypotheses.execution import mirror_trades
from ats.timeutil import cost_price
from ats.config import DATA_DIR
import os

def eur_gbp_divergence_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 1.2,
    slippage_pips: float = 0.5,
) -> pd.DataFrame:
    """
    H242: EURUSD / GBPUSD Intraday Divergence Fade
    Treatment: If EURUSD moves up > X ATR and GBPUSD moves down > X ATR, fade the move.
    """
    if symbol not in ["EURUSD", "GBPUSD"]:
        return pd.DataFrame()
        
    work = df.dropna(subset=["close"]).copy()
    if work.empty:
        return pd.DataFrame()
        
    # Load the OTHER pair
    other_sym = "GBPUSD" if symbol == "EURUSD" else "EURUSD"
    tf = params.get("timeframe", "M5")
    other_path = DATA_DIR / "raw" / f"{other_sym}_{tf}.parquet"
    if not other_path.exists():
        return pd.DataFrame()
            
    other = pd.read_parquet(other_path).set_index("time").dropna(subset=["close"])
    work = work.set_index("time")
    
    div_bars = int(params.get("div_bars", 3))
    # n-bar move
    move_main = work["close"] - work["close"].shift(div_bars)
    move_other = other["close"] - other["close"].shift(div_bars)
    
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
    times = work["time"].to_numpy()
    
    m_main = work["move_main"].to_numpy()
    m_other = work["move_other"].to_numpy()
    
    stop_mult = float(params.get("stop_atr", 1.0))
    tp_mult = float(params.get("target_atr", 1.0))
    hb = int(params.get("horizon_bars", 6))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    
    rows = []
    n = len(work)
    
    busy_until = -1
    for t in range(14, n - hb - 1):
        if t <= busy_until:
            continue
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
            
        # Divergence condition
        mm = m_main[t]
        mo = m_other[t]
        
        if not np.isfinite(mm) or not np.isfinite(mo):
            continue
            
        # Main up, Other down
        div_bull = (mm > div_atr_mult * a) and (mo < -div_atr_mult * a)
        # Main down, Other up
        div_bear = (mm < -div_atr_mult * a) and (mo > div_atr_mult * a)
        
        if not div_bull and not div_bear:
            continue
            
        entry = close[t]
        
        # Treatment: Fade the main pair's move
        # If main went up (div_bull), we short (-1). If down (div_bear), we long (1).
        direction = -1.0 if div_bull else 1.0
        
        side = "long" if direction == 1.0 else "short"
        rows.extend(mirror_trades(work, t, side, symbol, cost, hb, stop_mult, tp_mult))
        busy_until = t + hb
        
    return pd.DataFrame(rows)
