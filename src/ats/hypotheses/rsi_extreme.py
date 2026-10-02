import numpy as np
import pandas as pd
from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price

def rsi_extreme_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 240,
    slippage_pips: float = 50,
) -> pd.DataFrame:
    work = df.dropna(subset=["close"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
        
    cost = cost_price(symbol, spread_pips, slippage_pips)
    horizon = int(params.get("horizon_bars", 24))
    
    close = work["close"].astype(float)
    high = work["high"].astype(float)
    low = work["low"].astype(float)
    
    if "atr" not in work.columns or work["atr"].isna().all():
        prev = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev).abs(),
                (low - prev).abs(),
            ],
            axis=1,
        ).max(axis=1)
        work["atr"] = tr.rolling(14).mean()

    # Calculate RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    work['rsi'] = 100 - (100 / (1 + rs))
    
    rows = []
    busy_until = -1
    for i in range(20, len(work) - 1):
        if i + 1 + horizon >= len(work) or i <= busy_until:
            continue
            
        atr = float(work['atr'].iloc[i])
        if not np.isfinite(atr) or atr <= 0:
            continue
            
        c_rsi = float(work['rsi'].iloc[i])
        if pd.isna(c_rsi):
            continue
            
        bull_extreme = c_rsi < 20
        bear_extreme = c_rsi > 80
        
        treat = False
        side = None
        
        if bull_extreme:
            side = "long"
            treat = True
        elif bear_extreme:
            side = "short"
            treat = True
        else:
            # Baseline: Just buying RSI < 40
            if c_rsi < 40 and c_rsi >= 20:
                side = "long"
                treat = False
            elif c_rsi > 60 and c_rsi <= 80:
                side = "short"
                treat = False
            else:
                continue
                
        raw_open = float(work.at[i + 1, "open"])
        
        if side == "long":
            entry = raw_open + cost
            stop = entry - (atr * 1.5)
            target = entry + (atr * 1.5)
        else:
            entry = raw_open - cost
            stop = entry + (atr * 1.5)
            target = entry - (atr * 1.5)
            
        hit, r_mult = _forward(work, i, side, entry, target, stop, horizon)
        busy_until = i + horizon
        rows.append({
            "time": work.at[i + 1, "time"],
            "symbol": symbol,
            "treatment": treat,
            "side": side,
            "success": bool(hit),
            "success_cost_adj": bool(r_mult > 0),
            "r_mult": float(r_mult),
        })
        
    return pd.DataFrame(rows)
