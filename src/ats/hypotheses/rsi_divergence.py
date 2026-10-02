import numpy as np
import pandas as pd
from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price

def rsi_divergence_events(
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
    horizon = int(params.get("horizon_bars", 48))
    
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
    
    # Calculate rolling pivots (10 bars)
    work['lowest_10'] = low.rolling(10).min()
    work['highest_10'] = high.rolling(10).max()
    
    # Calculate RSI rolling pivots
    work['rsi_lowest_10'] = work['rsi'].rolling(10).min()
    work['rsi_highest_10'] = work['rsi'].rolling(10).max()
    
    rows = []
    busy_until = -1
    for i in range(20, len(work) - 1):
        if i + 1 + horizon >= len(work) or i <= busy_until:
            continue
            
        atr = float(work['atr'].iloc[i])
        if not np.isfinite(atr) or atr <= 0:
            continue
            
        c_close = float(close.iloc[i])
        c_low = float(low.iloc[i])
        c_high = float(high.iloc[i])
        c_rsi = float(work['rsi'].iloc[i])
        
        lowest_10 = float(work['lowest_10'].iloc[i])
        highest_10 = float(work['highest_10'].iloc[i])
        
        prev_lowest = float(work['lowest_10'].iloc[i-10])
        prev_highest = float(work['highest_10'].iloc[i-10])
        
        rsi_lowest = float(work['rsi_lowest_10'].iloc[i])
        rsi_highest = float(work['rsi_highest_10'].iloc[i])
        prev_rsi_lowest = float(work['rsi_lowest_10'].iloc[i-10])
        prev_rsi_highest = float(work['rsi_highest_10'].iloc[i-10])
        
        # Bullish Divergence: Price makes lower low, RSI makes higher low
        price_ll = (lowest_10 < prev_lowest) and (c_low == lowest_10)
        rsi_hl = (rsi_lowest > prev_rsi_lowest)
        bull_div = price_ll and rsi_hl
        
        # Bearish Divergence: Price makes higher high, RSI makes lower high
        price_hh = (highest_10 > prev_highest) and (c_high == highest_10)
        rsi_lh = (rsi_highest < prev_rsi_highest)
        bear_div = price_hh and rsi_lh
        
        treat = False
        side = None
        
        if bull_div:
            side = "long"
            treat = True
        elif bear_div:
            side = "short"
            treat = True
        else:
            # Baseline: Price makes lower low but RSI also makes lower low (No divergence)
            if price_ll and not rsi_hl:
                side = "long"
                treat = False
            elif price_hh and not rsi_lh:
                side = "short"
                treat = False
            else:
                continue
                
        raw_open = float(work.at[i + 1, "open"])
        
        if side == "long":
            entry = raw_open + cost
            stop = entry - (atr * 1.0)
            target = entry + (atr * 2.0)
        else:
            entry = raw_open - cost
            stop = entry + (atr * 1.0)
            target = entry - (atr * 2.0)
            
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
