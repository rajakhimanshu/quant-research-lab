import numpy as np
import pandas as pd
from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price

def strong_trend_events(
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

    work['sma50'] = close.rolling(50).mean()
    work['sma100'] = close.rolling(100).mean()
    work['sma200'] = close.rolling(200).mean()
    
    rows = []
    busy_until = -1
    for i in range(200, len(work) - 1):
        if i + 1 + horizon >= len(work) or i <= busy_until:
            continue
            
        atr = float(work['atr'].iloc[i])
        if not np.isfinite(atr) or atr <= 0:
            continue
            
        c_close = float(close.iloc[i])
        c_low = float(low.iloc[i])
        c_high = float(high.iloc[i])
        
        sma50 = float(work['sma50'].iloc[i])
        sma100 = float(work['sma100'].iloc[i])
        sma200 = float(work['sma200'].iloc[i])
        
        bull_stack = (sma50 > sma100) and (sma100 > sma200)
        bear_stack = (sma50 < sma100) and (sma100 < sma200)
        
        # Pullback to SMA 50
        bull_pullback = (c_low <= sma50) and (c_close > sma50)
        bear_pullback = (c_high >= sma50) and (c_close < sma50)
        
        treat = False
        side = None
        
        if bull_stack and bull_pullback:
            side = "long"
            treat = True
        elif bear_stack and bear_pullback:
            side = "short"
            treat = True
        else:
            # Baseline: Buying a pullback in a strong BEAR trend, or Shorting a pullback in a strong BULL trend.
            if bear_stack and bull_pullback:
                side = "long"
                treat = False
            elif bull_stack and bear_pullback:
                side = "short"
                treat = False
            else:
                continue
                
        raw_open = float(work.at[i + 1, "open"])
        
        if side == "long":
            entry = raw_open + cost
            stop = entry - (atr * 1.5)
            target = entry + (atr * 3.0)
        else:
            entry = raw_open - cost
            stop = entry + (atr * 1.5)
            target = entry - (atr * 3.0)
            
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
