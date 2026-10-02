import numpy as np
import pandas as pd
from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price

def mtf_pullback_events(
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
    horizon = int(params.get("horizon_bars", 48)) # Hold for 48 hours
    
    close = work["close"].astype(float)
    high = work["high"].astype(float)
    low = work["low"].astype(float)
    open_ = work["open"].astype(float)
    
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

    # MTF Trend Filters (Approximated on H1)
    work['sma1200'] = close.rolling(1200).mean() # Daily 50 SMA
    work['sma80'] = close.rolling(80).mean()     # H4 20 SMA
    
    rows = []
    busy_until = -1
    for i in range(1200, len(work) - 1):
        if i + 1 + horizon >= len(work) or i <= busy_until:
            continue
            
        atr = float(work['atr'].iloc[i])
        if not np.isfinite(atr) or atr <= 0:
            continue
            
        c_close = float(close.iloc[i])
        c_open = float(open_.iloc[i])
        c_high = float(high.iloc[i])
        c_low = float(low.iloc[i])
        
        p_high = float(high.iloc[i-1])
        
        sma1200 = float(work['sma1200'].iloc[i])
        sma80 = float(work['sma80'].iloc[i])
        
        # Bullish MTF setup
        bull_daily = c_close > sma1200
        bull_h4 = c_close > sma80
        # Pullback: The low of this or previous candle tagged the H4 SMA, but we closed above it
        bull_pullback = (c_low < sma80) or (float(low.iloc[i-1]) < float(work['sma80'].iloc[i-1]))
        # Trigger: Strong bullish candle taking out previous high
        bull_trigger = (c_close > c_open) and (c_close > p_high)
        
        # Bearish MTF setup
        bear_daily = c_close < sma1200
        bear_h4 = c_close < sma80
        bear_pullback = (c_high > sma80) or (float(high.iloc[i-1]) > float(work['sma80'].iloc[i-1]))
        bear_trigger = (c_close < c_open) and (c_close < float(low.iloc[i-1]))
        
        treat = False
        side = None
        
        if bull_daily and bull_h4 and bull_pullback and bull_trigger:
            side = "long"
            treat = True
        elif bear_daily and bear_h4 and bear_pullback and bear_trigger:
            side = "short"
            treat = True
            
        # Baseline: Just taking the trigger in the direction of the daily trend, WITHOUT waiting for the H4 pullback
        if not treat:
            if bull_daily and bull_trigger:
                side = "long"
            elif bear_daily and bear_trigger:
                side = "short"
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
