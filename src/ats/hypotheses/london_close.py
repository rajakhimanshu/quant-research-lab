import numpy as np
import pandas as pd
from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price

def london_close_events(
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

    # Calculate rolling daily High/Low over the last 15 hours
    work['day_high'] = high.rolling(15).max()
    work['day_low'] = low.rolling(15).min()
    work['day_range'] = work['day_high'] - work['day_low']
    
    rows = []
    busy_until = -1
    for i in range(15, len(work) - 1):
        if i + 1 + horizon >= len(work) or i <= busy_until:
            continue
            
        current_time = work['time'].iloc[i]
        
        # 15:00 UTC (10:00 AM NY or 11:00 AM NY) is the London Close Reversal window
        if current_time.hour != 15:
            continue
            
        atr = float(work['atr'].iloc[i])
        if not np.isfinite(atr) or atr <= 0:
            continue
            
        c_close = float(close.iloc[i])
        day_high = float(work['day_high'].iloc[i])
        day_low = float(work['day_low'].iloc[i])
        day_range = float(work['day_range'].iloc[i])
        
        if day_range < (atr * 1.5):
            continue # Only fade large trend days
            
        # Is price at the extreme?
        near_high = c_close >= (day_high - (day_range * 0.2))
        near_low = c_close <= (day_low + (day_range * 0.2))
        
        treat = False
        side = None
        
        if near_high:
            side = "short"
            treat = True
        elif near_low:
            side = "long"
            treat = True
            
        # Baseline: Fading the extreme at an arbitrary time (e.g., 08:00 UTC)
        # We can't do that easily here since we filter by hour == 15.
        # Let's use Baseline: trading WITH the trend at 15:00 UTC (Breakout continuation)
        if not treat:
            continue
            
        # Add baseline logic: if it's near the high, baseline traders buy the breakout.
        # We will set treatment = True for fading, False for following.
        # So we evaluate both.
        for t_side, is_treat in [(side, True), ("long" if side=="short" else "short", False)]:
            raw_open = float(work.at[i + 1, "open"])
            
            if t_side == "long":
                entry = raw_open + cost
                stop = entry - (atr * 1.0)
                target = entry + (atr * 1.5)
            else:
                entry = raw_open - cost
                stop = entry + (atr * 1.0)
                target = entry - (atr * 1.5)
                
            hit, r_mult = _forward(work, i, t_side, entry, target, stop, horizon)
            busy_until = i + horizon
            rows.append({
                "time": work.at[i + 1, "time"],
                "symbol": symbol,
                "treatment": is_treat,
                "side": t_side,
                "success": bool(hit),
                "success_cost_adj": bool(r_mult > 0),
                "r_mult": float(r_mult),
            })
        
    return pd.DataFrame(rows)
