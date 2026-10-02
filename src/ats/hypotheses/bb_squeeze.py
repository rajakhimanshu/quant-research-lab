import numpy as np
import pandas as pd
from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price

def bb_squeeze_events(
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
    if "atr" not in work.columns or work["atr"].isna().all():
        prev = close.shift(1)
        tr = pd.concat(
            [
                work["high"].astype(float) - work["low"].astype(float),
                (work["high"].astype(float) - prev).abs(),
                (work["low"].astype(float) - prev).abs(),
            ],
            axis=1,
        ).max(axis=1)
        work["atr"] = tr.rolling(14).mean()

    # Calculate Bollinger Bands (20, 2)
    work['sma20'] = close.rolling(20).mean()
    work['std20'] = close.rolling(20).std()
    work['upper'] = work['sma20'] + (work['std20'] * 2)
    work['lower'] = work['sma20'] - (work['std20'] * 2)
    work['bbw'] = (work['upper'] - work['lower']) / work['sma20']
    
    # Is BBW at a 20-period low?
    work['bbw_min20'] = work['bbw'].rolling(20).min()
    work['is_squeeze'] = work['bbw'] == work['bbw_min20']
    
    rows = []
    busy_until = -1
    for i in range(40, len(work) - 1):
        if i + 1 + horizon >= len(work) or i <= busy_until:
            continue
            
        atr = float(work['atr'].iloc[i])
        if not np.isfinite(atr) or atr <= 0:
            continue
            
        is_squeeze = bool(work['is_squeeze'].iloc[i-1]) # previous candle was squeeze
        
        c = float(work['close'].iloc[i])
        u = float(work['upper'].iloc[i])
        l = float(work['lower'].iloc[i])
        
        break_up = c > u
        break_down = c < l
        
        if not (break_up or break_down):
            continue
            
        side = "long" if break_up else "short"
        
        # Treatment: Breakout directly out of a volatility squeeze
        # Baseline: Breakout during normal volatility
        treat = is_squeeze
        
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
