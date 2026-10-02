import numpy as np
import pandas as pd
from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price

def trend_continuation_events(
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

    work['sma200'] = close.rolling(200).mean()
    work['sma20'] = close.rolling(20).mean()
    
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
        
        sma200 = float(work['sma200'].iloc[i])
        sma20 = float(work['sma20'].iloc[i])
        
        # Pullback to SMA 20
        touched_sma20_bull = c_low <= sma20 and c_close > sma20
        touched_sma20_bear = c_high >= sma20 and c_close < sma20
        
        bull_trend = c_close > sma200
        bear_trend = c_close < sma200
        
        treat = False
        side = None
        
        if touched_sma20_bull:
            side = "long"
            treat = bull_trend # Treatment: with trend. Baseline: against trend
        elif touched_sma20_bear:
            side = "short"
            treat = bear_trend # Treatment: with trend. Baseline: against trend
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
