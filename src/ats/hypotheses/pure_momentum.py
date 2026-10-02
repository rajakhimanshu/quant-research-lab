import numpy as np
import pandas as pd
from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price

def pure_momentum_events(
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

    # Daily Trend Filter (1200 H1 bars = 50 Days)
    work['sma1200'] = close.rolling(1200).mean()
    
    # 12-hour high/low breakout (Donchian channel)
    work['high12'] = high.rolling(12).max().shift(1)
    work['low12'] = low.rolling(12).min().shift(1)
    
    rows = []
    busy_until = -1
    for i in range(1200, len(work) - 1):
        if i + 1 + horizon >= len(work) or i <= busy_until:
            continue
            
        atr = float(work['atr'].iloc[i])
        if not np.isfinite(atr) or atr <= 0:
            continue
            
        c_close = float(close.iloc[i])
        sma1200 = float(work['sma1200'].iloc[i])
        high12 = float(work['high12'].iloc[i])
        low12 = float(work['low12'].iloc[i])
        
        bull_break = c_close > high12
        bear_break = c_close < low12
        
        bull_trend = c_close > sma1200
        bear_trend = c_close < sma1200
        
        if not (bull_break or bear_break):
            continue
            
        if bull_break:
            side = "long"
            treat = bull_trend # Treatment: with trend. Baseline: against trend
        else:
            side = "short"
            treat = bear_trend # Treatment: with trend. Baseline: against trend
            
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
