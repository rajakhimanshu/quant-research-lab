import numpy as np
import pandas as pd
from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price

def engulfing_pullback_events(
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

    # Trend Filter: 200 SMA
    work['sma200'] = close.rolling(200).mean()
    
    # 24-hour low/high
    work['low24'] = low.rolling(24).min()
    work['high24'] = high.rolling(24).max()
    
    rows = []
    busy_until = -1
    for i in range(200, len(work) - 1):
        if i + 1 + horizon >= len(work) or i <= busy_until:
            continue
            
        atr = float(work['atr'].iloc[i])
        if not np.isfinite(atr) or atr <= 0:
            continue
            
        c_open = float(open_.iloc[i])
        c_close = float(close.iloc[i])
        c_high = float(high.iloc[i])
        c_low = float(low.iloc[i])
        
        p_open = float(open_.iloc[i-1])
        p_close = float(close.iloc[i-1])
        p_high = float(high.iloc[i-1])
        p_low = float(low.iloc[i-1])
        
        sma200 = float(work['sma200'].iloc[i])
        low24 = float(work['low24'].iloc[i])
        high24 = float(work['high24'].iloc[i])
        
        # Bullish Engulfing at a 24-hour low in an Uptrend
        bull_trend = c_close > sma200
        was_low24 = (p_low == low24) # Previous candle was the lowest of last 24h
        bear_candle = p_close < p_open
        bull_engulf = (c_close > p_open) and (c_open < p_close)
        
        # Bearish Engulfing at a 24-hour high in a Downtrend
        bear_trend = c_close < sma200
        was_high24 = (p_high == high24)
        bull_candle = p_close > p_open
        bear_engulf = (c_close < p_open) and (c_open > p_close)
        
        treat = False
        side = None
        
        if bull_trend and was_low24 and bear_candle and bull_engulf:
            side = "long"
            treat = True
        elif bear_trend and was_high24 and bull_candle and bear_engulf:
            side = "short"
            treat = True
        else:
            # Baseline: Just trading regular engulfing candles in the trend (not necessarily at 24h extremes)
            if bull_trend and bear_candle and bull_engulf:
                side = "long"
                treat = False
            elif bear_trend and bull_candle and bear_engulf:
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
