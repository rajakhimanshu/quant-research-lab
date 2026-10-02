import numpy as np
import pandas as pd
from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price

def daily_engulfing_events(
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

    # Yesterday's completed UTC day (d_*) and the day before it (pd_*).
    day = pd.to_datetime(work["time"], utc=True).dt.floor("D")
    daily = work.groupby(day).agg(o=("open", "first"), c=("close", "last"), h=("high", "max"), l=("low", "min"))
    y, yy = daily.shift(1), daily.shift(2)
    work['d_close'] = day.map(y["c"]).to_numpy()
    work['d_open'] = day.map(y["o"]).to_numpy()
    work['d_high'] = day.map(y["h"]).to_numpy()
    work['d_low'] = day.map(y["l"]).to_numpy()
    work['pd_high'] = day.map(yy["h"]).to_numpy()
    work['pd_low'] = day.map(yy["l"]).to_numpy()
    
    # Calculate RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    work['rsi'] = 100 - (100 / (1 + rs))

    rows = []
    busy_until = -1
    for i in range(72, len(work) - 1):
        if i + 1 + horizon >= len(work) or i <= busy_until:
            continue
            
        atr = float(work['atr'].iloc[i])
        if not np.isfinite(atr) or atr <= 0:
            continue
            
        d_close = float(work['d_close'].iloc[i])
        d_open = float(work['d_open'].iloc[i])
        d_high = float(work['d_high'].iloc[i])
        d_low = float(work['d_low'].iloc[i])
        pd_high = float(work['pd_high'].iloc[i])
        pd_low = float(work['pd_low'].iloc[i])
        rsi = float(work['rsi'].iloc[i])
        
        # Bullish Engulfing Day
        bull_engulf = (d_close > d_open) and (d_high > pd_high) and (d_low < pd_low)
        bear_engulf = (d_close < d_open) and (d_high > pd_high) and (d_low < pd_low)
        
        treat = False
        side = None
        
        if bull_engulf:
            side = "long"
            treat = bool(rsi < 40) # Wait for pullback
        elif bear_engulf:
            side = "short"
            treat = bool(rsi > 60)
        else:
            continue
            
        # If not treat (meaning RSI didn't pull back), we still take the trade as baseline
            
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
        rows.append({
            "time": work.at[i + 1, "time"],
            "symbol": symbol,
            "treatment": treat,
            "side": side,
            "success": bool(hit),
            "success_cost_adj": bool(r_mult > 0),
            "r_mult": float(r_mult),
        })
        busy_until = i + horizon

    return pd.DataFrame(rows)
