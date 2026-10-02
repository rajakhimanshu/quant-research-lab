import numpy as np
import pandas as pd
from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price

def bb_exhaustion_events(
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

    period = 20
    dev = 2.0
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + (std * dev)
    lower = sma - (std * dev)
    
    rows = []
    busy_until = -1
    for i in range(period+1, len(work) - 1):
        if i + 1 + horizon >= len(work) or i <= busy_until:
            continue
            
        atr = float(work['atr'].iloc[i])
        if not np.isfinite(atr) or atr <= 0:
            continue
            
        c_high = float(high.iloc[i])
        c_low = float(low.iloc[i])
        c_close = float(close.iloc[i])
        c_open = float(work['open'].iloc[i])
        
        b_up = float(upper.iloc[i])
        b_dn = float(lower.iloc[i])
        b_mid = float(sma.iloc[i])
        
        tag_upper = c_high > b_up
        tag_lower = c_low < b_dn
        
        if tag_upper and tag_lower:
            continue
        elif tag_upper:
            side = "short"
        elif tag_lower:
            side = "long"
        else:
            continue
            
        # Treatment: Closed back inside the band as a reversal candle (pinbar/engulfing logic)
        treat = False
        if side == "short" and c_close < b_up and c_close < c_open:
            treat = True
            stop_dist = c_high - c_close
        elif side == "long" and c_close > b_dn and c_close > c_open:
            treat = True
            stop_dist = c_close - c_low
        else:
            stop_dist = atr * 1.0 # default for baseline
            
        raw_open = float(work.at[i + 1, "open"])
        
        if side == "long":
            entry = raw_open + cost
            target = b_mid
            stop = entry - (atr * 0.75)
            if target <= entry + (atr * 0.75):
                continue
        else:
            entry = raw_open - cost
            target = b_mid
            stop = entry + (atr * 0.75)
            if target >= entry - (atr * 0.75):
                continue
            
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
