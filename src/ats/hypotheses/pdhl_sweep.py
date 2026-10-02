import numpy as np
import pandas as pd
from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price

def pdhl_sweep_events(
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

    day = pd.to_datetime(work["time"], utc=True).dt.floor("D")
    prev_day = work.groupby(day).agg(h=("high", "max"), l=("low", "min")).shift(1)
    work['pdh'] = day.map(prev_day["h"]).to_numpy()
    work['pdl'] = day.map(prev_day["l"]).to_numpy()

    rows = []
    busy_until = -1
    for i in range(48, len(work) - 1):
        if i + 1 + horizon >= len(work) or i <= busy_until:
            continue
            
        atr = float(work['atr'].iloc[i])
        if not np.isfinite(atr) or atr <= 0:
            continue
            
        c_open = float(open_.iloc[i])
        c_close = float(close.iloc[i])
        c_high = float(high.iloc[i])
        c_low = float(low.iloc[i])
        
        pdh = float(work['pdh'].iloc[i])
        pdl = float(work['pdl'].iloc[i])
        
        if pd.isna(pdh) or pd.isna(pdl):
            continue
            
        # Treatment: Sweep of PDH/PDL and close back inside
        bull_sweep = (c_low < pdl) and (c_close > pdl)
        bear_sweep = (c_high > pdh) and (c_close < pdh)
        
        side = None
        treat = False
        
        if bull_sweep:
            side = "long"
            treat = True
            stop_dist = c_close - c_low
        elif bear_sweep:
            side = "short"
            treat = True
            stop_dist = c_high - c_close
        else:
            # Baseline: Buy if price is near PDL, Sell if price is near PDH (without sweep)
            if c_close > pdl and c_close < (pdl + atr):
                side = "long"
                treat = False
                stop_dist = atr * 1.0
            elif c_close < pdh and c_close > (pdh - atr):
                side = "short"
                treat = False
                stop_dist = atr * 1.0
            else:
                continue
                
        raw_open = float(work.at[i + 1, "open"])
        
        if side == "long":
            entry = raw_open + cost
            stop = entry - max(stop_dist, atr*0.5)
            target = entry + (atr * 1.5)
        else:
            entry = raw_open - cost
            stop = entry + max(stop_dist, atr*0.5)
            target = entry - (atr * 1.5)
            
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
