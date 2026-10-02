import numpy as np
import pandas as pd
from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price

def tuesday_turnaround_events(
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
    horizon = int(params.get("horizon_bars", 24)) # Hold for up to 24 hours (all of Tuesday)
    
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

    # 24-hour momentum
    work['mom_24'] = close - close.shift(24)
    
    rows = []
    for i in range(25, len(work) - 1):
        if i + 1 + horizon >= len(work):
            continue
            
        current_time = work['time'].iloc[i]
        
        # Tuesday 00:00 fades Monday; baseline is the same fade at Thursday 00:00.
        if current_time.hour != 0 or current_time.weekday() not in (1, 3):
            continue
        treat = current_time.weekday() == 1
            
        atr = float(work['atr'].iloc[i])
        if not np.isfinite(atr) or atr <= 0:
            continue
            
        mom = float(work['mom_24'].iloc[i])
        if mom == 0:
            continue
            
        # Fade Monday's move
        side = "short" if mom > 0 else "long"
        
        raw_open = float(work.at[i + 1, "open"])
        if side == "long":
            entry = raw_open + cost
            stop = entry - (atr * 1.5)
            target = entry + (atr * 2.0)
        else:
            entry = raw_open - cost
            stop = entry + (atr * 1.5)
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
        
    return pd.DataFrame(rows)
