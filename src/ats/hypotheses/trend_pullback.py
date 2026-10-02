import numpy as np
import pandas as pd
from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price

def trend_pullback_events(
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

    work['sma200'] = close.rolling(200).mean()
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    work['rsi'] = 100 - (100 / (1 + rs))
    
    rows = []
    busy_until = -1
    for i in range(200, len(work) - 1):
        if i + 1 + horizon >= len(work) or i <= busy_until:
            continue
            
        atr = float(work['atr'].iloc[i])
        if not np.isfinite(atr) or atr <= 0:
            continue
            
        c = float(work['close'].iloc[i])
        sma200 = float(work['sma200'].iloc[i])
        rsi = float(work['rsi'].iloc[i])
        
        if pd.isna(rsi) or pd.isna(sma200):
            continue
            
        side = "long" if c > sma200 else "short"
        
        if side == "long":
            treat = bool(rsi < 30)
            baseline = bool(45 < rsi < 55)
        else:
            treat = bool(rsi > 70)
            baseline = bool(45 < rsi < 55)
            
        if not (treat or baseline):
            continue
            
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
