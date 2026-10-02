import numpy as np
import pandas as pd
from ats.hypotheses.execution import mirror_trades
from ats.timeutil import cost_price

def rsi_extreme_fade_events(
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
    horizon = int(params.get("horizon_bars", 12)) 
    
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

    # Calculate RSI 14
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    work['rsi'] = 100 - (100 / (1 + rs))
    
    rows = []
    busy_until = -1
    for i in range(16, len(work) - 1):
        if i + 1 + horizon >= len(work) or i <= busy_until:
            continue
            
        atr = float(work['atr'].iloc[i])
        if not np.isfinite(atr) or atr <= 0:
            continue
            
        rsi = float(work['rsi'].iloc[i])
        if pd.isna(rsi):
            continue
            
        if rsi > 85:
            side = "short"
        elif rsi < 15:
            side = "long"
        else:
            continue
            
        rows.extend(mirror_trades(work, i, side, symbol, cost, horizon, 1.5, 1.5))
        busy_until = i + horizon

    return pd.DataFrame(rows)
