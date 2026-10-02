import numpy as np
import pandas as pd
from ats.hypotheses.execution import mirror_trades
from ats.timeutil import cost_price

def london_breakout_events(
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

    # Calculate Asian Range (00:00 to 07:00 UTC)
    # Since we are on H1, we can do a rolling max/min over the last 8 hours, 
    # but we ONLY sample it at hour == 7 (which closes at 08:00 UTC).
    work['asian_high'] = work['high'].rolling(8).max()
    work['asian_low'] = work['low'].rolling(8).min()
    
    # We forward fill the Asian high/low for the rest of the day until it resets.
    # A simple trick is to only keep values where hour == 7, then ffill.
    asian_h = work['asian_high'].where(work['time'].dt.hour == 7).ffill()
    asian_l = work['asian_low'].where(work['time'].dt.hour == 7).ffill()
    
    rows = []
    busy_until = -1
    for i in range(16, len(work) - 1):
        if i + 1 + horizon >= len(work) or i <= busy_until:
            continue
            
        current_time = work['time'].iloc[i]
        
        # Only trade the London Open candle (08:00 UTC candle, which closes at 09:00 UTC)
        # We check if it broke the asian range.
        if current_time.hour != 8:
            continue
            
        atr = float(work['atr'].iloc[i])
        if not np.isfinite(atr) or atr <= 0:
            continue
            
        c = float(work['close'].iloc[i])
        ah = float(asian_h.iloc[i])
        al = float(asian_l.iloc[i])
        
        break_up = c > ah
        break_down = c < al
        
        if not (break_up or break_down):
            continue
            
        side = "long" if break_up else "short"
        busy_until = i + horizon
        rows.extend(mirror_trades(work, i, side, symbol, cost, horizon, 1.0, 2.0))

    return pd.DataFrame(rows)
