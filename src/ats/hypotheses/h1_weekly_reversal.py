from __future__ import annotations

import numpy as np
import pandas as pd
from ats.hypotheses.execution import mirror_trades
from ats.timeutil import cost_price

def h1_weekly_reversal_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 1.2,
    slippage_pips: float = 0.5,
) -> pd.DataFrame:
    """
    H248: Friday Weekly Reversal (Profit Taking)
    Treatment: If the pair has trended > 2.0 ATR for the week by Friday 12:00 UTC, fade it into the close.
    """
    work = df.dropna(subset=["close"]).copy()
    if work.empty:
        return pd.DataFrame()
        
    work = work.reset_index(drop=True)
    
    if "atr" not in work.columns:
        tr1 = work["high"] - work["low"]
        tr2 = (work["high"] - work["close"].shift(1)).abs()
        tr3 = (work["low"] - work["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        work["atr"] = tr.rolling(14).mean()
        
    close = work["close"].to_numpy()
    high = work["high"].to_numpy()
    low = work["low"].to_numpy()
    atr = work["atr"].to_numpy()
    times = pd.Series(work["time"])
    
    # Identify Friday 12:00 UTC
    is_friday = times.dt.dayofweek == 4
    is_12 = times.dt.hour == 12
    trigger_bar = is_friday & is_12
    
    # Sunday-evening bars open the FX week, so shift 2h before taking the ISO week.
    iso = (times + pd.Timedelta(hours=2)).dt.isocalendar()
    week_key = iso["year"].astype(str) + "-" + iso["week"].astype(str)
    week_open = work["open"].groupby(week_key.to_numpy()).transform("first").to_numpy()
    lookback = 1
    
    trend_atr_mult = float(params.get("trend_atr_mult", 2.0))
    stop_mult = float(params.get("stop_atr", 1.0))
    tp_mult = float(params.get("target_atr", 1.0))
    hb = int(params.get("horizon_bars", 9)) # hold for 9 hours (until 21:00 Friday close)
    cost = cost_price(symbol, spread_pips, slippage_pips)
    
    trigger_arr = trigger_bar.to_numpy()
    
    rows = []
    n = len(work)
    
    busy_until = -1
    for t in range(lookback, n - hb - 1):
        if t <= busy_until:
            continue
        if not trigger_arr[t]:
            continue
            
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
            
        c = close[t]
        week_ret = c - week_open[t]
        
        extreme_bull = (week_ret > trend_atr_mult * a)
        extreme_bear = (week_ret < -trend_atr_mult * a)
        
        if not extreme_bull and not extreme_bear:
            continue
            
        entry = c
        direction = -1.0 if extreme_bull else 1.0
        
        side = "long" if direction == 1.0 else "short"
        rows.extend(mirror_trades(work, t, side, symbol, cost, hb, stop_mult, tp_mult))
        busy_until = t + hb
        
    return pd.DataFrame(rows)
