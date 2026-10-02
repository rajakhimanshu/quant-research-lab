from __future__ import annotations

import numpy as np
import pandas as pd
from ats.hypotheses.execution import mirror_trades
from ats.timeutil import cost_price
from ats.config import DATA_DIR
import os

def cross_pair_lag_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 1.2,
    slippage_pips: float = 0.5,
) -> pd.DataFrame:
    """
    H237: Cross-pair EURUSD-GBPUSD lag trade
    Treatment: Follow EURUSD's strong 15m momentum 3 bars (15m) later on GBPUSD.
    Baseline: Fade EURUSD's momentum on GBPUSD.
    """
    if symbol != "GBPUSD":
        return pd.DataFrame()
        
    work = df.dropna(subset=["close"]).copy()
    if work.empty:
        return pd.DataFrame()
        
    eur_path = DATA_DIR / "raw" / f"EURUSD_{params.get('timeframe', 'M5')}.parquet"
    if not eur_path.exists():
        return pd.DataFrame()
    eur = pd.read_parquet(eur_path)
    
    eur = eur.set_index("time").dropna(subset=["close"])
    work = work.set_index("time")
    
    # 3-bar (15m) move on EURUSD
    move = eur["close"] - eur["close"].shift(3)
    
    # ATR(14) approximation on EURUSD
    high, low, close = eur["high"], eur["low"], eur["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    london_start = int(params.get("london_start", 8))
    london_end = int(params.get("london_end", 16))
    hr = eur.index.hour
    is_london = (hr >= london_start) & (hr < london_end)
    
    mom_atr = float(params.get("eur_momentum_atr", 1.5))
    
    eur_bull = (move > mom_atr * atr) & is_london
    eur_bear = (move < -mom_atr * atr) & is_london
    
    # Lag by 2-3 bars. Let's just enter 3 bars later (15 min lag)
    delay = 3
    eur_bull = eur_bull.shift(delay)
    eur_bear = eur_bear.shift(delay)
    
    # Join onto GBPUSD
    work = work.join(pd.DataFrame({"eur_bull": eur_bull, "eur_bear": eur_bear}), how="left")
    work["eur_bull"] = work["eur_bull"].fillna(False)
    work["eur_bear"] = work["eur_bear"].fillna(False)
    
    # Basic logic
    times = work.index.to_numpy()
    close_gbp = work["close"].to_numpy()
    high_gbp = work["high"].to_numpy()
    low_gbp = work["low"].to_numpy()
    
    work = work.reset_index()
    if "atr" not in work.columns:
        # crude atr if missing
        tr1 = work["high"] - work["low"]
        tr2 = (work["high"] - work["close"].shift(1)).abs()
        tr3 = (work["low"] - work["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        work["atr"] = tr.rolling(14).mean()
    atr_g = work["atr"].to_numpy()
        
    bull_mask = work["eur_bull"].to_numpy(dtype=bool)
    bear_mask = work["eur_bear"].to_numpy(dtype=bool)
    
    stop_mult = float(params.get("stop_atr", 1.0))
    tp_mult = float(params.get("target_atr", 1.0))
    hb = int(params.get("horizon_bars", 3))
    cost = cost_price("GBPUSD", spread_pips, slippage_pips)
    
    rows = []
    n = len(work)
    
    busy_until = -1
    for t in range(50, n - hb - 1):
        if t <= busy_until:
            continue
        is_bull = bull_mask[t]
        is_bear = bear_mask[t]
        if not is_bull and not is_bear:
            continue
            
        entry = close_gbp[t]
        a = atr_g[t]
        if not np.isfinite(a) or a <= 0:
            continue
            
        direction = 1.0 if is_bull else -1.0
        
        side = "long" if direction == 1.0 else "short"
        rows.extend(mirror_trades(work, t, side, symbol, cost, hb, stop_mult, tp_mult))
        busy_until = t + hb
        
    return pd.DataFrame(rows)
