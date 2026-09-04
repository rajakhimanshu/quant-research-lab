"""H11: restore the usual gold–dollar inverse link after it temporarily breaks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ats.config import DATA_DIR
from ats.hypotheses.cot_spec_fade import _forward, _to_daily
from ats.timeutil import cost_price

DXY_CACHE = DATA_DIR / "ideas" / "dxy_daily.csv"


def _utc_day_index(idx) -> pd.DatetimeIndex:
    t = pd.to_datetime(idx, utc=True)
    return pd.DatetimeIndex(t).tz_convert("UTC").normalize()


def load_dxy(dxy: pd.Series | None = None, start: str = "2021-01-01") -> pd.Series:
    if dxy is not None:
        s = dxy.copy()
        s.index = _utc_day_index(s.index)
        return s.sort_index().rename("dxy")
    if DXY_CACHE.exists():
        raw = pd.read_csv(DXY_CACHE, parse_dates=["date"])
        s = pd.Series(raw["close"].to_numpy(), index=_utc_day_index(raw["date"]), name="dxy")
        return s.sort_index()
    from ats.ideas.cross_market import _yf_close

    try:
        s = _yf_close("DX-Y.NYB", start=start)
    except Exception:
        s = _yf_close("UUP", start=start)
    s.index = _utc_day_index(s.index)
    s = s.sort_index().rename("dxy")
    DXY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": s.index, "close": s.to_numpy()}).to_csv(DXY_CACHE, index=False)
    return s


def gold_dxy_relink_events(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    spread_pips: float = 240,
    slippage_pips: float = 50,
    dxy: pd.Series | None = None,
) -> pd.DataFrame:
    """Fade gold back to a negative dollar beta when 60d gold–DXY corr is abnormally high.

    treatment = corr_z >= extreme_z (link broken toward co-movement).
    baseline = |corr_z| <= mid_z (link near its own recent mean).
    Same dollar-implied side in both arms.
    """
    if df.empty:
        return pd.DataFrame()
    try:
        dx = load_dxy(dxy)
    except Exception:
        return pd.DataFrame()
    if dx.empty:
        return pd.DataFrame()

    corr_n = int(params.get("corr_window", 60))
    z_n = int(params.get("z_window", 252))
    extreme_z = float(params.get("extreme_z", 1.0))
    mid_z = float(params.get("mid_z", 0.35))
    look = int(params.get("dxy_lookback_days", 5))
    horizon = int(params.get("horizon_days", 5))
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    cost = cost_price(symbol, spread_pips, slippage_pips)

    daily = _to_daily(df)
    if daily.empty:
        return pd.DataFrame()
    work = daily.copy()
    work.index = _utc_day_index(work.index)
    work = work[~work.index.duplicated(keep="last")].sort_index()
    dx = dx[~dx.index.duplicated(keep="last")]
    work["dxy"] = dx.reindex(work.index).ffill()
    work = work.dropna(subset=["dxy", "atr"])
    work["g_ret"] = work["close"].pct_change()
    work["d_ret"] = work["dxy"].pct_change()
    work["corr_60"] = work["g_ret"].rolling(corr_n).corr(work["d_ret"])
    mu = work["corr_60"].rolling(z_n).mean()
    sd = work["corr_60"].rolling(z_n).std().replace(0, np.nan)
    work["corr_z"] = (work["corr_60"] - mu) / sd
    work["dxy_n"] = work["dxy"] / work["dxy"].shift(look) - 1.0

    rows = []
    n = len(work)
    for i in range(n - horizon - 2):
        z = work["corr_z"].iloc[i]
        dxy_n = work["dxy_n"].iloc[i]
        if not np.isfinite(z) or not np.isfinite(dxy_n) or dxy_n == 0:
            continue
        extreme = float(z) >= extreme_z
        mid = abs(float(z)) <= mid_z
        if not extreme and not mid:
            continue
        loc = i + 1
        if loc + horizon >= n:
            continue
        atr = float(work["atr"].iloc[loc])
        if not np.isfinite(atr) or atr <= 0:
            continue
        side = "short" if float(dxy_n) > 0 else "long"
        raw_open = float(work["open"].iloc[loc])
        if side == "long":
            entry = raw_open + cost
            stop = entry - stop_atr * atr
            target = entry + target_atr * atr
        else:
            entry = raw_open - cost
            stop = entry + stop_atr * atr
            target = entry - target_atr * atr
        hit, r_mult = _forward(work, loc, side, entry, target, stop, horizon)
        rows.append(
            {
                "time": work.index[loc],
                "symbol": symbol,
                "treatment": bool(extreme),
                "side": side,
                "success": bool(hit),
                "success_cost_adj": bool(r_mult > 0),
                "r_mult": float(r_mult),
                "corr_z": float(z),
            }
        )
    return pd.DataFrame(rows)
