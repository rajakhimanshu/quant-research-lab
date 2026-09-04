"""Cross-market dislocation snapshot. Not a strategy until a why is frozen."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from ats.config import DATA_DIR
from ats.data.mt5_client import load_raw

OUT = DATA_DIR / "ideas"


def _daily_close(symbol: str, timeframe: str) -> pd.Series:
    df = load_raw(symbol, timeframe)
    t = pd.to_datetime(df["time"], utc=True)
    s = pd.Series(df["close"].to_numpy(), index=t, name=symbol)
    return s.resample("1D").last().dropna()


def _yf_close(ticker: str, start: str = "2022-01-01") -> pd.Series:
    import yfinance as yf

    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False, threads=False)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"no yfinance data for {ticker}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index, utc=True)
    return close.dropna().rename(ticker)


def snapshot() -> dict:
    gold = _daily_close("XAUUSD", "M15")
    try:
        dxy = _yf_close("DX-Y.NYB")
    except Exception:
        dxy = _yf_close("UUP")
    try:
        tnx = _yf_close("^TNX")
    except Exception:
        tnx = None
    jpy = None
    try:
        jpy = _daily_close("USDJPY", "H1")
    except FileNotFoundError:
        pass

    aligned = pd.concat([gold.rename("gold"), dxy.rename("dxy")], axis=1).dropna()
    aligned["corr_60"] = aligned["gold"].pct_change().rolling(60).corr(aligned["dxy"].pct_change())
    mu = aligned["corr_60"].rolling(252).mean()
    sd = aligned["corr_60"].rolling(252).std()
    aligned["corr_z"] = (aligned["corr_60"] - mu) / sd.replace(0, np.nan)
    last = aligned.dropna().iloc[-1]
    out = {
        "asof": str(aligned.dropna().index[-1].date()),
        "gold_dxy_corr_60": float(last["corr_60"]),
        "gold_dxy_corr_z": float(last["corr_z"]) if np.isfinite(last["corr_z"]) else None,
        "note": "z < -2 is a dislocation lead, not a trade. Freeze a why before testing.",
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
    }
    if tnx is not None and jpy is not None:
        yj = pd.concat([tnx.rename("tnx"), jpy.rename("usdjpy")], axis=1).dropna()
        yj["corr_60"] = yj["tnx"].pct_change().rolling(60).corr(yj["usdjpy"].pct_change())
        last_j = yj["corr_60"].dropna()
        if len(last_j):
            out["tnx_usdjpy_corr_60"] = float(last_j.iloc[-1])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cross_market.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
