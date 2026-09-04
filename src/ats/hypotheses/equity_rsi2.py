from __future__ import annotations

import numpy as np
import pandas as pd


from ats.config import DATA_DIR

CACHE = DATA_DIR / "equity"


def rsi_wilder(close: pd.Series, length: int = 2) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    ag = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    al = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = ag / al.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.where(al != 0, 100.0)
    out = out.where(ag != 0, 0.0)
    return out


def atr(df: pd.DataFrame, length: int = 20) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _simulate(df: pd.DataFrame, params: dict, require_uptrend: bool) -> pd.DataFrame:
    """Match the frozen equity_rsi2_mean_reversion rules: signal on close, fill next open.

    Exit when prior close is back above SMA(5) or RSI(2) > exit, or max hold —
    not when price is still below SMA(5).
    """
    rsi_in = float(params.get("rsi_entry", 10))
    rsi_out = float(params.get("rsi_exit", 65))
    trend_n = int(params.get("trend_sma", 200))
    exit_n = int(params.get("exit_sma", 5))
    max_hold = int(params.get("max_hold", 10))
    atr_stop = float(params.get("atr_stop", 2.0))
    cost = float(params.get("cost_one_way", 0.00015))
    out = df.copy()
    out["rsi2"] = rsi_wilder(out["Close"], 2)
    out["sma_trend"] = out["Close"].rolling(trend_n).mean()
    out["sma_exit"] = out["Close"].rolling(exit_n).mean()
    out["atr"] = atr(out, 20)
    rows = []
    pos = None
    n = len(out)
    for i in range(1, n):
        prev = out.iloc[i - 1]
        row = out.iloc[i]
        if pos is not None:
            hold = i - pos["entry_i"]
            exit_signal = (
                (float(prev["Close"]) > float(prev["sma_exit"]))
                or (float(prev["rsi2"]) > rsi_out)
                or (hold >= max_hold)
            )
            hit_stop = float(row["Low"]) <= pos["stop"]
            if hit_stop:
                exit_px = pos["stop"] * (1 - cost)
            elif exit_signal:
                exit_px = float(row["Open"]) * (1 - cost)
            else:
                continue
            r_mult = (exit_px - pos["entry"]) / pos["risk"]
            rows.append(
                {
                    "time": pos["time"],
                    "exit_time": out.index[i],
                    "success": r_mult > 0,
                    "r_mult": r_mult,
                    "hold": hold,
                }
            )
            pos = None
            continue
        rsi = float(prev["rsi2"])
        close = float(prev["Close"])
        trend = float(prev["sma_trend"])
        atr_i = float(prev["atr"])
        if not (np.isfinite(rsi) and np.isfinite(trend) and np.isfinite(atr_i) and atr_i > 0):
            continue
        if rsi >= rsi_in:
            continue
        above = close > trend
        if require_uptrend and not above:
            continue
        if (not require_uptrend) and above:
            continue
        entry = float(row["Open"]) * (1 + cost)
        risk = atr_stop * atr_i
        pos = {
            "entry_i": i,
            "time": out.index[i],
            "entry": entry,
            "stop": entry - risk,
            "risk": max(risk, 1e-9),
        }
    if pos is not None:
        last = out.iloc[-1]
        exit_px = float(last["Close"]) * (1 - cost)
        r_mult = (exit_px - pos["entry"]) / pos["risk"]
        rows.append(
            {
                "time": pos["time"],
                "exit_time": out.index[-1],
                "success": r_mult > 0,
                "r_mult": r_mult,
                "hold": n - 1 - pos["entry_i"],
            }
        )
    return pd.DataFrame(rows)


CACHE = DATA_DIR / "equity"


def _load_ticker(ticker: str) -> pd.DataFrame | None:
    path = CACHE / f"{ticker.lower()}_daily.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    return None


def _download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    data = {}
    missing = []
    for t in tickers:
        local = _load_ticker(t)
        if local is not None and len(local) >= 300:
            data[t] = local
        else:
            missing.append(t)
    if not missing:
        return data
    import yfinance as yf

    raw = yf.download(
        missing,
        start="2010-01-01",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    for t in missing:
        piece = raw
        if isinstance(raw.columns, pd.MultiIndex):
            level = 0 if t in raw.columns.get_level_values(0) else 1
            try:
                piece = raw.xs(t, axis=1, level=level)
            except Exception:
                continue
        piece = piece.dropna(subset=["Close"])
        if len(piece) < 300:
            continue
        piece.index = pd.to_datetime(piece.index, utc=True)
        data[t] = piece
    return data


def load_equity_frames(params: dict, refresh: bool = False) -> dict[str, pd.DataFrame]:
    tickers = list(params.get("tickers") or ["SPY", "QQQ", "IWM", "DIA", "EEM", "XLK", "XLF", "GLD", "EFA", "XLE"])
    if refresh:
        missing = list(tickers)
        # force download path by skipping cache length check via temp
        data = {}
        import yfinance as yf

        raw = yf.download(
            missing,
            start="2010-01-01",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        for t in missing:
            piece = raw
            if isinstance(raw.columns, pd.MultiIndex):
                level = 0 if t in raw.columns.get_level_values(0) else 1
                try:
                    piece = raw.xs(t, axis=1, level=level)
                except Exception:
                    local = _load_ticker(t)
                    if local is not None:
                        data[t] = local
                    continue
            piece = piece.dropna(subset=["Close"])
            if len(piece) < 300:
                continue
            piece.index = pd.to_datetime(piece.index, utc=True)
            CACHE.mkdir(parents=True, exist_ok=True)
            out = piece.copy()
            if "Date" not in out.columns:
                out.index.name = "Date"
                out.to_csv(CACHE / f"{t.lower()}_daily.csv")
            data[t] = piece
        return data
    return _download(tickers)


def equity_rsi2_events(params: dict) -> pd.DataFrame:
    tickers = list(params.get("tickers") or ["SPY", "QQQ", "IWM", "DIA", "EEM", "XLK", "XLF", "GLD", "EFA", "XLE"])
    frames = _download(tickers)
    chunks = []
    for ticker, df in frames.items():
        treated = _simulate(df, params, require_uptrend=True)
        base = _simulate(df, params, require_uptrend=False)
        if not treated.empty:
            treated = treated.copy()
            treated["treatment"] = True
            treated["symbol"] = ticker
            treated["success_cost_adj"] = treated["success"]
            chunks.append(treated)
        if not base.empty:
            base = base.copy()
            base["treatment"] = False
            base["symbol"] = ticker
            base["success_cost_adj"] = base["success"]
            chunks.append(base)
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    return out.sort_values("time")
