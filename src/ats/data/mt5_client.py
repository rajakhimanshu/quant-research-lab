from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from datetime import datetime, timezone

from ats.config import DATA_DIR, load_settings
from ats.timeutil import TIMEFRAME_MINUTES, years_ago


KNOWN_TERMINALS = [
    r"C:\Program Files\MetaTrader 5 IC Markets Global\terminal64.exe",
    r"D:\exness\terminal64.exe",
    r"C:\Program Files\MetaTrader 5\terminal64.exe",
]


def discover_terminals() -> list[str]:
    found = []
    for path in KNOWN_TERMINALS:
        if Path(path).exists():
            found.append(path)
    return found


def _mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError(
            "MetaTrader5 package missing. Use Python 3.12: "
            "py -3.12 -m venv .venv && .venv\\Scripts\\pip install -r requirements.txt"
        ) from exc
    return mt5


def terminal_path() -> str | None:
    env = os.getenv("MT5_TERMINAL_PATH", "").strip().strip('"')
    if env:
        return env
    settings = load_settings()
    configured = (settings.get("broker") or {}).get("terminal_path") or ""
    configured = str(configured).strip().strip('"')
    return configured or None


def connect() -> None:
    mt5 = _mt5()
    path = terminal_path()
    candidates = [path] if path else [None, *discover_terminals()]
    errors = []
    for candidate in candidates:
        kwargs: dict = {}
        if candidate:
            kwargs["path"] = candidate
        login = os.getenv("MT5_LOGIN", "").strip()
        password = os.getenv("MT5_PASSWORD", "").strip()
        server = os.getenv("MT5_SERVER", "").strip()
        if login and password and server:
            kwargs.update(login=int(login), password=password, server=server)
        if mt5.initialize(**kwargs):
            return
        errors.append(f"{candidate or 'default'}: {mt5.last_error()}")
        mt5.shutdown()
    found = discover_terminals()
    raise RuntimeError(
        "MT5 initialize failed. Open the terminal, log in, enable Algo Trading. "
        f"Tried: {errors}. Detected installs: {found or 'none'}. "
        "Set MT5_TERMINAL_PATH in .env to the terminal64.exe you actually use."
    )


def shutdown() -> None:
    _mt5().shutdown()


def mt5_timeframe(name: str):
    mt5 = _mt5()
    mapping = {
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported timeframe {name}")
    return mapping[name]


def resolve_symbol(requested: str) -> str:
    mt5 = _mt5()
    suffix = ((load_settings().get("broker") or {}).get("symbol_suffix") or "").strip()
    for name in (requested + suffix, requested):
        if mt5.symbol_select(name, True):
            info = mt5.symbol_info(name)
            if info is not None:
                return name
    symbols = mt5.symbols_get()
    if not symbols:
        raise RuntimeError(f"No MT5 symbols available: {mt5.last_error()}")
    upper = requested.upper()
    matches = [s.name for s in symbols if s.name.upper().startswith(upper)]
    if not matches:
        raise RuntimeError(f"Symbol {requested} not found in this MT5 terminal")
    name = matches[0]
    mt5.symbol_select(name, True)
    return name


def account_snapshot() -> dict:
    mt5 = _mt5()
    info = mt5.account_info()
    term = mt5.terminal_info()
    return {
        "connected": bool(info),
        "login": getattr(info, "login", None),
        "server": getattr(info, "server", None),
        "company": getattr(info, "company", None),
        "trade_allowed": getattr(term, "trade_allowed", None),
        "terminal_path": getattr(term, "path", None),
        "last_error": mt5.last_error(),
    }


def copy_ohlc(symbol: str, timeframe: str, years: int) -> pd.DataFrame:
    mt5 = _mt5()
    resolved = resolve_symbol(symbol)
    tf = mt5_timeframe(timeframe)
    utc_from = years_ago(years).astimezone(timezone.utc).replace(tzinfo=None)
    utc_to = datetime.now(timezone.utc).replace(tzinfo=None)
    minutes = TIMEFRAME_MINUTES[timeframe]
    want = int(years * 365 * 24 * 60 / minutes) + 2000
    info = mt5.terminal_info()
    maxbars = int(getattr(info, "maxbars", 0) or 0)
    if maxbars and want > maxbars:
        print(
            f"WARNING: {resolved} {timeframe} wants {want} bars but this terminal "
            f"maxbars={maxbars} (Tools > Options > Charts > Max bars in chart). "
            f"MT5 is not unlimited. Set Unlimited, restart MT5, open the {resolved} "
            f"chart and scroll left, then pull again. Capping the request at {maxbars}."
        )
        want = maxbars
    chunk = 20000
    frames = []
    off = 0
    while off < want:
        n = min(chunk, want - off)
        part = mt5.copy_rates_from_pos(resolved, tf, off, n)
        if part is None or len(part) == 0:
            break
        frames.append(pd.DataFrame(part))
        if len(part) < n:
            break
        off += len(part)
    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        rates = mt5.copy_rates_range(resolved, tf, utc_from, utc_to)
        if rates is None or len(rates) == 0:
            rates = mt5.copy_rates_from(resolved, tf, utc_from, min(want, 20000))
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No bars for {resolved} {timeframe}: {mt5.last_error()}")
        df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    if "tick_volume" in df.columns:
        df = df.rename(columns={"tick_volume": "volume"})
    keep = ["time", "open", "high", "low", "close", "volume", "spread"]
    df = df[[c for c in keep if c in df.columns]].drop_duplicates("time").sort_values("time")
    df["symbol"] = resolved
    df["requested_symbol"] = symbol
    df["timeframe"] = timeframe
    if maxbars and len(df) >= maxbars:
        print(
            f"WARNING: {resolved} {timeframe} hit the {maxbars}-bar terminal cap. "
            "This is MT5 Max bars in chart, not a Python limit and not 'unlimited'."
        )
    return df.reset_index(drop=True)


def save_raw(df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
    path = DATA_DIR / "raw" / f"{symbol}_{timeframe}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_raw(symbol: str, timeframe: str) -> pd.DataFrame:
    path = DATA_DIR / "raw" / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m ats pull --symbols {symbol} --timeframe {timeframe}"
        )
    return pd.read_parquet(path)
