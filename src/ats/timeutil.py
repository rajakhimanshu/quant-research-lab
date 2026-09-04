from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

UTC = ZoneInfo("UTC")
IST = ZoneInfo("Asia/Kolkata")

TIMEFRAME_MINUTES = {
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


def parse_hhmm(value: str) -> time:
    h, m = value.split(":")
    return time(int(h), int(m))


def time_in_window(t: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= t < end
    return t >= start or t < end


def to_utc_index(series: pd.Series) -> pd.DatetimeIndex:
    idx = pd.to_datetime(series, utc=True)
    return pd.DatetimeIndex(idx)


@dataclass(frozen=True)
class Split:
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    oos_start: pd.Timestamp


def time_splits(index: pd.DatetimeIndex, train: float, validation: float) -> Split:
    if not index.is_monotonic_increasing:
        index = index.sort_values()
    n = len(index)
    i_train = int(n * train)
    i_val = int(n * (train + validation))
    return Split(
        train_end=index[max(i_train - 1, 0)],
        val_end=index[max(i_val - 1, 0)],
        oos_start=index[min(i_val, n - 1)],
    )


def _ts_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def locked_calendar_split(settings: dict, book: str) -> Split | None:
    """Frozen calendar cuts. Returns None if splits are not locked."""
    spec = (settings.get("splits") or {}).get(book)
    if not spec or not (settings.get("splits") or {}).get("locked"):
        return None
    train_end = _ts_utc(spec["train_end"])
    val_end = _ts_utc(spec["val_end"])
    return Split(train_end=train_end, val_end=val_end, oos_start=val_end)


def hyp_key(hypothesis_id: str) -> str:
    """Numeric id only. Do not use startswith('H1_') — H17 would match H1."""
    return hypothesis_id.split("_", 1)[0].upper()


def split_book_for(hypothesis_id: str) -> str:
    key = hyp_key(hypothesis_id)
    if key == "H5":
        return "equity_daily"
    if key in {"H3", "H4", "H6", "H7", "H9", "H11", "H13", "H16", "H17", "H18", "H19", "H20", "H21", "H23", "H24", "H25", "H26"}:
        return "gold_m15"
    if key in {"H14", "H15", "H22"}:
        return "gold_m5"
    if key == "H30":
        return "fx_m5"
    return "fx_h1"


def mask_split(index: pd.DatetimeIndex, split: Split, part: str) -> pd.Series:
    if part == "train":
        return index <= split.train_end
    if part == "validation":
        return (index > split.train_end) & (index <= split.val_end)
    if part == "oos":
        return index > split.val_end
    raise ValueError(part)


def pip_size(symbol: str) -> float:
    base = "".join(ch for ch in symbol.upper() if ch.isalpha())[:6]
    if "XAU" in base or "GOLD" in base:
        return 0.001
    if base.endswith("JPY"):
        return 0.01
    return 0.0001


def cost_price(symbol: str, spread_pips: float, slippage_pips: float) -> float:
    return (spread_pips + slippage_pips) * pip_size(symbol)


def years_ago(years: int) -> datetime:
    return datetime.now(tz=UTC) - timedelta(days=365 * years)
