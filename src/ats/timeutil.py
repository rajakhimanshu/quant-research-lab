from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

UTC = ZoneInfo("UTC")
IST = ZoneInfo("Asia/Kolkata")

TIMEFRAME_MINUTES = {
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


def mask_split(index: pd.DatetimeIndex, split: Split, part: str) -> pd.Series:
    if part == "train":
        return index <= split.train_end
    if part == "validation":
        return (index > split.train_end) & (index <= split.val_end)
    if part == "oos":
        return index > split.val_end
    raise ValueError(part)


def pip_size(symbol: str) -> float:
    base = symbol[:6].upper()
    if base.endswith("JPY"):
        return 0.01
    return 0.0001


def cost_price(symbol: str, spread_pips: float, slippage_pips: float) -> float:
    return (spread_pips + slippage_pips) * pip_size(symbol)


def years_ago(years: int) -> datetime:
    return datetime.now(tz=UTC) - timedelta(days=365 * years)
