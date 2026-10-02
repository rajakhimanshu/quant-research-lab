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
    if key in {"H48", "H50"}:
        return "fx_m15"
    if key == "H49":
        return "gold_m15"
    if key in {"H51", "H54", "H56"}:
        return "gold_m15"
    if key in {"H53", "H55"}:
        return "fx_m15"
    num = int(key[1:]) if key[1:].isdigit() else -1
    if num in {57, 67, 73}:
        return "gold_m15"
    if 57 <= num <= 76:
        return "fx_m15"
    if 90 <= num <= 95:
        return "fx_m15"
    if num == 96:
        return "gold_m15"
    if 97 <= num <= 101:
        return "fx_m15"
    if 103 <= num <= 105:
        return "fx_m15"
    if num in {106, 107}:
        return "fx_m15"
    if num == 108:
        return "gold_m15"
    if 109 <= num <= 111:
        return "fx_m15"
    if num in {112, 113}:
        return "fx_m15"
    if num == 114:
        return "gold_m15"
    if num in {115, 116}:
        return "fx_m15"
    if num == 117:
        return "gold_m15"
    if num in {118, 119}:
        return "fx_m15"
    if num == 120:
        return "gold_m15"
    if num in {121, 122}:
        return "fx_m15"
    if num == 123:
        return "gold_m15"
    if num in {124, 125}:
        return "fx_m15"
    if num == 126:
        return "gold_m15"
    if num in {127, 128}:
        return "fx_m15"
    if num == 129:
        return "gold_m15"
    if num in {130, 131}:
        return "fx_m15"
    if num == 132:
        return "gold_m15"
    if num in {133, 134}:
        return "fx_m15"
    if num == 135:
        return "gold_m15"
    if num in {136, 137}:
        return "fx_m15"
    if num == 138:
        return "gold_m15"
    if num in {139, 140}:
        return "fx_m15"
    if num == 141:
        return "gold_m15"
    if num in {142, 143}:
        return "fx_m15"
    if num == 144:
        return "gold_m15"
    if 145 <= num <= 147:
        return "fx_m15"
    if num in {148, 149}:
        return "fx_m15"
    if num == 150:
        return "gold_m15"
    if num in {151, 152}:
        return "fx_m15"
    if num == 153:
        return "gold_m15"
    if num in {154, 155}:
        return "fx_m15"
    if num == 156:
        return "gold_m15"
    if num in {157, 158}:
        return "fx_m15"
    if num == 159:
        return "gold_m15"
    if num in {160, 161}:
        return "fx_m15"
    if num == 162:
        return "gold_m15"
    if num in {163, 164}:
        return "fx_m15"
    if num == 165:
        return "gold_m15"
    if num in {166, 167}:
        return "fx_m15"
    if num == 168:
        return "gold_m15"
    if num in {169, 170}:
        return "fx_m15"
    if num == 171:
        return "gold_m15"
    if num == 172:
        return "gold_m15"
    if num in {173, 174}:
        return "fx_m15"
    if num == 175:
        return "gold_m15"
    if num in {176, 177}:
        return "fx_m15"
    if num == 178:
        return "gold_m15"
    if num in {179, 180}:
        return "fx_m15"
    if num in {181, 182}:
        return "fx_m15"
    if num == 183:
        return "gold_m15"
    if num == 184:
        return "gold_m15"
    if num in {185, 186}:
        return "fx_m15"
    if num in {187, 188}:
        return "fx_m15"
    if num == 189:
        return "gold_m15"
    if num in {190, 191}:
        return "fx_m15"
    if num == 192:
        return "gold_m15"
    if num == 193:
        return "fx_h1"
    if num in {194, 195}:
        return "gold_m15"
    if num in {196, 197}:
        return "fx_m15"
    if num == 198:
        return "gold_m15"
    if num in {199, 200}:
        return "fx_m15"
    if num == 201:
        return "gold_m15"
    if num in {202, 203}:
        return "fx_m15"
    if num == 204:
        return "gold_m15"
    if num in {205, 206}:
        return "fx_m15"
    if num == 207:
        return "gold_m15"
    if num in {208, 209}:
        return "fx_m15"
    if num == 210:
        return "gold_m15"
    if num in {211, 212}:
        return "fx_m15"
    if num == 213:
        return "gold_m15"
    if num == 214:
        return "gold_m5"
    if num in {237, 242, 243, 244}:
        return "fx_m5"
    if num in {239, 240}:
        return "gold_m5"
    if num == 245:
        return "fx_m15"
    if num in {241, 247}:
        return "gold_m15"
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
