from __future__ import annotations

from pathlib import Path

import pandas as pd

from ats.config import DATA_DIR

REQUIRED = ["datetime_utc", "currency", "impact", "event"]


def calendar_path() -> Path:
    return DATA_DIR / "calendar" / "high_impact.csv"


def load_calendar(path: Path | None = None) -> pd.DataFrame:
    path = path or calendar_path()
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Calendar CSV missing columns: {missing}")
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["impact"] = df["impact"].str.lower().str.strip()
    df["currency"] = df["currency"].str.upper().str.strip()
    return df.sort_values("datetime_utc").reset_index(drop=True)


def pair_currencies(symbol: str) -> tuple[str, str]:
    base = symbol[:6].upper()
    return base[:3], base[3:6]
