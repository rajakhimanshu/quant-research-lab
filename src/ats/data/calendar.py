from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from ats.config import DATA_DIR

REQUIRED = ["datetime_utc", "currency", "impact", "event"]
FF_CACHE_URL = (
    "https://huggingface.co/datasets/Ehsanrs2/Forex_Factory_Calendar/"
    "resolve/main/forex_factory_cache.csv"
)


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


def _normalize_impact(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower()
    out = pd.Series("low", index=s.index)
    out = out.mask(s.str.contains("medium|med", regex=True), "medium")
    out = out.mask(s.str.contains("high"), "high")
    return out


def import_public_calendar(years: int = 4) -> Path:
    """Download a public Forex Factory archive and keep high-impact rows."""
    dest = calendar_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    cache = dest.parent / "forex_factory_cache.csv"
    if not cache.exists():
        with urlopen(FF_CACHE_URL, timeout=120) as resp, cache.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    raw = pd.read_csv(cache)
    cols = {c.lower().replace("_", ""): c for c in raw.columns}

    def pick(*names: str) -> str:
        for name in names:
            key = name.lower().replace("_", "")
            if key in cols:
                return cols[key]
            for existing, original in cols.items():
                if name.lower() in existing:
                    return original
        raise ValueError(f"Calendar cache missing {names}. Have {list(raw.columns)}")

    dt_col = pick("datetime", "date")
    ccy_col = pick("currency")
    impact_col = pick("impact")
    event_col = pick("event", "name", "title")
    out = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(raw[dt_col], utc=True, errors="coerce"),
            "currency": raw[ccy_col].astype(str).str.upper().str.strip(),
            "impact": _normalize_impact(raw[impact_col]),
            "event": raw[event_col].astype(str),
        }
    ).dropna(subset=["datetime_utc"])
    cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=years)
    out = out.loc[(out["impact"] == "high") & (out["datetime_utc"] >= cutoff)]
    out = out.sort_values("datetime_utc")
    out.to_csv(dest, index=False)
    return dest
