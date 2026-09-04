"""Retail-positioning CSV drop. No broker login scrape."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ats.config import DATA_DIR

SENT_DIR = DATA_DIR / "sentiment"


def _col(df: pd.DataFrame, *needles: str) -> str:
    low = [str(c).lower().replace(" ", "_") for c in df.columns]
    for needle in needles:
        n = needle.lower().replace(" ", "_")
        for i, name in enumerate(low):
            if n in name:
                return str(df.columns[i])
    raise KeyError(f"need a column matching {needles}; got {list(df.columns)}")


def load_sentiment_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_c = _col(df, "date", "time", "asof")
    long_c = _col(df, "long_pct", "long%", "percent_long", "client_long")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_c], utc=True, errors="coerce"),
            "long_pct": pd.to_numeric(df[long_c], errors="coerce"),
        }
    )
    if "instrument" in {c.lower() for c in df.columns}:
        inst_c = _col(df, "instrument", "symbol", "market")
        out["instrument"] = df[inst_c].astype(str)
    out = out.dropna(subset=["date", "long_pct"]).sort_values("date")
    out["long_z"] = (out["long_pct"] - out["long_pct"].rolling(52, min_periods=20).mean()) / out[
        "long_pct"
    ].rolling(52, min_periods=20).std().replace(0, pd.NA)
    return out.reset_index(drop=True)


def list_csvs() -> list[Path]:
    SENT_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(SENT_DIR.glob("*.csv"))


def print_sentiment(path: Path | None = None) -> None:
    files = [path] if path else list_csvs()
    if not files or (path is None and not files):
        print("No sentiment CSV. Drop date,instrument,long_pct at data/sentiment/")
        print("Do not scrape a broker login. Extreme retail long is a lead, not a trade.")
        return
    for f in files:
        df = load_sentiment_csv(f)
        print(f"{f.name}: {len(df)} rows {df['date'].min().date()} .. {df['date'].max().date()}")
        last = df.dropna(subset=["long_z"]).tail(3)
        for _, row in last.iterrows():
            z = row["long_z"]
            flag = " extreme" if pd.notna(z) and abs(float(z)) >= 2 else ""
            print(f"  {row['date'].date()} long_pct={row['long_pct']:.1f} z={float(z):.2f}{flag}")
        print("Write a why and freeze a spec before python -m ats test. This is not a signal.")
