"""CFTC Commitment of Traders — official weekly files, not a scrape."""

from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime, timezone
from urllib.request import Request, urlopen

import pandas as pd

from ats.config import DATA_DIR

COT_DIR = DATA_DIR / "cot"
HISTORY = "https://www.cftc.gov/files/dea/history/deacot{year}.zip"
COMBINED = COT_DIR / "combined.csv"


def _get(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "ats-research-lab/1.0"})
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _find_col(df: pd.DataFrame, *needles: str) -> str:
    cols = list(df.columns)
    low = [_norm(c) for c in cols]
    for needle in needles:
        n = _norm(needle)
        for i, name in enumerate(low):
            if n in name:
                return cols[i]
    raise KeyError(f"no column matching {needles} in {cols[:12]}")


def _parse_annual(raw: bytes) -> pd.DataFrame:
    text = None
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = zf.namelist()
        pick = next((n for n in names if n.lower().endswith(".txt") or n.lower().endswith(".csv")), names[0])
        text = zf.read(pick)
    try:
        df = pd.read_csv(io.BytesIO(text), low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(text), low_memory=False, encoding="latin-1")
    mkt = _find_col(df, "market_and_exchange", "market_and_exchange_names")
    date_c = _find_col(df, "yyyy-mm-dd", "as_of_date_in_form_yyyy", "report_date")
    oi = _find_col(df, "open_interest_all")
    ncl = _find_col(df, "noncomm_positions_long_all", "noncommercial_positions_long")
    ncs = _find_col(df, "noncomm_positions_short_all", "noncommercial_positions_short")
    out = pd.DataFrame(
        {
            "market": df[mkt].astype(str),
            "asof": pd.to_datetime(df[date_c], errors="coerce", utc=True),
            "oi": pd.to_numeric(df[oi], errors="coerce"),
            "nc_long": pd.to_numeric(df[ncl], errors="coerce"),
            "nc_short": pd.to_numeric(df[ncs], errors="coerce"),
        }
    )
    return out.dropna(subset=["asof", "oi"])


def pull_cot(years: list[int] | None = None) -> pd.DataFrame:
    COT_DIR.mkdir(parents=True, exist_ok=True)
    now_y = datetime.now(timezone.utc).year
    years = years or list(range(now_y - 5, now_y + 1))
    frames = []
    for y in years:
        cache = COT_DIR / f"deacot{y}.csv"
        if cache.exists():
            frames.append(pd.read_csv(cache, parse_dates=["asof"]))
            continue
        try:
            raw = _get(HISTORY.format(year=y))
        except Exception as exc:
            print(f"  COT {y}: skip ({exc})")
            continue
        piece = _parse_annual(raw)
        piece.to_csv(cache, index=False)
        frames.append(piece)
        print(f"  COT {y}: {len(piece)} rows")
    if not frames:
        raise RuntimeError("No COT files downloaded. Check network / CFTC URL.")
    out = pd.concat(frames, ignore_index=True)
    out["asof"] = pd.to_datetime(out["asof"], utc=True)
    out["nc_net"] = out["nc_long"] - out["nc_short"]
    out["nc_net_oi"] = out["nc_net"] / out["oi"].replace(0, pd.NA)
    out = out.drop_duplicates(["market", "asof"]).sort_values(["market", "asof"])
    COMBINED.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(COMBINED, index=False)
    return out


def combined_path():
    return COMBINED


def load_combined() -> pd.DataFrame | None:
    if not COMBINED.exists():
        return None
    out = pd.read_csv(COMBINED)
    out["asof"] = pd.to_datetime(out["asof"], utc=True, errors="coerce")
    return out.dropna(subset=["asof"])


def gold_series(cot: pd.DataFrame) -> pd.DataFrame:
    m = cot["market"].str.upper()
    hit = m.str.contains("GOLD", regex=False) & m.str.contains("COMMODITY EXCHANGE", regex=False)
    hit &= ~m.str.contains("MICRO", regex=False) & ~m.str.contains("MINI", regex=False)
    g = cot.loc[hit].copy()
    if g.empty:
        g = cot.loc[m.str.contains("GOLD", regex=False)].copy()
    if "nc_net" not in g.columns and {"nc_long", "nc_short"} <= set(g.columns):
        g["nc_net"] = g["nc_long"] - g["nc_short"]
    if "nc_net_oi" not in g.columns and "oi" in g.columns and "nc_net" in g.columns:
        g["nc_net_oi"] = g["nc_net"] / g["oi"].replace(0, pd.NA)
    g = g.sort_values("asof").drop_duplicates("asof")
    return g.reset_index(drop=True)


def named_series(cot: pd.DataFrame, market: str) -> pd.DataFrame:
    """Exact CFTC market name. Do not fuzzy-match MICRO/XRATE books."""
    target = str(market).upper().strip()
    m = cot["market"].astype(str).str.upper().str.strip()
    g = cot.loc[m == target].copy()
    if g.empty:
        return g
    if "nc_net" not in g.columns and {"nc_long", "nc_short"} <= set(g.columns):
        g["nc_net"] = g["nc_long"] - g["nc_short"]
    if "nc_net_oi" not in g.columns and "oi" in g.columns and "nc_net" in g.columns:
        g["nc_net_oi"] = g["nc_net"] / g["oi"].replace(0, pd.NA)
    g = g.sort_values("asof").drop_duplicates("asof")
    return g.reset_index(drop=True)
