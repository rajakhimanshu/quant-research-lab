from __future__ import annotations

import numpy as np
import pandas as pd


def swing_points(df: pd.DataFrame, n: int) -> pd.DataFrame:
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n_bars = len(df)
    is_high = np.zeros(n_bars, dtype=bool)
    is_low = np.zeros(n_bars, dtype=bool)
    for i in range(n, n_bars - n):
        window_h = high[i - n : i + n + 1]
        window_l = low[i - n : i + n + 1]
        if high[i] == window_h.max() and np.argmax(window_h) == n:
            is_high[i] = True
        if low[i] == window_l.min() and np.argmin(window_l) == n:
            is_low[i] = True
    out = df.copy()
    out["swing_high"] = is_high
    out["swing_low"] = is_low
    return out


def cluster_levels(df: pd.DataFrame, tolerance_atr: float) -> pd.DataFrame:
    """Return one row per clustered swing level."""
    rows = []
    for kind, col, flag in (
        ("resistance", "high", "swing_high"),
        ("support", "low", "swing_low"),
    ):
        idx = df.index[df[flag]].tolist()
        used = set()
        for i in idx:
            if i in used:
                continue
            price = float(df.at[i, col])
            atr = float(df.at[i, "atr"]) if pd.notna(df.at[i, "atr"]) else np.nan
            if not np.isfinite(atr) or atr <= 0:
                continue
            tol = atr * tolerance_atr
            members = [i]
            used.add(i)
            for j in idx:
                if j in used:
                    continue
                other = float(df.at[j, col])
                if abs(other - price) <= tol:
                    members.append(j)
                    used.add(j)
            prices = df.loc[members, col]
            rows.append(
                {
                    "kind": kind,
                    "level": float(prices.mean()),
                    "first_bar": int(min(members)),
                    "last_bar": int(max(members)),
                    "members": members,
                    "tolerance": tol,
                }
            )
    return pd.DataFrame(rows)
