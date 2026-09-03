from __future__ import annotations

from datetime import time

import pandas as pd

from ats.timeutil import IST, parse_hhmm, time_in_window


def add_sessions(df: pd.DataFrame, sessions_ist: dict) -> pd.DataFrame:
    out = df.copy()
    ist = out["time"].dt.tz_convert(IST)
    clock = ist.dt.time
    named: dict[str, tuple[time, time]] = {}
    for name, pair in sessions_ist.items():
        named[name] = (parse_hhmm(pair[0]), parse_hhmm(pair[1]))
        out[f"session_{name}"] = clock.map(lambda t, s=named[name]: time_in_window(t, s[0], s[1]))
    out["ist_hour"] = ist.dt.hour + ist.dt.minute / 60.0
    out["ist_time"] = ist
    return out
