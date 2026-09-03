import pandas as pd

from ats.hypotheses.event_reversal import event_reversal_events
from tests.helpers import prepared


def test_news_large_candle_retrace_is_treatment_success():
    df = prepared(250, seed=3)
    i = 120
    # Force a large bullish spike, then a 50% retrace.
    atr = float(df.at[i, "atr"])
    df.at[i, "open"] = df.at[i, "close"]
    df.at[i, "low"] = df.at[i, "open"]
    df.at[i, "high"] = df.at[i, "open"] + 3.0 * atr
    df.at[i, "close"] = df.at[i, "high"]
    df.at[i, "range"] = df.at[i, "high"] - df.at[i, "low"]
    df.at[i, "body"] = df.at[i, "range"]
    df.at[i, "direction"] = 1
    df.at[i, "atr_multiple"] = 3.0
    mid = df.at[i, "high"] - 0.5 * df.at[i, "range"]
    for k in range(1, 4):
        df.at[i + k, "low"] = mid
        df.at[i + k, "high"] = df.at[i, "high"]
        df.at[i + k, "close"] = mid

    cal = pd.DataFrame(
        {
            "datetime_utc": [df.at[i, "time"]],
            "currency": ["USD"],
            "impact": ["high"],
            "event": ["test"],
        }
    )
    params = {
        "event_window_minutes": 30,
        "large_atr_mult": 2.5,
        "retrace_pct": 0.40,
        "horizon_bars": 3,
        "min_impact": "high",
    }
    ev = event_reversal_events(df, "EURUSD", params, calendar=cal)
    hit = ev.loc[ev["treatment"]]
    assert not hit.empty
    assert bool(hit.iloc[0]["success"])
