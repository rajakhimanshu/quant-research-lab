from ats.hypotheses.atr_momentum import atr_momentum_events
from tests.helpers import prepared


PARAMS = {
    "atr_multiplier": 1.5,
    "close_percent": 20.0,
    "sltp_mode": "percent",
    "stop_loss_pct": 1.0,
    "take_profit_pct": 2.0,
    "max_spread_atr_pct": 0.0,
}


def test_edge_close_is_treatment_and_hits_tp():
    df = prepared(250, seed=1)
    i = 80
    atr_prior = float(df.at[i - 1, "atr"])
    o = float(df.at[i, "open"])
    df.at[i, "low"] = o
    df.at[i, "high"] = o + 2.0 * atr_prior
    df.at[i, "close"] = df.at[i, "high"]
    df.at[i, "range"] = df.at[i, "high"] - df.at[i, "low"]

    entry = float(df.at[i + 1, "open"])
    tp = entry * 1.02
    sl = entry * 0.99
    for k in range(1, 8):
        df.at[i + k, "open"] = entry
        df.at[i + k, "close"] = tp
        df.at[i + k, "high"] = tp + 0.0001
        df.at[i + k, "low"] = sl + 0.0005

    ev = atr_momentum_events(df, "EURUSD", PARAMS, spread_pips=1.2, slippage_pips=0.5)
    hit = ev.loc[ev["treatment"] & (ev["time"] == df.at[i, "time"])]
    assert not hit.empty
    assert bool(hit.iloc[0]["success"])
    assert hit.iloc[0]["side"] == "long"


def test_mid_range_close_is_baseline():
    df = prepared(250, seed=2)
    i = 90
    atr_prior = float(df.at[i - 1, "atr"])
    o = float(df.at[i, "open"])
    hi = o + 2.0 * atr_prior
    lo = o - 0.1 * atr_prior
    df.at[i, "high"] = hi
    df.at[i, "low"] = lo
    df.at[i, "close"] = (hi + lo) / 2.0
    df.at[i, "range"] = hi - lo

    entry = float(df.at[i + 1, "open"])
    sl = entry * 0.99
    cap = entry * 1.005
    for k in range(1, 8):
        df.at[i + k, "open"] = entry
        df.at[i + k, "close"] = sl
        df.at[i + k, "high"] = cap
        df.at[i + k, "low"] = sl - 0.0001

    ev = atr_momentum_events(df, "EURUSD", PARAMS, spread_pips=1.2, slippage_pips=0.5)
    row = ev.loc[ev["time"] == df.at[i, "time"]]
    assert not row.empty
    assert not bool(row.iloc[0]["treatment"])
