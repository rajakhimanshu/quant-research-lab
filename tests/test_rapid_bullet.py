from ats.hypotheses.rapid_bullet import rapid_bullet_events
from tests.helpers import prepared


PARAMS = {
    "bars_n": 5,
    "lookback_bars": 40,
    "order_distance_pips": 10.0,
    "sl_pips": 20.0,
    "tp_pips": 20.0,
    "expiration_bars": 30,
    "donchian_n": 20,
}


def test_intact_swing_buy_stop_hits_tp():
    df = prepared(250, seed=1)
    mid = 1.1000
    for i in range(len(df)):
        df.at[i, "open"] = mid
        df.at[i, "close"] = mid
        df.at[i, "high"] = mid + 0.0002
        df.at[i, "low"] = mid - 0.0002
    pivot = 80
    entry = 1.1050
    df.at[pivot, "high"] = entry
    t = pivot + 6
    tp = entry + 0.0020
    sl = entry - 0.0020
    for k in range(1, 8):
        df.at[t + k, "open"] = mid
        df.at[t + k, "close"] = tp
        df.at[t + k, "high"] = tp + 0.0001
        df.at[t + k, "low"] = sl + 0.0003

    ev = rapid_bullet_events(df, "EURUSD", PARAMS, spread_pips=1.2, slippage_pips=0.5)
    treat = ev.loc[ev["treatment"] & (ev["side"] == "long")]
    assert not treat.empty
    assert bool(treat.iloc[0]["success"])


def test_rapid_bullet_does_not_crash():
    df = prepared(250, seed=4)
    ev = rapid_bullet_events(df, "EURUSD", PARAMS, spread_pips=1.2, slippage_pips=0.5)
    assert ev is not None
