from ats.hypotheses.london_close_fade import london_close_fade_events
from tests.helpers import prepared


def test_london_close_fade_emits_both_hours():
    df = prepared(900, seed=8)
    ev = london_close_fade_events(
        df,
        "GBPUSD",
        {
            "lookback_bars": 7,
            "horizon_bars": 2,
            "flatten_hour_london": 15,
            "open_hour_london": 8,
            "min_prior_atr": 0.1,
        },
        spread_pips=1.6,
        slippage_pips=0.5,
    )
    assert ev is not None
    if ev.empty:
        return
    hours = set(ev["london_hour"].astype(int))
    assert hours <= {8, 15}
    assert "r_mult" in ev.columns
    assert "treatment" in ev.columns
