from ats.timeutil import pip_size
from tests.helpers import prepared
from ats.hypotheses.ny_open_sweep import ny_open_sweep_events


from ats.hypotheses.session_orb import session_orb_events


def test_gold_pip_size():
    assert pip_size("XAUUSD") == 0.001
    assert pip_size("XAUUSDm") == 0.001


def test_exness_standard_gold_cost_is_twenty_cents_spread():
    from ats.timeutil import cost_price

    # 200 lab units * 0.001 = $0.20/oz = Exness 20 pips of 0.01
    assert abs(cost_price("XAUUSD", 200, 0) - 0.20) < 1e-9
    assert abs(cost_price("EURUSD", 0.8, 0.2) - 0.00010) < 1e-12


def test_sweep_params_do_not_crash():
    df = prepared(400, seed=4)
    ev = ny_open_sweep_events(
        df,
        "EURUSD",
        {"sweep_atr": 0.1, "target_atr": 0.5, "horizon_bars": 4, "ny_start": 9, "ny_end": 10},
        spread_pips=1.2,
        slippage_pips=0.5,
    )
    assert ev is not None
    london = ny_open_sweep_events(
        df,
        "EURUSD",
        {
            "sweep_atr": 0.1,
            "target_atr": 0.5,
            "horizon_bars": 4,
            "ny_start": 9,
            "ny_end": 10,
            "pool_ny_start": 3,
            "pool_ny_end": 8,
        },
        spread_pips=1.2,
        slippage_pips=0.5,
    )
    assert london is not None


def test_fade_mode_does_not_crash():
    df = prepared(400, seed=5)
    ev = session_orb_events(
        df,
        "EURUSD",
        {"mode": "fade", "asia_start_hour_utc": 0, "asia_end_hour_utc": 7, "ny_start_hour_utc": 13, "ny_end_hour_utc": 17},
        spread_pips=1.2,
        slippage_pips=0.5,
    )
    assert ev is not None


def test_sweep_params_do_not_crash():
    df = prepared(400, seed=4)
    ev = ny_open_sweep_events(
        df,
        "EURUSD",
        {"sweep_atr": 0.1, "target_atr": 0.5, "horizon_bars": 4, "ny_start": 9, "ny_end": 10},
        spread_pips=1.2,
        slippage_pips=0.5,
    )
    assert ev is not None
