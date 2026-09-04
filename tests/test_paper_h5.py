from ats.paper.forward import TARGET_FILLS, closed_stats
from ats.paper.h5_book import shares_for_risk
import pandas as pd


def test_shares_for_risk_caps_and_rounds_down():
    # $10k, 0.5% = $50 risk. Stop $2 away → 25 shares. $50 * 25 = $1,250 notional.
    assert shares_for_risk(10_000, 0.005, 100.0, 98.0) == 25
    assert shares_for_risk(10_000, 0.005, 100.0, 100.0) == 0
    assert shares_for_risk(50.0, 0.005, 100.0, 98.0) == 0


def test_forward_gate_starts_at_zero_fills():
    s = closed_stats(pd.DataFrame(columns=["event", "r"]))
    assert s["n"] == 0
    assert s["remaining"] == TARGET_FILLS
    assert "0/30" in s["gate"]
