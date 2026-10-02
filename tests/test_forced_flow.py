from ats.hypotheses.forced_flow import (
    lbma_pm_run_events,
    month_end_rebalance_events,
    streak_inventory_fade_events,
    weekend_gap_fade_events,
    wm_fix_follow_events,
    comex_session_run_events,
)
from ats.research.ledger import ledger_rows
from ats.config import load_hypotheses
import numpy as np
import pandas as pd


def test_ledger_covers_every_closed_id():
    hyp_ids = {h["id"] for h in load_hypotheses()}
    ledger_ids = {r["id"] for r in ledger_rows()}
    missing = sorted(hyp_ids - ledger_ids)
    assert not missing, missing
    decisions = {r["id"]: r["decision"] for r in ledger_rows()}
    assert decisions["H5_equity_rsi2"] == "ARCHIVED"
    assert decisions["H22_ny_box_fade_vs_break"] == "REJECT"
    assert decisions["H9_cot_spec_fade"] == "REJECT"
    assert decisions["H27_wm_fix_follow"] == "REJECT"
    assert decisions["H37_weekend_gap_fx"] == "NEEDS_MORE_DATA"
    assert decisions["H86_weekend_gap_g10"] == "REJECT"
    assert decisions["H192_xetra1730_fade_gold"] == "REJECT"
    assert decisions["H140_tnext_roll_gbp"] == "REJECT"


def _gold(n: int = 500, seed: int = 2, freq: str = "15min") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-02 00:00", periods=n, freq=freq, tz="UTC")
    close = 2050 + np.cumsum(rng.normal(0.02, 0.7, n))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": np.maximum(close, open_) + 0.8,
            "low": np.minimum(close, open_) - 0.8,
            "close": close,
            "atr": np.full(n, 1.5),
        }
    )


def test_forced_flow_handlers_run():
    m15 = _gold()
    h1 = _gold(n=400, freq="h")
    params = {"horizon_bars": 4}
    for fn, df in (
        (lbma_pm_run_events, m15),
        (month_end_rebalance_events, m15),
        (streak_inventory_fade_events, m15),
        (weekend_gap_fade_events, m15),
        (wm_fix_follow_events, h1),
        (comex_session_run_events, m15),
    ):
        ev = fn(df, "XAUUSD" if fn is not wm_fix_follow_events else "GBPUSD", params, 200, 20)
        assert ev is not None
        if not ev.empty:
            assert "treatment" in ev.columns
            assert set(ev["side"]).issubset({"long", "short"})
