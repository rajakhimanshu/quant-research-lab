from ats.config import load_settings
from ats.hypotheses.execution import mirror_trades
from ats.research.pipeline import decide, mirage_reason
from tests.helpers import prepared


def test_lab_can_pass_a_real_lift():
    settings = load_settings()
    train = {
        "n": 200,
        "successes": 150,
        "rate": 0.75,
        "baseline_n": 200,
        "baseline_successes": 100,
        "baseline_rate": 0.50,
        "p_value": 1e-6,
    }
    validation = {
        "n": 80,
        "successes": 56,
        "rate": 0.70,
        "baseline_n": 80,
        "baseline_successes": 40,
        "baseline_rate": 0.50,
        "p_value": 0.01,
    }
    decision, reason = decide(train, validation, settings, n_tests=1, notes=[])
    assert decision == "CANDIDATE"
    assert reason is None


def test_lab_rejects_a_coin_flip():
    settings = load_settings()
    train = {
        "n": 200,
        "successes": 100,
        "rate": 0.50,
        "baseline_n": 200,
        "baseline_successes": 102,
        "baseline_rate": 0.51,
        "p_value": 0.60,
    }
    validation = {
        "n": 80,
        "successes": 36,
        "rate": 0.45,
        "baseline_n": 80,
        "baseline_successes": 38,
        "baseline_rate": 0.475,
        "p_value": 0.70,
    }
    decision, reason = decide(train, validation, settings, n_tests=1, notes=[])
    assert decision == "REJECT"
    assert reason is not None


def test_lab_rejects_treatment_only_design():
    settings = load_settings()
    train = {"n": 200, "successes": 150, "rate": 0.75, "baseline_n": 0,
             "baseline_successes": 0, "baseline_rate": float("nan"), "p_value": None}
    validation = dict(train, n=80)
    decision, reason = decide(train, validation, settings, n_tests=1, notes=[])
    assert decision == "REJECT"
    assert "baseline" in reason


def test_mirage_gate_needs_positive_r_on_both_parts():
    assert mirage_reason(0.10, 0.05) is None
    assert mirage_reason(None, None) is None
    assert "mirage" in mirage_reason(-0.02, 0.30)
    assert "mirage" in mirage_reason(0.10, -0.01)
    assert "mirage" in mirage_reason(0.10, None)


def test_mirror_trades_returns_treatment_and_opposite_baseline():
    work = prepared().reset_index(drop=True)
    rows = mirror_trades(work, 250, "long", "EURUSD", 0.0001, 6, 1.0, 1.0)
    assert len(rows) == 2
    treat, base = rows
    assert treat["treatment"] and not base["treatment"]
    assert {treat["side"], base["side"]} == {"long", "short"}
    assert treat["entry"] > base["entry"]
