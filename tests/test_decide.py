from ats.config import load_settings
from ats.research.pipeline import decide


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
