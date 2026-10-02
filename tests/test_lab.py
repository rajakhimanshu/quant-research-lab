import json
from datetime import date

import numpy as np
import pandas as pd
import yaml

from ats.config import load_settings
from ats.data.clean import clean_ohlc
from ats.features.prepare import prepare_frame
from ats.lab import book
from ats.lab.loop import budget_block
from ats.lab.proposer import propose_from_lead, record_spec
from ats.lab.spec import freeze, signature, split_book, validate_spec
from ats.lab.templates import TEMPLATES, lead_lag_events, template_events

WHY = ("Corporate and fund hedgers must execute at the 16:00 London WM fix, so flow into the fix is "
       "one-sided and reverses once the benchmark window closes.")


def _spec(**over):
    spec = {
        "name": "WM fix fade test",
        "why": WHY,
        "payer": "Index funds and corporates forced to trade at the WM/Reuters benchmark",
        "causal_category": "forced_flow",
        "template": "clock_run",
        "symbols": ["EURUSD"],
        "timeframe": "M15",
        "params": {"tz": "Europe/London", "treat_hour": 16, "base_hour": 11, "mode": "fade"},
    }
    spec.update(over)
    return spec


def _frame(n=3000, freq="15min", seed=4, symbol="EURUSD", base=1.08, vol=0.0004):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    close = base + np.cumsum(rng.normal(0, vol, n))
    open_ = np.r_[close[0], close[:-1]]
    noise = rng.uniform(vol * 0.2, vol * 1.2, n)
    df = pd.DataFrame({
        "time": times, "open": open_,
        "high": np.maximum(open_, close) + noise, "low": np.minimum(open_, close) - noise,
        "close": close, "volume": 100, "spread": 10, "symbol": symbol,
        "requested_symbol": symbol, "timeframe": "M15",
    })
    return prepare_frame(clean_ohlc(df), load_settings())


def test_valid_spec_fills_defaults():
    clean, errors = validate_spec(_spec())
    assert errors == []
    assert clean["params"]["horizon_bars"] == 8
    assert clean["baseline"].startswith("Same rule at a control clock")


def test_validator_refuses_protocol_breaks():
    assert any("folklore" in e for e in validate_spec(_spec(why=WHY + " Confirm with RSI."))[1])
    assert any("universe" in e for e in validate_spec(_spec(symbols=["BTCUSD"]))[1])
    assert any("timeframe" in e for e in validate_spec(_spec(timeframe="H4"))[1])
    assert any("payer" in e for e in validate_spec(_spec(payer=""))[1])
    assert any("unknown params" in e for e in validate_spec(
        _spec(params={"tz": "Europe/London", "treat_hour": 16, "base_hour": 11, "mode": "fade", "rsi": 2}))[1])
    assert any("identical" in e for e in validate_spec(
        _spec(params={"tz": "Europe/London", "treat_hour": 16, "base_hour": 16, "mode": "fade"}))[1])
    assert any("above" in e for e in validate_spec(
        _spec(params={"tz": "Europe/London", "treat_hour": 16, "base_hour": 11, "mode": "fade",
                      "stop_atr": 9}))[1])


def test_lead_lag_leader_must_differ():
    spec = _spec(template="lead_lag", symbols=["GBPUSD"], timeframe="M5",
                 params={"leader": "GBPUSD"})
    assert any("leader" in e for e in validate_spec(spec)[1])


def test_signature_ignores_risk_knobs():
    a, _ = validate_spec(_spec())
    b, _ = validate_spec(_spec(params={"tz": "Europe/London", "treat_hour": 16, "base_hour": 11,
                                       "mode": "fade", "stop_atr": 2.0, "horizon_bars": 20}))
    c, _ = validate_spec(_spec(params={"tz": "Europe/London", "treat_hour": 15, "base_hour": 11,
                                       "mode": "fade"}))
    assert signature(a) == signature(b)
    assert signature(a) != signature(c)


def test_split_books_follow_market():
    assert split_book(["XAUUSD"], "M5") == "gold_m5"
    assert split_book(["EURUSD"], "M15") == "fx_m15"
    assert split_book(["XAUUSD"], "H1") == "fx_h1"


def test_templates_emit_both_arms_on_synthetic_data():
    df = _frame()
    ctx = {"timeframe": "M15", "load": lambda s, tf: _frame(seed=9, symbol=s)}
    cases = {
        "clock_run": {"tz": "Europe/London", "treat_hour": 16, "base_hour": 11, "mode": "fade",
                      "min_prior_atr": 0.1},
        "session_box": {"tz": "Europe/London", "box_start_hour": 8, "box_end_hour": 9},
        "sweep_reclaim": {},
    }
    for name, params in cases.items():
        clean, errors = validate_spec(_spec(template=name, params=params))
        assert errors == [], (name, errors)
        ev = template_events(name, df, "EURUSD", clean["params"], 0.8, 0.2, ctx)
        assert not ev.empty, name
        assert ev["treatment"].any() and (~ev["treatment"]).any(), name
        assert {"r_mult", "side", "time"} <= set(ev.columns), name


def test_follow_mode_swaps_mirror_arms():
    df = _frame()
    fade = template_events("sweep_reclaim", df, "EURUSD", {"mode": "fade"}, 0.8, 0.2, {})
    follow = template_events("sweep_reclaim", df, "EURUSD", {"mode": "follow"}, 0.8, 0.2, {})
    assert len(fade) == len(follow)
    assert (fade["treatment"].to_numpy() == ~follow["treatment"].to_numpy()).all()


def test_lead_lag_trades_follower_after_leader_move():
    follower = _frame(seed=1, symbol="GBPUSD")
    leader = _frame(seed=2)
    ev = lead_lag_events(follower, "GBPUSD", {"leader_move_atr": 0.8, "session_start_hour_utc": 0,
                                               "session_end_hour_utc": 24}, 1.0, 0.2, leader)
    assert not ev.empty
    assert ev["treatment"].sum() == (~ev["treatment"]).sum()
    times = pd.to_datetime(ev["time"]).sort_values()
    assert times.is_monotonic_increasing


def test_budget_allows_one_family_per_window():
    runs = [{"batch": "lab_x", "date": "2026-10-01"}]
    assert budget_block([], 7) is None
    assert "budget" in budget_block(runs, 7, today=date(2026, 10, 3))
    assert budget_block(runs, 7, today=date(2026, 10, 9)) is None


def test_record_spec_queues_then_flags_duplicate():
    rows: list[dict] = []
    first = record_spec(_spec(), "test", rows)
    second = record_spec(_spec(params={"tz": "Europe/London", "treat_hour": 16, "base_hour": 11,
                                       "mode": "fade", "target_atr": 2.0}), "test", rows)
    bad = record_spec(_spec(template="nope"), "test", rows)
    assert first["status"] == "queued"
    assert second["status"] == "duplicate"
    assert bad["status"] == "invalid"
    assert [r["id"] for r in rows] == ["P0001", "P0002", "P0003"]


def test_proposer_handles_spec_reject_and_garbage():
    rows: list[dict] = []
    ok = propose_from_lead({"id": "MECH_X", "name": "fix"},
                           lambda m: "```json\n" + json.dumps({"decision": "spec", "spec": _spec()}) + "\n```",
                           rows)
    no = propose_from_lead({"id": "MECH_Y"}, lambda m: '{"decision": "reject", "reason": "no payer"}', rows)
    err = propose_from_lead({"id": "MECH_Z"}, lambda m: "sorry", rows)
    assert ok["status"] == "queued" and ok["origin"] == "llm"
    assert no["status"] == "rejected_by_model"
    assert err["status"] == "error"


def test_prompt_contains_catalogue_and_rules():
    from ats.lab.proposer import build_prompt

    msgs = build_prompt({"id": "MECH_X"})
    text = msgs[1]["content"]
    for name in TEMPLATES:
        assert name in text
    assert "FORCED to trade" in text and "XAUUSD" in text


def test_freeze_appends_parseable_entries(tmp_path):
    path = tmp_path / "hypotheses.yaml"
    path.write_text("hypotheses:\n  - id: H1_x\n    why: test\n", encoding="utf-8")
    clean, _ = validate_spec(_spec())
    entries = freeze([clean], "lab_test", path=path)
    blob = yaml.safe_load(path.read_text(encoding="utf-8"))
    frozen = blob["hypotheses"][-1]
    assert frozen["id"] == entries[0]["id"]
    assert frozen["template"] == "clock_run" and frozen["enabled"] is False
    assert frozen["params"]["symbols"] == ["EURUSD"] and frozen["split_book"] == "fx_m15"


def test_book_row_from_report():
    class Rep:
        decision = "CANDIDATE"
        reject_reason = None
        train = {"rate": 0.55, "baseline_rate": 0.45, "n": 150}
        validation = {"rate": 0.52, "baseline_rate": 0.44, "n": 40}
        notes = ["train treatment mean R=0.1200 n=150 total R=18.0",
                 "validation treatment mean R=0.0500 n=40 total R=2.0"]

    entry = {"id": "H999_x", "split_book": "fx_m15", "template": "clock_run", "signature": "abc",
             "why": WHY, "payer": "p", "baseline": "b"}
    row = book.row_from_report(entry, Rep(), "lab_t", 4, 0.0125)
    assert row["decision"] == "CANDIDATE"
    assert "ats lab unlock --id H999_x" in row["why_closed"]
    assert "R +0.12" in row["train"]
