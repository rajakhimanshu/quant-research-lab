from ats.ideas.gates import calendar_prompt, fact_check, load_calendar, novelty_check


def test_calendar_loads_with_official_times():
    cal = load_calendar()
    assert cal["wmr_fix"]["time"] == "16:00" and cal["wmr_fix"]["tz"] == "Europe/London"
    assert cal["snb_decision"]["time"] == "09:30"
    assert "boj" not in " ".join(cal)


def test_calendar_prompt_lists_only_untested_families(monkeypatch):
    from ats.ideas import gates

    monkeypatch.setattr(gates, "_ledger_text", lambda: [("H1", "usdjpy tokyo fix 09:55 follow")])
    monkeypatch.setattr(gates, "_do_not_reopen", lambda: [])
    text = calendar_prompt()
    assert "tokyo_fix" not in text
    assert "snb_decision" in text


def test_fact_check_needs_listed_event_and_instrument():
    assert fact_check({"event_id": "snb_decision", "instrument": "USDCHF"}) is None
    assert "not a scheduled event" in fact_check({"event_id": "boj_0830", "instrument": "USDJPY"})
    assert "not a scheduled event" in fact_check({"instrument": "EURUSD"})
    assert "not traded around" in fact_check({"event_id": "snb_decision", "instrument": "NZDUSD"})


def test_novelty_rejects_tested_event_family():
    ledger = [("H28_fix", "h28 eurusd long at london 16:00 vs long at 08:00. krohn postfix.")]
    idea = {"event_id": "wmr_fix", "instrument": "EURUSD"}
    assert "H28_fix" in novelty_check(idea, ledger=ledger, dnr=[])
    assert novelty_check({"event_id": "wmr_fix", "instrument": "NZDUSD"}, ledger=ledger, dnr=[]) is None
    assert "do-not-reopen" in novelty_check({"event_id": "snb_decision", "instrument": "USDCHF"},
                                            ledger=[], dnr=["snb surprise fade"])
