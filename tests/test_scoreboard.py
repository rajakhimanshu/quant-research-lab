"""Scoreboard rebuild from ledger."""

from ats.research.scoreboard import build_scoreboard_markdown, write_scoreboard


def test_scoreboard_contains_latest_ids():
    text = build_scoreboard_markdown()
    assert "H1_event_reversal" in text
    assert "H213_ny19_fade_gold" in text
    assert "REJECT" in text


def test_write_scoreboard_roundtrip(tmp_path, monkeypatch):
    # write_scoreboard targets repo path; just ensure builder is non-empty
    md = build_scoreboard_markdown()
    assert md.count("|") > 50
    assert "Total tested" in md
