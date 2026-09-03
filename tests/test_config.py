import yaml

from ats.config import CONFIG_DIR
from ats.config import load_hypotheses


def test_every_hypothesis_has_a_why():
    items = load_hypotheses()
    assert items
    for item in items:
        assert item["why"].strip()


def test_hypotheses_file_lists_h1_and_h2():
    raw = yaml.safe_load((CONFIG_DIR / "hypotheses.yaml").read_text(encoding="utf-8"))
    ids = {h["id"] for h in raw["hypotheses"]}
    assert "H1_event_reversal" in ids
    assert "H2_tap_breakout" in ids
