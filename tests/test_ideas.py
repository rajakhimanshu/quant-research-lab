import io
import zipfile

import numpy as np
import pandas as pd

from ats.hypotheses.cot_spec_fade import cot_spec_fade_events, rolling_last_pct, _fade_crowded_side
from ats.ideas.arxiv import parse_arxiv_xml
from ats.ideas.cot import _find_col, _parse_annual, gold_series, named_series
from ats.ideas.inbox import list_rows
from ats.ideas.sentiment import load_sentiment_csv
from ats.timeutil import split_book_for


def test_inbox_lists_cot_lead():
    ids = {r["id"] for r in list_rows()}
    assert "H9_cot_spec_fade" in ids
    assert "H11_gold_dxy_relink" in ids
    assert "LEAD_arxiv" in ids


def test_split_books():
    assert split_book_for("H8_atr_momentum") == "fx_h1"
    assert split_book_for("H9_cot_spec_fade") == "gold_m15"
    assert split_book_for("H83_eur_cot_fade") == "fx_h1"


def test_parse_arxiv_atom():
    xml = b"""<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Currency mean reversion</title>
        <published>2024-03-01T00:00:00Z</published>
        <summary>A test paper about FX.</summary>
        <link rel="alternate" type="text/html" href="https://arxiv.org/abs/2403.00001"/>
      </entry>
    </feed>"""
    rows = parse_arxiv_xml(xml)
    assert rows[0]["title"] == "Currency mean reversion"
    assert "2403.00001" in rows[0]["link"]


def test_cot_column_and_gold_filter():
    df = pd.DataFrame(
        {
            "Market and Exchange Names": [
                "GOLD - COMMODITY EXCHANGE INC.",
                "MICRO GOLD - COMMODITY EXCHANGE INC.",
                "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
            ],
            "As of Date in Form YYYY-MM-DD": ["2024-01-02", "2024-01-02", "2024-01-02"],
            "Open Interest (All)": [200000, 1000, 300000],
            "Noncommercial Positions-Long (All)": [120000, 10, 1],
            "Noncommercial Positions-Short (All)": [80000, 5, 2],
        }
    )
    assert "Market" in _find_col(df, "market_and_exchange")
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as zf:
        zf.writestr("annual.txt", df.to_csv(index=False))
    parsed = _parse_annual(raw.getvalue())
    gold = gold_series(parsed)
    assert len(gold) == 1
    assert "MICRO" not in gold.iloc[0]["market"].upper()
    assert gold.iloc[0]["nc_net"] == 40000


def test_named_series_exact_market():
    cot = pd.DataFrame(
        {
            "market": [
                "EURO FX - CHICAGO MERCANTILE EXCHANGE",
                "EURO FX/JAPANESE YEN XRATE - CHICAGO MERCANTILE EXCHANGE",
            ],
            "asof": ["2024-01-02", "2024-01-02"],
            "oi": [100.0, 50.0],
            "nc_long": [80.0, 10.0],
            "nc_short": [20.0, 5.0],
        }
    )
    g = named_series(cot, "EURO FX - CHICAGO MERCANTILE EXCHANGE")
    assert len(g) == 1
    assert g.iloc[0]["nc_net"] == 60.0


def test_fade_crowded_side_quote_convention():
    assert _fade_crowded_side("XAUUSD", 1.0) == "short"
    assert _fade_crowded_side("EURUSD", 1.0) == "short"
    assert _fade_crowded_side("USDJPY", 1.0) == "long"
    assert _fade_crowded_side("USDJPY", -1.0) == "short"


def test_rolling_last_pct_ends_at_one_for_max():
    s = pd.Series([0.0, 1.0, 2.0, 3.0])
    pct = rolling_last_pct(s, 4)
    assert abs(float(pct.iloc[-1]) - 1.0) < 1e-9


def test_cot_spec_fade_has_treatment_and_baseline():
    bdays = pd.bdate_range("2021-01-04", periods=900, tz="UTC")
    rng = np.random.default_rng(1)
    close = 1800 + np.cumsum(rng.normal(0, 4, len(bdays)))
    df = pd.DataFrame(
        {
            "time": bdays,
            "open": close,
            "high": close + 10,
            "low": close - 10,
            "close": close,
        }
    )
    tues = pd.date_range("2021-01-05", periods=160, freq="W-TUE", tz="UTC")
    wave = np.sin(np.linspace(0, 10, len(tues)))
    cot = pd.DataFrame(
        {
            "market": "GOLD - COMMODITY EXCHANGE INC.",
            "asof": tues,
            "oi": 300000.0,
            "nc_long": 120000 + 70000 * wave,
            "nc_short": 120000 - 70000 * wave,
        }
    )
    cot["nc_net"] = cot["nc_long"] - cot["nc_short"]
    cot["nc_net_oi"] = cot["nc_net"] / cot["oi"]
    ev = cot_spec_fade_events(df, "XAUUSD", {}, 240, 50, cot_df=cot)
    assert not ev.empty
    assert bool(ev["treatment"].any())
    assert bool((~ev["treatment"]).any())
    assert "r_mult" in ev.columns


def test_sentiment_csv(tmp_path):
    path = tmp_path / "ig.csv"
    path.write_text("date,instrument,long_pct\n2024-01-01,XAUUSD,72\n2024-01-08,XAUUSD,81\n", encoding="utf-8")
    df = load_sentiment_csv(path)
    assert len(df) == 2
    assert df.iloc[-1]["long_pct"] == 81


def test_gold_dxy_relink_has_treatment_and_baseline():
    from ats.hypotheses.gold_dxy_relink import gold_dxy_relink_events

    n = 500
    idx = pd.bdate_range("2021-01-04", periods=n, tz="UTC")
    rng = np.random.default_rng(3)
    dxy = 100 + np.cumsum(rng.normal(0, 0.25, n))
    gold = np.empty(n)
    gold[0] = 1800.0
    for i in range(1, n):
        d_ret = (dxy[i] - dxy[i - 1]) / dxy[i - 1]
        beta = -0.9 if i < 320 else 0.9
        gold[i] = gold[i - 1] * (1 + beta * d_ret + rng.normal(0, 0.003))
    df = pd.DataFrame(
        {
            "time": idx,
            "open": gold,
            "high": np.asarray(gold) + 12,
            "low": np.asarray(gold) - 12,
            "close": gold,
        }
    )
    dxy_s = pd.Series(dxy, index=idx)
    ev = gold_dxy_relink_events(
        df,
        "XAUUSD",
        {"z_window": 80, "corr_window": 40, "extreme_z": 0.8, "mid_z": 0.25},
        240,
        50,
        dxy=dxy_s,
    )
    assert not ev.empty
    assert bool(ev["treatment"].any())
    assert bool((~ev["treatment"]).any())
