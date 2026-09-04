"""FX-only intraday/swing tests. No gold, no equities, not a reopen of H12/H27."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ats.hypotheses.m15_micro import _trade
from ats.timeutil import cost_price

LONDON = ZoneInfo("Europe/London")


def _params(params: dict, symbol: str, spread_pips: float, slippage_pips: float) -> tuple:
    horizon = int(params.get("horizon_bars", 6))
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    return horizon, stop_atr, target_atr, cost


def long_foreign_side(symbol: str) -> str:
    """Buy the non-USD. USDXXX quotes: short the pair. XXXUSD quotes: long the pair."""
    base = "".join(ch for ch in symbol.upper() if ch.isalpha())[:6]
    if base.startswith("USD"):
        return "short"
    return "long"


def short_foreign_side(symbol: str) -> str:
    return "short" if long_foreign_side(symbol) == "long" else "long"


def postfix_usd_fade_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """After London 16:00, buy the non-USD vs the same hold from 08:00 (Krohn postfix)."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    treat_hour = int(params.get("treat_hour_london", 16))
    base_hour = int(params.get("base_hour_london", 8))
    side = long_foreign_side(symbol)
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["lh"] = ldn.dt.hour
    rows = []
    for i in range(1, len(work)):
        hour = int(work.at[i, "lh"])
        if hour == treat_hour:
            treat = True
        elif hour == base_hour:
            treat = False
        else:
            continue
        row = _trade(work, i - 1, side, treat, symbol, cost, horizon, stop_atr, target_atr)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def pre_ecb_usd_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Into the ECB 14:15 Frankfurt window, sell the non-USD vs the same hold at 00:00 London."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    treat_hour = int(params.get("treat_hour_london", 7))
    base_hour = int(params.get("base_hour_london", 0))
    side = short_foreign_side(symbol)
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["lh"] = ldn.dt.hour
    rows = []
    for i in range(1, len(work)):
        hour = int(work.at[i, "lh"])
        if hour == treat_hour:
            treat = True
        elif hour == base_hour:
            treat = False
        else:
            continue
        row = _trade(work, i - 1, side, treat, symbol, cost, horizon, stop_atr, target_atr)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def london_asia_break_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Follow first London break of the Asia range vs fade that same break."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    asia_start = int(params.get("asia_start_london", 0))
    asia_end = int(params.get("asia_end_london", 7))
    ldn_start = int(params.get("london_start", 8))
    ldn_end = int(params.get("london_end", 11))
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["lh"] = ldn.dt.hour
    work["ldate"] = ldn.dt.date
    rows = []
    for _, g in work.groupby("ldate", sort=True):
        asia = g[(g["lh"] >= asia_start) & (g["lh"] < asia_end)]
        london = g[(g["lh"] >= ldn_start) & (g["lh"] < ldn_end)]
        if asia.empty or london.empty:
            continue
        rh, rl = float(asia["high"].max()), float(asia["low"].min())
        if rh - rl <= cost * 2:
            continue
        for i in london.index:
            close = float(work.at[i, "close"])
            follow = None
            if close > rh:
                follow = "long"
            elif close < rl:
                follow = "short"
            if follow is None:
                continue
            fade = "short" if follow == "long" else "long"
            for treat, side in (True, follow), (False, fade):
                row = _trade(work, int(i), side, treat, symbol, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
            break
    return pd.DataFrame(rows)


def h1_tsmom_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Once per day at 16:00 London, follow the prior 5-day return vs fade it."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    lookback = int(params.get("lookback_bars", 120))
    signal_hour = int(params.get("signal_hour_london", 16))
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["lh"] = ldn.dt.hour
    rows = []
    n = len(work)
    for i in range(lookback, n):
        if int(work.at[i, "lh"]) != signal_hour:
            continue
        prior = float(work.at[i, "close"]) - float(work.at[i - lookback, "close"])
        if prior == 0:
            continue
        follow = "long" if prior > 0 else "short"
        fade = "short" if follow == "long" else "long"
        for treat, side in (True, follow), (False, fade):
            row = _trade(work, i, side, treat, symbol, cost, horizon, stop_atr, target_atr)
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def ranaldo_local_hours_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Short EUR during London hours vs the same short during US hours (Ranaldo)."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    treat_hour = int(params.get("treat_hour_london", 8))
    base_hour = int(params.get("base_hour_london", 16))
    side = short_foreign_side(symbol)
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["lh"] = ldn.dt.hour
    rows = []
    for i in range(1, len(work)):
        hour = int(work.at[i, "lh"])
        if hour == treat_hour:
            treat = True
        elif hour == base_hour:
            treat = False
        else:
            continue
        row = _trade(work, i - 1, side, treat, symbol, cost, horizon, stop_atr, target_atr)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def inside_bar_break_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Follow a close beyond the mother bar after an inside bar vs fade that close."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    rows = []
    n = len(work)
    for j in range(2, n):
        m_h, m_l = float(work.at[j - 2, "high"]), float(work.at[j - 2, "low"])
        i_h, i_l = float(work.at[j - 1, "high"]), float(work.at[j - 1, "low"])
        if not (i_h <= m_h and i_l >= m_l and (i_h < m_h or i_l > m_l)):
            continue
        close = float(work.at[j, "close"])
        if close > m_h:
            follow = "long"
        elif close < m_l:
            follow = "short"
        else:
            continue
        fade = "short" if follow == "long" else "long"
        for treat, side in (True, follow), (False, fade):
            row = _trade(work, j, side, treat, symbol, cost, horizon, stop_atr, target_atr)
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def compression_expand_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Follow a new ATR expansion that was preceded by compression vs expansion with no compression."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    look = int(params.get("comp_lookback", 6))
    c_cut = float(params.get("compression_atr_ratio", 0.7))
    e_cut = float(params.get("expansion_atr_ratio", 1.3))
    avg = work["atr"].rolling(20, min_periods=20).mean()
    ratio = work["atr"] / avg.replace(0, np.nan)
    comp = ratio < c_cut
    exp = ratio > e_cut
    new_exp = exp & ~exp.shift(1, fill_value=False)
    prior_comp = comp.shift(1).astype(float).rolling(look, min_periods=1).max().fillna(0) > 0
    rows = []
    n = len(work)
    for i in range(20, n):
        if not bool(new_exp.iloc[i]):
            continue
        if float(work.at[i, "close"]) == float(work.at[i, "open"]):
            continue
        treat = bool(prior_comp.iloc[i])
        side = "long" if float(work.at[i, "close"]) > float(work.at[i, "open"]) else "short"
        row = _trade(work, i, side, treat, symbol, cost, horizon, stop_atr, target_atr)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def friday_flatten_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Fade the prior run at Friday 15:00 London vs the same fade on Wednesday."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    lookback = int(params.get("lookback_bars", 4))
    min_prior = float(params.get("min_prior_atr", 0.25))
    hour = int(params.get("flatten_hour_london", 15))
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["lh"] = ldn.dt.hour
    work["wd"] = ldn.dt.dayofweek
    rows = []
    n = len(work)
    for i in range(lookback, n):
        if int(work.at[i, "lh"]) != hour:
            continue
        wd = int(work.at[i, "wd"])
        if wd == 4:
            treat = True
        elif wd == 2:
            treat = False
        else:
            continue
        atr = float(work.at[i, "atr"])
        prior = float(work.at[i - 1, "close"]) - float(work.at[i - lookback, "close"])
        if not np.isfinite(atr) or atr <= 0 or abs(prior) < min_prior * atr:
            continue
        side = "short" if prior > 0 else "long"
        row = _trade(work, i - 1, side, treat, symbol, cost, horizon, stop_atr, target_atr)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def overlap_continuation_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Follow London morning into NY overlap vs follow the same morning after London close."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    morning_start = int(params.get("morning_start_london", 8))
    morning_end = int(params.get("morning_end_london", 12))
    treat_hour = int(params.get("treat_hour_london", 13))
    base_hour = int(params.get("base_hour_london", 16))
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["lh"] = ldn.dt.hour
    work["ldate"] = ldn.dt.date
    rows = []
    for _, g in work.groupby("ldate", sort=True):
        b_start = g[g["lh"] == morning_start]
        b_end = g[g["lh"] == morning_end]
        b_treat = g[g["lh"] == treat_hour]
        b_base = g[g["lh"] == base_hour]
        if b_start.empty or b_end.empty or b_treat.empty or b_base.empty:
            continue
        prior = float(work.at[b_end.index[0], "close"]) - float(work.at[b_start.index[0], "close"])
        if prior == 0:
            continue
        follow = "long" if prior > 0 else "short"
        for treat, idx in (True, int(b_treat.index[0])), (False, int(b_base.index[0])):
            row = _trade(work, idx - 1, follow, treat, symbol, cost, horizon, stop_atr, target_atr)
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def weekend_gap_fx_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Fade the weekend halt gap vs fade Wednesday's London-open overnight."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    min_gap = float(params.get("min_gap_hours", 36))
    min_atr = float(params.get("min_gap_atr", 0.15))
    base_hour = int(params.get("base_hour_london", 8))
    prior_hour = int(params.get("prior_hour_london", 16))
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    times = pd.to_datetime(work["time"], utc=True)
    work["lh"] = ldn.dt.hour
    work["wd"] = ldn.dt.dayofweek
    gaps = times.diff().dt.total_seconds() / 3600.0
    rows = []
    n = len(work)
    last_prior = None
    for i in range(1, n):
        hour = int(work.at[i, "lh"])
        if hour == prior_hour:
            last_prior = i
        gap_h = float(gaps.iloc[i]) if np.isfinite(gaps.iloc[i]) else 0.0
        atr = float(work.at[i, "atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue
        if gap_h >= min_gap:
            move = float(work.at[i, "open"]) - float(work.at[i - 1, "close"])
            if abs(move) < min_atr * atr:
                continue
            side = "short" if move > 0 else "long"
            row = _trade(work, i - 1, side, True, symbol, cost, horizon, stop_atr, target_atr)
            if row:
                rows.append(row)
            continue
        if hour == base_hour and int(work.at[i, "wd"]) == 2 and last_prior is not None:
            move = float(work.at[i, "open"]) - float(work.at[last_prior, "close"])
            if abs(move) < min_atr * atr:
                continue
            side = "short" if move > 0 else "long"
            row = _trade(work, i - 1, side, False, symbol, cost, horizon, stop_atr, target_atr)
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def tokyo_lunch_fade_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Fade USDJPY's Tokyo morning at lunch vs follow that same morning."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    lookback = int(params.get("lookback_bars", 3))
    lunch_hour = int(params.get("lunch_hour_tokyo", 12))
    tokyo = ZoneInfo("Asia/Tokyo")
    work["th"] = pd.to_datetime(work["time"], utc=True).dt.tz_convert(tokyo).dt.hour
    rows = []
    n = len(work)
    for i in range(lookback, n):
        if int(work.at[i, "th"]) != lunch_hour:
            continue
        prior = float(work.at[i - 1, "close"]) - float(work.at[i - lookback, "close"])
        if prior == 0:
            continue
        fade = "short" if prior > 0 else "long"
        follow = "long" if prior > 0 else "short"
        for treat, side in (True, fade), (False, follow):
            row = _trade(work, i - 1, side, treat, symbol, cost, horizon, stop_atr, target_atr)
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def xs_momentum_events(frames: dict, params: dict, settings: dict) -> pd.DataFrame:
    """Long the 5-day winner major and short the loser vs the reverse (Menkhoff XS)."""
    lookback = int(params.get("lookback_bars", 120))
    signal_hour = int(params.get("signal_hour_london", 16))
    horizon = int(params.get("horizon_bars", 24))
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    costs = settings["costs"]
    prepared: dict[str, pd.DataFrame] = {}
    hour_idx: dict[str, dict] = {}
    for symbol, df in frames.items():
        work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
        if work.empty:
            continue
        ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
        work["_lh"] = ldn.dt.hour
        work["_ldate"] = ldn.dt.date
        mapping = {}
        for i in range(len(work)):
            if int(work.at[i, "_lh"]) == signal_hour:
                mapping[work.at[i, "_ldate"]] = i
        if mapping:
            prepared[symbol] = work
            hour_idx[symbol] = mapping
    symbols = sorted(prepared)
    if len(symbols) < 3:
        return pd.DataFrame()
    dates = set.intersection(*(set(hour_idx[s]) for s in symbols))
    rows = []
    for d in sorted(dates):
        rets = {}
        idxs = {}
        skip = False
        for sym in symbols:
            i = hour_idx[sym][d]
            if i < lookback:
                skip = True
                break
            work = prepared[sym]
            rets[sym] = float(work.at[i, "close"]) - float(work.at[i - lookback, "close"])
            idxs[sym] = i
        if skip or len(set(rets.values())) < 2:
            continue
        winner = max(rets, key=rets.get)
        loser = min(rets, key=rets.get)
        if winner == loser:
            continue
        for treat, win_side, lose_side in ((True, "long", "short"), (False, "short", "long")):
            for sym, side in ((winner, win_side), (loser, lose_side)):
                work = prepared[sym]
                spread = float(costs["spread_pips"].get(sym, 1.5))
                slip = float((costs.get("slippage_by_symbol") or {}).get(sym, costs["slippage_pips"]))
                cost = cost_price(sym, spread, slip)
                row = _trade(work, idxs[sym], side, treat, sym, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
    return pd.DataFrame(rows)


def ny_close_asia_fade_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Fade the NY-day EURUSD run at NY close vs fade the London morning at noon."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    ny_start = int(params.get("ny_start_london", 13))
    ny_end = int(params.get("ny_end_london", 20))
    treat_hour = int(params.get("treat_hour_london", 21))
    ldn_start = int(params.get("london_start", 8))
    ldn_end = int(params.get("london_end", 12))
    base_hour = int(params.get("base_hour_london", 12))
    min_prior = float(params.get("min_prior_atr", 0.25))
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["lh"] = ldn.dt.hour
    work["ldate"] = ldn.dt.date
    rows = []
    for _, g in work.groupby("ldate", sort=True):
        b_ny_s, b_ny_e = g[g["lh"] == ny_start], g[g["lh"] == ny_end]
        b_treat = g[g["lh"] == treat_hour]
        b_ldn_s, b_ldn_e = g[g["lh"] == ldn_start], g[g["lh"] == ldn_end]
        b_base = g[g["lh"] == base_hour]
        atr = float(g["atr"].iloc[-1]) if len(g) else 0.0
        if not np.isfinite(atr) or atr <= 0:
            continue
        if not b_ny_s.empty and not b_ny_e.empty and not b_treat.empty:
            prior = float(work.at[b_ny_e.index[0], "close"]) - float(work.at[b_ny_s.index[0], "close"])
            if abs(prior) >= min_prior * atr:
                side = "short" if prior > 0 else "long"
                row = _trade(work, int(b_treat.index[0]) - 1, side, True, symbol, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
        if not b_ldn_s.empty and not b_ldn_e.empty and not b_base.empty:
            prior = float(work.at[b_ldn_e.index[0], "close"]) - float(work.at[b_ldn_s.index[0], "close"])
            if abs(prior) >= min_prior * atr:
                side = "short" if prior > 0 else "long"
                row = _trade(work, int(b_base.index[0]) - 1, side, False, symbol, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
    return pd.DataFrame(rows)


def prior_day_range_fade_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Fade first H1 close through yesterday's range vs follow that close."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["ldate"] = ldn.dt.date
    rows = []
    dates = list(work.groupby("ldate", sort=True))
    for k in range(1, len(dates)):
        _, prev = dates[k - 1]
        _, today = dates[k]
        rh, rl = float(prev["high"].max()), float(prev["low"].min())
        if rh - rl <= cost * 2:
            continue
        for i in today.index:
            close = float(work.at[i, "close"])
            if close > rh:
                fade, follow = "short", "long"
            elif close < rl:
                fade, follow = "long", "short"
            else:
                continue
            for treat, side in (True, fade), (False, follow):
                row = _trade(work, int(i), side, treat, symbol, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
            break
    return pd.DataFrame(rows)


def tokyo_close_flatten_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Fade USDJPY into Tokyo cash close vs the same fade at Tokyo lunch."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    lookback = int(params.get("lookback_bars", 3))
    treat_hour = int(params.get("close_hour_tokyo", 15))
    base_hour = int(params.get("lunch_hour_tokyo", 12))
    tokyo = ZoneInfo("Asia/Tokyo")
    work["th"] = pd.to_datetime(work["time"], utc=True).dt.tz_convert(tokyo).dt.hour
    rows = []
    n = len(work)
    for i in range(lookback, n):
        hour = int(work.at[i, "th"])
        if hour == treat_hour:
            treat = True
        elif hour == base_hour:
            treat = False
        else:
            continue
        prior = float(work.at[i - 1, "close"]) - float(work.at[i - lookback, "close"])
        if prior == 0:
            continue
        side = "short" if prior > 0 else "long"
        row = _trade(work, i - 1, side, treat, symbol, cost, horizon, stop_atr, target_atr)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def month_end_usd_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Short EURUSD (long USD) on the last 2 London sessions of the month vs days 10-12."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    hour = int(params.get("signal_hour_london", 16))
    n_end = int(params.get("month_end_days", 2))
    base_days = set(int(x) for x in params.get("base_days", [10, 11, 12]))
    side = short_foreign_side(symbol)
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["lh"] = ldn.dt.hour
    work["ldate"] = ldn.dt.date
    work["ym"] = ldn.dt.strftime("%Y-%m")
    end_dates: set = set()
    for _, g in work.groupby("ym", sort=True):
        days = sorted({d for d, h in zip(g["ldate"], g["lh"]) if int(h) == hour})
        end_dates.update(days[-n_end:])
    rows = []
    n = len(work)
    for i in range(1, n):
        if int(work.at[i, "lh"]) != hour:
            continue
        d = work.at[i, "ldate"]
        if d in end_dates:
            treat = True
        elif d.day in base_days:
            treat = False
        else:
            continue
        row = _trade(work, i - 1, side, treat, symbol, cost, horizon, stop_atr, target_atr)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)
