"""FX-only intraday/swing tests. No gold, no equities, not a reopen of H12/H27."""

from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ats.data.calendar import load_calendar
from ats.hypotheses.m15_micro import _trade
from ats.timeutil import cost_price, pip_size

LONDON = ZoneInfo("Europe/London")
NY = ZoneInfo("America/New_York")


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
    delay = int(params.get("fill_delay_bars", 0))
    follow = bool(params.get("follow_gap", False))
    use_bar = bool(params.get("use_bar_spread", False))
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
            if follow:
                side = "long" if move > 0 else "short"
            else:
                side = "short" if move > 0 else "long"
            fill_i = i + delay
            if fill_i >= n:
                continue
            trade_cost = cost
            if use_bar and "spread" in work.columns:
                pts = float(work.at[fill_i, "spread"])
                if np.isfinite(pts) and pts >= 0:
                    # MT5 copy_rates spread is points; 10 points = 1 pip on these majors.
                    trade_cost = cost_price(symbol, pts / 10.0, slippage_pips)
            row = _trade(work, fill_i - 1, side, True, symbol, trade_cost, horizon, stop_atr, target_atr)
            if row:
                row["gap_hours"] = gap_h
                row["reopen_dow"] = int(work.at[i, "wd"])
                row["gap_atr"] = abs(move) / atr
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


def monday_cash_gap_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Fade Friday-close to Monday cash-open vs Tuesday-close to Wednesday cash-open."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    min_atr = float(params.get("min_gap_atr", 0.15))
    cash_h = int(params.get("cash_hour", 8))
    tz = ZoneInfo(str(params.get("tz", "Europe/London")))
    use_bar = bool(params.get("use_bar_spread", False))
    local = pd.to_datetime(work["time"], utc=True).dt.tz_convert(tz)
    work["lh"] = local.dt.hour
    work["ldow"] = local.dt.dayofweek
    work["ldate"] = local.dt.date
    last_close = {}
    for i in range(len(work)):
        last_close[work.at[i, "ldate"]] = float(work.at[i, "close"])
    rows = []
    n = len(work)
    for i in range(1, n):
        if int(work.at[i, "lh"]) != cash_h:
            continue
        dow = int(work.at[i, "ldow"])
        day = work.at[i, "ldate"]
        if dow == 0:
            prior = day - timedelta(days=3)
            treat = True
        elif dow == 2:
            prior = day - timedelta(days=1)
            treat = False
        else:
            continue
        prev_c = last_close.get(prior)
        if prev_c is None:
            continue
        atr = float(work.at[i - 1, "atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue
        move = float(work.at[i, "open"]) - prev_c
        if abs(move) < min_atr * atr:
            continue
        side = "short" if move > 0 else "long"
        trade_cost = cost
        if use_bar and "spread" in work.columns:
            pts = float(work.at[i, "spread"])
            if np.isfinite(pts) and pts >= 0:
                trade_cost = cost_price(symbol, pts / 10.0, slippage_pips)
        row = _trade(work, i - 1, side, treat, symbol, trade_cost, horizon, stop_atr, target_atr)
        if row:
            row["gap_atr"] = abs(move) / atr
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


def overnight_intraday_fade_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Fade overnight return at London open vs follow that same overnight return.

    Overnight = prior London overnight_hour close → today signal_hour close.
    Weekend gaps skipped (max_gap_hours). Distinct from H40 (NY-day fade at
    Asia), H79 (unconditional overnight short), H102 (cross-section).
    """
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    signal_hour = int(params.get("signal_hour_london", 8))
    overnight_hour = int(params.get("overnight_hour_london", 21))
    min_prior = float(params.get("min_prior_atr", 0.25))
    max_gap_h = float(params.get("max_gap_hours", 14))
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["lh"] = ldn.dt.hour
    work["lm"] = ldn.dt.minute
    work["ts"] = pd.to_datetime(work["time"], utc=True)
    # First bar of the hour only (H1 always; M15 uses :00).
    need_minute = int(params.get("signal_minute", 0))
    ovn_idx: dict = {}
    rows = []
    for i in range(len(work)):
        if int(work.at[i, "lh"]) == overnight_hour and int(work.at[i, "lm"]) == need_minute:
            ovn_idx[work.at[i, "ts"].date()] = i
    for i in range(len(work)):
        if int(work.at[i, "lh"]) != signal_hour or int(work.at[i, "lm"]) != need_minute:
            continue
        ts = work.at[i, "ts"]
        if int(ts.dayofweek) > 4:
            continue
        # Prior London date that should hold overnight_hour (yesterday).
        prior_day = (ts - pd.Timedelta(hours=12)).date()
        j = ovn_idx.get(prior_day)
        if j is None or j >= i:
            continue
        gap_h = (ts - work.at[j, "ts"]).total_seconds() / 3600.0
        if gap_h <= 0 or gap_h > max_gap_h:
            continue
        atr = float(work.at[i, "atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue
        move = float(work.at[i, "close"]) - float(work.at[j, "close"])
        if abs(move) < min_prior * atr:
            continue
        fade = "short" if move > 0 else "long"
        follow = "long" if fade == "short" else "short"
        for treat, side in (True, fade), (False, follow):
            row = _trade(work, i, side, treat, symbol, cost, horizon, stop_atr, target_atr)
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def m15_to_h1(df: pd.DataFrame) -> pd.DataFrame:
    """Resample prepared M15 OHLC to H1 for single-symbol TSMOM (no separate H1 pull)."""
    work = df.dropna(subset=["atr"]).copy()
    if work.empty:
        return work
    work["time"] = pd.to_datetime(work["time"], utc=True)
    work = work.set_index("time").sort_index()
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "atr": "last",
    }
    if "tick_volume" in work.columns:
        agg["tick_volume"] = "sum"
    out = work.resample("1h", label="left", closed="left").agg(agg).dropna(subset=["open", "close", "atr"])
    return out.reset_index()


def overnight_xs_fade_events(frames: dict, params: dict, settings: dict) -> pd.DataFrame:
    """Fade overnight winner/loser at London 08:00 vs follow that overnight XS."""
    signal_hour = int(params.get("signal_hour_london", 8))
    prior_hour = int(params.get("prior_hour_london", 16))
    horizon = int(params.get("horizon_bars", 8))
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    costs = settings["costs"]
    prepared: dict[str, pd.DataFrame] = {}
    idx_sig: dict[str, dict] = {}
    idx_pri: dict[str, dict] = {}
    for symbol, df in frames.items():
        work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
        if work.empty:
            continue
        ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
        work["_lh"] = ldn.dt.hour
        work["_ldate"] = ldn.dt.date
        sig, pri = {}, {}
        for i in range(len(work)):
            hour = int(work.at[i, "_lh"])
            day = work.at[i, "_ldate"]
            if hour == signal_hour:
                sig[day] = i
            elif hour == prior_hour:
                pri[day] = i
        if sig and pri:
            prepared[symbol] = work
            idx_sig[symbol] = sig
            idx_pri[symbol] = pri
    symbols = sorted(prepared)
    if len(symbols) < 3:
        return pd.DataFrame()
    dates = set.intersection(*(set(idx_sig[s]) for s in symbols))
    rows = []
    for d in sorted(dates):
        rets = {}
        idxs = {}
        skip = False
        for sym in symbols:
            i8 = idx_sig[sym][d]
            if i8 < 1:
                skip = True
                break
            prior_dates = [p for p in idx_pri[sym] if p < d]
            if not prior_dates:
                skip = True
                break
            i16 = idx_pri[sym][max(prior_dates)]
            work = prepared[sym]
            atr = float(work.at[i8 - 1, "atr"])
            if not np.isfinite(atr) or atr <= 0:
                skip = True
                break
            rets[sym] = (float(work.at[i8, "open"]) - float(work.at[i16, "close"])) / atr
            idxs[sym] = i8
        if skip or len(set(rets.values())) < 2:
            continue
        winner = max(rets, key=rets.get)
        loser = min(rets, key=rets.get)
        if winner == loser:
            continue
        for treat, win_side, lose_side in ((True, "short", "long"), (False, "long", "short")):
            for sym, side in ((winner, win_side), (loser, lose_side)):
                work = prepared[sym]
                spread = float(costs["spread_pips"].get(sym, 1.5))
                slip = float((costs.get("slippage_by_symbol") or {}).get(sym, costs["slippage_pips"]))
                cost = cost_price(sym, spread, slip)
                row = _trade(work, idxs[sym] - 1, side, treat, sym, cost, horizon, stop_atr, target_atr)
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
    n_start = int(params.get("month_start_days", 0))
    n_end = int(params["month_end_days"]) if "month_end_days" in params else (0 if n_start else 2)
    base_days = set(int(x) for x in params.get("base_days", [10, 11, 12]))
    side = short_foreign_side(symbol)
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["lh"] = ldn.dt.hour
    work["ldate"] = ldn.dt.date
    work["ym"] = ldn.dt.strftime("%Y-%m")
    treat_dates: set = set()
    for _, g in work.groupby("ym", sort=True):
        days = sorted({d for d, h in zip(g["ldate"], g["lh"]) if int(h) == hour})
        if n_end:
            treat_dates.update(days[-n_end:])
        if n_start:
            treat_dates.update(days[:n_start])
    rows = []
    n = len(work)
    for i in range(1, n):
        if int(work.at[i, "lh"]) != hour:
            continue
        d = work.at[i, "ldate"]
        if d in treat_dates:
            treat = True
        elif d.day in base_days:
            treat = False
        else:
            continue
        row = _trade(work, i - 1, side, treat, symbol, cost, horizon, stop_atr, target_atr)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def wm_postfix_fade_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Fade the pre-WM run at 16:00 London vs fade the late-morning run at noon."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    t0 = int(params.get("treat_start_london", 14))
    t1 = int(params.get("treat_end_london", 16))
    b0 = int(params.get("base_start_london", 10))
    b1 = int(params.get("base_end_london", 12))
    min_prior = float(params.get("min_prior_atr", 0.25))
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["lh"] = ldn.dt.hour
    work["ldate"] = ldn.dt.date
    rows = []
    for _, g in work.groupby("ldate", sort=True):
        atr = float(g["atr"].iloc[-1]) if len(g) else 0.0
        if not np.isfinite(atr) or atr <= 0:
            continue
        b_t0, b_t1 = g[g["lh"] == t0], g[g["lh"] == t1]
        b_b0, b_b1 = g[g["lh"] == b0], g[g["lh"] == b1]
        if not b_t0.empty and not b_t1.empty:
            prior = float(work.at[b_t1.index[0], "close"]) - float(work.at[b_t0.index[0], "close"])
            if abs(prior) >= min_prior * atr:
                side = "short" if prior > 0 else "long"
                row = _trade(work, int(b_t1.index[0]) - 1, side, True, symbol, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
        if not b_b0.empty and not b_b1.empty:
            prior = float(work.at[b_b1.index[0], "close"]) - float(work.at[b_b0.index[0], "close"])
            if abs(prior) >= min_prior * atr:
                side = "short" if prior > 0 else "long"
                row = _trade(work, int(b_b1.index[0]) - 1, side, False, symbol, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
    return pd.DataFrame(rows)


def big_figure_fade_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Fade first H1 tag of a 100-pip figure vs first tag of yesterday's open."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    step = float(params.get("figure_pips", 100)) * pip_size(symbol)
    near_atr = float(params.get("near_atr", 0.40))
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["ldate"] = ldn.dt.date
    daily = work.groupby("ldate", sort=True).agg(o=("open", "first"))
    daily["prev_open"] = daily["o"].shift(1)
    rows = []
    for day, g in work.groupby("ldate", sort=True):
        prev_open = daily.at[day, "prev_open"] if day in daily.index else np.nan
        saw_r = saw_o = False
        for i in g.index:
            o, hi, lo = float(work.at[i, "open"]), float(work.at[i, "high"]), float(work.at[i, "low"])
            atr = float(work.at[i, "atr"])
            rnd = round(o / step) * step
            if not saw_r and abs(o - rnd) <= near_atr * atr and lo <= rnd <= hi:
                side = "short" if o < rnd else "long"
                row = _trade(work, int(i), side, True, symbol, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
                saw_r = True
            if not saw_o and np.isfinite(prev_open) and lo <= float(prev_open) <= hi:
                side = "short" if o < float(prev_open) else "long"
                row = _trade(work, int(i), side, False, symbol, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
                saw_o = True
            if saw_r and saw_o:
                break
    return pd.DataFrame(rows)


def stop_pool_20_fade_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Fade a new 20-H1 extreme vs fade a new 5-H1 extreme."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    n_treat = int(params.get("treat_lookback", 20))
    n_base = int(params.get("base_lookback", 5))
    rows = []
    n = len(work)
    start = max(n_treat, n_base)
    for i in range(start, n):
        hi = float(work.at[i, "high"])
        lo = float(work.at[i, "low"])
        prev20_h = float(work["high"].iloc[i - n_treat : i].max())
        prev20_l = float(work["low"].iloc[i - n_treat : i].min())
        prev5_h = float(work["high"].iloc[i - n_base : i].max())
        prev5_l = float(work["low"].iloc[i - n_base : i].min())
        treat_side = None
        if hi > prev20_h:
            treat_side = "short"
        elif lo < prev20_l:
            treat_side = "long"
        base_side = None
        if hi > prev5_h:
            base_side = "short"
        elif lo < prev5_l:
            base_side = "long"
        if treat_side is not None:
            row = _trade(work, i, treat_side, True, symbol, cost, horizon, stop_atr, target_atr)
            if row:
                rows.append(row)
        elif base_side is not None:
            row = _trade(work, i, base_side, False, symbol, cost, horizon, stop_atr, target_atr)
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def london_lunch_fade_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Fade the London morning at lunch vs the same fade at London open."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    lookback = int(params.get("lookback_bars", 3))
    treat_hour = int(params.get("lunch_hour_london", 12))
    base_hour = int(params.get("open_hour_london", 8))
    min_prior = float(params.get("min_prior_atr", 0.25))
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["lh"] = ldn.dt.hour
    rows = []
    n = len(work)
    for i in range(lookback, n):
        hour = int(work.at[i, "lh"])
        if hour == treat_hour:
            treat = True
        elif hour == base_hour:
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


def traditional_carry_side(symbol: str) -> str:
    """Textbook G10 vs USD: long AUD/NZD, fund JPY/CHF. Frozen. Do not flip."""
    base = "".join(ch for ch in symbol.upper() if ch.isalpha())[:6]
    if base in {"AUDUSD", "NZDUSD", "USDJPY", "USDCHF"}:
        return "long"
    raise ValueError(f"H52 carry side is not defined for {symbol}")


def carry_roll_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Hold textbook carry through the NY 17:00 tom-next vs the same hold at London 08:00."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    treat_hour = int(params.get("treat_hour_london", 16))
    base_hour = int(params.get("base_hour_london", 8))
    side = traditional_carry_side(symbol)
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["lh"] = ldn.dt.hour
    work["dow"] = ldn.dt.dayofweek
    rows = []
    for i in range(1, len(work)):
        if int(work.at[i, "dow"]) > 3:
            continue
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


def tokyo_fix_follow_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Follow the M15 run into the 9:55 Tokyo USDJPY fix vs the same run at 11:30 JST."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    min_prior = float(params.get("min_prior_atr", 0.25))
    treat_h, treat_m = int(params.get("treat_hour", 9)), int(params.get("treat_minute", 30))
    base_h, base_m = int(params.get("base_hour", 11)), int(params.get("base_minute", 30))
    tokyo = ZoneInfo("Asia/Tokyo")
    jst = pd.to_datetime(work["time"], utc=True).dt.tz_convert(tokyo)
    work["th"] = jst.dt.hour
    work["tm"] = jst.dt.minute
    work["tdate"] = jst.dt.date
    rows = []
    for _, g in work.groupby("tdate", sort=True):
        for treat, hour, minute in (True, treat_h, treat_m), (False, base_h, base_m):
            hit = g[(g["th"] == hour) & (g["tm"] == minute)]
            prev = g[(g["th"] == hour) & (g["tm"] == minute - 15)]
            if hit.empty or prev.empty:
                continue
            i = int(hit.index[-1])
            j = int(prev.index[-1])
            if j >= i:
                continue
            atr = float(work.at[i, "atr"])
            if not np.isfinite(atr) or atr <= 0:
                continue
            prior = float(work.at[i, "close"]) - float(work.at[j, "open"])
            if abs(prior) < min_prior * atr:
                continue
            side = "long" if prior > 0 else "short"
            row = _trade(work, i, side, treat, symbol, cost, horizon, stop_atr, target_atr)
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def nyse_open_fx_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Unconditional long the non-USD at NYSE 9:30 vs the same long at 8:00 NY."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    treat_h = int(params.get("treat_hour_ny", 9))
    treat_m = int(params.get("treat_minute_ny", 30))
    base_h = int(params.get("base_hour_ny", 8))
    base_m = int(params.get("base_minute_ny", 0))
    side = long_foreign_side(symbol)
    ny = pd.to_datetime(work["time"], utc=True).dt.tz_convert(NY)
    work["nh"] = ny.dt.hour
    work["nm"] = ny.dt.minute
    work["dow"] = ny.dt.dayofweek
    rows = []
    for i in range(len(work)):
        if int(work.at[i, "dow"]) > 4:
            continue
        hour = int(work.at[i, "nh"])
        minute = int(work.at[i, "nm"])
        if hour == treat_h and minute == treat_m:
            treat = True
        elif hour == base_h and minute == base_m:
            treat = False
        else:
            continue
        row = _trade(work, i, side, treat, symbol, cost, horizon, stop_atr, target_atr)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def two_day_streak_fade_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Fade a 2-session same-direction close at London 16:00 vs fade a 1-session close."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    signal_hour = int(params.get("signal_hour_london", 16))
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["lh"] = ldn.dt.hour
    work["ldate"] = ldn.dt.date
    daily = work.groupby("ldate", sort=True).agg(c=("close", "last"))
    daily["ret"] = daily["c"].diff()
    rows = []
    dates = list(daily.index)
    for k, day in enumerate(dates):
        # Signal at 16:00 uses completed sessions only (no same-day close leak).
        if k < 3:
            continue
        r_old = float(daily.at[dates[k - 2], "ret"])
        r_new = float(daily.at[dates[k - 1], "ret"])
        if r_new == 0 or not np.isfinite(r_new):
            continue
        treat = bool(np.sign(r_new) == np.sign(r_old) and r_old != 0 and np.isfinite(r_old))
        g = work.loc[work["ldate"] == day]
        hit = g[g["lh"] == signal_hour]
        if hit.empty:
            continue
        i = int(hit.index[-1])
        side = "short" if r_new > 0 else "long"
        row = _trade(work, i, side, treat, symbol, cost, horizon, stop_atr, target_atr)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def _snapped_print_local(event: str, datetime_utc, tz) -> pd.Timestamp | None:
    """This lab's FF cache is one NY calendar date early vs BLS/Fed print dates."""
    clocks = {
        "Non-Farm Employment Change": (8, 30),
        "CPI m/m": (8, 30),
        "FOMC Statement": (14, 0),
    }
    if event not in clocks:
        return None
    ny = pd.Timestamp(datetime_utc)
    if ny.tzinfo is None:
        ny = ny.tz_localize("UTC")
    ny = ny.tz_convert(tz).normalize() + pd.Timedelta(days=1)
    hour, minute = clocks[event]
    return ny.replace(hour=hour, minute=minute, second=0, microsecond=0)


def print_window_fade_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2, calendar=None) -> pd.DataFrame:
    """Fade the print-hour bar on named USD releases vs the same clock on a blank day."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon, stop_atr, target_atr, cost = _params(params, symbol, spread_pips, slippage_pips)
    tz = ZoneInfo(str(params.get("tz", "America/New_York")))
    treat_h = int(params.get("treat_hour", 8))
    treat_m = int(params.get("treat_minute", 0))
    names = {str(x) for x in params["event_names"]}
    weekday = params.get("same_weekday")
    impact = str(params.get("min_impact", "high")).lower()
    cal = calendar if calendar is not None else load_calendar()
    if cal is None or cal.empty:
        return pd.DataFrame()
    usd = cal[(cal["currency"].astype(str).str.upper() == "USD") & (cal["impact"].astype(str).str.lower() == impact)].copy()
    if usd.empty:
        return pd.DataFrame()
    et = pd.to_datetime(usd["datetime_utc"], utc=True)
    named_keys: set = set()
    for _, row in usd.iterrows():
        name = str(row["event"])
        if name not in names:
            continue
        snapped = _snapped_print_local(name, row["datetime_utc"], tz)
        if snapped is None:
            local_e = pd.Timestamp(row["datetime_utc"])
            if local_e.tzinfo is None:
                local_e = local_e.tz_localize("UTC")
            local_e = local_e.tz_convert(tz)
            named_keys.add((local_e.date(), int(local_e.hour)))
        else:
            named_keys.add((snapped.date(), int(snapped.hour)))
    any_local = et.dt.tz_convert(tz)
    any_keys = set(zip(any_local.dt.date, any_local.dt.hour)) | named_keys
    local = pd.to_datetime(work["time"], utc=True).dt.tz_convert(tz)
    work["h"] = local.dt.hour
    work["m"] = local.dt.minute
    work["dow"] = local.dt.dayofweek
    work["ldate"] = local.dt.date
    rows = []
    for i in range(len(work)):
        if int(work.at[i, "h"]) != treat_h or int(work.at[i, "m"]) != treat_m:
            continue
        dow = int(work.at[i, "dow"])
        if dow > 4:
            continue
        if weekday is not None and dow != int(weekday):
            continue
        key = (work.at[i, "ldate"], treat_h)
        if key in named_keys:
            treat = True
        elif key not in any_keys:
            treat = False
        else:
            continue
        o, c = float(work.at[i, "open"]), float(work.at[i, "close"])
        if c == o:
            continue
        side = "short" if c > o else "long"
        row = _trade(work, i, side, treat, symbol, cost, horizon, stop_atr, target_atr)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)
