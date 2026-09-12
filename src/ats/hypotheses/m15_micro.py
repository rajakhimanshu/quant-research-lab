"""Gold M15/M5 microstructure tests. No EMA stacks, no Dr.D retunes."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ats.hypotheses.cot_spec_fade import _forward
from ats.timeutil import cost_price

NY = ZoneInfo("America/New_York")
LONDON = ZoneInfo("Europe/London")


def _trade(
    work: pd.DataFrame,
    i: int,
    side: str,
    treat: bool,
    symbol: str,
    cost: float,
    horizon: int,
    stop_atr: float,
    target_atr: float,
) -> dict | None:
    n = len(work)
    if i + 1 + horizon >= n:
        return None
    atr = float(work.at[i, "atr"])
    if not np.isfinite(atr) or atr <= 0:
        return None
    raw_open = float(work.at[i + 1, "open"])
    if side == "long":
        entry = raw_open + cost
        stop = entry - stop_atr * atr
        target = entry + target_atr * atr
    else:
        entry = raw_open - cost
        stop = entry + stop_atr * atr
        target = entry - target_atr * atr
    hit, r_mult = _forward(work, i, side, entry, target, stop, horizon)
    return {
        "time": work.at[i + 1, "time"],
        "symbol": symbol,
        "treatment": treat,
        "side": side,
        "success": bool(hit),
        "success_cost_adj": bool(r_mult > 0),
        "r_mult": float(r_mult),
        "entry": float(entry),
        "stop": float(stop),
        "target": float(target),
        "atr": float(atr),
        "raw_open": float(raw_open),
    }


def pdh_pdl_fade_events(df, symbol, params, spread_pips=200, slippage_pips=20) -> pd.DataFrame:
    """First touch of prior-day high/low fades vs first trade through prior-day mid."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon = int(params.get("horizon_bars", 8))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    ny = pd.to_datetime(work["time"], utc=True).dt.tz_convert(NY)
    work["ny_date"] = ny.dt.date
    daily = work.groupby("ny_date", sort=True).agg(h=("high", "max"), l=("low", "min"))
    daily["pdh"] = daily["h"].shift(1)
    daily["pdl"] = daily["l"].shift(1)
    daily["mid"] = (daily["pdh"] + daily["pdl"]) / 2.0
    rows = []
    for day, g in work.groupby("ny_date", sort=True):
        if day not in daily.index:
            continue
        pdh, pdl, mid = daily.at[day, "pdh"], daily.at[day, "pdl"], daily.at[day, "mid"]
        if not (np.isfinite(pdh) and np.isfinite(pdl) and np.isfinite(mid)):
            continue
        saw_ext = saw_mid = False
        for i in g.index:
            hi, lo, close = float(work.at[i, "high"]), float(work.at[i, "low"]), float(work.at[i, "close"])
            if not saw_ext and (hi >= pdh or lo <= pdl):
                side = "short" if hi >= pdh else "long"
                row = _trade(work, int(i), side, True, symbol, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
                saw_ext = True
            if not saw_mid and lo <= mid <= hi:
                side = "short" if close >= mid else "long"
                row = _trade(work, int(i), side, False, symbol, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
                saw_mid = True
            if saw_ext and saw_mid:
                break
    return pd.DataFrame(rows)


def asia_range_london_fade_events(df, symbol, params, spread_pips=200, slippage_pips=20) -> pd.DataFrame:
    """Fade first London tag of Asia high/low vs first tag of Asia mid."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon = int(params.get("horizon_bars", 8))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    asia_end = int(params.get("asia_end_hour_london", 7))
    cash_h = int(params.get("london_hour", 8))
    cash_end = int(params.get("london_end_hour", 12))
    ldn = pd.to_datetime(work["time"], utc=True).dt.tz_convert(LONDON)
    work["lh"] = ldn.dt.hour
    work["ldate"] = ldn.dt.date
    rows = []
    for day, g in work.groupby("ldate", sort=True):
        asia = g[g["lh"] <= asia_end]
        cash = g[(g["lh"] >= cash_h) & (g["lh"] < cash_end)]
        if asia.empty or cash.empty:
            continue
        ah, al = float(asia["high"].max()), float(asia["low"].min())
        mid = (ah + al) / 2.0
        if not (np.isfinite(ah) and np.isfinite(al) and ah > al):
            continue
        saw_ext = saw_mid = False
        for i in cash.index:
            hi, lo, close = float(work.at[i, "high"]), float(work.at[i, "low"]), float(work.at[i, "close"])
            if not saw_ext and (hi >= ah or lo <= al):
                side = "short" if hi >= ah else "long"
                row = _trade(work, int(i), side, True, symbol, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
                saw_ext = True
            if not saw_mid and lo <= mid <= hi:
                side = "short" if close >= mid else "long"
                row = _trade(work, int(i), side, False, symbol, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
                saw_mid = True
            if saw_ext and saw_mid:
                break
    return pd.DataFrame(rows)


def session_range_later_fade_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Fade first later-session tag of a completed session H/L vs first tag of that mid."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon = int(params.get("horizon_bars", 8))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    src_tz = ZoneInfo(str(params.get("source_tz", "Europe/London")))
    hunt_tz = ZoneInfo(str(params.get("hunt_tz", "America/New_York")))
    src_sh = int(params.get("source_start_hour", 8))
    src_eh = int(params.get("source_end_hour", 12))
    hunt_sh = int(params.get("hunt_start_hour", 8))
    hunt_eh = int(params.get("hunt_end_hour", 12))
    skip_weekend = bool(params.get("skip_weekend", True))
    utc = pd.to_datetime(work["time"], utc=True)
    src_local = utc.dt.tz_convert(src_tz)
    hunt_local = utc.dt.tz_convert(hunt_tz)
    work["src_h"] = src_local.dt.hour
    work["src_d"] = src_local.dt.date
    work["hunt_h"] = hunt_local.dt.hour
    work["hunt_d"] = hunt_local.dt.date
    work["hunt_dow"] = hunt_local.dt.dayofweek
    rows = []
    for day, g in work.groupby("hunt_d", sort=True):
        if skip_weekend and int(g["hunt_dow"].iloc[0]) >= 5:
            continue
        source = work[(work["src_d"] == day) & (work["src_h"] >= src_sh) & (work["src_h"] < src_eh)]
        hunt = g[(g["hunt_h"] >= hunt_sh) & (g["hunt_h"] < hunt_eh)]
        if source.empty or hunt.empty:
            continue
        ah, al = float(source["high"].max()), float(source["low"].min())
        mid = (ah + al) / 2.0
        if not (np.isfinite(ah) and np.isfinite(al) and ah > al):
            continue
        saw_ext = saw_mid = False
        for i in hunt.index:
            hi, lo, close = float(work.at[i, "high"]), float(work.at[i, "low"]), float(work.at[i, "close"])
            if not saw_ext and (hi >= ah or lo <= al):
                if hi >= ah and lo <= al:
                    continue
                side = "short" if hi >= ah else "long"
                row = _trade(work, int(i), side, True, symbol, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
                saw_ext = True
            if not saw_mid and lo <= mid <= hi:
                side = "short" if close >= mid else "long"
                row = _trade(work, int(i), side, False, symbol, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
                saw_mid = True
            if saw_ext and saw_mid:
                break
    return pd.DataFrame(rows)


def large_bar_fade_events(df, symbol, params, spread_pips=200, slippage_pips=20) -> pd.DataFrame:
    """Fade a 2.5+ ATR M15 bar vs fade a 1.0-1.5 ATR bar (exhaustion vs ordinary)."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon = int(params.get("horizon_bars", 4))
    big = float(params.get("large_atr", 2.5))
    lo, hi = float(params.get("mid_atr_lo", 1.0)), float(params.get("mid_atr_hi", 1.5))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    atr_p = work["atr"].shift(1)
    rng = work["high"] - work["low"]
    large = rng > big * atr_p
    mid = (rng > lo * atr_p) & (rng <= hi * atr_p)
    rows = []
    n = len(work)
    for i in range(1, n):
        is_t, is_b = bool(large.iloc[i]), bool(mid.iloc[i])
        if not is_t and not is_b:
            continue
        if float(work.at[i, "close"]) == float(work.at[i, "open"]):
            continue
        side = "short" if float(work.at[i, "close"]) > float(work.at[i, "open"]) else "long"
        row = _trade(work, i, side, is_t, symbol, cost, horizon, stop_atr, target_atr)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def vwap_extreme_fade_events(df, symbol, params, spread_pips=200, slippage_pips=20) -> pd.DataFrame:
    """Fade when |price-session VWAP| > 1.5 ATR vs fade when that gap is < 0.35 ATR."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty or "volume" not in work.columns:
        return pd.DataFrame()
    horizon = int(params.get("horizon_bars", 8))
    far = float(params.get("far_atr", 1.5))
    near = float(params.get("near_atr", 0.35))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    ny = pd.to_datetime(work["time"], utc=True).dt.tz_convert(NY)
    work["ny_date"] = ny.dt.date
    tp = (work["high"] + work["low"] + work["close"]) / 3.0
    vol = work["volume"].replace(0, np.nan).fillna(1.0)
    pv = tp * vol
    work["vwap"] = pv.groupby(work["ny_date"]).cumsum() / vol.groupby(work["ny_date"]).cumsum()
    dist = (work["close"] - work["vwap"]).abs() / work["atr"].replace(0, np.nan)
    ext = dist > far
    near_m = dist < near
    ext_on = ext & ~ext.shift(1, fill_value=False)
    near_on = near_m & ~near_m.shift(1, fill_value=False)
    rows = []
    n = len(work)
    for i in range(n):
        is_t, is_b = bool(ext_on.iloc[i]), bool(near_on.iloc[i])
        if not is_t and not is_b:
            continue
        vwap = float(work.at[i, "vwap"])
        close = float(work.at[i, "close"])
        if not np.isfinite(vwap):
            continue
        side = "short" if close > vwap else "long"
        row = _trade(work, i, side, is_t, symbol, cost, horizon, stop_atr, target_atr)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def volume_spike_cont_events(df, symbol, params, spread_pips=200, slippage_pips=20) -> pd.DataFrame:
    """Follow a 2x tick-volume spike vs fade the same spike (unfinished vs exhausted flow)."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty or "volume" not in work.columns:
        return pd.DataFrame()
    horizon = int(params.get("horizon_bars", 6))
    mult = float(params.get("vol_mult", 2.0))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    vma = work["volume"].rolling(20, min_periods=20).mean()
    spike = work["volume"] > mult * vma
    spike_on = spike & ~spike.shift(1, fill_value=False)
    rows = []
    n = len(work)
    for i in np.flatnonzero(spike_on.to_numpy()):
        if float(work.at[i, "close"]) == float(work.at[i, "open"]):
            continue
        follow = "long" if float(work.at[i, "close"]) > float(work.at[i, "open"]) else "short"
        fade = "short" if follow == "long" else "long"
        for treat, side in (True, follow), (False, fade):
            row = _trade(work, int(i), side, treat, symbol, cost, horizon, stop_atr, target_atr)
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def round_bounce_events(df, symbol, params, spread_pips=200, slippage_pips=20) -> pd.DataFrame:
    """First tag of a $5 round fades vs first tag of yesterday's open (not a round magnet)."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon = int(params.get("horizon_bars", 8))
    step = float(params.get("round_step", 5.0))
    near_atr = float(params.get("near_atr", 0.40))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    ny = pd.to_datetime(work["time"], utc=True).dt.tz_convert(NY)
    work["ny_date"] = ny.dt.date
    daily = work.groupby("ny_date", sort=True).agg(o=("open", "first"))
    daily["prev_open"] = daily["o"].shift(1)
    rows = []
    for day, g in work.groupby("ny_date", sort=True):
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


def ny_box_fade_events(df, symbol, params, spread_pips=200, slippage_pips=20) -> pd.DataFrame:
    """Fade the first break of the NY 9:00-9:30 box vs trade that break (M5)."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon = int(params.get("horizon_bars", 12))
    box_bars = int(params.get("box_bars", 6))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    ny = pd.to_datetime(work["time"], utc=True).dt.tz_convert(NY)
    work["ny_date"] = ny.dt.date
    work["ny_hour"] = ny.dt.hour
    rows = []
    for day, g in work.groupby("ny_date", sort=True):
        box = g[(g["ny_hour"] >= 9) & (g["ny_hour"] < 10)].head(box_bars)
        if len(box) < box_bars:
            continue
        rh, rl = float(box["high"].max()), float(box["low"].min())
        last_box_i = int(box.index[-1])
        rest = g.loc[g.index > last_box_i]
        rest = rest[rest["ny_hour"] < 12]
        broke = False
        for i in rest.index:
            hi, lo, close = float(work.at[i, "high"]), float(work.at[i, "low"]), float(work.at[i, "close"])
            side = None
            if close > rh:
                side = "long"
            elif close < rl:
                side = "short"
            if side is None:
                continue
            fade = "short" if side == "long" else "long"
            for treat, s in (True, fade), (False, side):
                row = _trade(work, int(i), s, treat, symbol, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
            broke = True
            break
        _ = broke
    return pd.DataFrame(rows)


def session_box_fade_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Fade first break of a named session opening box vs follow that break."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon = int(params.get("horizon_bars", 8))
    box_bars = int(params.get("box_bars", 2))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    tz = ZoneInfo(str(params.get("tz", "Europe/London")))
    box_sh = int(params.get("box_start_hour", 8))
    box_sm = int(params.get("box_start_minute", 0))
    box_eh = int(params.get("box_end_hour", 8))
    box_em = int(params.get("box_end_minute", 30))
    hunt_end = int(params.get("hunt_end_hour", 12))
    skip_weekend = bool(params.get("skip_weekend", True))
    local = pd.to_datetime(work["time"], utc=True).dt.tz_convert(tz)
    work["ldate"] = local.dt.date
    work["lh"] = local.dt.hour
    work["lm"] = local.dt.minute
    work["dow"] = local.dt.dayofweek
    start_m = box_sh * 60 + box_sm
    end_m = box_eh * 60 + box_em
    rows = []
    for _, g in work.groupby("ldate", sort=True):
        if skip_weekend and int(g["dow"].iloc[0]) >= 5:
            continue
        mins = g["lh"] * 60 + g["lm"]
        box = g[(mins >= start_m) & (mins < end_m)].head(box_bars)
        if len(box) < box_bars:
            continue
        rh, rl = float(box["high"].max()), float(box["low"].min())
        last_box_i = int(box.index[-1])
        rest = g.loc[g.index > last_box_i]
        rest = rest[rest["lh"] < hunt_end]
        for i in rest.index:
            close = float(work.at[i, "close"])
            side = None
            if close > rh:
                side = "long"
            elif close < rl:
                side = "short"
            if side is None:
                continue
            fade = "short" if side == "long" else "long"
            for treat, s in (True, fade), (False, side):
                row = _trade(work, int(i), s, treat, symbol, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
            break
    return pd.DataFrame(rows)


def _fvg_zone(work: pd.DataFrame, i: int, min_gap_atr: float):
    if i < 2:
        return None
    h0 = float(work.at[i - 2, "high"])
    l0 = float(work.at[i - 2, "low"])
    h2 = float(work.at[i, "high"])
    l2 = float(work.at[i, "low"])
    atr = float(work.at[i, "atr"])
    if not np.isfinite(atr) or atr <= 0:
        return None
    if l2 > h0 and (l2 - h0) >= min_gap_atr * atr:
        return "long", h0, l2
    if h2 < l0 and (l0 - h2) >= min_gap_atr * atr:
        return "short", h2, l0
    return None


def prior_hour_extreme_fade_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Fade first M15 tag of the prior clock-hour high/low in London AM vs NY AM."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon = int(params.get("horizon_bars", 8))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    start_h = int(params.get("window_start_hour", 8))
    end_h = int(params.get("window_end_hour", 12))
    skip_weekend = bool(params.get("skip_weekend", True))
    treat_tz = ZoneInfo(str(params.get("treat_tz", "Europe/London")))
    base_tz = ZoneInfo(str(params.get("base_tz", "America/New_York")))
    rows = []
    for treat, tz in ((True, treat_tz), (False, base_tz)):
        local = pd.to_datetime(work["time"], utc=True).dt.tz_convert(tz)
        ldate = local.dt.date
        lh = local.dt.hour
        dow = local.dt.dayofweek
        hour_hl = (
            pd.DataFrame({"ldate": ldate, "lh": lh, "high": work["high"], "low": work["low"]})
            .groupby(["ldate", "lh"], sort=True)
            .agg(hh=("high", "max"), ll=("low", "min"))
        )
        for day in pd.unique(ldate):
            mask_day = ldate == day
            if skip_weekend and int(dow[mask_day].iloc[0]) >= 5:
                continue
            for h in range(start_h, end_h):
                if (day, h - 1) not in hour_hl.index:
                    continue
                ph = float(hour_hl.at[(day, h - 1), "hh"])
                pl = float(hour_hl.at[(day, h - 1), "ll"])
                if not (np.isfinite(ph) and np.isfinite(pl)):
                    continue
                hour_idx = work.loc[mask_day & (lh == h)].index
                for i in hour_idx:
                    hi = float(work.at[i, "high"])
                    lo = float(work.at[i, "low"])
                    hit_hi = hi >= ph
                    hit_lo = lo <= pl
                    if hit_hi and hit_lo:
                        continue
                    if hit_hi:
                        side = "short"
                    elif hit_lo:
                        side = "long"
                    else:
                        continue
                    row = _trade(work, int(i), side, treat, symbol, cost, horizon, stop_atr, target_atr)
                    if row:
                        rows.append(row)
                    break
    return pd.DataFrame(rows)


def hour_close_flatten_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Fade the hour-to-date move at :45 in London AM vs the same flatten in NY AM."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon = int(params.get("horizon_bars", 8))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    start_h = int(params.get("window_start_hour", 8))
    end_h = int(params.get("window_end_hour", 12))
    fire_m = int(params.get("fire_minute", 45))
    min_prior = float(params.get("min_prior_atr", 0.25))
    skip_weekend = bool(params.get("skip_weekend", True))
    treat_tz = ZoneInfo(str(params.get("treat_tz", "Europe/London")))
    base_tz = ZoneInfo(str(params.get("base_tz", "America/New_York")))
    rows = []
    for treat, tz in ((True, treat_tz), (False, base_tz)):
        local = pd.to_datetime(work["time"], utc=True).dt.tz_convert(tz)
        ldate = local.dt.date
        lh = local.dt.hour
        lm = local.dt.minute
        dow = local.dt.dayofweek
        for day in pd.unique(ldate):
            mask_day = ldate == day
            if skip_weekend and int(dow[mask_day].iloc[0]) >= 5:
                continue
            for h in range(start_h, end_h):
                open_bars = work.loc[mask_day & (lh == h) & (lm == 0)]
                fire_bars = work.loc[mask_day & (lh == h) & (lm == fire_m)]
                if open_bars.empty or fire_bars.empty:
                    continue
                i = int(fire_bars.index[0])
                hour_open = float(open_bars["open"].iloc[0])
                fire_close = float(work.at[i, "close"])
                atr = float(work.at[i, "atr"])
                if not np.isfinite(atr) or atr <= 0:
                    continue
                move = fire_close - hour_open
                if abs(move) < min_prior * atr:
                    continue
                side = "short" if move > 0 else "long"
                row = _trade(work, i, side, treat, symbol, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
    return pd.DataFrame(rows)


def hour_open_gap_fade_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Fade the :00 gap vs the prior hour's last close in London AM vs NY AM."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon = int(params.get("horizon_bars", 8))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    start_h = int(params.get("window_start_hour", 8))
    end_h = int(params.get("window_end_hour", 12))
    min_prior = float(params.get("min_prior_atr", 0.25))
    skip_weekend = bool(params.get("skip_weekend", True))
    treat_tz = ZoneInfo(str(params.get("treat_tz", "Europe/London")))
    base_tz = ZoneInfo(str(params.get("base_tz", "America/New_York")))
    rows = []
    for treat, tz in ((True, treat_tz), (False, base_tz)):
        local = pd.to_datetime(work["time"], utc=True).dt.tz_convert(tz)
        ldate = local.dt.date
        lh = local.dt.hour
        lm = local.dt.minute
        dow = local.dt.dayofweek
        for day in pd.unique(ldate):
            mask_day = ldate == day
            if skip_weekend and int(dow[mask_day].iloc[0]) >= 5:
                continue
            for h in range(start_h, end_h):
                prev = work.loc[mask_day & (lh == h - 1)]
                opens = work.loc[mask_day & (lh == h) & (lm == 0)]
                if prev.empty or opens.empty:
                    continue
                i = int(opens.index[0])
                gap = float(work.at[i, "open"]) - float(prev["close"].iloc[-1])
                atr = float(work.at[i, "atr"])
                if not np.isfinite(atr) or atr <= 0:
                    continue
                if abs(gap) < min_prior * atr:
                    continue
                side = "short" if gap > 0 else "long"
                row = _trade(work, i, side, treat, symbol, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
    return pd.DataFrame(rows)


def prior_close_tag_fade_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Fade first M15 tag of the prior weekday close in London AM vs NY AM."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon = int(params.get("horizon_bars", 8))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    start_h = int(params.get("window_start_hour", 8))
    end_h = int(params.get("window_end_hour", 12))
    skip_weekend = bool(params.get("skip_weekend", True))
    treat_tz = ZoneInfo(str(params.get("treat_tz", "Europe/London")))
    base_tz = ZoneInfo(str(params.get("base_tz", "America/New_York")))
    rows = []
    for treat, tz in ((True, treat_tz), (False, base_tz)):
        local = pd.to_datetime(work["time"], utc=True).dt.tz_convert(tz)
        ldate = local.dt.date
        lh = local.dt.hour
        dow = local.dt.dayofweek
        pdc_for: dict = {}
        last_wd_close = None
        for day in pd.unique(ldate):
            mask_day = ldate == day
            d0 = int(dow[mask_day].iloc[0])
            if skip_weekend and d0 >= 5:
                continue
            pdc_for[day] = last_wd_close
            last_wd_close = float(work.loc[mask_day, "close"].iloc[-1])
        for day, pdc in pdc_for.items():
            if pdc is None or not np.isfinite(pdc):
                continue
            mask_day = ldate == day
            hour_idx = work.loc[mask_day & (lh >= start_h) & (lh < end_h)].index
            for i in hour_idx:
                hi = float(work.at[i, "high"])
                lo = float(work.at[i, "low"])
                px = float(work.at[i, "close"])
                if not (lo <= pdc <= hi):
                    continue
                if px > pdc:
                    side = "short"
                elif px < pdc:
                    side = "long"
                else:
                    continue
                row = _trade(work, int(i), side, treat, symbol, cost, horizon, stop_atr, target_atr)
                if row:
                    rows.append(row)
                break
    return pd.DataFrame(rows)


def silver_bullet_fvg_events(df, symbol, params, spread_pips=1.5, slippage_pips=0.2) -> pd.DataFrame:
    """Follow first M15 FVG tag in a named NY hour vs the same rule in the next hour."""
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    horizon = int(params.get("horizon_bars", 8))
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    min_gap = float(params.get("min_gap_atr", 0.15))
    treat_start = int(params.get("treat_hour", 10))
    base_start = int(params.get("base_hour", 11))
    cost = cost_price(symbol, spread_pips, slippage_pips)
    ny = pd.to_datetime(work["time"], utc=True).dt.tz_convert(NY)
    work["ny_date"] = ny.dt.date
    work["ny_hour"] = ny.dt.hour
    rows = []
    for _, g in work.groupby("ny_date", sort=True):
        for treat, start in ((True, treat_start), (False, base_start)):
            end = start + 1
            window = g[(g["ny_hour"] >= start) & (g["ny_hour"] < end)]
            if window.empty:
                continue
            taken = False
            for i in window.index:
                zone = _fvg_zone(work, int(i), min_gap)
                if zone is None:
                    continue
                side, zlo, zhi = zone
                later = window.loc[window.index > i]
                for k in later.index:
                    lo, hi = float(work.at[k, "low"]), float(work.at[k, "high"])
                    if lo <= zhi and hi >= zlo:
                        row = _trade(work, int(k), side, treat, symbol, cost, horizon, stop_atr, target_atr)
                        if row:
                            rows.append(row)
                        taken = True
                        break
                if taken:
                    break
    return pd.DataFrame(rows)
