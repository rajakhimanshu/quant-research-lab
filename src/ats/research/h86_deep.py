"""H86 last-level battery. Frozen 36h / pairs. Bar spread, M5 path, cluster, LOO. Not an EA."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ats.config import DATA_DIR
from ats.data.mt5_client import load_raw
from ats.features.prepare import prepare_frame
from ats.paper.h86_book import h86_params, pair_cost
from ats.research.h86_audit import _all_events, _block
from ats.research.ledger import ledger_rows
from ats.timeutil import locked_calendar_split, pip_size

HORIZON_H = 8
POINTS_PER_PIP = 10.0
LOT = 0.01


def _mean(x) -> float | None:
    if x is None or len(x) == 0:
        return None
    return float(np.mean(x))


def _split_mask(times: pd.Series, split, name: str) -> pd.Series:
    if name == "train":
        return times <= split.train_end
    if name == "validation":
        return (times > split.train_end) & (times <= split.val_end)
    return times > split.val_end


def _first_bar(side: str, high: float, low: float, stop: float, target: float) -> str:
    if side == "long":
        hit_stop, hit_tgt = low <= stop, high >= target
    else:
        hit_stop, hit_tgt = high >= stop, low <= target
    if hit_stop and hit_tgt:
        return "both_stop_first"
    if hit_stop:
        return "stop"
    if hit_tgt:
        return "target"
    return "neither"


def _walk_bars(bars: pd.DataFrame, side: str, entry: float, stop: float, target: float) -> dict:
    risk = abs(entry - stop)
    if bars.empty:
        return {"end": "no_bars", "r_mult": None, "same_bar_both": False, "n_bars": 0}
    for _, bar in bars.iterrows():
        tag = _first_bar(side, float(bar["high"]), float(bar["low"]), stop, target)
        if tag == "both_stop_first":
            return {"end": "same_bar_stop_first", "r_mult": -1.0, "same_bar_both": True, "n_bars": int(len(bars))}
        if tag == "stop":
            return {"end": "stop", "r_mult": -1.0, "same_bar_both": False, "n_bars": int(len(bars))}
        if tag == "target":
            return {"end": "target", "r_mult": 1.0, "same_bar_both": False, "n_bars": int(len(bars))}
    last = float(bars.iloc[-1]["close"])
    if risk <= 0:
        r = 0.0
    elif side == "long":
        r = (last - entry) / risk
    else:
        r = (entry - last) / risk
    return {"end": "time", "r_mult": float(r), "same_bar_both": False, "n_bars": int(len(bars))}


def _pip_value_usd(symbol: str, price: float, lot: float = LOT) -> float:
    units = 100_000.0 * lot
    if "JPY" in symbol.upper():
        return 0.01 * units / price if price else 0.0
    return 0.0001 * units


def _cash_pnl(row: pd.Series, lot: float = LOT) -> float:
    atr = float(row["atr"])
    pip = pip_size(str(row["symbol"]))
    stop_pips = atr / pip if pip else 0.0
    px = float(row.get("raw_open") or row.get("entry") or 0.0)
    return float(row["r_mult"]) * stop_pips * _pip_value_usd(str(row["symbol"]), px, lot)


def _spread_profile(raw: pd.DataFrame, symbol: str) -> dict:
    df = raw.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    ldn = df["time"].dt.tz_convert("Europe/London")
    pips = df["spread"] / POINTS_PER_PIP
    wd = ldn.dt.dayofweek < 5
    sun = ldn.dt.dayofweek == 6
    reopen = sun & ldn.dt.hour.isin([21, 22])
    return {
        "symbol": symbol,
        "weekday_median_pips": float(pips.loc[wd].median()) if wd.any() else None,
        "sunday_median_pips": float(pips.loc[sun].median()) if sun.any() else None,
        "sunday_reopen_median_pips": float(pips.loc[reopen].median()) if reopen.any() else None,
        "sunday_reopen_p90_pips": float(pips.loc[reopen].quantile(0.9)) if reopen.any() else None,
        "lab_pips": None,
        "sunday_vs_lab_mult": None,
        "n_sunday_reopen_bars": int(reopen.sum()),
    }


def _bucket(treated: pd.DataFrame, split) -> list[dict]:
    t = treated.copy()
    edges = [0.15, 0.30, 0.50, 10.0]
    labels = ["0.15-0.30", "0.30-0.50", ">=0.50"]
    t["bucket"] = pd.cut(t["gap_atr"], bins=edges, labels=labels, right=False)
    rows = []
    for lab in labels:
        g = t.loc[t["bucket"] == lab]
        row = {"bucket": lab, "n": int(len(g)), "mean_r": _mean(g["r_mult"]) if len(g) else None}
        for name in ("train", "validation", "oos"):
            sub = g.loc[_split_mask(g["time"], split, name)]
            row[f"{name}_n"] = int(len(sub))
            row[f"{name}_mean_r"] = _mean(sub["r_mult"]) if len(sub) else None
        rows.append(row)
    return rows


def _m5_for_trade(m5: pd.DataFrame, start, hours: int = HORIZON_H) -> pd.DataFrame:
    end = start + pd.Timedelta(hours=hours)
    return m5.loc[(m5["time"] >= start) & (m5["time"] < end)]


def run_deep() -> dict:
    from ats.config import load_settings

    settings = load_settings()
    split = locked_calendar_split(settings, "fx_h1")
    params = h86_params()
    symbols = list(params["symbols"])
    h1_raw = {s: load_raw(s, "H1") for s in symbols}
    frames = {s: prepare_frame(h1_raw[s], settings) for s in symbols}
    m5 = {}
    for s in symbols:
        raw = load_raw(s, "M5")
        raw = raw.copy()
        raw["time"] = pd.to_datetime(raw["time"], utc=True)
        m5[s] = raw.sort_values("time").reset_index(drop=True)

    lab = _all_events(frames)
    treated = lab.loc[lab["treatment"]].copy()
    treated["time"] = pd.to_datetime(treated["time"], utc=True)
    london = treated["time"].dt.tz_convert("Europe/London")
    treated["week"] = london.dt.strftime("%G-W%V")

    bar = _all_events(frames, {"use_bar_spread": True})
    bar_t = bar.loc[bar["treatment"]].copy() if not bar.empty else pd.DataFrame()
    if not bar_t.empty:
        bar_t["time"] = pd.to_datetime(bar_t["time"], utc=True)

    spread_rows = []
    for s in symbols:
        row = _spread_profile(h1_raw[s], s)
        lab_pips, _slip = pair_cost(s, settings)
        row["lab_pips"] = float(lab_pips)
        if row["sunday_reopen_median_pips"] and lab_pips:
            row["sunday_vs_lab_mult"] = float(row["sunday_reopen_median_pips"] / lab_pips)
        spread_rows.append(row)

    first_bar = {"stop": 0, "target": 0, "both_stop_first": 0, "neither": 0}
    for _, tr in treated.iterrows():
        df = frames[str(tr["symbol"])]
        times = pd.to_datetime(df["time"], utc=True)
        hit = times[times == tr["time"]]
        if hit.empty:
            continue
        i = int(hit.index[0])
        tag = _first_bar(
            str(tr["side"]),
            float(df.at[i, "high"]),
            float(df.at[i, "low"]),
            float(tr["stop"]),
            float(tr["target"]),
        )
        first_bar[tag] = first_bar.get(tag, 0) + 1

    m5_rows = []
    m5_spread_r = []
    for _, tr in treated.iterrows():
        sym = str(tr["symbol"])
        start = tr["time"]
        bars = _m5_for_trade(m5[sym], start)
        walk = _walk_bars(bars, str(tr["side"]), float(tr["entry"]), float(tr["stop"]), float(tr["target"]))
        first_spread = None
        if not bars.empty and "spread" in bars.columns:
            pts = float(bars.iloc[0]["spread"])
            if np.isfinite(pts):
                first_spread = pts / POINTS_PER_PIP
        split_name = (
            "train"
            if start <= split.train_end
            else ("validation" if start <= split.val_end else "oos")
        )
        m5_rows.append(
            {
                "time": start,
                "symbol": sym,
                "split": split_name,
                "covered": walk["n_bars"] > 0,
                "end": walk["end"],
                "r_mult": walk["r_mult"],
                "same_bar_both": walk["same_bar_both"],
                "first_m5_spread_pips": first_spread,
            }
        )
        if first_spread is not None and walk["r_mult"] is not None:
            # Haircut: extra cost vs lab, in R. 1R = 1 ATR.
            atr = float(tr["atr"])
            pip = pip_size(sym)
            lab_pips, slip = pair_cost(sym, settings)
            extra_pips = max(0.0, first_spread - lab_pips)
            extra_r = (extra_pips * pip) / atr if atr else 0.0
            m5_spread_r.append(
                {
                    "split": split_name,
                    "r_lab_path": float(walk["r_mult"]),
                    "r_m5_spread_haircut": float(walk["r_mult"]) - extra_r,
                    "first_m5_spread_pips": first_spread,
                    "extra_r": extra_r,
                }
            )
    m5_df = pd.DataFrame(m5_rows)
    m5_cov = m5_df.loc[m5_df["covered"]] if not m5_df.empty else pd.DataFrame()
    m5_sp = pd.DataFrame(m5_spread_r)

    loo = []
    for drop in symbols:
        keep = treated.loc[treated["symbol"] != drop]
        row = {"drop": drop}
        for name in ("train", "validation", "oos"):
            sub = keep.loc[_split_mask(keep["time"], split, name)]
            row[f"{name}_n"] = int(len(sub))
            row[f"{name}_mean_r"] = _mean(sub["r_mult"]) if len(sub) else None
        loo.append(row)

    weekly = treated.groupby("week").agg(
        n=("r_mult", "size"),
        sum_r=("r_mult", "sum"),
        time=("time", "min"),
        symbols=("symbol", lambda s: ",".join(sorted(set(s)))),
    )
    worst = weekly.nsmallest(5, "sum_r")
    cluster = []
    for name in ("train", "validation", "oos"):
        w = weekly.loc[_split_mask(weekly["time"], split, name)]
        cluster.append(
            {
                "split": name,
                "weeks": int(len(w)),
                "mean_week_sum_r": _mean(w["sum_r"]) if len(w) else None,
                "worst_week_r": float(w["sum_r"].min()) if len(w) else None,
                "share_multi_pair": float((w["n"] >= 2).mean()) if len(w) else None,
            }
        )

    lot_rows = []
    for name in ("train", "validation", "oos"):
        sub = treated.loc[_split_mask(treated["time"], split, name)]
        cash = sub.apply(_cash_pnl, axis=1) if len(sub) else pd.Series(dtype=float)
        lot_rows.append(
            {
                "split": name,
                "n": int(len(sub)),
                "lot": LOT,
                "sum_usd": float(cash.sum()) if len(cash) else 0.0,
                "mean_usd": float(cash.mean()) if len(cash) else None,
                "note": "0.01 lot cash, not 1% risk. $100-200 demo cannot 1% a 15-pip stop.",
            }
        )

    def _split_block(ev: pd.DataFrame, name: str) -> dict:
        if ev.empty:
            return {"split": name, "n": 0, "mean_r": None}
        part = ev.loc[_split_mask(ev["time"], split, name)]
        if part.empty:
            return {"split": name, "n": 0, "mean_r": None}
        return {
            "split": name,
            "n": int(len(part)),
            "mean_r": _mean(part["r_mult"]),
            "win_rate": float(part["success"].mean()) if "success" in part else None,
        }

    bar_splits = {name: _split_block(bar_t, name) for name in ("train", "validation", "oos")}
    lab_splits = {name: _block(name, lab.loc[_split_mask(lab["time"], split, name)]) for name in ("train", "validation", "oos")}

    sunday = treated.loc[treated["reopen_dow"] == 6] if "reopen_dow" in treated else treated
    holiday = treated.loc[treated["reopen_dow"] != 6] if "reopen_dow" in treated else pd.DataFrame()

    def _m5_split(name: str) -> dict:
        if m5_cov.empty:
            return {"split": name, "n": 0}
        sub = m5_cov.loc[m5_cov["split"] == name]
        return {
            "split": name,
            "n": int(len(sub)),
            "mean_r": _mean(sub["r_mult"]),
            "stop_first_share": float((sub["end"] == "stop").mean()) if len(sub) else None,
            "target_share": float((sub["end"] == "target").mean()) if len(sub) else None,
            "same_bar_both": int(sub["same_bar_both"].sum()) if len(sub) else 0,
        }

    def _m5_cost_split(name: str) -> dict:
        if m5_sp.empty:
            return {"split": name, "n": 0}
        sub = m5_sp.loc[m5_sp["split"] == name]
        return {
            "split": name,
            "n": int(len(sub)),
            "mean_first_m5_spread_pips": _mean(sub["first_m5_spread_pips"]),
            "mean_r_lab_m5_path": _mean(sub["r_lab_path"]),
            "mean_r_after_m5_spread_haircut": _mean(sub["r_m5_spread_haircut"]),
            "mean_extra_r": _mean(sub["extra_r"]),
        }

    n_hyps = len(ledger_rows())
    oos_p = lab_splits["oos"].get("p_value")
    blockers = []
    oos_bar = bar_splits["oos"].get("mean_r")
    if oos_bar is None or oos_bar <= 0:
        blockers.append("h1_bar_spread_oos_mean_r<=0")
    oos_m5 = _m5_split("oos")
    if oos_m5.get("n", 0) >= 20 and (oos_m5.get("mean_r") is None or oos_m5["mean_r"] <= 0):
        blockers.append("m5_path_oos_mean_r<=0")
    oos_m5c = _m5_cost_split("oos")
    if oos_m5c.get("n", 0) >= 20 and (
        oos_m5c.get("mean_r_after_m5_spread_haircut") is None
        or oos_m5c["mean_r_after_m5_spread_haircut"] <= 0
    ):
        blockers.append("m5_open_spread_haircut_oos_mean_r<=0")
    if any((r.get("oos_mean_r") or 0) <= 0 for r in loo):
        blockers.append("leave_one_symbol_oos_mean_r<=0")
    oos_cl = next(c for c in cluster if c["split"] == "oos")
    if oos_cl.get("mean_week_sum_r") is None or oos_cl["mean_week_sum_r"] <= 0:
        blockers.append("weekend_cluster_oos_mean_r<=0")

    payload = {
        "hypothesis": "H86_weekend_gap_g10",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "frozen": True,
        "lab_cost_splits": lab_splits,
        "h1_bar_spread_splits": bar_splits,
        "h1_sunday_spread": spread_rows,
        "first_h1_bar_path": first_bar,
        "gap_atr_buckets": _bucket(treated, split),
        "sunday_vs_holiday": {
            "sunday_n": int(len(sunday)),
            "sunday_mean_r": _mean(sunday["r_mult"]) if len(sunday) else None,
            "holiday_n": int(len(holiday)),
            "holiday_mean_r": _mean(holiday["r_mult"]) if len(holiday) else None,
        },
        "leave_one_symbol_out": loo,
        "weekend_cluster": cluster,
        "worst_weeks": [
            {"week": str(i), "n": int(r["n"]), "sum_r": float(r["sum_r"]), "symbols": r["symbols"]}
            for i, r in worst.iterrows()
        ],
        "m5_path": {name: _m5_split(name) for name in ("train", "validation", "oos")},
        "m5_open_spread_haircut": {name: _m5_cost_split(name) for name in ("train", "validation", "oos")},
        "m5_coverage_note": "M5 files are 100k-bar capped (~May 2025+). Path is val/OOS-heavy, not full train.",
        "lot_0_01_cash_usd": lot_rows,
        "multiple_testing": {
            "ledger_closed_n": n_hyps,
            "oos_p_vs_wednesday": oos_p,
            "bonferroni_alpha": 0.05 / n_hyps if n_hyps else None,
            "oos_p_survives_bonferroni": bool(oos_p is not None and oos_p < (0.05 / n_hyps)) if n_hyps else False,
            "note": "OOS was locked until this hyp was picked. Bonferroni vs the whole ledger is conservative, not a retune switch.",
        },
        "blockers": blockers,
        "last_level_clear": not blockers,
        "notes": [
            "Do not retune 36h or min_gap_atr.",
            "Do not add NZD/CAD/CHF/EURUSD.",
            "H1 bar spread is MT5 points/10. Sunday reopen is the fill, not weekday median.",
            "M5 first-bar spread haircut is the open-print cost. H1 bar spread can be a later-in-hour snapshot.",
            "Not live. EA only if you accept last_level_clear after reading blockers.",
        ],
    }
    path = DATA_DIR / "results" / "H86_deep.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    payload["_path"] = str(path)
    return payload


def print_deep(blob: dict) -> None:
    print("H86 last-level — frozen weekend-gap fade. No retune. Not an EA.")
    print("Lab-cost splits:")
    for name, s in blob["lab_cost_splits"].items():
        print(f"  {name:12} n={s.get('n')} wr={s.get('win_rate')} meanR={s.get('mean_r')}")
    print("H1 bar-spread cost (this bar's MT5 spread, not weekday median):")
    for name, s in blob["h1_bar_spread_splits"].items():
        print(f"  {name:12} n={s.get('n')} meanR={s.get('mean_r')} wr={s.get('win_rate')}")
    print("Sunday H1 reopen spread vs lab:")
    for r in blob["h1_sunday_spread"]:
        print(
            f"  {r['symbol']:7} lab={r['lab_pips']} sun_reopen_med={r['sunday_reopen_median_pips']} "
            f"x{r['sunday_vs_lab_mult']} p90={r['sunday_reopen_p90_pips']}"
        )
    print(f"First H1 bar path (stop-first if both): {blob['first_h1_bar_path']}")
    print("Leave-one-symbol-out OOS mean R:")
    for r in blob["leave_one_symbol_out"]:
        print(f"  drop {r['drop']:7} oos n={r['oos_n']} meanR={r['oos_mean_r']}")
    print("Weekend cluster (sum R of all pairs that Sunday):")
    for c in blob["weekend_cluster"]:
        print(
            f"  {c['split']:12} weeks={c['weeks']} meanSumR={c['mean_week_sum_r']} "
            f"worst={c['worst_week_r']} multi={c['share_multi_pair']}"
        )
    print("M5 stop-first path (100k cap, late sample):")
    for name, s in blob["m5_path"].items():
        print(f"  {name:12} n={s.get('n')} meanR={s.get('mean_r')} tgt={s.get('target_share')} both={s.get('same_bar_both')}")
    print("M5 first-print spread haircut:")
    for name, s in blob["m5_open_spread_haircut"].items():
        print(
            f"  {name:12} n={s.get('n')} m5spr={s.get('mean_first_m5_spread_pips')} "
            f"pathR={s.get('mean_r_lab_m5_path')} after={s.get('mean_r_after_m5_spread_haircut')}"
        )
    print("0.01 lot cash USD (not 1% risk):")
    for r in blob["lot_0_01_cash_usd"]:
        print(f"  {r['split']:12} n={r['n']} sum=${r['sum_usd']:.2f} mean=${r.get('mean_usd')}")
    mt = blob["multiple_testing"]
    print(
        f"Multiple testing: ledger n={mt['ledger_closed_n']} OOS p={mt['oos_p_vs_wednesday']} "
        f"Bonferroni a={mt['bonferroni_alpha']} survives={mt['oos_p_survives_bonferroni']}"
    )
    print(f"LAST_LEVEL_CLEAR={blob['last_level_clear']}")
    print(f"Blockers: {blob['blockers']}")
    print(f"Wrote {blob.get('_path')}")
    print("Your call on the EA. This script does not start MQL5.")
