"""H192 last-level. Train/val only. Bar spread, M5 path, cost haircut. Not an EA. No OOS peek."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ats.config import DATA_DIR, load_settings
from ats.data.mt5_client import load_raw
from ats.features.prepare import prepare_frame
from ats.hypotheses.clock_family import clock_run_events
from ats.timeutil import locked_calendar_split, pip_size

BERLIN = ZoneInfo("Europe/Berlin")
PARAMS = {
    "tz": "Europe/Berlin",
    "treat_hour": 17,
    "treat_minute": 30,
    "base_hour": 14,
    "base_minute": 0,
    "mode": "fade",
    "skip_weekend": True,
    "min_prior_atr": 0.25,
    "horizon_bars": 8,
    "stop_atr": 1.0,
    "target_atr": 1.0,
}


def _mean(x) -> float | None:
    if x is None or len(x) == 0:
        return None
    arr = np.asarray(list(x), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    return float(np.mean(arr))


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
        return {"end": "no_bars", "r_mult": None, "same_bar_both": False}
    for _, bar in bars.iterrows():
        tag = _first_bar(side, float(bar["high"]), float(bar["low"]), stop, target)
        if tag == "both_stop_first":
            return {"end": "same_bar_stop_first", "r_mult": -1.0, "same_bar_both": True}
        if tag == "stop":
            return {"end": "stop", "r_mult": -1.0, "same_bar_both": False}
        if tag == "target":
            return {"end": "target", "r_mult": 1.0, "same_bar_both": False}
    last = float(bars.iloc[-1]["close"])
    if risk <= 0:
        r = 0.0
    elif side == "long":
        r = (last - entry) / risk
    else:
        r = (entry - last) / risk
    return {"end": "time", "r_mult": float(r), "same_bar_both": False}


def run_deep() -> dict:
    settings = load_settings()
    split = locked_calendar_split(settings, "gold_m15")
    costs = settings["costs"]
    lab_pips = float(costs["spread_pips"]["XAUUSD"])
    slip = float(costs.get("slippage_by_symbol", {}).get("XAUUSD", costs.get("slippage_pips", 20)))
    lab_total = lab_pips + slip
    raw = load_raw("XAUUSD", "M15")
    m15 = prepare_frame(raw, settings)
    ev = clock_run_events(m15, "XAUUSD", PARAMS, lab_pips, slip)
    treated = ev.loc[ev["treatment"]].copy()
    treated["time"] = pd.to_datetime(treated["time"], utc=True)
    train_m = treated["time"] <= split.train_end
    val_m = (treated["time"] > split.train_end) & (treated["time"] <= split.val_end)
    locked = treated.loc[train_m | val_m].copy()

    raw_s = raw.copy()
    raw_s["time"] = pd.to_datetime(raw_s["time"], utc=True)
    fill = locked.merge(raw_s[["time", "spread"]], on="time", how="left")
    # Gold: MT5 spread points match lab units (med ~200); do not divide by 10.
    fill["spread_pips"] = fill["spread"].astype(float)
    berlin = fill["time"].dt.tz_convert(BERLIN)
    fill["berlin_hour"] = berlin.dt.hour
    fill["berlin_minute"] = berlin.dt.minute
    wd = raw_s.copy()
    wd["berlin"] = wd["time"].dt.tz_convert(BERLIN)
    wd_pips = wd["spread"].astype(float)
    weekday = wd["berlin"].dt.dayofweek < 5

    pip = pip_size("XAUUSD")
    fill["extra_pips"] = (fill["spread_pips"] - lab_total).clip(lower=0)
    fill["extra_r"] = fill["extra_pips"] * pip / fill["atr"]
    fill["r_haircut"] = fill["r_mult"] - fill["extra_r"]

    m5_raw = load_raw("XAUUSD", "M5")
    m5_raw["time"] = pd.to_datetime(m5_raw["time"], utc=True)
    m5_raw = m5_raw.sort_values("time")
    path_rows = []
    first_bar = {"stop": 0, "target": 0, "both_stop_first": 0, "neither": 0}
    m15_path_n = 0
    m5_path_n = 0
    m15_idx = m15.copy()
    m15_idx["time"] = pd.to_datetime(m15_idx["time"], utc=True)
    for _, tr in locked.iterrows():
        hit = m15_idx.index[m15_idx["time"] == tr["time"]]
        if len(hit):
            i = int(hit[0])
            tag = _first_bar(
                str(tr["side"]),
                float(m15_idx.at[i, "high"]),
                float(m15_idx.at[i, "low"]),
                float(tr["stop"]),
                float(tr["target"]),
            )
            first_bar[tag] = first_bar.get(tag, 0) + 1
        start = tr["time"]
        bars = m5_raw.loc[(m5_raw["time"] >= start) & (m5_raw["time"] < start + pd.Timedelta(hours=2))]
        path_tf = "M5"
        if bars.empty:
            # Broker gold M5 starts ~2025-04; train ends 2024-12. Fall back to M15 path.
            bars = m15_idx.loc[(m15_idx["time"] > start) & (m15_idx["time"] <= start + pd.Timedelta(hours=2))]
            path_tf = "M15"
            m15_path_n += 1
        else:
            m5_path_n += 1
        walk = _walk_bars(bars, str(tr["side"]), float(tr["entry"]), float(tr["stop"]), float(tr["target"]))
        first_spread = None
        if not bars.empty and "spread" in bars.columns:
            pts = float(bars.iloc[0]["spread"])
            if np.isfinite(pts):
                first_spread = pts
        elif path_tf == "M15" and "spread" in fill.columns:
            row = fill.loc[fill["time"] == start]
            if not row.empty and np.isfinite(float(row.iloc[0]["spread_pips"])):
                first_spread = float(row.iloc[0]["spread_pips"])
        extra_r = 0.0
        if first_spread is not None:
            extra_r = max(0.0, first_spread - lab_total) * pip / float(tr["atr"])
        path_r = walk["r_mult"]
        path_rows.append(
            {
                "split": "train" if tr["time"] <= split.train_end else "validation",
                "end": walk["end"],
                "r_path": path_r,
                "r_haircut": None if path_r is None else float(path_r) - extra_r,
                "first_m5_spread_pips": first_spread,
                "same_bar_both": walk["same_bar_both"],
                "path_tf": path_tf,
            }
        )
    paths = pd.DataFrame(path_rows)

    def _split_stats(mask: pd.Series, col: str) -> dict:
        sub = fill.loc[mask]
        return {"n": int(len(sub)), "mean_r": _mean(sub[col]), "median_spread_pips": _mean(sub["spread_pips"])}

    def _path_stats(name: str) -> dict:
        sub = paths.loc[paths["split"] == name] if not paths.empty else paths
        return {
            "n": int(len(sub)),
            "mean_r_path": _mean(sub["r_path"]) if not sub.empty else None,
            "mean_r_haircut": _mean(sub["r_haircut"]) if not sub.empty else None,
            "mean_m5_spread_pips": _mean(sub["first_m5_spread_pips"]) if not sub.empty else None,
            "same_bar_both_n": int(sub["same_bar_both"].sum()) if not sub.empty else 0,
        }

    train_lab = _split_stats(fill["time"] <= split.train_end, "r_mult")
    val_lab = _split_stats(
        (fill["time"] > split.train_end) & (fill["time"] <= split.val_end), "r_mult"
    )
    train_hc = _split_stats(fill["time"] <= split.train_end, "r_haircut")
    val_hc = _split_stats(
        (fill["time"] > split.train_end) & (fill["time"] <= split.val_end), "r_haircut"
    )
    train_path = _path_stats("train")
    val_path = _path_stats("validation")
    fill_med = _mean(fill["spread_pips"])
    wd_med = float(wd_pips.loc[weekday].median()) if weekday.any() else None
    h1730 = (berlin.dt.hour == 17) & (berlin.dt.minute == 30)
    h14 = (wd["berlin"].dt.hour == 14) & weekday
    def _ok_r(x) -> bool:
        return x is not None and np.isfinite(x) and x > 0

    blockers = []
    if fill_med is None or (wd_med and fill_med > 3.0 * max(lab_total, wd_med)):
        blockers.append("fill_bar_spread_gt_3x_lab")
    if not _ok_r(train_hc["mean_r"]):
        blockers.append("train_spread_haircut_mean_r<=0")
    if not _ok_r(val_hc["mean_r"]):
        blockers.append("val_spread_haircut_mean_r<=0")
    if train_path["n"] < 30 or not _ok_r(train_path["mean_r_haircut"]):
        blockers.append("train_m5_path_haircut_mean_r<=0")
    if val_path["n"] < 30 or not _ok_r(val_path["mean_r_haircut"]):
        blockers.append("val_m5_path_haircut_mean_r<=0")
    return {
        "id": "H192_xetra1730_fade_gold",
        "asof": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "oos_peeked": False,
        "lab_spread_pips": lab_pips,
        "lab_total_pips": lab_total,
        "weekday_median_spread_pips": wd_med,
        "fill_median_spread_pips": fill_med,
        "berlin_1730_median_spread_pips": float(fill.loc[h1730, "spread_pips"].median()) if h1730.any() else None,
        "berlin_14_weekday_median_spread_pips": float(wd_pips.loc[h14].median()) if h14.any() else None,
        "fill_vs_lab_mult": None if not fill_med else float(fill_med / lab_total),
        "train_lab": train_lab,
        "val_lab": val_lab,
        "train_spread_haircut": train_hc,
        "val_spread_haircut": val_hc,
        "train_m5_path": train_path,
        "val_m5_path": val_path,
        "m15_first_bar": first_bar,
        "path_tf_counts": {"M5": m5_path_n, "M15_fallback": m15_path_n},
        "blockers": blockers,
        "LAST_LEVEL_CLEAR": not blockers,
        "notes": [
            "Train/val only. OOS locked.",
            "Fill is the 17:30 Berlin M15 treat bar (next-open fill in handler). Gold MT5 spread points = lab units.",
            "Haircut is extra fill spread vs lab 200+20, in R (1R = 1 ATR).",
            "Path uses M5 when available; train falls back to M15 because broker gold M5 starts ~2025-04.",
        ],
    }


def main() -> None:
    blob = run_deep()
    out = DATA_DIR / "results" / "H192_deep.json"
    out.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(f"LAST_LEVEL_CLEAR={blob['LAST_LEVEL_CLEAR']}")
    print(f"blockers={blob['blockers']}")
    print(
        f"fill_spread={blob['fill_median_spread_pips']} pips "
        f"(lab {blob['lab_total_pips']}, weekday {blob['weekday_median_spread_pips']})"
    )
    print(f"train haircut R={blob['train_spread_haircut']}")
    print(f"val haircut R={blob['val_spread_haircut']}")
    print(f"train M5 path={blob['train_m5_path']}")
    print(f"val M5 path={blob['val_m5_path']}")
    print(f"m15 first bar={blob['m15_first_bar']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
