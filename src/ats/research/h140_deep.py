"""H140 last-level. Train/val only. Bar spread, M5 path, cost haircut. Not an EA. No OOS peek."""

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

POINTS_PER_PIP = 10.0
NY = ZoneInfo("America/New_York")
PARAMS = {
    "tz": "America/New_York",
    "treat_hour": 17,
    "treat_minute": 0,
    "base_hour": 16,
    "base_minute": 0,
    "mode": "fade",
    "skip_friday": True,
    "skip_weekend": True,
    "min_prior_atr": 0.25,
    "horizon_bars": 8,
    "stop_atr": 1.0,
    "target_atr": 1.0,
}


def _mean(x) -> float | None:
    if x is None or len(x) == 0:
        return None
    return float(np.mean(x))


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
    split = locked_calendar_split(settings, "fx_m15")
    costs = settings["costs"]
    lab_pips = float(costs["spread_pips"]["GBPUSD"])
    slip = float(costs.get("slippage_pips", 0.2))
    lab_total = lab_pips + slip
    raw = load_raw("GBPUSD", "M15")
    m15 = prepare_frame(raw, settings)
    ev = clock_run_events(m15, "GBPUSD", PARAMS, lab_pips, slip)
    treated = ev.loc[ev["treatment"]].copy()
    treated["time"] = pd.to_datetime(treated["time"], utc=True)
    train_m = treated["time"] <= split.train_end
    val_m = (treated["time"] > split.train_end) & (treated["time"] <= split.val_end)
    locked = treated.loc[train_m | val_m].copy()

    raw_s = raw.copy()
    raw_s["time"] = pd.to_datetime(raw_s["time"], utc=True)
    fill = locked.merge(raw_s[["time", "spread"]], on="time", how="left")
    fill["spread_pips"] = fill["spread"] / POINTS_PER_PIP
    ny = fill["time"].dt.tz_convert(NY)
    fill["ny_hour"] = ny.dt.hour
    wd = raw_s.copy()
    wd["ny"] = wd["time"].dt.tz_convert(NY)
    wd_pips = wd["spread"] / POINTS_PER_PIP
    weekday = wd["ny"].dt.dayofweek < 5

    pip = pip_size("GBPUSD")
    fill["extra_pips"] = (fill["spread_pips"] - lab_total).clip(lower=0)
    fill["extra_r"] = fill["extra_pips"] * pip / fill["atr"]
    fill["r_haircut"] = fill["r_mult"] - fill["extra_r"]

    m5_raw = load_raw("GBPUSD", "M5")
    m5_raw["time"] = pd.to_datetime(m5_raw["time"], utc=True)
    m5_raw = m5_raw.sort_values("time")
    path_rows = []
    first_bar = {"stop": 0, "target": 0, "both_stop_first": 0, "neither": 0}
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
        walk = _walk_bars(bars, str(tr["side"]), float(tr["entry"]), float(tr["stop"]), float(tr["target"]))
        first_spread = None
        if not bars.empty and "spread" in bars.columns:
            pts = float(bars.iloc[0]["spread"])
            if np.isfinite(pts):
                first_spread = pts / POINTS_PER_PIP
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
    h17 = ny.dt.hour == 17
    h16 = (wd["ny"].dt.hour == 16) & weekday
    blockers = []
    if fill_med is None or (wd_med and fill_med > 3.0 * max(lab_total, wd_med)):
        blockers.append("fill_bar_spread_gt_3x_lab")
    if train_hc["mean_r"] is None or train_hc["mean_r"] <= 0:
        blockers.append("train_spread_haircut_mean_r<=0")
    if val_hc["mean_r"] is None or val_hc["mean_r"] <= 0:
        blockers.append("val_spread_haircut_mean_r<=0")
    if train_path["mean_r_haircut"] is None or train_path["mean_r_haircut"] <= 0:
        blockers.append("train_m5_path_haircut_mean_r<=0")
    if val_path["mean_r_haircut"] is None or val_path["mean_r_haircut"] <= 0:
        blockers.append("val_m5_path_haircut_mean_r<=0")
    blob = {
        "id": "H140_tnext_roll_gbp",
        "asof": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "oos_peeked": False,
        "lab_spread_pips": lab_pips,
        "lab_total_pips": lab_total,
        "weekday_median_spread_pips": wd_med,
        "fill_median_spread_pips": fill_med,
        "ny17_median_spread_pips": float(fill.loc[h17, "spread_pips"].median()) if h17.any() else None,
        "ny16_weekday_median_spread_pips": float(wd_pips.loc[h16].median()) if h16.any() else None,
        "fill_vs_lab_mult": None if not fill_med else float(fill_med / lab_total),
        "train_lab": train_lab,
        "val_lab": val_lab,
        "train_spread_haircut": train_hc,
        "val_spread_haircut": val_hc,
        "train_m5_path": train_path,
        "val_m5_path": val_path,
        "m15_first_bar": first_bar,
        "blockers": blockers,
        "LAST_LEVEL_CLEAR": not blockers,
        "notes": [
            "Train/val only. OOS locked.",
            "Fill is next M15 after 17:00 NY (17:15). Spread is that bar's MT5 points/10.",
            "Haircut is extra fill spread vs lab 1.0+0.2 pips, in R (1R = 1 ATR).",
        ],
    }
    return blob


def main() -> None:
    blob = run_deep()
    out = DATA_DIR / "results" / "H140_deep.json"
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
