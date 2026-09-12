"""H86 audit battery. Frozen params. No retune of 36h or pairs."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ats.config import DATA_DIR, load_settings
from ats.data.mt5_client import load_raw
from ats.features.prepare import prepare_frame
from ats.hypotheses.fx_session import weekend_gap_fx_events
from ats.paper.h86_book import h86_params, pair_cost
from ats.research.stats import rates
from ats.timeutil import locked_calendar_split

FOLDS = [
    ("holdout_2023", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("holdout_2024", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("holdout_2025", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("holdout_2026", "2025-12-31", "2026-01-01", "2026-09-04"),
]
COST_MULT = [1.0, 2.0, 3.0, 5.0]
BOOT = 2000
RNG = np.random.default_rng(86)


def _block(name: str, part: pd.DataFrame) -> dict:
    t = part.loc[part["treatment"]] if "treatment" in part.columns and not part.empty else part
    b = part.loc[~part["treatment"]] if "treatment" in part.columns and not part.empty else pd.DataFrame()
    if t.empty:
        return {"split": name, "n": 0}
    wins = t.loc[t["r_mult"] > 0, "r_mult"].sum()
    losses = -t.loc[t["r_mult"] <= 0, "r_mult"].sum()
    eq = t.sort_values("time")["r_mult"].cumsum()
    dd = eq - eq.cummax()
    out = {
        "split": name,
        "n": int(len(t)),
        "win_rate": float(t["success"].mean()),
        "mean_r": float(t["r_mult"].mean()),
        "median_r": float(t["r_mult"].median()),
        "total_r": float(t["r_mult"].sum()),
        "profit_factor": float(wins / losses) if losses > 0 else None,
        "max_dd_r": float(dd.min()) if len(dd) else None,
        "longest_loss_streak": _loss_streak(t.sort_values("time")["r_mult"]),
    }
    if not b.empty:
        rr = rates(part["success"], part["treatment"])
        out["base_n"] = int(len(b))
        out["base_win_rate"] = float(b["success"].mean())
        out["base_mean_r"] = float(b["r_mult"].mean())
        out["p_value"] = rr.p_value
    return out


def _loss_streak(r: pd.Series) -> int:
    best = cur = 0
    for x in r:
        if x <= 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _all_events(frames: dict, extra: dict | None = None, cost_mult: float = 1.0) -> pd.DataFrame:
    settings = load_settings()
    params = {**h86_params(), **(extra or {})}
    chunks = []
    for symbol, df in frames.items():
        spread, slip = pair_cost(symbol, settings)
        ev = weekend_gap_fx_events(df, symbol, params, spread * cost_mult, slip * cost_mult)
        if ev is not None and not ev.empty:
            chunks.append(ev)
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    return out.sort_values("time")


def _bootstrap_mean(x: np.ndarray, n: int = BOOT) -> dict:
    if len(x) < 5:
        return {"mean": None, "ci_low": None, "ci_high": None}
    means = [float(RNG.choice(x, size=len(x), replace=True).mean()) for _ in range(n)]
    return {
        "mean": float(np.mean(x)),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "p_leq_zero": float(np.mean(np.array(means) <= 0)),
    }


def run_audit() -> dict:
    settings = load_settings()
    split = locked_calendar_split(settings, "fx_h1")
    params = h86_params()
    frames = {
        symbol: prepare_frame(load_raw(symbol, params.get("timeframe", "H1")), settings)
        for symbol in params["symbols"]
    }
    ev = _all_events(frames)
    tmask = ev["treatment"]
    treated = ev.loc[tmask].copy()
    times = ev["time"]
    masks = {
        "train": times <= split.train_end,
        "validation": (times > split.train_end) & (times <= split.val_end),
        "oos": times > split.val_end,
    }
    splits = {name: _block(name, ev.loc[mask]) for name, mask in masks.items()}

    per_symbol = []
    for sym, g in treated.groupby("symbol"):
        row = {"symbol": sym}
        gt = g["time"]
        for name, lo, hi in (
            ("train", None, split.train_end),
            ("validation", split.train_end, split.val_end),
            ("oos", split.val_end, None),
        ):
            if lo is None:
                sub = g.loc[gt <= hi]
            elif hi is None:
                sub = g.loc[gt > lo]
            else:
                sub = g.loc[(gt > lo) & (gt <= hi)]
            row[name] = {
                "n": int(len(sub)),
                "win_rate": float(sub["success"].mean()) if len(sub) else None,
                "mean_r": float(sub["r_mult"].mean()) if len(sub) else None,
            }
        per_symbol.append(row)

    folds = []
    for label, train_to, test_from, test_to in FOLDS:
        a = pd.Timestamp(test_from, tz="UTC")
        b = pd.Timestamp(test_to, tz="UTC")
        prior = treated.loc[treated["time"] <= pd.Timestamp(train_to, tz="UTC")]
        test = treated.loc[(treated["time"] >= a) & (treated["time"] <= b)]
        folds.append(
            {
                "fold": label,
                "prior_n": int(len(prior)),
                "prior_mean_r": float(prior["r_mult"].mean()) if len(prior) else None,
                "test_n": int(len(test)),
                "test_mean_r": float(test["r_mult"].mean()) if len(test) else None,
                "test_win": float(test["success"].mean()) if len(test) else None,
            }
        )
    fold_ok = sum(1 for f in folds if f["test_n"] >= 30 and (f["test_mean_r"] or 0) > 0)

    cost_rows = []
    for m in COST_MULT:
        c_ev = _all_events(frames, cost_mult=m)
        ct = c_ev.loc[c_ev["treatment"]]
        oos = ct.loc[ct["time"] > split.val_end]
        cost_rows.append(
            {
                "cost_mult": m,
                "n": int(len(ct)),
                "mean_r": float(ct["r_mult"].mean()) if len(ct) else None,
                "oos_n": int(len(oos)),
                "oos_mean_r": float(oos["r_mult"].mean()) if len(oos) else None,
            }
        )

    delay = _all_events(frames, {"fill_delay_bars": 1})
    dt = delay.loc[delay["treatment"]] if not delay.empty else pd.DataFrame()
    delay_oos = dt.loc[dt["time"] > split.val_end] if not dt.empty else pd.DataFrame()
    follow = _all_events(frames, {"follow_gap": True})
    ft = follow.loc[follow["treatment"]] if not follow.empty else pd.DataFrame()

    london = treated["time"].dt.tz_convert("Europe/London")
    treated = treated.copy()
    treated["week"] = london.dt.strftime("%G-W%V")
    weekly = treated.groupby("week")["r_mult"].sum()
    same_sunday = treated.groupby("week").size().value_counts().sort_index()

    boot_trade = _bootstrap_mean(treated["r_mult"].to_numpy())
    boot_week = _bootstrap_mean(weekly.to_numpy())
    oos_t = treated.loc[treated["time"] > split.val_end]
    boot_oos = _bootstrap_mean(oos_t["r_mult"].to_numpy()) if len(oos_t) else {}

    # Permutation: random ±1 on |R| (a shuffle of observed signs keeps the win rate).
    obs = float(treated["r_mult"].mean())
    mag = np.abs(treated["r_mult"].to_numpy())
    perm = np.array(
        [(RNG.choice(np.array([-1.0, 1.0]), size=len(mag)) * mag).mean() for _ in range(BOOT)]
    )
    perm_p = float(np.mean(perm >= obs))

    yearly = []
    for y, g in treated.groupby(treated["time"].dt.year):
        yearly.append({"year": int(y), "n": int(len(g)), "mean_r": float(g["r_mult"].mean()), "total_r": float(g["r_mult"].sum())})

    invert_mean = None if ft.empty else float(ft["r_mult"].mean())
    cost3_oos = cost_rows[2]["oos_mean_r"]
    fail_reasons = []
    if splits["oos"]["n"] < 30:
        fail_reasons.append("oos_n<30")
    if not (splits["oos"]["mean_r"] > 0):
        fail_reasons.append("oos_mean_r<=0")
    if not (splits["train"]["mean_r"] > 0):
        fail_reasons.append("train_mean_r<=0")
    if not (splits["validation"]["mean_r"] > 0):
        fail_reasons.append("val_mean_r<=0")
    if fold_ok < 3:
        fail_reasons.append("walk_forward_positive_folds<3")
    if cost3_oos is None or not (cost3_oos > 0):
        fail_reasons.append("cost_x3_oos_mean_r<=0")
    if invert_mean is not None and invert_mean >= obs:
        fail_reasons.append("follow_gap_not_worse_than_fade")
    economic = not fail_reasons
    payload = {
        "hypothesis": "H86_weekend_gap_g10",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "frozen": True,
        "splits": splits,
        "per_symbol": per_symbol,
        "walk_forward": folds,
        "walk_forward_positive_folds": fold_ok,
        "cost_mult": cost_rows,
        "fill_delay_1bar": {
            "n": int(len(dt)),
            "mean_r": float(dt["r_mult"].mean()) if len(dt) else None,
            "oos_n": int(len(delay_oos)),
            "oos_mean_r": float(delay_oos["r_mult"].mean()) if len(delay_oos) else None,
        },
        "follow_gap_instead": {
            "n": int(len(ft)),
            "mean_r": float(ft["r_mult"].mean()) if len(ft) else None,
        },
        "bootstrap_trade_mean_r": boot_trade,
        "bootstrap_week_sum_r": boot_week,
        "bootstrap_oos_mean_r": boot_oos,
        "permutation_p_mean_r": perm_p,
        "same_weekend_pair_counts": {int(k): int(v) for k, v in same_sunday.items()},
        "yearly": yearly,
        "economic_pass": economic,
        "economic_fail_reasons": fail_reasons,
        "notes": [
            "Do not retune 36h or min_gap_atr from this battery.",
            "Do not add NZD/CAD/CHF.",
            "Fill-delay and follow-gap are falsification, not new hyps.",
            "Lab costs are weekday median spreads. Sunday reopen is wider; cost x2/x3 is the stress.",
            "Not live. EA only if economic_pass.",
        ],
    }
    path = DATA_DIR / "results" / "H86_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    payload["_path"] = str(path)
    return payload


def print_audit(blob: dict) -> None:
    print("H86 audit — frozen weekend-gap fade. No retune.")
    for name, s in blob["splits"].items():
        print(
            f"  {name:12} n={s.get('n')} wr={s.get('win_rate')} "
            f"meanR={s.get('mean_r')} vs base {s.get('base_win_rate')} "
            f"pf={s.get('profit_factor')} ddR={s.get('max_dd_r')} streak={s.get('longest_loss_streak')}"
        )
    print("Walk-forward holdouts:")
    for f in blob["walk_forward"]:
        print(f"  {f['fold']:14} priorR={f['prior_mean_r']} test n={f['test_n']} meanR={f['test_mean_r']}")
    print("Cost multiplier (oos mean R):")
    for c in blob["cost_mult"]:
        print(f"  x{c['cost_mult']} allR={c['mean_r']} oosR={c['oos_mean_r']}")
    d = blob["fill_delay_1bar"]
    print(f"Fill delay +1 H1: meanR={d['mean_r']} oosR={d['oos_mean_r']}")
    print(f"Follow gap (invert): meanR={blob['follow_gap_instead']['mean_r']}")
    print(f"Bootstrap trade mean R 95% CI: {blob['bootstrap_trade_mean_r']}")
    print(f"Bootstrap OOS mean R 95% CI: {blob['bootstrap_oos_mean_r']}")
    print(f"Permutation p (obs mean R vs random ±1 on |R|): {blob['permutation_p_mean_r']}")
    print(f"ECONOMIC_PASS={blob['economic_pass']}")
    if blob.get("economic_fail_reasons"):
        print(f"Fail reasons: {blob['economic_fail_reasons']}")
    print(f"Wrote {blob.get('_path')}")
    print("Not live. Not an EA until this print is green and you still want MT5 tester.")
