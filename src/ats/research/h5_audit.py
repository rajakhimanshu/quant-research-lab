"""Pre-specified H5 battery. Frozen params. No retune."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from ats.config import DATA_DIR, load_hypotheses, load_settings
from ats.hypotheses.equity_rsi2 import equity_rsi2_events
from ats.timeutil import time_splits

FOLDS = [
    ("2016-2017", "2015-12-31", "2016-01-01", "2017-12-31"),
    ("2018-2019", "2017-12-31", "2018-01-01", "2019-12-31"),
    ("2020-2021", "2019-12-31", "2020-01-01", "2021-12-31"),
    ("2022-2023", "2021-12-31", "2022-01-01", "2023-12-31"),
    ("2024-2026", "2023-12-31", "2024-01-01", "2026-12-31"),
]
COSTS = [0.00015, 0.00030, 0.00050]
MAX_OPEN = 4


def _block(name: str, part: pd.DataFrame) -> dict:
    t = part.loc[part["treatment"]] if "treatment" in part.columns else part
    b = part.loc[~part["treatment"]] if "treatment" in part.columns else pd.DataFrame()
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
        "avg_hold": float(t["hold"].mean()) if "hold" in t.columns else None,
    }
    if not b.empty:
        out["base_n"] = int(len(b))
        out["base_win_rate"] = float(b["success"].mean())
        out["base_mean_r"] = float(b["r_mult"].mean())
    return out


def _cap_concurrent(trades: pd.DataFrame, max_open: int) -> pd.DataFrame:
    t = trades.sort_values("time").copy()
    if "exit_time" not in t.columns:
        return t
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)
    t["time"] = pd.to_datetime(t["time"], utc=True)
    kept = []
    open_exits: list[pd.Timestamp] = []
    for _, row in t.iterrows():
        open_exits = [e for e in open_exits if e > row["time"]]
        if len(open_exits) >= max_open:
            continue
        kept.append(row)
        open_exits.append(row["exit_time"])
    return pd.DataFrame(kept)


def run_audit() -> dict:
    hyp = next(h for h in load_hypotheses() if h["id"] == "H5_equity_rsi2")
    settings = load_settings()
    params = dict(hyp["params"])
    events = equity_rsi2_events(params)
    events["time"] = pd.to_datetime(events["time"], utc=True)
    if "exit_time" in events.columns:
        events["exit_time"] = pd.to_datetime(events["exit_time"], utc=True)
    treated = events.loc[events["treatment"]].copy()
    v = settings["validation"]
    split = time_splits(pd.DatetimeIndex(events["time"]), float(v["train"]), float(v["validation"]))
    times = events["time"]
    masks = {
        "train": times <= split.train_end,
        "validation": (times > split.train_end) & (times <= split.val_end),
        "oos": times > split.val_end,
    }

    splits = {name: _block(name, events.loc[mask]) for name, mask in masks.items()}
    train_r = splits["train"]["mean_r"]
    oos_r = splits["oos"]["mean_r"]
    val_r = splits["validation"]["mean_r"]

    per_ticker = []
    for sym, g in treated.groupby("symbol"):
        row = {"symbol": sym}
        gt = g["time"]
        parts = {
            "train": g.loc[gt <= split.train_end],
            "validation": g.loc[(gt > split.train_end) & (gt <= split.val_end)],
            "oos": g.loc[gt > split.val_end],
        }
        for name, sub in parts.items():
            row[name] = {
                "n": int(len(sub)),
                "win_rate": float(sub["success"].mean()) if len(sub) else None,
                "mean_r": float(sub["r_mult"].mean()) if len(sub) else None,
            }
        row["full_mean_r"] = float(g["r_mult"].mean())
        g70 = g.sort_values("time")
        cut = int(len(g70) * 0.70)
        hold = g70.iloc[cut:]
        row["legacy_70_30"] = {
            "in_n": cut,
            "holdout_n": int(len(hold)),
            "in_mean_r": float(g70.iloc[:cut]["r_mult"].mean()) if cut else None,
            "holdout_mean_r": float(hold["r_mult"].mean()) if len(hold) else None,
        }
        per_ticker.append(row)

    legacy_pass = sum(
        1
        for r in per_ticker
        if (r["legacy_70_30"]["holdout_n"] or 0) >= 20
        and (r["legacy_70_30"]["in_mean_r"] or 0) > 0
        and (r["legacy_70_30"]["holdout_mean_r"] or 0) > 0
    )

    folds = []
    for label, train_to, test_from, test_to in FOLDS:
        train_to_ts = pd.Timestamp(train_to, tz="UTC")
        a = pd.Timestamp(test_from, tz="UTC")
        b = pd.Timestamp(test_to, tz="UTC")
        test = treated.loc[(treated["time"] >= a) & (treated["time"] <= b)]
        prior = treated.loc[treated["time"] <= train_to_ts]
        folds.append(
            {
                "fold": label,
                "train_to": train_to,
                "test_n": int(len(test)),
                "test_mean_r": float(test["r_mult"].mean()) if len(test) else None,
                "prior_n": int(len(prior)),
                "prior_mean_r": float(prior["r_mult"].mean()) if len(prior) else None,
            }
        )
    fold_positive = sum(1 for f in folds if f["test_n"] >= 30 and (f["test_mean_r"] or 0) > 0)

    cost_rows = []
    for c in COSTS:
        p = dict(params)
        p["cost_one_way"] = c
        ev = equity_rsi2_events(p)
        ev["time"] = pd.to_datetime(ev["time"], utc=True)
        t = ev.loc[ev["treatment"]]
        cost_rows.append(
            {
                "cost_one_way_bp": round(c * 10000, 2),
                "n": int(len(t)),
                "mean_r": float(t["r_mult"].mean()) if len(t) else None,
                "oos_mean_r": float(t.loc[t["time"] > split.val_end, "r_mult"].mean())
                if (t["time"] > split.val_end).any()
                else None,
            }
        )

    capped = _cap_concurrent(treated, MAX_OPEN)
    same_day = (
        treated.assign(day=treated["time"].dt.floor("D"))
        .groupby("day")
        .size()
        .value_counts()
        .sort_index()
    )
    yearly = []
    for y, g in treated.groupby(treated["time"].dt.year):
        t = g["time"]
        if (t <= split.train_end).all():
            bucket = "train"
        elif (t > split.val_end).all():
            bucket = "oos"
        elif ((t > split.train_end) & (t <= split.val_end)).all():
            bucket = "validation"
        else:
            bucket = "mixed"
        yearly.append(
            {
                "year": int(y),
                "n": int(len(g)),
                "mean_r": float(g["r_mult"].mean()),
                "total_r": float(g["r_mult"].sum()),
                "bucket": bucket,
            }
        )

    oos_gap_r = abs(train_r - oos_r) if train_r == train_r and oos_r == oos_r else None
    val_gap_r = abs(train_r - val_r)
    economic = (
        splits["oos"]["n"] >= 30
        and splits["oos"]["mean_r"] > 0
        and splits["validation"]["mean_r"] > 0
        and splits["train"]["mean_r"] > 0
        and (oos_gap_r is None or oos_gap_r <= 0.20)
        and val_gap_r <= 0.20
        and fold_positive >= 3
        and cost_rows[0]["oos_mean_r"] is not None
        and cost_rows[0]["oos_mean_r"] > 0
    )
    report = {
        "hypothesis": "H5_equity_rsi2",
        "frozen": True,
        "params": params,
        "split_dates": {
            "train_end": str(split.train_end),
            "val_end": str(split.val_end),
        },
        "splits": splits,
        "train_val_gap_r": val_gap_r,
        "train_oos_gap_r": oos_gap_r,
        "per_ticker": per_ticker,
        "legacy_70_30_pass": f"{legacy_pass}/{len(per_ticker)}",
        "walk_forward": folds,
        "walk_forward_positive_folds": f"{fold_positive}/{len(folds)}",
        "cost_stress": cost_rows,
        "portfolio_max4": _block("max4", capped.assign(treatment=True)),
        "portfolio_uncapped": _block("uncapped", treated.assign(treatment=True)),
        "same_day_cluster_counts": {int(k): int(v) for k, v in same_day.items()},
        "yearly": yearly,
        "economic_gate": "PASS" if economic else "FAIL",
        "notes": [
            "Parameters frozen. No RSI/SMA/hold search.",
            "Win rate vs below-SMA200 is the original contrast; mean R is the economic test.",
            "SPY/QQQ/DIA/XLK overlap — n is not independent.",
            "Not a live go. Paper only if economic_gate PASS.",
        ],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = DATA_DIR / "results" / f"H5_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["path"] = str(path)
    return report


def print_audit(report: dict) -> None:
    print("=" * 64)
    print("H5 equity RSI(2) — frozen audit (OOS unlocked)")
    print(f"Train end {report['split_dates']['train_end']}  val end {report['split_dates']['val_end']}")
    for name, block in report["splits"].items():
        print(
            f"  {name:12} n={block.get('n')} wr={block.get('win_rate')} "
            f"meanR={block.get('mean_r')} baseR={block.get('base_mean_r')} "
            f"dd={block.get('max_dd_r')}"
        )
    print(f"Train-val gap R={report['train_val_gap_r']}  train-OOS gap R={report['train_oos_gap_r']}")
    print(f"Legacy 70/30 names with +E in and holdout: {report['legacy_70_30_pass']}")
    print("Walk-forward test folds:")
    for f in report["walk_forward"]:
        print(f"  {f['fold']} n={f['test_n']} meanR={f['test_mean_r']}")
    print("Cost stress (treatment, all years + OOS):")
    for c in report["cost_stress"]:
        print(f"  {c['cost_one_way_bp']}bp one-way  all={c['mean_r']}  oos={c['oos_mean_r']}")
    print(
        "Portfolio uncapped",
        report["portfolio_uncapped"].get("mean_r"),
        "max4",
        report["portfolio_max4"].get("mean_r"),
        "n",
        report["portfolio_max4"].get("n"),
    )
    print(f"ECONOMIC GATE: {report['economic_gate']}")
    print(f"Wrote {report['path']}")
    print("=" * 64)


if __name__ == "__main__":
    print_audit(run_audit())
