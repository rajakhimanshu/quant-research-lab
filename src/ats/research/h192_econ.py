"""H192 economics. Frozen spec only. Not a live return promise. Not an EA.

Computes R-based annualization and 0.01-lot USD paths on train/val/oos.
OOS may already be unlocked; this does not change the decision gate.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ats.config import DATA_DIR, load_settings
from ats.data.mt5_client import load_raw
from ats.features.prepare import prepare_frame
from ats.hypotheses.clock_family import clock_run_events
from ats.timeutil import locked_calendar_split

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
LOT = 0.01  # Exness min; ~1 oz on many gold contracts


def _oz_per_lot(lot: float) -> float:
    # Standard MT5 gold: 1.00 lot = 100 oz → 0.01 lot = 1 oz.
    return 100.0 * lot


def _part(treated: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    sub = treated.loc[mask].copy()
    if sub.empty:
        return {"label": label, "n": 0}
    t0 = pd.to_datetime(sub["time"].min(), utc=True)
    t1 = pd.to_datetime(sub["time"].max(), utc=True)
    years = max((t1 - t0).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    r = sub["r_mult"].astype(float)
    atr = sub["atr"].astype(float)
    # USD PnL for 0.01 lot: R * ATR($/oz) * oz
    pnl = (r * atr * _oz_per_lot(LOT)).to_numpy()
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return {
        "label": label,
        "n": int(len(sub)),
        "years": float(years),
        "trades_per_year": float(len(sub) / years),
        "mean_r": float(r.mean()),
        "total_r": float(r.sum()),
        "median_r": float(r.median()),
        "mean_atr_usd": float(atr.mean()),
        "total_pnl_usd_0p01": float(pnl.sum()),
        "mean_pnl_usd_0p01": float(pnl.mean()),
        "pnl_per_year_usd_0p01": float(pnl.sum() / years),
        "max_dd_usd_0p01": float(dd.min()) if len(dd) else None,
        "hit_rate": float((r > 0).mean()),
        "start": t0.isoformat(),
        "end": t1.isoformat(),
        # Not account-% without a chosen starting balance; show R/year instead.
        "r_per_year": float(r.sum() / years),
    }


def run_econ() -> dict:
    settings = load_settings()
    split = locked_calendar_split(settings, "gold_m15")
    costs = settings["costs"]
    lab = float(costs["spread_pips"]["XAUUSD"])
    slip = float((costs.get("slippage_by_symbol") or {}).get("XAUUSD", costs["slippage_pips"]))
    raw = load_raw("XAUUSD", "M15")
    m15 = prepare_frame(raw, settings)
    ev = clock_run_events(m15, "XAUUSD", PARAMS, lab, slip)
    treated = ev.loc[ev["treatment"]].copy()
    treated["time"] = pd.to_datetime(treated["time"], utc=True)
    train_m = treated["time"] <= split.train_end
    val_m = (treated["time"] > split.train_end) & (treated["time"] <= split.val_end)
    oos_m = treated["time"] > split.val_end
    parts = {
        "train": _part(treated, train_m, "train"),
        "validation": _part(treated, val_m, "validation"),
        "oos": _part(treated, oos_m, "oos"),
    }
    # Pre-committed OOS gates (written before unlock)
    oos = parts["oos"]
    train = parts["train"]
    gates = {
        "oos_n_ge_50": oos.get("n", 0) >= 50,
        "oos_mean_r_gt_0": (oos.get("mean_r") or 0) > 0,
        "train_oos_rate_gap_le_20pp": None,  # filled from rates below
    }
    # rates from success column if present
    if "success" in treated.columns:
        tr = float(treated.loc[train_m, "success"].mean()) if train_m.any() else None
        orate = float(treated.loc[oos_m, "success"].mean()) if oos_m.any() else None
        gap = None if tr is None or orate is None else abs(tr - orate)
        gates["train_oos_rate_gap"] = gap
        gates["train_oos_rate_gap_le_20pp"] = gap is not None and gap <= 0.20
        gates["oos_rate"] = orate
        gates["train_rate"] = tr
        base_oos = None
        if "treatment" in ev.columns:
            base = ev.loc[~ev["treatment"] & (pd.to_datetime(ev["time"], utc=True) > split.val_end)]
            if not base.empty and "success" in base.columns:
                base_oos = float(base["success"].mean())
        gates["oos_rate_above_baseline"] = (
            orate is not None and base_oos is not None and orate > base_oos
        )
        gates["oos_baseline_rate"] = base_oos
    gates["PASS"] = bool(
        gates.get("oos_n_ge_50")
        and gates.get("oos_mean_r_gt_0")
        and gates.get("train_oos_rate_gap_le_20pp")
        and gates.get("oos_rate_above_baseline")
    )
    return {
        "id": "H192_xetra1730_fade_gold",
        "asof": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lot": LOT,
        "oz_per_trade": _oz_per_lot(LOT),
        "split": {"train_end": str(split.train_end), "val_end": str(split.val_end)},
        "parts": parts,
        "gates": gates,
        "notes": [
            "Per-annum $ is lab PnL on 0.01 lot (1 oz), not % of a $100/$1000 account.",
            "1R = 1 ATR price move. Mean R ~0.02 is two cents of R per trade before account scaling.",
            "On $100 with 0.01 lot, risk per trade is ~ATR USD — often >>1% of equity (retail path note).",
            "OOS unlock is a one-shot honest check. Fail = do not retune. Not an EA.",
        ],
    }


def main() -> None:
    blob = run_econ()
    out = DATA_DIR / "results" / "H192_econ.json"
    out.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    g = blob["gates"]
    print(f"GATES_PASS={g['PASS']}")
    print(f"gates={json.dumps(g, indent=2)}")
    for name, p in blob["parts"].items():
        if p.get("n", 0) == 0:
            print(f"{name}: n=0")
            continue
        print(
            f"{name}: n={p['n']} years={p['years']:.2f} trades/y={p['trades_per_year']:.0f} "
            f"mean_R={p['mean_r']:.4f} R/y={p['r_per_year']:.2f} "
            f"$/y@0.01lot={p['pnl_per_year_usd_0p01']:.1f} "
            f"total$={p['total_pnl_usd_0p01']:.1f} maxDD$={p['max_dd_usd_0p01']:.1f}"
        )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
