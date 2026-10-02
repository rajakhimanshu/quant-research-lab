"""Frozen event templates the automated lab is allowed to run.

A template is a tested event engine with a built-in baseline, a closed
parameter schema, and the lab fill model (next-bar open +/- cost, net R).
Proposals (from papers, the mechanism library, an LLM, or a person) can only
choose a template and values inside its schema. Nothing outside this file
becomes a runnable test without a human writing and reviewing code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ats.hypotheses.clock_family import clock_run_events
from ats.hypotheses.execution import mirror_trades
from ats.hypotheses.forced_flow import month_end_rebalance_events, weekend_gap_fade_events
from ats.hypotheses.m15_micro import session_box_fade_events
from ats.hypotheses.stop_cascade import stop_cascade_absorption_events
from ats.timeutil import cost_price

UNIVERSE = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "XAUUSD")
TIMEFRAMES = ("M5", "M15", "H1")
TIMEZONES = (
    "UTC", "Europe/London", "Europe/Berlin", "America/New_York",
    "Asia/Tokyo", "Asia/Singapore", "Asia/Hong_Kong", "Australia/Sydney", "Asia/Dubai",
)


@dataclass(frozen=True)
class Param:
    kind: type
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None
    default: Any = None
    required: bool = False

    def check(self, name: str, value: Any) -> tuple[Any, str | None]:
        try:
            if self.kind is bool:
                if not isinstance(value, bool):
                    return value, f"{name} must be true/false"
                v = value
            else:
                v = self.kind(value)
        except (TypeError, ValueError):
            return value, f"{name}={value!r} is not {self.kind.__name__}"
        if self.choices is not None and v not in self.choices:
            return v, f"{name}={v!r} not in {list(self.choices)}"
        if self.lo is not None and v < self.lo:
            return v, f"{name}={v} below {self.lo}"
        if self.hi is not None and v > self.hi:
            return v, f"{name}={v} above {self.hi}"
        return v, None


def _risk() -> dict[str, Param]:
    return {
        "horizon_bars": Param(int, 2, 48, default=8),
        "stop_atr": Param(float, 0.5, 3.0, default=1.0),
        "target_atr": Param(float, 0.5, 4.0, default=1.0),
    }


@dataclass(frozen=True)
class Template:
    name: str
    mechanism: str
    baseline: str
    timeframes: tuple[str, ...]
    params: dict[str, Param]
    run: Callable[..., pd.DataFrame]
    needs_leader: bool = False
    rules: tuple[Callable[[dict], str | None], ...] = field(default_factory=tuple)

    def catalogue_entry(self) -> dict:
        schema = {}
        for k, p in self.params.items():
            d: dict[str, Any] = {"type": p.kind.__name__}
            if p.choices is not None:
                d["choices"] = list(p.choices)
            if p.lo is not None:
                d["min"], d["max"] = p.lo, p.hi
            d["required" if p.required else "default"] = True if p.required else p.default
            schema[k] = d
        return {"template": self.name, "mechanism": self.mechanism, "baseline": self.baseline,
                "timeframes": list(self.timeframes), "params": schema}


def _flip_mirror(ev: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Mirror-baseline engines emit fade as treatment; follow swaps the arms."""
    if ev.empty or mode != "follow":
        return ev
    ev = ev.copy()
    ev["treatment"] = ~ev["treatment"].astype(bool)
    return ev


def _clock_run(df, symbol, params, spread, slip, ctx):
    return clock_run_events(df, symbol, params, spread, slip)


def _session_box(df, symbol, params, spread, slip, ctx):
    return _flip_mirror(session_box_fade_events(df, symbol, params, spread, slip), params.get("mode", "fade"))


def _sweep_reclaim(df, symbol, params, spread, slip, ctx):
    return _flip_mirror(stop_cascade_absorption_events(df, symbol, params, spread, slip), params.get("mode", "fade"))


def _weekend_gap(df, symbol, params, spread, slip, ctx):
    return weekend_gap_fade_events(df, symbol, params, spread, slip)


def _month_end(df, symbol, params, spread, slip, ctx):
    return month_end_rebalance_events(df, symbol, params, spread, slip)


def lead_lag_events(df, symbol, params, spread, slip, leader: pd.DataFrame) -> pd.DataFrame:
    """Leader moves > k ATR over n bars in a UTC window; follower trades `delay` bars later.

    Treatment follows the leader's direction; baseline is the opposite side on the same bar.
    """
    work = df.dropna(subset=["atr"]).copy().reset_index(drop=True)
    lead = leader.dropna(subset=["atr"]).copy()
    if work.empty or lead.empty:
        return pd.DataFrame()
    n = int(params.get("leader_move_bars", 3))
    k = float(params.get("leader_move_atr", 1.5))
    delay = int(params.get("delay_bars", 1))
    start_h = int(params.get("session_start_hour_utc", 7))
    end_h = int(params.get("session_end_hour_utc", 16))
    horizon = int(params.get("horizon_bars", 3))
    stop_atr = float(params.get("stop_atr", 1.0))
    target_atr = float(params.get("target_atr", 1.0))
    cost = cost_price(symbol, spread, slip)

    lt = pd.to_datetime(lead["time"], utc=True)
    move = lead["close"].astype(float) - lead["close"].astype(float).shift(n)
    hour = lt.dt.hour
    in_window = (hour >= start_h) & (hour < end_h)
    sig = pd.Series(0.0, index=lead.index)
    sig[(move > k * lead["atr"]) & in_window] = 1.0
    sig[(move < -k * lead["atr"]) & in_window] = -1.0
    signals = pd.Series(sig.to_numpy(), index=lt.to_numpy())

    ft = pd.to_datetime(work["time"], utc=True)
    aligned = signals.reindex(ft.to_numpy()).fillna(0.0).to_numpy()
    rows: list[dict] = []
    busy_until = -1
    for s in np.flatnonzero(aligned != 0.0):
        t = int(s) + delay
        if t <= busy_until or t + 1 + horizon >= len(work):
            continue
        side = "long" if aligned[s] > 0 else "short"
        rows.extend(mirror_trades(work, t, side, symbol, cost, horizon, stop_atr, target_atr))
        busy_until = t + horizon
    return _flip_mirror(pd.DataFrame(rows), params.get("mode", "follow"))


def _lead_lag(df, symbol, params, spread, slip, ctx):
    leader = ctx["load"](params["leader"], ctx["timeframe"])
    return lead_lag_events(df, symbol, params, spread, slip, leader)


def _clock_distinct(p: dict) -> str | None:
    if (p.get("treat_dow") is None) != (p.get("base_dow") is None):
        return "treat_dow and base_dow must be set together (else no baseline arm)"
    same_time = (p["treat_hour"], p.get("treat_minute", 0)) == (p["base_hour"], p.get("base_minute", 0))
    if same_time and p.get("treat_dow") == p.get("base_dow"):
        return "treat clock and base clock are identical (no baseline)"
    return None


def _box_order(p: dict) -> str | None:
    start = p["box_start_hour"] * 60 + p.get("box_start_minute", 0)
    end = p["box_end_hour"] * 60 + p.get("box_end_minute", 0)
    if end <= start:
        return "box end must be after box start"
    if p.get("hunt_end_hour", 12) * 60 <= end:
        return "hunt_end_hour must be after the box"
    return None


def _window_order(p: dict) -> str | None:
    if p.get("session_end_hour_utc", 16) <= p.get("session_start_hour_utc", 7):
        return "session_end_hour_utc must be after session_start_hour_utc"
    return None


def _mid_month(p: dict) -> str | None:
    if p.get("mid_dom_hi", 14) < p.get("mid_dom_lo", 10):
        return "mid_dom_hi must be >= mid_dom_lo"
    return None


MINUTES = (0, 15, 30, 45)

TEMPLATES: dict[str, Template] = {
    t.name: t
    for t in (
        Template(
            name="clock_run",
            mechanism="A named clock (fix, auction, session open/close, settlement) forces flow; "
                      "fade or follow the bar into that clock.",
            baseline="Same rule at a control clock on the same days.",
            timeframes=("M5", "M15", "H1"),
            params={
                "tz": Param(str, choices=TIMEZONES, required=True),
                "treat_hour": Param(int, 0, 23, required=True),
                "treat_minute": Param(int, choices=MINUTES, default=0),
                "base_hour": Param(int, 0, 23, required=True),
                "base_minute": Param(int, choices=MINUTES, default=0),
                "mode": Param(str, choices=("fade", "follow"), required=True),
                "min_prior_atr": Param(float, 0.1, 1.5, default=0.25),
                "skip_weekend": Param(bool, default=True),
                "treat_dow": Param(int, 0, 4),
                "base_dow": Param(int, 0, 4),
                **_risk(),
            },
            run=_clock_run,
            rules=(_clock_distinct,),
        ),
        Template(
            name="session_box",
            mechanism="Stops cluster around a session's opening range; the first break is a stop run "
                      "(fade) or initiative flow (follow).",
            baseline="Opposite side on the same break bar.",
            timeframes=("M5", "M15"),
            params={
                "tz": Param(str, choices=TIMEZONES, required=True),
                "box_start_hour": Param(int, 0, 23, required=True),
                "box_start_minute": Param(int, choices=MINUTES, default=0),
                "box_end_hour": Param(int, 0, 23, required=True),
                "box_end_minute": Param(int, choices=MINUTES, default=0),
                "box_bars": Param(int, 1, 12, default=2),
                "hunt_end_hour": Param(int, 1, 23, default=12),
                "mode": Param(str, choices=("fade", "follow"), default="fade"),
                "skip_weekend": Param(bool, default=True),
                **_risk(),
            },
            run=_session_box,
            rules=(_box_order,),
        ),
        Template(
            name="sweep_reclaim",
            mechanism="Dealers absorb stop orders just beyond a recent swing; a shallow sweep that "
                      "closes back inside reverts as inventory is unwound.",
            baseline="Opposite side on the same sweep bar.",
            timeframes=("M5", "M15", "H1"),
            params={
                "swing_lookback": Param(int, 5, 100, default=20),
                "min_bars_since": Param(int, 1, 20, default=5),
                "mode": Param(str, choices=("fade", "follow"), default="fade"),
                **_risk(),
            },
            run=_sweep_reclaim,
        ),
        Template(
            name="lead_lag",
            mechanism="Price discovery happens first in the more liquid market; the follower's dealers "
                      "reprice with a lag.",
            baseline="Opposite side on the same follower bar.",
            timeframes=("M5", "M15"),
            params={
                "leader": Param(str, choices=UNIVERSE, required=True),
                "leader_move_bars": Param(int, 1, 12, default=3),
                "leader_move_atr": Param(float, 0.5, 3.0, default=1.5),
                "delay_bars": Param(int, 0, 6, default=1),
                "session_start_hour_utc": Param(int, 0, 23, default=7),
                "session_end_hour_utc": Param(int, 1, 24, default=16),
                "mode": Param(str, choices=("fade", "follow"), default="follow"),
                **_risk(),
            },
            run=_lead_lag,
            needs_leader=True,
            rules=(_window_order,),
        ),
        Template(
            name="weekend_gap",
            mechanism="Weekend news reprices with no liquidity; the reopen gap overshoots and is faded "
                      "as dealers return.",
            baseline="Fade of a same-size London-open jump on a normal day.",
            timeframes=("M5", "M15", "H1"),
            params={
                "min_gap_atr": Param(float, 0.1, 2.0, default=0.3),
                "weekend_gap_hours": Param(float, 24, 72, default=24),
                "london_open_hour": Param(int, 6, 10, default=8),
                **_risk(),
            },
            run=_weekend_gap,
        ),
        Template(
            name="month_end",
            mechanism="Month-end portfolio rebalancing forces flow in the month-to-date direction "
                      "over the last sessions.",
            baseline="Same follow on mid-month sessions.",
            timeframes=("M15", "H1"),
            params={
                "last_n_sessions": Param(int, 1, 5, default=3),
                "mid_dom_lo": Param(int, 5, 20, default=10),
                "mid_dom_hi": Param(int, 5, 20, default=14),
                "entry_hour_ny": Param(int, 0, 23, default=9),
                **_risk(),
            },
            run=_month_end,
            rules=(_mid_month,),
        ),
    )
}


def catalogue() -> list[dict]:
    return [t.catalogue_entry() for t in TEMPLATES.values()]


def template_events(name: str, df: pd.DataFrame, symbol: str, params: dict,
                    spread: float, slip: float, ctx: dict) -> pd.DataFrame:
    return TEMPLATES[name].run(df, symbol, params, spread, slip, ctx)
