from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest


@dataclass
class SplitRates:
    n: int
    successes: int
    rate: float
    baseline_n: int
    baseline_successes: int
    baseline_rate: float
    p_value: float | None


@dataclass
class HypothesisReport:
    hypothesis: str
    name: str
    why: str
    sample_size: int
    success_rate: float | None
    baseline_rate: float | None
    p_value: float | None
    train: dict
    validation: dict
    oos: dict
    regime_breakdown: dict
    cost_adjusted_success_rate: float | None
    train_val_gap: float | None
    decision: str
    reject_reason: str | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def proportion_test(successes: int, n: int, base_successes: int, base_n: int) -> float | None:
    if min(n, base_n) < 10:
        return None
    if successes > n or base_successes > base_n:
        return None
    stat, p = proportions_ztest(
        np.array([successes, base_successes]),
        np.array([n, base_n]),
        alternative="larger",
    )
    if not np.isfinite(p):
        return None
    return float(p)


def rates(success: pd.Series, treatment: pd.Series) -> SplitRates:
    s = success.fillna(False).astype(bool)
    t = treatment.fillna(False).astype(bool)
    n = int(t.sum())
    succ = int((s & t).sum())
    bn = int((~t).sum())
    bsucc = int((s & ~t).sum())
    rate = succ / n if n else float("nan")
    brate = bsucc / bn if bn else float("nan")
    p = proportion_test(succ, n, bsucc, bn) if n and bn else None
    return SplitRates(n, succ, rate, bn, bsucc, brate, p)


def bonferroni(alpha: float, n_tests: int) -> float:
    return alpha / max(n_tests, 1)
