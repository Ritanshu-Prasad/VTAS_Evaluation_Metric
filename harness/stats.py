"""Bootstrap confidence intervals and paired bootstrap.

Rows of ``data`` are the resampling unit (images). A statistic maps the resampled
array to one float, e.g. :func:`mean` for per-image scores or :func:`ratio_of_sums`
for rates such as CHAIRi (column 0 = numerator, column 1 = denominator).

Defaults (1,000 resamples, 95% percentile interval, seed 0) match harness/gates.yaml.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

Statistic = Callable[[np.ndarray], float]


def mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def ratio_of_sums(x: np.ndarray) -> float:
    """sum(column 0) / sum(column 1); 0.0 when the denominator is 0."""
    den = float(np.sum(x[:, 1]))
    return float(np.sum(x[:, 0])) / den if den else 0.0


@dataclass(frozen=True)
class CI:
    point: float
    low: float
    high: float
    level: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PairedResult:
    a: float
    b: float
    diff: CI           # statistic(a) - statistic(b) with its bootstrap CI
    p_value: float     # two-sided, H0: no difference

    def as_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "diff": self.diff.as_dict(), "p_value": self.p_value}


def _as_rows(data) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 0 or arr.shape[0] == 0:
        raise ValueError("need at least one row to bootstrap")
    return arr


def _indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


def _percentile(samples: np.ndarray, ci: float) -> tuple[float, float]:
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def bootstrap_ci(data, statistic: Statistic = mean, n_resamples: int = 1000,
                 ci: float = 0.95, seed: int = 0) -> CI:
    """Percentile bootstrap CI of ``statistic`` over the rows of ``data``."""
    arr = _as_rows(data)
    idx = _indices(arr.shape[0], n_resamples, seed)
    samples = np.array([statistic(arr[i]) for i in idx])
    lo, hi = _percentile(samples, ci)
    return CI(statistic(arr), lo, hi, ci)


def paired_bootstrap(data_a, data_b, statistic: Statistic = mean, n_resamples: int = 1000,
                     ci: float = 0.95, seed: int = 0) -> PairedResult:
    """Paired bootstrap of statistic(a) - statistic(b).

    Row i of ``data_a`` and ``data_b`` must be the same image; each resample draws the
    same rows for both systems. The two-sided p-value is
    2 * min(P*(diff <= 0), P*(diff >= 0)) with add-one smoothing, capped at 1.
    """
    a, b = _as_rows(data_a), _as_rows(data_b)
    if a.shape[0] != b.shape[0]:
        raise ValueError("paired data must have the same number of rows (images)")
    idx = _indices(a.shape[0], n_resamples, seed)
    diffs = np.array([statistic(a[i]) - statistic(b[i]) for i in idx])
    sa, sb = statistic(a), statistic(b)
    lo, hi = _percentile(diffs, ci)
    le = (np.sum(diffs <= 0) + 1) / (n_resamples + 1)
    ge = (np.sum(diffs >= 0) + 1) / (n_resamples + 1)
    p = float(min(1.0, 2.0 * min(le, ge)))
    return PairedResult(sa, sb, CI(sa - sb, lo, hi, ci), p)
