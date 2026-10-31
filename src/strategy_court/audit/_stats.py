"""Statistical primitives shared by the inference tools.

Conventions: Sharpe ratios here are *per period* (not annualised) unless the name says
otherwise; kurtosis is raw (normal = 3); moments use the plain (bias=True) estimators, the
form used in the PSR/DSR papers.
"""

import math

import numpy as np
from scipy import stats

from strategy_court.backtest.metrics import kurtosis_raw, sharpe_ratio, skewness

EULER_GAMMA = 0.5772156649015329


def as_1d(x: object) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def sharpe_per_period(r: np.ndarray) -> float:
    return sharpe_ratio(r, 0.0, 1)


def sharpe_moments(r: np.ndarray) -> tuple[float, float, float, int]:
    """(per-period SR, skewness, raw kurtosis, n)."""
    return sharpe_per_period(r), skewness(r), kurtosis_raw(r), int(r.size)


def sharpe_variance_factor(sr: float, skew: float, kurt: float) -> float:
    """Asymptotic variance of the SR estimator times n for iid non-normal returns:
    1 - skew*SR + (kurt - 1)/4 * SR^2 (Mertens 2002, used by Bailey & López de Prado 2012).
    Reduces to 1 + SR^2/2 under normality (Lo 2002)."""
    return 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr


def norm_cdf(x: float) -> float:
    return float(stats.norm.cdf(x))


def norm_ppf(p: float) -> float:
    return float(stats.norm.ppf(p))


def newey_west_lag(n: int) -> int:
    """Automatic Bartlett-kernel truncation lag floor(4 (n/100)^(2/9)) (Newey & West 1994)."""
    return max(0, int(4.0 * math.pow(n / 100.0, 2.0 / 9.0)))  # int() == floor for x >= 0


def long_run_covariance(y: np.ndarray, lags: int | None = None) -> np.ndarray:
    """Newey-West (Bartlett) HAC long-run covariance of the columns of y (demeaned)."""
    y2 = np.asarray(y, dtype=np.float64)
    if y2.ndim == 1:
        y2 = y2[:, None]
    n = y2.shape[0]
    e = y2 - y2.mean(axis=0)
    lag = newey_west_lag(n) if lags is None else lags
    s = e.T @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        g = e[k:].T @ e[:-k] / n
        s += (1.0 - k / (lag + 1.0)) * (g + g.T)
    return np.asarray(s)


def long_run_variance(x: np.ndarray, lags: int | None = None) -> float:
    return float(long_run_covariance(x, lags)[0, 0])
