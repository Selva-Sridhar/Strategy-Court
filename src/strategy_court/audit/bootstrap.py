"""Dependence-robust inference for the Sharpe ratio.

* Stationary bootstrap (Politis & Romano 1994): resample blocks of geometric length with mean b,
  which keeps serial dependence (volatility clustering, autocorrelated P&L) in the resamples.
* Automatic mean block length b (Politis & White 2004, with the correction of Patton, Politis
  & White 2009): flat-top lag-window estimates of g(0) and G = sum |k| R(k), giving
  b_opt = (2 G^2 / D)^(1/3) n^(1/3) with D = 2 g(0)^2.
* Confidence intervals: percentile and basic (reverse-percentile) bootstrap intervals, plus
  the asymptotic iid interval SE = sqrt((1 - g3 SR + (g4 - 1)/4 SR^2) / n) (Lo 2002; Mertens).
* Sharpe-difference test vs a benchmark (Ledoit & Wolf 2008): delta method with a
  Newey-West HAC covariance of (r1, r2, r1^2, r2^2). LW recommend prewhitened QS kernels; we
  use the Bartlett kernel with the Newey-West (1994) lag, stated as a simplification.
"""

import math
from dataclasses import dataclass

import numpy as np

from strategy_court.audit._common import inconclusive, make_evidence
from strategy_court.audit._stats import (
    as_1d,
    long_run_covariance,
    norm_cdf,
    norm_ppf,
    sharpe_moments,
    sharpe_variance_factor,
)
from strategy_court.audit.citations import (
    LEDOIT_WOLF_2008,
    LO_2002,
    NEWEY_WEST_1994,
    PATTON_POLITIS_WHITE_2009,
    POLITIS_ROMANO_1994,
    POLITIS_WHITE_2004,
)
from strategy_court.schemas import Evidence, Severity

TOOL = "bootstrap_sharpe"
CITATIONS = (
    POLITIS_ROMANO_1994,
    POLITIS_WHITE_2004,
    PATTON_POLITIS_WHITE_2009,
    LO_2002,
    LEDOIT_WOLF_2008,
    NEWEY_WEST_1994,
)


def _flat_top(t: np.ndarray) -> np.ndarray:
    a = np.abs(t)
    return np.where(a <= 0.5, 1.0, np.where(a <= 1.0, 2.0 * (1.0 - a), 0.0))


def optimal_block_length(x: object) -> float:
    """Politis-White (2004) / Patton-Politis-White (2009) mean block length for the
    stationary bootstrap. Returns 1.0 for (near-)uncorrelated data."""
    r = as_1d(x)
    n = r.size
    if n < 10:
        return 1.0
    e = r - r.mean()
    var = float(e @ e) / n
    if var <= 0:
        return 1.0
    k_n = max(5, math.ceil(math.sqrt(math.log10(n))))
    m_max = math.ceil(math.sqrt(n)) + k_n
    b_max = math.ceil(min(3.0 * math.sqrt(n), n / 3.0))
    lags = np.arange(0, m_max + k_n + 1)
    acov = np.array([float(e[k:] @ e[: n - k]) / n for k in lags])
    rho = acov / var
    c = 2.0 * math.sqrt(math.log10(n) / n)
    insignificant = np.abs(rho[1:]) < c
    m_hat = m_max
    for m in range(1, m_max + 1):
        if insignificant[m - 1 : m - 1 + k_n].all():
            m_hat = m - 1
            break
    big_m = min(2 * max(m_hat, 1), m_max)
    k = np.arange(-big_m, big_m + 1)
    w = _flat_top(k / big_m)
    r_k = acov[np.abs(k)]
    g0 = float((w * r_k).sum())
    g = float((w * np.abs(k) * r_k).sum())
    d = 2.0 * g0**2
    if d <= 0 or g == 0:
        return 1.0
    b = (2.0 * g**2 / d) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(min(max(b, 1.0), b_max))


def stationary_bootstrap_indices(
    n: int, n_boot: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """(n_boot, n) indices: each step starts a new block with prob 1/b, else continues
    (wrapping circularly), as in Politis & Romano (1994)."""
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    starts = rng.random((n_boot, n)) < p
    fresh = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(starts[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def _sharpe_rows(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, x.mean(axis=1) / sd, np.nan)


@dataclass(frozen=True)
class SharpeDifference:
    diff_annual: float
    se_annual: float
    z: float
    p_value: float  # two-sided, H0: equal Sharpe ratios


def sharpe_difference_hac(r1: object, r2: object, periods_per_year: int = 252) -> SharpeDifference:
    """Ledoit & Wolf (2008) delta-method HAC test of SR1 = SR2 (paired observations)."""
    a = np.asarray(r1, dtype=np.float64)
    b = np.asarray(r2, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = a.size
    mu1, mu2 = a.mean(), b.mean()
    g1, g2 = (a**2).mean(), (b**2).mean()
    v1, v2 = g1 - mu1**2, g2 - mu2**2
    if n < 10 or v1 <= 0 or v2 <= 0:
        nan = float("nan")
        return SharpeDifference(nan, nan, nan, nan)
    diff = mu1 / math.sqrt(v1) - mu2 / math.sqrt(v2)
    grad = np.array([g1 / v1**1.5, -g2 / v2**1.5, -mu1 / (2 * v1**1.5), mu2 / (2 * v2**1.5)])
    psi = long_run_covariance(np.column_stack([a, b, a**2, b**2]))
    se = math.sqrt(max(float(grad @ psi @ grad), 0.0) / n)
    ann = math.sqrt(periods_per_year)
    z = diff / se if se > 0 else float("nan")
    p = 2 * (1 - norm_cdf(abs(z))) if math.isfinite(z) else float("nan")
    return SharpeDifference(diff * ann, se * ann, z, p)


@dataclass(frozen=True)
class BootstrapResult:
    sharpe_annual: float
    ci_percentile: tuple[float, float]
    ci_basic: tuple[float, float]
    ci_asymptotic: tuple[float, float]
    se_bootstrap_annual: float
    p_value_le_zero: float
    block_length: float
    n_boot: int
    n_obs: int
    alpha: float
    benchmark: SharpeDifference | None = None

    def to_evidence(self, strategy_id: str) -> Evidence:
        lo, hi = self.ci_percentile
        excludes_zero = lo > 0
        sev = Severity.INFO if excludes_zero else Severity.HIGH if hi <= 0.5 else Severity.MEDIUM
        conf = round(100 * (1 - self.alpha))
        summary = (
            f"Annual Sharpe {self.sharpe_annual:.2f}, {conf}% stationary-bootstrap CI "
            f"[{lo:.2f}, {hi:.2f}] (mean block {self.block_length:.1f} days); "
            f"P(Sharpe <= 0) = {self.p_value_le_zero:.3f}."
        )
        details: dict[str, object] = {
            "sharpe_annual": self.sharpe_annual,
            "ci_percentile": list(self.ci_percentile),
            "ci_basic": list(self.ci_basic),
            "ci_asymptotic_iid": list(self.ci_asymptotic),
            "se_bootstrap_annual": self.se_bootstrap_annual,
            "mean_block_length": self.block_length,
            "n_boot": self.n_boot,
            "n_obs": self.n_obs,
            "alpha": self.alpha,
        }
        if self.benchmark is not None:
            details |= {
                "benchmark_sharpe_diff_annual": self.benchmark.diff_annual,
                "benchmark_diff_se_annual": self.benchmark.se_annual,
                "benchmark_diff_p_value": self.benchmark.p_value,
            }
        return make_evidence(
            TOOL,
            strategy_id,
            flag=not excludes_zero,
            severity=sev,
            statistic_name="sharpe_ci_lower",
            statistic=lo,
            p_value=self.p_value_le_zero,
            summary=summary,
            details=details,
            citations=CITATIONS,
        )


def bootstrap_sharpe(
    returns: object,
    n_boot: int = 2000,
    alpha: float = 0.05,
    block_length: float | None = None,
    periods_per_year: int = 252,
    benchmark_returns: object | None = None,
    seed: int = 0,
    chunk: int = 250,
) -> BootstrapResult:
    r = as_1d(returns)
    n = r.size
    rng = np.random.default_rng(seed)
    b = optimal_block_length(r) if block_length is None else float(block_length)
    ann = math.sqrt(periods_per_year)
    sr, skew, kurt, _ = sharpe_moments(r)
    boots = []
    for start in range(0, n_boot, chunk):
        idx = stationary_bootstrap_indices(n, min(chunk, n_boot - start), b, rng)
        boots.append(_sharpe_rows(r[idx]))
    sims = np.concatenate(boots)
    sims = sims[np.isfinite(sims)]
    q_lo, q_hi = np.quantile(sims, [alpha / 2, 1 - alpha / 2])
    se_iid = math.sqrt(max(sharpe_variance_factor(sr, skew, kurt), 0.0) / n)
    z = norm_ppf(1 - alpha / 2)
    bench = None
    if benchmark_returns is not None:
        bench = sharpe_difference_hac(r, as_1d(benchmark_returns)[:n], periods_per_year)
    return BootstrapResult(
        sharpe_annual=sr * ann,
        ci_percentile=(float(q_lo) * ann, float(q_hi) * ann),
        ci_basic=((2 * sr - float(q_hi)) * ann, (2 * sr - float(q_lo)) * ann),
        ci_asymptotic=((sr - z * se_iid) * ann, (sr + z * se_iid) * ann),
        se_bootstrap_annual=float(sims.std(ddof=1)) * ann,
        p_value_le_zero=float((1 + (sims <= 0).sum()) / (1 + sims.size)),
        block_length=b,
        n_boot=int(sims.size),
        n_obs=n,
        alpha=alpha,
        benchmark=bench,
    )


def bootstrap_evidence(returns: object, strategy_id: str, **kw: object) -> Evidence:
    r = as_1d(returns)
    if r.size < 30:
        return inconclusive(TOOL, strategy_id, "fewer than 30 return observations", CITATIONS)
    return bootstrap_sharpe(r, **kw).to_evidence(strategy_id)  # type: ignore[arg-type]
