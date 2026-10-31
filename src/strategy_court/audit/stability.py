"""Subperiod stability and structural breaks in strategy returns.

* Subperiods: Sharpe ratio in K equal blocks (default 5) and per calendar year; a robust
  edge is positive in most of them.
* Equality of means across the K blocks: HAC Wald test, W = d' (R V R')^-1 d ~ chi2(K-1), with
  block means estimated jointly and a Newey-West long-run variance.
* Unknown break date (Andrews 1993): sup-Wald test for a single shift in mean return,
  W(tau) = (mean_after - mean_before)^2 / (s2_LR (1/n1 + 1/n2)), maximised over tau in
  [0.15 n, 0.85 n]. Its limit is sup_{pi} BB(pi)^2 / (pi (1 - pi)) for a Brownian bridge BB;
  p-values come from simulating that limit (fixed seed, cached), not from remembered tables.
  s2_LR is estimated under H0 (constant mean) — conservative under the alternative.
Decay after publication / discovery is a documented empirical regularity (McLean & Pontiff 2016),
which is why a break is reported with its date and before/after Sharpe ratios.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from strategy_court.audit._common import inconclusive, make_evidence
from strategy_court.audit._stats import long_run_variance, sharpe_per_period
from strategy_court.audit.citations import ANDREWS_1993, MCLEAN_PONTIFF_2016, NEWEY_WEST_1994
from strategy_court.schemas import Evidence, Severity

TOOL = "subperiod_stability"
CITATIONS = (ANDREWS_1993, MCLEAN_PONTIFF_2016, NEWEY_WEST_1994)


@lru_cache(maxsize=4)
def sup_wald_null(
    trim: float = 0.15, grid: int = 1000, reps: int = 20_000, seed: int = 7
) -> np.ndarray:
    """Sorted draws from the asymptotic null distribution of the sup-Wald statistic (1 break)."""
    rng = np.random.default_rng(seed)
    pi = np.arange(1, grid) / grid
    keep = (pi >= trim) & (pi <= 1 - trim)
    out = np.empty(reps)
    for start in range(0, reps, 2000):
        k = min(2000, reps - start)
        w = np.cumsum(rng.standard_normal((k, grid)), axis=1) / math.sqrt(grid)
        bridge = w[:, :-1] - pi * w[:, -1:]
        out[start : start + k] = (bridge[:, keep] ** 2 / (pi[keep] * (1 - pi[keep]))).max(axis=1)
    return np.sort(out)


def sup_wald_pvalue(stat: float, trim: float = 0.15) -> float:
    null = sup_wald_null(trim)
    return float((1 + null.size - np.searchsorted(null, stat, side="left")) / (1 + null.size))


def sup_wald(r: np.ndarray, trim: float = 0.15) -> tuple[float, int]:
    """(sup-Wald statistic, break index = first bar of the second regime)."""
    n = r.size
    lo, hi = max(1, math.ceil(trim * n)), min(n - 1, math.floor((1 - trim) * n))
    s2 = long_run_variance(r)
    if s2 <= 0 or hi <= lo:
        return float("nan"), -1
    csum = np.cumsum(r)
    tau = np.arange(lo, hi + 1)
    m1 = csum[tau - 1] / tau
    m2 = (csum[-1] - csum[tau - 1]) / (n - tau)
    w = (m2 - m1) ** 2 / (s2 * (1 / tau + 1 / (n - tau)))
    k = int(np.argmax(w))
    return float(w[k]), int(tau[k])


def equal_means_wald(r: np.ndarray, k: int) -> tuple[float, float]:
    """HAC Wald test that K contiguous block means are equal: (statistic, p-value)."""
    blocks = np.array_split(r, k)
    means = np.array([b.mean() for b in blocks])
    sizes = np.array([b.size for b in blocks], dtype=float)
    s2 = long_run_variance(r)
    if s2 <= 0:
        return float("nan"), float("nan")
    d = means[1:] - means[:-1]
    cov = np.diag(s2 / sizes)
    contrast = np.eye(k)[1:] - np.eye(k)[:-1]
    v = contrast @ cov @ contrast.T
    stat = float(d @ np.linalg.solve(v, d))
    return stat, float(stats.chi2.sf(stat, k - 1))


@dataclass(frozen=True)
class StabilityResult:
    block_sharpes: tuple[float, ...]
    yearly_sharpes: dict[str, float]
    frac_blocks_positive: float
    wald_equal_means: float
    wald_p: float
    sup_wald: float
    sup_wald_p: float
    break_date: str | None
    sharpe_before: float
    sharpe_after: float
    rolling_sharpe: pd.Series

    def to_evidence(self, strategy_id: str) -> Evidence:
        broke = self.sup_wald_p < 0.05
        decayed = broke and self.sharpe_after < 0.5 * self.sharpe_before
        if self.frac_blocks_positive < 0.5 or decayed:
            sev = Severity.HIGH
        elif broke or self.wald_p < 0.05 or self.frac_blocks_positive < 0.8:
            sev = Severity.MEDIUM
        else:
            sev = Severity.INFO
        when = f"; break around {self.break_date}" if broke else ""
        roll = self.rolling_sharpe.dropna()
        step = max(1, len(roll) // 300)
        return make_evidence(
            TOOL,
            strategy_id,
            flag=sev >= Severity.MEDIUM,
            severity=sev,
            statistic_name="sup_wald",
            statistic=self.sup_wald,
            p_value=self.sup_wald_p,
            summary=(
                f"Sharpe positive in {self.frac_blocks_positive:.0%} of subperiods; break test "
                f"p = {self.sup_wald_p:.3f}{when} (Sharpe {self.sharpe_before:.2f} before, "
                f"{self.sharpe_after:.2f} after)."
            ),
            details={
                "block_sharpes_annual": list(self.block_sharpes),
                "yearly_sharpes_annual": self.yearly_sharpes,
                "fraction_blocks_positive": self.frac_blocks_positive,
                "wald_equal_means": self.wald_equal_means,
                "wald_equal_means_p": self.wald_p,
                "sup_wald": self.sup_wald,
                "sup_wald_p": self.sup_wald_p,
                "break_date": self.break_date,
                "sharpe_before_break_annual": self.sharpe_before,
                "sharpe_after_break_annual": self.sharpe_after,
            },
            plot_data={
                "rolling_dates": [str(d.date()) for d in roll.index[::step]],
                "rolling_sharpe": roll.to_numpy()[::step],
            },
            citations=CITATIONS,
        )


def stability_test(
    returns: pd.Series, n_blocks: int = 5, window: int = 252, periods_per_year: int = 252
) -> StabilityResult:
    r = returns.dropna()
    x = r.to_numpy(dtype=np.float64)
    ann = math.sqrt(periods_per_year)
    blocks = tuple(sharpe_per_period(b) * ann for b in np.array_split(x, n_blocks))
    yearly = {}
    if isinstance(r.index, pd.DatetimeIndex):
        for year, grp in r.groupby(r.index.year):
            if len(grp) >= periods_per_year // 4:
                yearly[str(year)] = sharpe_per_period(grp.to_numpy()) * ann
    w_stat, w_p = equal_means_wald(x, n_blocks)
    sw, tau = sup_wald(x)
    rolling = (r.rolling(window).mean() / r.rolling(window).std()) * ann
    return StabilityResult(
        block_sharpes=blocks,
        yearly_sharpes=yearly,
        frac_blocks_positive=float(np.mean([b > 0 for b in blocks])),
        wald_equal_means=w_stat,
        wald_p=w_p,
        sup_wald=sw,
        sup_wald_p=sup_wald_pvalue(sw) if math.isfinite(sw) else float("nan"),
        break_date=str(pd.Timestamp(r.index[tau]).date())
        if tau > 0 and isinstance(r.index, pd.DatetimeIndex)
        else (str(tau) if tau > 0 else None),
        sharpe_before=sharpe_per_period(x[:tau]) * ann if tau > 0 else float("nan"),
        sharpe_after=sharpe_per_period(x[tau:]) * ann if tau > 0 else float("nan"),
        rolling_sharpe=rolling,
    )


def stability_evidence(returns: pd.Series, strategy_id: str, **kw: int) -> Evidence:
    if returns.dropna().size < 250:
        return inconclusive(TOOL, strategy_id, "fewer than 250 observations", CITATIONS)
    return stability_test(returns, **kw).to_evidence(strategy_id)
