"""Probabilistic Sharpe Ratio and Minimum Track Record Length (Bailey & López de Prado 2012).

PSR(SR*) = Phi( (SR - SR*) sqrt(n - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2) )

is the probability that the true per-period Sharpe exceeds SR*, accounting for sample length,
skewness g3 and raw kurtosis g4 of returns. MinTRL is the number of observations needed for
PSR(SR*) to reach 1 - alpha if the observed moments persist:

MinTRL = 1 + (1 - g3 SR + (g4 - 1)/4 SR^2) (z_{1-alpha} / (SR - SR*))^2.

Assumptions: returns are iid (serial correlation is handled by the bootstrap tool); the
variance factor is the non-normal iid asymptotic variance of the SR estimator (Mertens 2002).
Known limitation (verified in tests): the correction is asymptotic. It controls size well for
realistic skew/kurtosis, but with extreme skew (|skew| ~ 6, kurtosis > 100) and n ~ 250 it still
over-rejects (~8% at a 5% level), though far less than the normal-assumption formula.
"""

import math
from dataclasses import dataclass

import numpy as np

from strategy_court.audit._common import inconclusive, make_evidence
from strategy_court.audit._stats import (
    as_1d,
    norm_cdf,
    norm_ppf,
    sharpe_moments,
    sharpe_variance_factor,
)
from strategy_court.audit.citations import BAILEY_LDP_2012, LO_2002
from strategy_court.schemas import Evidence, Severity

TOOL = "psr"
CITATIONS = (BAILEY_LDP_2012, LO_2002)


def probabilistic_sharpe_ratio(
    sr: float, n: int, skew: float, kurt: float, sr_benchmark: float = 0.0
) -> float:
    """All Sharpe ratios per period. NaN if the variance factor is non-positive."""
    v = sharpe_variance_factor(sr, skew, kurt)
    if n < 2 or not math.isfinite(v) or v <= 0:
        return float("nan")
    return norm_cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(v))


def min_track_record_length(
    sr: float, skew: float, kurt: float, sr_benchmark: float = 0.0, alpha: float = 0.05
) -> float:
    """Observations needed for PSR(sr_benchmark) >= 1 - alpha. inf if SR <= benchmark."""
    if sr <= sr_benchmark:
        return float("inf")
    v = sharpe_variance_factor(sr, skew, kurt)
    if v <= 0:
        return float("nan")
    return 1.0 + v * (norm_ppf(1 - alpha) / (sr - sr_benchmark)) ** 2


@dataclass(frozen=True)
class PSRResult:
    sr: float
    skew: float
    kurt: float
    n: int
    sr_benchmark: float
    psr: float
    psr_normal: float  # same test assuming normal returns, to show the moment correction
    min_trl: float
    alpha: float
    periods_per_year: int

    @property
    def sr_annual(self) -> float:
        return self.sr * math.sqrt(self.periods_per_year)

    @property
    def min_trl_years(self) -> float:
        return self.min_trl / self.periods_per_year

    @property
    def years(self) -> float:
        return self.n / self.periods_per_year

    def to_evidence(self, strategy_id: str) -> Evidence:
        significant = math.isfinite(self.psr) and self.psr >= 1 - self.alpha
        if not math.isfinite(self.psr):
            sev = Severity.MEDIUM
        elif self.psr < 0.5:
            sev = Severity.HIGH
        elif not significant:
            sev = Severity.MEDIUM
        else:
            sev = Severity.INFO
        bench = self.sr_benchmark * math.sqrt(self.periods_per_year)
        trl = (
            f"needs {self.min_trl_years:.1f} years, has {self.years:.1f}"
            if math.isfinite(self.min_trl)
            else "no track-record length suffices (SR <= benchmark)"
        )
        return make_evidence(
            TOOL,
            strategy_id,
            flag=not significant,
            severity=sev,
            statistic_name="psr",
            statistic=self.psr,
            p_value=1 - self.psr if math.isfinite(self.psr) else None,
            summary=(
                f"Probability the true annual Sharpe exceeds {bench:.2f}: {self.psr:.3f} "
                f"(observed {self.sr_annual:.2f}); track record {trl}."
            ),
            details={
                "sharpe_annual": self.sr_annual,
                "sharpe_per_period": self.sr,
                "skewness": self.skew,
                "kurtosis": self.kurt,
                "n_obs": self.n,
                "years": self.years,
                "benchmark_sharpe_annual": bench,
                "psr": self.psr,
                "psr_if_normal": self.psr_normal,
                "min_track_record_obs": self.min_trl,
                "min_track_record_years": self.min_trl_years,
                "alpha": self.alpha,
            },
            citations=CITATIONS,
        )


def psr_test(
    returns: object,
    benchmark_sharpe_annual: float = 0.0,
    periods_per_year: int = 252,
    alpha: float = 0.05,
) -> PSRResult:
    r = as_1d(returns)
    sr, skew, kurt, n = sharpe_moments(r)
    sr_b = benchmark_sharpe_annual / math.sqrt(periods_per_year)
    if not np.isfinite([sr, skew, kurt]).all():
        nan = float("nan")
        return PSRResult(sr, skew, kurt, n, sr_b, nan, nan, nan, alpha, periods_per_year)
    return PSRResult(
        sr=sr,
        skew=skew,
        kurt=kurt,
        n=n,
        sr_benchmark=sr_b,
        psr=probabilistic_sharpe_ratio(sr, n, skew, kurt, sr_b),
        psr_normal=probabilistic_sharpe_ratio(sr, n, 0.0, 3.0, sr_b),
        min_trl=min_track_record_length(sr, skew, kurt, sr_b, alpha),
        alpha=alpha,
        periods_per_year=periods_per_year,
    )


def psr_evidence(
    returns: object, strategy_id: str, benchmark_sharpe_annual: float = 0.0, **kw: int | float
) -> Evidence:
    r = as_1d(returns)
    if r.size < 10:
        return inconclusive(TOOL, strategy_id, "fewer than 10 return observations", CITATIONS)
    return psr_test(r, benchmark_sharpe_annual, **kw).to_evidence(strategy_id)  # type: ignore[arg-type]
