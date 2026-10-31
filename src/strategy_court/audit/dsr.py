"""Deflated Sharpe Ratio (Bailey & López de Prado 2014).

After N trials, the maximum Sharpe ratio is inflated even when every strategy is worthless
(the "False Strategy Theorem"). Its expected value under the null is approximately

SR0 = sqrt(V[SR_n]) * ((1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e)))

with gamma the Euler-Mascheroni constant and V[SR_n] the cross-sectional variance of the
trials' Sharpe ratios. DSR = PSR(SR0): the probability the selected strategy's true Sharpe
beats what pure luck would deliver after N tries.

Correlated trials overstate N. We report DSR at the raw N and at an *effective* N: the number of
independent trials with the same expected maximum as the actual correlated trials (simulated
under their correlation matrix, `effective_trials_for_max`) — a transparent stand-in for the
clustering approach of López de Prado (2019). The decision uses the effective N when the full
search is disclosed. (The eigenvalue participation ratio is kept as a diagnostic; it is not used
for decisions because it counts sign-flipped variants as one trial.)

Which trials count: only the search that produced the claim. The court's own parameter
neighbourhood (`TrialSet.source == "probe"`) is not the claimant's search; with a probe set the
trial count is 1 + `extra_trials` (disclosed count) and the probe set only supplies V[SR]. With
no disclosed count the deflation is not applicable (inconclusive), never silently skipped.
"""

import math
from dataclasses import dataclass

import numpy as np

from strategy_court.audit._common import inconclusive, make_evidence
from strategy_court.audit._stats import EULER_GAMMA, norm_ppf, sharpe_moments
from strategy_court.audit.citations import BAILEY_LDP_2014, BBLZ_2014, LDP_2019
from strategy_court.audit.psr import probabilistic_sharpe_ratio
from strategy_court.audit.trials import TrialSet
from strategy_court.schemas import Evidence, Severity

TOOL = "deflated_sharpe"
CITATIONS = (BAILEY_LDP_2014, BBLZ_2014, LDP_2019)


def expected_max_sharpe(n_trials: float, sr_variance: float) -> float:
    """E[max SR] over n_trials independent zero-skill trials (per-period units)."""
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    z1 = norm_ppf(1 - 1 / n_trials)
    z2 = norm_ppf(1 - 1 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe_ratio(
    sr: float, n_obs: int, skew: float, kurt: float, n_trials: float, sr_variance: float
) -> tuple[float, float]:
    """(DSR, SR0). All Sharpe ratios per period."""
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(sr, n_obs, skew, kurt, sr0), sr0


def effective_trials_for_max(returns: np.ndarray, n_sims: int = 4000, seed: int = 0) -> float:
    """Number N of *independent* trials whose expected maximum equals the expected maximum of
    the actual, correlated trials under the null.

    Draw Gaussian vectors with the trials' correlation matrix, average their maximum, and invert
    the iid expected-maximum formula. Unlike the participation ratio this counts a sign-flipped
    variant (trend vs contrarian, correlation -1) as a genuine extra trial, which it is for a
    best-of-N selection."""
    x = np.asarray(returns, dtype=np.float64)
    x = x[:, x.std(axis=0) > 0]
    m = x.shape[1]
    if m <= 1:
        return float(m)
    eig, vec = np.linalg.eigh(np.corrcoef(x, rowvar=False))
    loadings = vec * np.sqrt(np.clip(eig, 0.0, None))
    z = np.random.default_rng(seed).standard_normal((n_sims, m)) @ loadings.T
    target = float(z.max(axis=1).mean())
    if target <= 0:
        return 1.0
    lo, hi = 1.0, 10.0 * m
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if expected_max_sharpe(mid, 1.0) < target else (lo, mid)
    return math.sqrt(lo * hi)


def effective_trials(returns: np.ndarray) -> float:
    """Participation ratio of the correlation spectrum: 1 for identical trials, N for
    uncorrelated ones. Constant columns are dropped."""
    x = np.asarray(returns, dtype=np.float64)
    x = x[:, x.std(axis=0) > 0]
    if x.shape[1] <= 1:
        return float(x.shape[1])
    eig = np.clip(np.linalg.eigvalsh(np.corrcoef(x, rowvar=False)), 0.0, None)
    return float(eig.sum() ** 2 / (eig**2).sum())


@dataclass(frozen=True)
class DSRResult:
    sr: float
    skew: float
    kurt: float
    n_obs: int
    n_trials: int
    n_effective: float
    sr_variance: float
    sr0: float
    sr0_effective: float
    dsr: float
    dsr_effective: float
    alpha: float
    periods_per_year: int
    use_effective: bool = True

    @property
    def decision_dsr(self) -> float:
        return self.dsr_effective if self.use_effective else self.dsr

    def to_evidence(self, strategy_id: str) -> Evidence:
        ann = math.sqrt(self.periods_per_year)
        d = self.decision_dsr
        passed = math.isfinite(d) and d >= 1 - self.alpha
        sev = (
            Severity.INFO
            if passed
            else Severity.HIGH
            if not math.isfinite(d) or d < 0.5
            else Severity.MEDIUM
        )
        return make_evidence(
            TOOL,
            strategy_id,
            flag=not passed,
            severity=sev,
            statistic_name="dsr_effective" if self.use_effective else "dsr",
            statistic=d,
            p_value=1 - d if math.isfinite(d) else None,
            summary=(
                f"After {self.n_trials} trials luck alone is expected to produce an annual Sharpe "
                f"of {self.sr0 * ann:.2f}; observed {self.sr * ann:.2f}; deflated Sharpe "
                f"{self.dsr:.3f} (with {self.n_effective:.1f} effective trials: "
                f"{self.dsr_effective:.3f})."
            ),
            details={
                "sharpe_annual": self.sr * ann,
                "n_obs": self.n_obs,
                "n_trials": self.n_trials,
                "n_effective_trials": self.n_effective,
                "trial_sharpe_std_annual": math.sqrt(self.sr_variance) * ann,
                "expected_max_sharpe_annual": self.sr0 * ann,
                "expected_max_sharpe_annual_effective": self.sr0_effective * ann,
                "dsr": self.dsr,
                "dsr_effective": self.dsr_effective,
                "decision_uses_effective_trials": self.use_effective,
                "skewness": self.skew,
                "kurtosis": self.kurt,
                "alpha": self.alpha,
            },
            citations=CITATIONS,
        )


def dsr_test(
    trials: TrialSet,
    extra_trials: int = 0,
    periods_per_year: int = 252,
    alpha: float = 0.05,
) -> DSRResult:
    """`extra_trials` adds configurations tried but not in the matrix (e.g. the K replication
    interpretations or a disclosed grid size)."""
    r = trials.selected_returns
    sr, skew, kurt, n = sharpe_moments(r)
    sharpes = trials.sharpes()
    sharpes = sharpes[np.isfinite(sharpes)]
    var = float(np.var(sharpes, ddof=1)) if sharpes.size >= 2 else 0.0
    if trials.source == "probe":
        n_trials = 1 + extra_trials
        n_eff = float(n_trials)
    else:
        n_trials = trials.n_trials + extra_trials
        n_eff = min(float(n_trials), effective_trials_for_max(trials.returns) + extra_trials)
    dsr, sr0 = deflated_sharpe_ratio(sr, n, skew, kurt, n_trials, var)
    dsr_e, sr0_e = deflated_sharpe_ratio(sr, n, skew, kurt, n_eff, var)
    return DSRResult(
        sr, skew, kurt, n, n_trials, n_eff, var, sr0, sr0_e, dsr, dsr_e, alpha, periods_per_year
    )


def disclosed_trials(trials: TrialSet, extra_trials: int) -> int:
    """Number of trials attributable to the claimant's own search."""
    return (1 if trials.source == "probe" else trials.n_trials) + extra_trials


def dsr_evidence(trials: TrialSet, strategy_id: str, extra_trials: int = 0) -> Evidence:
    if disclosed_trials(trials, extra_trials) < 2:
        return inconclusive(
            TOOL,
            strategy_id,
            "search not disclosed (single trial); deflation not applicable",
            CITATIONS,
        )
    if trials.n_bars < 10:
        return inconclusive(TOOL, strategy_id, "fewer than 10 return observations", CITATIONS)
    return dsr_test(trials, extra_trials).to_evidence(strategy_id)
