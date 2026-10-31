"""Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation
(Bailey, Borwein, López de Prado & Zhu 2017).

1. Split the (T x N) trial-return matrix into S equal, contiguous row groups.
2. For every one of the C(S, S/2) ways to pick half the groups as in-sample (IS), the rest as
   out-of-sample (OOS): take n* = the trial with the best IS Sharpe and its OOS relative rank
   w = rank_OOS(n*) / (N + 1); logit lambda = ln(w / (1 - w)).
3. PBO = P(lambda <= 0): how often the IS winner is below the OOS median.

Also reported: the performance-degradation regression of OOS on IS Sharpe of the selected
trial, and the probability that the selected trial loses money OOS.
Sharpe ratios for all combinations come from per-group sufficient statistics (sums and sums of
squares), so the C(16, 8) = 12,870 combinations cost one matrix product.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from strategy_court.audit._common import inconclusive, make_evidence
from strategy_court.audit.citations import BBLZ_2014, BBLZ_2017
from strategy_court.audit.trials import TrialSet
from strategy_court.schemas import Evidence, Severity

TOOL = "pbo"
CITATIONS = (BBLZ_2017, BBLZ_2014)


def _combo_sharpes(
    sums: np.ndarray, sumsq: np.ndarray, sizes: np.ndarray, pick: np.ndarray
) -> np.ndarray:
    n = pick @ sizes
    s = pick @ sums
    q = pick @ sumsq
    mean = s / n[:, None]
    var = (q - n[:, None] * mean**2) / (n[:, None] - 1)
    sd = np.sqrt(np.clip(var, 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, mean / sd, 0.0)


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    logits: np.ndarray
    is_sharpe: np.ndarray  # selected trial, per combination (per period)
    oos_sharpe: np.ndarray
    prob_oos_loss: float
    degradation_slope: float
    degradation_intercept: float
    n_groups: int
    n_trials: int
    periods_per_year: int

    def to_evidence(self, strategy_id: str) -> Evidence:
        sev = (
            Severity.HIGH
            if self.pbo > 0.5
            else Severity.MEDIUM
            if self.pbo > 0.2
            else Severity.INFO
        )
        ann = math.sqrt(self.periods_per_year)
        hist, edges = np.histogram(self.logits[np.isfinite(self.logits)], bins=30)
        return make_evidence(
            TOOL,
            strategy_id,
            flag=self.pbo > 0.2,
            severity=sev,
            statistic_name="pbo",
            statistic=self.pbo,
            summary=(
                f"Probability of backtest overfitting {self.pbo:.0%} across "
                f"{self.logits.size} CSCV splits of {self.n_trials} trials; the in-sample "
                f"winner loses money out-of-sample {self.prob_oos_loss:.0%} of the time."
            ),
            details={
                "pbo": self.pbo,
                "prob_oos_loss": self.prob_oos_loss,
                "degradation_slope": self.degradation_slope,
                "degradation_intercept_annual": self.degradation_intercept * ann,
                "mean_is_sharpe_annual": float(self.is_sharpe.mean()) * ann,
                "mean_oos_sharpe_annual": float(self.oos_sharpe.mean()) * ann,
                "n_splits": int(self.logits.size),
                "n_groups": self.n_groups,
                "n_trials": self.n_trials,
            },
            plot_data={
                "logit_hist_counts": hist,
                "logit_hist_edges": edges,
                "is_vs_oos_sample": np.column_stack([self.is_sharpe, self.oos_sharpe])[:500] * ann,
            },
            citations=CITATIONS,
        )


def pbo_cscv(returns: np.ndarray, n_groups: int = 16, periods_per_year: int = 252) -> PBOResult:
    r = np.asarray(returns, dtype=np.float64)
    t, n_trials = r.shape
    if n_groups % 2 or n_groups < 4:
        raise ValueError("n_groups must be an even number >= 4")
    if n_trials < 2:
        raise ValueError("PBO needs at least two trials")
    usable = (t // n_groups) * n_groups
    r = r[t - usable :]  # drop the oldest rows so groups are equal
    g = r.reshape(n_groups, usable // n_groups, n_trials)
    sums, sumsq = g.sum(axis=1), (g**2).sum(axis=1)
    sizes = np.full(n_groups, usable // n_groups, dtype=np.float64)
    combos = np.array(list(itertools.combinations(range(n_groups), n_groups // 2)))
    pick = np.zeros((len(combos), n_groups))
    np.put_along_axis(pick, combos, 1.0, axis=1)
    sr_is = _combo_sharpes(sums, sumsq, sizes, pick)
    sr_oos = _combo_sharpes(sums, sumsq, sizes, 1.0 - pick)
    best = sr_is.argmax(axis=1)
    rows = np.arange(len(combos))
    ranks = stats.rankdata(sr_oos, axis=1)[rows, best]
    w = ranks / (n_trials + 1)
    logits = np.log(w / (1 - w))
    is_sel, oos_sel = sr_is[rows, best], sr_oos[rows, best]
    slope, intercept = (
        np.polyfit(is_sel, oos_sel, 1) if np.ptp(is_sel) > 0 else (0.0, float(oos_sel.mean()))
    )
    return PBOResult(
        pbo=float((logits <= 0).mean()),
        logits=logits,
        is_sharpe=is_sel,
        oos_sharpe=oos_sel,
        prob_oos_loss=float((oos_sel < 0).mean()),
        degradation_slope=float(slope),
        degradation_intercept=float(intercept),
        n_groups=n_groups,
        n_trials=n_trials,
        periods_per_year=periods_per_year,
    )


def pbo_evidence(trials: TrialSet, strategy_id: str, n_groups: int = 16) -> Evidence:
    if trials.returns.shape[1] < 2:
        return inconclusive(TOOL, strategy_id, "needs at least two trials", CITATIONS)
    if trials.n_bars < 10 * n_groups:
        return inconclusive(
            TOOL, strategy_id, f"fewer than {10 * n_groups} observations", CITATIONS
        )
    return pbo_cscv(trials.returns, n_groups).to_evidence(strategy_id)
