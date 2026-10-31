"""Data-snooping tests over the whole trial set.

H0: no trial outperforms the benchmark (max_k E[d_k] <= 0, d_k = trial k minus benchmark).

* White's (2000) Reality Check: V = sqrt(n) max_k mean(d_k), null distribution from the
  stationary bootstrap of max_k sqrt(n)(mean*(d_k) - mean(d_k)) (least-favourable centring).
* Hansen's (2005) SPA test: studentised statistic T = max(0, max_k sqrt(n) mean(d_k)/w_k) and
  data-dependent recentring that stops clearly inferior trials from inflating the p-value.
  Reported with the three estimators of the null mean: lower (mu = min(dbar, 0)), consistent
  (mu = dbar if sqrt(n) dbar / w <= -sqrt(2 ln ln n), else 0) and upper (mu = 0).
  p_l <= p_c <= p_u; the consistent version is the recommended one.

The bootstrap resamples whole rows (dates) so the cross-correlation of trials is preserved.
w_k is the bootstrap standard deviation of sqrt(n) mean*(d_k).
"""

import math
from dataclasses import dataclass

import numpy as np

from strategy_court.audit._common import inconclusive, make_evidence, severity_from_p
from strategy_court.audit.bootstrap import optimal_block_length, stationary_bootstrap_indices
from strategy_court.audit.citations import HANSEN_2005, STW_1999, WHITE_2000
from strategy_court.audit.trials import TrialSet
from strategy_court.schemas import Evidence

TOOL = "data_snooping"
CITATIONS = (WHITE_2000, HANSEN_2005, STW_1999)


@dataclass(frozen=True)
class SnoopingResult:
    rc_p: float
    spa_p_lower: float
    spa_p_consistent: float
    spa_p_upper: float
    best_trial: int
    best_mean: float
    n_trials: int
    block_length: float
    n_boot: int

    def to_evidence(self, strategy_id: str, labels: tuple[str, ...] = ()) -> Evidence:
        p = self.spa_p_consistent
        best = labels[self.best_trial] if labels else str(self.best_trial)
        return make_evidence(
            TOOL,
            strategy_id,
            flag=p > 0.05,
            severity=severity_from_p(p),
            statistic_name="spa_p_consistent",
            statistic=p,
            p_value=p,
            summary=(
                f"Across {self.n_trials} trials, can any beat the benchmark beyond data-snooping "
                f"luck? Hansen SPA p = {p:.3f}, White Reality Check p = {self.rc_p:.3f}."
            ),
            details={
                "white_rc_p": self.rc_p,
                "spa_p_lower": self.spa_p_lower,
                "spa_p_consistent": self.spa_p_consistent,
                "spa_p_upper": self.spa_p_upper,
                "best_trial": best,
                "best_mean_excess_per_period": self.best_mean,
                "n_trials": self.n_trials,
                "mean_block_length": self.block_length,
                "n_boot": self.n_boot,
            },
            citations=CITATIONS,
        )


def reality_check(
    trial_returns: np.ndarray,
    benchmark_returns: np.ndarray | None = None,
    n_boot: int = 1000,
    block_length: float | None = None,
    seed: int = 0,
    chunk: int = 25,
) -> SnoopingResult:
    r = np.asarray(trial_returns, dtype=np.float64)
    n, m = r.shape
    bench = np.zeros(n) if benchmark_returns is None else np.asarray(benchmark_returns)[:n]
    d = r - bench[:, None]
    dbar = d.mean(axis=0)
    best = int(np.argmax(dbar))
    b = optimal_block_length(d[:, best]) if block_length is None else float(block_length)
    rng = np.random.default_rng(seed)
    boot_means = np.empty((n_boot, m))
    for start in range(0, n_boot, chunk):
        k = min(chunk, n_boot - start)
        idx = stationary_bootstrap_indices(n, k, b, rng)
        boot_means[start : start + k] = d[idx].mean(axis=1)
    root_n = math.sqrt(n)
    centred = root_n * (boot_means - dbar)
    omega = centred.std(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, np.inf)

    v = root_n * dbar.max()
    rc_p = float((1 + (centred.max(axis=1) >= v).sum()) / (1 + n_boot))

    t_obs = max(0.0, float((root_n * dbar / omega).max()))
    thresh = -math.sqrt(2 * math.log(math.log(n))) if n > 15 else -np.inf
    mus = {
        "lower": np.minimum(dbar, 0.0),
        "consistent": np.where(root_n * dbar / omega <= thresh, dbar, 0.0),
        "upper": np.zeros(m),
    }
    pvals = {}
    for name, mu in mus.items():
        t_star = np.maximum(0.0, ((centred + root_n * mu) / omega).max(axis=1))
        pvals[name] = float((1 + (t_star >= t_obs).sum()) / (1 + n_boot))
    return SnoopingResult(
        rc_p,
        pvals["lower"],
        pvals["consistent"],
        pvals["upper"],
        best,
        float(dbar[best]),
        m,
        b,
        n_boot,
    )


def snooping_evidence(
    trials: TrialSet, strategy_id: str, benchmark_returns: np.ndarray | None = None, **kw: object
) -> Evidence:
    if trials.returns.shape[1] < 2:
        return inconclusive(TOOL, strategy_id, "needs at least two trials", CITATIONS)
    if trials.n_bars < 50:
        return inconclusive(TOOL, strategy_id, "fewer than 50 observations", CITATIONS)
    res = reality_check(trials.returns, benchmark_returns, **kw)  # type: ignore[arg-type]
    return res.to_evidence(strategy_id, trials.labels)
