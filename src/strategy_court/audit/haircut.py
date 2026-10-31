"""Haircut Sharpe ratio (Harvey & Liu 2015; Harvey, Liu & Zhu 2016).

The selected strategy's Sharpe is turned into a t-statistic, t = SR_annual * sqrt(years), whose
p-value is adjusted for the number of strategies tried, then mapped back to the Sharpe ratio
that the *adjusted* p-value would correspond to (the haircut Sharpe). Adjustments:

* Bonferroni (FWER): p * M
* Holm (FWER, step-down; Holm 1979)
* BHY (FDR under arbitrary dependence; Benjamini & Yekutieli 2001): BH scaled by c(M) = sum 1/i

Where the full trial set is available we use the *actual* p-values of every trial (Harvey &
Liu instead model the unseen tests). If only the count M is known, the other trials are
treated as null (p = 1), which makes Holm and BHY reduce to conservative closed forms.

The annual Sharpe is computed with Lo's (2002) serial-correlation correction:
SR_q = eta(q) * SR, eta(q) = q / sqrt(q + 2 sum_{k=1}^{q-1} (q - k) rho_k),
with sample autocorrelations rho_k up to `max_lag` (higher lags set to zero).
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from strategy_court.audit._common import clean, inconclusive, make_evidence
from strategy_court.audit._stats import as_1d, norm_cdf, norm_ppf, sharpe_per_period
from strategy_court.audit.citations import (
    BENJAMINI_YEKUTIELI_2001,
    HARVEY_LIU_2015,
    HARVEY_LIU_ZHU_2016,
    HOLM_1979,
    LO_2002,
)
from strategy_court.audit.dsr import disclosed_trials, effective_trials_for_max
from strategy_court.audit.trials import TrialSet
from strategy_court.schemas import Evidence, Severity

TOOL = "haircut_sharpe"
CITATIONS = (HARVEY_LIU_2015, HARVEY_LIU_ZHU_2016, LO_2002, HOLM_1979, BENJAMINI_YEKUTIELI_2001)
Method = Literal["bonferroni", "holm", "bhy"]
METHODS: tuple[Method, ...] = ("bonferroni", "holm", "bhy")


def lo_annualisation_factor(returns: object, q: int = 252, max_lag: int = 10) -> float:
    """eta(q); equals sqrt(q) for serially uncorrelated returns."""
    r = as_1d(returns)
    r = r - r.mean()
    denom = float(r @ r)
    rho = np.zeros(q)
    if denom > 0:
        for k in range(1, min(max_lag, q - 1, r.size - 1) + 1):
            rho[k] = float(r[k:] @ r[:-k]) / denom
    lags = np.arange(1, q)
    inner = q + 2.0 * float(((q - lags) * rho[1:]).sum())
    return q / math.sqrt(inner) if inner > 0 else float("nan")


def adjust_pvalues(pvals: object, method: Method) -> np.ndarray:
    """Standard adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    ps = p[order]
    if method == "bonferroni":
        adj = np.minimum(1.0, ps * m)
    elif method == "holm":
        adj = np.minimum(1.0, np.maximum.accumulate(ps * (m - np.arange(m))))
    else:
        c = float((1.0 / np.arange(1, m + 1)).sum())
        raw = ps * m * c / np.arange(1, m + 1)
        adj = np.minimum(1.0, np.minimum.accumulate(raw[::-1])[::-1])
    out = np.empty(m)
    out[order] = adj
    return out


def _p_from_t(t: float, one_sided: bool) -> float:
    return 1 - norm_cdf(t) if one_sided else 2 * (1 - norm_cdf(abs(t)))


def _t_from_p(p: float, one_sided: bool) -> float:
    p = min(max(p, 1e-300), 1.0)
    return norm_ppf(1 - p) if one_sided else norm_ppf(1 - p / 2)


@dataclass(frozen=True)
class HaircutResult:
    sharpe_annual: float
    years: float
    t_stat: float
    p_value: float
    n_trials: int
    one_sided: bool
    adjusted_p: dict[str, float]
    haircut_sharpe: dict[str, float]
    used_actual_trial_pvalues: bool

    def haircut(self, method: Method) -> float:
        if self.sharpe_annual <= 0:
            return float("nan")
        return 1.0 - self.haircut_sharpe[method] / self.sharpe_annual

    def to_evidence(self, strategy_id: str, decision_method: Method = "holm") -> Evidence:
        hsr = self.haircut_sharpe[decision_method]
        p_adj = self.adjusted_p[decision_method]
        sev = Severity.INFO if p_adj <= 0.05 else Severity.HIGH if hsr <= 0 else Severity.MEDIUM
        return make_evidence(
            TOOL,
            strategy_id,
            flag=p_adj > 0.05,
            severity=sev,
            statistic_name=f"haircut_sharpe_{decision_method}",
            statistic=hsr,
            p_value=p_adj,
            summary=(
                f"Sharpe {self.sharpe_annual:.2f} after {self.n_trials} trials: haircut Sharpe "
                f"{hsr:.2f} ({decision_method}), adjusted p = {p_adj:.3f}."
            ),
            details={
                "sharpe_annual_lo_adjusted": self.sharpe_annual,
                "years": self.years,
                "t_stat": self.t_stat,
                "p_value_single_test": self.p_value,
                "n_trials": self.n_trials,
                "one_sided": self.one_sided,
                "used_actual_trial_pvalues": self.used_actual_trial_pvalues,
                **{f"adjusted_p_{m}": v for m, v in self.adjusted_p.items()},
                **{f"haircut_sharpe_{m}": v for m, v in self.haircut_sharpe.items()},
                **{f"haircut_pct_{m}": self.haircut(m) for m in METHODS},
            },
            citations=CITATIONS,
        )


def _annual_sharpe(r: np.ndarray, periods_per_year: int, max_lag: int) -> float:
    return sharpe_per_period(r) * lo_annualisation_factor(r, periods_per_year, max_lag)


def haircut_sharpe(
    returns: object,
    n_trials: int = 1,
    other_trial_returns: np.ndarray | None = None,
    periods_per_year: int = 252,
    one_sided: bool = True,
    max_lag: int = 10,
) -> HaircutResult:
    """`n_trials` counts every trial including the selected one; trials beyond the selected and
    `other_trial_returns` are treated as null (p = 1)."""
    r = as_1d(returns)
    years = r.size / periods_per_year
    sr = _annual_sharpe(r, periods_per_year, max_lag)
    t = sr * math.sqrt(years)
    p = _p_from_t(t, one_sided)
    others = np.empty(0)
    if other_trial_returns is not None and other_trial_returns.size:
        others = np.array(
            [
                _p_from_t(
                    _annual_sharpe(col, periods_per_year, max_lag) * math.sqrt(years), one_sided
                )
                for col in np.asarray(other_trial_returns, dtype=np.float64).T
            ]
        )
    m = max(n_trials, 1 + others.size)
    pvals = np.concatenate([[p], others, np.ones(m - 1 - others.size)])
    adjusted: dict[str, float] = {k: float(adjust_pvalues(pvals, k)[0]) for k in METHODS}
    sqrt_y = math.sqrt(years)
    hsr: dict[str, float] = {
        k: max(0.0, _t_from_p(v, one_sided)) / sqrt_y if v < 1 else 0.0 for k, v in adjusted.items()
    }
    if sr <= 0:
        hsr = {k: sr for k in METHODS}
    return HaircutResult(sr, years, t, p, m, one_sided, adjusted, hsr, others.size > 0)


def haircut_evidence(trials: TrialSet, strategy_id: str, extra_trials: int = 0) -> Evidence:
    """Decision on the *effective* number of independent trials (correlated variants of one rule
    are not independent tests); the raw-count adjustments on actual p-values are reported too.
    The court's own probe neighbourhood is not the claimant's search and is never counted."""
    if trials.n_bars < 20:
        return inconclusive(TOOL, strategy_id, "fewer than 20 return observations", CITATIONS)
    if disclosed_trials(trials, extra_trials) < 2:
        return inconclusive(
            TOOL,
            strategy_id,
            "search not disclosed (single trial); no haircut applicable",
            CITATIONS,
        )
    selected = trials.selected_returns
    if trials.source == "probe":
        return haircut_sharpe(selected, n_trials=1 + extra_trials).to_evidence(strategy_id)
    m_eff = math.ceil(min(trials.n_trials, effective_trials_for_max(trials.returns))) + extra_trials
    decision = haircut_sharpe(selected, n_trials=max(m_eff, 1))
    raw = haircut_sharpe(
        selected,
        n_trials=trials.n_trials + extra_trials,
        other_trial_returns=np.delete(trials.returns, trials.selected, axis=1),
    )
    ev = decision.to_evidence(strategy_id)
    details = clean(
        {
            **ev.details,
            "effective_trials": m_eff,
            "raw_trials": raw.n_trials,
            **{f"raw_adjusted_p_{k}": v for k, v in raw.adjusted_p.items()},
            **{f"raw_haircut_sharpe_{k}": v for k, v in raw.haircut_sharpe.items()},
        }
    )
    assert isinstance(details, dict)
    return ev.model_copy(update={"details": details})
