"""Purged and embargoed cross-validation (López de Prado 2018, ch. 7 and 12).

Labels in finance span time: the outcome attributed to a decision at bar i is realised over
[i, i + h]. Ordinary K-fold lets a training observation whose label overlaps the test period
leak test information into the fit. Two fixes:

* Purging: drop training observations i whose label interval [i, i + h] overlaps the test
  label span [a, b - 1 + h] of a test block [a, b).
* Embargo: also drop training observations starting within `embargo` bars after the test
  label span (serial correlation outlives the label).

Combinatorial purged CV (CPCV) splits the sample into N groups and tests on every combination of
k groups, giving phi = k/N * C(N, k) complete out-of-sample backtest paths instead of one.

Audit use: the *selection procedure* (pick the best configuration of a trial set) is
cross-validated. Each fold picks the trial with the best purged in-sample Sharpe and records its
out-of-sample returns; the concatenated OOS series is what the research process would actually
have delivered.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np

from strategy_court.audit._common import inconclusive, make_evidence
from strategy_court.audit._stats import sharpe_per_period
from strategy_court.audit.citations import LOPEZ_DE_PRADO_2018
from strategy_court.audit.psr import psr_test
from strategy_court.audit.trials import TrialSet
from strategy_court.schemas import Evidence, Severity

TOOL = "purged_cv"
CITATIONS = (LOPEZ_DE_PRADO_2018,)
Split = tuple[np.ndarray, np.ndarray]


def _blocks(n: int, n_groups: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n, n_groups + 1).astype(int)
    return [(int(a), int(b)) for a, b in itertools.pairwise(edges)]


def _train_mask(
    n: int, test_blocks: list[tuple[int, int]], horizon: int, embargo: int
) -> np.ndarray:
    keep = np.ones(n, dtype=bool)
    i = np.arange(n)
    for a, b in test_blocks:
        span_end = b - 1 + horizon
        keep &= ~((i + horizon >= a) & (i <= span_end))
        keep &= ~((i > span_end) & (i <= span_end + embargo))
    return keep


def purged_kfold(n: int, n_splits: int = 5, horizon: int = 0, embargo: int = 0) -> list[Split]:
    """Contiguous folds; training indices purged of label overlap and embargoed after the test."""
    if not 2 <= n_splits <= n:
        raise ValueError("need 2 <= n_splits <= n")
    out = []
    for a, b in _blocks(n, n_splits):
        train = np.flatnonzero(_train_mask(n, [(a, b)], horizon, embargo))
        out.append((train, np.arange(a, b)))
    return out


def cpcv_splits(
    n: int, n_groups: int = 6, n_test_groups: int = 2, horizon: int = 0, embargo: int = 0
) -> tuple[list[Split], list[tuple[int, ...]]]:
    """All C(N, k) purged splits and, for each, the tuple of test-group ids."""
    if not 1 <= n_test_groups < n_groups:
        raise ValueError("need 1 <= n_test_groups < n_groups")
    blocks = _blocks(n, n_groups)
    splits, combos = [], []
    for combo in itertools.combinations(range(n_groups), n_test_groups):
        tb = [blocks[g] for g in combo]
        train = np.flatnonzero(_train_mask(n, tb, horizon, embargo))
        test = np.concatenate([np.arange(a, b) for a, b in tb])
        splits.append((train, test))
        combos.append(combo)
    return splits, combos


def cpcv_paths(n_groups: int, combos: list[tuple[int, ...]]) -> list[list[int]]:
    """phi backtest paths; path p uses, for every group g, the p-th split testing g."""
    by_group: list[list[int]] = [[] for _ in range(n_groups)]
    for s, combo in enumerate(combos):
        for g in combo:
            by_group[g].append(s)
    n_paths = len(by_group[0])
    assert all(len(v) == n_paths for v in by_group)
    return [[by_group[g][p] for g in range(n_groups)] for p in range(n_paths)]


def n_cpcv_paths(n_groups: int, n_test_groups: int) -> int:
    return math.comb(n_groups, n_test_groups) * n_test_groups // n_groups


def _pick(returns: np.ndarray, train: np.ndarray) -> int:
    sr = np.array([sharpe_per_period(returns[train, j]) for j in range(returns.shape[1])])
    return int(np.nanargmax(np.where(np.isfinite(sr), sr, -np.inf)))


@dataclass(frozen=True)
class PurgedCVResult:
    is_sharpe: float
    oos_sharpe: float
    oos_sharpe_unpurged: float
    oos_psr: float
    selection_stability: float  # share of folds that pick the full-sample choice
    path_sharpes: np.ndarray  # CPCV OOS Sharpe per path
    n_splits: int
    horizon: int
    embargo: int
    periods_per_year: int

    def to_evidence(self, strategy_id: str) -> Evidence:
        ann = math.sqrt(self.periods_per_year)
        is_ann, oos_ann = self.is_sharpe * ann, self.oos_sharpe * ann
        degradation = 1 - self.oos_sharpe / self.is_sharpe if self.is_sharpe > 0 else float("nan")
        frac_neg = (
            float((self.path_sharpes <= 0).mean()) if self.path_sharpes.size else float("nan")
        )
        weak = not (self.oos_psr >= 0.95)
        sev = (
            Severity.HIGH
            if oos_ann <= 0
            else Severity.MEDIUM
            if weak or (math.isfinite(degradation) and degradation > 0.5)
            else Severity.INFO
        )
        return make_evidence(
            TOOL,
            strategy_id,
            flag=sev >= Severity.MEDIUM,
            severity=sev,
            statistic_name="oos_sharpe_annual",
            statistic=oos_ann,
            p_value=1 - self.oos_psr if math.isfinite(self.oos_psr) else None,
            summary=(
                f"Purged {self.n_splits}-fold CV of the selection: in-sample Sharpe {is_ann:.2f} "
                f"-> out-of-sample {oos_ann:.2f}; {frac_neg:.0%} of CPCV paths lose money."
            ),
            details={
                "in_sample_sharpe_annual": is_ann,
                "oos_sharpe_annual": oos_ann,
                "oos_sharpe_unpurged_annual": self.oos_sharpe_unpurged * ann,
                "degradation": degradation,
                "oos_psr": self.oos_psr,
                "selection_stability": self.selection_stability,
                "cpcv_paths": int(self.path_sharpes.size),
                "cpcv_median_sharpe_annual": float(np.median(self.path_sharpes)) * ann
                if self.path_sharpes.size
                else None,
                "cpcv_fraction_paths_negative": frac_neg,
                "label_horizon": self.horizon,
                "embargo": self.embargo,
            },
            plot_data={"cpcv_path_sharpes_annual": self.path_sharpes * ann},
            citations=CITATIONS,
        )


def purged_cv_selection(
    trials: TrialSet,
    n_splits: int = 5,
    horizon: int = 1,
    embargo_pct: float = 0.01,
    cpcv_groups: int = 6,
    cpcv_test_groups: int = 2,
    periods_per_year: int = 252,
) -> PurgedCVResult:
    r = trials.returns
    n = r.shape[0]
    embargo = math.ceil(embargo_pct * n)
    full_pick = trials.selected
    oos = np.full(n, np.nan)
    oos_naive = np.full(n, np.nan)
    picks = []
    for (train, test), (train_naive, _) in zip(
        purged_kfold(n, n_splits, horizon, embargo), purged_kfold(n, n_splits), strict=True
    ):
        j = _pick(r, train)
        picks.append(j)
        oos[test] = r[test, j]
        oos_naive[test] = r[test, _pick(r, train_naive)]
    splits, combos = cpcv_splits(n, cpcv_groups, cpcv_test_groups, horizon, embargo)
    picks_cpcv = [_pick(r, train) for train, _ in splits]
    blocks = _blocks(n, cpcv_groups)
    paths = []
    for path in cpcv_paths(cpcv_groups, combos):
        series = np.concatenate(
            [r[blocks[g][0] : blocks[g][1], picks_cpcv[s]] for g, s in enumerate(path)]
        )
        paths.append(sharpe_per_period(series))
    return PurgedCVResult(
        is_sharpe=sharpe_per_period(r[:, full_pick]),
        oos_sharpe=sharpe_per_period(oos),
        oos_sharpe_unpurged=sharpe_per_period(oos_naive),
        oos_psr=psr_test(oos, periods_per_year=periods_per_year).psr,
        selection_stability=float(np.mean([p == full_pick for p in picks])),
        path_sharpes=np.array(paths),
        n_splits=n_splits,
        horizon=horizon,
        embargo=embargo,
        periods_per_year=periods_per_year,
    )


def purged_cv_evidence(trials: TrialSet, strategy_id: str, **kw: object) -> Evidence:
    if trials.returns.shape[1] < 2:
        return inconclusive(
            TOOL, strategy_id, "needs at least two trials to select from", CITATIONS
        )
    if trials.n_bars < 100:
        return inconclusive(TOOL, strategy_id, "fewer than 100 observations", CITATIONS)
    return purged_cv_selection(trials, **kw).to_evidence(strategy_id)  # type: ignore[arg-type]
