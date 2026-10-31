import itertools
import math

import numpy as np
import pytest

from strategy_court.audit.pbo import pbo_cscv, pbo_evidence
from strategy_court.audit.purged_cv import (
    cpcv_paths,
    cpcv_splits,
    n_cpcv_paths,
    purged_cv_evidence,
    purged_cv_selection,
    purged_kfold,
)
from strategy_court.audit.trials import TrialSet

# ------------------------------------------------------------------ purging / embargo


def test_purged_folds_have_no_label_overlap_and_respect_embargo() -> None:
    n, h, emb = 1000, 10, 20
    for train, test in purged_kfold(n, 5, horizon=h, embargo=emb):
        a, b = test.min(), test.max() + 1
        train_spans = np.stack([train, train + h], axis=1)
        overlap = (train_spans[:, 1] >= a) & (train_spans[:, 0] <= b - 1 + h)
        assert not overlap.any()
        after = train[train > b - 1 + h]
        assert after.size == 0 or after.min() > b - 1 + h + emb
        assert np.intersect1d(train, test).size == 0


def test_purging_removes_overlapping_label_leak() -> None:
    """Known-answer demo (López de Prado 2018 ch.7): labels are overlapping 20-bar forward sums
    of pure noise; a 1-nearest-neighbour model on the time index 'learns' the neighbouring
    label. Without purging the OOS correlation is large; with purging + embargo it vanishes."""
    rng = np.random.default_rng(0)
    n, h = 2000, 20
    e = rng.standard_normal(n + h)
    y = np.array([e[i + 1 : i + 1 + h].sum() for i in range(n)])

    def oos_corr(splits: list[tuple[np.ndarray, np.ndarray]]) -> float:
        pred = np.empty(n)
        for train, test in splits:
            nearest = train[np.abs(test[:, None] - train[None, :]).argmin(axis=1)]
            pred[test] = y[nearest]
        return float(np.corrcoef(pred, y)[0, 1])

    leaky = oos_corr(purged_kfold(n, 50))
    clean = oos_corr(purged_kfold(n, 50, horizon=h, embargo=5))
    assert leaky > 0.4
    assert abs(clean) < 0.1


def test_cpcv_counts_and_paths_cover_every_group_once() -> None:
    splits, combos = cpcv_splits(600, n_groups=6, n_test_groups=2, horizon=3, embargo=5)
    assert len(splits) == math.comb(6, 2)
    paths = cpcv_paths(6, combos)
    assert len(paths) == n_cpcv_paths(6, 2) == 5
    for path in paths:
        assert all(g in combos[s] for g, s in enumerate(path))
    for g in range(6):
        assert sorted(p[g] for p in paths) == sorted(s for s, c in enumerate(combos) if g in c)


def test_selection_audit_on_noise_vs_real_edge() -> None:
    rng = np.random.default_rng(1)
    noise = TrialSet.from_returns(rng.normal(0, 0.01, (1500, 40)))
    res = purged_cv_selection(noise)
    assert res.is_sharpe > 0 and res.oos_sharpe < res.is_sharpe
    ev = res.to_evidence("s")
    assert ev.flag and ev.details["cpcv_paths"] == 5
    real = rng.normal(0, 0.01, (1500, 40))
    real[:, 5] += 0.0015
    good = purged_cv_selection(TrialSet.from_returns(real))
    assert good.selection_stability == 1.0 and good.oos_sharpe > 0.05
    assert not good.to_evidence("s").flag
    one = TrialSet.from_returns(real[:, :1])
    assert purged_cv_evidence(one, "s").details["status"] == "inconclusive"


# ------------------------------------------------------------------ PBO


def test_pbo_of_pure_noise_is_about_one_half() -> None:
    rng = np.random.default_rng(2)
    vals = [pbo_cscv(rng.normal(0, 0.01, (800, 50)), n_groups=10).pbo for _ in range(8)]
    assert 0.35 < float(np.mean(vals)) < 0.65


def test_pbo_near_zero_with_a_genuinely_superior_trial() -> None:
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, (1600, 30))
    r[:, 11] += 0.004
    res = pbo_cscv(r)
    assert res.pbo < 0.05 and res.prob_oos_loss < 0.05
    assert res.logits.size == math.comb(16, 8)


def test_pbo_matches_brute_force_on_small_case() -> None:
    rng = np.random.default_rng(4)
    r = rng.normal(0, 0.01, (120, 5))
    res = pbo_cscv(r, n_groups=6)
    groups = np.split(r, 6)

    def sr(x: np.ndarray) -> np.ndarray:
        return x.mean(axis=0) / x.std(axis=0, ddof=1)

    expected = []
    for combo in itertools.combinations(range(6), 3):
        is_ = np.vstack([groups[g] for g in combo])
        oos = np.vstack([groups[g] for g in range(6) if g not in combo])
        best = int(np.argmax(sr(is_)))
        rank = (sr(oos) <= sr(oos)[best]).sum()
        w = rank / 6
        expected.append(math.log(w / (1 - w)))
    np.testing.assert_allclose(res.logits, expected)


def test_pbo_validation_and_evidence() -> None:
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError):
        pbo_cscv(rng.normal(size=(100, 3)), n_groups=5)
    ev = pbo_evidence(TrialSet.from_returns(rng.normal(0, 0.01, (800, 20))), "s")
    assert ev.tool == "pbo" and ev.statistic is not None
    short = TrialSet.from_returns(rng.normal(0, 0.01, (50, 20)))
    assert pbo_evidence(short, "s").details["status"] == "inconclusive"
