import numpy as np
import pandas as pd
import pytest
from scipy import stats

from strategy_court.data import (
    GarchParams,
    garch_t_returns,
    random_walk_prices,
    survivorship_universe,
    weak_signal_prices,
)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.log(prices)).diff().dropna()


def acf1(x: np.ndarray) -> float:
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def test_random_walk_has_zero_mean_fat_tails_and_vol_clustering() -> None:
    r = log_returns(random_walk_prices(n_days=10_000, n_assets=4, seed=1)).to_numpy()
    for col in r.T:
        se = col.std(ddof=1) / np.sqrt(len(col))
        assert abs(col.mean()) < 4 * se
        assert stats.kurtosis(col) > 1.0
        assert abs(acf1(col)) < 0.05
        assert acf1(col**2) > 0.05
    annual_vol = r.std(axis=0, ddof=1).mean() * np.sqrt(252)
    assert annual_vol == pytest.approx(0.16, rel=0.25)


def test_weak_signal_has_known_autocorrelation() -> None:
    phi = 0.10
    r = log_returns(weak_signal_prices(n_days=20_000, n_assets=4, ar1=phi, seed=2)).to_numpy()
    observed = np.mean([acf1(col) for col in r.T])
    assert 0.5 * phi < observed < 1.5 * phi


def test_generators_are_seed_deterministic() -> None:
    a = random_walk_prices(n_days=300, n_assets=2, seed=7)
    pd.testing.assert_frame_equal(a, random_walk_prices(n_days=300, n_assets=2, seed=7))
    assert not a.equals(random_walk_prices(n_days=300, n_assets=2, seed=8))
    assert a.index.is_monotonic_increasing and a.index.name == "date"


def test_survivorship_universe_drops_losers() -> None:
    u = survivorship_universe(n_days=2520, n_assets=50, seed=3)
    assert 0 < len(u.delisted) < 50
    for name, when in u.delisted.items():
        col = u.prices[name]
        assert col[col.index > when].isna().all()
        assert col[when] < 0.4 * col.iloc[0]
    assert u.survivor_prices.notna().all().all()
    total = u.prices.ffill().iloc[-1] / u.prices.iloc[0]
    assert total[u.survivors].mean() > total.mean()  # the bias the court must catch


@pytest.mark.parametrize("kwargs", [{"alpha": 0.5, "beta": 0.6}, {"nu": 2.0}, {"annual_vol": 0.0}])
def test_garch_params_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        GarchParams(**kwargs)


def test_ar1_bounds() -> None:
    with pytest.raises(ValueError):
        garch_t_returns(10, 1, np.random.default_rng(0), ar1=1.0)
