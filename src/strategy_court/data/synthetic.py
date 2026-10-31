"""Synthetic markets with known ground truth for the planted-flaw benchmark.

Returns follow a GARCH(1,1) process with standardised Student-t innovations
(Bollerslev 1986; Bollerslev 1987), giving fat tails and volatility clustering while the
conditional mean stays zero, so true alpha is exactly zero unless predictability is injected.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252
START = "2010-01-04"


@dataclass(frozen=True)
class GarchParams:
    alpha: float = 0.08
    beta: float = 0.90
    nu: float = 5.0
    annual_vol: float = 0.16

    def __post_init__(self) -> None:
        if not (self.alpha >= 0 and self.beta >= 0 and self.alpha + self.beta < 1):
            raise ValueError("GARCH requires alpha, beta >= 0 and alpha + beta < 1")
        if self.nu <= 2:
            raise ValueError("Student-t degrees of freedom must exceed 2 (finite variance)")
        if self.annual_vol <= 0:
            raise ValueError("annual_vol must be positive")


DEFAULT_GARCH = GarchParams()
SINGLE_STOCK_GARCH = GarchParams(annual_vol=0.40)


def garch_t_returns(
    n_days: int,
    n_assets: int,
    rng: np.random.Generator,
    params: GarchParams = DEFAULT_GARCH,
    ar1: float = 0.0,
    annual_drift: float = 0.0,
) -> np.ndarray:
    """Daily log returns, shape (n_days, n_assets).

    `ar1` injects predictability: standardised shocks follow x_t = ar1*x_{t-1} + sqrt(1-ar1^2)*z_t,
    so the lag-1 autocorrelation of returns is approximately `ar1` (the known signal strength).
    """
    if not -1 < ar1 < 1:
        raise ValueError("ar1 must be in (-1, 1)")
    p = params
    daily_var = p.annual_vol**2 / TRADING_DAYS
    omega = daily_var * (1 - p.alpha - p.beta)
    z = rng.standard_t(p.nu, size=(n_days, n_assets)) * np.sqrt((p.nu - 2) / p.nu)
    x = np.empty_like(z)
    x[0] = z[0]
    scale = np.sqrt(1 - ar1**2)
    for t in range(1, n_days):
        x[t] = ar1 * x[t - 1] + scale * z[t]
    eps = np.empty_like(x)
    sigma2 = np.full(n_assets, daily_var)
    eps[0] = np.sqrt(sigma2) * x[0]
    for t in range(1, n_days):
        sigma2 = omega + p.alpha * eps[t - 1] ** 2 + p.beta * sigma2
        eps[t] = np.sqrt(sigma2) * x[t]
    return np.asarray(eps + annual_drift / TRADING_DAYS, dtype=np.float64)


def _to_prices(log_returns: np.ndarray, prefix: str = "SYN") -> pd.DataFrame:
    n_days, n_assets = log_returns.shape
    index = pd.bdate_range(START, periods=n_days, name="date")
    columns = [f"{prefix}{i:03d}" for i in range(n_assets)]
    return pd.DataFrame(100 * np.exp(np.cumsum(log_returns, axis=0)), index=index, columns=columns)


def random_walk_prices(
    n_days: int = 2520, n_assets: int = 1, seed: int = 0, params: GarchParams = DEFAULT_GARCH
) -> pd.DataFrame:
    """Zero-alpha prices: any strategy that looks good on these is a false positive."""
    return _to_prices(garch_t_returns(n_days, n_assets, np.random.default_rng(seed), params))


def weak_signal_prices(
    n_days: int = 2520,
    n_assets: int = 1,
    ar1: float = 0.05,
    seed: int = 0,
    params: GarchParams = DEFAULT_GARCH,
) -> pd.DataFrame:
    """Prices with a small, known short-horizon momentum (lag-1 autocorrelation ~ `ar1`)."""
    rng = np.random.default_rng(seed)
    return _to_prices(garch_t_returns(n_days, n_assets, rng, params, ar1=ar1))


@dataclass(frozen=True)
class SurvivorshipUniverse:
    """Full universe incl. names that were delisted (NaN after their delisting date)."""

    prices: pd.DataFrame
    delisted: dict[str, pd.Timestamp] = field(default_factory=dict)

    @property
    def survivors(self) -> list[str]:
        return [c for c in self.prices.columns if c not in self.delisted]

    @property
    def survivor_prices(self) -> pd.DataFrame:
        """What a survivorship-biased dataset looks like: only names alive at the end."""
        return self.prices[self.survivors]


def survivorship_universe(
    n_days: int = 2520,
    n_assets: int = 50,
    seed: int = 0,
    delist_level: float = 0.4,
    params: GarchParams = SINGLE_STOCK_GARCH,
) -> SurvivorshipUniverse:
    """Zero-alpha universe where a name is delisted the first day it closes below
    `delist_level` x its starting price (losers disappear; cf. Brown et al. 1992)."""
    if not 0 < delist_level < 1:
        raise ValueError("delist_level must be in (0, 1)")
    prices = random_walk_prices(n_days, n_assets, seed, params)
    breached = prices.lt(delist_level * prices.iloc[0])
    delisted: dict[str, pd.Timestamp] = {}
    for col in prices.columns:
        hits = breached.index[breached[col].to_numpy()]
        if len(hits):
            delisted[str(col)] = pd.Timestamp(hits[0])
            prices.loc[prices.index > hits[0], col] = np.nan
    return SurvivorshipUniverse(prices, delisted)
