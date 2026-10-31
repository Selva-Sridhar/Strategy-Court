"""Benchmark strategies as *source code*, so the static scanner sees exactly what the court sees.

These are repo-authored and trusted, so they are compiled in-process (D2a). Untrusted,
LLM-written strategies go through the Docker sandbox instead.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from strategy_court.audit.trials import StrategyFactory
from strategy_court.backtest.strategy import Strategy
from strategy_court.schemas import ParamValue

HEADER = "import numpy as np\nimport pandas as pd\n\n"

TEMPLATES: dict[str, str] = {
    # ---------------------------------------------------------------- clean
    "trend_ma": '''
class Strategy:
    """Long/short each asset on a fast/slow moving-average crossover (direction -1 = fade it)."""

    def __init__(self, fast=20, slow=80, direction=1):
        self.params = {"fast": int(fast), "slow": int(slow), "direction": int(direction)}

    def signals(self, prices):
        fast = prices.rolling(self.params["fast"]).mean()
        slow = prices.rolling(self.params["slow"]).mean()
        trend = np.sign(fast - slow).fillna(0.0)
        return self.params["direction"] * trend / prices.shape[1]
''',
    "time_momentum": '''
class Strategy:
    """Long/short each asset on the sign of its trailing return."""

    def __init__(self, lookback=60):
        self.params = {"lookback": int(lookback)}

    def signals(self, prices):
        past = prices.pct_change(self.params["lookback"], fill_method=None)
        return np.sign(past).fillna(0.0) / prices.shape[1]
''',
    "loser_reversal": '''
class Strategy:
    """Long, equally weighted, the bottom quantile of trailing returns ("buy the losers")."""

    def __init__(self, lookback=20, quantile=0.2):
        self.params = {"lookback": int(lookback), "quantile": float(quantile)}

    def signals(self, prices):
        past = prices.pct_change(self.params["lookback"], fill_method=None)
        losers = (past.rank(axis=1, pct=True) <= self.params["quantile"]).astype(float)
        count = losers.sum(axis=1).replace(0.0, np.nan)
        return losers.div(count, axis=0).fillna(0.0)
''',
    # ---------------------------------------------------------------- flawed
    "trend_ma_lookahead": '''
class Strategy:
    """Moving-average crossover 'improved' with a confirmation filter that peeks two bars ahead."""

    def __init__(self, fast=20, slow=80):
        self.params = {"fast": int(fast), "slow": int(slow)}

    def signals(self, prices):
        fast = prices.rolling(self.params["fast"]).mean()
        slow = prices.rolling(self.params["slow"]).mean()
        trend = np.sign(fast - slow).fillna(0.0)
        confirm = np.sign(prices.pct_change(fill_method=None).shift(-2)).fillna(0.0)
        return (0.85 * trend + 0.15 * confirm) / prices.shape[1]
''',
    "zscore_full_sample": '''
class Strategy:
    """Mean reversion to a 'fair value' z-score normalised with full-sample moments."""

    def __init__(self, band=0.5):
        self.params = {"band": float(band)}

    def signals(self, prices):
        z = (prices - prices.mean()) / prices.std()
        w = -np.sign(z) * (z.abs() > self.params["band"])
        return w.fillna(0.0) / prices.shape[1]
''',
}


def source(name: str) -> str:
    return HEADER + TEMPLATES[name].lstrip("\n")


def compile_strategy(name: str) -> type:
    namespace: dict[str, object] = {"np": np, "pd": pd}
    exec(compile(source(name), f"<benchmark:{name}>", "exec"), namespace)
    cls = namespace["Strategy"]
    assert isinstance(cls, type)
    return cls


def factory(name: str) -> StrategyFactory:
    cls = compile_strategy(name)

    def make(params: Mapping[str, ParamValue]) -> Strategy:
        instance = cls(**params)
        assert isinstance(instance, Strategy)
        return instance

    return make
