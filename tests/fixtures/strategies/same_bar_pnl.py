"""Not the court interface: a typical self-made backtest that fills on the signal bar."""

import numpy as np
import pandas as pd


def backtest(prices: pd.DataFrame) -> pd.Series:
    rets = prices.pct_change(fill_method=None)
    position = np.sign(rets.rolling(10).mean())
    pnl = position * rets
    honest = position.shift(1) * rets
    return pnl.sum(axis=1) + 0 * honest.sum(axis=1)
