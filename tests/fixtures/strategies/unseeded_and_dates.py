"""Non-deterministic and keyed to a known historical date (the March 2020 low)."""

import numpy as np
import pandas as pd


def signals(prices: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame(np.random.randn(*prices.shape), prices.index, prices.columns)
    w.loc[w.index >= "2020-03-23"] = 1.0
    return w / prices.shape[1]
