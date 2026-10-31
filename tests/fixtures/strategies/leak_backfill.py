"""Leak: weekly signal back-filled onto the days before it was known."""

import numpy as np
import pandas as pd


def signals(prices: pd.DataFrame) -> pd.DataFrame:
    weekly = prices.pct_change(5, fill_method=None).iloc[::5]
    daily = weekly.reindex(prices.index).bfill()
    return np.sign(daily) / prices.shape[1]
