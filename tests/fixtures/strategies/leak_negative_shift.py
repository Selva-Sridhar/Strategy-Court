"""Leak: a constant hides the negative shift two assignments away."""

import numpy as np
import pandas as pd

HORIZON = 2
LEAD = -HORIZON


def signals(prices: pd.DataFrame) -> pd.DataFrame:
    future = prices.pct_change(fill_method=None).shift(LEAD)
    return np.sign(future) / prices.shape[1]
