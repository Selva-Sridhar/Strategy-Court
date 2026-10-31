import numpy as np
import pandas as pd
import pytest

from strategy_court.backtest import MarketData
from strategy_court.data import DataError

NAN = np.nan
IDX = pd.bdate_range("2021-01-04", periods=4)


def df(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, IDX, ["A", "B"])


def test_listing_and_delisting_masks() -> None:
    m = MarketData(df([[NAN, 1], [1, 1], [1, 1], [1, NAN]]))
    np.testing.assert_array_equal(m.alive[:, 0], [False, True, True, True])
    np.testing.assert_array_equal(m.alive[:, 1], [True, True, True, False])
    assert m.delisted_at == {"B": IDX[2]}
    assert m.shape == (4, 2) and m.assets == ["A", "B"]


@pytest.mark.parametrize(
    ("close", "match"),
    [
        ([[1, 1], [NAN, 1], [1, 1], [1, 1]], "interior gaps"),
        ([[1, 1], [0, 1], [1, 1], [1, 1]], "positive"),
        ([[1, NAN], [1, NAN], [1, NAN], [1, NAN]], "no data"),
        ([[1, np.inf], [1, 1], [1, 1], [1, 1]], "finite"),
    ],
)
def test_close_validation(close: list[list[float]], match: str) -> None:
    with pytest.raises(DataError, match=match):
        MarketData(df(close))


def test_index_and_alignment_validation() -> None:
    good = df([[1, 1]] * 4)
    with pytest.raises(DataError, match="DatetimeIndex"):
        MarketData(good.reset_index(drop=True))
    with pytest.raises(DataError, match="increasing"):
        MarketData(good.iloc[::-1])
    with pytest.raises(DataError, match="same index"):
        MarketData(good, open=good.iloc[:3])
    with pytest.raises(DataError, match="volume"):
        MarketData(good, volume=-good)
