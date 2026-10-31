"""yfinance loader tests. All offline (mocked) except one `network` test for the PoC machine."""

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategy_court.data import DataError, DataStore, load_prices
from strategy_court.data.yahoo import normalise_close

START, END = date(2020, 1, 1), date(2020, 2, 1)


def fake_yf_frame(tickers: Sequence[str], tz: str | None = "America/New_York") -> pd.DataFrame:
    idx = pd.date_range("2020-01-02", periods=4, freq="B", tz=tz)
    cols = pd.MultiIndex.from_product([["Close", "Volume"], list(tickers)])
    return pd.DataFrame(np.arange(4 * len(cols), dtype=float).reshape(4, -1) + 1, idx, cols)


def test_normalise_multiindex_frame() -> None:
    out = normalise_close(fake_yf_frame(["SPY", "QQQ"]), ["SPY", "QQQ"])
    assert list(out.columns) == ["SPY", "QQQ"]
    assert out.index.tz is None and out.index.name == "date"
    assert out.index[0] == pd.Timestamp("2020-01-02")


def test_normalise_flat_single_ticker() -> None:
    flat = fake_yf_frame(["SPY"], tz=None).droplevel(1, axis=1)
    assert list(normalise_close(flat, ["SPY"]).columns) == ["SPY"]


def test_normalise_rejects_ticker_without_data() -> None:
    raw = fake_yf_frame(["SPY", "BAD"])
    raw[("Close", "BAD")] = np.nan
    with pytest.raises(DataError, match="BAD"):
        normalise_close(raw, ["SPY", "BAD"])


def test_load_prices_downloads_once_then_serves_cache(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def downloader(tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        calls.append(tuple(tickers))
        return normalise_close(fake_yf_frame(tickers), tickers)

    store = DataStore(tmp_path)
    first = load_prices(["SPY"], START, END, store, downloader=downloader)
    second = load_prices(["SPY"], START, END, store, downloader=downloader)
    assert len(calls) == 1
    assert first.sha256 == second.sha256
    load_prices(["SPY"], START, END, store, refresh=True, downloader=downloader)
    assert len(calls) == 2


def test_load_prices_validates_arguments(tmp_path: Path) -> None:
    with pytest.raises(DataError):
        load_prices([], START, END, DataStore(tmp_path))
    with pytest.raises(DataError):
        load_prices(["SPY"], END, START, DataStore(tmp_path))


@pytest.mark.network
def test_live_yfinance_download(tmp_path: Path) -> None:
    ds = load_prices(["SPY", "QQQ"], date(2023, 1, 1), date(2023, 3, 1), DataStore(tmp_path))
    assert list(ds.frame.columns) == ["SPY", "QQQ"]
    assert len(ds.frame) > 30
    assert ds.frame.notna().all().all()
