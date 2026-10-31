"""Real market data from Yahoo Finance, downloaded once and served from the hashed cache."""

import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import date

import pandas as pd

from strategy_court.data.store import DataError, Dataset, DataStore

Downloader = Callable[[Sequence[str], date, date], pd.DataFrame]


def download_close(tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
    """Split/dividend-adjusted daily closes, one column per ticker (network call)."""
    import yfinance as yf

    raw = yf.download(
        list(tickers),
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise DataError(f"yfinance returned no data for {list(tickers)}")
    return normalise_close(raw, tickers)


def normalise_close(raw: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Turn a yfinance frame (MultiIndex or flat columns) into a clean wide close-price frame."""
    if isinstance(raw.columns, pd.MultiIndex):
        selected = pd.DataFrame(raw["Close"])
    elif len(tickers) == 1:
        selected = pd.DataFrame(raw[["Close"]]).set_axis(list(tickers), axis=1)
    else:
        raise DataError("unexpected yfinance column layout")
    index = pd.DatetimeIndex(selected.index)
    index = (index.tz_localize(None) if index.tz is not None else index).normalize()
    close = pd.DataFrame(
        selected.reindex(columns=list(tickers)).to_numpy(dtype=float),
        index=pd.DatetimeIndex(index, name="date"),
        columns=list(tickers),
    )
    close = close.sort_index().dropna(how="all")
    empty = [t for t in tickers if close[t].isna().all()]
    if empty:
        raise DataError(f"no data for tickers: {empty}")
    return close


def cache_name(tickers: Sequence[str], start: date, end: date) -> str:
    key = json.dumps([list(tickers), start.isoformat(), end.isoformat()])
    return "yf_" + hashlib.sha256(key.encode()).hexdigest()[:16]


def load_prices(
    tickers: Sequence[str],
    start: date,
    end: date,
    store: DataStore,
    refresh: bool = False,
    downloader: Downloader = download_close,
) -> Dataset:
    """Serve from cache when present (hash-verified); otherwise download once and cache."""
    if not tickers:
        raise DataError("at least one ticker is required")
    if start >= end:
        raise DataError("start must be before end")
    name = cache_name(tickers, start, end)
    if store.exists(name) and not refresh:
        return store.load(name)
    return store.save(name, downloader(tickers, start, end))
