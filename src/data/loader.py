"""Data acquisition: download India VIX and NIFTY 50 history and cache it locally.

Raw files are stored in `data/raw/` as simple CSVs with the columns
Date, Open, High, Low, Close, Volume. Once a file exists it is reused, so the
project gives the same results every run unless a refresh is requested.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import (
    DATA_SOURCE,
    END_DATE,
    RAW_DATA_DIR,
    RAW_METADATA_FILE,
    START_DATE,
    TICKERS,
)

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class DataDownloadError(RuntimeError):
    """Raised when market data cannot be downloaded."""


def raw_file_path(name: str) -> Path:
    """Return the cache file path for a dataset name such as 'vix' or 'nifty'."""
    return RAW_DATA_DIR / f"{name}.csv"


def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily prices for one ticker from Yahoo Finance.

    Returns a DataFrame indexed by Date with the columns in PRICE_COLUMNS.
    Raises DataDownloadError if the download fails or returns no rows.
    """
    try:
        prices = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,  # Indices have no splits/dividends, so this changes nothing.
            progress=False,
            multi_level_index=False,
        )
    except Exception as error:
        raise DataDownloadError(f"Download failed for {ticker}: {error}") from error

    # yfinance usually reports failures by returning an empty table, not by raising.
    if prices is None or prices.empty:
        raise DataDownloadError(
            f"No data returned for {ticker} between {start} and {end}. "
            "Check the internet connection and the ticker symbol."
        )

    prices = prices[PRICE_COLUMNS]
    prices.index.name = "Date"
    return prices


def save_raw_prices(prices: pd.DataFrame, path: Path) -> None:
    """Save raw prices as a single-header CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path)


def read_raw_prices(path: Path) -> pd.DataFrame:
    """Read a raw price CSV written by `save_raw_prices`."""
    return pd.read_csv(path, index_col="Date", parse_dates=["Date"])


def record_metadata(name: str, ticker: str, prices: pd.DataFrame) -> None:
    """Record where a raw dataset came from and what it covers."""
    metadata = {}
    if RAW_METADATA_FILE.exists():
        metadata = json.loads(RAW_METADATA_FILE.read_text())

    metadata[name] = {
        "ticker": ticker,
        "source": DATA_SOURCE,
        "requested_start": START_DATE,
        "requested_end_exclusive": END_DATE,
        "first_date": prices.index.min().strftime("%Y-%m-%d"),
        "last_date": prices.index.max().strftime("%Y-%m-%d"),
        "rows": len(prices),
        "downloaded_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }
    RAW_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    RAW_METADATA_FILE.write_text(json.dumps(metadata, indent=2) + "\n")


def load_raw_prices(name: str, refresh: bool = False) -> pd.DataFrame:
    """Load one raw dataset ('vix' or 'nifty').

    Uses the cached CSV when it exists. Downloads (and caches) it when the
    file is missing or when `refresh=True`.
    """
    if name not in TICKERS:
        raise ValueError(f"Unknown dataset '{name}'. Expected one of {list(TICKERS)}.")

    path = raw_file_path(name)
    if path.exists() and not refresh:
        return read_raw_prices(path)

    ticker = TICKERS[name]
    prices = download_prices(ticker, START_DATE, END_DATE)
    save_raw_prices(prices, path)
    record_metadata(name, ticker, prices)
    return prices


def load_raw_data(refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Load both raw datasets. Returns {'vix': DataFrame, 'nifty': DataFrame}."""
    return {name: load_raw_prices(name, refresh=refresh) for name in TICKERS}
