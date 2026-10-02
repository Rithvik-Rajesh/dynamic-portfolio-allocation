"""Data cleaning: turn the two raw price files into one aligned daily dataset.

Output columns (indexed by `date`):
    vix           India VIX closing level
    nifty         NIFTY 50 closing level
    nifty_return  NIFTY 50 simple daily return, close-to-close

Cleaning decisions (all explicit):
- Only closing values are used.
- Rows whose close is missing or not positive are dropped (never filled).
- Duplicate dates keep the first occurrence.
- Dates are aligned with an inner join: a day is kept only if BOTH VIX and
  NIFTY have a close. Prices are never forward-filled, because a filled
  price would create an artificial 0% return.
- `nifty_return` is calculated AFTER alignment, so when a date is dropped
  the next return covers both days and no NIFTY movement is lost.
"""

import pandas as pd


def clean_close_series(raw_prices: pd.DataFrame, name: str) -> pd.Series:
    """Extract a clean, sorted, de-duplicated series of closing values."""
    close = pd.to_numeric(raw_prices["Close"], errors="coerce")
    close.index = pd.to_datetime(raw_prices.index).normalize()
    close.index.name = "date"

    close = close.sort_index()
    close = close[~close.index.duplicated(keep="first")]
    close = close[close > 0]  # Also drops NaN, since NaN > 0 is False.
    return close.rename(name)


def align_series(vix_close: pd.Series, nifty_close: pd.Series) -> pd.DataFrame:
    """Keep only the dates on which both VIX and NIFTY have a closing value."""
    return pd.concat([vix_close, nifty_close], axis="columns", join="inner")


def add_nifty_return(market_data: pd.DataFrame) -> pd.DataFrame:
    """Add the simple daily NIFTY return. The first row has no return (NaN)."""
    market_data = market_data.copy()
    market_data["nifty_return"] = market_data["nifty"].pct_change()
    return market_data


def build_market_data(raw_vix: pd.DataFrame, raw_nifty: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline from raw downloads to the aligned dataset."""
    vix_close = clean_close_series(raw_vix, "vix")
    nifty_close = clean_close_series(raw_nifty, "nifty")
    market_data = align_series(vix_close, nifty_close)
    return add_nifty_return(market_data)


def unmatched_dates(raw_vix: pd.DataFrame, raw_nifty: pd.DataFrame) -> dict[str, list]:
    """List the dates present in one dataset but not the other (dropped by alignment)."""
    vix_dates = clean_close_series(raw_vix, "vix").index
    nifty_dates = clean_close_series(raw_nifty, "nifty").index
    return {
        "only_in_vix": list(vix_dates.difference(nifty_dates).date),
        "only_in_nifty": list(nifty_dates.difference(vix_dates).date),
    }


def save_market_data(market_data: pd.DataFrame, path) -> None:
    """Save the processed dataset to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    market_data.to_csv(path)


def read_market_data(path) -> pd.DataFrame:
    """Read a processed dataset written by `save_market_data`."""
    return pd.read_csv(path, index_col="date", parse_dates=["date"])
