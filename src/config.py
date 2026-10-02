"""Project-wide settings: file locations, data source and study period.

Strategy assumptions (VIX thresholds, allocations, ...) live in
`src/strategy/config.py`, not here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_METADATA_FILE = RAW_DATA_DIR / "metadata.json"
MARKET_DATA_FILE = PROCESSED_DATA_DIR / "market_data.csv"
VIX_REGIMES_FILE = PROCESSED_DATA_DIR / "vix_regimes.csv"

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
DATA_SOURCE = "Yahoo Finance (via the yfinance package)"

# Dataset name -> Yahoo Finance ticker.
# ^NSEI is the NIFTY 50 *price* index (dividends are not included).
TICKERS = {
    "vix": "^INDIAVIX",
    "nifty": "^NSEI",
}

# ---------------------------------------------------------------------------
# Study period
# ---------------------------------------------------------------------------
# yfinance treats END_DATE as exclusive, so this covers 2015-01-01 to 2025-12-31.
START_DATE = "2015-01-01"
END_DATE = "2026-01-01"

# ---------------------------------------------------------------------------
# Market conventions
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
