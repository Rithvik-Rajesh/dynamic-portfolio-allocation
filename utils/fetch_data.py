from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DATA = Path("data/raw")
RAW_DATA.mkdir(parents=True, exist_ok=True)


def main():
    vix = yf.download(
        "^INDIAVIX",
        start="2015-01-01",
        end="2026-01-01",
    )

    nifty = yf.download(
        "^NSEI",
        start="2015-01-01",
        end="2026-01-01",
    )

    if vix is None or nifty is None:
        raise RuntimeError("Failed to download market data")

    vix.to_csv(RAW_DATA / "vix.csv")
    nifty.to_csv(RAW_DATA / "nifty.csv")

    print(f"VIX:   {len(vix):,} rows → {RAW_DATA / 'vix.csv'}")
    print(f"NIFTY: {len(nifty):,} rows → {RAW_DATA / 'nifty.csv'}")


if __name__ == "__main__":
    main()
