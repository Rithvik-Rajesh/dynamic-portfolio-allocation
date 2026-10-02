"""Run the pipeline built so far (Milestones 1-3).

    uv run python main.py            # use cached raw data
    uv run python main.py --refresh  # re-download raw data first

Steps: load raw data -> clean & align -> validate -> save processed data
-> VIX percentile & regimes -> regime analysis -> charts.
"""

import argparse

import pandas as pd

from src.analysis.regime_analysis import (
    nifty_behaviour_by_regime,
    regime_changes,
    regime_frequency,
)
from src.config import FIGURES_DIR, MARKET_DATA_FILE, VIX_REGIMES_FILE
from src.data.cleaner import build_market_data, save_market_data, unmatched_dates
from src.data.loader import load_raw_data
from src.data.validator import (
    describe_coverage,
    describe_market_data,
    validate_market_data,
)
from src.strategy.config import RegimeConfig
from src.strategy.regimes import add_vix_regimes
from src.visualization.charts import (
    plot_market_with_regimes,
    plot_regime_summary,
    plot_vix_percentile,
    save_figure,
)


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def main(refresh: bool = False) -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.4f}".format)

    # Milestone 1: raw data
    raw = load_raw_data(refresh=refresh)
    print_section("Raw data")
    for name, prices in raw.items():
        print(f"{name:>6}: {len(prices):,} rows, "
              f"{prices.index.min().date()} to {prices.index.max().date()}")

    # Milestone 2: clean, align, validate
    market_data = build_market_data(raw["vix"], raw["nifty"])
    validate_market_data(market_data)
    save_market_data(market_data, MARKET_DATA_FILE)

    print_section("Aligned dataset")
    print(describe_coverage(market_data))
    dropped = unmatched_dates(raw["vix"], raw["nifty"])
    print(f"Dates dropped by alignment: {len(dropped['only_in_vix'])} only in VIX, "
          f"{len(dropped['only_in_nifty'])} only in NIFTY")
    print(describe_market_data(market_data))
    print(f"Saved -> {MARKET_DATA_FILE}")

    # Milestone 3: VIX percentile, regimes, analysis
    config = RegimeConfig()
    regime_data = add_vix_regimes(market_data, config)
    regime_data.to_csv(VIX_REGIMES_FILE)

    print_section("VIX regimes")
    print(config)
    first_regime_date = regime_data["vix_regime"].first_valid_index().date()
    print(f"First date with a regime (after warm-up): {first_regime_date}")
    print(f"Regime changes: {regime_changes(regime_data)}")
    frequency = regime_frequency(regime_data)
    print(frequency)

    print_section("Next-day NIFTY behaviour by regime")
    behaviour = nifty_behaviour_by_regime(regime_data)
    print(behaviour)
    print(f"Saved -> {VIX_REGIMES_FILE}")

    save_figure(plot_market_with_regimes(regime_data), FIGURES_DIR / "market_with_regimes.png")
    save_figure(plot_vix_percentile(regime_data, config), FIGURES_DIR / "vix_percentile.png")
    save_figure(plot_regime_summary(frequency, behaviour), FIGURES_DIR / "regime_summary.png")
    print(f"Charts saved -> {FIGURES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download raw data")
    main(refresh=parser.parse_args().refresh)
