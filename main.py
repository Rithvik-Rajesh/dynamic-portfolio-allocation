"""Run the pipeline built so far (Milestones 1-6).

    uv run python main.py            # use cached raw data
    uv run python main.py --refresh  # re-download raw data first

Steps: load raw data -> clean & align -> validate -> VIX percentile & regimes
-> target allocation -> backtest -> performance & risk metrics -> charts.
"""

import argparse

import pandas as pd

from src.analysis.performance import summarise_performance
from src.analysis.regime_analysis import (
    nifty_behaviour_by_regime,
    regime_changes,
    regime_frequency,
)
from src.config import BACKTEST_FILE, FIGURES_DIR, MARKET_DATA_FILE, VIX_REGIMES_FILE
from src.data.cleaner import build_market_data, save_market_data, unmatched_dates
from src.data.loader import load_raw_data
from src.data.validator import (
    describe_coverage,
    describe_market_data,
    validate_market_data,
)
from src.pipeline import run_vix_strategy
from src.strategy.config import AllocationConfig, BacktestConfig, RegimeConfig
from src.strategy.regimes import add_vix_regimes
from src.visualization.charts import (
    plot_backtest,
    plot_market_with_regimes,
    plot_regime_summary,
    plot_vix_percentile,
    save_figure,
)


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def prepare_market_data(refresh: bool) -> pd.DataFrame:
    """Milestones 1-2: load raw data, clean, align, validate and save."""
    raw = load_raw_data(refresh=refresh)
    print_section("Raw data")
    for name, prices in raw.items():
        print(f"{name:>6}: {len(prices):,} rows, "
              f"{prices.index.min().date()} to {prices.index.max().date()}")

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
    return market_data


def analyse_regimes(market_data: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    """Milestone 3: VIX percentile, regimes, regime analysis and charts."""
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
    return regime_data


def backtest_strategy(market_data: pd.DataFrame, regime_config: RegimeConfig,
                      allocation_config: AllocationConfig,
                      backtest_config: BacktestConfig) -> pd.DataFrame:
    """Milestones 4-6: target allocation, backtest and performance metrics."""
    history = run_vix_strategy(market_data, regime_config, allocation_config, backtest_config)
    history.to_csv(BACKTEST_FILE)

    print_section("Strategy settings")
    print(regime_config)
    print(allocation_config)
    print(backtest_config)

    print_section("Backtest performance (VIX strategy)")
    metrics = summarise_performance(history, backtest_config.cash_annual_rate,
                                    backtest_config.initial_capital)
    for name, value in metrics.items():
        shown = f"{value:,.4f}" if isinstance(value, float) else value
        print(f"{name:>30}: {shown}")
    print(f"Saved -> {BACKTEST_FILE}")

    save_figure(plot_backtest(history), FIGURES_DIR / "backtest.png")
    return history


def main(refresh: bool = False) -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.4f}".format)

    # Every tuneable setting lives in these three objects. Change them here
    # (e.g. RegimeConfig(percentile_window="rolling", rolling_window=504) or
    # BacktestConfig(start_date="2018-01-01", execution_price="open")).
    regime_config = RegimeConfig()
    allocation_config = AllocationConfig()
    backtest_config = BacktestConfig()

    market_data = prepare_market_data(refresh)
    analyse_regimes(market_data, regime_config)
    backtest_strategy(market_data, regime_config, allocation_config, backtest_config)
    print(f"\nCharts saved -> {FIGURES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download raw data")
    main(refresh=parser.parse_args().refresh)
