"""Run the full research pipeline (Milestones 1-9).

    uv run python main.py            # use cached raw data
    uv run python main.py --refresh  # re-download raw data first

Steps: load raw data -> clean & align -> validate -> VIX percentile & regimes
-> target allocation -> backtest -> performance & risk metrics
-> benchmark comparison -> sensitivity experiments -> walk-forward validation.
Tables are saved to reports/tables/, charts to reports/figures/.
"""

import argparse

import pandas as pd

from src.analysis.benchmarks import (
    STRATEGY_NAME,
    compare_performance,
    portfolio_values,
    run_benchmarks,
)
from src.analysis.performance import summarise_performance
from src.analysis.regime_analysis import (
    nifty_behaviour_by_regime,
    regime_changes,
    regime_frequency,
)
from src.analysis.sensitivity import run_all_experiments
from src.analysis.walk_forward import (
    out_of_sample_summary,
    portfolio_order,
    run_walk_forward,
)
from src.config import (
    BACKTEST_FILE,
    FIGURES_DIR,
    MARKET_DATA_FILE,
    TABLES_DIR,
    VIX_REGIMES_FILE,
)
from src.data.cleaner import build_market_data, save_market_data, unmatched_dates
from src.data.loader import load_raw_data
from src.data.validator import (
    describe_coverage,
    describe_market_data,
    validate_market_data,
)
from src.pipeline import run_settings
from src.strategy.config import (
    AllocationConfig,
    BacktestConfig,
    BenchmarkConfig,
    RegimeConfig,
    StrategySettings,
)
from src.strategy.regimes import add_vix_regimes
from src.visualization.charts import (
    plot_backtest,
    plot_benchmark_comparison,
    plot_market_with_regimes,
    plot_regime_summary,
    plot_sensitivity_overview,
    plot_vix_percentile,
    plot_walk_forward,
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


def save_table(table: pd.DataFrame, name: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / f"{name}.csv")


def backtest_strategy(market_data: pd.DataFrame, settings: StrategySettings) -> pd.DataFrame:
    """Milestones 4-6: target allocation, backtest and performance metrics."""
    history = run_settings(market_data, settings)
    history.to_csv(BACKTEST_FILE)

    print_section("Strategy settings")
    print(settings.regime)
    print(settings.allocation)
    print(settings.backtest)

    print_section("Backtest performance (VIX strategy)")
    metrics = summarise_performance(history, settings.backtest.cash_annual_rate,
                                    settings.backtest.initial_capital)
    for name, value in metrics.items():
        shown = f"{value:,.4f}" if isinstance(value, float) else value
        print(f"{name:>30}: {shown}")
    print(f"Saved -> {BACKTEST_FILE}")

    save_figure(plot_backtest(history), FIGURES_DIR / "backtest.png")
    return history


def compare_with_benchmarks(market_data: pd.DataFrame, strategy_history: pd.DataFrame,
                            settings: StrategySettings, benchmark_config: BenchmarkConfig) -> None:
    """Milestone 7: strategy vs buy-and-hold vs fixed allocation."""
    histories = {STRATEGY_NAME: strategy_history}
    histories.update(run_benchmarks(market_data, strategy_history, settings.backtest,
                                    benchmark_config.fixed_nifty_weight))
    comparison = compare_performance(histories, settings.backtest)
    save_table(comparison, "benchmark_comparison")

    print_section("Strategy vs benchmarks")
    print(benchmark_config)
    print(comparison.T)
    save_figure(plot_benchmark_comparison(portfolio_values(histories)),
                FIGURES_DIR / "benchmark_comparison.png")


def run_experiments(market_data: pd.DataFrame, settings: StrategySettings,
                    benchmark_config: BenchmarkConfig) -> None:
    """Milestone 8: one-at-a-time sensitivity experiments."""
    experiments = run_all_experiments(market_data, settings, benchmark_config.fixed_nifty_weight)
    save_table(experiments, "sensitivity_experiments")

    print_section("Sensitivity experiments (strategy minus buy-and-hold)")
    columns = ["experiment", "variant", "strategy_cagr", "strategy_sharpe_ratio",
               "strategy_max_drawdown", "cagr_vs_buy_and_hold", "sharpe_vs_buy_and_hold",
               "max_drawdown_vs_buy_and_hold", "strategy_number_of_trades"]
    print(experiments[columns].to_string(index=False))
    save_figure(plot_sensitivity_overview(experiments), FIGURES_DIR / "sensitivity_overview.png")


def validate_walk_forward(market_data: pd.DataFrame, settings: StrategySettings,
                          benchmark_config: BenchmarkConfig) -> None:
    """Milestone 9: walk-forward validation."""
    folds, chained_returns = run_walk_forward(market_data, settings,
                                              benchmark_config.fixed_nifty_weight)
    names = portfolio_order(benchmark_config.fixed_nifty_weight)
    summary = out_of_sample_summary(chained_returns[names], settings.backtest.cash_annual_rate)
    save_table(folds, "walk_forward_folds")
    save_table(summary, "walk_forward_summary")

    print_section("Walk-forward: selection and test-year results")
    print(folds.T)
    print_section("Walk-forward: chained out-of-sample performance")
    print(summary)
    save_figure(plot_walk_forward(folds, chained_returns, names), FIGURES_DIR / "walk_forward.png")


def main(refresh: bool = False) -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.4f}".format)

    # Every tuneable setting lives in these objects. Change them here, e.g.
    # RegimeConfig(percentile_window="rolling", rolling_window=504) or
    # BacktestConfig(start_date="2018-01-01", execution_price="open").
    settings = StrategySettings(
        regime=RegimeConfig(),
        allocation=AllocationConfig(),
        backtest=BacktestConfig(),
    )
    benchmark_config = BenchmarkConfig()

    market_data = prepare_market_data(refresh)
    analyse_regimes(market_data, settings.regime)
    strategy_history = backtest_strategy(market_data, settings)
    compare_with_benchmarks(market_data, strategy_history, settings, benchmark_config)
    run_experiments(market_data, settings, benchmark_config)
    validate_walk_forward(market_data, settings, benchmark_config)
    print(f"\nTables saved -> {TABLES_DIR}")
    print(f"Charts saved -> {FIGURES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download raw data")
    main(refresh=parser.parse_args().refresh)
