"""Strategy experiments (Milestone 8): how sensitive are the results to each setting?

Method: one-at-a-time variation. Start from a base StrategySettings, change
ONE setting to each value in a list, keep everything else at the base, and
backtest. Every variant is compared with buy-and-hold and the fixed-allocation
benchmark run with the same backtest settings over the same dates.

The purpose is robustness, not optimisation: the question is whether the
strategy's behaviour holds across reasonable settings, not which setting had
the best historical number. Nothing in this module picks a "winner".
"""

from dataclasses import replace

import pandas as pd

from src.analysis.benchmarks import (
    BUY_AND_HOLD_NAME,
    STRATEGY_NAME,
    compare_performance,
    fixed_allocation_name,
    run_benchmarks,
)
from src.pipeline import run_settings
from src.strategy.config import AllocationConfig, StrategySettings

# ---------------------------------------------------------------------------
# Values tried in each experiment (edit here to change the experiments).
# ---------------------------------------------------------------------------
HIGH_THRESHOLDS = [0.70, 0.75, 0.80, 0.90]
EXTREME_THRESHOLDS = [0.90, 0.95, 0.98]
LOW_THRESHOLDS = [0.10, 0.25, 0.40]
ALLOCATION_SETS = {
    "100/90/80/70 (mild)": AllocationConfig(low=1.00, normal=0.90, high=0.80, extreme=0.70),
    "100/100/75/50 (aggressive)": AllocationConfig(low=1.00, normal=1.00, high=0.75, extreme=0.50),
    "100/75/50/25 (base)": AllocationConfig(low=1.00, normal=0.75, high=0.50, extreme=0.25),
    "100/50/25/0 (defensive)": AllocationConfig(low=1.00, normal=0.50, high=0.25, extreme=0.00),
}
REBALANCE_FREQUENCIES = ["daily", "weekly", "monthly"]
DRIFT_TOLERANCES = [0.0, 0.02, 0.05, 0.10, 0.20]
TRANSACTION_COST_RATES = [0.0, 0.0005, 0.001, 0.002, 0.005]
CASH_RATES = [0.0, 0.04, 0.06, 0.08]
PERCENTILE_WINDOWS = {  # label -> (percentile_window, rolling_window)
    "expanding": ("expanding", 504),
    "rolling 252d": ("rolling", 252),
    "rolling 504d": ("rolling", 504),
    "rolling 756d": ("rolling", 756),
    "rolling 1008d": ("rolling", 1008),
}
EXECUTION_PRICES = ["close", "open"]
START_YEARS = list(range(2016, 2025))

# Metrics copied into the experiment tables.
TABLE_METRICS = ["cagr", "annualised_volatility", "sharpe_ratio", "sortino_ratio",
                 "max_drawdown", "number_of_trades", "total_transaction_costs",
                 "average_nifty_weight"]


def vary(base: StrategySettings, section: str, **changes) -> StrategySettings:
    """Copy of `base` with fields of one section ('regime', 'allocation',
    'backtest') changed, e.g. vary(base, "regime", high_threshold=0.8)."""
    return replace(base, **{section: replace(getattr(base, section), **changes)})


def build_experiments(base: StrategySettings) -> dict[str, dict[str, StrategySettings]]:
    """All experiments: {experiment name: {variant label: settings}}."""
    return {
        "high_threshold": {f"{value:.0%}": vary(base, "regime", high_threshold=value)
                           for value in HIGH_THRESHOLDS},
        "extreme_threshold": {f"{value:.0%}": vary(base, "regime", extreme_threshold=value)
                              for value in EXTREME_THRESHOLDS},
        "low_threshold": {f"{value:.0%}": vary(base, "regime", low_threshold=value)
                          for value in LOW_THRESHOLDS},
        "allocation": {label: replace(base, allocation=allocation)
                       for label, allocation in ALLOCATION_SETS.items()},
        "percentile_window": {
            label: vary(base, "regime", percentile_window=window, rolling_window=size)
            for label, (window, size) in PERCENTILE_WINDOWS.items()
        },
        "rebalance_frequency": {value: vary(base, "backtest", rebalance_frequency=value)
                                for value in REBALANCE_FREQUENCIES},
        "drift_tolerance": {f"{value * 100:g} pts": vary(base, "backtest", drift_tolerance=value)
                            for value in DRIFT_TOLERANCES},
        "transaction_cost": {f"{value:.2%}": vary(base, "backtest", transaction_cost_rate=value)
                             for value in TRANSACTION_COST_RATES},
        "cash_rate": {f"{value:.0%}": vary(base, "backtest", cash_annual_rate=value)
                      for value in CASH_RATES},
        "execution_price": {value: vary(base, "backtest", execution_price=value)
                            for value in EXECUTION_PRICES},
        "start_year": {str(year): vary(base, "backtest", start_date=f"{year}-01-01")
                       for year in START_YEARS},
    }


def evaluate_settings(market_data: pd.DataFrame, settings: StrategySettings,
                      fixed_nifty_weight: float) -> pd.DataFrame:
    """Backtest the strategy and both benchmarks; one row of metrics each."""
    strategy_history = run_settings(market_data, settings)
    histories = {STRATEGY_NAME: strategy_history}
    histories.update(run_benchmarks(market_data, strategy_history, settings.backtest,
                                    fixed_nifty_weight))
    return compare_performance(histories, settings.backtest)


def flatten_comparison(comparison: pd.DataFrame, fixed_nifty_weight: float) -> dict:
    """One flat row: strategy metrics, benchmark metrics and strategy-minus-benchmark gaps."""
    fixed_name = fixed_allocation_name(fixed_nifty_weight)
    strategy = comparison.loc[STRATEGY_NAME]
    buy_and_hold = comparison.loc[BUY_AND_HOLD_NAME]
    fixed = comparison.loc[fixed_name]

    row = {"start_date": strategy["start_date"], "end_date": strategy["end_date"]}
    row.update({f"strategy_{metric}": strategy[metric] for metric in TABLE_METRICS})
    for prefix, benchmark in [("buy_and_hold", buy_and_hold), (fixed_name, fixed)]:
        for metric in ["cagr", "sharpe_ratio", "max_drawdown"]:
            row[f"{prefix}_{metric}"] = benchmark[metric]

    # Positive = strategy better. A drawdown is negative, so a shallower
    # strategy drawdown gives a positive gap.
    row["cagr_vs_buy_and_hold"] = strategy["cagr"] - buy_and_hold["cagr"]
    row["sharpe_vs_buy_and_hold"] = strategy["sharpe_ratio"] - buy_and_hold["sharpe_ratio"]
    row["max_drawdown_vs_buy_and_hold"] = strategy["max_drawdown"] - buy_and_hold["max_drawdown"]
    row["sharpe_vs_fixed"] = strategy["sharpe_ratio"] - fixed["sharpe_ratio"]
    return row


def run_experiment(market_data: pd.DataFrame, variants: dict[str, StrategySettings],
                   base: StrategySettings, fixed_nifty_weight: float) -> pd.DataFrame:
    """Evaluate every variant of one experiment. One row per variant."""
    rows = {}
    for label, settings in variants.items():
        comparison = evaluate_settings(market_data, settings, fixed_nifty_weight)
        row = flatten_comparison(comparison, fixed_nifty_weight)
        row["is_base"] = settings == base
        rows[label] = row
    table = pd.DataFrame(rows).T
    table.index.name = "variant"
    return table


def run_all_experiments(market_data: pd.DataFrame, base: StrategySettings,
                        fixed_nifty_weight: float) -> pd.DataFrame:
    """Run every experiment and stack the results into one table."""
    tables = []
    for name, variants in build_experiments(base).items():
        table = run_experiment(market_data, variants, base, fixed_nifty_weight)
        table.insert(0, "experiment", name)
        tables.append(table.reset_index())
    return pd.concat(tables, ignore_index=True)
