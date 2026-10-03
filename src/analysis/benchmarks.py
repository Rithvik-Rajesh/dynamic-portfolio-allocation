"""Benchmarks: simple static portfolios the VIX strategy is compared against.

    buy_and_hold   100% NIFTY, 0% cash, never traded after the first purchase
    fixed_<w>      a constant w% NIFTY / (100-w)% cash, kept near target with the
                   same rebalancing schedule and drift band as the strategy

Fairness rules: every benchmark is run through the SAME backtest engine with
the SAME BacktestConfig as the strategy (initial capital, transaction costs,
cash rate, execution price/lag, rebalancing), over EXACTLY the same dates.
Only the target weight differs.
"""

from dataclasses import replace

import pandas as pd

from src.analysis.performance import summarise_performance
from src.backtesting.engine import run_backtest
from src.strategy.config import BacktestConfig

STRATEGY_NAME = "vix_strategy"
BUY_AND_HOLD_NAME = "buy_and_hold"


def fixed_allocation_name(nifty_weight: float) -> str:
    """Label such as 'fixed_75' for a 75% NIFTY benchmark."""
    return f"fixed_{nifty_weight * 100:g}"


def run_constant_weight(market_data: pd.DataFrame, nifty_weight: float,
                        config: BacktestConfig) -> pd.DataFrame:
    """Backtest a portfolio whose target NIFTY weight never changes."""
    target = pd.Series(nifty_weight, index=market_data.index)
    return run_backtest(market_data, target, config)


def run_benchmarks(market_data: pd.DataFrame, strategy_history: pd.DataFrame,
                   config: BacktestConfig, fixed_nifty_weight: float) -> dict[str, pd.DataFrame]:
    """Run buy-and-hold and the fixed-allocation benchmark over the strategy's dates.

    The strategy cannot start before its VIX warm-up ends, so the benchmarks
    are given the strategy's actual first and last dates.
    """
    same_dates = replace(
        config,
        start_date=strategy_history.index[0].strftime("%Y-%m-%d"),
        end_date=strategy_history.index[-1].strftime("%Y-%m-%d"),
    )
    return {
        BUY_AND_HOLD_NAME: run_constant_weight(market_data, 1.0, same_dates),
        fixed_allocation_name(fixed_nifty_weight): run_constant_weight(
            market_data, fixed_nifty_weight, same_dates
        ),
    }


def compare_performance(histories: dict[str, pd.DataFrame],
                        config: BacktestConfig) -> pd.DataFrame:
    """One row of headline metrics per portfolio, using the same metric functions."""
    rows = {
        name: summarise_performance(history, config.cash_annual_rate, config.initial_capital)
        for name, history in histories.items()
    }
    return pd.DataFrame(rows).T


def portfolio_values(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Portfolio value of each portfolio side by side (one column each)."""
    return pd.DataFrame({name: history["portfolio_value"] for name, history in histories.items()})
