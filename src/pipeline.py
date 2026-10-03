"""One function that runs the VIX strategy end to end for a given set of settings.

This is the single entry point for anything that wants to try different
settings (main.py now, the Streamlit dashboard and experiments later):

    market data -> VIX percentile & regime -> target allocation -> backtest

Every tuneable assumption is in the three config objects (see
`src/strategy/config.py`).
"""

import pandas as pd

from src.backtesting.engine import add_signal_columns, run_backtest
from src.strategy.allocation import add_target_allocation
from src.strategy.config import (
    AllocationConfig,
    BacktestConfig,
    RegimeConfig,
    StrategySettings,
)
from src.strategy.regimes import add_vix_regimes


def run_vix_strategy(market_data: pd.DataFrame, regime_config: RegimeConfig,
                     allocation_config: AllocationConfig,
                     backtest_config: BacktestConfig) -> pd.DataFrame:
    """Backtest the VIX strategy and return its daily history with signal columns.

    Regimes are calculated on the FULL market history before the backtest
    period is cut, so a later start date still has the VIX history that was
    available at the time.
    """
    regime_data = add_vix_regimes(market_data, regime_config)
    strategy_data = add_target_allocation(regime_data, allocation_config)
    history = run_backtest(strategy_data, strategy_data["target_nifty_weight"], backtest_config)
    return add_signal_columns(history, strategy_data)


def run_settings(market_data: pd.DataFrame, settings: StrategySettings) -> pd.DataFrame:
    """Same as `run_vix_strategy`, taking one StrategySettings bundle."""
    return run_vix_strategy(market_data, settings.regime, settings.allocation, settings.backtest)
