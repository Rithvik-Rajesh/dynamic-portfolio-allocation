"""Allocation rule: convert each day's VIX regime into a target portfolio.

This module only decides WHAT the portfolio should hold. When the target is
traded (execution lag, rebalancing schedule) is handled by the backtest engine.
"""

import pandas as pd

from src.strategy.config import AllocationConfig


def target_nifty_weight(vix_regime: pd.Series, config: AllocationConfig) -> pd.Series:
    """Target NIFTY weight for each day. Days without a regime (warm-up) get NaN."""
    weights = vix_regime.map(config.as_dict()).astype(float)
    return weights.rename("target_nifty_weight")


def add_target_allocation(regime_data: pd.DataFrame, config: AllocationConfig) -> pd.DataFrame:
    """Return a copy with `target_nifty_weight` and `target_cash_weight` added."""
    regime_data = regime_data.copy()
    regime_data["target_nifty_weight"] = target_nifty_weight(regime_data["vix_regime"], config)
    regime_data["target_cash_weight"] = 1 - regime_data["target_nifty_weight"]
    return regime_data
