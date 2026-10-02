import numpy as np
import pandas as pd
import pytest

from src.strategy.allocation import add_target_allocation, target_nifty_weight
from src.strategy.config import REGIME_LABELS, AllocationConfig, BacktestConfig


def regimes(labels):
    return pd.Series(pd.Categorical(labels, categories=REGIME_LABELS, ordered=True))


def test_default_mapping():
    weights = target_nifty_weight(regimes(["low", "normal", "high", "extreme"]),
                                  AllocationConfig())
    assert weights.tolist() == [1.0, 0.75, 0.5, 0.25]


def test_custom_mapping_and_warm_up_nan():
    config = AllocationConfig(low=0.9, normal=0.6, high=0.3, extreme=0.0)
    weights = target_nifty_weight(regimes([None, "extreme", "low"]), config)
    assert np.isnan(weights.iloc[0])
    assert weights.iloc[1:].tolist() == [0.0, 0.9]


def test_cash_weight_is_remainder():
    data = pd.DataFrame({"vix_regime": regimes(["low", "high"])})
    result = add_target_allocation(data, AllocationConfig())
    assert result["target_cash_weight"].tolist() == [0.0, 0.5]
    assert "target_nifty_weight" not in data.columns  # input not modified


@pytest.mark.parametrize("weight", [-0.1, 1.5])
def test_allocation_outside_zero_one_rejected(weight):
    with pytest.raises(ValueError):
        AllocationConfig(high=weight)


@pytest.mark.parametrize("settings", [
    {"rebalance_frequency": "hourly"},
    {"initial_capital": 0},
    {"transaction_cost_rate": -0.001},
    {"drift_tolerance": 2},
    {"execution_lag_days": -1},
])
def test_invalid_backtest_config_rejected(settings):
    with pytest.raises(ValueError):
        BacktestConfig(**settings)
