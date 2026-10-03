import numpy as np
import pandas as pd

from src.pipeline import run_vix_strategy
from src.strategy.config import AllocationConfig, BacktestConfig, RegimeConfig


def synthetic_market(days=600, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=days)
    nifty = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.01, days)), index=dates)
    return pd.DataFrame({
        "vix": rng.uniform(10, 40, days),
        "nifty_open": nifty.shift(1).fillna(100) * (1 + rng.normal(0, 0.002, days)),
        "nifty": nifty,
        "nifty_return": nifty.pct_change(),
    }, index=dates)


def test_later_start_date_keeps_earlier_vix_history():
    """Starting to invest later must not throw away the VIX history known by then."""
    market = synthetic_market()
    regime_config = RegimeConfig(min_history=100)

    full = run_vix_strategy(market, regime_config, AllocationConfig(), BacktestConfig())
    late = run_vix_strategy(market, regime_config, AllocationConfig(),
                            BacktestConfig(start_date="2019-06-03"))

    assert late.index[0] == pd.Timestamp("2019-06-03")
    pd.testing.assert_series_equal(late["vix_percentile"],
                                   full.loc[late.index, "vix_percentile"])


def test_every_setting_combination_runs():
    market = synthetic_market()
    for window in ["expanding", "rolling"]:
        for frequency in ["daily", "weekly", "monthly"]:
            for price in ["close", "open"]:
                history = run_vix_strategy(
                    market,
                    RegimeConfig(min_history=60, percentile_window=window, rolling_window=120),
                    AllocationConfig(),
                    BacktestConfig(rebalance_frequency=frequency, execution_price=price,
                                   initial_capital=5_000, drift_tolerance=0.1),
                )
                assert history["portfolio_value"].iloc[0] > 0
                assert (history["cash"] >= 0).all()
                assert history["nifty_weight"].between(0, 1).all()
