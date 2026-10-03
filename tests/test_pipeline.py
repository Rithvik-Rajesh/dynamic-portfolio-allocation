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


def test_full_backtest_has_no_look_ahead():
    """End-to-end check: if everything after a cut-off date is replaced, every
    row of the backtest up to the cut-off (signal, trades, portfolio value)
    must be unchanged."""
    market = synthetic_market(days=600)
    settings = (RegimeConfig(min_history=100), AllocationConfig(),
                BacktestConfig(rebalance_frequency="daily", drift_tolerance=0.0))
    cutoff = market.index[450]

    altered = market.copy()
    future = altered.index > cutoff
    altered.loc[future, "vix"] = 80.0
    altered.loc[future, "nifty"] = altered.loc[future, "nifty"] * 0.3
    altered.loc[future, "nifty_open"] = altered.loc[future, "nifty_open"] * 0.3
    altered["nifty_return"] = altered["nifty"].pct_change()

    original = run_vix_strategy(market, *settings)
    changed = run_vix_strategy(altered, *settings)

    pd.testing.assert_frame_equal(original.loc[:cutoff], changed.loc[:cutoff])
    assert not original.loc[cutoff:].equals(changed.loc[cutoff:])  # the change did matter
