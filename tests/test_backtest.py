import numpy as np
import pandas as pd
import pytest

from src.backtesting.costs import transaction_cost
from src.backtesting.engine import executable_target, rebalance_days, run_backtest
from src.backtesting.portfolio import needs_rebalance, rebalance
from src.rates import period_rate, risk_free_returns
from src.strategy.config import BacktestConfig


def market(nifty_prices, start="2024-01-01"):
    """Market data on consecutive business days."""
    dates = pd.bdate_range(start, periods=len(nifty_prices))
    nifty = pd.Series(nifty_prices, index=dates, dtype=float)
    return pd.DataFrame({"nifty": nifty, "nifty_return": nifty.pct_change()})


def no_frictions(**overrides):
    settings = dict(initial_capital=1000.0, rebalance_frequency="daily", drift_tolerance=0.0,
                    transaction_cost_rate=0.0, cash_annual_rate=0.0, execution_lag_days=1)
    settings.update(overrides)
    return BacktestConfig(**settings)


# --- Costs and rebalancing arithmetic ---------------------------------------

def test_transaction_cost_is_symmetric():
    assert transaction_cost(1000, 0.001) == pytest.approx(1.0)
    assert transaction_cost(-1000, 0.001) == pytest.approx(1.0)


@pytest.mark.parametrize("nifty_value, cash, target", [
    (0.0, 1000.0, 1.0),     # initial full investment
    (0.0, 1000.0, 0.75),
    (750.0, 250.0, 0.25),   # sell
    (500.0, 500.0, 0.5),    # already on target
])
def test_rebalance_hits_target_exactly_after_costs(nifty_value, cash, target):
    rate = 0.001
    new_nifty, new_cash, trade, cost = rebalance(nifty_value, cash, target, rate)

    assert cost == pytest.approx(abs(trade) * rate)
    assert new_nifty / (new_nifty + new_cash) == pytest.approx(target)
    assert new_nifty + new_cash == pytest.approx(nifty_value + cash - cost)
    assert new_cash >= 0


def test_needs_rebalance_rules():
    assert needs_rebalance(0.0, 0.75, None, 0.05)        # first trade always
    assert needs_rebalance(0.75, 0.5, 0.75, 0.05)        # target changed
    assert not needs_rebalance(0.78, 0.75, 0.75, 0.05)   # small drift
    assert needs_rebalance(0.81, 0.75, 0.75, 0.05)       # drift beyond band


# --- Schedule and timing ----------------------------------------------------

def test_weekly_and_monthly_rebalance_days():
    dates = pd.bdate_range("2024-01-29", "2024-02-09")  # Mon 29 Jan .. Fri 9 Feb
    weekly = rebalance_days(dates, "weekly")
    monthly = rebalance_days(dates, "monthly")

    assert list(weekly[weekly].index.strftime("%Y-%m-%d")) == ["2024-01-29", "2024-02-05"]
    assert list(monthly[monthly].index.strftime("%Y-%m-%d")) == ["2024-01-29", "2024-02-01"]
    assert rebalance_days(dates, "daily").all()


def test_weekly_rebalance_uses_first_trading_day_after_holiday():
    # Monday 2024-01-01 missing (holiday) -> Tuesday is the first day of that week.
    dates = pd.DatetimeIndex(["2023-12-29", "2024-01-02", "2024-01-03"])
    assert rebalance_days(dates, "weekly").tolist() == [True, True, False]


def test_execution_lag_shifts_signal():
    signal = pd.Series([1.0, 0.5, 0.25])
    assert executable_target(signal, 1).tolist()[1:] == [1.0, 0.5]
    assert executable_target(signal, 0).tolist() == [1.0, 0.5, 0.25]


def test_signal_does_not_earn_the_return_of_its_own_day():
    """A crash on day 2 lifts VIX on day 2. With a 1-day lag the strategy can
    only cut exposure at day 3's close, so it must take the full day-2 and
    day-3 losses at its old weight."""
    data = market([100, 50, 25, 25])
    target = pd.Series([1.0, 0.0, 0.0, 0.0], index=data.index)

    history = run_backtest(data, target, no_frictions())

    # Start day 1 (lagged target from day 0 = 100%), fully invested at 100.
    assert history.index[0] == data.index[1]
    assert history["nifty_weight"].iloc[0] == pytest.approx(1.0)
    # Day 2: NIFTY 50 -> 25, still fully invested; sells at the close.
    assert history["portfolio_value"].iloc[1] == pytest.approx(500.0)
    assert history["traded"].iloc[1]
    assert history["nifty_weight"].iloc[1] == pytest.approx(0.0)


def test_same_day_execution_when_lag_is_zero():
    data = market([100, 100, 50])
    target = pd.Series([1.0, 0.0, 0.0], index=data.index)
    history = run_backtest(data, target, no_frictions(execution_lag_days=0))
    # Sold at day 1's close, so the day-2 crash is avoided.
    assert history["portfolio_value"].iloc[-1] == pytest.approx(1000.0)


# --- Whole-engine behaviour -------------------------------------------------

def test_full_investment_tracks_nifty():
    data = market([100, 110, 99, 120])
    target = pd.Series(1.0, index=data.index)
    history = run_backtest(data, target, no_frictions())
    # Invested at 110 on the first backtest day, so value tracks NIFTY / 110.
    expected = 1000 * data["nifty"].iloc[1:] / 110
    np.testing.assert_allclose(history["portfolio_value"], expected)


def test_all_cash_earns_cash_rate():
    data = market([100, 90, 80, 70, 60], start="2024-01-01")
    target = pd.Series(0.0, index=data.index)
    history = run_backtest(data, target, no_frictions(cash_annual_rate=0.06))

    days = (history.index[-1] - history.index[0]).days
    assert history["portfolio_value"].iloc[-1] == pytest.approx(1000 * 1.06 ** (days / 365))


def test_drift_band_and_weekly_schedule_limit_trading():
    data = market([100] * 10)  # flat market: no drift
    target = pd.Series(0.75, index=data.index)
    history = run_backtest(data, target, no_frictions(rebalance_frequency="weekly",
                                                      drift_tolerance=0.05))
    assert history["traded"].sum() == 1  # only the initial investment


def test_costs_reduce_portfolio_value():
    data = market([100, 100, 100])
    target = pd.Series(1.0, index=data.index)
    history = run_backtest(data, target, no_frictions(transaction_cost_rate=0.001))
    assert history["transaction_cost"].iloc[0] == pytest.approx(1000 / 1.001 * 0.001)
    assert history["portfolio_value"].iloc[0] == pytest.approx(1000 / 1.001)
    assert history["portfolio_return"].iloc[0] == pytest.approx(1 / 1.001 - 1)


def test_backtest_is_deterministic():
    rng = np.random.default_rng(1)
    data = market(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)))
    target = pd.Series(rng.choice([0.25, 0.5, 0.75, 1.0], 300), index=data.index)
    config = BacktestConfig(rebalance_frequency="weekly")
    pd.testing.assert_frame_equal(run_backtest(data, target, config),
                                  run_backtest(data, target, config))


def test_missing_target_after_start_raises():
    data = market([100, 101, 102, 103])
    target = pd.Series([1.0, 1.0, np.nan, 1.0], index=data.index)
    with pytest.raises(ValueError):
        run_backtest(data, target, no_frictions())


# --- Rates ------------------------------------------------------------------

def test_rates_compound_to_annual_rate():
    assert period_rate(0.06, 365) == pytest.approx(0.06)
    dates = pd.DatetimeIndex(["2024-01-05", "2024-01-08"])  # Friday -> Monday
    assert risk_free_returns(dates, 0.06).tolist() == pytest.approx([0.0, 1.06 ** (3 / 365) - 1])
