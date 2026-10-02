import numpy as np
import pandas as pd
import pytest

from src.analysis.performance import (
    annual_turnover,
    cagr,
    sharpe_ratio,
    sortino_ratio,
    summarise_performance,
    total_return,
)
from src.analysis.risk import (
    annualised_volatility,
    downside_deviation,
    drawdown_series,
    max_drawdown,
    max_drawdown_details,
)


def series(values, start="2024-01-01", freq="B"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq),
                     dtype=float)


# --- Returns -----------------------------------------------------------------

def test_total_return_compounds():
    assert total_return(series([0.10, -0.10])) == pytest.approx(1.1 * 0.9 - 1)


def test_cagr_over_two_calendar_years():
    # 21% total over exactly 731 days (2020-01-01 -> 2022-01-01) = 2.0014 years
    returns = pd.Series([0.0, 0.21], index=pd.to_datetime(["2020-01-01", "2022-01-01"]))
    assert cagr(returns) == pytest.approx(1.21 ** (365.25 / 731) - 1)


# --- Risk --------------------------------------------------------------------

def test_annualised_volatility():
    returns = series([0.01, -0.01, 0.01, -0.01])
    assert annualised_volatility(returns) == pytest.approx(returns.std() * np.sqrt(252))


def test_drawdown_hand_calculated():
    values = series([100, 120, 90, 130, 104])
    assert drawdown_series(values).tolist() == pytest.approx([0, 0, -0.25, 0, -0.2])
    assert max_drawdown(values) == pytest.approx(-0.25)


def test_drawdown_details_and_recovery():
    values = series([100, 120, 90, 110, 125, 100])
    details = max_drawdown_details(values)
    assert details["max_drawdown"] == pytest.approx(-0.25)
    assert details["peak_date"] == values.index[1]
    assert details["trough_date"] == values.index[2]
    assert details["recovery_date"] == values.index[4]


def test_drawdown_without_recovery():
    assert max_drawdown_details(series([100, 80, 90]))["recovery_date"] is None


def test_rising_series_has_no_drawdown():
    assert max_drawdown(series([1, 2, 3])) == 0


def test_downside_deviation_ignores_gains():
    # Only -0.02 counts: sqrt(0.02^2 / 4) = 0.01
    assert downside_deviation(series([0.03, -0.02, 0.0, 0.05])) == pytest.approx(0.01)


# --- Risk-adjusted -----------------------------------------------------------

def test_sharpe_with_zero_risk_free():
    returns = series([0.01, 0.02, -0.01, 0.00])
    expected = returns.mean() / returns.std() * np.sqrt(252)
    assert sharpe_ratio(returns, 0.0) == pytest.approx(expected)


def test_sharpe_subtracts_risk_free_by_calendar_days():
    returns = series([0.0, 0.01, 0.01], start="2024-01-04")  # Thu, Fri, Mon
    excess = returns - pd.Series([0.0, 1.06 ** (1 / 365) - 1, 1.06 ** (3 / 365) - 1],
                                 index=returns.index)
    expected = excess.mean() / excess.std() * np.sqrt(252)
    assert sharpe_ratio(returns, 0.06) == pytest.approx(expected)


def test_sortino_hand_calculated():
    returns = series([0.03, -0.02, 0.0, 0.05])
    expected = returns.mean() / 0.01 * np.sqrt(252)
    assert sortino_ratio(returns, 0.0) == pytest.approx(expected)


def test_ratios_undefined_without_risk():
    flat = series([0.0, 0.0, 0.0])
    assert np.isnan(sharpe_ratio(flat, 0.0))
    assert np.isnan(sortino_ratio(flat, 0.0))


# --- Trading and summary -----------------------------------------------------

def make_history():
    index = pd.to_datetime(["2023-01-02", "2023-07-03", "2024-01-02"])
    return pd.DataFrame({
        "portfolio_value": [100.0, 100.0, 100.0],
        "portfolio_return": [0.0, 0.0, 0.0],
        "trade_value": [100.0, -50.0, 0.0],
        "transaction_cost": [0.1, 0.05, 0.0],
        "traded": [True, True, False],
        "nifty_weight": [1.0, 0.5, 0.5],
    }, index=index)


def test_annual_turnover():
    # 150 traded / 100 average value / 1.0 year (365 days / 365.25)
    assert annual_turnover(make_history()) == pytest.approx(1.5 * 365.25 / 365)


def test_summary_contains_all_metrics():
    summary = summarise_performance(make_history(), 0.0, initial_capital=100.0)
    assert summary["number_of_trades"] == 2
    assert summary["total_transaction_costs"] == pytest.approx(0.15)
    assert summary["costs_pct_of_initial_capital"] == pytest.approx(0.0015)
    for key in ["total_return", "cagr", "annualised_volatility", "sharpe_ratio",
                "sortino_ratio", "max_drawdown", "annual_turnover"]:
        assert key in summary
