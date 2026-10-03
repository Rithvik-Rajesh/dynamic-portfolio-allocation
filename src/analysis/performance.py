"""Performance metrics for any backtested portfolio.

Every metric is a small function on pandas Series, and `summarise_performance`
combines them for a backtest history (as produced by
`src.backtesting.engine.run_backtest`). Strategy and benchmarks must use these
same functions so their numbers are comparable.

Conventions
-----------
- Returns are daily simple returns.
- Annualisation uses 252 trading days for volatility-style figures and actual
  calendar time (365.25-day years) for CAGR.
- The risk-free rate is the same annual rate the backtest pays on cash,
  converted per day with `src.rates`, so a 100% cash portfolio has an excess
  return of exactly zero.
"""

import numpy as np
import pandas as pd

from src.analysis.risk import (
    annualised_volatility,
    downside_deviation,
    max_drawdown_details,
)
from src.config import TRADING_DAYS_PER_YEAR
from src.rates import risk_free_returns

DAYS_PER_YEAR = 365.25


# --- Returns -----------------------------------------------------------------

def total_return(returns: pd.Series) -> float:
    """Compounded return over the whole period."""
    return float((1 + returns).prod() - 1)


def years_between(dates: pd.DatetimeIndex) -> float:
    """Length of the period in years (calendar time)."""
    return (dates[-1] - dates[0]).days / DAYS_PER_YEAR


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate: (1 + total return)^(1 / years) - 1."""
    years = years_between(returns.index)
    if years <= 0:
        return np.nan
    return float((1 + total_return(returns)) ** (1 / years) - 1)


def calendar_year_returns(returns: pd.Series) -> pd.Series:
    """Compounded return within each calendar year (partial years included)."""
    yearly = (1 + returns).groupby(returns.index.year).prod() - 1
    yearly.index.name = "year"
    return yearly


# --- Risk-adjusted -----------------------------------------------------------

def excess_returns(returns: pd.Series, risk_free_annual_rate: float) -> pd.Series:
    """Daily return minus the daily risk-free return."""
    return returns - risk_free_returns(returns.index, risk_free_annual_rate)


def sharpe_ratio(returns: pd.Series, risk_free_annual_rate: float) -> float:
    """Annualised Sharpe ratio: mean(excess) / std(excess) x sqrt(252)."""
    excess = excess_returns(returns, risk_free_annual_rate)
    if excess.std() == 0:
        return np.nan
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series, risk_free_annual_rate: float) -> float:
    """Annualised Sortino ratio: mean(excess) / downside deviation x sqrt(252)."""
    excess = excess_returns(returns, risk_free_annual_rate)
    downside = downside_deviation(excess)
    if downside == 0:
        return np.nan
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS_PER_YEAR))


# --- Trading -----------------------------------------------------------------

def number_of_trades(history: pd.DataFrame) -> int:
    """Days on which a trade happened (includes the initial investment)."""
    return int(history["traded"].sum())


def annual_turnover(history: pd.DataFrame) -> float:
    """Average yearly traded value as a multiple of average portfolio value.

    1.0 means trading the equivalent of the whole portfolio once a year.
    Includes the initial investment.
    """
    traded_value = history["trade_value"].abs().sum()
    average_value = history["portfolio_value"].mean()
    years = years_between(history.index)
    if years <= 0:
        return np.nan
    return float(traded_value / average_value / years)


def total_transaction_costs(history: pd.DataFrame) -> float:
    """Sum of all transaction costs paid, in rupees."""
    return float(history["transaction_cost"].sum())


# --- Summary -----------------------------------------------------------------

def summarise_performance(history: pd.DataFrame, risk_free_annual_rate: float,
                          initial_capital: float) -> dict:
    """All headline metrics for one backtest history."""
    returns = history["portfolio_return"]
    drawdown = max_drawdown_details(history["portfolio_value"])
    costs = total_transaction_costs(history)

    return {
        "start_date": history.index[0].date(),
        "end_date": history.index[-1].date(),
        "final_value": float(history["portfolio_value"].iloc[-1]),
        "total_return": total_return(returns),
        "cagr": cagr(returns),
        "annualised_volatility": annualised_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_annual_rate),
        "sortino_ratio": sortino_ratio(returns, risk_free_annual_rate),
        "max_drawdown": drawdown["max_drawdown"],
        "max_drawdown_peak": drawdown["peak_date"].date(),
        "max_drawdown_trough": drawdown["trough_date"].date(),
        "max_drawdown_recovery": (drawdown["recovery_date"].date()
                                  if drawdown["recovery_date"] is not None else None),
        "number_of_trades": number_of_trades(history),
        "annual_turnover": annual_turnover(history),
        "total_transaction_costs": costs,
        "costs_pct_of_initial_capital": costs / initial_capital,
        "average_nifty_weight": float(history["nifty_weight"].mean()),
    }
