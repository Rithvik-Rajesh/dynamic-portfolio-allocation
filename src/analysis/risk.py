"""Risk metrics: volatility, downside deviation and drawdowns.

All functions take plain pandas Series so they work for any portfolio
(strategy or benchmark).
"""

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS_PER_YEAR


def annualised_volatility(returns: pd.Series) -> float:
    """Standard deviation of daily returns x sqrt(252)."""
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def downside_deviation(excess_returns: pd.Series) -> float:
    """Daily downside deviation: sqrt(mean(min(excess return, 0)^2)).

    Only returns below the target (the risk-free rate) count as risk; all
    days are kept in the average.
    """
    downside = excess_returns.clip(upper=0)
    return float(np.sqrt((downside**2).mean()))


def drawdown_series(portfolio_values: pd.Series) -> pd.Series:
    """Percentage below the highest value seen so far (0 at a new peak, negative below)."""
    running_peak = portfolio_values.cummax()
    return (portfolio_values / running_peak - 1).rename("drawdown")


def max_drawdown(portfolio_values: pd.Series) -> float:
    """Largest peak-to-trough fall, as a negative number (e.g. -0.35 = -35%)."""
    return float(drawdown_series(portfolio_values).min())


def max_drawdown_details(portfolio_values: pd.Series) -> dict:
    """Peak, trough and recovery dates of the maximum drawdown.

    recovery_date is the first date the value regains the previous peak, or
    None if it never recovered within the data.
    """
    drawdown = drawdown_series(portfolio_values)
    trough_date = drawdown.idxmin()
    peak_date = portfolio_values.loc[:trough_date].idxmax()

    after_trough = portfolio_values.loc[trough_date:]
    recovered = after_trough[after_trough >= portfolio_values.loc[peak_date]]
    recovery_date = recovered.index[0] if not recovered.empty else None

    return {
        "max_drawdown": float(drawdown.min()),
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
    }
