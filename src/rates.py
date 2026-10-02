"""Interest-rate helpers shared by the backtest (cash growth) and the
performance metrics (risk-free rate), so both use exactly the same numbers.

Cash accrues interest for every CALENDAR day between two trading days, so a
Monday includes Saturday and Sunday. Over a full year this compounds to
exactly the stated annual rate.
"""

import pandas as pd

CALENDAR_DAYS_PER_YEAR = 365


def period_rate(annual_rate: float, calendar_days: float) -> float:
    """Compounded return of `annual_rate` over a number of calendar days."""
    return (1 + annual_rate) ** (calendar_days / CALENDAR_DAYS_PER_YEAR) - 1


def risk_free_returns(dates: pd.DatetimeIndex, annual_rate: float) -> pd.Series:
    """Cash / risk-free return for each date, measured since the previous date.

    The first date has a return of 0 (no time has passed).
    """
    calendar_gaps = dates.to_series().diff().dt.days.fillna(0)
    return calendar_gaps.apply(lambda days: period_rate(annual_rate, days))
