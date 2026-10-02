"""Descriptive analysis of VIX regimes: how often each regime occurs and how
NIFTY behaved afterwards.

This is research/description only. It looks at returns AFTER each day's
regime is known, which is fine for describing history, but it must never be
used to choose a day's allocation.
"""

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS_PER_YEAR


def regime_frequency(regime_data: pd.DataFrame) -> pd.DataFrame:
    """Number and share of trading days in each regime (warm-up days excluded)."""
    counts = regime_data["vix_regime"].value_counts(sort=False)
    return pd.DataFrame({"days": counts, "share": counts / counts.sum()})


def add_next_day_return(regime_data: pd.DataFrame) -> pd.DataFrame:
    """Add the NIFTY return of the FOLLOWING trading day.

    A regime known at the close of day t can only affect returns from
    day t+1, so next-day return is the relevant one for analysis.
    """
    regime_data = regime_data.copy()
    regime_data["next_day_nifty_return"] = regime_data["nifty_return"].shift(-1)
    return regime_data


def nifty_behaviour_by_regime(regime_data: pd.DataFrame) -> pd.DataFrame:
    """Summary of next-day NIFTY returns for each regime.

    Columns:
        days                    number of observations
        mean_daily_return       average next-day return
        daily_volatility        standard deviation of next-day returns
        annualised_volatility   daily_volatility * sqrt(252)
        share_negative_days     fraction of next days with a negative return
        worst_day / best_day    smallest / largest next-day return
    """
    data = add_next_day_return(regime_data).dropna(
        subset=["vix_regime", "next_day_nifty_return"]
    )
    grouped = data.groupby("vix_regime", observed=False)["next_day_nifty_return"]

    summary = pd.DataFrame(
        {
            "days": grouped.count(),
            "mean_daily_return": grouped.mean(),
            "daily_volatility": grouped.std(),
            "share_negative_days": grouped.apply(lambda returns: (returns < 0).mean()),
            "worst_day": grouped.min(),
            "best_day": grouped.max(),
        }
    )
    summary.insert(
        3,
        "annualised_volatility",
        summary["daily_volatility"] * np.sqrt(TRADING_DAYS_PER_YEAR),
    )
    return summary


def regime_changes(regime_data: pd.DataFrame) -> int:
    """Number of days on which the regime differs from the previous day's regime."""
    regime = regime_data["vix_regime"].dropna()
    return int((regime != regime.shift()).iloc[1:].sum())
