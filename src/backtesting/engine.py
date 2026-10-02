"""Backtest engine: simulate a NIFTY + cash portfolio day by day.

The engine is given a TARGET NIFTY weight for each day (from any rule: the VIX
strategy now, benchmarks later) and simulates trading it.

Timing (no look-ahead)
----------------------
The target for day t is based on the VIX close of day t. With
`execution_lag_days = 1`, it is traded at the NIFTY close of day t+1 and the
new holdings earn returns from day t+2. So a trade never benefits from the
market move that happened on the day it was decided.

Daily steps, for each trading day d:
    1. Grow holdings: NIFTY by the day's NIFTY return, cash by the cash rate.
    2. If d is a rebalance day, read the (lagged) target weight.
    3. If the target changed, or the weight drifted too far, trade at the close.
       Costs are paid from cash.
    4. Record the portfolio.

The backtest starts on the first day a lagged target exists, with all capital
in cash; the initial investment is traded at that day's close.
"""

import pandas as pd

from src.backtesting.portfolio import needs_rebalance, nifty_weight, rebalance
from src.rates import risk_free_returns
from src.strategy.config import BacktestConfig


def rebalance_days(dates: pd.DatetimeIndex, frequency: str) -> pd.Series:
    """True on the trading days when trading is allowed.

    daily   every trading day
    weekly  first trading day of each calendar week
    monthly first trading day of each calendar month
    """
    if frequency == "daily":
        return pd.Series(True, index=dates)

    period_code = {"weekly": "W", "monthly": "M"}[frequency]
    # Compare each date's period with the PREVIOUS ROW's period. (Note:
    # PeriodIndex.shift would move every period forward in time instead.)
    periods = pd.Series(dates.to_period(period_code), index=dates)
    return periods != periods.shift(1)  # first row compares with NaN -> True


def executable_target(target_nifty_weight: pd.Series, execution_lag_days: int) -> pd.Series:
    """The target weight that may be traded on each day (signal shifted by the lag)."""
    return target_nifty_weight.shift(execution_lag_days)


def run_backtest(market_data: pd.DataFrame, target_nifty_weight: pd.Series,
                 config: BacktestConfig) -> pd.DataFrame:
    """Simulate the portfolio and return its full daily history.

    market_data must contain `nifty` and `nifty_return`. target_nifty_weight
    must share its index and may be NaN only before the first valid signal.

    History columns:
        tradable_target_weight  target that may be traded today, i.e. the
                                signal from `execution_lag_days` earlier
        rebalance_day / traded  whether trading was allowed / happened today
        trade_value             NIFTY bought (+) or sold (-), in rupees
        transaction_cost        cost paid today, in rupees
        nifty_value, cash       end-of-day holdings, in rupees
        portfolio_value         nifty_value + cash
        nifty_weight            actual end-of-day NIFTY weight
        portfolio_return        daily return (day one = initial trading cost)
    """
    trade_target = executable_target(target_nifty_weight, config.execution_lag_days)
    start_date = trade_target.first_valid_index()
    if start_date is None:
        raise ValueError("No valid target weight: nothing to backtest.")

    period = market_data.loc[start_date:]
    trade_target = trade_target.loc[start_date:]
    if trade_target.isna().any():
        raise ValueError("Target weight is missing after the backtest start date.")

    is_rebalance_day = rebalance_days(period.index, config.rebalance_frequency)
    cash_returns = risk_free_returns(period.index, config.cash_annual_rate)

    nifty_value = 0.0
    cash = config.initial_capital
    previous_target = None
    rows = []

    for i, date in enumerate(period.index):
        # 1. Market moves since yesterday's close (none on the first day).
        if i > 0:
            nifty_value *= 1 + period["nifty_return"].iloc[i]
            cash *= 1 + cash_returns.iloc[i]
        value_before_trade = nifty_value + cash

        # 2-3. Trade at today's close if allowed and needed.
        target = trade_target.iloc[i]
        rebalance_allowed = bool(is_rebalance_day.iloc[i]) or previous_target is None
        trade_value = cost = 0.0
        traded = False
        if rebalance_allowed and needs_rebalance(nifty_weight(nifty_value, cash), target,
                                                 previous_target, config.drift_tolerance):
            nifty_value, cash, trade_value, cost = rebalance(
                nifty_value, cash, target, config.transaction_cost_rate
            )
            previous_target = target
            traded = True

        # 4. Record the end-of-day portfolio.
        portfolio_value = nifty_value + cash
        rows.append({
            "date": date,
            "nifty": period["nifty"].iloc[i],
            "nifty_return": period["nifty_return"].iloc[i] if i > 0 else 0.0,
            "tradable_target_weight": target,
            "rebalance_day": bool(is_rebalance_day.iloc[i]),
            "traded": traded,
            "trade_value": trade_value,
            "transaction_cost": cost,
            "nifty_value": nifty_value,
            "cash": cash,
            "portfolio_value": portfolio_value,
            "nifty_weight": nifty_weight(nifty_value, cash),
            "value_before_trade": value_before_trade,
        })

    history = pd.DataFrame(rows).set_index("date")
    previous_value = history["portfolio_value"].shift(1).fillna(config.initial_capital)
    history["portfolio_return"] = history["portfolio_value"] / previous_value - 1
    return history.drop(columns="value_before_trade")


def add_signal_columns(history: pd.DataFrame, regime_data: pd.DataFrame) -> pd.DataFrame:
    """Attach the VIX signal columns (as known on each date) for inspection."""
    signal_columns = ["vix", "vix_percentile", "vix_regime"]
    return regime_data.loc[history.index, signal_columns].join(history)
