"""Backtest engine: simulate a NIFTY + cash portfolio day by day.

The engine is given a TARGET NIFTY weight for each day (from any rule: the VIX
strategy now, benchmarks later) and simulates trading it.

Timing (no look-ahead)
----------------------
The target for day t is based on the VIX close of day t. With
`execution_lag_days = 1` it is traded on day t+1:
    execution_price='close'  at the NIFTY close of t+1 (about 24 hours later)
    execution_price='open'   at the NIFTY open of t+1 (the next morning)
So a trade never benefits from the market move of the day it was decided.

Daily steps, for each trading day d
-----------------------------------
Trading at the close:
    1. Grow holdings over the day: NIFTY by close(d)/close(d-1), cash by the
       cash rate.
    2. On a rebalance day, trade at the close if the target changed or the
       weight drifted too far. Costs are paid from cash.
Trading at the open:
    1. Grow NIFTY overnight by open(d)/close(d-1); grow cash by the cash rate.
    2. On a rebalance day, trade at the open (same rule as above).
    3. Grow NIFTY during the day by close(d)/open(d).
Both: record the end-of-day portfolio.

Cash interest is credited once per day (for the calendar days since the
previous trading day), before any trade. When trading at the open this
ignores the few hours between the open and the close, which is negligible.

Backtest period
---------------
The backtest starts on the later of `config.start_date` and the first day a
lagged target exists, with all capital in cash; the initial investment is
the first trade. It ends on `config.end_date` (or the last date in the data).
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


def backtest_dates(trade_target: pd.Series, config: BacktestConfig) -> pd.DatetimeIndex:
    """Trading days of the backtest: from the later of start_date and the
    first tradable target, up to end_date."""
    first_signal_date = trade_target.first_valid_index()
    if first_signal_date is None:
        raise ValueError("No valid target weight: nothing to backtest.")

    start = first_signal_date
    if config.start_date is not None:
        start = max(start, pd.Timestamp(config.start_date))
    end = pd.Timestamp(config.end_date) if config.end_date is not None else trade_target.index[-1]

    dates = trade_target.loc[start:end].index
    if len(dates) < 2:
        raise ValueError(
            f"Backtest period {start.date()} to {end.date()} has fewer than 2 trading days. "
            f"The first date with a tradable signal is {first_signal_date.date()}."
        )
    return dates


def run_backtest(market_data: pd.DataFrame, target_nifty_weight: pd.Series,
                 config: BacktestConfig) -> pd.DataFrame:
    """Simulate the portfolio and return its full daily history.

    market_data must contain `nifty`, `nifty_return` and, when trading at the
    open, `nifty_open`. target_nifty_weight must share its index and may be
    NaN only before the first valid signal.

    History columns:
        tradable_target_weight  target that may be traded today, i.e. the
                                signal from `execution_lag_days` earlier
        rebalance_day / traded  whether trading was allowed / happened today
        trade_value             NIFTY bought (+) or sold (-), in rupees
        transaction_cost        cost paid today, in rupees
        nifty_value, cash       end-of-day holdings, in rupees
        portfolio_value         nifty_value + cash
        nifty_weight            actual end-of-day NIFTY weight
        portfolio_return        daily return (day one includes the initial
                                trading cost and, at the open, day one's
                                intraday move)
    """
    trade_target = executable_target(target_nifty_weight, config.execution_lag_days)
    dates = backtest_dates(trade_target, config)
    period = market_data.loc[dates]
    trade_target = trade_target.loc[dates]
    if trade_target.isna().any():
        raise ValueError("Target weight is missing after the backtest start date.")

    trade_at_open = config.execution_price == "open"
    close = period["nifty"]
    previous_close = market_data["nifty"].shift(1).loc[dates]
    open_price = period["nifty_open"] if trade_at_open else None

    is_rebalance_day = rebalance_days(dates, config.rebalance_frequency)
    cash_returns = risk_free_returns(dates, config.cash_annual_rate)

    nifty_value = 0.0
    cash = config.initial_capital
    previous_target = None
    rows = []

    for i, date in enumerate(dates):
        # 1. Market moves up to the moment of trading (none before the first trade).
        if i > 0:
            trade_price = open_price.iloc[i] if trade_at_open else close.iloc[i]
            nifty_value *= trade_price / previous_close.iloc[i]
            cash *= 1 + cash_returns.iloc[i]

        # 2. Trade if allowed today and needed.
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

        # 3. When trading at the open, the rest of the day happens after the trade.
        if trade_at_open:
            nifty_value *= close.iloc[i] / open_price.iloc[i]

        # 4. Record the end-of-day portfolio.
        rows.append({
            "date": date,
            "nifty": close.iloc[i],
            "nifty_return": period["nifty_return"].iloc[i] if i > 0 else 0.0,
            "tradable_target_weight": target,
            "rebalance_day": bool(is_rebalance_day.iloc[i]),
            "traded": traded,
            "trade_value": trade_value,
            "transaction_cost": cost,
            "nifty_value": nifty_value,
            "cash": cash,
            "portfolio_value": nifty_value + cash,
            "nifty_weight": nifty_weight(nifty_value, cash),
        })

    history = pd.DataFrame(rows).set_index("date")
    previous_value = history["portfolio_value"].shift(1).fillna(config.initial_capital)
    history["portfolio_return"] = history["portfolio_value"] / previous_value - 1
    return history


def add_signal_columns(history: pd.DataFrame, regime_data: pd.DataFrame) -> pd.DataFrame:
    """Attach the VIX signal columns (as known on each date) for inspection."""
    signal_columns = ["vix", "vix_percentile", "vix_regime"]
    return regime_data.loc[history.index, signal_columns].join(history)
