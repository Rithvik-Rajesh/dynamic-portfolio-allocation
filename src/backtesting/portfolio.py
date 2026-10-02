"""Portfolio arithmetic: rebalancing a NIFTY + cash portfolio to a target weight.

The portfolio holds two things:
    nifty_value  rupees invested in NIFTY 50 (moves with the index)
    cash         rupees in cash (earns the cash rate)

Transaction costs are paid from cash.
"""

from src.backtesting.costs import transaction_cost


def nifty_weight(nifty_value: float, cash: float) -> float:
    """Current share of the portfolio held in NIFTY."""
    return nifty_value / (nifty_value + cash)


def rebalance_trade_value(nifty_value: float, cash: float, target_weight: float,
                          cost_rate: float) -> float:
    """Rupee amount of NIFTY to buy (+) or sell (-) to reach `target_weight`.

    The cost of the trade reduces the portfolio, so the trade is sized so that
    the weight is exactly on target AFTER paying the cost:

        new_value = value - cost
        target_weight x new_value = nifty_value + trade
        cost = |trade| x cost_rate

    Solving for the trade gives:
        buy  (trade > 0): trade = (w x V - N) / (1 + w x rate)
        sell (trade < 0): trade = (w x V - N) / (1 - w x rate)
    """
    portfolio_value = nifty_value + cash
    gap = target_weight * portfolio_value - nifty_value
    if gap >= 0:
        return gap / (1 + target_weight * cost_rate)
    return gap / (1 - target_weight * cost_rate)


def rebalance(nifty_value: float, cash: float, target_weight: float,
              cost_rate: float) -> tuple[float, float, float, float]:
    """Trade to `target_weight`.

    Returns (new_nifty_value, new_cash, trade_value, cost).
    """
    trade_value = rebalance_trade_value(nifty_value, cash, target_weight, cost_rate)
    cost = transaction_cost(trade_value, cost_rate)
    new_nifty_value = nifty_value + trade_value
    new_cash = cash - trade_value - cost
    if abs(new_cash) < 1e-6:  # remove floating-point dust (e.g. -3e-11) at 100% NIFTY
        new_cash = 0.0
    return new_nifty_value, new_cash, trade_value, cost


def needs_rebalance(current_weight: float, target_weight: float, previous_target: float | None,
                    drift_tolerance: float) -> bool:
    """Decide whether to trade on a rebalance day.

    Trade if the target changed since the last trade (e.g. new regime), or if
    market moves pushed the actual weight further than `drift_tolerance` from
    the target. The first trade of a backtest (previous_target is None) always
    happens.
    """
    if previous_target is None or target_weight != previous_target:
        return True
    return abs(current_weight - target_weight) > drift_tolerance
