"""Transaction-cost model: a fixed percentage of traded value.

    transaction cost = |trade value| x cost rate

Buying and selling cost the same. The rate is set in BacktestConfig.
"""


def transaction_cost(trade_value: float, cost_rate: float) -> float:
    """Cost of a trade. `trade_value` is positive for buys, negative for sells."""
    return abs(trade_value) * cost_rate
