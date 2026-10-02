"""Strategy and backtest assumptions, kept in one place so they are easy to
find, change and test.

- RegimeConfig      how VIX percentiles become regimes        (Milestone 3)
- AllocationConfig  how regimes become a target NIFTY weight   (Milestone 4)
- BacktestConfig    how the portfolio is traded and costed     (Milestone 5)

All defaults are illustrative starting points, not "correct" values.
"""

from dataclasses import dataclass

REGIME_LABELS = ["low", "normal", "high", "extreme"]
REBALANCE_FREQUENCIES = ["daily", "weekly", "monthly"]


@dataclass(frozen=True)
class RegimeConfig:
    """How a VIX percentile (0-1) is mapped to a regime.

    percentile <  low_threshold                      -> low
    low_threshold     <= percentile < high_threshold -> normal
    high_threshold    <= percentile < extreme_threshold -> high
    extreme_threshold <= percentile                  -> extreme

    min_history: number of trading days of VIX history required before a
    percentile (and therefore a regime) is produced. Earlier days have no
    regime. 252 trading days is about one year.
    """

    low_threshold: float = 0.25
    high_threshold: float = 0.75
    extreme_threshold: float = 0.95
    min_history: int = 252

    def __post_init__(self):
        if not 0 < self.low_threshold < self.high_threshold < self.extreme_threshold < 1:
            raise ValueError(
                "Thresholds must satisfy 0 < low < high < extreme < 1, got "
                f"low={self.low_threshold}, high={self.high_threshold}, "
                f"extreme={self.extreme_threshold}"
            )
        if self.min_history < 1:
            raise ValueError(f"min_history must be at least 1, got {self.min_history}")


@dataclass(frozen=True)
class AllocationConfig:
    """Target NIFTY weight (0-1) for each VIX regime. The rest is held in cash.

    No leverage and no short selling: every weight must be between 0 and 1.
    """

    low: float = 1.00
    normal: float = 0.75
    high: float = 0.50
    extreme: float = 0.25

    def __post_init__(self):
        for regime in REGIME_LABELS:
            weight = getattr(self, regime)
            if not 0 <= weight <= 1:
                raise ValueError(f"Allocation for '{regime}' must be between 0 and 1, got {weight}")

    def as_dict(self) -> dict[str, float]:
        return {regime: getattr(self, regime) for regime in REGIME_LABELS}


@dataclass(frozen=True)
class BacktestConfig:
    """How the portfolio is simulated.

    initial_capital        starting portfolio value in rupees.
    rebalance_frequency    'daily', 'weekly' or 'monthly'. Trades can only happen
                           on the first trading day of each period.
    drift_tolerance        on a rebalance day, trade if the target weight changed
                           OR the actual NIFTY weight is more than this many
                           weight points away from target (0.05 = 5 points).
    transaction_cost_rate  cost as a fraction of traded value (0.001 = 0.10%).
                           A single simplified rate for brokerage, STT, exchange
                           fees, GST, stamp duty and slippage; real Indian costs
                           differ by broker, instrument (ETF/futures) and size.
    cash_annual_rate       annual return earned on cash (0.06 = 6% p.a.),
                           approximating Indian T-bill / liquid-fund yields over
                           2015-2025. Also used as the risk-free rate in
                           Sharpe and Sortino ratios.
    execution_lag_days     trading days between the signal (VIX close) and the
                           trade. 1 = the regime known at close t is traded at
                           the close of t+1.
    """

    initial_capital: float = 100_000.0
    rebalance_frequency: str = "weekly"
    drift_tolerance: float = 0.05
    transaction_cost_rate: float = 0.001
    cash_annual_rate: float = 0.06
    execution_lag_days: int = 1

    def __post_init__(self):
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.rebalance_frequency not in REBALANCE_FREQUENCIES:
            raise ValueError(
                f"rebalance_frequency must be one of {REBALANCE_FREQUENCIES}, "
                f"got '{self.rebalance_frequency}'"
            )
        if not 0 <= self.drift_tolerance <= 1:
            raise ValueError("drift_tolerance must be between 0 and 1")
        if not 0 <= self.transaction_cost_rate < 0.1:
            raise ValueError("transaction_cost_rate must be between 0 and 0.1")
        if not -0.05 <= self.cash_annual_rate <= 0.5:
            raise ValueError("cash_annual_rate looks implausible (expected -5% to 50%)")
        if self.execution_lag_days < 0:
            raise ValueError("execution_lag_days cannot be negative")
