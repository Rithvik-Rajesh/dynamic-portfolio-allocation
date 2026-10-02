"""Strategy assumptions, kept in one place so they are easy to find and change.

Milestone 3 only needs the VIX regime settings. Allocation, rebalancing and
transaction-cost settings will be added here in later milestones.
"""

from dataclasses import dataclass

REGIME_LABELS = ["low", "normal", "high", "extreme"]


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
