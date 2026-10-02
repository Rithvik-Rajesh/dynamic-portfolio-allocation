"""VIX signal: expanding percentile and regime classification.

Look-ahead rule
---------------
The percentile on date t compares VIX(t) only with VIX values from the start
of the data up to and including date t. No later value is ever used.

Timing: VIX(t) is the closing value, so the percentile and regime for date t
are only known at the END of day t. Any allocation based on them can only
earn returns from day t+1 onwards. The backtesting engine (Milestone 5) must
respect this.
"""

import numpy as np
import pandas as pd

from src.strategy.config import REGIME_LABELS, RegimeConfig


def calculate_vix_percentile(vix: pd.Series, min_history: int) -> pd.Series:
    """Expanding percentile of each day's VIX within its own history.

    percentile(t) = share of VIX observations from the first day up to and
    including day t that are less than or equal to VIX(t).

    The result is between 0 and 1. Days with fewer than `min_history`
    observations get NaN.
    """
    percentile = vix.expanding(min_periods=min_history).rank(method="max", pct=True)
    return percentile.rename("vix_percentile")


def classify_vix_regime(vix_percentile: pd.Series, config: RegimeConfig) -> pd.Series:
    """Map each percentile to a regime label. NaN percentiles stay NaN."""
    conditions = [
        vix_percentile < config.low_threshold,
        vix_percentile < config.high_threshold,
        vix_percentile < config.extreme_threshold,
        vix_percentile >= config.extreme_threshold,
    ]
    labels = np.select(conditions, REGIME_LABELS, default=None)

    regime = pd.Series(labels, index=vix_percentile.index, name="vix_regime")
    return regime.astype(pd.CategoricalDtype(REGIME_LABELS, ordered=True))


def add_vix_regimes(market_data: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    """Return a copy of the market data with `vix_percentile` and `vix_regime` added."""
    market_data = market_data.copy()
    market_data["vix_percentile"] = calculate_vix_percentile(
        market_data["vix"], config.min_history
    )
    market_data["vix_regime"] = classify_vix_regime(market_data["vix_percentile"], config)
    return market_data
