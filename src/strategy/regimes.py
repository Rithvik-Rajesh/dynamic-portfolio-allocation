"""VIX signal: VIX percentile (expanding or rolling) and regime classification.

Look-ahead rule
---------------
The percentile on date t compares VIX(t) only with VIX values up to and
including date t: either all history since the start of the data
('expanding') or the most recent N trading days ('rolling'). No later value
is ever used.

Timing: VIX(t) is the closing value, so the percentile and regime for date t
are only known at the END of day t. Any allocation based on them can only
earn returns from day t+1 onwards. The backtesting engine (Milestone 5) must
respect this.
"""

import numpy as np
import pandas as pd

from src.strategy.config import REGIME_LABELS, RegimeConfig


def calculate_vix_percentile(vix: pd.Series, min_history: int,
                             rolling_window: int | None = None) -> pd.Series:
    """Percentile of each day's VIX within its own past.

    percentile(t) = share of the VIX observations in the comparison window
    that are less than or equal to VIX(t). The window always ends on day t:
        rolling_window=None -> expanding: every day from the first day of data
        rolling_window=N    -> rolling: the last N trading days

    The result is between 0 and 1. Days with fewer than `min_history`
    observations in their window get NaN.
    """
    if rolling_window is None:
        window = vix.expanding(min_periods=min_history)
    else:
        window = vix.rolling(rolling_window, min_periods=min_history)
    percentile = window.rank(method="max", pct=True)
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
    rolling_window = config.rolling_window if config.percentile_window == "rolling" else None
    market_data["vix_percentile"] = calculate_vix_percentile(
        market_data["vix"], config.min_history, rolling_window
    )
    market_data["vix_regime"] = classify_vix_regime(market_data["vix_percentile"], config)
    return market_data
