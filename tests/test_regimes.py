import numpy as np
import pandas as pd
import pytest

from src.analysis.regime_analysis import (
    add_next_day_return,
    nifty_behaviour_by_regime,
    regime_frequency,
)
from src.strategy.config import RegimeConfig
from src.strategy.regimes import add_vix_regimes, calculate_vix_percentile, classify_vix_regime


def series(values):
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)),
                     dtype=float)


# --- Percentile --------------------------------------------------------------

def test_percentile_hand_calculated():
    # Day 3: values so far [10, 20, 15]; 2 of 3 are <= 15 -> 0.667
    # Day 4: values so far [10, 20, 15, 30]; 4 of 4 are <= 30 -> 1.0
    # Day 5: values so far [10, 20, 15, 30, 5]; 1 of 5 is <= 5 -> 0.2
    percentile = calculate_vix_percentile(series([10, 20, 15, 30, 5]), min_history=3)

    assert percentile.iloc[:2].isna().all()
    assert percentile.iloc[2:].tolist() == pytest.approx([2 / 3, 1.0, 0.2])


def test_percentile_ties_count_as_less_or_equal():
    percentile = calculate_vix_percentile(series([10, 10, 10]), min_history=1)
    assert percentile.tolist() == [1.0, 1.0, 1.0]


def test_percentile_has_no_look_ahead():
    """Changing or adding FUTURE values must not change any past percentile."""
    rng = np.random.default_rng(seed=0)
    vix = series(rng.uniform(10, 40, size=400))
    original = calculate_vix_percentile(vix, min_history=50)

    cutoff = 300
    altered_future = vix.copy()
    altered_future.iloc[cutoff:] = 1000.0  # extreme future spike
    altered = calculate_vix_percentile(altered_future, min_history=50)

    truncated = calculate_vix_percentile(vix.iloc[:cutoff], min_history=50)

    pd.testing.assert_series_equal(original.iloc[:cutoff], altered.iloc[:cutoff])
    pd.testing.assert_series_equal(original.iloc[:cutoff], truncated)


# --- Regime classification ---------------------------------------------------

def test_regime_boundaries():
    config = RegimeConfig(low_threshold=0.25, high_threshold=0.75, extreme_threshold=0.95)
    percentile = series([0.0, 0.2499, 0.25, 0.7499, 0.75, 0.9499, 0.95, 1.0, np.nan])

    regime = classify_vix_regime(percentile, config)

    assert regime.iloc[:8].tolist() == [
        "low", "low", "normal", "normal", "high", "high", "extreme", "extreme",
    ]
    assert pd.isna(regime.iloc[8])
    assert regime.cat.ordered


def test_custom_thresholds_are_used():
    config = RegimeConfig(low_threshold=0.1, high_threshold=0.5, extreme_threshold=0.9)
    regime = classify_vix_regime(series([0.2, 0.6]), config)
    assert regime.tolist() == ["normal", "high"]


@pytest.mark.parametrize("thresholds", [(0.5, 0.4, 0.9), (0.0, 0.5, 0.9), (0.2, 0.5, 1.0),
                                        (0.2, 0.5, 0.5)])
def test_invalid_thresholds_rejected(thresholds):
    low, high, extreme = thresholds
    with pytest.raises(ValueError):
        RegimeConfig(low_threshold=low, high_threshold=high, extreme_threshold=extreme)


def test_invalid_min_history_rejected():
    with pytest.raises(ValueError):
        RegimeConfig(min_history=0)


def test_add_vix_regimes_does_not_modify_input():
    market_data = pd.DataFrame({"vix": series([10, 20, 15])})
    result = add_vix_regimes(market_data, RegimeConfig(min_history=1))

    assert list(market_data.columns) == ["vix"]
    assert {"vix_percentile", "vix_regime"} <= set(result.columns)


# --- Regime analysis ---------------------------------------------------------

def test_next_day_return_is_shifted_forward():
    data = pd.DataFrame({"nifty_return": series([np.nan, 0.01, 0.02, 0.03])})
    result = add_next_day_return(data)
    assert result["next_day_nifty_return"].iloc[0] == 0.01
    assert result["next_day_nifty_return"].iloc[2] == 0.03
    assert np.isnan(result["next_day_nifty_return"].iloc[3])


def test_behaviour_uses_next_day_returns():
    regime = pd.Categorical(["low", "extreme", "low", "low"],
                            categories=["low", "normal", "high", "extreme"], ordered=True)
    data = pd.DataFrame({"vix_regime": regime,
                         "nifty_return": [np.nan, 0.01, -0.05, 0.02]},
                        index=pd.bdate_range("2020-01-01", periods=4))

    behaviour = nifty_behaviour_by_regime(data)

    # Day 1 is 'low', followed by +1%. Day 2 is 'extreme', followed by -5%.
    # Day 3 is 'low', followed by +2%. Day 4 has no next day.
    assert behaviour.loc["low", "days"] == 2
    assert behaviour.loc["low", "mean_daily_return"] == pytest.approx(0.015)
    assert behaviour.loc["extreme", "mean_daily_return"] == pytest.approx(-0.05)
    assert behaviour.loc["normal", "days"] == 0

    frequency = regime_frequency(data)
    assert frequency.loc["low", "days"] == 3
    assert frequency["share"].sum() == pytest.approx(1.0)


# --- Rolling window ----------------------------------------------------------

def test_rolling_percentile_hand_calculated():
    # Window of 3, at least 2 observations:
    # day 2 [10, 20] -> 20 is 2/2; day 3 [10, 20, 15] -> 2/3; day 4 [20, 15, 30] -> 3/3;
    # day 5 [15, 30, 5] -> 1/3; day 6 [30, 5, 12] -> 2/3
    percentile = calculate_vix_percentile(series([10, 20, 15, 30, 5, 12]), min_history=2,
                                          rolling_window=3)
    assert np.isnan(percentile.iloc[0])
    assert percentile.iloc[1:].tolist() == pytest.approx([1.0, 2 / 3, 1.0, 1 / 3, 2 / 3])


def test_rolling_window_forgets_old_spike():
    # A spike to 80 on day 0. Five days later VIX is 20.
    vix = series([80, 10, 10, 10, 10, 20])
    expanding = calculate_vix_percentile(vix, min_history=1)
    rolling = calculate_vix_percentile(vix, min_history=1, rolling_window=5)
    assert expanding.iloc[-1] == pytest.approx(5 / 6)  # the spike still counts
    assert rolling.iloc[-1] == pytest.approx(1.0)      # the spike has dropped out


def test_rolling_percentile_has_no_look_ahead():
    rng = np.random.default_rng(seed=0)
    vix = series(rng.uniform(10, 40, size=400))
    original = calculate_vix_percentile(vix, min_history=50, rolling_window=100)
    altered_future = vix.copy()
    altered_future.iloc[300:] = 1000.0
    altered = calculate_vix_percentile(altered_future, min_history=50, rolling_window=100)
    pd.testing.assert_series_equal(original.iloc[:300], altered.iloc[:300])


def test_add_vix_regimes_uses_rolling_setting():
    market_data = pd.DataFrame({"vix": series([80, 10, 10, 10, 10, 20])})
    rolling = add_vix_regimes(market_data, RegimeConfig(min_history=1, percentile_window="rolling",
                                                        rolling_window=5))
    expanding = add_vix_regimes(market_data, RegimeConfig(min_history=1))
    assert rolling["vix_regime"].iloc[-1] == "extreme"
    assert expanding["vix_regime"].iloc[-1] == "high"


@pytest.mark.parametrize("settings", [
    {"percentile_window": "weekly"},
    {"percentile_window": "rolling", "rolling_window": 100, "min_history": 252},
    {"percentile_window": "rolling", "rolling_window": 1, "min_history": 1},
])
def test_invalid_window_settings_rejected(settings):
    with pytest.raises(ValueError):
        RegimeConfig(**settings)
