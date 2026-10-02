import numpy as np
import pandas as pd
import pytest

from src.data.validator import find_validation_errors, validate_market_data


def valid_data():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    nifty = pd.Series([100.0, 101.0, 99.0], index=dates)
    return pd.DataFrame(
        {"vix": [15.0, 16.0, 18.0], "nifty": nifty, "nifty_return": nifty.pct_change()},
        index=pd.Index(dates, name="date"),
    )


def test_valid_data_passes():
    validate_market_data(valid_data())


def test_unsorted_dates_fail():
    data = valid_data().iloc[[1, 0, 2]]
    assert any("chronological" in error for error in find_validation_errors(data))


def test_duplicate_dates_fail():
    data = valid_data()
    data.index = pd.DatetimeIndex(["2020-01-01", "2020-01-01", "2020-01-03"], name="date")
    assert any("Duplicate" in error for error in find_validation_errors(data))


def test_missing_vix_fails():
    data = valid_data()
    data.loc[data.index[1], "vix"] = np.nan
    with pytest.raises(ValueError, match="'vix' has missing values"):
        validate_market_data(data)


def test_missing_return_after_first_row_fails():
    data = valid_data()
    data.loc[data.index[2], "nifty_return"] = np.nan
    assert any("nifty_return" in error for error in find_validation_errors(data))


def test_negative_price_fails():
    data = valid_data()
    data.loc[data.index[0], "nifty"] = -1.0
    assert any("negative" in error for error in find_validation_errors(data))


def test_implausible_values_fail():
    data = valid_data()
    data.loc[data.index[1], "vix"] = 500.0
    data.loc[data.index[2], "nifty_return"] = 0.5
    errors = find_validation_errors(data)
    assert any("VIX outside" in error for error in errors)
    assert any("sanity limit" in error for error in errors)


def test_missing_column_fails():
    assert find_validation_errors(valid_data().drop(columns="vix")) == [
        "Missing required columns: ['vix']"
    ]
