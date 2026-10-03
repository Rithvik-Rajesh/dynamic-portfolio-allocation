import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import build_market_data, clean_price_series, unmatched_dates


def raw(dates, closes, opens=None):
    opens = closes if opens is None else opens
    return pd.DataFrame({"Open": opens, "Close": closes}, index=pd.Index(dates, name="Date"))


def test_clean_close_sorts_and_removes_duplicates():
    prices = raw(["2020-01-03", "2020-01-01", "2020-01-01"], [3.0, 1.0, 99.0])
    close = clean_price_series(prices, "nifty")

    assert list(close.index.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-01-03"]
    assert close.loc["2020-01-01"] == 1.0  # first occurrence kept


def test_clean_close_drops_missing_and_non_positive():
    prices = raw(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"],
                 [10.0, np.nan, 0.0, 12.0])
    close = clean_price_series(prices, "vix")
    assert list(close.values) == [10.0, 12.0]


def test_alignment_keeps_only_common_dates():
    vix = raw(["2020-01-01", "2020-01-02", "2020-01-03"], [15.0, 16.0, 17.0])
    nifty = raw(["2020-01-02", "2020-01-03", "2020-01-06"], [100.0, 110.0, 121.0])

    market_data = build_market_data(vix, nifty)

    assert list(market_data.index.strftime("%Y-%m-%d")) == ["2020-01-02", "2020-01-03"]
    assert unmatched_dates(vix, nifty) == {
        "only_in_vix": [pd.Timestamp("2020-01-01").date()],
        "only_in_nifty": [pd.Timestamp("2020-01-06").date()],
    }


def test_return_spans_a_dropped_date():
    # NIFTY has 2020-01-02 but VIX does not, so that day is dropped. The next
    # return must cover both days: 100 -> 121 = +21%, not lost or split.
    vix = raw(["2020-01-01", "2020-01-03"], [15.0, 16.0])
    nifty = raw(["2020-01-01", "2020-01-02", "2020-01-03"], [100.0, 110.0, 121.0])

    market_data = build_market_data(vix, nifty)

    assert np.isnan(market_data["nifty_return"].iloc[0])
    assert market_data["nifty_return"].iloc[1] == pytest.approx(0.21)


def test_open_column_is_kept_and_rows_without_open_are_dropped():
    vix = raw(["2020-01-01", "2020-01-02", "2020-01-03"], [15.0, 16.0, 17.0])
    nifty = raw(["2020-01-01", "2020-01-02", "2020-01-03"], [100.0, 110.0, 121.0],
                opens=[99.0, np.nan, 120.0])

    market_data = build_market_data(vix, nifty)

    assert list(market_data.columns) == ["vix", "nifty_open", "nifty", "nifty_return"]
    assert list(market_data.index.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-01-03"]
    assert market_data["nifty_return"].iloc[1] == pytest.approx(0.21)
