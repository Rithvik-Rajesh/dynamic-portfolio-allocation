import pandas as pd
import pytest

from src.data import loader


def make_prices(dates):
    index = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    values = range(1, len(dates) + 1)
    return pd.DataFrame(
        {column: values for column in ["Open", "High", "Low", "Close", "Volume"]},
        index=index,
    )


def test_download_raises_on_empty_result(monkeypatch):
    monkeypatch.setattr(loader.yf, "download", lambda *args, **kwargs: pd.DataFrame())
    with pytest.raises(loader.DataDownloadError):
        loader.download_prices("^NSEI", "2015-01-01", "2016-01-01")


def test_download_raises_on_exception(monkeypatch):
    def failing_download(*args, **kwargs):
        raise ConnectionError("no internet")

    monkeypatch.setattr(loader.yf, "download", failing_download)
    with pytest.raises(loader.DataDownloadError, match="no internet"):
        loader.download_prices("^NSEI", "2015-01-01", "2016-01-01")


def test_save_and_read_round_trip(tmp_path):
    prices = make_prices(["2020-01-01", "2020-01-02"])
    path = tmp_path / "prices.csv"
    loader.save_raw_prices(prices, path)
    pd.testing.assert_frame_equal(loader.read_raw_prices(path), prices, check_dtype=False,
                                  check_freq=False)


def test_load_uses_cache_without_downloading(monkeypatch, tmp_path):
    monkeypatch.setattr(loader, "RAW_DATA_DIR", tmp_path)
    loader.save_raw_prices(make_prices(["2020-01-01"]), tmp_path / "vix.csv")

    def must_not_download(*args, **kwargs):
        raise AssertionError("download should not be called when a cache exists")

    monkeypatch.setattr(loader.yf, "download", must_not_download)
    assert len(loader.load_raw_prices("vix")) == 1


def test_load_downloads_and_caches_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(loader, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(loader, "RAW_METADATA_FILE", tmp_path / "metadata.json")
    monkeypatch.setattr(loader.yf, "download",
                        lambda *args, **kwargs: make_prices(["2020-01-01", "2020-01-02"]))

    prices = loader.load_raw_prices("nifty")

    assert len(prices) == 2
    assert (tmp_path / "nifty.csv").exists()
    assert "nifty" in (tmp_path / "metadata.json").read_text()


def test_unknown_dataset_name():
    with pytest.raises(ValueError):
        loader.load_raw_prices("bitcoin")
