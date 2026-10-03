"""Smoke tests: the dashboard runs headlessly and reacts to its inputs."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


@pytest.fixture
def app():
    return AppTest.from_file(APP, default_timeout=60).run()


def metric_values(app):
    return {metric.label: metric.value for metric in app.metric}


def test_dashboard_runs_with_default_settings(app):
    assert not app.exception
    assert not app.error
    assert [tab.label for tab in app.tabs] == [
        "Overview", "Data", "Strategy", "Backtest", "Risk & Performance",
        "Comparison", "Experiments", "Walk-forward",
    ]
    assert set(metric_values(app)) == {"CAGR", "Volatility", "Sharpe ratio", "Max drawdown"}


def test_changing_a_setting_changes_the_result(app):
    before = metric_values(app)
    app.selectbox[0].set_value("monthly").run()  # rebalancing frequency
    assert not app.exception
    assert metric_values(app) != before


def test_rolling_window_reveals_window_slider(app):
    labels_before = [slider.label for slider in app.slider]
    app.radio[0].set_value("rolling").run()  # percentile window
    labels_after = [slider.label for slider in app.slider]
    assert "Rolling window (trading days)" not in labels_before
    assert "Rolling window (trading days)" in labels_after
    assert not app.exception


def test_invalid_thresholds_show_an_error_instead_of_crashing(app):
    high_slider = next(s for s in app.slider if s.label == "High regime from percentile")
    high_slider.set_value(10).run()  # below the 25% low threshold
    assert not app.exception
    assert any("Invalid settings" in error.value for error in app.error)


def test_open_execution_runs(app):
    app.radio[1].set_value("open").run()  # execution price
    assert not app.exception
    assert not app.error
