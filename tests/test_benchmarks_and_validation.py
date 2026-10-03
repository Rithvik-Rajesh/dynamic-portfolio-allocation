
import numpy as np
import pandas as pd
import pytest

from src.analysis.benchmarks import (
    BUY_AND_HOLD_NAME,
    compare_performance,
    fixed_allocation_name,
    run_benchmarks,
)
from src.analysis.sensitivity import build_experiments, run_experiment, vary
from src.analysis.walk_forward import (
    BASE_NAME,
    SELECTED_NAME,
    build_candidates,
    development_scores,
    out_of_sample_summary,
    run_walk_forward,
    select_candidate,
)
from src.pipeline import run_settings
from src.strategy.config import BenchmarkConfig, RegimeConfig, StrategySettings


def synthetic_market(start="2015-01-01", end="2019-12-31", seed=7):
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    nifty = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(dates))), index=dates)
    return pd.DataFrame({
        "vix": rng.uniform(10, 40, len(dates)),
        "nifty_open": nifty.shift(1).fillna(100.0),
        "nifty": nifty,
        "nifty_return": nifty.pct_change(),
    }, index=dates)


BASE = StrategySettings(regime=RegimeConfig(min_history=120))


# --- Benchmarks ---------------------------------------------------------------

def test_benchmarks_use_exactly_the_strategy_dates():
    market = synthetic_market()
    strategy = run_settings(market, BASE)
    benchmarks = run_benchmarks(market, strategy, BASE.backtest, 0.75)

    assert set(benchmarks) == {BUY_AND_HOLD_NAME, "fixed_75"}
    for history in benchmarks.values():
        assert history.index.equals(strategy.index)


def test_buy_and_hold_tracks_nifty_after_initial_cost():
    market = synthetic_market()
    strategy = run_settings(market, BASE)
    buy_and_hold = run_benchmarks(market, strategy, BASE.backtest, 0.75)[BUY_AND_HOLD_NAME]

    nifty = market.loc[strategy.index, "nifty"]
    expected = BASE.backtest.initial_capital / (1 + BASE.backtest.transaction_cost_rate) \
        * nifty / nifty.iloc[0]
    np.testing.assert_allclose(buy_and_hold["portfolio_value"], expected)
    assert buy_and_hold["traded"].sum() == 1


def test_fixed_benchmark_stays_near_its_weight():
    market = synthetic_market()
    strategy = run_settings(market, BASE)
    fixed = run_benchmarks(market, strategy, BASE.backtest, 0.6)["fixed_60"]
    # Weekly rebalancing with a 5-point band keeps it within ~5 points plus one week's drift.
    assert fixed["nifty_weight"].between(0.5, 0.7).all()


def test_compare_performance_has_one_row_per_portfolio():
    market = synthetic_market()
    strategy = run_settings(market, BASE)
    histories = {"vix_strategy": strategy, **run_benchmarks(market, strategy, BASE.backtest, 0.75)}
    table = compare_performance(histories, BASE.backtest)
    assert list(table.index) == ["vix_strategy", BUY_AND_HOLD_NAME, "fixed_75"]
    assert table.loc[BUY_AND_HOLD_NAME, "average_nifty_weight"] == pytest.approx(1.0)


def test_fixed_allocation_name_and_config():
    assert fixed_allocation_name(0.75) == "fixed_75"
    assert fixed_allocation_name(0.6) == "fixed_60"
    with pytest.raises(ValueError):
        BenchmarkConfig(fixed_nifty_weight=1.0)


# --- Sensitivity experiments ----------------------------------------------------

def test_vary_changes_only_one_field():
    changed = vary(BASE, "regime", high_threshold=0.8)
    assert changed.regime.high_threshold == 0.8
    assert changed.regime.low_threshold == BASE.regime.low_threshold
    assert changed.backtest == BASE.backtest
    assert BASE.regime.high_threshold == 0.75  # original untouched


def test_every_experiment_contains_valid_settings():
    experiments = build_experiments(StrategySettings())
    assert {"high_threshold", "rebalance_frequency", "transaction_cost", "percentile_window",
            "start_year", "execution_price", "drift_tolerance"} <= set(experiments)
    # The base settings appear in the experiments that vary a base value.
    assert any(settings == StrategySettings()
               for settings in experiments["rebalance_frequency"].values())


def test_run_experiment_marks_base_and_compares_with_benchmarks():
    market = synthetic_market()
    variants = {"weekly": BASE, "monthly": vary(BASE, "backtest", rebalance_frequency="monthly")}
    table = run_experiment(market, variants, BASE, 0.75)

    assert table.loc["weekly", "is_base"]
    assert not table.loc["monthly", "is_base"]
    row = table.loc["weekly"]
    assert row["sharpe_vs_buy_and_hold"] == pytest.approx(
        row["strategy_sharpe_ratio"] - row["buy_and_hold_sharpe_ratio"])


# --- Walk-forward -----------------------------------------------------------------

def test_candidates_are_distinct_and_valid():
    candidates = build_candidates(StrategySettings())
    assert len(candidates) == 16
    assert len(set(candidates.values())) == 16


def test_selection_ties_go_to_first_candidate():
    assert select_candidate(pd.Series({"a": 1.0, "b": 1.0, "c": np.nan})) == "a"
    assert select_candidate(pd.Series({"a": np.nan, "b": 0.5})) == "b"


def test_selection_does_not_use_future_data():
    """Scores for the development window ending 2017 must not change when the
    2018-2019 data is replaced with something completely different."""
    market = synthetic_market()
    candidates = build_candidates(BASE)
    original = development_scores(market, candidates, "2017-12-31", "sharpe_ratio")

    altered = market.copy()
    future = altered.index > "2017-12-31"
    altered.loc[future, "vix"] = 99.0
    altered.loc[future, "nifty"] = altered.loc[future, "nifty"] * 0.5
    altered["nifty_return"] = altered["nifty"].pct_change()
    changed = development_scores(altered, candidates, "2017-12-31", "sharpe_ratio")

    pd.testing.assert_series_equal(original, changed)


def test_walk_forward_test_years_are_out_of_sample_and_chained():
    market = synthetic_market()
    folds, chained = run_walk_forward(market, BASE, 0.75, first_test_year=2018,
                                      last_test_year=2019)

    assert list(folds.index) == [2018, 2019]
    assert chained.index.min() >= pd.Timestamp("2018-01-01")
    assert chained.index.is_unique and chained.index.is_monotonic_increasing
    assert {SELECTED_NAME, BASE_NAME, BUY_AND_HOLD_NAME, "fixed_75"} <= set(chained.columns)

    # The fold's yearly return equals the compounded daily returns of that year.
    year_2019 = chained.loc["2019", BUY_AND_HOLD_NAME]
    assert folds.loc[2019, f"{BUY_AND_HOLD_NAME}_return"] == pytest.approx(
        (1 + year_2019).prod() - 1)


def test_out_of_sample_summary_matches_chained_returns():
    returns = pd.DataFrame({"a": [0.0, 0.1, -0.05]},
                           index=pd.to_datetime(["2020-01-01", "2020-07-01", "2021-01-01"]))
    summary = out_of_sample_summary(returns, 0.0)
    assert summary.loc["a", "total_return"] == pytest.approx(1.1 * 0.95 - 1)
    assert summary.loc["a", "max_drawdown"] == pytest.approx(-0.05)
