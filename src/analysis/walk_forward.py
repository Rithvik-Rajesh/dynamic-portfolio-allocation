"""Walk-forward validation (Milestone 9).

For each test year Y:
    1. DEVELOP: backtest every candidate setting on the development window,
       from the first available signal up to 31 December of Y-1.
    2. SELECT: freeze the candidate with the best development score
       (default: Sharpe ratio). Ties go to the earlier candidate in the list.
    3. TEST: run the frozen setting on year Y only, a period it has never
       seen, next to:
         - the fixed base rules (the default settings, never re-selected),
         - buy-and-hold and the fixed-allocation benchmark.
    4. Move forward one year and repeat.

No future information is used: selection only looks at data before Y, and
the VIX percentile itself only uses VIX history up to each day.

Each test year is an independent backtest that starts in cash on the first
trading day of the year, so every portfolio misses the market move of that
single day and pays one initial trading cost per year. This affects the
strategy and benchmarks equally. The yearly returns are then chained into
one continuous out-of-sample record.
"""

from dataclasses import replace

import pandas as pd

from src.analysis.benchmarks import BUY_AND_HOLD_NAME, fixed_allocation_name, run_benchmarks
from src.analysis.performance import (
    cagr,
    sharpe_ratio,
    sortino_ratio,
    summarise_performance,
    total_return,
)
from src.analysis.risk import annualised_volatility, max_drawdown
from src.analysis.sensitivity import vary
from src.pipeline import run_settings
from src.strategy.config import StrategySettings

SELECTED_NAME = "walk_forward_selected"
BASE_NAME = "fixed_base_rules"

FIRST_TEST_YEAR = 2019  # 2016-2018 is the first development window
LAST_TEST_YEAR = 2025
SELECTION_METRIC = "sharpe_ratio"

# Candidate settings the walk-forward may choose between. Kept deliberately
# small: the more candidates, the easier it is to overfit the past.
CANDIDATE_HIGH_THRESHOLDS = [0.70, 0.75, 0.80, 0.90]
CANDIDATE_WINDOWS = {"expanding": ("expanding", 504), "rolling 504d": ("rolling", 504)}
CANDIDATE_FREQUENCIES = ["weekly", "monthly"]


def build_candidates(base: StrategySettings) -> dict[str, StrategySettings]:
    """Every combination of the candidate values, all other settings from `base`."""
    candidates = {}
    for window_label, (window, size) in CANDIDATE_WINDOWS.items():
        for high in CANDIDATE_HIGH_THRESHOLDS:
            for frequency in CANDIDATE_FREQUENCIES:
                settings = vary(base, "regime", high_threshold=high,
                                percentile_window=window, rolling_window=size)
                settings = vary(settings, "backtest", rebalance_frequency=frequency)
                candidates[f"high {high:.0%}, {window_label}, {frequency}"] = settings
    return candidates


def with_period(settings: StrategySettings, start: str | None, end: str) -> StrategySettings:
    """Copy of settings with the backtest limited to [start, end]."""
    return vary(settings, "backtest", start_date=start, end_date=end)


def development_scores(market_data: pd.DataFrame, candidates: dict[str, StrategySettings],
                       development_end: str, metric: str) -> pd.Series:
    """Score every candidate on the development window (up to development_end)."""
    scores = {}
    for label, settings in candidates.items():
        history = run_settings(market_data, with_period(settings, None, development_end))
        metrics = summarise_performance(history, settings.backtest.cash_annual_rate,
                                        settings.backtest.initial_capital)
        scores[label] = metrics[metric]
    return pd.Series(scores, name=metric)


def select_candidate(scores: pd.Series) -> str:
    """Label of the best score. Ties (and NaNs) resolve to the earlier candidate."""
    return scores.fillna(float("-inf")).idxmax()


def run_test_year(market_data: pd.DataFrame, selected: StrategySettings, base: StrategySettings,
                  year: int, fixed_nifty_weight: float) -> dict[str, pd.DataFrame]:
    """Histories of all portfolios over one test year."""
    start, end = f"{year}-01-01", f"{year}-12-31"
    selected_history = run_settings(market_data, with_period(selected, start, end))
    histories = {
        SELECTED_NAME: selected_history,
        BASE_NAME: run_settings(market_data, with_period(base, start, end)),
    }
    histories.update(run_benchmarks(market_data, selected_history, base.backtest,
                                    fixed_nifty_weight))
    return histories


def run_walk_forward(market_data: pd.DataFrame, base: StrategySettings,
                     fixed_nifty_weight: float, first_test_year: int = FIRST_TEST_YEAR,
                     last_test_year: int = LAST_TEST_YEAR,
                     metric: str = SELECTION_METRIC) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the walk-forward procedure.

    Returns:
        folds    one row per test year: the selected candidate, its development
                 score, and each portfolio's test-year return / drawdown
        returns  daily returns of every portfolio over all test years, chained
    """
    candidates = build_candidates(base)
    fold_rows = []
    test_returns = []

    for year in range(first_test_year, last_test_year + 1):
        scores = development_scores(market_data, candidates, f"{year - 1}-12-31", metric)
        selected_label = select_candidate(scores)
        histories = run_test_year(market_data, candidates[selected_label], base, year,
                                  fixed_nifty_weight)

        row = {"test_year": year, "selected": selected_label,
               f"development_{metric}": scores[selected_label],
               f"base_development_{metric}": development_base_score(market_data, base, year, metric)}
        for name, history in histories.items():
            row[f"{name}_return"] = total_return(history["portfolio_return"])
            row[f"{name}_max_drawdown"] = max_drawdown(history["portfolio_value"])
        fold_rows.append(row)
        test_returns.append(pd.DataFrame(
            {name: history["portfolio_return"] for name, history in histories.items()}
        ))

    folds = pd.DataFrame(fold_rows).set_index("test_year")
    return folds, pd.concat(test_returns)


def development_base_score(market_data: pd.DataFrame, base: StrategySettings, year: int,
                           metric: str) -> float:
    """Development score of the base rules, for reference next to the selected one."""
    history = run_settings(market_data, with_period(base, None, f"{year - 1}-12-31"))
    return summarise_performance(history, base.backtest.cash_annual_rate,
                                 base.backtest.initial_capital)[metric]


def out_of_sample_summary(chained_returns: pd.DataFrame, cash_annual_rate: float) -> pd.DataFrame:
    """Headline metrics of each portfolio over the chained test years."""
    rows = {}
    for name in chained_returns.columns:
        returns = chained_returns[name]
        values = (1 + returns).cumprod()
        rows[name] = {
            "total_return": total_return(returns),
            "cagr": cagr(returns),
            "annualised_volatility": annualised_volatility(returns),
            "sharpe_ratio": sharpe_ratio(returns, cash_annual_rate),
            "sortino_ratio": sortino_ratio(returns, cash_annual_rate),
            "max_drawdown": max_drawdown(values),
        }
    return pd.DataFrame(rows).T


def portfolio_order(fixed_nifty_weight: float) -> list[str]:
    """Display order of the walk-forward portfolios."""
    return [SELECTED_NAME, BASE_NAME, BUY_AND_HOLD_NAME, fixed_allocation_name(fixed_nifty_weight)]
