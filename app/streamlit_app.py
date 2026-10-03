"""Streamlit dashboard for the India VIX dynamic NIFTY allocation study.

    uv run streamlit run app/streamlit_app.py

The app contains no financial logic of its own: every number comes from the
functions in `src/`, called with the settings chosen in the sidebar.
"""

import sys
from pathlib import Path

# `streamlit run` only puts the app/ folder on the import path; add the
# project root so `src` can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.analysis.benchmarks import (  # noqa: E402
    BUY_AND_HOLD_NAME,
    STRATEGY_NAME,
    compare_performance,
    portfolio_values,
    run_benchmarks,
)
from src.analysis.performance import calendar_year_returns  # noqa: E402
from src.analysis.regime_analysis import (  # noqa: E402
    nifty_behaviour_by_regime,
    regime_frequency,
)
from src.analysis.sensitivity import run_all_experiments  # noqa: E402
from src.analysis.walk_forward import (  # noqa: E402
    CANDIDATE_FREQUENCIES,
    CANDIDATE_HIGH_THRESHOLDS,
    CANDIDATE_WINDOWS,
    FIRST_TEST_YEAR,
    LAST_TEST_YEAR,
    SELECTION_METRIC,
    out_of_sample_summary,
    portfolio_order,
    run_walk_forward,
)
from src.config import DATA_SOURCE, TICKERS  # noqa: E402
from src.data.cleaner import build_market_data, unmatched_dates  # noqa: E402
from src.data.loader import load_raw_data  # noqa: E402
from src.data.validator import (  # noqa: E402
    describe_coverage,
    describe_market_data,
    validate_market_data,
)
from src.pipeline import run_settings  # noqa: E402
from src.strategy.config import (  # noqa: E402
    REGIME_LABELS,
    AllocationConfig,
    BacktestConfig,
    BenchmarkConfig,
    RegimeConfig,
    StrategySettings,
)
from src.strategy.regimes import add_vix_regimes  # noqa: E402
from src.visualization.charts import (  # noqa: E402
    plot_backtest,
    plot_benchmark_comparison,
    plot_calendar_year_returns,
    plot_market_with_regimes,
    plot_regime_summary,
    plot_sensitivity_overview,
    plot_vix_percentile,
    plot_walk_forward,
)
from src.visualization.tables import (  # noqa: E402
    display_name,
    format_metrics_table,
    format_percent_table,
)

# ---------------------------------------------------------------------------
# Cached calculations
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading market data...")
def load_market_data() -> tuple[pd.DataFrame, dict]:
    raw = load_raw_data()
    market_data = build_market_data(raw["vix"], raw["nifty"])
    validate_market_data(market_data)
    return market_data, unmatched_dates(raw["vix"], raw["nifty"])


@st.cache_data(show_spinner="Running backtest...")
def run_comparison(_market_data: pd.DataFrame, settings: StrategySettings,
                   fixed_nifty_weight: float) -> tuple[dict, pd.DataFrame]:
    """Strategy + benchmark histories and their metrics for these settings."""
    strategy_history = run_settings(_market_data, settings)
    histories = {STRATEGY_NAME: strategy_history}
    histories.update(run_benchmarks(_market_data, strategy_history, settings.backtest,
                                    fixed_nifty_weight))
    return histories, compare_performance(histories, settings.backtest)


@st.cache_data(show_spinner="Running all experiments (about 20 seconds)...")
def run_experiments(_market_data: pd.DataFrame, settings: StrategySettings,
                    fixed_nifty_weight: float) -> pd.DataFrame:
    return run_all_experiments(_market_data, settings, fixed_nifty_weight)


@st.cache_data(show_spinner="Running walk-forward validation (about 10 seconds)...")
def run_validation(_market_data: pd.DataFrame, settings: StrategySettings,
                   fixed_nifty_weight: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    return run_walk_forward(_market_data, settings, fixed_nifty_weight)


def show(figure) -> None:
    """Draw a matplotlib figure in the page and free its memory."""
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Sidebar: every tuneable setting
# ---------------------------------------------------------------------------

def sidebar_settings(market_data: pd.DataFrame) -> tuple[StrategySettings, BenchmarkConfig]:
    """Read all settings from the sidebar. Stops the app if they are invalid."""
    sidebar = st.sidebar
    first_date, last_date = market_data.index[0].date(), market_data.index[-1].date()

    sidebar.header("Settings")

    sidebar.subheader("Investment")
    initial_capital = sidebar.number_input("Initial amount (₹)", min_value=1_000,
                                           value=100_000, step=10_000)
    start_date = sidebar.date_input("Start investing on", value=first_date,
                                    min_value=first_date, max_value=last_date)
    end_date = sidebar.date_input("End on", value=last_date,
                                  min_value=first_date, max_value=last_date)

    sidebar.subheader("VIX signal")
    window_type = sidebar.radio("Percentile window", ["expanding", "rolling"], horizontal=True,
                                help="Expanding: compare with all VIX history so far. "
                                     "Rolling: compare with the last N trading days only.")
    rolling_window = 504
    if window_type == "rolling":
        rolling_window = sidebar.slider("Rolling window (trading days)", 63, 1260, 504, step=21,
                                        help="252 ≈ 1 year, 504 ≈ 2 years")
    min_history = sidebar.slider("Warm-up (trading days)", 21, 504, 252, step=21,
                                 help="VIX history needed before the first regime.")
    low = sidebar.slider("Low regime below percentile", 1, 98, 25)
    high = sidebar.slider("High regime from percentile", 2, 98, 75)
    extreme = sidebar.slider("Extreme regime from percentile", 3, 99, 95)

    sidebar.subheader("NIFTY allocation by regime (%)")
    weights = {regime: sidebar.slider(regime.capitalize(), 0, 100, default, step=5)
               for regime, default in zip(REGIME_LABELS, [100, 75, 50, 25], strict=True)}

    sidebar.subheader("Trading")
    frequency = sidebar.selectbox("Rebalancing", ["weekly", "daily", "monthly"])
    drift = sidebar.slider("Drift band (weight points)", 0, 30, 5,
                           help="Trade back to target if the actual NIFTY weight is "
                                "further than this from target.")
    execution_price = sidebar.radio("Trade at the next day's", ["close", "open"], horizontal=True)
    cost_percent = sidebar.number_input("Transaction cost (% of traded value)", 0.0, 2.0, 0.10,
                                        step=0.05, format="%.2f")
    cash_percent = sidebar.number_input("Cash return / risk-free rate (% per year)", 0.0, 15.0,
                                        6.0, step=0.5, format="%.1f")

    sidebar.subheader("Benchmark")
    fixed_weight = sidebar.slider("Fixed-allocation benchmark NIFTY weight (%)", 5, 95, 75, step=5)

    try:
        settings = StrategySettings(
            regime=RegimeConfig(low_threshold=low / 100, high_threshold=high / 100,
                                extreme_threshold=extreme / 100, min_history=min_history,
                                percentile_window=window_type, rolling_window=rolling_window),
            allocation=AllocationConfig(**{regime: weight / 100
                                           for regime, weight in weights.items()}),
            backtest=BacktestConfig(initial_capital=float(initial_capital),
                                    rebalance_frequency=frequency,
                                    drift_tolerance=drift / 100,
                                    transaction_cost_rate=cost_percent / 100,
                                    cash_annual_rate=cash_percent / 100,
                                    execution_price=execution_price,
                                    start_date=start_date.isoformat(),
                                    end_date=end_date.isoformat()),
        )
        benchmark = BenchmarkConfig(fixed_nifty_weight=fixed_weight / 100)
    except ValueError as error:
        st.error(f"Invalid settings: {error}")
        st.stop()
    return settings, benchmark


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def overview_tab(settings: StrategySettings, comparison: pd.DataFrame) -> None:
    st.markdown(
        "**Research question.** Can India VIX be used as a market-risk signal to adjust NIFTY 50 "
        "exposure, and does that give better risk-adjusted performance or downside protection "
        "than simple static portfolios?"
    )
    st.markdown(
        "**How the strategy works.** Each day the VIX close is ranked against its own history "
        "(a percentile). The percentile sets the regime, the regime sets the target NIFTY "
        "weight, and the rest is held in cash. Trades happen only on rebalance days, after the "
        "signal is known, and pay transaction costs. It manages risk; it does not predict prices."
    )

    strategy, buy_and_hold = comparison.loc[STRATEGY_NAME], comparison.loc[BUY_AND_HOLD_NAME]
    st.subheader(f"Result with current settings ({strategy['start_date']} to "
                 f"{strategy['end_date']})")
    st.caption("Deltas are versus buy-and-hold over the same dates.")
    columns = st.columns(4)
    for column, (metric, label, inverse) in zip(columns, [
        ("cagr", "CAGR", False),
        ("annualised_volatility", "Volatility", True),
        ("sharpe_ratio", "Sharpe ratio", False),
        ("max_drawdown", "Max drawdown", False),
    ], strict=True):
        is_ratio = metric == "sharpe_ratio"
        value, gap = strategy[metric], strategy[metric] - buy_and_hold[metric]
        column.metric(label, f"{value:.2f}" if is_ratio else f"{value:.1%}",
                      f"{gap:+.2f}" if is_ratio else f"{gap * 100:+.1f} pts",
                      delta_color="inverse" if inverse else "normal")

    st.subheader("Current settings")
    st.table(settings_table(settings))


def settings_table(settings: StrategySettings) -> pd.DataFrame:
    regime, allocation, backtest = settings.regime, settings.allocation, settings.backtest
    window = ("expanding" if regime.percentile_window == "expanding"
              else f"rolling, {regime.rolling_window} trading days")
    rows = {
        "Percentile window": window,
        "Regime thresholds": f"{regime.low_threshold:.0%} / {regime.high_threshold:.0%} / "
                             f"{regime.extreme_threshold:.0%}",
        "NIFTY weight (low / normal / high / extreme)":
            " / ".join(f"{weight:.0%}" for weight in allocation.as_dict().values()),
        "Rebalancing": f"{backtest.rebalance_frequency}, drift band "
                       f"{backtest.drift_tolerance * 100:g} pts",
        "Execution": f"next day's {backtest.execution_price}",
        "Transaction cost": f"{backtest.transaction_cost_rate:.2%} of traded value",
        "Cash return": f"{backtest.cash_annual_rate:.1%} per year",
        "Initial amount": f"₹{backtest.initial_capital:,.0f}",
    }
    return pd.DataFrame({"Setting": rows.keys(), "Value": rows.values()}).set_index("Setting")


def data_tab(market_data: pd.DataFrame, dropped: dict) -> None:
    coverage = describe_coverage(market_data)
    st.markdown(
        f"**Source:** {DATA_SOURCE}. Tickers: India VIX `{TICKERS['vix']}`, NIFTY 50 "
        f"`{TICKERS['nifty']}` (price index, dividends not included).  \n"
        f"**Coverage:** {coverage['first_date']} to {coverage['last_date']}, "
        f"{coverage['trading_days']:,} trading days on which both series have data. "
        f"{len(dropped['only_in_vix'])} VIX-only and {len(dropped['only_in_nifty'])} "
        "NIFTY-only dates were dropped."
    )
    show(plot_market_with_regimes(market_data, show_regimes=False))
    st.subheader("Descriptive statistics")
    st.dataframe(describe_market_data(market_data).round(4))


def strategy_tab(settings: StrategySettings, market_data: pd.DataFrame) -> None:
    regime, allocation = settings.regime, settings.allocation
    bounds = [0, regime.low_threshold, regime.high_threshold, regime.extreme_threshold, 1]
    rules = pd.DataFrame({
        "VIX percentile": [f"{bounds[i]:.0%} – {bounds[i + 1]:.0%}" for i in range(4)],
        "NIFTY": [f"{weight:.0%}" for weight in allocation.as_dict().values()],
        "Cash": [f"{1 - weight:.0%}" for weight in allocation.as_dict().values()],
    }, index=[label.capitalize() for label in REGIME_LABELS])
    rules.index.name = "Regime"
    st.subheader("Allocation rule")
    st.table(rules)

    regime_data = add_vix_regimes(market_data, regime)
    show(plot_market_with_regimes(regime_data))
    show(plot_vix_percentile(regime_data, regime))

    frequency = regime_frequency(regime_data)
    behaviour = nifty_behaviour_by_regime(regime_data)
    show(plot_regime_summary(frequency, behaviour))
    st.subheader("NIFTY on the day after each regime")
    st.caption("Descriptive only: uses returns after each day's regime was known.")
    table = behaviour.copy()
    for column in ["mean_daily_return", "daily_volatility", "annualised_volatility",
                   "share_negative_days", "worst_day", "best_day"]:
        table[column] = table[column].map(lambda value: f"{value:.2%}")
    st.dataframe(table)


def backtest_tab(history: pd.DataFrame) -> None:
    show(plot_backtest(history))
    trades = history[history["traded"]][
        ["vix_regime", "tradable_target_weight", "trade_value", "transaction_cost",
         "nifty_weight", "portfolio_value"]
    ]
    st.subheader(f"Trades ({len(trades)})")
    st.dataframe(trades.style.format({
        "tradable_target_weight": "{:.0%}", "trade_value": "₹{:,.0f}",
        "transaction_cost": "₹{:,.0f}", "nifty_weight": "{:.1%}", "portfolio_value": "₹{:,.0f}",
    }))
    st.download_button("Download daily history (CSV)", history.to_csv().encode(),
                       file_name="vix_strategy_backtest.csv", mime="text/csv")


def risk_tab(histories: dict, comparison: pd.DataFrame) -> None:
    st.subheader("VIX strategy: performance and risk")
    st.table(format_metrics_table(comparison.loc[[STRATEGY_NAME]]))
    st.subheader("Calendar-year returns")
    yearly = pd.DataFrame({name: calendar_year_returns(history["portfolio_return"])
                           for name, history in histories.items()})
    show(plot_calendar_year_returns(yearly))
    st.dataframe(format_percent_table(yearly.rename(columns=display_name)))


def comparison_tab(histories: dict, comparison: pd.DataFrame) -> None:
    st.caption("All portfolios use the same dates, initial amount, transaction costs, "
               "cash rate and execution settings. Only the target NIFTY weight differs.")
    show(plot_benchmark_comparison(portfolio_values(histories)))
    st.table(format_metrics_table(comparison))


def experiments_tab(market_data: pd.DataFrame, settings: StrategySettings,
                    benchmark: BenchmarkConfig) -> None:
    st.markdown(
        "Each experiment changes **one** setting from the current sidebar settings and keeps "
        "the rest. Every variant is compared with buy-and-hold over the same dates. The aim is "
        "to see how robust the strategy is, not to pick the best historical setting."
    )
    if st.button("Run experiments", key="run_experiments"):
        st.session_state["experiments_for"] = (settings, benchmark)
    if st.session_state.get("experiments_for") != (settings, benchmark):
        st.info("Press the button to run the experiments for the current settings.")
        return

    experiments = run_experiments(market_data, settings, benchmark.fixed_nifty_weight)
    show(plot_sensitivity_overview(experiments))
    name = st.selectbox("Experiment details", experiments["experiment"].unique())
    rows = experiments[experiments["experiment"] == name].set_index("variant")
    percent_columns = ["strategy_cagr", "strategy_annualised_volatility", "strategy_max_drawdown",
                       "buy_and_hold_cagr", "buy_and_hold_max_drawdown"]
    table = rows[percent_columns + ["strategy_sharpe_ratio", "buy_and_hold_sharpe_ratio",
                                    "strategy_number_of_trades"]].copy()
    table[percent_columns] = format_percent_table(table[percent_columns])
    for column in ["strategy_sharpe_ratio", "buy_and_hold_sharpe_ratio"]:
        table[column] = table[column].map(lambda value: f"{value:.2f}")
    st.dataframe(table)


def walk_forward_tab(market_data: pd.DataFrame, settings: StrategySettings,
                     benchmark: BenchmarkConfig) -> None:
    windows = ", ".join(CANDIDATE_WINDOWS)
    st.markdown(
        f"For each test year {FIRST_TEST_YEAR}–{LAST_TEST_YEAR}: backtest every candidate "
        f"setting on all earlier data, freeze the one with the best "
        f"{SELECTION_METRIC.replace('_', ' ')}, then test it on the unseen year. "
        f"Candidates: high threshold {', '.join(f'{v:.0%}' for v in CANDIDATE_HIGH_THRESHOLDS)}"
        f" × window {windows} × rebalancing {', '.join(CANDIDATE_FREQUENCIES)}; other settings "
        "come from the sidebar. The sidebar start and end dates are not used here."
    )
    if st.button("Run walk-forward validation", key="run_walk_forward"):
        st.session_state["walk_forward_for"] = (settings, benchmark)
    if st.session_state.get("walk_forward_for") != (settings, benchmark):
        st.info("Press the button to run walk-forward validation for the current settings.")
        return

    folds, chained = run_validation(market_data, settings, benchmark.fixed_nifty_weight)
    names = portfolio_order(benchmark.fixed_nifty_weight)
    show(plot_walk_forward(folds, chained, names))

    st.subheader("Chained out-of-sample performance")
    summary = out_of_sample_summary(chained[names], settings.backtest.cash_annual_rate)
    st.table(format_metrics_table(summary))

    st.subheader("Selected setting and return in each test year")
    table = folds[["selected"] + [f"{name}_return" for name in names]].copy()
    table[[f"{name}_return" for name in names]] = format_percent_table(
        table[[f"{name}_return" for name in names]])
    table.columns = ["Selected setting"] + [display_name(name) for name in names]
    st.dataframe(table)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="VIX Dynamic NIFTY Allocation", layout="wide")
    st.title("India VIX–based dynamic NIFTY 50 allocation")

    market_data, dropped = load_market_data()
    settings, benchmark = sidebar_settings(market_data)
    try:
        histories, comparison = run_comparison(market_data, settings,
                                               benchmark.fixed_nifty_weight)
    except ValueError as error:
        st.error(f"Cannot run the backtest with these settings: {error}")
        st.stop()

    tabs = st.tabs(["Overview", "Data", "Strategy", "Backtest", "Risk & Performance",
                    "Comparison", "Experiments", "Walk-forward"])
    with tabs[0]:
        overview_tab(settings, comparison)
    with tabs[1]:
        data_tab(market_data, dropped)
    with tabs[2]:
        strategy_tab(settings, market_data)
    with tabs[3]:
        backtest_tab(histories[STRATEGY_NAME])
    with tabs[4]:
        risk_tab(histories, comparison)
    with tabs[5]:
        comparison_tab(histories, comparison)
    with tabs[6]:
        experiments_tab(market_data, settings, benchmark)
    with tabs[7]:
        walk_forward_tab(market_data, settings, benchmark)


main()
