# India VIX-Based Dynamic NIFTY 50 Allocation

A Financial Engineering project that uses **India VIX as a market-risk signal** to dynamically adjust NIFTY 50 exposure.

The project evaluates the strategy through **historical backtesting, risk-adjusted performance analysis, benchmark comparison, and walk-forward validation**.

---

## Overview

Market volatility changes over time, and a fixed equity allocation does not adapt to these changing conditions.

This project investigates whether **India VIX can be used to systematically adjust NIFTY 50 exposure** based on the prevailing volatility regime.

The strategy converts the historical position of India VIX into volatility regimes and maps those regimes to predefined NIFTY 50 allocations, with the remaining portfolio held as cash.

### Core Flow

```text
India VIX + NIFTY 50 Data
          ↓
     Data Processing
          ↓
   VIX Percentile
          ↓
    VIX Regime
          ↓
 Target NIFTY Allocation
          ↓
    Rebalancing
          ↓
      Backtesting
          ↓
 Performance & Risk Analysis
          ↓
 Benchmark Comparison
```

---

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/). The raw data is committed, so no download is needed.

```bash
uv sync                                  # create .venv and install dependencies
uv run streamlit run app/streamlit_app.py   # open the dashboard
uv run python main.py                    # run the full research pipeline, save tables + charts
uv run pytest                            # run the tests (~35 s; dashboard tests are the slowest)
```

Without uv: `pip install -r requirements.txt pytest`, then use `streamlit run ...`, `python main.py` and `pytest` directly.

`python main.py --refresh` re-downloads the raw data from Yahoo Finance.

## Dashboard

`app/streamlit_app.py`. All settings are in the sidebar: investment amount and period, percentile window (expanding / rolling and its length), warm-up, regime thresholds, NIFTY weight per regime, rebalancing frequency, drift band, next-day open or close execution, transaction cost, cash rate, and the fixed-benchmark weight. Invalid combinations show an error instead of running.

| Tab | Shows |
|---|---|
| Overview | Research question, how the strategy works, headline result vs buy-and-hold, current settings |
| Data | Source, coverage, NIFTY and VIX history, descriptive statistics |
| Strategy | Allocation rule, regimes over time, VIX percentile, NIFTY behaviour by regime |
| Backtest | Portfolio value, target vs actual NIFTY weight, drawdown, trade list, CSV download |
| Risk & Performance | All metrics, calendar-year returns |
| Comparison | Strategy vs buy-and-hold vs fixed allocation |
| Experiments | One-at-a-time sensitivity experiments around the current settings (button, ~20 s) |
| Walk-forward | Out-of-sample validation around the current settings (button, ~10 s) |

The app has no financial logic of its own; it calls the same functions as `main.py`.

## Repository Structure

```text
main.py                       Runs the full pipeline, saves tables and charts
app/streamlit_app.py          Dashboard (Milestone 10)
src/config.py                 Paths, tickers, study period, output locations
src/data/loader.py            Download + cache raw data (Milestone 1)
src/data/cleaner.py           Clean and align VIX/NIFTY (Milestone 2)
src/data/validator.py         Validation checks + descriptive stats (Milestone 2)
src/rates.py                  Cash / risk-free rate conversion (shared)
src/pipeline.py               run_vix_strategy(): one call, any settings
src/strategy/config.py        ALL strategy, backtest and benchmark assumptions
src/strategy/regimes.py       VIX percentile (expanding/rolling) + regimes (Milestone 3)
src/strategy/allocation.py    Regime -> target NIFTY/cash weight (Milestone 4)
src/backtesting/engine.py     Day-by-day portfolio simulation (Milestone 5)
src/backtesting/portfolio.py  Rebalancing arithmetic and drift rule
src/backtesting/costs.py      Transaction-cost model
src/analysis/performance.py   Return, Sharpe, Sortino, trading metrics (Milestone 6)
src/analysis/risk.py          Volatility, downside deviation, drawdowns
src/analysis/regime_analysis.py   Regime frequency and NIFTY behaviour by regime
src/analysis/benchmarks.py    Buy-and-hold and fixed-allocation benchmarks (Milestone 7)
src/analysis/sensitivity.py   One-at-a-time parameter experiments (Milestone 8)
src/analysis/walk_forward.py  Walk-forward validation (Milestone 9)
src/visualization/charts.py   Charts (matplotlib)
src/visualization/tables.py   Display formatting for metric tables
tests/                        Unit, look-ahead, end-to-end and dashboard tests
data/raw/                     Cached raw downloads + metadata.json
data/processed/               Generated datasets
reports/figures/              Generated charts
reports/tables/               Generated result tables (CSV)
```

## Testing

`uv run pytest` runs 111 tests. Besides hand-calculated checks of every metric and of the cost and rebalancing arithmetic, they include:

- **Look-ahead tests**: replacing all data after a cut-off date must not change any VIX percentile, any walk-forward selection, or any row of a full backtest up to that date.
- **Timing tests**: a signal can never earn the market move of the day it was generated.
- **Benchmark tests**: benchmarks use exactly the strategy's dates; buy-and-hold tracks NIFTY.
- **Dashboard tests**: the app runs headlessly, reacts to its inputs, and shows an error for invalid settings.

## Data

| Series | Yahoo ticker | Period |
|---|---|---|
| India VIX | `^INDIAVIX` | 2015-01-01 to 2025-12-31 |
| NIFTY 50 | `^NSEI` | 2015-01-02 to 2025-12-31 |

Source and download time are recorded in `data/raw/metadata.json`.

## Methodology Notes and Assumptions

- **Prices used.** Daily VIX close; NIFTY close (and NIFTY open, only when trading at the open).
- **Alignment.** A date is kept only if both series have a close (inner join). Prices are never forward-filled. NIFTY returns are calculated after alignment, so no price movement is lost when a date is dropped.
- **Price index.** `^NSEI` is the NIFTY 50 price index; dividends are not included. This understates equity returns and will slightly favour strategies that hold less equity. This is a known limitation.
- **VIX percentile.** On day *t*, the share of VIX closes in the comparison window that are ≤ VIX(*t*). The window always ends on day *t*, so no future data is used. **Expanding** (default): all VIX history since the start of the data. **Rolling**: only the last *N* trading days. The first 252 trading days are a warm-up period with no regime, so regimes start in January 2016.
- **Regimes** (configurable in `src/strategy/config.py`): low < 25th percentile ≤ normal < 75th ≤ high < 95th ≤ extreme.
- **Timing.** A regime is known only at the close of day *t*, so it can only affect returns from day *t+1* onwards.

### Strategy and backtest defaults (`src/strategy/config.py`)

| Assumption | Default | Notes |
|---|---|---|
| Regime thresholds | 25th / 75th / 95th percentile | `RegimeConfig.low_threshold`, `high_threshold`, `extreme_threshold`. |
| Percentile window | Expanding | `RegimeConfig.percentile_window` = `"expanding"` or `"rolling"`; `rolling_window` = 504 trading days (≈2 years) when rolling. |
| Warm-up | 252 trading days | `RegimeConfig.min_history`. Must not exceed the rolling window. |
| Allocation by regime | Low 100% · Normal 75% · High 50% · Extreme 25% | `AllocationConfig`. Rest in cash. No leverage, no shorting. |
| Execution | Next day's close | `BacktestConfig.execution_price` = `"close"` or `"open"`; `execution_lag_days` = 1. Trading at the open needs a lag of at least 1. |
| Rebalancing | Weekly | Trades only on the first trading day of each week (or month / every day). |
| Drift band | 5 weight points | On a rebalance day, trade if the target changed or the actual weight is more than 5 points off target. |
| Transaction cost | 0.10% of traded value | One simplified rate for brokerage, taxes, fees and slippage. Real Indian costs vary by broker, instrument and size. |
| Cash return / risk-free rate | 6% p.a. | Approximates Indian T-bill / liquid-fund yields 2015–2025 (actual yields varied, roughly 3–8%). Accrues per calendar day. |
| Initial capital | ₹100,000 | Starts in cash; the first trade invests it. |
| Investment period | Whole dataset | `BacktestConfig.start_date` / `end_date` (`"YYYY-MM-DD"`). The VIX percentile still uses all VIX history before the start date, since it was known at the time. |

Trades are sized so the weight is exactly on target after costs, and costs are paid from cash.

All settings are passed to `src.pipeline.run_vix_strategy(market_data, regime_config, allocation_config, backtest_config)`. In `main.py` they are set at the top of `main()`.

### Metrics (`src/analysis/`)

- **Total return / CAGR**: compounded daily returns; CAGR uses calendar years (365.25 days).
- **Volatility**: daily standard deviation × √252.
- **Sharpe**: mean daily excess return over the cash rate ÷ its standard deviation × √252.
- **Sortino**: mean daily excess return ÷ downside deviation × √252.
- **Maximum drawdown**: largest fall from a running peak of portfolio value, with peak, trough and recovery dates.
- **Trading**: number of trades, annual turnover (traded value ÷ average portfolio value per year), total costs. The initial investment counts as a trade.

### Benchmarks (`src/analysis/benchmarks.py`)

- **Buy-and-hold**: 100% NIFTY from the first day.
- **Fixed allocation**: constant NIFTY weight (default 75%, `BenchmarkConfig.fixed_nifty_weight`), rebalanced with the same schedule and drift band as the strategy.
- Benchmarks run through the same engine with the same `BacktestConfig` (capital, costs, cash rate, execution) over exactly the strategy's dates. Only the target weight differs.

### Sensitivity experiments (`src/analysis/sensitivity.py`)

One setting is changed at a time from the base settings: regime thresholds, allocation levels, percentile window, rebalancing frequency, drift band, transaction cost, cash rate, execution price and start year. Every variant is compared with the benchmarks over the same dates. The values tried are listed at the top of the module. The experiments measure robustness; they do not pick a best setting.

### Walk-forward validation (`src/analysis/walk_forward.py`)

For each test year 2019–2025: backtest 16 candidate settings (high threshold 70/75/80/90% × expanding/rolling-504d window × weekly/monthly rebalancing) on all data up to the previous year-end, freeze the one with the best Sharpe ratio, and test it on the unseen year. The fixed base rules and the benchmarks are tested on the same year. Each test year starts in cash on its first trading day (so all portfolios miss that day's move and pay one initial cost per year); yearly results are chained into one out-of-sample record.
