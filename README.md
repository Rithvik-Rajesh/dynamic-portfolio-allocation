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

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # create .venv and install dependencies
uv run python main.py        # run the pipeline (uses cached raw data)
uv run pytest                # run the tests
```

Without uv: `pip install -r requirements.txt pytest`, then `python main.py`.

`python main.py --refresh` re-downloads the raw data from Yahoo Finance.

## Repository Structure

```text
main.py                       Runs the pipeline built so far
src/config.py                 Paths, tickers, study period
src/data/loader.py            Download + cache raw data (Milestone 1)
src/data/cleaner.py           Clean and align VIX/NIFTY (Milestone 2)
src/data/validator.py         Validation checks + descriptive stats (Milestone 2)
src/config.py                 (also) output file locations
src/rates.py                  Cash / risk-free rate conversion (shared)
src/strategy/config.py        ALL strategy + backtest assumptions
src/strategy/regimes.py       Expanding VIX percentile + regimes (Milestone 3)
src/strategy/allocation.py    Regime -> target NIFTY/cash weight (Milestone 4)
src/backtesting/engine.py     Day-by-day portfolio simulation (Milestone 5)
src/backtesting/portfolio.py  Rebalancing arithmetic and drift rule
src/backtesting/costs.py      Transaction-cost model
src/analysis/performance.py   Return, Sharpe, Sortino, trading metrics (Milestone 6)
src/analysis/risk.py          Volatility, downside deviation, drawdowns
src/analysis/regime_analysis.py   Regime frequency and NIFTY behaviour by regime
src/visualization/charts.py   Charts
tests/                        Unit tests
data/raw/                     Cached raw downloads + metadata.json (committed)
data/processed/               Generated datasets (not committed)
reports/figures/              Generated charts (not committed)
```

## Data

| Series | Yahoo ticker | Period |
|---|---|---|
| India VIX | `^INDIAVIX` | 2015-01-01 to 2025-12-31 |
| NIFTY 50 | `^NSEI` | 2015-01-02 to 2025-12-31 |

Source and download time are recorded in `data/raw/metadata.json`.

## Methodology Notes and Assumptions

- **Closing values only.** Daily close of VIX and NIFTY.
- **Alignment.** A date is kept only if both series have a close (inner join). Prices are never forward-filled. NIFTY returns are calculated after alignment, so no price movement is lost when a date is dropped.
- **Price index.** `^NSEI` is the NIFTY 50 price index; dividends are not included. This understates equity returns and will slightly favour strategies that hold less equity. This is a known limitation.
- **VIX percentile.** Expanding percentile: on day *t*, the share of all VIX closes from the start of the data up to and including day *t* that are ≤ VIX(*t*). No future data is used. The first 252 trading days are a warm-up period with no regime, so regimes start in January 2016.
- **Regimes** (configurable in `src/strategy/config.py`): low < 25th percentile ≤ normal < 75th ≤ high < 95th ≤ extreme.
- **Timing.** A regime is known only at the close of day *t*, so it can only affect returns from day *t+1* onwards.

### Strategy and backtest defaults (`src/strategy/config.py`)

| Assumption | Default | Notes |
|---|---|---|
| Allocation by regime | Low 100% · Normal 75% · High 50% · Extreme 25% | Rest in cash. No leverage, no shorting. |
| Execution | Next day's close | Regime from close *t* is traded at close *t+1*. |
| Rebalancing | Weekly | Trades only on the first trading day of each week (or month / every day). |
| Drift band | 5 weight points | On a rebalance day, trade if the target changed or the actual weight is more than 5 points off target. |
| Transaction cost | 0.10% of traded value | One simplified rate for brokerage, taxes, fees and slippage. Real Indian costs vary by broker, instrument and size. |
| Cash return / risk-free rate | 6% p.a. | Approximates Indian T-bill / liquid-fund yields 2015–2025 (actual yields varied, roughly 3–8%). Accrues per calendar day. |
| Initial capital | ₹100,000 | Starts in cash; the first trade invests it. |

Trades are sized so the weight is exactly on target after costs, and costs are paid from cash.

### Metrics (`src/analysis/`)

- **Total return / CAGR**: compounded daily returns; CAGR uses calendar years (365.25 days).
- **Volatility**: daily standard deviation × √252.
- **Sharpe**: mean daily excess return over the cash rate ÷ its standard deviation × √252.
- **Sortino**: mean daily excess return ÷ downside deviation × √252.
- **Maximum drawdown**: largest fall from a running peak of portfolio value, with peak, trough and recovery dates.
- **Trading**: number of trades, annual turnover (traded value ÷ average portfolio value per year), total costs. The initial investment counts as a trade.
