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
