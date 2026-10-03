"""Turn metric tables into readable display tables (percentages, rupees, ratios).

Used by the dashboard. Formatting is kept separate from the calculations so
the numbers in `src/analysis/` stay plain floats.
"""

import pandas as pd

# metric name -> (display name, format)
METRIC_FORMATS = {
    "start_date": ("Start date", "text"),
    "end_date": ("End date", "text"),
    "final_value": ("Final value", "rupees"),
    "total_return": ("Total return", "percent"),
    "cagr": ("CAGR", "percent"),
    "annualised_volatility": ("Annualised volatility", "percent"),
    "sharpe_ratio": ("Sharpe ratio", "ratio"),
    "sortino_ratio": ("Sortino ratio", "ratio"),
    "max_drawdown": ("Maximum drawdown", "percent"),
    "max_drawdown_peak": ("Drawdown peak", "text"),
    "max_drawdown_trough": ("Drawdown trough", "text"),
    "max_drawdown_recovery": ("Drawdown recovered", "text"),
    "number_of_trades": ("Number of trades", "integer"),
    "annual_turnover": ("Annual turnover", "multiple"),
    "total_transaction_costs": ("Total transaction costs", "rupees"),
    "costs_pct_of_initial_capital": ("Costs as % of initial capital", "percent"),
    "average_nifty_weight": ("Average NIFTY weight", "percent"),
}


def format_value(value, kind: str) -> str:
    """Format one value. Missing values become an em dash."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if kind == "percent":
        return f"{value:.2%}"
    if kind == "rupees":
        return f"₹{value:,.0f}"
    if kind == "ratio":
        return f"{value:.2f}"
    if kind == "multiple":
        return f"{value:.2f}×"
    if kind == "integer":
        return f"{int(value):,}"
    return str(value)


def format_metrics_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Metrics with one ROW per portfolio -> display table with one COLUMN per
    portfolio and readable metric names."""
    rows = {}
    for metric, (label, kind) in METRIC_FORMATS.items():
        if metric in metrics.columns:
            rows[label] = [format_value(value, kind) for value in metrics[metric]]
    return pd.DataFrame(rows, index=[display_name(name) for name in metrics.index]).T


def display_name(name: str) -> str:
    """'buy_and_hold' -> 'Buy and hold', 'vix_strategy' -> 'VIX strategy'."""
    return name.replace("_", " ").capitalize().replace("Vix", "VIX")


def format_percent_table(table: pd.DataFrame) -> pd.DataFrame:
    """Every cell as a percentage string."""
    return table.apply(lambda column: column.map(lambda value: format_value(value, "percent")))
