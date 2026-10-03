"""Static charts (matplotlib): VIX regimes, backtest, benchmarks, experiments
and walk-forward validation.

Each function returns a matplotlib Figure so it can be saved to a file now
and shown in the dashboard later.
"""

import matplotlib

matplotlib.use("Agg")  # Draw to files; no window needed.

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.risk import drawdown_series
from src.strategy.config import REGIME_LABELS, RegimeConfig
from src.visualization.tables import display_name

# One colour per portfolio, used in every comparison chart.
PORTFOLIO_COLOURS = {
    "vix_strategy": "#000000",
    "walk_forward_selected": "#000000",
    "fixed_base_rules": "#7f7f7f",
    "buy_and_hold": "#2f6db3",
}
OTHER_PORTFOLIO_COLOUR = "#b0b0b0"  # fixed-allocation benchmark

REGIME_COLOURS = {
    "low": "#4c9f70",
    "normal": "#9e9e9e",
    "high": "#e09f3e",
    "extreme": "#c0392b",
}


def _shade_regimes(axis, regime: pd.Series) -> None:
    """Shade the background of a time-series axis by regime."""
    regime = regime.dropna()
    if regime.empty:
        return

    # A block starts on every day where the regime differs from the day before
    # and lasts until the next block starts (or the last date).
    block_starts = regime[regime != regime.shift()]
    block_ends = list(block_starts.index[1:]) + [regime.index[-1]]
    for (start, label), end in zip(block_starts.items(), block_ends, strict=True):
        axis.axvspan(start, end, color=REGIME_COLOURS[label], alpha=0.25, linewidth=0)


def _regime_legend(axis) -> None:
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=REGIME_COLOURS[label], alpha=0.4)
        for label in REGIME_LABELS
    ]
    axis.legend(handles, [label.capitalize() for label in REGIME_LABELS],
                loc="upper left", ncol=4, frameon=False, fontsize=9)


def plot_market_with_regimes(regime_data: pd.DataFrame, show_regimes: bool = True):
    """NIFTY (top) and India VIX (bottom), optionally shaded by VIX regime."""
    figure, (nifty_axis, vix_axis) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )

    for axis in (nifty_axis, vix_axis):
        if show_regimes:
            _shade_regimes(axis, regime_data["vix_regime"])
        axis.grid(alpha=0.3)

    nifty_axis.plot(regime_data.index, regime_data["nifty"], color="black", linewidth=1)
    nifty_axis.set_ylabel("NIFTY 50")
    if show_regimes:
        nifty_axis.set_title("NIFTY 50 and India VIX, shaded by VIX regime")
        _regime_legend(nifty_axis)
    else:
        nifty_axis.set_title("NIFTY 50 and India VIX (daily close)")

    vix_axis.plot(regime_data.index, regime_data["vix"], color="black", linewidth=1)
    vix_axis.set_ylabel("India VIX")

    figure.tight_layout()
    return figure


def plot_vix_percentile(regime_data: pd.DataFrame, config: RegimeConfig):
    """Expanding VIX percentile with the regime thresholds marked."""
    figure, axis = plt.subplots(figsize=(12, 4))

    axis.plot(regime_data.index, regime_data["vix_percentile"], color="black", linewidth=0.8)
    for threshold in (config.low_threshold, config.high_threshold, config.extreme_threshold):
        axis.axhline(threshold, color="#555555", linestyle="--", linewidth=0.8)
        axis.text(regime_data.index[-1], threshold, f" {threshold:.0%}",
                  va="center", fontsize=9, color="#555555")

    axis.set_ylim(0, 1.02)
    axis.set_ylabel("VIX percentile (expanding)")
    axis.set_title("Expanding VIX percentile and regime thresholds")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def plot_regime_summary(frequency: pd.DataFrame, behaviour: pd.DataFrame):
    """Left: share of days per regime. Right: annualised next-day NIFTY volatility."""
    figure, (share_axis, vol_axis) = plt.subplots(1, 2, figsize=(11, 4))
    colours = [REGIME_COLOURS[label] for label in REGIME_LABELS]
    names = [label.capitalize() for label in REGIME_LABELS]

    share_axis.bar(names, frequency.loc[REGIME_LABELS, "share"] * 100, color=colours)
    share_axis.set_ylabel("% of trading days")
    share_axis.set_title("How often each regime occurred")

    vol_axis.bar(names, behaviour.loc[REGIME_LABELS, "annualised_volatility"] * 100, color=colours)
    vol_axis.set_ylabel("Annualised volatility (%)")
    vol_axis.set_title("NIFTY volatility on the day after each regime")

    for axis in (share_axis, vol_axis):
        axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    return figure


def plot_backtest(history: pd.DataFrame):
    """Portfolio value, NIFTY weight (target vs actual) and drawdown over time."""
    figure, (value_axis, weight_axis, drawdown_axis) = plt.subplots(
        3, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2]}
    )

    value_axis.plot(history.index, history["portfolio_value"], color="black", linewidth=1)
    value_axis.set_ylabel("Portfolio value (₹)")
    value_axis.set_title("VIX strategy backtest")

    weight_axis.step(history.index, history["tradable_target_weight"] * 100, where="post",
                     color="#555555", linewidth=0.8, linestyle="--", label="Target")
    weight_axis.plot(history.index, history["nifty_weight"] * 100, color="black",
                     linewidth=0.8, label="Actual")
    weight_axis.set_ylabel("NIFTY weight (%)")
    weight_axis.set_ylim(0, 105)
    weight_axis.legend(loc="lower left", frameon=False, fontsize=9)

    drawdown = drawdown_series(history["portfolio_value"]) * 100
    drawdown_axis.fill_between(history.index, drawdown, 0, color="#c0392b", alpha=0.4, linewidth=0)
    drawdown_axis.set_ylabel("Drawdown (%)")

    for axis in (value_axis, weight_axis, drawdown_axis):
        axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def _portfolio_colour(name: str) -> str:
    return PORTFOLIO_COLOURS.get(name, OTHER_PORTFOLIO_COLOUR)


def plot_benchmark_comparison(values: pd.DataFrame):
    """Growth of each portfolio (top, log scale) and its drawdown (bottom)."""
    figure, (value_axis, drawdown_axis) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    for name in values.columns:
        colour = _portfolio_colour(name)
        value_axis.plot(values.index, values[name], color=colour, linewidth=1,
                        label=display_name(name))
        drawdown = drawdown_series(values[name]) * 100
        drawdown_axis.plot(values.index, drawdown, color=colour, linewidth=0.8)

    value_axis.set_yscale("log")
    value_axis.set_ylabel("Portfolio value (₹, log scale)")
    value_axis.set_title("VIX strategy vs benchmarks (same dates, costs and settings)")
    value_axis.legend(loc="upper left", frameon=False)
    drawdown_axis.set_ylabel("Drawdown (%)")
    for axis in (value_axis, drawdown_axis):
        axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def plot_sensitivity_overview(experiments: pd.DataFrame):
    """For every experiment variant: strategy minus buy-and-hold in CAGR,
    Sharpe and max drawdown. Bars to the right = strategy better."""
    columns = [
        ("cagr_vs_buy_and_hold", "CAGR difference (pct points)", 100),
        ("sharpe_vs_buy_and_hold", "Sharpe ratio difference", 1),
        ("max_drawdown_vs_buy_and_hold", "Max drawdown difference (pct points)", 100),
    ]
    labels = [f"{row.experiment}: {row.variant}" for row in experiments.itertuples()]
    positions = range(len(experiments))

    figure, axes = plt.subplots(1, 3, figsize=(15, 0.28 * len(experiments) + 1.5), sharey=True)
    for axis, (column, title, scale) in zip(axes, columns, strict=True):
        gaps = experiments[column].astype(float) * scale
        colours = ["#000000" if is_base else "#8c8c8c" for is_base in experiments["is_base"]]
        axis.barh(positions, gaps, color=colours, height=0.7)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(title, fontsize=10)
        axis.grid(axis="x", alpha=0.3)

    axes[0].set_yticks(list(positions), labels, fontsize=8)
    axes[0].invert_yaxis()
    figure.suptitle("Strategy minus buy-and-hold for each setting (black = base settings; "
                    "right of zero = strategy better)", fontsize=11)
    figure.tight_layout()
    return figure


def plot_walk_forward(folds: pd.DataFrame, chained_returns: pd.DataFrame, names: list[str]):
    """Left: each portfolio's return in every test year. Right: the chained
    out-of-sample growth of ₹1."""
    figure, (year_axis, growth_axis) = plt.subplots(1, 2, figsize=(14, 5))

    bar_width = 0.8 / len(names)
    years = list(folds.index)
    for i, name in enumerate(names):
        offsets = [position + (i - (len(names) - 1) / 2) * bar_width
                   for position in range(len(years))]
        year_axis.bar(offsets, folds[f"{name}_return"] * 100, width=bar_width,
                      color=_portfolio_colour(name), label=display_name(name))
        growth = (1 + chained_returns[name]).cumprod()
        line_style = "--" if name == "fixed_base_rules" else "-"
        growth_axis.plot(growth.index, growth, color=_portfolio_colour(name), linewidth=1,
                         linestyle=line_style, label=display_name(name))

    year_axis.set_xticks(range(len(years)), years)
    year_axis.axhline(0, color="black", linewidth=0.8)
    year_axis.set_ylabel("Return in test year (%)")
    year_axis.set_title("Out-of-sample return in each test year")
    year_axis.legend(frameon=False, fontsize=8)
    growth_axis.set_ylabel("Growth of ₹1")
    growth_axis.set_title("Chained out-of-sample performance")
    for axis in (year_axis, growth_axis):
        axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def plot_calendar_year_returns(yearly_returns: pd.DataFrame):
    """Grouped bars: each portfolio's return in each calendar year."""
    figure, axis = plt.subplots(figsize=(12, 4.5))
    names = list(yearly_returns.columns)
    bar_width = 0.8 / len(names)
    for i, name in enumerate(names):
        offsets = [position + (i - (len(names) - 1) / 2) * bar_width
                   for position in range(len(yearly_returns))]
        axis.bar(offsets, yearly_returns[name] * 100, width=bar_width,
                 color=_portfolio_colour(name), label=display_name(name))

    axis.set_xticks(range(len(yearly_returns)), yearly_returns.index)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Return (%)")
    axis.set_title("Calendar-year returns (first and last years may be partial)")
    axis.legend(frameon=False, fontsize=9)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    return figure


def save_figure(figure, path) -> None:
    """Save a figure as PNG and release its memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
