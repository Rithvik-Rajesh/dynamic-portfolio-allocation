"""Static charts for the VIX regime analysis (matplotlib).

Each function returns a matplotlib Figure so it can be saved to a file now
and shown in the dashboard later.
"""

import matplotlib

matplotlib.use("Agg")  # Draw to files; no window needed.

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.risk import drawdown_series
from src.strategy.config import REGIME_LABELS, RegimeConfig

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
    for (start, label), end in zip(block_starts.items(), block_ends):
        axis.axvspan(start, end, color=REGIME_COLOURS[label], alpha=0.25, linewidth=0)


def _regime_legend(axis) -> None:
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=REGIME_COLOURS[label], alpha=0.4)
        for label in REGIME_LABELS
    ]
    axis.legend(handles, [label.capitalize() for label in REGIME_LABELS],
                loc="upper left", ncol=4, frameon=False, fontsize=9)


def plot_market_with_regimes(regime_data: pd.DataFrame):
    """NIFTY (top) and India VIX (bottom), background shaded by VIX regime."""
    figure, (nifty_axis, vix_axis) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )

    for axis in (nifty_axis, vix_axis):
        _shade_regimes(axis, regime_data["vix_regime"])
        axis.grid(alpha=0.3)

    nifty_axis.plot(regime_data.index, regime_data["nifty"], color="black", linewidth=1)
    nifty_axis.set_ylabel("NIFTY 50")
    nifty_axis.set_title("NIFTY 50 and India VIX, shaded by VIX regime")
    _regime_legend(nifty_axis)

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


def save_figure(figure, path) -> None:
    """Save a figure as PNG and release its memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
