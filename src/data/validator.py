"""Data validation: check that the aligned dataset is safe for the strategy layer.

`validate_market_data` raises a ValueError listing every problem it finds.
`describe_market_data` produces simple descriptive statistics for reporting.
"""

import pandas as pd

REQUIRED_COLUMNS = ["vix", "nifty_open", "nifty", "nifty_return"]
PRICE_COLUMNS = ["vix", "nifty_open", "nifty"]

# Sanity bounds. These are not filters: data outside them stops the pipeline
# so a human can look at it. India VIX traded roughly 9-84 during 2015-2025,
# and NIFTY's largest daily move was about -13% (23 March 2020).
VIX_MIN_PLAUSIBLE = 5.0
VIX_MAX_PLAUSIBLE = 150.0
MAX_ABS_DAILY_RETURN = 0.20


def find_validation_errors(market_data: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    errors = []

    if market_data.empty:
        return ["Dataset is empty."]

    missing = [column for column in REQUIRED_COLUMNS if column not in market_data.columns]
    if missing:
        return [f"Missing required columns: {missing}"]

    index = market_data.index
    if not isinstance(index, pd.DatetimeIndex):
        errors.append("Index must be a DatetimeIndex of dates.")
    if index.has_duplicates:
        errors.append(f"Duplicate dates found: {index[index.duplicated()].tolist()[:5]}")
    if not index.is_monotonic_increasing:
        errors.append("Dates are not in chronological order.")

    for column in REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(market_data[column]):
            errors.append(f"Column '{column}' is not numeric.")
    if errors:
        return errors

    for column in PRICE_COLUMNS:
        if market_data[column].isna().any():
            errors.append(f"Column '{column}' has missing values.")
        if (market_data[column] <= 0).any():
            errors.append(f"Column '{column}' has zero or negative values.")

    # Only the first return is allowed to be missing (no previous close).
    if market_data["nifty_return"].iloc[1:].isna().any():
        errors.append("Column 'nifty_return' has missing values after the first row.")

    vix = market_data["vix"]
    if (vix < VIX_MIN_PLAUSIBLE).any() or (vix > VIX_MAX_PLAUSIBLE).any():
        errors.append(
            f"VIX outside plausible range [{VIX_MIN_PLAUSIBLE}, {VIX_MAX_PLAUSIBLE}]: "
            f"min={vix.min():.2f}, max={vix.max():.2f}"
        )

    overnight_gap = market_data["nifty_open"] / market_data["nifty"].shift(1) - 1
    largest_gap = overnight_gap.abs().max()
    if largest_gap > MAX_ABS_DAILY_RETURN:
        errors.append(
            f"NIFTY overnight gap (open vs previous close) of {largest_gap:.1%} exceeds "
            f"the {MAX_ABS_DAILY_RETURN:.0%} sanity limit."
        )

    largest_move = market_data["nifty_return"].abs().max()
    if largest_move > MAX_ABS_DAILY_RETURN:
        errors.append(
            f"NIFTY daily return of {largest_move:.1%} exceeds the "
            f"{MAX_ABS_DAILY_RETURN:.0%} sanity limit."
        )

    return errors


def validate_market_data(market_data: pd.DataFrame) -> None:
    """Raise ValueError if the dataset fails any validation check."""
    errors = find_validation_errors(market_data)
    if errors:
        raise ValueError("Market data failed validation:\n- " + "\n- ".join(errors))


def describe_market_data(market_data: pd.DataFrame) -> pd.DataFrame:
    """Basic descriptive statistics for VIX, NIFTY and NIFTY daily returns."""
    summary = market_data[REQUIRED_COLUMNS].describe().T
    summary["unchanged_days"] = (market_data[PRICE_COLUMNS].diff() == 0).sum()
    return summary


def describe_coverage(market_data: pd.DataFrame) -> dict:
    """Date coverage of the dataset."""
    return {
        "first_date": market_data.index.min().date(),
        "last_date": market_data.index.max().date(),
        "trading_days": len(market_data),
    }
