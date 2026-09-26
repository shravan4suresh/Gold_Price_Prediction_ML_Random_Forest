"""Data loading and cleaning utilities for the gold price project."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["Date", "Close"]
OPTIONAL_NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class CleaningReport:
    """Human-readable details about what was found and changed."""

    original_shape: tuple[int, int]
    cleaned_shape: tuple[int, int]
    original_missing_values: pd.Series
    missing_percentages: pd.Series
    duplicate_rows_found: int
    duplicate_dates_found: int
    invalid_numeric_values: dict[str, int]
    actions: list[str]


def load_gold_data(csv_file) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(csv_file)


def inspect_dataset(gold_data: pd.DataFrame) -> dict[str, object]:
    """Return basic inspection outputs used in notebooks and Streamlit."""
    return {
        "first_observations": gold_data.head(),
        "last_observations": gold_data.tail(),
        "shape": gold_data.shape,
        "columns": list(gold_data.columns),
        "data_types": gold_data.dtypes.astype(str),
    }


def validate_required_columns(gold_data: pd.DataFrame) -> None:
    """Ensure the minimum columns needed for this project are available."""
    missing_required_columns = [
        column for column in REQUIRED_COLUMNS if column not in gold_data.columns
    ]
    if missing_required_columns:
        raise ValueError(
            "Missing required column(s): " + ", ".join(missing_required_columns)
        )


def _standardize_column_names(gold_data: pd.DataFrame) -> pd.DataFrame:
    """Trim extra spaces from column names while keeping familiar names."""
    cleaned_data = gold_data.copy()
    cleaned_data.columns = [str(column).strip() for column in cleaned_data.columns]
    return cleaned_data


def clean_gold_data(gold_data: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Clean historical gold data and document each cleaning action.

    The function avoids silent changes by returning a CleaningReport. Rows with
    invalid dates or missing target prices cannot be used for modeling, so they
    are removed and reported.
    """
    cleaned_data = _standardize_column_names(gold_data)
    validate_required_columns(cleaned_data)

    original_shape = cleaned_data.shape
    original_missing_values = cleaned_data.isna().sum()
    missing_percentages = (cleaned_data.isna().mean() * 100).round(2)
    duplicate_rows_found = int(cleaned_data.duplicated().sum())
    duplicate_dates_found = int(cleaned_data.duplicated(subset=["Date"]).sum())
    invalid_numeric_values: dict[str, int] = {}
    actions: list[str] = []

    cleaned_data["Date"] = pd.to_datetime(cleaned_data["Date"], errors="coerce")
    invalid_dates = int(cleaned_data["Date"].isna().sum())
    if invalid_dates:
        cleaned_data = cleaned_data.dropna(subset=["Date"])
        actions.append(f"Removed {invalid_dates} row(s) with invalid Date values.")
    else:
        actions.append("Converted Date column to datetime format.")

    numeric_columns = [
        column for column in OPTIONAL_NUMERIC_COLUMNS if column in cleaned_data.columns
    ]
    for column in numeric_columns:
        before_missing = int(cleaned_data[column].isna().sum())
        cleaned_data[column] = pd.to_numeric(cleaned_data[column], errors="coerce")
        after_missing = int(cleaned_data[column].isna().sum())
        invalid_numeric_values[column] = max(after_missing - before_missing, 0)

    total_invalid_numeric = sum(invalid_numeric_values.values())
    if total_invalid_numeric:
        actions.append(
            f"Converted non-numeric price/volume values to NaN "
            f"({total_invalid_numeric} value(s))."
        )
    else:
        actions.append("Checked numeric columns; no invalid numeric values found.")

    if duplicate_rows_found:
        cleaned_data = cleaned_data.drop_duplicates()
        actions.append(f"Removed {duplicate_rows_found} exact duplicate row(s).")
    else:
        actions.append("Checked exact duplicate rows; none found.")

    duplicate_dates_after_row_drop = int(cleaned_data.duplicated(subset=["Date"]).sum())
    if duplicate_dates_after_row_drop:
        cleaned_data = cleaned_data.sort_values("Date").drop_duplicates(
            subset=["Date"], keep="last"
        )
        actions.append(
            f"Removed {duplicate_dates_after_row_drop} duplicate date row(s), "
            "keeping the latest occurrence after sorting."
        )
    else:
        actions.append("Checked duplicate dates; none found.")

    missing_close = int(cleaned_data["Close"].isna().sum())
    if missing_close:
        cleaned_data = cleaned_data.dropna(subset=["Close"])
        actions.append(
            f"Removed {missing_close} row(s) with missing Close values because "
            "the target price cannot be learned without them."
        )

    cleaned_data = cleaned_data.sort_values("Date").reset_index(drop=True)
    actions.append("Sorted observations chronologically from oldest to newest.")

    cleaned_shape = cleaned_data.shape
    report = CleaningReport(
        original_shape=original_shape,
        cleaned_shape=cleaned_shape,
        original_missing_values=original_missing_values,
        missing_percentages=missing_percentages,
        duplicate_rows_found=duplicate_rows_found,
        duplicate_dates_found=duplicate_dates_found,
        invalid_numeric_values=invalid_numeric_values,
        actions=actions,
    )
    return cleaned_data, report


def calculate_descriptive_statistics(gold_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate count, mean, median, spread, range, and quartiles."""
    numeric_data = gold_data.select_dtypes(include=[np.number])
    descriptive_statistics = numeric_data.describe().T
    descriptive_statistics["median"] = numeric_data.median()
    ordered_columns = [
        "count",
        "mean",
        "median",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]
    return descriptive_statistics[ordered_columns]

