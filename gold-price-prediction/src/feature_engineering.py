"""Leakage-safe feature engineering for next-day gold price prediction."""

from __future__ import annotations

import pandas as pd


BASE_FEATURE_CANDIDATES = ["Open", "High", "Low", "Close", "Volume"]
ENGINEERED_FEATURES = [
    "Close_Lag_1",
    "Close_Lag_2",
    "Close_Lag_3",
    "Close_Lag_5",
    "Close_Lag_10",
    "Return_Lag_1",
    "MA_5",
    "MA_10",
    "Rolling_Std_5",
]


def add_time_series_features(gold_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create target and historical features without using future information.

    Target is tomorrow's close: Close shifted by -1. Every feature on row t is
    based only on values available at or before row t.
    """
    featured_data = gold_data.copy().sort_values("Date").reset_index(drop=True)

    featured_data["Target_Next_Close"] = featured_data["Close"].shift(-1)
    featured_data["Daily_Return"] = featured_data["Close"].pct_change()

    for lag in [1, 2, 3, 5, 10]:
        featured_data[f"Close_Lag_{lag}"] = featured_data["Close"].shift(lag)

    featured_data["Return_Lag_1"] = featured_data["Daily_Return"].shift(1)
    featured_data["MA_5"] = featured_data["Close"].rolling(window=5).mean()
    featured_data["MA_10"] = featured_data["Close"].rolling(window=10).mean()
    featured_data["Rolling_Std_5"] = featured_data["Close"].rolling(window=5).std()

    return featured_data


def get_feature_columns(featured_data: pd.DataFrame) -> list[str]:
    """
    Select model features that are available in the current dataset.

    Same-day Open/High/Low/Close/Volume are included only when present. In this
    project's prediction setup, they represent information known at the end of
    the current trading day and are used to predict the next trading day's close.
    """
    base_features = [
        column for column in BASE_FEATURE_CANDIDATES if column in featured_data.columns
    ]
    engineered_features = [
        column for column in ENGINEERED_FEATURES if column in featured_data.columns
    ]
    return base_features + engineered_features


def prepare_modeling_data(
    gold_data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """
    Add features and remove rows with feature-generated NaN values.

    The returned NaN summary helps distinguish original missing data from NaN
    values naturally created by lagging, rolling windows, and target shifting.
    """
    featured_data = add_time_series_features(gold_data)
    feature_columns = get_feature_columns(featured_data)
    modeling_columns = ["Date", "Target_Next_Close"] + feature_columns

    generated_nan_counts = (
        featured_data[modeling_columns].isna().sum().astype(int).to_dict()
    )
    modeling_data = featured_data.dropna(subset=modeling_columns).reset_index(drop=True)

    return modeling_data, feature_columns, generated_nan_counts


def build_latest_feature_row(
    gold_data: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, float]:
    """
    Construct the most recent feature row for next-trading-day prediction.

    This uses only the historical observations available up to the latest row.
    """
    featured_data = add_time_series_features(gold_data)
    latest_feature_row = featured_data.dropna(subset=feature_columns).tail(1)

    if latest_feature_row.empty:
        raise ValueError(
            "Not enough historical rows to build lagged and rolling features. "
            "At least 11 chronological observations are recommended."
        )

    latest_gold_price = float(latest_feature_row["Close"].iloc[0])
    return latest_feature_row[feature_columns], latest_gold_price

