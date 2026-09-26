"""Next-trading-day prediction helper."""

from __future__ import annotations

import pandas as pd

from .feature_engineering import build_latest_feature_row


def predict_next_trading_day(
    random_forest_model,
    gold_data: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, float]:
    """Predict the next trading day's closing price from the latest history."""
    latest_feature_row, latest_gold_price = build_latest_feature_row(
        gold_data, feature_columns
    )
    predicted_next_price = float(random_forest_model.predict(latest_feature_row)[0])
    predicted_change = predicted_next_price - latest_gold_price
    predicted_percentage_change = (predicted_change / latest_gold_price) * 100

    return {
        "Latest Gold Price": latest_gold_price,
        "Predicted Next Trading Day Gold Price": predicted_next_price,
        "Predicted Change": predicted_change,
        "Predicted Percentage Change": predicted_percentage_change,
    }

