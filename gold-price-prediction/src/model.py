"""Model training and evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def chronological_train_test_split(
    modeling_data: pd.DataFrame,
    feature_columns: list[str],
    train_fraction: float = 0.8,
) -> dict[str, object]:
    """Split data chronologically so the model is tested on newer observations."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    split_index = int(len(modeling_data) * train_fraction)
    if split_index == 0 or split_index == len(modeling_data):
        raise ValueError("Not enough observations for a chronological train/test split.")

    training_data = modeling_data.iloc[:split_index].copy()
    testing_data = modeling_data.iloc[split_index:].copy()

    return {
        "training_data": training_data,
        "testing_data": testing_data,
        "training_features": training_data[feature_columns],
        "testing_features": testing_data[feature_columns],
        "training_target": training_data["Target_Next_Close"],
        "testing_target": testing_data["Target_Next_Close"],
    }


def create_random_forest_model(
    n_estimators: int = 300,
    max_depth: int | None = 8,
    min_samples_split: int = 4,
    min_samples_leaf: int = 2,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Create an understandable Random Forest Regressor."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )


def train_random_forest_model(
    training_features: pd.DataFrame,
    training_target: pd.Series,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Train Random Forest only on the chronological training period."""
    random_forest_model = create_random_forest_model(random_state=random_state)
    random_forest_model.fit(training_features, training_target)
    return random_forest_model


def calculate_naive_baseline(testing_data: pd.DataFrame) -> np.ndarray:
    """Baseline: predicted tomorrow price equals today's closing price."""
    return testing_data["Close"].to_numpy()


def evaluate_predictions(
    actual_gold_price: pd.Series | np.ndarray,
    predicted_gold_price: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calculate MAE, RMSE, and R-squared."""
    mae = mean_absolute_error(actual_gold_price, predicted_gold_price)
    mse = mean_squared_error(actual_gold_price, predicted_gold_price)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_gold_price, predicted_gold_price)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def build_results_dataframe(
    testing_data: pd.DataFrame,
    predicted_gold_price: np.ndarray,
) -> pd.DataFrame:
    """Create row-level prediction results for the test period."""
    results_data = pd.DataFrame(
        {
            "Date": testing_data["Date"].values,
            "Actual_Price": testing_data["Target_Next_Close"].values,
            "Predicted_Price": predicted_gold_price,
        }
    )
    results_data["Prediction_Error"] = (
        results_data["Actual_Price"] - results_data["Predicted_Price"]
    )
    results_data["Absolute_Error"] = results_data["Prediction_Error"].abs()
    return results_data


def compare_model_with_baseline(
    baseline_metrics: dict[str, float],
    random_forest_metrics: dict[str, float],
) -> str:
    """State whether Random Forest beats the naive baseline on MAE and RMSE."""
    beats_mae = random_forest_metrics["MAE"] < baseline_metrics["MAE"]
    beats_rmse = random_forest_metrics["RMSE"] < baseline_metrics["RMSE"]

    if beats_mae and beats_rmse:
        return "Random Forest beats the naive baseline on both MAE and RMSE."
    if beats_mae or beats_rmse:
        return "Random Forest beats the naive baseline on one metric, but not both."
    return "Random Forest does not beat the naive baseline on MAE or RMSE."


def get_feature_importance(
    random_forest_model: RandomForestRegressor, feature_columns: list[str]
) -> pd.DataFrame:
    """Return feature importances sorted from highest to lowest."""
    return (
        pd.DataFrame(
            {
                "Feature": feature_columns,
                "Importance": random_forest_model.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
