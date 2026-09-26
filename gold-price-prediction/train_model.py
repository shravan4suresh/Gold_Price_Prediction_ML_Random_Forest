"""Train and save the Random Forest gold price prediction model."""

from __future__ import annotations

from pathlib import Path

import joblib

from src.data_processing import clean_gold_data, load_gold_data
from src.feature_engineering import prepare_modeling_data
from src.model import (
    build_results_dataframe,
    calculate_naive_baseline,
    chronological_train_test_split,
    compare_model_with_baseline,
    evaluate_predictions,
    get_feature_importance,
    train_random_forest_model,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "gold_prices.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_gold.pkl"


def run_training_pipeline(data_path: Path = DATA_PATH) -> dict[str, object]:
    """Run the complete modeling pipeline and save the trained model."""
    raw_gold_data = load_gold_data(data_path)
    clean_data, cleaning_report = clean_gold_data(raw_gold_data)
    modeling_data, feature_columns, generated_nan_counts = prepare_modeling_data(
        clean_data
    )

    split_data = chronological_train_test_split(modeling_data, feature_columns)
    random_forest_model = train_random_forest_model(
        split_data["training_features"], split_data["training_target"]
    )

    predicted_gold_price = random_forest_model.predict(split_data["testing_features"])
    naive_predictions = calculate_naive_baseline(split_data["testing_data"])

    random_forest_metrics = evaluate_predictions(
        split_data["testing_target"], predicted_gold_price
    )
    baseline_metrics = evaluate_predictions(split_data["testing_target"], naive_predictions)
    results_data = build_results_dataframe(
        split_data["testing_data"], predicted_gold_price
    )
    feature_importance = get_feature_importance(random_forest_model, feature_columns)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": random_forest_model,
            "feature_columns": feature_columns,
            "metrics": random_forest_metrics,
            "baseline_metrics": baseline_metrics,
        },
        MODEL_PATH,
    )

    return {
        "clean_data": clean_data,
        "cleaning_report": cleaning_report,
        "modeling_data": modeling_data,
        "feature_columns": feature_columns,
        "generated_nan_counts": generated_nan_counts,
        "split_data": split_data,
        "model": random_forest_model,
        "random_forest_metrics": random_forest_metrics,
        "baseline_metrics": baseline_metrics,
        "comparison": compare_model_with_baseline(
            baseline_metrics, random_forest_metrics
        ),
        "results_data": results_data,
        "feature_importance": feature_importance,
        "model_path": MODEL_PATH,
    }


if __name__ == "__main__":
    outputs = run_training_pipeline()
    split = outputs["split_data"]
    print("Training complete.")
    print(f"Saved model to: {outputs['model_path']}")
    print(f"Features ({len(outputs['feature_columns'])}): {outputs['feature_columns']}")
    print(
        "Training period:",
        split["training_data"]["Date"].min().date(),
        "to",
        split["training_data"]["Date"].max().date(),
    )
    print(
        "Testing period:",
        split["testing_data"]["Date"].min().date(),
        "to",
        split["testing_data"]["Date"].max().date(),
    )
    print("Random Forest metrics:", outputs["random_forest_metrics"])
    print("Naive baseline metrics:", outputs["baseline_metrics"])
    print(outputs["comparison"])

