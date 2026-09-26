"""Streamlit app for Gold Price Prediction Using Random Forest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_processing import (
    calculate_descriptive_statistics,
    clean_gold_data,
    inspect_dataset,
    load_gold_data,
)
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
from src.prediction import predict_next_trading_day
from src.visualization import (
    plot_actual_vs_predicted,
    plot_close_distribution,
    plot_closing_price,
    plot_correlation_matrix,
    plot_daily_returns,
    plot_feature_importance,
    plot_moving_averages,
    plot_prediction_error,
    plot_return_distribution,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "gold_prices.csv"


@st.cache_data
def load_default_data() -> pd.DataFrame:
    return load_gold_data(DEFAULT_DATA_PATH)


@st.cache_data
def run_pipeline(raw_gold_data: pd.DataFrame) -> dict[str, object]:
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
    next_day_prediction = predict_next_trading_day(
        random_forest_model, clean_data, feature_columns
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
        "results_data": results_data,
        "feature_importance": feature_importance,
        "next_day_prediction": next_day_prediction,
        "comparison": compare_model_with_baseline(
            baseline_metrics, random_forest_metrics
        ),
    }


def show_metric_row(metrics: dict[str, float], prefix: str) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{prefix} MAE", f"{metrics['MAE']:.2f}")
    col2.metric(f"{prefix} RMSE", f"{metrics['RMSE']:.2f}")
    col3.metric(f"{prefix} R²", f"{metrics['R2']:.3f}")


def format_date_for_metric(value: object) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def main() -> None:
    st.set_page_config(
        page_title="Gold Price Prediction Using Random Forest",
        layout="wide",
    )
    st.title("Gold Price Prediction Using Random Forest")

    uploaded_file = st.sidebar.file_uploader("Upload gold price CSV", type=["csv"])
    raw_gold_data = load_gold_data(uploaded_file) if uploaded_file else load_default_data()

    try:
        outputs = run_pipeline(raw_gold_data)
    except Exception as error:
        st.error(f"Unable to run pipeline: {error}")
        st.stop()

    clean_data = outputs["clean_data"]
    cleaning_report = outputs["cleaning_report"]
    split_data = outputs["split_data"]

    tabs = st.tabs(
        [
            "Overview",
            "Dataset",
            "EDA",
            "Model",
            "Performance",
            "Prediction",
            "Charts",
        ]
    )

    with tabs[0]:
        st.subheader("Objective")
        st.write(
            "Use historical gold-price information and leakage-safe lagged "
            "features to predict the next trading day's closing price."
        )
        st.subheader("Dataset and Target")
        st.write(
            "The app expects a CSV with Date and Close columns. Open, High, "
            "Low, and Volume are used when available. The target is Close "
            "shifted by -1 trading day, meaning today's information predicts "
            "tomorrow's close."
        )
        st.subheader("Model")
        st.write(
            "The primary model is RandomForestRegressor. A naive baseline also "
            "predicts tomorrow's price as today's close, so the Random Forest "
            "must beat a simple time-series benchmark."
        )

    with tabs[1]:
        inspection = inspect_dataset(raw_gold_data)
        col1, col2, col3 = st.columns(3)
        col1.metric("Raw Rows", inspection["shape"][0])
        col2.metric("Raw Columns", inspection["shape"][1])
        col3.metric("Clean Rows", cleaning_report.cleaned_shape[0])

        st.subheader("Preview")
        st.write("First observations")
        st.dataframe(inspection["first_observations"], use_container_width=True)
        st.write("Last observations")
        st.dataframe(inspection["last_observations"], use_container_width=True)

        st.subheader("Dataset Details")
        st.write("Column names:", inspection["columns"])
        st.dataframe(inspection["data_types"].rename("Data Type"))

        col1, col2 = st.columns(2)
        col1.metric("Start Date", format_date_for_metric(clean_data["Date"].min()))
        col2.metric("End Date", format_date_for_metric(clean_data["Date"].max()))

        st.subheader("Missing Values and Duplicates")
        missing_summary = pd.DataFrame(
            {
                "Missing Values": cleaning_report.original_missing_values,
                "Missing Percentage": cleaning_report.missing_percentages,
            }
        )
        st.dataframe(missing_summary, use_container_width=True)
        st.write(f"Exact duplicate rows found: {cleaning_report.duplicate_rows_found}")
        st.write(f"Duplicate dates found: {cleaning_report.duplicate_dates_found}")
        st.write("Cleaning actions performed:")
        for action in cleaning_report.actions:
            st.write(f"- {action}")

    with tabs[2]:
        st.subheader("Descriptive Statistics")
        st.dataframe(
            calculate_descriptive_statistics(clean_data), use_container_width=True
        )
        st.subheader("Exploratory Charts")
        chart_col1, chart_col2 = st.columns(2)
        chart_col1.pyplot(plot_closing_price(clean_data))
        chart_col2.pyplot(plot_daily_returns(clean_data))
        chart_col1.pyplot(plot_close_distribution(clean_data))
        chart_col2.pyplot(plot_return_distribution(clean_data))
        st.pyplot(plot_moving_averages(clean_data))
        st.pyplot(plot_correlation_matrix(clean_data))

    with tabs[3]:
        st.subheader("Features Used")
        st.write(outputs["feature_columns"])
        st.write(f"Number of features: {len(outputs['feature_columns'])}")
        st.write(
            "Lagged and rolling features are based only on values available at "
            "or before the prediction date."
        )
        st.subheader("Feature-Generated NaN Values")
        st.dataframe(
            pd.Series(outputs["generated_nan_counts"], name="NaN Count"),
            use_container_width=True,
        )
        st.subheader("Chronological Split")
        col1, col2 = st.columns(2)
        col1.write(
            f"Training period: {split_data['training_data']['Date'].min().date()} "
            f"to {split_data['training_data']['Date'].max().date()}"
        )
        col1.write(f"Training observations: {len(split_data['training_data'])}")
        col2.write(
            f"Testing period: {split_data['testing_data']['Date'].min().date()} "
            f"to {split_data['testing_data']['Date'].max().date()}"
        )
        col2.write(f"Testing observations: {len(split_data['testing_data'])}")
        st.subheader("Random Forest Settings")
        st.write(
            {
                "n_estimators": 300,
                "max_depth": 8,
                "min_samples_split": 4,
                "min_samples_leaf": 2,
                "random_state": 42,
            }
        )

    with tabs[4]:
        st.subheader("Random Forest Performance")
        show_metric_row(outputs["random_forest_metrics"], "Random Forest")
        st.subheader("Naive Baseline Performance")
        show_metric_row(outputs["baseline_metrics"], "Baseline")
        st.info(outputs["comparison"])
        st.write(
            "A high R² alone is not enough. The important comparison is whether "
            "Random Forest improves out-of-sample MAE and RMSE over the naive "
            "baseline."
        )
        st.subheader("Prediction Results")
        st.dataframe(outputs["results_data"], use_container_width=True)

    with tabs[5]:
        prediction = outputs["next_day_prediction"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Gold Price", f"{prediction['Latest Gold Price']:.2f}")
        col2.metric(
            "Predicted Next Trading Day Price",
            f"{prediction['Predicted Next Trading Day Gold Price']:.2f}",
        )
        col3.metric("Expected Change", f"{prediction['Predicted Change']:.2f}")
        col4.metric(
            "Expected % Change",
            f"{prediction['Predicted Percentage Change']:.2f}%",
        )
        st.warning(
            "This is a model estimate for academic demonstration, not a "
            "guaranteed future market price or financial advice."
        )

    with tabs[6]:
        st.pyplot(plot_actual_vs_predicted(outputs["results_data"]))
        st.pyplot(plot_prediction_error(outputs["results_data"]))
        st.pyplot(plot_feature_importance(outputs["feature_importance"]))


if __name__ == "__main__":
    main()
