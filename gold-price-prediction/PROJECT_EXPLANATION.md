# Project Explanation for Professor Demonstration

## One-Sentence Summary

This project predicts the next trading day's gold closing price using historical
gold prices, lagged features, and a Random Forest Regressor while preventing
time-series data leakage.

## What the Project Does

The project loads a gold-price CSV, cleans the data, performs exploratory data
analysis, creates supervised learning features, trains a Random Forest model,
compares it with a naive baseline, evaluates performance, and displays the
results in a Streamlit app.

## Dataset Cleaning

The cleaning pipeline:

1. Converts `Date` to datetime.
2. Sorts rows from oldest to newest.
3. Checks missing values.
4. Checks exact duplicate rows.
5. Checks duplicate dates.
6. Converts price and volume columns to numeric values.
7. Reports every cleaning action.

Rows with invalid dates or missing `Close` values are removed because they
cannot be used safely for next-day price prediction.

## Target Variable

The target is the next trading day's close:

```python
Target_Next_Close = Close.shift(-1)
```

That means each row uses today's available information to predict tomorrow's
closing price.

## Features

The important engineered features are:

- Closing-price lags: `Close_Lag_1`, `Close_Lag_2`, `Close_Lag_3`,
  `Close_Lag_5`, `Close_Lag_10`
- Previous return: `Return_Lag_1`
- Moving averages: `MA_5`, `MA_10`
- Rolling volatility: `Rolling_Std_5`

These features are useful because they summarize recent price direction,
momentum, and volatility.

## Data Leakage Prevention

This is the most important rule in the project. The model must not use future
information.

The project prevents leakage by:

- Sorting data chronologically.
- Creating features from current or past observations only.
- Using tomorrow's close only as the target.
- Splitting training and testing chronologically.
- Never using random shuffling.

The guiding question is:

```text
Would this information have been available at the time of prediction?
```

## Train/Test Split

The first 80% of observations are used for training. The most recent 20% are
used for testing. This simulates a real forecasting situation where older data
is used to predict newer data.

## Baseline Model

The baseline prediction is:

```text
Tomorrow's predicted close = today's close
```

This is a strong simple benchmark for price-level forecasting. The Random
Forest should only be considered useful if it improves MAE and RMSE compared
with this baseline.

## Random Forest Model

Random Forest builds many decision trees and averages their predictions. It is
used because it can model non-linear relationships while remaining explainable
through feature importance.

The main settings are:

- `n_estimators=300`
- `max_depth=8`
- `min_samples_split=4`
- `min_samples_leaf=2`
- `random_state=42`

## Evaluation

The project reports:

- MAE: average absolute error
- RMSE: error measure that penalizes large mistakes
- R²: variance explained

The app explicitly compares Random Forest with the naive baseline. A high R² is
not enough by itself because time-series prices often look predictable due to
persistence.

## Streamlit App Sections

The app contains:

- Overview
- Dataset
- EDA
- Model
- Performance
- Prediction
- Charts

The prediction section displays the latest known gold price, predicted next
trading day price, expected change, and expected percentage change.

## Final Note

The prediction is a model estimate for academic demonstration. It is not a
guaranteed future market price and should not be treated as financial advice.

