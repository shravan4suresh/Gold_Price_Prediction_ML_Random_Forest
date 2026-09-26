# Gold Price Prediction Using Random Forest

This project predicts the next trading day's gold closing price using historical
gold prices, leakage-safe lagged features, and a Random Forest Regressor.

## Business / Problem Statement

Gold prices are important for investors, commodity analysts, central banks, and
businesses exposed to precious-metal markets. The goal is not to guarantee future
prices, but to build an explainable machine learning workflow that estimates the
next trading day's closing price from historical market information.

## Project Objective

Use historical gold-price information and engineered lagged features to predict:

```text
Tomorrow's Close = f(today's and previous historical gold-price information)
```

The model is designed for an academic Applied Machine Learning demonstration.

## Dataset Description

The application accepts a CSV file with:

- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

Only `Date` and `Close` are strictly required. The included
`data/gold_prices.csv` is a small sample dataset for running the project
immediately. Replace it or upload your own historical gold-price CSV in the
Streamlit app.

## Methodology

1. Load historical gold prices.
2. Inspect the first and last rows, shape, columns, and data types.
3. Clean dates, numeric columns, missing values, duplicate rows, and duplicate
   dates.
4. Perform exploratory data analysis.
5. Create a next-day target using `Close.shift(-1)`.
6. Create lagged and rolling features using only current and past information.
7. Split the data chronologically into training and testing periods.
8. Train a Random Forest Regressor.
9. Compare against a naive baseline.
10. Predict the next trading day's gold closing price.

## EDA

The project includes:

- Descriptive statistics
- Missing-value analysis
- Duplicate analysis
- Gold closing price over time
- Daily returns
- Closing-price distribution
- Daily-return distribution
- Moving averages
- Correlation matrix

## Feature Engineering

The target is:

```python
Target_Next_Close = Close.shift(-1)
```

Features include:

- `Close_Lag_1`
- `Close_Lag_2`
- `Close_Lag_3`
- `Close_Lag_5`
- `Close_Lag_10`
- `Return_Lag_1`
- `MA_5`
- `MA_10`
- `Rolling_Std_5`

When available, `Open`, `High`, `Low`, `Close`, and `Volume` are also used as
same-day values known at the end of day `t` to predict day `t + 1`.

## Why Lagged Features Are Required

Machine learning models do not automatically understand time order. Lagged
features turn historical time-series values into supervised learning inputs.
For example, `Close_Lag_1` tells the model what yesterday's close was when
making a prediction from today's row.

## Random Forest Explanation

Random Forest is an ensemble of decision trees. Each tree learns relationships
between historical price features and the next day's close. The final prediction
is the average prediction across many trees.

Main parameters:

- `n_estimators`: number of trees in the forest
- `max_depth`: maximum depth of each tree
- `min_samples_split`: minimum samples needed to split a node
- `min_samples_leaf`: minimum samples required in a leaf node
- `random_state`: makes results reproducible

## Train/Test Methodology

The project uses a chronological 80/20 split:

- First 80% of observations: training
- Most recent 20% of observations: testing

The data is never randomly shuffled because this is time-series data.

## Data Leakage Prevention

The key rule is:

```text
Would this information have been available when the prediction was made?
```

If the answer is no, the information is not used.

Leakage prevention steps:

- No random train/test split
- Features are built using only current and past values
- Tomorrow's close is used only as the target, never as an input feature
- Rolling features use backward-looking windows
- Test observations occur after training observations

## Evaluation Metrics

The project reports:

- MAE: average absolute prediction error
- RMSE: penalizes larger errors more strongly
- R²: explains variance, but is not enough by itself for time-series success

## Baseline Comparison

The naive baseline is:

```text
Predicted tomorrow price = today's closing price
```

Random Forest should be judged by whether it improves out-of-sample MAE and
RMSE compared with this baseline.

## How to Run the Project

Create and activate a virtual environment if desired, then install packages:

```bash
pip install -r requirements.txt
```

Train and save the model:

```bash
python train_model.py
```

## How to Run Streamlit

```bash
streamlit run app.py
```

The app opens a browser UI where you can use the sample data or upload your own
CSV file.

## How to Run with Docker

Build the Docker image:

```bash
docker build -t gold-price-prediction .
```

Run the Streamlit app in a container:

```bash
docker run --rm -p 8501:8501 gold-price-prediction
```

Then open:

```text
http://localhost:8501
```

## How to Publish to Git

Initialize the repository, commit the project, and push it to your Git host:

```bash
git init
git add .
git commit -m "Add Dockerized gold price prediction app"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

Replace `<your-repository-url>` with the GitHub, GitLab, Bitbucket, or internal
Git repository URL you want to share with your team.

## Repository Structure

```text
gold-price-prediction/
├── data/
│   └── gold_prices.csv
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb
│   └── 02_random_forest_model.ipynb
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── prediction.py
│   └── visualization.py
├── models/
│   └── random_forest_gold.pkl
├── app.py
├── train_model.py
├── requirements.txt
├── PROJECT_EXPLANATION.md
└── README.md
```

## Limitations

- Historical prices alone may not capture macroeconomic news, interest rates,
  inflation expectations, currency changes, or geopolitical events.
- Random Forest does not explicitly model long-term time-series dynamics.
- A strong naive baseline can be difficult to beat for daily price levels.
- The sample dataset is provided for demonstration and should be replaced with
  real historical data for serious analysis.

## Future Improvements

- Add external features such as USD index, interest rates, inflation, or ETF
  flows.
- Test additional leakage-safe lag and rolling-window features.
- Add walk-forward validation.
- Add model persistence and prediction logging for repeated demonstrations.
- Compare with other non-neural, non-prohibited models such as linear regression
  or ridge regression.
