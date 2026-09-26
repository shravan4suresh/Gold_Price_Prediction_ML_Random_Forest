"""Matplotlib visualisations for EDA and model interpretation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_closing_price(gold_data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gold_data["Date"], gold_data["Close"], label="Close", color="#1f77b4")
    ax.set_title("Gold Closing Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_daily_returns(gold_data: pd.DataFrame):
    returns = gold_data["Close"].pct_change()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gold_data["Date"], returns, label="Daily Return", color="#2ca02c")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Daily Gold Returns Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_close_distribution(gold_data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gold_data["Close"], bins=30, color="#1f77b4", edgecolor="white")
    ax.set_title("Distribution of Gold Closing Prices")
    ax.set_xlabel("Closing Price")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def plot_return_distribution(gold_data: pd.DataFrame):
    returns = gold_data["Close"].pct_change().dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(returns, bins=30, color="#2ca02c", edgecolor="white")
    ax.set_title("Distribution of Daily Gold Returns")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def plot_moving_averages(gold_data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gold_data["Date"], gold_data["Close"], label="Close", color="#1f77b4")
    ax.plot(
        gold_data["Date"],
        gold_data["Close"].rolling(5).mean(),
        label="5-Day Moving Average",
        color="#ff7f0e",
    )
    ax.plot(
        gold_data["Date"],
        gold_data["Close"].rolling(10).mean(),
        label="10-Day Moving Average",
        color="#9467bd",
    )
    ax.set_title("Gold Price Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_correlation_matrix(gold_data: pd.DataFrame):
    numeric_data = gold_data.select_dtypes(include="number")
    correlation_matrix = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix")
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(correlation_matrix.columns)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(results_data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        results_data["Date"],
        results_data["Actual_Price"],
        label="Actual Price",
        color="#1f77b4",
    )
    ax.plot(
        results_data["Date"],
        results_data["Predicted_Price"],
        label="Predicted Price",
        color="#d62728",
    )
    ax.set_title("Actual vs Predicted Gold Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Gold Closing Price")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_prediction_error(results_data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        results_data["Date"],
        results_data["Prediction_Error"],
        label="Prediction Error",
        color="#d62728",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Prediction Error Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Actual - Predicted")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_feature_importance(feature_importance: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        feature_importance["Feature"],
        feature_importance["Importance"],
        color="#9467bd",
    )
    ax.invert_yaxis()
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig

