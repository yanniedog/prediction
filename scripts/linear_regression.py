# linear_regression.py

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta  # Added import for datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import logging

def perform_linear_regression(
    data: pd.DataFrame,
    correlations: dict,
    max_lag: int,
    time_interval: str,
    timestamp: str,
    base_csv_filename: str,
    future_datetime: datetime,  # Ensure datetime is imported
    lag_periods: int
) -> None:
    """
    Perform linear regression predictions based on top correlated indicators.

    Parameters:
    - data: Original DataFrame containing the data.
    - correlations: Dictionary with indicator names as keys and list of correlations as values.
    - max_lag: Maximum lag considered.
    - time_interval: Time interval string (e.g., '1w').
    - timestamp: Current timestamp for filename uniqueness.
    - base_csv_filename: Base name derived from symbol and timeframe.
    - future_datetime: Datetime object for future prediction.
    - lag_periods: Number of periods ahead to predict.
    """
    logging.info(f"Performing linear regression for lag periods: {lag_periods} {time_interval}(s)...")
    
    # Select top indicators based on correlation
    top_indicators = sorted(correlations.keys(), key=lambda x: abs(correlations[x][lag_periods-1]), reverse=True)[:5]
    logging.info(f"Top indicators selected for regression: {top_indicators}")
    
    # Prepare features and target
    feature_cols = top_indicators + ['Volume', 'Open', 'High', 'Low']
    if not all(col in data.columns for col in feature_cols):
        missing = [col for col in feature_cols if col not in data.columns]
        logging.error(f"Missing columns for regression: {missing}")
        return
    
    data_copy = data.copy()
    data_copy['Target'] = data_copy['Close'].shift(-lag_periods)
    data_copy.dropna(inplace=True)
    
    X = data_copy[feature_cols]
    y = data_copy['Target']
    
    # Split into training and testing
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    logging.info("Linear regression model trained.")
    
    # Predict
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    logging.info(f"Regression MSE: {mse}, MAE: {mae}")
    
    # Save predictions
    predictions_data = {
        'Timestamp': [timestamp]*len(y_pred),
        'Symbol': [base_csv_filename.split('_')[0]]*len(y_pred),
        'Timeframe': [base_csv_filename.split('_')[1]]*len(y_pred),
        'Lag': [lag_periods]*len(y_pred),
        'Predicted_Close': y_pred,
        'Actual_Close': y_test.values,
        'MSE': [mse]*len(y_pred),
        'MAE': [mae]*len(y_pred)
    }
    predictions_df = pd.DataFrame(predictions_data)
    predictions_filepath = os.path.join('predictions', 'csv', f"{timestamp}_{base_csv_filename}_linear_regression_predictions.csv")
    os.makedirs(os.path.dirname(predictions_filepath), exist_ok=True)
    predictions_df.to_csv(predictions_filepath, index=False)
    logging.info(f"Saved linear regression predictions at {predictions_filepath}.")
    
    # Plot predictions
    plt.figure(figsize=(12,6))
    plt.plot(data_copy['Date'], y_test, label='Actual Close Price', color='blue')
    plt.plot(data_copy['Date'], y_pred, label='Predicted Close Price', color='red', linestyle='--')
    plt.title('Linear Regression Predictions vs Actual Close Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filepath = os.path.join('predictions', 'plots', f"{timestamp}_{base_csv_filename}_linear_regression_plot.png")
    os.makedirs(os.path.dirname(plot_filepath), exist_ok=True)
    plt.savefig(plot_filepath)
    plt.close()
    logging.info(f"Saved linear regression plot at {plot_filepath}.")
