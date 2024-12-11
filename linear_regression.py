# Filename: linear_regression.py

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import resample
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def perform_linear_regression(data, correlations, max_lag, time_interval, timestamp, base_csv_filename, future_datetime, lag_periods):
    """
    Performs linear regression to predict future prices using the top correlated indicators.

    Args:
        data (pd.DataFrame): The dataset containing indicators and 'Close' price.
        correlations (dict): Dictionary containing correlations for each indicator.
        max_lag (int): Maximum lag to consider.
        time_interval (str): Time interval between data points.
        timestamp (str): Current timestamp for file naming.
        base_csv_filename (str): Base filename of the CSV file.
        future_datetime (datetime): The future date/time for which to predict the price.
        lag_periods (int): The lag period corresponding to future_datetime.
    """
    max_user_lag = lag_periods

    print(f"Calculated lag period: {max_user_lag} {time_interval}(s)")

    # Ask if the user wants to calculate predictions for every lag period up to the specified date/time
    calc_all_lags = input(f"Do you want to calculate predictions for every lag period up to {max_user_lag} {time_interval}(s)? (y/n): ").strip().lower() == 'y'

    if calc_all_lags:
        lag_range = range(1, max_user_lag + 1)
    else:
        lag_range = [max_user_lag]

    predictions = {}
    confidence_intervals = {}
    coefficients_data = []
    predictions_data = []

    # Determine the number of significant figures in the original 'Close' price
    close_prices = data['Close'].dropna().astype(str)
    sig_figs = close_prices.apply(lambda x: len(x.replace('.', '').replace('-', '').lstrip('0'))).max()

    # Create directories for saving predictions and plots
    predictions_dir = 'predictions'
    csv_dir = os.path.join(predictions_dir, 'csv')
    json_dir = os.path.join(predictions_dir, 'json')
    plots_dir = os.path.join(predictions_dir, 'plots')
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Prepare filenames
    future_datetime_str = future_datetime.strftime('%Y%m%d-%H%M%S')
    csv_filename = f"prediction_for_{future_datetime_str}_{base_csv_filename}.csv"
    json_filename = f"prediction_for_{future_datetime_str}_{base_csv_filename}.json"
    plot_filename = f"prediction_plot_for_{future_datetime_str}_{base_csv_filename}.png"
    csv_filepath = os.path.join(csv_dir, csv_filename)
    json_filepath = os.path.join(json_dir, json_filename)
    plot_filepath = os.path.join(plots_dir, plot_filename)

    # Collect data for plotting
    plot_dates = []
    plot_predictions = []
    plot_lower_bounds = []
    plot_upper_bounds = []

    for lag in lag_range:
        # Calculate the actual date/time for each lag period
        lag_timedelta = timedelta(seconds=lag * time_interval_seconds(time_interval))
        prediction_datetime = data['Date'].max() + lag_timedelta

        print(f"\nCalculating prediction for lag {lag} {time_interval}(s) (Prediction Date/Time: {prediction_datetime})...")

        # Identify the top N indicators with the highest absolute correlations for the given lag
        lag_index = lag - 1
        lag_correlations = {
            col: correlations[col][lag_index] if lag_index < len(correlations[col]) else np.nan
            for col in correlations
        }

        # Remove NaN correlations
        lag_correlations = {col: corr for col, corr in lag_correlations.items() if not np.isnan(corr)}

        if not lag_correlations:
            print(f"No valid correlations found for lag {lag}. Skipping.")
            continue

        # Select the top N indicators
        N = 5  # You can adjust N as needed

        sorted_correlations = sorted(lag_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        top_indicators = [col for col, corr in sorted_correlations[:N]]

        # Prepare the dataset for linear regression
        data_copy = data.copy()

        # Shift the target variable by the negative lag to represent future prices
        data_copy['Target'] = data_copy['Close'].shift(-lag)

        # Include the top indicators as features
        feature_columns = top_indicators + ['Volume', 'Open', 'High', 'Low']
        data_copy = data_copy[feature_columns + ['Target', 'Date']]

        # Drop rows with NaN values
        data_copy.dropna(inplace=True)

        X = data_copy[feature_columns]
        y = data_copy['Target']

        if X.empty or y.empty:
            print(f"Not enough data to train model for lag {lag}. Skipping.")
            continue

        # Split the data into training and testing sets
        split_index = int(len(X) * 0.8)
        X_train = X[:split_index]
        y_train = y[:split_index]
        X_test = X[split_index:]
        y_test = y[split_index:]

        # Train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make prediction for the specified future date/time
        # Ensure future_features is a DataFrame with feature names
        future_features = X.iloc[[-1]].copy()

        predicted_price = model.predict(future_features)[0]

        # Format the predicted price to match the significant figures
        predicted_price_formatted = format_significant_figures(predicted_price, sig_figs)

        # Parallelized Bootstrapping
        def bootstrap_prediction(seed):
            np.random.seed(seed)
            X_resampled, y_resampled = resample(X_train, y_train)
            model_boot = LinearRegression()
            model_boot.fit(X_resampled, y_resampled)
            # Use future_features with feature names
            boot_pred = model_boot.predict(future_features)[0]
            return boot_pred

        num_bootstraps = 1000  # Adjust as needed
        seeds = np.random.randint(0, 1000000, size=num_bootstraps)

        # Use all available CPU cores
        boot_predictions = Parallel(n_jobs=-1)(
            delayed(bootstrap_prediction)(seed) for seed in seeds
        )

        # Compute the 95% confidence interval
        lower_bound = np.percentile(boot_predictions, 2.5)
        upper_bound = np.percentile(boot_predictions, 97.5)

        # Format confidence intervals
        lower_bound_formatted = format_significant_figures(lower_bound, sig_figs)
        upper_bound_formatted = format_significant_figures(upper_bound, sig_figs)

        predictions[lag] = predicted_price_formatted
        confidence_intervals[lag] = (lower_bound_formatted, upper_bound_formatted)

        print(f"Predicted price for lag {lag} ({prediction_datetime}): {predicted_price_formatted}")
        print(f"95% Confidence Interval: [{lower_bound_formatted}, {upper_bound_formatted}]")

        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        variance = np.var(y_test)

        # Compute the ratio of MSE to variance
        mse_variance_ratio = mse / variance if variance != 0 else np.nan

        # Interpret the MSE
        mse_interpretation = interpret_mse(mse_variance_ratio)

        print(f"Mean Squared Error on test set for lag {lag}: {mse:.4f}")
        print(f"Root Mean Squared Error on test set for lag {lag}: {rmse:.4f}")
        print(f"Mean Absolute Error on test set for lag {lag}: {mae:.4f}")
        print(f"Variance of target variable: {variance:.4f}")
        print(f"MSE to Variance Ratio: {mse_variance_ratio:.4f}")
        print(f"Interpretation of MSE: {mse_interpretation}")

        # Collect coefficients data
        coefficients = pd.DataFrame({
            'Feature': ['Intercept'] + feature_columns,
            'Coefficient': [model.intercept_] + list(model.coef_)
        })
        coefficients['Lag'] = lag
        coefficients['Prediction_DateTime'] = prediction_datetime
        coefficients_data.append(coefficients)

        # Collect predictions data
        predictions_data.append({
            'Lag': lag,
            'Prediction_DateTime': prediction_datetime,
            'Predicted_Price': predicted_price_formatted,
            'Lower_Bound': lower_bound_formatted,
            'Upper_Bound': upper_bound_formatted,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Variance': variance,
            'MSE/Variance Ratio': mse_variance_ratio,
            'MSE Interpretation': mse_interpretation
        })

        # Save the coefficients and predictions progressively
        save_predictions_and_coefficients(
            predictions_data, coefficients_data, csv_filepath, json_filepath
        )

        # Collect data for plotting
        plot_dates.append(prediction_datetime)
        plot_predictions.append(predicted_price)
        plot_lower_bounds.append(lower_bound)
        plot_upper_bounds.append(upper_bound)

    if not predictions:
        print("No predictions were made.")
        return

    # Prepare data for plotting
    # Get the last 50 data points
    recent_data = data.tail(50)
    plot_actual_dates = pd.to_datetime(recent_data['Date'])
    plot_actual_prices = recent_data['Close']

    # Combine actual and predicted data
    all_dates = list(plot_actual_dates) + plot_dates
    all_prices = list(plot_actual_prices) + plot_predictions
    all_lower_bounds = [np.nan] * len(plot_actual_dates) + plot_lower_bounds
    all_upper_bounds = [np.nan] * len(plot_actual_dates) + plot_upper_bounds

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot actual prices
    plt.plot(plot_actual_dates, plot_actual_prices, label='Actual Prices', color='blue')

    # Plot predicted prices
    plt.plot(plot_dates, plot_predictions, label='Predicted Prices', color='green', linestyle='--')

    # Plot confidence intervals
    plt.fill_between(plot_dates, plot_lower_bounds, plot_upper_bounds, color='orange', alpha=0.3, label='95% Confidence Interval')

    # Configure x-axis
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Price Prediction using Linear Regression')
    plt.legend()
    plt.grid(True)

    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plt.savefig(plot_filepath)
    plt.close()

    print(f"Prediction plot saved to {plot_filepath}")

def time_interval_seconds(time_interval):
    """Converts time interval string to seconds."""
    intervals = {
        'second': 1,
        'minute': 60,
        'hour': 3600,
        'day': 86400,
        'week': 604800
    }
    return intervals.get(time_interval.lower(), None)

def format_significant_figures(value, sig_figs):
    """Formats a number to the specified number of significant figures."""
    if value == 0 or np.isnan(value):
        return '0'
    else:
        decimals = sig_figs - int(np.floor(np.log10(abs(value)))) - 1
        if decimals > 0:
            return f"{value:.{decimals}f}"
        else:
            return f"{round(value, -decimals)}"

def interpret_mse(mse_variance_ratio):
    """Interprets the MSE to Variance Ratio."""
    if np.isnan(mse_variance_ratio):
        return "Variance is zero; cannot interpret MSE."
    elif mse_variance_ratio < 0.1:
        return "Tiny MSE relative to variance."
    elif mse_variance_ratio < 0.5:
        return "Small MSE relative to variance."
    elif mse_variance_ratio < 1.0:
        return "Average MSE relative to variance."
    else:
        return "Large MSE relative to variance."

def save_predictions_and_coefficients(predictions_data, coefficients_data, csv_filepath, json_filepath):
    """Saves predictions and coefficients to CSV and JSON files."""
    # Save predictions
    predictions_df = pd.DataFrame(predictions_data)
    predictions_df.sort_values('Lag', inplace=True)
    # Concatenate all coefficients
    coefficients_df = pd.concat(coefficients_data, ignore_index=True)
    # Merge predictions and coefficients
    combined_df = pd.merge(
        predictions_df,
        coefficients_df,
        left_on=['Lag', 'Prediction_DateTime'],
        right_on=['Lag', 'Prediction_DateTime'],
        how='left'
    )
    # Save to CSV
    combined_df.to_csv(csv_filepath, index=False)
    # Save to JSON
    combined_df.to_json(json_filepath, orient='records', lines=True)