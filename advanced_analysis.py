import os
import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def advanced_price_prediction(data, correlations, max_lag, time_interval, timestamp, base_csv_filename, future_datetime, lag_periods):
    """
    Performs advanced analysis and price prediction using machine learning techniques.

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

    # We already have future_datetime and lag_periods calculated in main()
    # So no need to ask the user again

    lag = lag_periods

    print(f"\nPerforming advanced analysis for lag {lag} {time_interval}(s)...")

    # Determine the number of significant figures in the original 'Close' price
    close_prices = data['Close'].dropna().astype(str)
    sig_figs = close_prices.apply(lambda x: len(x.replace('.', '').replace('-', '').lstrip('0'))).max()

    # Create directories for saving predictions
    predictions_dir = os.path.join('predictions', 'advanced_analysis')
    csv_dir = os.path.join(predictions_dir, 'csv')
    json_dir = os.path.join(predictions_dir, 'json')
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    # Prepare filenames
    future_datetime_str = future_datetime.strftime('%Y%m%d-%H%M%S')
    csv_filename = f"advanced_prediction_for_{future_datetime_str}_{base_csv_filename}.csv"
    json_filename = f"advanced_prediction_for_{future_datetime_str}_{base_csv_filename}.json"
    csv_filepath = os.path.join(csv_dir, csv_filename)
    json_filepath = os.path.join(json_dir, json_filename)

    # Prepare the dataset
    data = data.copy()

    # Create lagged features
    lagged_features = []
    for i in range(1, lag + 1):
        for col in ['Close', 'Volume']:
            data[f'{col}_lag_{i}'] = data[col].shift(i)
            lagged_features.append(f'{col}_lag_{i}')

    # Create differenced features
    data['Close_diff'] = data['Close'].diff()
    data['Volume_diff'] = data['Volume'].diff()

    # Generate interaction terms
    data['Close_Volume'] = data['Close'] * data['Volume']

    # Shift the target variable by the negative lag to represent future prices
    data['Target'] = data['Close'].shift(-lag)

    # Include the top N indicators as features
    N = 20  # Number of top indicators to include
    lag_index = lag - 1
    lag_correlations = {
        col: correlations[col][lag_index] if lag_index < len(correlations[col]) else np.nan
        for col in correlations
    }
    lag_correlations = {col: corr for col, corr in lag_correlations.items() if not np.isnan(corr)}
    sorted_correlations = sorted(lag_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    top_indicators = [col for col, corr in sorted_correlations[:N]]

    # Ensure all top indicators are in the data
    top_indicators = [col for col in top_indicators if col in data.columns]

    # Define feature columns
    feature_columns = top_indicators + lagged_features + ['Close_diff', 'Volume_diff', 'Close_Volume']

    # Handle missing values using KNN Imputer
    imputer = KNNImputer(n_neighbors=5)
    data_imputed = pd.DataFrame(imputer.fit_transform(data[feature_columns + ['Target']]), columns=feature_columns + ['Target'])

    # Scale features using StandardScaler
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data_imputed[feature_columns]), columns=feature_columns)

    # Define X and y
    X = data_scaled
    y = data_imputed['Target']

    # Drop rows with NaN in target variable
    valid_indices = ~y.isna()
    X = X[valid_indices]
    y = y[valid_indices]

    if X.empty or y.empty:
        print("Not enough data to train the model.")
        return

    # Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Model selection and hyperparameter tuning
    models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42, objective='reg:squarederror')
    }

    param_grid = {
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 6],
        }
    }

    best_models = {}
    predictions_data = []

    for name, model in models.items():
        print(f"\nTraining and tuning {name} model...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid[name],
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        best_models[name] = grid_search.best_estimator_
        print(f"Best {name} model: {grid_search.best_params_}")

    # Ensemble predictions
    predictions = []
    for name, model in best_models.items():
        pred = model.predict(X.iloc[[-1]])
        predicted_price = pred[0]
        # Format the predicted price to match the significant figures
        predicted_price_formatted = format_significant_figures(predicted_price, sig_figs)

        # Evaluate the model
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred)

        # Collect predictions data
        predictions_data.append({
            'Model': name,
            'Lag': lag,
            'Future_DateTime': future_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'Predicted_Price': predicted_price_formatted,
            'MSE': mse,
            'MAE': mae,
            'MAPE (%)': mape * 100
        })

    # Save the predictions
    predictions_df = pd.DataFrame(predictions_data)
    predictions_df.to_csv(csv_filepath, index=False)
    predictions_df.to_json(json_filepath, orient='records', lines=True)
    print(f"\nAdvanced predictions saved to {csv_filepath} and {json_filepath}")

def format_significant_figures(value, sig_figs):
    """Formats a number to the specified number of significant figures."""
    if value == 0:
        return '0'
    else:
        return f"{value:.{sig_figs - int(np.floor(np.log10(abs(value)))) - 1}f}"