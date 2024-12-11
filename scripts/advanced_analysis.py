# advanced_analysis.py

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def advanced_price_prediction(
    data: pd.DataFrame, 
    correlations: dict, 
    max_lag: int, 
    time_interval: str, 
    timestamp: str, 
    base_csv_filename: str, 
    future_datetime: datetime, 
    lag_periods: int
) -> None:
    """
    Perform advanced price prediction based on correlations.

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
    logging.info(f"Performing advanced analysis for lag {lag_periods} {time_interval}(s)...")
    close_prices = data['Close'].dropna().astype(str)
    sig_figs = close_prices.apply(lambda x: len(x.replace('.', '').replace('-', '').lstrip('0'))).max()
    predictions_dir = os.path.join('predictions', 'advanced_analysis')
    csv_dir = os.path.join(predictions_dir, 'csv')
    json_dir = os.path.join(predictions_dir, 'json')
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    future_datetime_str = future_datetime.strftime('%Y%m%d-%H%M%S')
    csv_filename = f"advanced_prediction_for_{future_datetime_str}_{base_csv_filename}.csv"
    json_filename = f"advanced_prediction_for_{future_datetime_str}_{base_csv_filename}.json"
    csv_filepath = os.path.join(csv_dir, csv_filename)
    json_filepath = os.path.join(json_dir, json_filename)
    data = data.copy()
    
    # Create lagged features
    lag = lag_periods
    logging.info(f"Creating lagged features for lag: {lag}")
    for i in range(1, lag + 1):
        for col in ['Close', 'Volume', 'Open', 'High', 'Low']:
            data[f'{col}_lag_{i}'] = data[col].shift(i)
    
    data['Target'] = data['Close'].shift(-lag)
    
    # Select top N correlated indicators
    N = 20
    lag_index = lag - 1
    lag_correlations = {col: correlations[col][lag_index] if lag_index < len(correlations[col]) else np.nan for col in correlations}
    lag_correlations = {col: c for col, c in lag_correlations.items() if not np.isnan(c)}
    sorted_correlations = sorted(lag_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    top_indicators = [col for col, _ in sorted_correlations[:N] if col in data.columns]
    logging.info(f"Top {N} indicators selected based on correlation: {top_indicators}")
    
    feature_columns = top_indicators + [f'{col}_lag_{lag}' for col in ['Close', 'Volume', 'Open', 'High', 'Low']] + ['Target']
    
    # Handle missing values
    imputer = KNNImputer(n_neighbors=5)
    data_imputed = pd.DataFrame(imputer.fit_transform(data[feature_columns]), columns=feature_columns)
    
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data_imputed[top_indicators + [f'{col}_lag_{lag}' for col in ['Close', 'Volume', 'Open', 'High', 'Low']]]), 
                                columns=top_indicators + [f'{col}_lag_{lag}' for col in ['Close', 'Volume', 'Open', 'High', 'Low']],
                                index=data_imputed.index)
    
    X = data_scaled
    y = data_imputed['Target']
    valid_indices = ~y.isna()
    X, y = X[valid_indices], y[valid_indices]
    
    if X.empty or y.empty:
        logging.error("Not enough data to train the model.")
        return
    
    tscv = TimeSeriesSplit(n_splits=5)
    models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42, objective='reg:squarederror')
    }
    param_grid = {
        'RandomForest': {'n_estimators': [100, 200],'max_depth': [5, 10, None],'min_samples_split': [2, 5]},
        'XGBoost': {'n_estimators': [100, 200],'learning_rate': [0.01, 0.1],'max_depth': [3, 6]}
    }
    best_models = {}
    predictions_data = []
    for name, model in models.items():
        logging.info(f"Training and tuning {name} model...")
        grid_search = GridSearchCV(model, param_grid[name], cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X, y)
        best_models[name] = grid_search.best_estimator_
        logging.info(f"Best {name} model parameters: {grid_search.best_params_}")
    
    for name, model in best_models.items():
        pred = model.predict(X.iloc[[-1]])[0]
        predicted_price_formatted = format_significant_figures(pred, sig_figs)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred)
        predictions_data.append({
            'Model': name,
            'Lag': lag,
            'Future_DateTime': future_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'Predicted_Price': predicted_price_formatted,
            'MSE': mse,
            'MAE': mae,
            'MAPE (%)': mape * 100
        })
    
    pd.DataFrame(predictions_data).to_csv(csv_filepath, index=False)
    pd.DataFrame(predictions_data).to_json(json_filepath, orient='records', lines=True)
    logging.info(f"Advanced predictions saved to {csv_filepath} and {json_filepath}")
    
def format_significant_figures(value, sig_figs):
    if value == 0:
        return '0'
    return f"{value:.{sig_figs - int(np.floor(np.log10(abs(value)))) - 1}f}"

