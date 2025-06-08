# predictor.py
"""Prediction module for analyzing and predicting price movements."""

import logging
import pandas as pd
import numpy as np
from typing import (
    Optional, Tuple, Dict, Any, List, Sequence, Union, cast,
    TypeVar, Protocol, Callable, TYPE_CHECKING, Mapping,
    TypeAlias
)
from numpy.typing import ArrayLike
from pathlib import Path
from datetime import datetime, timedelta, timezone
import re
import sqlite3 # For specific error handling

# Use non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Type aliases for better type hints
DateTimeArray: TypeAlias = ArrayLike
FloatArray: TypeAlias = ArrayLike

# Import statsmodels conditionally
try:
    import statsmodels.api as sm
    from statsmodels.api import add_constant, OLS
    from statsmodels.regression.linear_model import RegressionResultsWrapper
    STATSMODELS_AVAILABLE = True
except ImportError:
    sm = None
    add_constant = OLS = RegressionResultsWrapper = None  # type: ignore
    STATSMODELS_AVAILABLE = False
    logging.getLogger(__name__).error(
        "Predictor requires 'statsmodels'. Please install it (`pip install statsmodels`). Prediction functionality will be disabled."
    )

# Import project modules
import config # Import config for constants
import utils
import data_manager
import indicator_factory
import leaderboard_manager
import sqlite_manager

logger = logging.getLogger(__name__)

# Get constant from config
MIN_REGRESSION_POINTS = config.DEFAULTS.get("min_regression_points", 30) # Fallback default

# Create a module-level factory instance
_indicator_factory = indicator_factory.IndicatorFactory()

# Type aliases for better readability
IndicatorCache: TypeAlias = Dict[str, pd.DataFrame]
PriceSequence: TypeAlias = Sequence[float]
DateSequence: TypeAlias = Sequence[datetime]

# Type definitions for statsmodels
class RegressionModel(Protocol):
    """Protocol for regression model results."""
    nobs: int
    params: pd.Series
    rsquared: float
    summary: Any
    model: Any
    
    def get_prediction(self, exog: pd.DataFrame) -> Any: ...

RegressionModelT = TypeVar('RegressionModelT', bound=RegressionModel)

# --- Helper Functions ---

def _get_latest_data_point(db_path: Path) -> Optional[pd.DataFrame]:
    """Loads only the most recent data point from the database."""
    logger.info(f"Loading latest data point from {db_path}...")
    conn = sqlite_manager.create_connection(str(db_path))
    if not conn: return None
    try:
        cursor = conn.cursor()
        # Verify table exists first to avoid errors on empty DB
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='historical_data' LIMIT 1;")
        if cursor.fetchone() is None:
             logger.error(f"Table 'historical_data' not found in {db_path}. Cannot load latest point.")
             return None

        query = "SELECT * FROM historical_data ORDER BY open_time DESC LIMIT 1"
        df = pd.read_sql_query(query, conn)

        if df.empty: logger.warning("Latest data DF is empty."); return None

        # Basic validation of the loaded latest point
        df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
        df.dropna(subset=['open_time'], inplace=True)
        if df.empty: logger.warning("Latest data point has invalid open_time."); return None

        df['date'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df.dropna(subset=['date'], inplace=True)
        if df.empty: logger.warning("Latest data point has invalid date conversion."); return None

        core_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in core_cols:
            if col not in df.columns:
                logger.error(f"Latest data point missing required column: '{col}'"); return None
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Drop if ANY core value is NaN
        df.dropna(subset=core_cols, inplace=True)
        if df.empty: logger.warning("Latest data point has NaN in core price/volume."); return None

        logger.info(f"Latest data point loaded: {df.iloc[0]['date']} (OpenTime: {df.iloc[0]['open_time']})")
        return df

    except (pd.errors.DatabaseError, sqlite3.Error) as db_err:
        logger.error(f"Database error loading latest data point from {db_path}: {db_err}", exc_info=True)
        return None
    except Exception as e: logger.error(f"Unexpected error loading latest data point: {e}", exc_info=True); return None
    finally:
        if conn: conn.close()


def _get_historical_indicator_price_pairs(
    db_path: Path,
    symbol_id: int,
    timeframe_id: int,
    indicator_config: Dict[str, Any],
    lag: int,
    data: pd.DataFrame,
    indicator_series_cache: IndicatorCache
) -> Optional[pd.DataFrame]:
    """Get historical pairs of indicator values and future price changes.
    
    Args:
        db_path: Path to the database file
        symbol_id: ID of the symbol being analyzed
        timeframe_id: ID of the timeframe being analyzed
        indicator_config: Configuration for the indicator
        lag: Lag value to use for correlation
        data: DataFrame containing price data
        indicator_series_cache: Cache of previously calculated indicator series
        
    Returns:
        DataFrame with indicator values and future price changes, or None if error
    """
    try:
        # Get and validate indicator name
        indicator_name = indicator_config.get('indicator_name')
        if not isinstance(indicator_name, str) or not indicator_name:
            logger.error("Missing or invalid indicator name in config")
            return None
            
        # Get and validate config ID
        config_id = indicator_config.get('config_id')
        if not isinstance(config_id, int):
            logger.error(f"Invalid config_id type for {indicator_name}")
            return None
            
        # Get and validate params
        params = indicator_config.get('params')
        if not isinstance(params, dict):
            logger.error(f"Invalid params type for {indicator_name}")
            return None

        # Calculate or retrieve indicator values
        indicator_df = _calculate_or_get_cached_indicator(
            data,
            indicator_name,
            params,
            config_id,
            indicator_series_cache
        )
        
        if indicator_df is None:
            return None

        # Get the indicator column name
        indicator_col = f"{indicator_name}_{config_id}"
        if indicator_col not in indicator_df.columns:
            logger.error(f"Indicator column {indicator_col} not found")
            return None

        # Calculate future price changes
        future_returns = data['close'].pct_change(periods=lag).shift(-lag)
        
        # Combine into pairs DataFrame
        pairs_df = pd.DataFrame({
            'indicator': indicator_df[indicator_col],
            'future_return': future_returns
        })
        
        # Drop any rows with NaN values
        pairs_df = pairs_df.dropna()
        
        if len(pairs_df) < MIN_REGRESSION_POINTS:
            logger.warning(
                f"Insufficient pairs ({len(pairs_df)}) for {indicator_name} "
                f"at lag {lag}. Need {MIN_REGRESSION_POINTS}."
            )
            return None
            
        return pairs_df
        
    except Exception as e:
        logger.error(f"Error getting historical pairs: {e}", exc_info=True)
        return None

def _calculate_or_get_cached_indicator(
    data: pd.DataFrame,
    indicator_name: str,
    params: Dict[str, Any],
    config_id: int,
    indicator_series_cache: Dict[str, pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """Calculate indicator values or retrieve from cache.
    
    Args:
        data: DataFrame containing price data
        indicator_name: Name of the indicator
        params: Parameters for the indicator
        config_id: Configuration ID
        indicator_series_cache: Cache of previously calculated indicator series
        
    Returns:
        DataFrame with indicator values, or None if error
    """
    try:
        # Generate cache key
        param_hash = utils.get_config_hash(params)
        cache_key = f"{indicator_name}_{param_hash}"
        
        # Check cache first
        if cache_key in indicator_series_cache:
            return indicator_series_cache[cache_key]
            
        # Calculate new indicator values
        indicator_df = _indicator_factory._compute_single_indicator(  # Use internal method
            data=data.copy(),
            name=indicator_name,
            config={
                'indicator_name': indicator_name,
                'params': params,
                'config_id': config_id
            }
        )
        
        if indicator_df is not None:
            indicator_series_cache[cache_key] = indicator_df
            
        return indicator_df
        
    except Exception as e:
        logger.error(f"Error calculating indicator {indicator_name}: {e}", exc_info=True)
        return None

def _perform_prediction_regression(
    pairs_df: pd.DataFrame,
    current_indicator_value: float,
    lag: int
) -> Optional[Tuple[float, float]]:
    """Perform regression to predict future returns.
    
    Args:
        pairs_df: DataFrame with indicator values and future returns
        current_indicator_value: Current value of the indicator
        lag: Lag value used for correlation
        
    Returns:
        Tuple of (predicted return, confidence score), or None if error
    """
    if not STATSMODELS_AVAILABLE or add_constant is None or OLS is None:
        logger.error("statsmodels not available for regression")
        return None
        
    try:
        if len(pairs_df) < MIN_REGRESSION_POINTS:
            logger.warning(
                f"Insufficient pairs ({len(pairs_df)}) for regression. "
                f"Need {MIN_REGRESSION_POINTS}."
            )
            return None
            
        # Prepare data for regression
        X = add_constant(pairs_df['indicator'])
        y = pairs_df['future_return']
        
        # Fit regression model
        model = OLS(y, X).fit()
        
        # Make prediction
        current_X = pd.DataFrame({
            'const': [1.0],
            'indicator': [current_indicator_value]
        })
        prediction = model.get_prediction(current_X)
        
        # Get prediction interval
        pred_mean = prediction.predicted_mean[0]
        pred_std = prediction.se_mean[0]
        
        # Calculate confidence score (inverse of standard error)
        confidence = 1.0 / (1.0 + pred_std)
        
        return pred_mean, confidence
        
    except Exception as e:
        logger.error(f"Error in regression: {e}", exc_info=True)
        return None

def _export_prediction_details(
    prediction_results: List[Dict],
    file_prefix: str,
    symbol: str,
    timeframe: str,
    latest_date: datetime,
    current_price: float
):
    """Exports the detailed prediction path to a text file."""
    if not prediction_results:
        logger.warning("No prediction results to export.")
        return

    output_filepath = config.REPORTS_DIR / f"{file_prefix}_prediction_details.txt"
    current_timestamp_str = datetime.now(timezone.utc).isoformat(timespec='milliseconds') + 'Z'
    logger.info(f"Exporting prediction details to: {output_filepath}")

    try:
        df = pd.DataFrame(prediction_results)

        # Select and rename columns for clarity
        # Ensure all expected keys from reg_res are included
        export_cols = [
            'lag', 'target_date', 'predicted_value', 'ci_lower', 'ci_upper',
            'predictor_name', 'predictor_cfg_id', 'predictor_lb_corr',
            'current_indicator_value', 'r_squared', 'adj_r_squared', 'correlation',
            'slope', 'intercept', 'n_observations'
        ]
        # Add model params/pvalues if needed, but can make file very long
        # 'model_params', 'model_pvalues'

        # Filter df to only include existing columns to avoid KeyErrors
        existing_export_cols = [col for col in export_cols if col in df.columns]
        df_export = df[existing_export_cols].copy()

        df_export.rename(columns={
            'lag': 'Lag',
            'target_date': 'Target Date (Est. UTC)',
            'predicted_value': 'Predicted Price',
            'ci_lower': 'CI Lower (95%)',
            'ci_upper': 'CI Upper (95%)',
            'predictor_name': 'Predictor',
            'predictor_cfg_id': 'Predictor CfgID',
            'predictor_lb_corr': 'Predictor LB Corr',
            'current_indicator_value': 'Indicator Val @ Lag 0',
            'r_squared': 'Regression R2',
            'adj_r_squared': 'Regression Adj R2',
            'correlation': 'Regression Corr',
            'slope': 'Regression Slope',
            'intercept': 'Regression Intercept',
            'n_observations': 'Regression Obs.'
        }, inplace=True)

        # Formatting
        prec = utils.estimate_price_precision(current_price) # Use dynamic precision
        na_rep='N/A' # Define NA representation

        # Format numeric columns safely, handling potential non-numeric entries
        num_cols_price = ['Predicted Price', 'CI Lower (95%)', 'CI Upper (95%)', 'Regression Intercept']
        num_cols_indicator = ['Indicator Val @ Lag 0'] # Might have different precision needs
        num_cols_corr = ['Predictor LB Corr', 'Regression R2', 'Regression Adj R2', 'Regression Corr', 'Regression Slope']
        int_cols = ['Regression Obs.', 'Lag', 'Predictor CfgID'] # Also format Lag and ID as int

        for col in num_cols_price:
            if col in df_export.columns: df_export[col] = df_export[col].apply(lambda x: f"{float(x):.{prec}f}" if pd.notna(x) and isinstance(x, (int, float, np.number)) else na_rep)
        for col in num_cols_indicator:
             if col in df_export.columns: df_export[col] = df_export[col].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) and isinstance(x, (int, float, np.number)) else na_rep) # Example 4dp
        for col in num_cols_corr:
             if col in df_export.columns: df_export[col] = df_export[col].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) and isinstance(x, (int, float, np.number)) else na_rep)
        for col in int_cols:
             if col in df_export.columns: df_export[col] = df_export[col].apply(lambda x: f"{int(x)}" if pd.notna(x) and isinstance(x, (int, float, np.number)) else na_rep)

        # Format datetime objects correctly
        if 'Target Date (Est. UTC)' in df_export.columns:
            df_export['Target Date (Est. UTC)'] = pd.to_datetime(df_export['Target Date (Est. UTC)'], errors='coerce', utc=True).dt.strftime('%Y-%m-%d %H:%M').fillna(na_rep)

        # Fill remaining NAs for object columns
        for col in df_export.select_dtypes(include=['object']).columns:
             if col not in num_cols_price + num_cols_indicator + num_cols_corr + int_cols + ['Target Date (Est. UTC)']:
                 df_export[col] = df_export[col].fillna(na_rep)

        # Generate Output String
        output_string = f"Prediction Details - {symbol} ({timeframe})\n"
        output_string += f"Generated: {current_timestamp_str}\n"
        output_string += f"Based on Latest Data: {latest_date.strftime('%Y-%m-%d %H:%M')} UTC (Close: {current_price:.{prec}f})\n"
        output_string += "=" * 150 + "\n" # Adjust width if needed
        with pd.option_context('display.width', 1000, 'display.max_colwidth', 40):
            output_string += df_export.to_string(index=False, justify='left', na_rep=na_rep)
        output_string += "\n" + "=" * 150

        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        output_filepath.write_text(output_string, encoding='utf-8')
        logger.info(f"Successfully exported prediction details to {output_filepath}")
        print(f"Prediction details saved to: {output_filepath}")

    except Exception as e:
        logger.error(f"Error exporting prediction details: {e}", exc_info=True)
        print("\nError saving prediction details file.")


# --- Main Prediction Function ---
def predict_price(db_path: Path, symbol: str, timeframe: str, final_target_lag: int) -> None:
    """
    Main prediction function orchestrator. Predicts for all lags up to final_target_lag.
    """
    if not STATSMODELS_AVAILABLE:
        print("\nError: Prediction requires 'statsmodels'. Cannot proceed.")
        logger.error("Prediction skipped: statsmodels missing.")
        return
    if not isinstance(final_target_lag, int) or final_target_lag <= 0:
        logger.error(f"Invalid final_target_lag ({final_target_lag}) passed to predict_price.")
        print(f"\nError: Invalid target lag ({final_target_lag}) provided for prediction.")
        return

    utils.clear_screen(); print(f"\n--- Price Prediction for {symbol} ({timeframe}) ---");
    logger.info(f"Starting prediction: {symbol}/{timeframe}, Target Lag: {final_target_lag}")

    # 1. Get latest data point
    latest_data_df = _get_latest_data_point(db_path)
    if latest_data_df is None or latest_data_df.empty:
        print("Error: Cannot load latest data point. Prediction aborted."); return
    latest_data = latest_data_df.iloc[0]; current_price = latest_data['close']; latest_date = latest_data['date']
    price_prec = utils.estimate_price_precision(current_price) # Get precision for display
    print(f"Latest Data: {latest_date.strftime('%Y-%m-%d %H:%M')} UTC - Current Close: {current_price:.{price_prec}f}")
    print(f"Predicting price path for Lags 1 to {final_target_lag}...")

    # 2. Load full historical data ONCE
    full_historical_data = data_manager.load_data(db_path)
    if full_historical_data is None or full_historical_data.empty:
        print("Error: Failed load full historical data. Prediction aborted."); return
    # Validate essential columns after loading
    if 'close' not in full_historical_data.columns or not pd.api.types.is_numeric_dtype(full_historical_data['close']):
        print("Error: 'close' column missing/invalid. Prediction aborted.")
        logger.error("Prediction aborted: historical data missing/invalid 'close'.")
        return
    if 'date' not in full_historical_data.columns or not pd.api.types.is_datetime64_any_dtype(full_historical_data['date']):
         print("Error: 'date' column missing/invalid. Prediction aborted.")
         logger.error("Prediction aborted: historical data missing/invalid 'date'.")
         return
    # Final check on historical data length relative to target lag
    if len(full_historical_data) < final_target_lag + MIN_REGRESSION_POINTS:
         logger.error(f"Insufficient history ({len(full_historical_data)}) for lag {final_target_lag} + {MIN_REGRESSION_POINTS} regression points.")
         print(f"\nError: Not enough historical data ({len(full_historical_data)}) to predict up to lag {final_target_lag}. Need at least {final_target_lag + MIN_REGRESSION_POINTS}.")
         return

    # 3. Get Symbol/Timeframe IDs (Error handling improved)
    conn_ids = None; sym_id = -1; tf_id = -1
    try:
        conn_ids = sqlite_manager.create_connection(str(db_path))
        if conn_ids is None: raise ConnectionError("Failed to connect to DB for IDs.")
        conn_ids.execute("BEGIN;"); # Start transaction for IDs
        symbol_id = sqlite_manager._get_or_create_id(conn_ids, 'symbols', 'symbol', symbol)
        timeframe_id = sqlite_manager._get_or_create_id(conn_ids, 'timeframes', 'timeframe', timeframe)
        conn_ids.commit() # Commit IDs
        logger.info(f"Using DB IDs - Symbol: {symbol_id}, Timeframe: {timeframe_id}")
    except Exception as id_err:
        logger.error(f"Failed get/create DB Symbol/Timeframe IDs: {id_err}", exc_info=True)
        if conn_ids:
            try: conn_ids.rollback(); logger.warning("Rolled back transaction for sym/tf IDs due to error.")
            except Exception as rb_err: logger.error(f"Rollback failed after ID error: {rb_err}")
        print("\nError: Failed to get Symbol/Timeframe ID. Prediction aborted.")
        return # Exit if IDs failed
    finally:
        if conn_ids: conn_ids.close()

    # --- Prediction Loop ---
    prediction_results: List[Dict] = []
    # Cache computed indicators within this prediction run (local scope)
    indicator_series_cache_predict: Dict[int, pd.DataFrame] = {}

    print("\nCalculating predictions for each lag...")
    skipped_lags = 0
    for current_lag in range(1, final_target_lag + 1):
        print(f" Processing Lag: {current_lag}/{final_target_lag}", end='\r') # Use \r for overwrite

        # 4. Find best predictor for CURRENT lag from leaderboard
        # logger.info(f"Querying leaderboard for best predictor: Lag {current_lag}") # Reduce noise
        predictor_info = leaderboard_manager.find_best_predictor_for_lag(current_lag)
        if not predictor_info:
            logger.warning(f"No predictor found for Lag = {current_lag}. Skipping this lag."); skipped_lags+=1; continue

        # Validate predictor_info structure
        ind_name = predictor_info.get('indicator_name')
        params = predictor_info.get('params') # Already a dict from find_best_predictor...
        cfg_id = predictor_info.get('config_id_source_db')
        lb_corr = predictor_info.get('correlation_value')

        if not all([ind_name, isinstance(params, dict), cfg_id is not None, lb_corr is not None]):
             logger.error(f"Incomplete predictor info from leaderboard for Lag {current_lag}: {predictor_info}. Skipping.")
             skipped_lags += 1; continue

        # logger.info(f" Predictor for Lag {current_lag}: {ind_name} (CfgID: {cfg_id}), LB Corr: {lb_corr:.4f}") # Reduce noise
        indicator_config = {'indicator_name': ind_name, 'params': params, 'config_id': cfg_id}

        # 5. Calculate current indicator value (using cached indicator if possible)
        current_ind_val = None
        indicator_df_full = indicator_series_cache_predict.get(cfg_id)
        is_cached_failure_pred = False

        if indicator_df_full is None: # Not cached
            # logger.info(f"Calculating indicator series for current value {ind_name} (ID: {cfg_id})...") # Reduce noise
            # Use the already loaded full historical data
            temp_df = _indicator_factory._compute_single_indicator(
                data=full_historical_data.copy(),
                name=ind_name,
                config=indicator_config
            )
            if temp_df is not None and not temp_df.empty:
                indicator_series_cache_predict[cfg_id] = temp_df # Cache result
                indicator_df_full = temp_df
            else:
                logger.error(f"Failed compute indicator {ind_name} Cfg {cfg_id} for current value. Skipping lag {current_lag}.")
                indicator_series_cache_predict[cfg_id] = pd.DataFrame() # Cache failure
                skipped_lags+=1; continue
        elif indicator_df_full.empty: # Cached failure
            logger.warning(f"Skipping lag {current_lag}: Indicator Cfg {cfg_id} previously failed calculation.")
            is_cached_failure_pred = True # Mark as cached failure
            skipped_lags+=1; continue

        if is_cached_failure_pred: continue # Skip if marked as failure

        # Extract the current value from the potentially multi-output DataFrame
        pattern = re.compile(rf"^{re.escape(ind_name)}_{cfg_id}(_.*)?$")
        potential_cols = [col for col in indicator_df_full.columns if pattern.match(col)]
        if not potential_cols:
            logger.error(f"Could not find output col for CfgID {cfg_id} in cached/calculated DF. Cols: {list(indicator_df_full.columns)}"); skipped_lags+=1; continue
        current_ind_col = potential_cols[0] # Use first match

        # Get the last non-NaN value for the current indicator value
        current_ind_series = indicator_df_full[current_ind_col].dropna()
        if current_ind_series.empty:
            logger.error(f"Current indicator series '{current_ind_col}' (Cfg {cfg_id}) is all NaN. Skipping lag {current_lag}.")
            skipped_lags+=1; continue
        current_ind_val = current_ind_series.iloc[-1]
        # logger.info(f" Current Indicator Value (Lag {current_lag}, Cfg {cfg_id}, Col {current_ind_col}): {current_ind_val:.4f}") # Reduce noise

        # 6. Get historical pairs (pass the prediction-specific cache)
        hist_pairs = _get_historical_indicator_price_pairs(
            db_path, symbol_id, timeframe_id, indicator_config, current_lag,
            full_historical_data, indicator_series_cache_predict # Pass prediction cache
            )
        if hist_pairs is None:
             logger.warning(f"Could not get historical pairs for lag {current_lag}. Skipping lag."); skipped_lags+=1; continue

        # 7. Perform regression
        reg_res = _perform_prediction_regression(hist_pairs, current_ind_val, current_lag)
        if reg_res is None:
            logger.warning(f"Regression failed for lag {current_lag}. Skipping lag."); skipped_lags+=1; continue

        # 8. Estimate target date for this specific lag
        estimated_target_date = utils.estimate_future_date(latest_date, current_lag, timeframe)
        if not estimated_target_date:
            logger.warning(f"Could not estimate target date for lag {current_lag}. Using placeholder offset.")
            estimated_target_date = latest_date + timedelta(days=current_lag) # Crude fallback

        # 9. Store results
        prediction_results.append({
            "lag": current_lag,
            "target_date": estimated_target_date,
            "predictor_cfg_id": cfg_id,
            "predictor_name": ind_name,
            "predictor_params": params, # Store the params dict
            "predictor_lb_corr": lb_corr,
            "current_indicator_value": current_ind_val,
            **reg_res # Unpack all results from regression function
        })
    # --- End Prediction Loop ---
    print() # Newline after loop finishes

    if skipped_lags > 0:
        print(f"Note: Skipped {skipped_lags} lags due to missing predictors or calculation errors.")

    if not prediction_results:
        print("\nError: No successful predictions were made for any lag.")
        return

    # 10. Export Prediction Details to Text File
    try:
        safe_symbol = re.sub(r'[\\/*?:"<>|\s]+', '_', symbol)
        timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
        export_prefix = f"{timestamp_str}_{safe_symbol}_{timeframe}_predpath_{final_target_lag}lags"
        _export_prediction_details(
            prediction_results, export_prefix,
            symbol, timeframe, latest_date, current_price
        )
    except Exception as export_err:
        logger.error(f"Failed to export prediction details: {export_err}", exc_info=True)

    # 11. Display Final Results Summary (for the MAX lag predicted)
    final_prediction = prediction_results[-1] # Get the result for the last successful lag
    final_lag_actual = final_prediction['lag'] # Actual final lag achieved

    pred_p = final_prediction['predicted_value']; ci_l = final_prediction['ci_lower']; ci_u = final_prediction['ci_upper']
    r2 = final_prediction['r_squared']; reg_corr = final_prediction['correlation']; lb_corr_final = final_prediction['predictor_lb_corr']
    slope = final_prediction['slope']; intercept = final_prediction['intercept']
    final_cfg_id = final_prediction['predictor_cfg_id']; final_ind_name = final_prediction['predictor_name']
    final_target_dt_actual = final_prediction['target_date']

    print("\n--- Final Prediction Summary ---")
    print(f"Target: {final_lag_actual} periods ({final_target_dt_actual.strftime('%Y-%m-%d %H:%M') if final_target_dt_actual else 'N/A'} UTC)")
    print(f"Final Predictor: {final_ind_name} (CfgID: {final_cfg_id})")
    print(f"Predicted Price: {pred_p:.{price_prec}f}")
    print(f"95% Confidence Interval: [{ci_l:.{price_prec}f} - {ci_u:.{price_prec}f}]")
    print(f"Regression R2: {r2:.4f} (Final Lag)")
    print(f"Regression Corr: {reg_corr:.4f} (Leaderboard Corr: {lb_corr_final:.4f} for final lag predictor)")
    print(f"Model (Final Lag): Price[t+{final_lag_actual}] = {slope:.4f} * Ind[t] + {intercept:.{price_prec}f}")

    # 12. Calculate Yield to Final Target
    pct_chg = ((pred_p - current_price) / current_price) * 100 if current_price != 0 else 0
    print(f"\nExpected Gain/Loss vs Current (to Lag {final_lag_actual}): {pct_chg:.2f}%")
    approx_days = utils.estimate_days_in_periods(final_lag_actual, timeframe)
    if approx_days is not None and approx_days > 0.1:
        daily_yield = np.clip(pct_chg / approx_days, -100.0, 100.0) # Clip extreme values
        print(f"Approx Daily Yield: {daily_yield:.3f}% (over ~{approx_days:.1f} days)")
    else: print("Cannot estimate meaningful daily yield.")

    # 13. Generate Plot with full prediction path
    try:
        # Use the same prefix generated for the export function
        plot_prefix = export_prefix
        # Ensure dates/prices start from the actual current data point (lag 0)
        plot_dates = [latest_date] + [res['target_date'] for res in prediction_results]
        plot_prices = [current_price] + [res['predicted_value'] for res in prediction_results]
        plot_ci_lower = [current_price] + [res['ci_lower'] for res in prediction_results]
        plot_ci_upper = [current_price] + [res['ci_upper'] for res in prediction_results]

        plot_predicted_path(
            plot_dates, plot_prices, plot_ci_lower, plot_ci_upper,
            timeframe, symbol, plot_prefix, final_target_lag, # Pass original target lag
            prediction_results # <<< --- ***** PASS RESULTS HERE *****
        )
    except Exception as plot_err: logger.error(f"Failed generate prediction plot: {plot_err}", exc_info=True); print("\nWarning: Could not generate plot.")


# --- Plotting Function ---
def plot_predicted_path(
    dates: DateSequence,
    prices: PriceSequence,
    ci_lower: PriceSequence,
    ci_upper: PriceSequence,
    timeframe: str, symbol: str, file_prefix: str, final_target_lag: int,
    prediction_results: List[Dict] # <<< --- ***** ADD ARGUMENT HERE *****
):
    """Generates plot showing the predicted path and CI bands across lags."""
    logger.info("Generating prediction path plot...")
    if not dates or len(dates) != len(prices) or len(dates) != len(ci_lower) or len(dates) != len(ci_upper):
        logger.error("Mismatched data lengths for plotting prediction path.")
        return
    if len(dates) < 2: # Need at least start and one prediction
        logger.warning("Not enough data points (need >= 2) to plot prediction path.")
        return

    start_date = dates[0]
    start_price = prices[0]
    # Use the last valid date from the plotted data
    target_date = dates[-1]

    # Ensure dates are timezone-aware for plotting (assume UTC if naive)
    aware_dates = [d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d for d in dates]

    plot_dpi = config.DEFAULTS.get("plot_dpi", 150) # Use configured DPI
    fig, ax = plt.subplots(figsize=(12, 7), dpi=plot_dpi)

    # Filter out non-finite values before plotting
    valid_indices = [i for i, (p, l, u) in enumerate(zip(prices, ci_lower, ci_upper)) if all(np.isfinite([p, l, u]))]
    if len(valid_indices) < 1: # Need at least the start point
        logger.error("Not enough finite data points to plot prediction path after filtering.")
        plt.close(fig)
        return

    # Use only valid points for plotting
    plot_aware_dates = [aware_dates[i] for i in valid_indices]
    plot_prices = [prices[i] for i in valid_indices]
    plot_ci_lower = [ci_lower[i] for i in valid_indices]
    plot_ci_upper = [ci_upper[i] for i in valid_indices]

    # Plot predicted path and CI bands
    if len(plot_aware_dates) >= 2: # Need at least two points for a line/fill
        ax.plot(plot_aware_dates, plot_prices, marker='.', linestyle='-', markersize=4, color='blue', label='Predicted Price Path')
        ax.fill_between(plot_aware_dates, plot_ci_lower, plot_ci_upper, color='skyblue', alpha=0.4, interpolate=True, label='95% CI Band')
        # Plot final point marker only if we have more than just the start
        ax.plot(plot_aware_dates[-1], plot_prices[-1], marker='*', markersize=10, color='red', linestyle='None', label=f'Final Prediction ({target_date.strftime("%Y-%m-%d %H:%M")})') # Added linestyle='None'
        # Annotate final point
        price_prec = utils.estimate_price_precision(start_price) # Use dynamic precision
        # Annotate using the actual lag of the final prediction
        final_actual_lag = prediction_results[-1]['lag'] if prediction_results else len(plot_prices) - 1
        ax.text(
            plot_aware_dates[-1],
            plot_prices[-1],
            f' Lag {final_actual_lag}\n ${plot_prices[-1]:.{price_prec}f}',
            va='bottom',
            ha='left',
            fontsize=9,
            color='red',
        )

    # Always plot start point
    ax.plot(plot_aware_dates[0], plot_prices[0], marker='o', markersize=8, color='black', linestyle='None', label=f'Start ({start_date.strftime("%Y-%m-%d %H:%M")})') # Added linestyle='None'
    # Annotate start point
    price_prec = utils.estimate_price_precision(start_price)
    ax.text(plot_aware_dates[0], plot_prices[0], f' Start\n ${plot_prices[0]:.{price_prec}f}', va='bottom', ha='right', fontsize=9, color='black')


    ax.set_title(f"Predicted Price Path: {symbol} ({timeframe}) - Up to {final_target_lag} Periods Attempted")
    ax.set_xlabel("Date (UTC)"); ax.set_ylabel("Price")
    ax.grid(True, linestyle='--', alpha=0.6); ax.legend(loc='best')

    # Date formatting (robust handling)
    fig.autofmt_xdate(rotation=30)
    try: # Estimate time delta robustly
        if len(plot_aware_dates) >= 2:
             time_delta_days = (plot_aware_dates[-1] - plot_aware_dates[0]).total_seconds() / 86400.0
        else: time_delta_days = 1
    except: time_delta_days = 30 # Default if error
    date_fmt = '%Y-%m-%d %H:%M' if time_delta_days <= 5 else ('%Y-%m-%d' if time_delta_days <= 90 else '%b %Y')
    try:
         ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt, tz=timezone.utc))
         ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=12, tz=timezone.utc))
    except Exception as fmt_err:
         logger.warning(f"Could not apply advanced date formatting: {fmt_err}")

    # Adjust y-axis limits based on CI range to ensure visibility
    try:
        min_y = min(min(plot_ci_lower), start_price)
        max_y = max(max(plot_ci_upper), start_price)
        y_range = max_y - min_y if max_y > min_y else abs(start_price * 0.1) if start_price != 0 else 1.0 # Avoid zero/small range
        y_pad = y_range * 0.1
        ax.set_ylim(min_y - y_pad, max_y + y_pad)
    except Exception as ylim_err:
        logger.warning(f"Could not dynamically set Y limits: {ylim_err}")

    fig.tight_layout()
    output_filepath = config.REPORTS_DIR / f"{file_prefix}_plot.png"
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    try: fig.savefig(output_filepath); logger.info(f"Saved prediction plot: {output_filepath}"); print(f"\nPrediction plot saved to: {output_filepath}")
    except Exception as e: logger.error(f"Failed save plot {output_filepath.name}: {e}", exc_info=True)
    finally: plt.close(fig) # Ensure figure is closed


class Predictor:
    def __init__(self):
        """Initialize the predictor."""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels is required for prediction functionality. Please install it with 'pip install statsmodels'.")
    
    def get_latest_data_point(self, db_path: Path) -> Optional[pd.DataFrame]:
        """Get the latest data point from the database."""
        return _get_latest_data_point(db_path)
    
    def get_historical_indicator_price_pairs(
        self,
        db_path: Path,
        symbol_id: int,
        timeframe_id: int,
        indicator_config: Dict[str, Any],
        lag: int,
        full_historical_data: pd.DataFrame,
        indicator_series_cache: Dict[str, pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Get historical indicator and price pairs for prediction."""
        return _get_historical_indicator_price_pairs(
            db_path, symbol_id, timeframe_id, indicator_config,
            lag, full_historical_data, indicator_series_cache
        )
    
    def perform_prediction_regression(
        self,
        hist_pairs: pd.DataFrame,
        current_indicator_value: float,
        current_lag: int
    ) -> Optional[Tuple[float, float]]:
        """Perform regression for prediction."""
        return _perform_prediction_regression(hist_pairs, current_indicator_value, current_lag)
    
    def export_prediction_details(
        self,
        prediction_results: List[Dict],
        file_prefix: str,
        symbol: str,
        timeframe: str,
        latest_date: datetime,
        current_price: float
    ):
        """Export prediction details to files."""
        return _export_prediction_details(
            prediction_results, file_prefix, symbol,
            timeframe, latest_date, current_price
        )
    
    def predict_price(
        self,
        db_path: Path,
        symbol: str,
        timeframe: str,
        final_target_lag: int
    ) -> None:
        """Predict future price based on historical data."""
        return predict_price(db_path, symbol, timeframe, final_target_lag)
    
    def plot_predicted_path(
        self,
        dates: DateSequence,
        prices: PriceSequence,
        ci_lower: PriceSequence,
        ci_upper: PriceSequence,
        timeframe: str,
        symbol: str,
        file_prefix: str,
        final_target_lag: int,
        prediction_results: List[Dict]
    ):
        """Plot the predicted price path with confidence intervals."""
        return plot_predicted_path(
            dates, prices, ci_lower, ci_upper,
            timeframe, symbol, file_prefix,
            final_target_lag, prediction_results
        )


if __name__ == '__main__':
    print("Predictor module. Run via main.py.")