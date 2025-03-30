# predictor.py
import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta, timezone
import re
import sqlite3 # For specific error handling

# Use non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import statsmodels conditionally
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False; sm = None
    logging.getLogger(__name__).error("Predictor requires 'statsmodels'. Please install it (`pip install statsmodels`). Prediction functionality will be disabled.")

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

# --- Helper Functions ---

def _get_latest_data_point(db_path: Path) -> Optional[pd.DataFrame]:
    """Loads only the most recent data point from the database."""
    logger.info(f"Loading latest data point from {db_path}...")
    conn = sqlite_manager.create_connection(str(db_path))
    if not conn: return None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='historical_data' LIMIT 1;")
        if cursor.fetchone() is None: logger.error(f"'historical_data' table not found in {db_path}."); return None

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
        df.dropna(subset=core_cols, inplace=True)
        if df.empty: logger.warning("Latest data point has NaN in core price/volume."); return None

        logger.info(f"Latest data point loaded: {df.iloc[0]['date']} (Open: {df.iloc[0]['open_time']})")
        return df

    except (pd.errors.DatabaseError, sqlite3.Error) as db_err:
        logger.error(f"Database error loading latest data point from {db_path}: {db_err}", exc_info=True)
        return None
    except Exception as e: logger.error(f"Unexpected error loading latest data point: {e}", exc_info=True); return None
    finally:
        if conn: conn.close()


def _get_historical_indicator_price_pairs(
    db_path: Path, symbol_id: int, timeframe_id: int, indicator_config: Dict[str, Any], lag: int,
    full_historical_data: pd.DataFrame, indicator_series_cache: Dict[int, pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """
    Fetches/calculates historical indicator, returns aligned (Indicator[t], Close[t+lag]) pairs.
    Uses provided full historical data and indicator cache.
    """
    config_id = indicator_config.get('config_id')
    indicator_name = indicator_config.get('indicator_name', 'Unknown')
    if config_id is None:
        logger.error("Indicator config missing 'config_id'. Cannot get historical pairs.")
        return None

    logger.info(f"Preparing historical pairs: Cfg {config_id} ('{indicator_name}'), Lag {lag}...")

    # Use cached indicator series if available
    indicator_df = indicator_series_cache.get(config_id)
    if indicator_df is not None:
        if not isinstance(indicator_df, pd.DataFrame):
             logger.warning(f"Invalid cached data type for Cfg {config_id}. Recalculating.")
             indicator_df = None # Force recalc
             if config_id in indicator_series_cache: del indicator_series_cache[config_id]
        elif indicator_df.empty:
             logger.debug(f"Cached indicator series for Cfg {config_id} is empty (likely previous failure). Won't recalculate here.")
             return None # Return None if cached failure
        else:
             logger.debug(f"Using cached indicator series for Cfg {config_id}")

    # Calculate if not cached
    if indicator_df is None:
        logger.debug(f"Calculating '{indicator_name}' historical series for pairs (ID: {config_id})")
        # Pass a copy to ensure the original full_historical_data isn't modified by the factory
        indicator_df = indicator_factory._compute_single_indicator(full_historical_data.copy(), indicator_config)
        if indicator_df is None or indicator_df.empty:
            logger.error(f"Failed to calculate historical indicator Cfg {config_id}.")
            indicator_series_cache[config_id] = pd.DataFrame() # Cache empty df on failure
            return None
        indicator_series_cache[config_id] = indicator_df # Cache the result
        logger.debug(f"Cached new indicator series for Cfg {config_id}")

    # Find the correct output column(s)
    # Use a regex to handle potential suffixes more robustly
    pattern = re.compile(rf"^{re.escape(indicator_name)}_{config_id}(_.*)?$")
    potential_cols = [col for col in indicator_df.columns if pattern.match(col)]

    if not potential_cols:
        logger.error(f"Could not find output column matching pattern for {indicator_name}_{config_id} in calculated indicator DF. Columns: {list(indicator_df.columns)}")
        return None

    indicator_col_name = potential_cols[0] # Use the first matching column
    if len(potential_cols) > 1:
        logger.debug(f"Multiple outputs for {indicator_name} Cfg {config_id}: {potential_cols}. Using first: '{indicator_col_name}'.")
    logger.debug(f"Using indicator column '{indicator_col_name}' for regression for lag {lag}.")

    # Align Indicator[t] and Close[t+lag] using the full historical data
    try:
        df_reg = pd.DataFrame(index=full_historical_data.index)
        # Reindex indicator result to match the main dataframe's index BEFORE assigning
        # Ensure the selected column exists before trying to access it
        if indicator_col_name not in indicator_df.columns:
            logger.error(f"Selected indicator column '{indicator_col_name}' not found in indicator DataFrame.")
            return None
        indicator_reindexed = indicator_df[indicator_col_name].reindex(full_historical_data.index)

        # Ensure source columns are numeric before proceeding
        if not pd.api.types.is_numeric_dtype(indicator_reindexed):
            logger.warning(f"Indicator column '{indicator_col_name}' (Cfg {config_id}) is not numeric after reindex. Attempting conversion.")
            indicator_reindexed = pd.to_numeric(indicator_reindexed, errors='coerce')
        if not pd.api.types.is_numeric_dtype(full_historical_data['close']):
             logger.error("'close' column is not numeric in source data.")
             return None

        df_reg['Indicator_t'] = indicator_reindexed
        df_reg['Close_t_plus_lag'] = full_historical_data['close'].shift(-lag)

        initial_rows = len(df_reg)
        df_reg.dropna(subset=['Indicator_t', 'Close_t_plus_lag'], inplace=True)
        rows_dropped = initial_rows - len(df_reg)
        logger.info(f"Found {len(df_reg)} valid (Indicator[t], Close[t+{lag}]) pairs (dropped {rows_dropped} NaN rows).")

        if len(df_reg) < MIN_REGRESSION_POINTS:
            logger.error(f"Insufficient historical pairs ({len(df_reg)}) for regression for lag {lag} (min: {MIN_REGRESSION_POINTS}).")
            return None

        return df_reg[['Indicator_t', 'Close_t_plus_lag']]

    except Exception as e:
        logger.error(f"Error creating historical pairs for Cfg {config_id}, Lag {lag}: {e}", exc_info=True)
        return None


def _perform_prediction_regression(
    hist_pairs: pd.DataFrame, current_indicator_value: float, current_lag: int
) -> Optional[Dict[str, Any]]:
    """
    Performs OLS regression and predicts future close price for a specific lag.

    Note: This uses a simple univariate OLS model (Price[t+L] ~ Indicator[t]).
          Future enhancements could explore:
          - Multivariate regression (using multiple indicators).
          - More complex time series models (ARIMA, VAR).
          - Machine learning models (Random Forest, Gradient Boosting, Neural Networks).
          However, these add significant complexity in terms of feature engineering,
          model selection, training, and validation.
    """
    if not STATSMODELS_AVAILABLE: logger.error("Statsmodels unavailable."); return None
    logger.info(f"Performing linear regression via statsmodels for Lag={current_lag}...")

    if 'Indicator_t' not in hist_pairs.columns or 'Close_t_plus_lag' not in hist_pairs.columns:
        logger.error(f"Lag={current_lag}: Missing required columns 'Indicator_t' or 'Close_t_plus_lag'.")
        return None

    X = hist_pairs['Indicator_t']; y = hist_pairs['Close_t_plus_lag']

    # Check for constant predictor or target, or all NaNs
    if X.nunique(dropna=True) <= 1: logger.error(f"Lag={current_lag}: Predictor '{X.name}' is constant or all NaN after dropna."); return None
    if y.nunique(dropna=True) <= 1: logger.error(f"Lag={current_lag}: Target '{y.name}' is constant or all NaN after dropna."); return None
    # Ensure current indicator value is valid
    if not np.isfinite(current_indicator_value):
        logger.error(f"Lag={current_lag}: Invalid current_indicator_value ({current_indicator_value}) for prediction.")
        return None

    try:
        # Add constant, handle potential issues explicitly
        X_sm = sm.add_constant(X, prepend=True, has_constant='raise')
        # Fit model, dropping any remaining NaNs in the pair
        model = sm.OLS(y, X_sm, missing='drop').fit()

        if not hasattr(model, 'summary') or not hasattr(model, 'params'):
             logger.error(f"Statsmodels OLS fit failed for Lag={current_lag}. Model object invalid."); return None

        # Basic check on model results
        if model.params.isnull().any(): logger.warning(f"Lag={current_lag}: Regression resulted in NaN parameters.")
        logger.debug(f"Regression Summary (Lag={current_lag}):\n" + str(model.summary())) # Log summary at DEBUG

        # --- Predict ---
        # Create prediction input DataFrame matching model's design
        # Use the actual name of the indicator column from X
        indicator_col_actual_name = X.name
        # Ensure structure matches model.model.exog_names which includes 'const'
        x_pred_data = {'const': [1.0], indicator_col_actual_name: [current_indicator_value]}
        x_pred_df = pd.DataFrame(x_pred_data)
        # Reorder columns to strictly match the model's expectation
        try:
            x_pred_df = x_pred_df[model.model.exog_names]
        except KeyError as e:
            logger.error(f"Lag={current_lag}: Error matching prediction columns to model columns. Model expects: {model.model.exog_names}, Got: {list(x_pred_df.columns)}. Error: {e}")
            return None

        pred_res = model.get_prediction(x_pred_df)
        pred_summary = pred_res.summary_frame(alpha=0.05) # 95% CI

        # Extract results carefully
        predicted_value = pred_summary['mean'].iloc[0]
        ci_lower = pred_summary['mean_ci_lower'].iloc[0]
        ci_upper = pred_summary['mean_ci_upper'].iloc[0]

        # --- Metrics ---
        r_squared = model.rsquared; adj_r_squared = model.rsquared_adj
        # Access slope using the actual indicator column name
        slope = model.params.get(indicator_col_actual_name, np.nan)
        intercept = model.params.get('const', np.nan)
        # Implied correlation from R-squared and slope sign
        corr_from_r2 = np.sqrt(max(0, r_squared)) * np.sign(slope) if pd.notna(slope) and r_squared >= 0 else np.nan

        # Check for infinite/NaN results which can happen with poor data/models
        if not all(np.isfinite([predicted_value, ci_lower, ci_upper, slope, intercept, r_squared])):
            logger.error(f"Lag={current_lag}: Regression produced non-finite results. Pred={predicted_value}, Slope={slope}, R2={r_squared}")
            return None

        logger.info(f"Regression Prediction (Lag={current_lag}): {predicted_value:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        logger.info(f"  Metrics: R2={r_squared:.4f}, AdjR2={adj_r_squared:.4f}, ImpliedCorr={corr_from_r2:.4f}, Slope={slope:.4f}, N={int(model.nobs)}")

        return {
            "predicted_value": predicted_value, "ci_lower": ci_lower, "ci_upper": ci_upper,
            "r_squared": r_squared, "adj_r_squared": adj_r_squared, "correlation": corr_from_r2,
            "intercept": intercept, "slope": slope, "model_params": model.params.to_dict(),
            "model_pvalues": model.pvalues.to_dict(), "n_observations": int(model.nobs),
        }
    except Exception as e: logger.error(f"Error during prediction regression for Lag={current_lag}: {e}", exc_info=True); return None


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
    current_timestamp_str = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
    logger.info(f"Exporting prediction details to: {output_filepath}")

    try:
        df = pd.DataFrame(prediction_results)

        # Select and rename columns for clarity
        df_export = df[[
            'lag', 'target_date', 'predicted_value', 'ci_lower', 'ci_upper',
            'predictor_name', 'predictor_cfg_id', 'predictor_lb_corr',
            'current_indicator_value', 'r_squared', 'correlation', 'slope', 'intercept',
            'n_observations' # Added observation count
        ]].copy()

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
            'correlation': 'Regression Corr',
            'slope': 'Regression Slope',
            'intercept': 'Regression Intercept',
            'n_observations': 'Regression Obs.' # Added Obs. count
        }, inplace=True)

        # Formatting
        prec = 2 if abs(current_price) > 100 else 4
        # Format numeric columns safely, handling potential non-numeric entries
        for col in ['Predicted Price', 'CI Lower (95%)', 'CI Upper (95%)', 'Indicator Val @ Lag 0', 'Regression Intercept']:
            df_export[col] = df_export[col].apply(lambda x: f"{float(x):.{prec}f}" if pd.notna(x) and isinstance(x, (int, float)) else 'N/A')
        for col in ['Predictor LB Corr', 'Regression R2', 'Regression Corr', 'Regression Slope']:
             df_export[col] = df_export[col].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) and isinstance(x, (int, float)) else 'N/A')
        # Format datetime objects correctly
        df_export['Target Date (Est. UTC)'] = pd.to_datetime(df_export['Target Date (Est. UTC)'], errors='coerce', utc=True).dt.strftime('%Y-%m-%d %H:%M').fillna('N/A')
        # Format integer columns safely
        df_export['Regression Obs.'] = df_export['Regression Obs.'].apply(lambda x: f"{int(x)}" if pd.notna(x) and isinstance(x, (int, float)) else 'N/A')

        # Generate Output String
        output_string = f"Prediction Details - {symbol} ({timeframe})\n"
        output_string += f"Generated: {current_timestamp_str}\n"
        output_string += f"Based on Latest Data: {latest_date.strftime('%Y-%m-%d %H:%M')} UTC (Close: {current_price:.{prec}f})\n"
        output_string += "=" * 140 + "\n" # Increased width
        with pd.option_context('display.width', 1000, 'display.max_colwidth', 40):
            output_string += df_export.to_string(index=False, justify='left', na_rep='N/A') # Use na_rep
        output_string += "\n" + "=" * 140

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
    print(f"Latest Data: {latest_date.strftime('%Y-%m-%d %H:%M')} UTC - Current Close: {current_price:.4f}")
    print(f"Predicting price path for Lags 1 to {final_target_lag}...")

    # 2. Load full historical data ONCE
    full_historical_data = data_manager.load_data(db_path)
    if full_historical_data is None or full_historical_data.empty:
        print("Error: Failed load full historical data. Prediction aborted."); return
    # Validate essential columns after loading
    if 'close' not in full_historical_data.columns or not pd.api.types.is_numeric_dtype(full_historical_data['close']):
        print("Error: 'close' column missing or not numeric in historical data. Prediction aborted.")
        logger.error("Prediction aborted: historical data missing/invalid 'close' column.")
        return
    if 'date' not in full_historical_data.columns or not pd.api.types.is_datetime64_any_dtype(full_historical_data['date']):
         print("Error: 'date' column missing or not datetime in historical data. Prediction aborted.")
         logger.error("Prediction aborted: historical data missing/invalid 'date' column.")
         return
    # Final check on historical data length relative to target lag
    if len(full_historical_data) < final_target_lag + MIN_REGRESSION_POINTS:
         logger.error(f"Insufficient historical data ({len(full_historical_data)}) for target lag {final_target_lag} and min regression points {MIN_REGRESSION_POINTS}.")
         print(f"\nError: Not enough historical data ({len(full_historical_data)}) to support prediction up to lag {final_target_lag}.")
         return

    # 3. Get Symbol/Timeframe IDs
    conn_ids = sqlite_manager.create_connection(str(db_path)); sym_id = -1; tf_id = -1
    if conn_ids:
        try:
            conn_ids.execute("BEGIN;")
            sym_id = sqlite_manager._get_or_create_id(conn_ids, 'symbols', 'symbol', symbol)
            tf_id = sqlite_manager._get_or_create_id(conn_ids, 'timeframes', 'timeframe', timeframe)
            conn_ids.commit()
        except Exception as id_err:
            logger.error(f"Failed get sym/tf IDs: {id_err}", exc_info=True)
            try: conn_ids.rollback(); logger.warning("Rolled back transaction for sym/tf IDs due to error.")
            except Exception as rb_err: logger.error(f"Rollback failed after ID error: {rb_err}")
        finally:
            if conn_ids: conn_ids.close()
    else: logger.error("Failed connect for sym/tf IDs.")

    if sym_id == -1 or tf_id == -1:
        print("\nError: Failed to get Symbol/Timeframe ID from database. Prediction aborted.")
        logger.error("Prediction aborted: Could not get Symbol/Timeframe IDs.")
        return

    # --- Prediction Loop ---
    prediction_results: List[Dict] = []
    # Cache computed indicators within this prediction run (local scope)
    indicator_series_cache_predict: Dict[int, pd.DataFrame] = {}

    print("\nCalculating predictions for each lag...")
    skipped_lags = 0
    for current_lag in range(1, final_target_lag + 1):
        print(f" Processing Lag: {current_lag}/{final_target_lag}", end='\r') # Use \r for overwrite

        # 4. Find best predictor for CURRENT lag from leaderboard
        logger.info(f"Querying leaderboard for best predictor: Lag {current_lag}")
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

        logger.info(f" Predictor for Lag {current_lag}: {ind_name} (CfgID: {cfg_id}), LB Corr: {lb_corr:.4f}")
        indicator_config = {'indicator_name': ind_name, 'params': params, 'config_id': cfg_id}

        # 5. Calculate current indicator value (using cached indicator if possible)
        # This requires the full historical series of the indicator
        current_ind_val = None
        indicator_df_full = indicator_series_cache_predict.get(cfg_id)

        if indicator_df_full is None: # Not cached
            logger.info(f"Calculating indicator series for current value {ind_name} (ID: {cfg_id})...")
            # Use the already loaded full historical data
            temp_df = indicator_factory._compute_single_indicator(full_historical_data.copy(), indicator_config)
            if temp_df is not None and not temp_df.empty:
                indicator_series_cache_predict[cfg_id] = temp_df # Cache result
                indicator_df_full = temp_df
            else:
                logger.error(f"Failed compute indicator {ind_name} Cfg {cfg_id} for current value. Skipping lag {current_lag}.")
                indicator_series_cache_predict[cfg_id] = pd.DataFrame() # Cache failure
                skipped_lags+=1; continue
        elif indicator_df_full.empty: # Cached failure
            logger.warning(f"Skipping lag {current_lag}: Indicator Cfg {cfg_id} previously failed calculation.")
            skipped_lags+=1; continue

        # --- Moved Regex Compilation Here ---
        # Use the same robust pattern matching as _get_historical_indicator_price_pairs
        pattern = re.compile(rf"^{re.escape(ind_name)}_{cfg_id}(_.*)?$")
        # --- End Move ---

        # Extract the current value from the potentially multi-output DataFrame
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
        logger.info(f" Current Indicator Value (Lag {current_lag}, Cfg {cfg_id}, Col {current_ind_col}): {current_ind_val:.4f}")

        # 6. Get historical pairs (pass the prediction-specific cache)
        hist_pairs = _get_historical_indicator_price_pairs(
            db_path, sym_id, tf_id, indicator_config, current_lag,
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

    prec = 2 if abs(current_price) > 100 else 4
    print("\n--- Final Prediction Summary ---")
    print(f"Target: {final_lag_actual} periods ({final_target_dt_actual.strftime('%Y-%m-%d %H:%M') if final_target_dt_actual else 'N/A'} UTC)")
    print(f"Final Predictor: {final_ind_name} (CfgID: {final_cfg_id})")
    print(f"Predicted Price: {pred_p:.{prec}f}")
    print(f"95% Confidence Interval: [{ci_l:.{prec}f} - {ci_u:.{prec}f}]")
    print(f"Regression R2: {r2:.4f} (Final Lag)")
    print(f"Regression Corr: {reg_corr:.4f} (Leaderboard Corr: {lb_corr_final:.4f} for final lag predictor)")
    print(f"Model (Final Lag): Price[t+{final_lag_actual}] = {slope:.4f} * Ind[t] + {intercept:.{prec}f}")

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
            timeframe, symbol, plot_prefix, final_target_lag # Pass original target lag for title
        )
    except Exception as plot_err: logger.error(f"Failed generate prediction plot: {plot_err}", exc_info=True); print("\nWarning: Could not generate plot.")


# --- Plotting Function ---
def plot_predicted_path(
    dates: List[datetime],
    prices: List[float],
    ci_lower: List[float],
    ci_upper: List[float],
    timeframe: str, symbol: str, file_prefix: str, final_target_lag: int
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
    target_date = dates[-1] # Final target date in the successful predictions

    # Ensure dates are timezone-aware for plotting (assume UTC if naive)
    aware_dates = [d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d for d in dates]

    plot_dpi = config.DEFAULTS.get("plot_dpi", 300)
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
        ax.plot(plot_aware_dates[-1], plot_prices[-1], marker='*', markersize=10, color='red', label=f'Final Prediction ({target_date.strftime("%Y-%m-%d %H:%M")})')
        # Annotate final point
        prec = 2 if abs(start_price) > 100 else 4
        final_actual_lag = len(prediction_results) # Correct lag number for annotation
        ax.text(plot_aware_dates[-1], plot_prices[-1], f' Lag {final_actual_lag}\n ${plot_prices[-1]:.{prec}f}', va='bottom', ha='left', fontsize=9, color='red')

    # Always plot start point
    ax.plot(plot_aware_dates[0], plot_prices[0], marker='o', markersize=8, color='black', label=f'Start ({start_date.strftime("%Y-%m-%d %H:%M")})')
    # Annotate start point
    prec = 2 if abs(start_price) > 100 else 4
    ax.text(plot_aware_dates[0], plot_prices[0], f' Start\n ${plot_prices[0]:.{prec}f}', va='bottom', ha='right', fontsize=9, color='black')


    ax.set_title(f"Predicted Price Path: {symbol} ({timeframe}) - Up to {final_target_lag} Periods Attempted")
    ax.set_xlabel("Date (UTC)"); ax.set_ylabel("Price")
    ax.grid(True, linestyle='--', alpha=0.6); ax.legend(loc='best')

    # Date formatting
    fig.autofmt_xdate(rotation=30)
    try: # Estimate time delta robustly
        time_delta_days = (plot_aware_dates[-1] - plot_aware_dates[0]).days if len(plot_aware_dates) >= 2 else 1
    except: time_delta_days = 30 # Default if error
    date_fmt = '%Y-%m-%d %H:%M' if time_delta_days <= 5 else ('%Y-%m-%d' if time_delta_days <= 90 else '%b %Y')
    try:
         ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt, tz=timezone.utc))
         ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=12, tz=timezone.utc))
    except Exception as fmt_err:
         logger.warning(f"Could not apply advanced date formatting: {fmt_err}")

    # Adjust y-axis limits based on CI range to ensure visibility
    min_y = min(min(plot_ci_lower), start_price)
    max_y = max(max(plot_ci_upper), start_price)
    y_range = max_y - min_y if max_y > min_y else 1.0 # Avoid zero range
    y_pad = y_range * 0.1
    ax.set_ylim(min_y - y_pad, max_y + y_pad)

    fig.tight_layout()
    output_filepath = config.REPORTS_DIR / f"{file_prefix}_plot.png"
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    try: fig.savefig(output_filepath); logger.info(f"Saved prediction plot: {output_filepath}"); print(f"\nPrediction plot saved to: {output_filepath}")
    except Exception as e: logger.error(f"Failed save plot {output_filepath.name}: {e}", exc_info=True)
    finally: plt.close(fig) # Ensure figure is closed


if __name__ == '__main__':
    print("Predictor module. Run via main.py.")