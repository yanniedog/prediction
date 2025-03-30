# predictor.py
import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta, timezone
import re
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False; sm = None
    # Log error instead of printing directly during import
    logging.getLogger(__name__).error("Predictor requires 'statsmodels'. Please install it (`pip install statsmodels`). Prediction functionality will be disabled.")

import config # Ensure config is imported
import utils
import data_manager
import indicator_factory
import leaderboard_manager
import sqlite_manager

logger = logging.getLogger(__name__)

MIN_REGRESSION_POINTS = 30 # Minimum points for regression

# --- Helper Functions ---
def _get_latest_data_point(db_path: Path) -> Optional[pd.DataFrame]:
    """Loads only the most recent data point from the database."""
    logger.info(f"Loading latest data point from {db_path}...")
    conn = sqlite_manager.create_connection(str(db_path))
    if not conn: return None
    try:
        cursor = conn.cursor()
        # Check if table exists first
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='historical_data';")
        if cursor.fetchone() is None: logger.error(f"'historical_data' table not found in {db_path}."); return None

        query = "SELECT * FROM historical_data ORDER BY open_time DESC LIMIT 1"
        df = pd.read_sql_query(query, conn)

        if df.empty: logger.warning("Latest data DF is empty."); return None

        # Basic validation of the loaded latest point
        df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
        df.dropna(subset=['open_time'], inplace=True)
        if df.empty: logger.warning("Latest data point has invalid open_time."); return None

        df['date'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df.dropna(subset=['date'], inplace=True) # Drop if date conversion failed
        if df.empty: logger.warning("Latest data point has invalid date conversion."); return None

        # Ensure core price/volume columns exist and are numeric
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

# Removed _calculate_periods_to_date - calculation is now based on max_lag from main

def _get_historical_indicator_price_pairs(
    db_path: Path, symbol_id: int, timeframe_id: int, indicator_config: Dict[str, Any], lag: int,
    full_historical_data: pd.DataFrame, indicator_series_cache: Dict[int, pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """
    Fetches/calculates historical indicator, returns aligned (Indicator[t], Close[t+lag]) pairs.
    Uses provided full historical data and indicator cache.
    """
    config_id = indicator_config.get('config_id', 'N/A')
    indicator_name = indicator_config.get('indicator_name', 'Unknown')
    logger.info(f"Preparing historical pairs: Cfg {config_id} ('{indicator_name}'), Lag {lag}...")

    # Use cached indicator series if available
    if config_id in indicator_series_cache:
        indicator_df = indicator_series_cache[config_id]
        if not isinstance(indicator_df, pd.DataFrame):
             logger.warning(f"Invalid cached data for Cfg {config_id}. Recalculating.")
             indicator_df = None # Force recalc
        else:
             logger.debug(f"Using cached indicator series for Cfg {config_id}")
    else:
        indicator_df = None

    # Calculate if not cached or cache was invalid
    if indicator_df is None:
        logger.debug(f"Calculating '{indicator_name}' historical for pairs (ID: {config_id})")
        # Pass a copy to ensure the original full_historical_data isn't modified by the factory
        indicator_df = indicator_factory._compute_single_indicator(full_historical_data.copy(), indicator_config)
        if indicator_df is None or indicator_df.empty:
            logger.error(f"Failed calc historical indicator Cfg {config_id}.")
            indicator_series_cache[config_id] = pd.DataFrame() # Cache empty df on failure
            return None
        indicator_series_cache[config_id] = indicator_df # Cache the result
        logger.debug(f"Cached new indicator series for Cfg {config_id}")

    # Find the correct output column (handles multi-output indicators)
    potential_cols = [col for col in indicator_df.columns if col.startswith(f"{indicator_name}_{config_id}")]
    if not potential_cols: logger.error(f"Could not find output col like '{indicator_name}_{config_id}_...'"); return None
    indicator_col_name = potential_cols[0]
    if len(potential_cols) > 1: logger.warning(f"Multiple outputs for {indicator_name} Cfg {config_id}: {potential_cols}. Using first: '{indicator_col_name}'.")
    logger.debug(f"Using indicator column '{indicator_col_name}' for regression for lag {lag}.")

    # Align Indicator[t] and Close[t+lag] using the full historical data
    df_reg = pd.DataFrame(index=full_historical_data.index)
    # Reindex indicator result to match the main dataframe's index BEFORE assigning
    indicator_reindexed = indicator_df[indicator_col_name].reindex(full_historical_data.index)
    df_reg['Indicator_t'] = indicator_reindexed
    df_reg['Close_t_plus_lag'] = full_historical_data['close'].shift(-lag)

    initial_rows = len(df_reg)
    df_reg.dropna(subset=['Indicator_t', 'Close_t_plus_lag'], inplace=True)
    logger.info(f"Found {len(df_reg)} valid (Indicator[t], Close[t+{lag}]) pairs (dropped {initial_rows - len(df_reg)} NaN rows).")

    if len(df_reg) < MIN_REGRESSION_POINTS:
        logger.error(f"Insufficient historical pairs ({len(df_reg)}) for regression for lag {lag} (min: {MIN_REGRESSION_POINTS}).")
        return None

    return df_reg[['Indicator_t', 'Close_t_plus_lag']]


def _perform_prediction_regression(hist_pairs: pd.DataFrame, current_indicator_value: float, current_lag: int) -> Optional[Dict[str, Any]]:
    """Performs OLS regression and predicts future close price for a specific lag."""
    if not STATSMODELS_AVAILABLE: logger.error("Statsmodels unavailable."); return None
    logger.info(f"Performing linear regression via statsmodels for Lag={current_lag}...")
    X = hist_pairs['Indicator_t']; y = hist_pairs['Close_t_plus_lag']

    # Check for constant predictor or target (can cause issues)
    if X.nunique() <= 1: logger.error(f"Lag={current_lag}: Predictor '{X.name}' is constant."); return None
    if y.nunique() <= 1: logger.error(f"Lag={current_lag}: Target '{y.name}' is constant."); return None
    if X.isnull().all() or y.isnull().all(): logger.error(f"Lag={current_lag}: Predictor or target all NaN."); return None

    try:
        X_sm = sm.add_constant(X, prepend=True, has_constant='raise') # Use 'raise' to catch issues explicitly
        model = sm.OLS(y, X_sm, missing='drop').fit() # Use missing='drop'
        if not hasattr(model, 'summary') or not hasattr(model, 'params'):
             logger.error(f"Statsmodels OLS fit failed for Lag={current_lag}. Model object invalid."); return None

        # Basic check on model results
        if model.params.isnull().any(): logger.warning(f"Lag={current_lag}: Regression resulted in NaN parameters.")
        # Log summary at DEBUG level to avoid flooding console
        logger.debug(f"Regression Summary (Lag={current_lag}):\n" + str(model.summary()))

        # Predict
        # Ensure column names match exactly what sm.add_constant created
        # Use the actual name of the indicator column from hist_pairs
        indicator_col_actual_name = X.name
        x_pred_df = pd.DataFrame({'const': [1.0], indicator_col_actual_name: [current_indicator_value]})
        # Ensure column order matches the fitted model's exog_names
        x_pred_df = x_pred_df[model.model.exog_names]

        pred_res = model.get_prediction(x_pred_df); pred_summary = pred_res.summary_frame(alpha=0.05)
        predicted_value = pred_summary['mean'].iloc[0]
        ci_lower = pred_summary['mean_ci_lower'].iloc[0]; ci_upper = pred_summary['mean_ci_upper'].iloc[0]

        # Metrics
        r_squared = model.rsquared; adj_r_squared = model.rsquared_adj
        # Make sure to access the slope using the correct column name from X
        slope = model.params.get(indicator_col_actual_name, np.nan)
        intercept = model.params.get('const', np.nan)
        corr_from_r2 = np.sqrt(max(0, r_squared)) * np.sign(slope) if pd.notna(slope) else np.nan

        # Add checks for infinite/NaN results which can happen with poor data/models
        if not all(np.isfinite([predicted_value, ci_lower, ci_upper, slope, intercept, r_squared])):
            logger.error(f"Lag={current_lag}: Regression produced non-finite results. Pred={predicted_value}, Slope={slope}, R2={r_squared}")
            return None

        logger.info(f"Regression Prediction (Lag={current_lag}): {predicted_value:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        logger.info(f"R2: {r_squared:.4f}, Adj R2: {adj_r_squared:.4f}, Implied Corr: {corr_from_r2:.4f}")

        return {
            "predicted_value": predicted_value, "ci_lower": ci_lower, "ci_upper": ci_upper,
            "r_squared": r_squared, "adj_r_squared": adj_r_squared, "correlation": corr_from_r2,
            "intercept": intercept, "slope": slope, "model_params": model.params.to_dict(),
            "model_pvalues": model.pvalues.to_dict(), "n_observations": int(model.nobs), # Ensure int
        }
    except Exception as e: logger.error(f"Error during prediction regression for Lag={current_lag}: {e}", exc_info=True); return None


# --- NEW HELPER FUNCTION ---
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
        for col in ['Predicted Price', 'CI Lower (95%)', 'CI Upper (95%)', 'Indicator Val @ Lag 0', 'Regression Intercept']:
            df_export[col] = df_export[col].map(f'{{:.{prec}f}}'.format).fillna('N/A')
        for col in ['Predictor LB Corr', 'Regression R2', 'Regression Corr', 'Regression Slope']:
            df_export[col] = df_export[col].map('{:.4f}'.format).fillna('N/A')
        df_export['Target Date (Est. UTC)'] = pd.to_datetime(df_export['Target Date (Est. UTC)']).dt.strftime('%Y-%m-%d %H:%M')
        df_export['Regression Obs.'] = df_export['Regression Obs.'].map('{:.0f}'.format).fillna('N/A') # Format Obs count

        # Generate Output String
        output_string = f"Prediction Details - {symbol} ({timeframe})\n"
        output_string += f"Generated: {current_timestamp_str}\n"
        output_string += f"Based on Latest Data: {latest_date.strftime('%Y-%m-%d %H:%M')} UTC (Close: {current_price:.{prec}f})\n"
        output_string += "=" * 140 + "\n" # Increased width
        with pd.option_context('display.width', 1000, 'display.max_colwidth', 40): # Adjust width/colwidth as needed
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
    if not STATSMODELS_AVAILABLE: print("\nError: Prediction requires 'statsmodels'."); logger.error("Pred skipped: statsmodels missing."); return
    # Ensure final_target_lag is valid
    if not isinstance(final_target_lag, int) or final_target_lag <= 0:
        logger.error(f"Invalid final_target_lag ({final_target_lag}) passed to predict_price.")
        print(f"\nError: Invalid target lag ({final_target_lag}) provided for prediction.")
        return

    utils.clear_screen(); print(f"\n--- Price Prediction for {symbol} ({timeframe}) ---");
    logger.info(f"Starting prediction: {symbol}/{timeframe}, Target Lag: {final_target_lag}")

    # 1. Get latest data point
    latest_data_df = _get_latest_data_point(db_path)
    if latest_data_df is None or latest_data_df.empty: print("Error: Cannot load latest data point."); return
    latest_data = latest_data_df.iloc[0]; current_price = latest_data['close']; latest_date = latest_data['date']
    print(f"Latest Data: {latest_date.strftime('%Y-%m-%d %H:%M')} UTC - Current Close: {current_price:.4f}")
    print(f"Predicting price path for Lags 1 to {final_target_lag}...")

    # 2. Load full historical data ONCE
    full_historical_data = data_manager.load_data(db_path)
    if full_historical_data is None or full_historical_data.empty: print("Error: Failed load full historical data."); return
    # Ensure 'close' is numeric after loading
    if 'close' not in full_historical_data.columns or not pd.api.types.is_numeric_dtype(full_historical_data['close']):
        print("Error: 'close' column missing or not numeric in historical data.")
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
    if sym_id == -1 or tf_id == -1: print("\nError: Failed to get Symbol/Timeframe ID from database."); return

    # --- Prediction Loop ---
    prediction_results: List[Dict] = []
    indicator_series_cache: Dict[int, pd.DataFrame] = {} # Cache computed indicators within this run

    print("\nCalculating predictions for each lag...")
    skipped_lags = 0
    for current_lag in range(1, final_target_lag + 1):
        print(f" Processing Lag: {current_lag}/{final_target_lag}", end='\r') # Use \r for overwrite

        # 4. Find best predictor for CURRENT lag from leaderboard
        logger.info(f"Querying leaderboard for best predictor: Lag {current_lag}")
        predictor_info = leaderboard_manager.find_best_predictor_for_lag(current_lag)
        if not predictor_info:
            logger.warning(f"No predictor found for Lag = {current_lag}. Skipping this lag."); skipped_lags+=1; continue
        ind_name = predictor_info['indicator_name']; params = predictor_info['params']; cfg_id = predictor_info['config_id_source_db']
        lb_corr = predictor_info['correlation_value']; lb_corr_type = predictor_info['correlation_type']
        logger.info(f" Predictor for Lag {current_lag}: {ind_name} (CfgID: {cfg_id}), Corr: {lb_corr:.4f}")

        # 5. Calculate current indicator value (using cached indicator if possible)
        indicator_config = {'indicator_name': ind_name, 'params': params, 'config_id': cfg_id}
        current_ind_val = None
        if cfg_id in indicator_series_cache:
            indicator_df_full = indicator_series_cache[cfg_id]
            if isinstance(indicator_df_full, pd.DataFrame):
                 logger.debug(f"Using cached indicator series for current value (Cfg {cfg_id})")
            else: # Cached item was invalid (e.g., empty DF from previous failure)
                 logger.warning(f"Invalid cached series found for Cfg {cfg_id}. Retrying calculation.")
                 indicator_df_full = None # Force recalc
                 del indicator_series_cache[cfg_id] # Remove bad entry
        else:
            indicator_df_full = None

        if indicator_df_full is None: # Not cached or cache was bad
            logger.info(f"Calculating indicator series for current value {ind_name} (ID: {cfg_id})...")
            # Pass a copy to ensure the original full_historical_data isn't modified by the factory
            indicator_df_full = indicator_factory._compute_single_indicator(full_historical_data.copy(), indicator_config)
            if indicator_df_full is not None and not indicator_df_full.empty:
                indicator_series_cache[cfg_id] = indicator_df_full # Cache result
            else:
                logger.error(f"Failed compute indicator {ind_name} Cfg {cfg_id}. Skipping lag {current_lag}."); skipped_lags+=1; continue
                indicator_series_cache[cfg_id] = pd.DataFrame() # Cache empty to prevent repeated failure

        # Extract the current value
        if indicator_df_full is None or indicator_df_full.empty:
             logger.error(f"Indicator computation failed Cfg {cfg_id}. Cannot get current value."); skipped_lags+=1; continue
        potential_cols = [col for col in indicator_df_full.columns if col.startswith(f"{ind_name}_{cfg_id}")]
        if not potential_cols:
            logger.error(f"Could not find output col for CfgID {cfg_id}."); skipped_lags+=1; continue
        current_ind_col = potential_cols[0]
        if len(potential_cols) > 1: logger.warning(f"Predictor {ind_name} Cfg {cfg_id} multiple outputs: {potential_cols}. Using first: '{current_ind_col}'.")
        # Get the last non-NaN value for the current indicator value
        current_ind_series = indicator_df_full[current_ind_col].dropna()
        if current_ind_series.empty:
            logger.error(f"Current indicator '{current_ind_col}' is all NaN."); skipped_lags+=1; continue
        current_ind_val = current_ind_series.iloc[-1]
        logger.info(f" Current Indicator Value (Lag {current_lag}, Cfg {cfg_id}, Col {current_ind_col}): {current_ind_val:.4f}")

        # 6. Get historical pairs (using cache)
        hist_pairs = _get_historical_indicator_price_pairs(db_path, sym_id, tf_id, indicator_config, current_lag, full_historical_data, indicator_series_cache)
        if hist_pairs is None:
             logger.warning(f"Could not get historical pairs for lag {current_lag}"); skipped_lags+=1; continue

        # 7. Perform regression
        reg_res = _perform_prediction_regression(hist_pairs, current_ind_val, current_lag)
        if reg_res is None:
            logger.warning(f"Regression failed for lag {current_lag}"); skipped_lags+=1; continue

        # 8. Estimate target date for this specific lag
        estimated_target_date = utils.estimate_future_date(latest_date, current_lag, timeframe)
        if not estimated_target_date:
            logger.warning(f"Could not estimate target date for lag {current_lag}. Using placeholder offset.")
            approx_days_inc = utils.estimate_days_in_periods(1, timeframe) or (1/24.0) # Estimate days for one period
            estimated_target_date = latest_date + timedelta(days=current_lag * approx_days_inc) # Crude fallback

        # 9. Store results
        prediction_results.append({
            "lag": current_lag,
            "target_date": estimated_target_date,
            "predictor_cfg_id": cfg_id,
            "predictor_name": ind_name,
            "predictor_params": params, # Store the params dict
            "predictor_lb_corr": lb_corr,
            "current_indicator_value": current_ind_val,
            "predicted_value": reg_res['predicted_value'],
            "ci_lower": reg_res['ci_lower'],
            "ci_upper": reg_res['ci_upper'],
            "r_squared": reg_res['r_squared'],
            "correlation": reg_res['correlation'],
            "slope": reg_res['slope'],
            "intercept": reg_res['intercept'],
            "n_observations": reg_res['n_observations'] # Store obs count
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
        # Generate a unique prefix for this specific prediction run's outputs
        safe_symbol = re.sub(r'[\\/*?:"<>|\s]+', '_', symbol)
        timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
        export_prefix = f"{timestamp_str}_{safe_symbol}_{timeframe}_predpath_{final_target_lag}lags"
        _export_prediction_details(
            prediction_results,
            export_prefix,
            symbol,
            timeframe,
            latest_date,
            current_price
        )
    except Exception as export_err:
        logger.error(f"Failed to export prediction details: {export_err}", exc_info=True)


    # 11. Display Final Results Summary (for the MAX lag)
    final_prediction = prediction_results[-1] # Get the result for the final target lag
    pred_p = final_prediction['predicted_value']; ci_l = final_prediction['ci_lower']; ci_u = final_prediction['ci_upper']
    r2 = final_prediction['r_squared']; reg_corr = final_prediction['correlation']; lb_corr_final = final_prediction['predictor_lb_corr']
    slope = final_prediction['slope']; intercept = final_prediction['intercept']
    final_cfg_id = final_prediction['predictor_cfg_id']; final_ind_name = final_prediction['predictor_name']
    final_target_dt_actual = final_prediction['target_date']
    final_lag_actual = final_prediction['lag'] # Should be final_target_lag if loop completed

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
        plot_ci_lower = [current_price] + [res['ci_lower'] for res in prediction_results] # Use current price as CI bounds at t=0
        plot_ci_upper = [current_price] + [res['ci_upper'] for res in prediction_results]

        plot_predicted_path(
            plot_dates, plot_prices, plot_ci_lower, plot_ci_upper,
            timeframe, symbol, plot_prefix, final_target_lag
        )
    except Exception as plot_err: logger.error(f"Failed generate prediction plot: {plot_err}", exc_info=True); print("\nWarning: Could not generate plot.")


# --- Plotting Function (Modified to show full path) ---
def plot_predicted_path(
    dates: List[datetime],
    prices: List[float],
    ci_lower: List[float],
    ci_upper: List[float],
    timeframe: str, symbol: str, file_prefix: str, final_lag: int
):
    """Generates plot showing the predicted path and CI bands across lags."""
    logger.info("Generating prediction path plot...")
    if not dates or len(dates) != len(prices) or len(dates) != len(ci_lower) or len(dates) != len(ci_upper):
        logger.error("Mismatched data lengths for plotting prediction path.")
        return
    if len(dates) < 2: # Need at least start and one prediction
        logger.warning("Not enough data points (need > 1) to plot prediction path.")
        return

    start_date = dates[0]
    start_price = prices[0]
    target_date = dates[-1] # Final target date

    # Ensure dates are timezone-aware for plotting
    aware_dates = []
    for d in dates:
        if d.tzinfo is None:
            aware_dates.append(d.replace(tzinfo=timezone.utc)) # Assume UTC if naive
        else:
            aware_dates.append(d) # Use existing timezone

    plot_dpi = config.DEFAULTS.get("plot_dpi", 300)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=plot_dpi) # Adjusted size slightly

    # Plot predicted path and CI bands
    # Ensure we only plot valid, finite data
    valid_indices = [i for i, p in enumerate(prices) if np.isfinite(p) and np.isfinite(ci_lower[i]) and np.isfinite(ci_upper[i])]
    if len(valid_indices) < 2:
        logger.error("Not enough finite data points to plot prediction path.")
        plt.close(fig)
        return

    plot_aware_dates = [aware_dates[i] for i in valid_indices]
    plot_prices = [prices[i] for i in valid_indices]
    plot_ci_lower = [ci_lower[i] for i in valid_indices]
    plot_ci_upper = [ci_upper[i] for i in valid_indices]


    ax.plot(plot_aware_dates, plot_prices, marker='.', linestyle='-', markersize=4, color='blue', label='Predicted Price Path')
    ax.fill_between(plot_aware_dates, plot_ci_lower, plot_ci_upper, color='skyblue', alpha=0.4, interpolate=True, label='95% CI Band')

    # Plot start and end points prominently
    ax.plot(plot_aware_dates[0], plot_prices[0], marker='o', markersize=8, color='black', label=f'Start ({start_date.strftime("%Y-%m-%d %H:%M")})')
    ax.plot(plot_aware_dates[-1], plot_prices[-1], marker='*', markersize=10, color='red', label=f'Final Prediction ({target_date.strftime("%Y-%m-%d %H:%M")})')

    # Annotations (optional, can be added for start/end)
    prec = 2 if abs(start_price) > 100 else 4
    ax.text(plot_aware_dates[-1], plot_prices[-1], f' Lag {final_lag}\n ${plot_prices[-1]:.{prec}f}', va='bottom', ha='left', fontsize=9, color='red')
    ax.text(plot_aware_dates[0], plot_prices[0], f' Start\n ${plot_prices[0]:.{prec}f}', va='bottom', ha='right', fontsize=9, color='black')

    ax.set_title(f"Predicted Price Path: {symbol} ({timeframe}) - {final_lag} Periods")
    ax.set_xlabel("Date (UTC)"); ax.set_ylabel("Price")
    ax.grid(True, linestyle='--', alpha=0.6); ax.legend(loc='best')

    # Date formatting
    fig.autofmt_xdate(rotation=30)
    try: time_delta_days = (target_date - start_date).days if target_date and start_date else 30
    except: time_delta_days = 30 # Default if error
    date_fmt = '%Y-%m-%d %H:%M' if time_delta_days <= 5 else ('%Y-%m-%d' if time_delta_days <= 90 else '%b %Y')
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt, tz=timezone.utc))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=12, tz=timezone.utc))

    # Adjust y-axis limits based on CI range to ensure visibility
    min_y = min(min(plot_ci_lower), start_price)
    max_y = max(max(plot_ci_upper), start_price)
    y_range = max_y - min_y
    # Add a bit more padding if range is very small
    y_pad = y_range * 0.1 if y_range > 1e-6 else abs(start_price * 0.1) + 0.1
    ax.set_ylim(min_y - y_pad, max_y + y_pad)


    fig.tight_layout()
    output_filepath = config.REPORTS_DIR / f"{file_prefix}_plot.png"
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    try: fig.savefig(output_filepath); logger.info(f"Saved prediction plot: {output_filepath}"); print(f"\nPrediction plot saved to: {output_filepath}")
    except Exception as e: logger.error(f"Failed save plot {output_filepath.name}: {e}", exc_info=True)
    finally: plt.close(fig)


if __name__ == '__main__':
    print("Predictor module. Run via main.py.")