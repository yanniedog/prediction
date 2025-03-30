# backtester.py
# TODO: Implement a point-in-time backtesting approach for realistic performance simulation.
#       This would involve:
#       1. Splitting data into Training and Backtest sets.
#       2. Generating a leaderboard *only* using the Training set.
#       3. Running the backtest loop on the Backtest set, selecting predictors based *only* on the Training leaderboard.
#       4. Potentially using walk-forward optimization for even greater realism (though computationally expensive).
#       The current implementation checks historical regression stability of FINAL predictors, it does NOT simulate real trading.

import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime, timezone
import math
import json # Added for parsing params

# Import config for constants
import config
import utils
import data_manager
import indicator_factory
import leaderboard_manager
import sqlite_manager
import predictor # To reuse regression logic and pair generation

logger = logging.getLogger(__name__)

# Get constant from config
MIN_REGRESSION_POINTS = config.DEFAULTS.get("min_regression_points", 30) # Fallback default

def run_backtest(
    db_path: Path,
    symbol: str,
    timeframe: str,
    max_lag_backtest: int,
    num_backtest_points: int = 50 # Number of historical points to test per lag
) -> None:
    """
    Performs a simplified "historical predictor check", NOT a true backtest.

    *** CRITICAL WARNING ***
    This function uses the FINAL leaderboard generated from the ENTIRE dataset
    to select predictors. This introduces SEVERE LOOKAHEAD BIAS.
    The results DO NOT represent realistic trading performance or how the
    strategy would have performed historically based only on past data.
    It primarily serves to check the historical stability of the regression
    relationship for the predictors that were ultimately found to be best overall.
    *** CRITICAL WARNING ***

    Args:
        db_path: Path to the database containing historical price data.
        symbol: Trading symbol (e.g., 'BTCUSDT').
        timeframe: Timeframe string (e.g., '1d').
        max_lag_backtest: Maximum lag to check predictors for.
        num_backtest_points: Number of historical points to test per lag.
    """
    utils.clear_screen()
    # --- Enhanced Warnings ---
    print("\n" + "="*60)
    print("--- HISTORICAL PREDICTOR CHECK (NOT A REALISTIC BACKTEST) ---")
    print("="*60)
    print(f"\nSymbol: {symbol} ({timeframe})")
    print(f"Checking last {num_backtest_points} points for lags 1 to {max_lag_backtest}")
    print("\n" + "***" * 20)
    print("*** WARNING: SEVERE LOOKAHEAD BIAS PRESENT ***")
    print("*** Uses FINAL leaderboard (knows future best predictors). ***")
    print("*** Results DO NOT reflect realistic trading performance. ***")
    print("*** Checks historical regression stability of final predictors ONLY. ***")
    print("***" * 20 + "\n")
    logger.critical("Starting Historical Predictor Check - SEVERE LOOKAHEAD BIAS PRESENT.")
    logger.warning(f"Historical Check Parameters: {symbol}/{timeframe}, Lags: 1-{max_lag_backtest}, Points: {num_backtest_points}. Uses FINAL leaderboard.")
    # --- End Enhanced Warnings ---

    if not predictor.STATSMODELS_AVAILABLE:
        print("\nError: This check requires 'statsmodels'.")
        logger.error("Historical Check skipped: statsmodels missing.")
        return

    # 1. Load full historical data
    full_historical_data = data_manager.load_data(db_path)
    # Check data validity thoroughly
    if full_historical_data is None or full_historical_data.empty:
        print(f"Error: Could not load historical data from {db_path}.")
        logger.error(f"Historical Check failed: Cannot load data from {db_path}.")
        return
    required_cols = ['date', 'close', 'open_time'] # Check essential columns
    if not all(col in full_historical_data.columns for col in required_cols):
        missing = [c for c in required_cols if c not in full_historical_data.columns]
        print(f"Error: Historical data missing required columns: {missing}.")
        logger.error(f"Historical Check failed: Data missing columns {missing}.")
        return
    if not pd.api.types.is_numeric_dtype(full_historical_data['close']):
        print("Error: 'close' column is not numeric.")
        logger.error("Historical Check failed: 'close' column not numeric.")
        return
    if len(full_historical_data) < max_lag_backtest + num_backtest_points + MIN_REGRESSION_POINTS:
        print(f"Error: Insufficient historical data ({len(full_historical_data)} rows). Need at least {max_lag_backtest + num_backtest_points + MIN_REGRESSION_POINTS}.")
        logger.error(f"Insufficient data for historical check. Have {len(full_historical_data)}, Need {max_lag_backtest + num_backtest_points + MIN_REGRESSION_POINTS}.")
        return

    # 2. Load final leaderboard data (emphasize this is the source of bias)
    logger.warning("Loading FINAL leaderboard state for historical check (Source of LOOKAHEAD BIAS).")
    final_leaderboard = leaderboard_manager.load_leaderboard()
    if not final_leaderboard:
        print("Error: Could not load FINAL leaderboard data. Cannot proceed.")
        logger.error("Historical Check failed: cannot load final leaderboard.")
        return

    # 3. Get Symbol/Timeframe IDs (Needed for pair generation if not cached)
    conn_ids = sqlite_manager.create_connection(str(db_path)); sym_id = -1; tf_id = -1
    if conn_ids:
        try:
            conn_ids.execute("BEGIN;")
            sym_id = sqlite_manager._get_or_create_id(conn_ids, 'symbols', 'symbol', symbol)
            tf_id = sqlite_manager._get_or_create_id(conn_ids, 'timeframes', 'timeframe', timeframe)
            conn_ids.commit()
        except Exception as id_err:
             logger.error(f"Historical Check: Failed get sym/tf IDs: {id_err}", exc_info=True)
             try: conn_ids.rollback()
             except Exception as rb_err: logger.error(f"Rollback failed: {rb_err}")
        finally:
             if conn_ids: conn_ids.close()
    else: logger.error("Historical Check: Failed connect for sym/tf IDs.")

    if sym_id == -1 or tf_id == -1:
        print("\nError: Failed to get Symbol/Timeframe ID from database.")
        logger.error("Historical Check failed: Could not get Symbol/Timeframe IDs.")
        return

    # --- Historical Check Loop ---
    backtest_results = []
    # Cache indicators calculated during this check (local scope)
    indicator_series_cache_local: Dict[int, pd.DataFrame] = {}

    total_iterations = max_lag_backtest * num_backtest_points
    completed_iterations = 0

    print("\nRunning historical check iterations...")
    for lag in range(1, max_lag_backtest + 1):
        # Find best predictor for this lag ONCE from the FINAL leaderboard
        predictor_key_pos = (lag, 'positive')
        predictor_key_neg = (lag, 'negative')
        best_predictor_info = None
        corr_pos = final_leaderboard.get(predictor_key_pos, {}).get('correlation_value', -np.inf)
        corr_neg = final_leaderboard.get(predictor_key_neg, {}).get('correlation_value', np.inf)

        # Choose based on absolute correlation value from the final leaderboard
        # Ensure values are valid floats before comparing
        valid_pos = isinstance(corr_pos, (int, float)) and pd.notna(corr_pos)
        valid_neg = isinstance(corr_neg, (int, float)) and pd.notna(corr_neg)
        abs_pos = abs(corr_pos) if valid_pos else -np.inf
        abs_neg = abs(corr_neg) if valid_neg else -np.inf

        if abs_pos >= abs_neg and valid_pos:
            best_predictor_info = final_leaderboard.get(predictor_key_pos)
        elif valid_neg:
            best_predictor_info = final_leaderboard.get(predictor_key_neg)

        if best_predictor_info:
            best_predictor_info['lag'] = lag # Add lag info for context

        # Validate predictor info structure
        if (not best_predictor_info or
            not best_predictor_info.get('config_id_source_db') or
            not best_predictor_info.get('indicator_name') or
            not best_predictor_info.get('config_json')):
            logger.warning(f"Historical Check: No valid predictor found for Lag = {lag} in FINAL leaderboard. Skipping lag.")
            completed_iterations += num_backtest_points # Increment progress even if skipped
            continue

        ind_name = best_predictor_info['indicator_name']
        cfg_id = best_predictor_info['config_id_source_db']
        try:
            # Attempt to parse params here. If fails, skip lag.
            params = json.loads(best_predictor_info['config_json'])
            indicator_config = {'indicator_name': ind_name, 'params': params, 'config_id': cfg_id}
        except json.JSONDecodeError:
            logger.error(f"Historical Check: Failed to parse params for predictor CfgID {cfg_id} (Lag {lag}). Skipping lag.")
            completed_iterations += num_backtest_points; continue
        except Exception as e:
             logger.error(f"Historical Check: Error preparing predictor config for CfgID {cfg_id} (Lag {lag}): {e}. Skipping lag.")
             completed_iterations += num_backtest_points; continue

        logger.info(f"Historical Check Lag {lag}: Using Predictor CfgID {cfg_id} ('{ind_name}') from FINAL leaderboard.")

        for i in range(num_backtest_points):
            # t = index for predictor calculation
            # target_idx = index for actual price verification
            # We iterate from the most recent point backwards
            t = len(full_historical_data) - 1 - lag - i
            target_idx = t + lag

            # Basic bounds check
            if t < 0 or target_idx >= len(full_historical_data):
                logger.warning(f"Historical Check: Index out of bounds (t={t}, target={target_idx}). Stopping early for lag {lag}.")
                break # Stop testing this lag if we run out of data

            current_progress = (completed_iterations / total_iterations) * 100 if total_iterations > 0 else 0
            print(f" Progress: {current_progress:.1f}% (Lag {lag}, Point {i+1}/{num_backtest_points})", end='\r')

            try:
                # Data slices
                # Data for regression uses history *up to* point t
                data_for_regression = full_historical_data.iloc[:t+1]
                actual_price = full_historical_data.iloc[target_idx]['close']
                actual_date = full_historical_data.iloc[target_idx]['date']
                predictor_date = full_historical_data.iloc[t]['date']

                # Calculate current indicator value at time t
                # Use the local cache, but calculate if missing
                current_ind_val = None
                indicator_df_cached = indicator_series_cache_local.get(cfg_id)

                if indicator_df_cached is None: # Not cached or previously failed
                    logger.debug(f"Historical Check: Calculating full indicator series for CfgID {cfg_id} (up to point {t})")
                    # Pass copy to avoid modification issues, use full history for stable calculation
                    indicator_df_full_hist = indicator_factory._compute_single_indicator(
                        full_historical_data.copy(), # Use full history for calc
                        indicator_config
                    )
                    if indicator_df_full_hist is not None and not indicator_df_full_hist.empty:
                        indicator_series_cache_local[cfg_id] = indicator_df_full_hist
                        indicator_df_cached = indicator_df_full_hist # Use the newly calculated df
                    else:
                        logger.error(f"Historical Check: Failed compute indicator Cfg {cfg_id} for point {t}. Skipping point.")
                        indicator_series_cache_local[cfg_id] = pd.DataFrame() # Cache empty to avoid re-trying
                        completed_iterations += 1; continue
                # else: logger.debug(f"Historical Check: Using cached indicator series for CfgID {cfg_id}")

                # Get the indicator column name (handles multi-output)
                potential_cols = [col for col in indicator_df_cached.columns if col.startswith(f"{ind_name}_{cfg_id}")]
                if not potential_cols:
                    logger.error(f"Historical Check: No output col found for CfgID {cfg_id}. Skipping point {t}, lag {lag}.")
                    completed_iterations += 1; continue
                current_ind_col = potential_cols[0]

                # Get value at index t from the cached series
                if t < len(indicator_df_cached):
                    current_ind_val = indicator_df_cached[current_ind_col].iloc[t]
                    if pd.isna(current_ind_val):
                         logger.warning(f"Historical Check: Indicator value NaN at index {t} for CfgID {cfg_id}, Lag {lag}. Skipping point.")
                         completed_iterations += 1; continue
                else: # Should not happen if caching logic is correct, but safety check
                    logger.error(f"Historical Check: Index {t} out of bounds for indicator CfgID {cfg_id}. Skipping point.")
                    completed_iterations += 1; continue

                # Get historical pairs using data *up to time t*
                # Pass an empty dict to force recalc on slice (don't use main series cache here)
                hist_pairs = predictor._get_historical_indicator_price_pairs(
                    db_path, sym_id, tf_id, indicator_config, lag,
                    data_for_regression, # Use data only up to t
                    {} # Use a temporary empty cache for pair generation
                )
                if hist_pairs is None or len(hist_pairs) < MIN_REGRESSION_POINTS:
                    logger.warning(f"Historical Check: Insufficient regression pairs ({len(hist_pairs) if hist_pairs is not None else 0}) at point {t}, lag {lag}. Min={MIN_REGRESSION_POINTS}. Skipping.")
                    completed_iterations += 1; continue

                # Perform regression using the historical pairs up to t
                reg_res = predictor._perform_prediction_regression(hist_pairs, current_ind_val, lag)
                if reg_res is None:
                    logger.warning(f"Historical Check: Regression failed at point {t}, lag {lag}. Skipping.")
                    completed_iterations += 1; continue

                # Store result
                predicted_price = reg_res['predicted_value']
                error = predicted_price - actual_price
                pct_error = (error / actual_price) * 100 if actual_price != 0 else np.inf

                backtest_results.append({
                    'Lag': lag,
                    'Test Point Index (i)': i,
                    'Predictor Time (t)': predictor_date, # Use datetime objects
                    'Target Time (t+lag)': actual_date,   # Use datetime objects
                    'Actual Price': actual_price,
                    'Predicted Price': predicted_price,
                    'Error': error,
                    'Percent Error': pct_error,
                    'Predictor CfgID': cfg_id,
                    'Predictor Name': ind_name,
                    'Indicator Value @ t': current_ind_val,
                    'Regression R2': reg_res['r_squared']
                })

            except Exception as iter_err:
                logger.error(f"Historical Check: Error during iteration (Lag {lag}, Point {i}): {iter_err}", exc_info=True)
            finally:
                completed_iterations += 1 # Ensure progress increments even on error within loop

    print("\nHistorical check iterations complete.") # Final newline after progress indicator

    # 4. Analyze and Report Results
    if not backtest_results:
        print("\nHistorical check finished with no results.")
        logger.warning("Historical check completed but no results were generated.")
        return

    results_df = pd.DataFrame(backtest_results)

    # Calculate overall metrics
    mae = np.mean(np.abs(results_df['Error']))
    rmse = np.sqrt(np.mean(results_df['Error']**2))
    # Calculate MAPE carefully, excluding zero actual prices and infs
    valid_pct_err = results_df.loc[(results_df['Actual Price'] != 0) & np.isfinite(results_df['Percent Error']), 'Percent Error']
    mape = np.mean(np.abs(valid_pct_err)) if not valid_pct_err.empty else np.nan

    print("\n--- Historical Predictor Check Overall Summary ---")
    print("REMINDER: Results affected by LOOKAHEAD BIAS in predictor selection.")
    print(f"Total Predictions Tested: {len(results_df)}")
    print(f"MAE (Mean Absolute Error):  {mae:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%" if pd.notna(mape) else "MAPE: N/A")

    # Calculate metrics per lag
    metrics_per_lag = results_df.groupby('Lag').agg(
        Predictions=('Error', 'size'),
        MAE=('Error', lambda x: np.mean(np.abs(x))),
        RMSE=('Error', lambda x: np.sqrt(np.mean(x**2))),
        # Calculate MAPE per group, handling potential infs/NaNs within the lambda
        MAPE=('Percent Error', lambda x: np.mean(np.abs(x.loc[np.isfinite(x) & (x != 0)])) if not x.loc[np.isfinite(x) & (x != 0)].empty else np.nan),
        Mean_R2=('Regression R2', 'mean')
    ).reset_index()

    print("\n--- Historical Predictor Check Metrics Per Lag ---")
    metrics_per_lag['MAPE'] = metrics_per_lag['MAPE'].map(lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A')
    metrics_per_lag['MAE'] = metrics_per_lag['MAE'].map('{:.4f}'.format)
    metrics_per_lag['RMSE'] = metrics_per_lag['RMSE'].map('{:.4f}'.format)
    metrics_per_lag['Mean_R2'] = metrics_per_lag['Mean_R2'].map('{:.3f}'.format)
    # Increase display width for better table formatting
    with pd.option_context('display.width', 1000, 'display.max_columns', None):
        print(metrics_per_lag.to_string(index=False))

    # Save detailed results to CSV
    output_filename = f"{symbol}_{timeframe}_historical_check_details_{max_lag_backtest}lags_{num_backtest_points}pts.csv"
    output_filepath = config.REPORTS_DIR / output_filename
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        results_df_sorted = results_df.sort_values(by=['Lag', 'Test Point Index (i)']).copy()
        # Format dates for CSV readability (UTC)
        results_df_sorted['Predictor Time (t)'] = results_df_sorted['Predictor Time (t)'].dt.strftime('%Y-%m-%d %H:%M:%S')
        results_df_sorted['Target Time (t+lag)'] = results_df_sorted['Target Time (t+lag)'].dt.strftime('%Y-%m-%d %H:%M:%S')

        results_df_sorted.to_csv(output_filepath, index=False, float_format='%.6f')
        print(f"\nDetailed historical check results saved to: {output_filepath}")
        logger.info(f"Historical Check details saved to: {output_filepath}")
    except Exception as e:
        print(f"\nError saving detailed historical check results: {e}")
        logger.error(f"Failed to save historical check CSV: {e}", exc_info=True)

# Example of how to potentially call it (e.g., from main.py prompt)
# if __name__ == '__main__':
#     print("Backtester (Historical Check) module should be run via main.py.")