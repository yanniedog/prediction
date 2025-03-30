# backtester.py (New File)
import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime, timezone
import math
import json # Added for parsing params

import config
import utils
import data_manager
import indicator_factory
import leaderboard_manager
import sqlite_manager
import predictor # To reuse regression logic and pair generation

logger = logging.getLogger(__name__)

def run_backtest(
    db_path: Path,
    symbol: str,
    timeframe: str,
    max_lag_backtest: int,
    num_backtest_points: int = 50 # Number of historical points to test per lag
) -> None:
    """
    Performs a simplified backtest using the final leaderboard.
    WARNING: Contains lookahead bias as predictor selection uses the final leaderboard.
    Tests the regression accuracy for the selected predictor historically.
    """
    utils.clear_screen()
    print(f"\n--- Simplified Backtest for {symbol} ({timeframe}) ---")
    print(f"Testing last {num_backtest_points} points for lags 1 to {max_lag_backtest}")
    print("WARNING: Uses final leaderboard, contains lookahead bias.")
    logger.info(f"Starting simplified backtest: {symbol}/{timeframe}, Lags: 1-{max_lag_backtest}, Points: {num_backtest_points}")

    if not predictor.STATSMODELS_AVAILABLE:
        print("\nError: Backtesting requires 'statsmodels'.")
        logger.error("Backtest skipped: statsmodels missing.")
        return

    # 1. Load full historical data
    full_historical_data = data_manager.load_data(db_path)
    if full_historical_data is None or full_historical_data.empty or len(full_historical_data) < max_lag_backtest + num_backtest_points + predictor.MIN_REGRESSION_POINTS:
        print("Error: Insufficient historical data for the requested backtest range.")
        logger.error("Insufficient data for backtest.")
        return
    # Ensure 'close' is numeric
    if 'close' not in full_historical_data.columns or not pd.api.types.is_numeric_dtype(full_historical_data['close']):
        print("Error: 'close' column missing or not numeric.")
        return

    # 2. Load final leaderboard data (from DB)
    # It's better to load fresh than pass potentially stale data
    logger.info("Loading final leaderboard state for backtesting...")
    final_leaderboard = leaderboard_manager.load_leaderboard()
    if not final_leaderboard:
        print("Error: Could not load leaderboard data.")
        logger.error("Backtest failed: cannot load leaderboard.")
        return

    # 3. Get Symbol/Timeframe IDs (Needed for pair generation if not cached)
    conn_ids = sqlite_manager.create_connection(str(db_path)); sym_id = -1; tf_id = -1
    if conn_ids:
        try:
            conn_ids.execute("BEGIN;")
            sym_id = sqlite_manager._get_or_create_id(conn_ids, 'symbols', 'symbol', symbol)
            tf_id = sqlite_manager._get_or_create_id(conn_ids, 'timeframes', 'timeframe', timeframe)
            conn_ids.commit()
        except Exception as id_err: logger.error(f"Backtest: Failed get sym/tf IDs: {id_err}")
        finally: conn_ids.close()
    else: logger.error("Backtest: Failed connect for sym/tf IDs.")
    if sym_id == -1 or tf_id == -1: print("\nError: Failed to get Symbol/Timeframe ID for backtest."); return


    # --- Backtesting Loop ---
    backtest_results = []
    indicator_series_cache: Dict[int, pd.DataFrame] = {} # Cache indicators calculated during backtest

    total_iterations = max_lag_backtest * num_backtest_points
    completed_iterations = 0

    print("Running backtest iterations...")
    for lag in range(1, max_lag_backtest + 1):
        # Find best predictor for this lag ONCE from the FINAL leaderboard
        predictor_key_pos = (lag, 'positive')
        predictor_key_neg = (lag, 'negative')
        best_predictor_info = None
        # Choose based on absolute correlation value from the final leaderboard
        corr_pos = final_leaderboard.get(predictor_key_pos, {}).get('correlation_value', -np.inf)
        corr_neg = final_leaderboard.get(predictor_key_neg, {}).get('correlation_value', np.inf)

        if abs(corr_pos) >= abs(corr_neg) and pd.notna(corr_pos):
            best_predictor_info = final_leaderboard.get(predictor_key_pos)
            if best_predictor_info: best_predictor_info['lag'] = lag # Add lag info
        elif pd.notna(corr_neg):
            best_predictor_info = final_leaderboard.get(predictor_key_neg)
            if best_predictor_info: best_predictor_info['lag'] = lag # Add lag info

        if not best_predictor_info or not best_predictor_info.get('config_id_source_db'):
            logger.warning(f"Backtest: No valid predictor found for Lag = {lag} in final leaderboard. Skipping lag.")
            completed_iterations += num_backtest_points # Increment progress even if skipped
            continue

        ind_name = best_predictor_info['indicator_name']
        cfg_id = best_predictor_info['config_id_source_db']
        try:
            # Attempt to parse params here. If fails, skip lag.
            params = json.loads(best_predictor_info['config_json'])
            indicator_config = {'indicator_name': ind_name, 'params': params, 'config_id': cfg_id}
        except json.JSONDecodeError:
            logger.error(f"Backtest: Failed to parse params for predictor CfgID {cfg_id} (Lag {lag}). Skipping lag.")
            completed_iterations += num_backtest_points
            continue
        except Exception as e:
             logger.error(f"Backtest: Error preparing predictor config for CfgID {cfg_id} (Lag {lag}): {e}. Skipping lag.")
             completed_iterations += num_backtest_points
             continue

        logger.info(f"Backtest Lag {lag}: Using Predictor CfgID {cfg_id} ('{ind_name}')")

        for i in range(num_backtest_points):
            # t = index for predictor calculation
            # target_idx = index for actual price verification
            t = len(full_historical_data) - 1 - lag - i
            target_idx = t + lag

            # Basic bounds check
            if t < 0 or target_idx >= len(full_historical_data):
                logger.warning(f"Backtest: Index out of bounds (t={t}, target={target_idx}). Stopping early for lag {lag}.")
                break # Stop testing this lag if we run out of data

            current_progress = (completed_iterations / total_iterations) * 100 if total_iterations > 0 else 0
            print(f" Progress: {current_progress:.1f}% (Lag {lag}, Point {i+1})", end='\r')

            try:
                # Data slices
                data_for_regression = full_historical_data.iloc[:t+1] # Data up to predictor time t
                actual_price = full_historical_data.iloc[target_idx]['close']
                actual_date = full_historical_data.iloc[target_idx]['date']
                predictor_date = full_historical_data.iloc[t]['date']

                # Calculate current indicator value at time t
                # We need the *full* series calculated on *all* data up to t
                # Use the main cache, but calculate if missing
                current_ind_val = None
                if cfg_id not in indicator_series_cache:
                    logger.debug(f"Backtest: Calculating full indicator series for CfgID {cfg_id} up to point {t}")
                    # Pass copy to avoid modification issues
                    indicator_df_full_hist = indicator_factory._compute_single_indicator(
                        full_historical_data.copy(), # Use full history to ensure stable calc
                        indicator_config
                    )
                    if indicator_df_full_hist is not None and not indicator_df_full_hist.empty:
                        indicator_series_cache[cfg_id] = indicator_df_full_hist
                    else:
                        logger.error(f"Backtest: Failed compute indicator Cfg {cfg_id} for point {t}. Skipping point.")
                        completed_iterations += 1
                        continue
                else:
                     logger.debug(f"Backtest: Using cached indicator series for CfgID {cfg_id}")

                indicator_df_cached = indicator_series_cache[cfg_id]
                potential_cols = [col for col in indicator_df_cached.columns if col.startswith(f"{ind_name}_{cfg_id}")]
                if not potential_cols:
                    logger.error(f"Backtest: No output col found for CfgID {cfg_id}. Skipping point {t}, lag {lag}.")
                    completed_iterations += 1
                    continue
                current_ind_col = potential_cols[0]
                # Get value at index t
                if t < len(indicator_df_cached):
                    current_ind_val = indicator_df_cached[current_ind_col].iloc[t]
                    if pd.isna(current_ind_val):
                         logger.warning(f"Backtest: Indicator value NaN at index {t} for CfgID {cfg_id}, Lag {lag}. Skipping point.")
                         completed_iterations += 1
                         continue
                else:
                    logger.error(f"Backtest: Index {t} out of bounds for indicator CfgID {cfg_id}. Skipping point.")
                    completed_iterations += 1
                    continue


                # Get historical pairs using data *up to time t*
                # Pass an empty dict to force recalc on slice (don't use main series cache here)
                hist_pairs = predictor._get_historical_indicator_price_pairs(
                    db_path, sym_id, tf_id, indicator_config, lag,
                    data_for_regression, # Use data only up to t
                    {} # Use a temporary empty cache for pair generation to force recalc on slice
                )
                if hist_pairs is None or len(hist_pairs) < predictor.MIN_REGRESSION_POINTS:
                    logger.warning(f"Backtest: Insufficient regression pairs ({len(hist_pairs) if hist_pairs is not None else 0}) at point {t}, lag {lag}. Min={predictor.MIN_REGRESSION_POINTS}. Skipping.")
                    completed_iterations += 1
                    continue

                # Perform regression
                reg_res = predictor._perform_prediction_regression(hist_pairs, current_ind_val, lag)
                if reg_res is None:
                    logger.warning(f"Backtest: Regression failed at point {t}, lag {lag}. Skipping.")
                    completed_iterations += 1
                    continue

                # Store result
                predicted_price = reg_res['predicted_value']
                error = predicted_price - actual_price
                pct_error = (error / actual_price) * 100 if actual_price != 0 else np.inf

                backtest_results.append({
                    'Lag': lag,
                    'Test Point Index (i)': i,
                    'Predictor Time (t)': predictor_date,
                    'Target Time (t+lag)': actual_date,
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
                logger.error(f"Backtest: Error during iteration (Lag {lag}, Point {i}): {iter_err}", exc_info=True)
            finally:
                completed_iterations += 1 # Ensure progress increments even on error within loop

    print("\nBacktest iterations complete.") # Final newline after loop

    # 4. Analyze and Report Results
    if not backtest_results:
        print("\nBacktest finished with no results.")
        logger.warning("Backtest completed but no results were generated.")
        return

    results_df = pd.DataFrame(backtest_results)

    # Calculate overall metrics
    mae = np.mean(np.abs(results_df['Error']))
    rmse = np.sqrt(np.mean(results_df['Error']**2))
    # Calculate MAPE carefully, excluding zero actual prices and infs
    valid_pct_err = results_df.loc[(results_df['Actual Price'] != 0) & np.isfinite(results_df['Percent Error']), 'Percent Error']
    mape = np.mean(np.abs(valid_pct_err)) if not valid_pct_err.empty else np.nan


    print("\n--- Backtest Overall Summary ---")
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
        MAPE=('Percent Error', lambda x: np.mean(np.abs(x[(x != 0) & np.isfinite(x)])) if not x[(x != 0) & np.isfinite(x)].empty else np.nan),
        Mean_R2=('Regression R2', 'mean')
    ).reset_index()


    print("\n--- Backtest Metrics Per Lag ---")
    metrics_per_lag['MAPE'] = metrics_per_lag['MAPE'].map('{:.2f}%'.format).replace('nan%', 'N/A')
    metrics_per_lag['MAE'] = metrics_per_lag['MAE'].map('{:.4f}'.format)
    metrics_per_lag['RMSE'] = metrics_per_lag['RMSE'].map('{:.4f}'.format)
    metrics_per_lag['Mean_R2'] = metrics_per_lag['Mean_R2'].map('{:.3f}'.format)
    with pd.option_context('display.width', 1000):
        print(metrics_per_lag.to_string(index=False))

    # Save detailed results to CSV
    output_filepath = config.REPORTS_DIR / f"{symbol}_{timeframe}_backtest_details_{max_lag_backtest}lags_{num_backtest_points}pts.csv"
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        results_df.sort_values(by=['Lag', 'Test Point Index (i)'], inplace=True)
        # Format dates for CSV readability
        results_df['Predictor Time (t)'] = pd.to_datetime(results_df['Predictor Time (t)']).dt.strftime('%Y-%m-%d %H:%M')
        results_df['Target Time (t+lag)'] = pd.to_datetime(results_df['Target Time (t+lag)']).dt.strftime('%Y-%m-%d %H:%M')
        results_df.to_csv(output_filepath, index=False, float_format='%.6f')
        print(f"\nDetailed backtest results saved to: {output_filepath}")
        logger.info(f"Backtest details saved to: {output_filepath}")
    except Exception as e:
        print(f"\nError saving detailed backtest results: {e}")
        logger.error(f"Failed to save backtest CSV: {e}", exc_info=True)

# Example of how to potentially call it (e.g., from main.py prompt)
# if __name__ == '__main__':
#     print("Backtester module should be run via main.py or called directly.")
#     # Example direct call (replace with actual paths/values)
#     # import logging_setup # Need logging setup if running standalone
#     # logging_setup.setup_logging()
#     # test_db_path = Path("C:/code/prediction/database/BTCUSDT_1d.db")
#     # test_symbol = "BTCUSDT"
#     # test_tf = "1d"
#     # test_max_lag = 7
#     # test_num_points = 20
#     # run_backtest(test_db_path, test_symbol, test_tf, test_max_lag, test_num_points)
