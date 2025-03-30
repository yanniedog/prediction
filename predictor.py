# predictor.py
import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta, timezone
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False; sm = None
    logging.getLogger(__name__).error("Predictor requires 'statsmodels'. Please install it (`pip install statsmodels`).")

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
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='historical_data';")
        if cursor.fetchone() is None: logger.error(f"'historical_data' table not found in {db_path}."); return None
        query = "SELECT * FROM historical_data ORDER BY open_time DESC LIMIT 1"
        df = pd.read_sql_query(query, conn)
        if df.empty: logger.warning("Latest data DF is empty."); return None
        df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
        df.dropna(subset=['open_time'], inplace=True)
        if df.empty: logger.warning("Latest data point invalid open_time."); return None
        df['date'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        logger.info(f"Latest data point loaded: {df.iloc[0]['date']} (Open: {df.iloc[0]['open_time']})")
        return df
    except Exception as e: logger.error(f"Error loading latest data: {e}", exc_info=True); return None
    finally:
        if conn: conn.close()

def _calculate_periods_to_date(latest_date: datetime, target_date_str: str, timeframe: str) -> Optional[int]:
    """Calculate number of periods between latest date and target date string."""
    try:
        # Attempt to parse with or without time, defaulting to start of day UTC
        try:
            target_date = datetime.strptime(target_date_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
        except ValueError:
            try:
                target_date = datetime.strptime(target_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            except ValueError:
                 logger.error(f"Invalid date format: '{target_date_str}'. Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS.")
                 return None

        if target_date <= latest_date:
            logger.error(f"Target date {target_date_str} not after latest data date {latest_date}.")
            return None
        periods = utils.calculate_periods_between_dates(latest_date, target_date, timeframe)
        if periods is None or periods <= 0:
             logger.error(f"Could not calculate valid positive periods for {timeframe} from {latest_date} to {target_date}.")
             return None
        logger.info(f"Calculated {periods} periods from {latest_date} to {target_date} for {timeframe}.")
        return periods
    except Exception as e: logger.error(f"Error calculating periods to date: {e}", exc_info=True); return None

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
        logger.debug(f"Using cached indicator series for Cfg {config_id}")
    else:
        logger.debug(f"Calculating '{indicator_name}' historical for pairs: {indicator_config}")
        # Pass a copy to ensure the original full_historical_data isn't modified by the factory
        indicator_df = indicator_factory._compute_single_indicator(full_historical_data.copy(), indicator_config)
        if indicator_df is None or indicator_df.empty:
            logger.error(f"Failed calc historical indicator Cfg {config_id}.")
            # Cache failure explicitly? Maybe not, let it retry next time.
            return None
        indicator_series_cache[config_id] = indicator_df # Cache the result
        logger.debug(f"Cached indicator series for Cfg {config_id}")

    potential_cols = [col for col in indicator_df.columns if col.startswith(f"{indicator_name}_{config_id}")]
    if not potential_cols: logger.error(f"Could not find output col like '{indicator_name}_{config_id}_...'"); return None
    indicator_col_name = potential_cols[0]
    if len(potential_cols) > 1: logger.warning(f"Multiple outputs for {indicator_name} Cfg {config_id}: {potential_cols}. Using first: '{indicator_col_name}'.")
    logger.debug(f"Using indicator column '{indicator_col_name}' for regression for lag {lag}.")

    # Align Indicator[t] and Close[t+lag] using the full historical data
    df_reg = pd.DataFrame(index=full_historical_data.index)
    # Reindex indicator result to match the main dataframe's index
    df_reg['Indicator_t'] = indicator_df[indicator_col_name].reindex(full_historical_data.index)
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
    # Check for constant predictor (can cause issues)
    if X.nunique() <= 1:
        logger.error(f"Cannot perform regression for Lag={current_lag}: Predictor variable '{X.name}' is constant.")
        return None

    X_sm = sm.add_constant(X, prepend=True, has_constant='raise') # Use 'raise' to catch issues explicitly
    try:
        model = sm.OLS(y, X_sm, missing='drop').fit() # Use missing='drop'
        if not hasattr(model, 'summary'): logger.error(f"Statsmodels OLS fit failed for Lag={current_lag}."); return None
        logger.debug(f"Regression Summary (Lag={current_lag}):\n" + str(model.summary()))

        # Predict
        # Ensure column names match exactly what sm.add_constant created
        x_pred_df = pd.DataFrame({'const': [1.0], X.name: [current_indicator_value]})[X_sm.columns]

        pred_res = model.get_prediction(x_pred_df); pred_summary = pred_res.summary_frame(alpha=0.05)
        predicted_value = pred_summary['mean'].iloc[0]
        ci_lower = pred_summary['mean_ci_lower'].iloc[0]; ci_upper = pred_summary['mean_ci_upper'].iloc[0]

        # Metrics
        r_squared = model.rsquared; adj_r_squared = model.rsquared_adj
        # Make sure to access the slope using the correct column name from X
        slope = model.params.get(X.name, np.nan)
        corr_from_r2 = np.sqrt(max(0, r_squared)) * np.sign(slope) if pd.notna(slope) else np.nan
        intercept = model.params.get('const', np.nan)

        logger.info(f"Regression Prediction (Lag={current_lag}): {predicted_value:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        logger.info(f"R2: {r_squared:.4f}, Adj R2: {adj_r_squared:.4f}, Implied Corr: {corr_from_r2:.4f}")

        return {
            "predicted_value": predicted_value, "ci_lower": ci_lower, "ci_upper": ci_upper,
            "r_squared": r_squared, "adj_r_squared": adj_r_squared, "correlation": corr_from_r2,
            "intercept": intercept, "slope": slope, "model_params": model.params.to_dict(),
            "model_pvalues": model.pvalues.to_dict(), "n_observations": model.nobs,
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
            'current_indicator_value', 'r_squared', 'correlation', 'slope', 'intercept'
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
            'intercept': 'Regression Intercept'
        }, inplace=True)

        # Formatting
        prec = 2 if abs(current_price) > 100 else 4
        for col in ['Predicted Price', 'CI Lower (95%)', 'CI Upper (95%)', 'Indicator Val @ Lag 0', 'Regression Intercept']:
            df_export[col] = df_export[col].map(f'{{:.{prec}f}}'.format).fillna('N/A')
        for col in ['Predictor LB Corr', 'Regression R2', 'Regression Corr', 'Regression Slope']:
            df_export[col] = df_export[col].map('{:.4f}'.format).fillna('N/A')
        df_export['Target Date (Est. UTC)'] = pd.to_datetime(df_export['Target Date (Est. UTC)']).dt.strftime('%Y-%m-%d %H:%M')

        # Generate Output String
        output_string = f"Prediction Details - {symbol} ({timeframe})\n"
        output_string += f"Generated: {current_timestamp_str}\n"
        output_string += f"Based on Latest Data: {latest_date.strftime('%Y-%m-%d %H:%M')} UTC (Close: {current_price:.{prec}f})\n"
        output_string += "=" * 120 + "\n"
        with pd.option_context('display.width', 1000, 'display.max_colwidth', 40): # Adjust width/colwidth as needed
            output_string += df_export.to_string(index=False, justify='left')
        output_string += "\n" + "=" * 120

        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        output_filepath.write_text(output_string, encoding='utf-8')
        logger.info(f"Successfully exported prediction details to {output_filepath}")
        print(f"Prediction details saved to: {output_filepath}")

    except Exception as e:
        logger.error(f"Error exporting prediction details: {e}", exc_info=True)
        print("\nError saving prediction details file.")


# --- Main Prediction Function ---
def predict_price(db_path: Path, symbol: str, timeframe: str, forecast_target: str) -> None:
    """Main prediction function orchestrator."""
    if not STATSMODELS_AVAILABLE: print("\nError: Prediction requires 'statsmodels'."); logger.error("Pred skipped: statsmodels missing."); return
    utils.clear_screen(); print(f"\n--- Price Prediction for {symbol} ({timeframe}) ---"); logger.info(f"Starting prediction: {symbol}/{timeframe}, Target: {forecast_target}")

    # 1. Get latest data
    latest_data_df = _get_latest_data_point(db_path)
    if latest_data_df is None or latest_data_df.empty: print("Error: Cannot load latest data."); return
    latest_data = latest_data_df.iloc[0]; current_price = latest_data['close']; latest_date = latest_data['date']
    print(f"Latest Data: {latest_date.strftime('%Y-%m-%d %H:%M')} UTC - Current Close: {current_price:.4f}")

    # 2. Determine FINAL target lag
    final_target_lag = None
    final_target_date_input_str = None # Store user date if provided
    if forecast_target.startswith('+'):
        try: final_target_lag = int(forecast_target[1:]); assert final_target_lag > 0; print(f"Forecasting up to {final_target_lag} periods ahead.")
        except (ValueError, AssertionError): print(f"Error: Invalid period format '{forecast_target}'. Use '+N'."); return
    else:
        final_target_date_input_str = forecast_target
        final_target_lag = _calculate_periods_to_date(latest_date, forecast_target, timeframe)
        if final_target_lag is None: print(f"Error: Cannot determine lag for date '{forecast_target}'. Check format (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS) and ensure it's after latest data."); return
        print(f"Forecasting to date {forecast_target} (Total Lag: {final_target_lag} periods).")
    if final_target_lag is None or final_target_lag <= 0: print("Error: Invalid target lag determined."); return

    # 3. Load full historical data ONCE
    full_historical_data = data_manager.load_data(db_path)
    if full_historical_data is None or full_historical_data.empty: print("Error: Failed load full historical data."); return
    # Ensure 'close' is numeric after loading
    if 'close' not in full_historical_data.columns or not pd.api.types.is_numeric_dtype(full_historical_data['close']):
        print("Error: 'close' column missing or not numeric in historical data.")
        return

    # 4. Get Symbol/Timeframe IDs
    conn_ids = sqlite_manager.create_connection(str(db_path)); sym_id = -1; tf_id = -1
    if conn_ids:
        try:
            conn_ids.execute("BEGIN;")
            sym_id = sqlite_manager._get_or_create_id(conn_ids, 'symbols', 'symbol', symbol)
            tf_id = sqlite_manager._get_or_create_id(conn_ids, 'timeframes', 'timeframe', timeframe)
            conn_ids.commit()
        except Exception as id_err:
            logger.error(f"Failed get sym/tf IDs: {id_err}")
            try:
                conn_ids.rollback()
                logger.warning("Rolled back transaction for sym/tf IDs due to error.")
            except Exception as rb_err:
                logger.error(f"Rollback failed after ID error: {rb_err}")
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

        # 5. Find best predictor for CURRENT lag
        logger.info(f"Querying leaderboard for best predictor: Lag {current_lag}")
        predictor_info = leaderboard_manager.find_best_predictor_for_lag(current_lag)
        if not predictor_info:
            logger.warning(f"No predictor found for Lag = {current_lag}. Skipping this lag."); skipped_lags+=1; continue
        ind_name = predictor_info['indicator_name']; params = predictor_info['params']; cfg_id = predictor_info['config_id_source_db']
        lb_corr = predictor_info['correlation_value']; lb_corr_type = predictor_info['correlation_type']
        logger.info(f" Predictor for Lag {current_lag}: {ind_name} (CfgID: {cfg_id}), Corr: {lb_corr:.4f}")

        # 6. Calculate current indicator value (using cached indicator if possible)
        indicator_config = {'indicator_name': ind_name, 'params': params, 'config_id': cfg_id}
        current_ind_val = None
        if cfg_id in indicator_series_cache:
            indicator_df_full = indicator_series_cache[cfg_id]
            logger.debug(f"Using cached indicator series for current value (Cfg {cfg_id})")
        else:
            logger.info(f"Calculating indicator series for current value {ind_name} (ID: {cfg_id})...")
            # Pass a copy to ensure the original full_historical_data isn't modified by the factory
            indicator_df_full = indicator_factory._compute_single_indicator(full_historical_data.copy(), indicator_config)
            if indicator_df_full is not None and not indicator_df_full.empty:
                indicator_series_cache[cfg_id] = indicator_df_full # Cache result
            else:
                logger.error(f"Failed compute indicator {ind_name} Cfg {cfg_id}."); skipped_lags+=1; continue

        # Extract the current value
        if indicator_df_full is None or indicator_df_full.empty:
             logger.error(f"Indicator computation failed Cfg {cfg_id}."); skipped_lags+=1; continue
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

        # 7. Get historical pairs (using cache)
        hist_pairs = _get_historical_indicator_price_pairs(db_path, sym_id, tf_id, indicator_config, current_lag, full_historical_data, indicator_series_cache)
        if hist_pairs is None:
             logger.warning(f"Could not get historical pairs for lag {current_lag}"); skipped_lags+=1; continue

        # 8. Perform regression
        reg_res = _perform_prediction_regression(hist_pairs, current_ind_val, current_lag)
        if reg_res is None:
            logger.warning(f"Regression failed for lag {current_lag}"); skipped_lags+=1; continue

        # 9. Estimate target date for this specific lag
        estimated_target_date = utils.estimate_future_date(latest_date, current_lag, timeframe)
        if not estimated_target_date:
            logger.warning(f"Could not estimate target date for lag {current_lag}. Using placeholder offset.")
            approx_days_inc = utils.estimate_days_in_periods(1, timeframe) or (1/24.0) # Estimate days for one period
            estimated_target_date = latest_date + timedelta(days=current_lag * approx_days_inc) # Crude fallback

        # 10. Store results
        prediction_results.append({
            "lag": current_lag,
            "target_date": estimated_target_date,
            "predictor_cfg_id": cfg_id,
            "predictor_name": ind_name,
            "predictor_params": params,
            "predictor_lb_corr": lb_corr,
            "current_indicator_value": current_ind_val,
            "predicted_value": reg_res['predicted_value'],
            "ci_lower": reg_res['ci_lower'],
            "ci_upper": reg_res['ci_upper'],
            "r_squared": reg_res['r_squared'],
            "correlation": reg_res['correlation'],
            "slope": reg_res['slope'],
            "intercept": reg_res['intercept']
        })
    # --- End Prediction Loop ---
    print() # Newline after loop finishes

    if skipped_lags > 0:
        print(f"Note: Skipped {skipped_lags} lags due to missing predictors or calculation errors.")

    if not prediction_results:
        print("\nError: No successful predictions were made for any lag.")
        return

    # 11. Export Prediction Details <--- MODIFIED CALL LOCATION
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


    # 12. Display Final Results Summary
    final_prediction = prediction_results[-1] # Get the result for the final target lag
    pred_p = final_prediction['predicted_value']; ci_l = final_prediction['ci_lower']; ci_u = final_prediction['ci_upper']
    r2 = final_prediction['r_squared']; reg_corr = final_prediction['correlation']; lb_corr_final = final_prediction['predictor_lb_corr']
    slope = final_prediction['slope']; intercept = final_prediction['intercept']
    final_cfg_id = final_prediction['predictor_cfg_id']; final_ind_name = final_prediction['predictor_name']
    final_target_dt_actual = final_prediction['target_date']

    prec = 2 if abs(current_price) > 100 else 4
    print("\n--- Final Prediction Summary ---")
    print(f"Target: {final_target_lag} periods ({final_target_dt_actual.strftime('%Y-%m-%d %H:%M') if final_target_dt_actual else 'N/A'} UTC)")
    print(f"Final Predictor: {final_ind_name} (CfgID: {final_cfg_id})")
    print(f"Predicted Price: {pred_p:.{prec}f}")
    print(f"95% Confidence Interval: [{ci_l:.{prec}f} - {ci_u:.{prec}f}]")
    print(f"Regression R2: {r2:.4f} (Final Lag)")
    print(f"Regression Corr: {reg_corr:.4f} (Leaderboard Corr: {lb_corr_final:.4f} for final lag predictor)")
    print(f"Model (Final Lag): Price[t+{final_target_lag}] = {slope:.4f} * Ind[t] + {intercept:.{prec}f}")

    # 13. Calculate Yield
    pct_chg = ((pred_p - current_price) / current_price) * 100 if current_price != 0 else 0
    print(f"\nExpected Gain/Loss vs Current: {pct_chg:.2f}%")
    approx_days = utils.estimate_days_in_periods(final_target_lag, timeframe)
    if approx_days is not None and approx_days > 0.1:
        daily_yield = np.clip(pct_chg / approx_days, -100.0, 100.0)
        print(f"Approx Daily Yield: {daily_yield:.3f}% (over ~{approx_days:.1f} days)")
    else: print("Cannot estimate meaningful daily yield.")

    # 14. Generate Plot with path
    try:
        # Use the same prefix generated for the export function
        plot_prefix = export_prefix
        plot_dates = [latest_date] + [res['target_date'] for res in prediction_results]
        plot_prices = [current_price] + [res['predicted_value'] for res in prediction_results]
        plot_ci_lower = [current_price] + [res['ci_lower'] for res in prediction_results] # Use current price as CI bounds at t=0
        plot_ci_upper = [current_price] + [res['ci_upper'] for res in prediction_results]

        plot_predicted_path(
            plot_dates, plot_prices, plot_ci_lower, plot_ci_upper,
            timeframe, symbol, plot_prefix, final_target_lag
        )
    except Exception as plot_err: logger.error(f"Failed generate prediction plot: {plot_err}", exc_info=True); print("\nWarning: Could not generate plot.")


# --- Plotting Function (Modified) ---
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
        logger.error("Mismatched data lengths for plotting.")
        return

    start_date = dates[0]
    start_price = prices[0]
    target_date = dates[-1] # Final target date

    # Ensure dates are timezone-aware for plotting
    aware_dates = []
    for d in dates:
        if d.tzinfo is None:
            aware_dates.append(d.replace(tzinfo=timezone.utc))
        else:
            aware_dates.append(d)

    plot_dpi = config.DEFAULTS.get("plot_dpi", 300)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=plot_dpi) # Adjusted size slightly

    # Plot predicted path and CI bands
    ax.plot(aware_dates, prices, marker='.', linestyle='-', markersize=4, color='blue', label='Predicted Price Path')
    ax.fill_between(aware_dates, ci_lower, ci_upper, color='skyblue', alpha=0.4, interpolate=True, label='95% CI Band')

    # Plot start and end points prominently
    ax.plot(aware_dates[0], prices[0], marker='o', markersize=8, color='black', label=f'Start ({start_date.strftime("%Y-%m-%d %H:%M")})')
    ax.plot(aware_dates[-1], prices[-1], marker='*', markersize=10, color='red', label=f'Final Prediction ({target_date.strftime("%Y-%m-%d %H:%M")})')

    # Annotations (optional, can be added for start/end)
    prec = 2 if abs(start_price) > 100 else 4
    ax.text(aware_dates[-1], prices[-1], f' Lag {final_lag}\n ${prices[-1]:.{prec}f}', va='bottom', ha='left', fontsize=9, color='red')
    ax.text(aware_dates[0], prices[0], f' Start\n ${prices[0]:.{prec}f}', va='bottom', ha='right', fontsize=9, color='black')

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
    min_y = min(min(ci_lower), start_price)
    max_y = max(max(ci_upper), start_price)
    y_range = max_y - min_y
    ax.set_ylim(min_y - y_range * 0.1, max_y + y_range * 0.1)


    fig.tight_layout()
    output_filepath = config.REPORTS_DIR / f"{file_prefix}_plot.png"
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    try: fig.savefig(output_filepath); logger.info(f"Saved prediction plot: {output_filepath}"); print(f"\nPrediction plot saved to: {output_filepath}")
    except Exception as e: logger.error(f"Failed save plot {output_filepath.name}: {e}", exc_info=True)
    finally: plt.close(fig)


if __name__ == '__main__':
    print("Predictor module. Run via main.py.")
