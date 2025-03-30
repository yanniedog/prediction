# main.py
import logging
import logging_setup # Import the setup module

# >> Logging setup moved inside the main execution block <<

import sys
import json
from datetime import datetime, timezone
import pandas as pd
import os
import shutil
from pathlib import Path
import numpy as np
import re
import math

# Import project modules
import utils
import config # Need config early for log path
import sqlite_manager
import data_manager
import parameter_generator
import parameter_optimizer
import indicator_factory
import correlation_calculator # Make sure this is imported
import visualization_generator
import custom_indicators # Keep import for checking definitions
import leaderboard_manager
import predictor
import backtester # Added import

# >> Logger initialization moved inside main execution block <<

# --- Cleanup Function ---
def cleanup_previous_content():
    """Deletes specified previous analysis output files and directories, INCLUDING the leaderboard."""
    # --- Logger reference needed here ---
    # Get logger instance assuming it might have been re-initialized
    logger = logging.getLogger(__name__)
    # ---------------------------------
    logger.info("Starting cleanup of previous outputs (INCLUDING leaderboard)...")
    deleted_count = 0; error_count = 0
    locations_to_clean = [
        (config.DB_DIR, "*.db"),
        config.LEADERBOARD_DB_PATH, # Add leaderboard DB path directly
        (config.LOG_DIR, "*.log"), # Keep log dir, remove old .log files
        config.REPORTS_DIR, # Remove entire reports dir and its contents
    ]
    current_log_path_abs = (config.LOG_DIR / "logfile.txt").resolve()

    for item in locations_to_clean:
        try:
            if isinstance(item, tuple): # Handle (directory, pattern)
                target_dir, pattern = item
                if target_dir.exists() and target_dir.is_dir():
                    logger.debug(f"Cleaning '{pattern}' in '{target_dir}'...")
                    for file_path in target_dir.glob(pattern):
                        file_path_abs = file_path.resolve()
                        if file_path_abs == current_log_path_abs:
                             logger.debug(f"Skipping deletion of active log file: {file_path}")
                             continue
                        if file_path.is_file():
                            try: file_path.unlink(); logger.info(f"Deleted: {file_path}"); deleted_count += 1
                            except OSError as e: logger.error(f"Error deleting file {file_path}: {e}"); error_count += 1
            elif isinstance(item, Path): # Handle directory or specific file Path objects
                target_path = item
                if target_path.is_file():
                    # Specific handling for the leaderboard file
                    if target_path.resolve() == config.LEADERBOARD_DB_PATH.resolve():
                         logger.debug(f"Deleting leaderboard DB file: '{target_path}'...")
                         try: target_path.unlink(); logger.info(f"Deleted leaderboard DB: {target_path}"); deleted_count += 1
                         except OSError as e: logger.error(f"Error deleting leaderboard DB {target_path}: {e}"); error_count += 1
                    else:
                        logger.warning(f"Skipping deletion of unexpected specific file: {target_path}") # Only handle leaderboard or dirs

                elif target_path.is_dir(): # Handle directories (like REPORTS_DIR)
                     # Protect project root, log dir, and leaderboard parent
                     protected_dirs = [config.PROJECT_ROOT.resolve(), config.LOG_DIR.resolve(), config.LEADERBOARD_DB_PATH.parent.resolve()]
                     if target_path.resolve() not in protected_dirs:
                        if target_path.exists(): # Check again after is_dir()
                            logger.debug(f"Removing directory tree: '{target_path}'...")
                            try: shutil.rmtree(target_path); logger.info(f"Deleted dir tree: {target_path}"); deleted_count += 1
                            except OSError as e: logger.error(f"Error deleting dir {target_path}: {e}"); error_count += 1
                        else: logger.debug(f"Directory doesn't exist, skipping: {target_path}")
                     else: logger.warning(f"Skipping deletion of protected directory: {target_path}")
                else: # Doesn't exist or is not a file/dir
                    logger.debug(f"Item {target_path} not found or not a file/dir. Skipping.")
            else: logger.debug(f"Skipping unknown cleanup item type: {item}")
        except Exception as e: logger.error(f"Error during cleanup processing item {item}: {e}"); error_count += 1

    # Recreate essential directories if they were deleted
    config.DB_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.LEADERBOARD_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


    logger.info(f"Cleanup finished. Deleted items/trees: {deleted_count}. Errors: {error_count}.")
    if error_count > 0: print("WARNING: Errors occurred during cleanup. Check logs.")


# --- Main Analysis Orchestration ---
def run_analysis():
    """Main orchestration function for the analysis pipeline."""
    # --- Logger reference needed here ---
    logger = logging.getLogger(__name__)
    # ---------------------------------
    utils.clear_screen()
    start_time = datetime.now(timezone.utc)
    timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
    logger.info(f"--- Starting Analysis Run: {timestamp_str} ---")

    # Initialize Leaderboard DB (might be deleted below)
    logger.info("Initializing leaderboard database (pre-cleanup)...")
    if not leaderboard_manager.initialize_leaderboard_db():
        logger.error("Failed initialize leaderboard DB. Features will be affected.")
    else: logger.info("Leaderboard database initialized (pre-cleanup check).")

    # --- Optional Cleanup ---
    try:
        # --- MODIFIED PROMPT ---
        cleanup_choice = input("Delete previous analysis content (DBs, reports, leaderboard)? [Y/n]: ").strip().lower() or 'y'
        if cleanup_choice == 'y':
            print("Proceeding with cleanup (INCLUDING leaderboard)...")
            # --- END MODIFIED PROMPT ---
            # --- Close logging handlers BEFORE cleanup ---
            handlers = logging.getLogger().handlers[:]
            for handler in handlers: handler.close(); logging.getLogger().removeHandler(handler)
            logging.shutdown()
            # ------------------------------------------
            cleanup_previous_content() # This now includes leaderboard
            # --- Re-initialize logging AFTER cleanup ---
            logging_setup.setup_logging() # This will open logfile.txt in 'w' mode
            logger = logging.getLogger(__name__) # Re-get logger after setup
            logger.info(f"Re-initialized logging after cleanup. Continuing run: {timestamp_str}")
            # ------------------------------------------
            # Leaderboard DB *must* be re-initialized after cleanup if it was deleted
            logger.info("Re-initializing leaderboard database after cleanup...")
            if not leaderboard_manager.initialize_leaderboard_db():
                 logger.error("CRITICAL: Failed re-init leaderboard DB after cleanup. Exiting.")
                 sys.exit(1) # Exit if leaderboard cannot be recreated
            else: logger.info("Leaderboard database successfully re-initialized after cleanup.")
        else: print("Skipping cleanup."); logger.info("Skipping cleanup.")
    except Exception as e: logger.error(f"Error during cleanup prompt/exec: {e}", exc_info=True); print("Cleanup error, continuing..."); sys.exit(1)


    # --- Mode Selection: Analysis vs Custom vs Backtest ---
    while True:
        print("\n--- Mode Selection ---")
        print("[A]nalysis: Run full analysis (Download/Indicators/Correlations/Reports/Predict).")
        print("[C]ustom:   Generate reports/predictions from existing database.")
        print("[B]acktest: Run simplified backtest (requires final leaderboard).")
        mode_choice = input("Select mode [A/c/b]: ").strip().lower() or 'a'

        if mode_choice == 'c':
            logger.info("--- Entering Custom Mode (Report/Predict from Existing DB) ---")
            print("\n--- Custom Mode ---")
            print("Select the database containing the pre-calculated correlations.")
            db_path_custom = data_manager.select_existing_database()

            if not db_path_custom:
                print("No database selected. Exiting Custom Mode.")
                continue # Go back to Mode Selection

            logger.info(f"Custom Mode using DB: {db_path_custom.name}")

            # Extract symbol/timeframe
            try:
                base_name = db_path_custom.stem
                parts = base_name.split('_', 1)
                if len(parts) == 2:
                    symbol_custom, timeframe_custom = parts
                    symbol_custom = symbol_custom.upper()
                    logger.info(f"Extracted context: Symbol={symbol_custom}, Timeframe={timeframe_custom}")
                else:
                    raise ValueError("Invalid filename format")
            except Exception as e:
                logger.error(f"Cannot parse symbol/timeframe from DB name '{db_path_custom.name}': {e}")
                print("Error: Invalid database filename format. Cannot proceed.")
                continue # Go back to Mode Selection

            # Get necessary info from the selected DB
            conn_custom = None
            max_lag_custom = None
            indicator_configs_processed_custom = []
            correlations_by_config_id_custom = {}
            custom_data_ok = False

            try:
                conn_custom = sqlite_manager.create_connection(str(db_path_custom))
                if not conn_custom: raise ConnectionError("Failed to connect to selected DB.")

                cursor = conn_custom.cursor()
                cursor.execute("SELECT id FROM symbols WHERE LOWER(symbol) = ?", (symbol_custom.lower(),))
                res_sym = cursor.fetchone()
                symbol_id_custom = res_sym[0] if res_sym else None
                cursor.execute("SELECT id FROM timeframes WHERE LOWER(timeframe) = ?", (timeframe_custom.lower(),))
                res_tf = cursor.fetchone()
                timeframe_id_custom = res_tf[0] if res_tf else None

                if symbol_id_custom is None or timeframe_id_custom is None:
                     raise ValueError(f"Symbol '{symbol_custom}' or Timeframe '{timeframe_custom}' not found in the selected DB's metadata tables.")

                max_lag_custom = sqlite_manager.get_max_lag_for_pair(conn_custom, symbol_id_custom, timeframe_id_custom)
                if max_lag_custom is None or max_lag_custom <= 0:
                    raise ValueError("Could not determine max lag from correlations in the DB.")

                config_ids_custom = sqlite_manager.get_distinct_config_ids_for_pair(conn_custom, symbol_id_custom, timeframe_id_custom)
                if not config_ids_custom:
                    raise ValueError("No indicator configurations with correlation data found in the DB.")

                indicator_configs_processed_custom = sqlite_manager.get_indicator_configs_by_ids(conn_custom, config_ids_custom)
                if not indicator_configs_processed_custom:
                    raise ValueError("Could not retrieve indicator configuration details from the DB.")

                correlations_by_config_id_custom = sqlite_manager.fetch_correlations(conn_custom, symbol_id_custom, timeframe_id_custom, config_ids_custom)
                if not correlations_by_config_id_custom:
                    raise ValueError("Could not fetch correlation data from the DB.")

                valid_corrs_found = False
                max_lag_fetched = 0
                for cfg_id, corrs in correlations_by_config_id_custom.items():
                    if corrs and isinstance(corrs, list):
                         max_lag_fetched = max(max_lag_fetched, len(corrs))
                         if len(corrs) >= max_lag_custom:
                             valid_corrs_found = True
                if max_lag_fetched < max_lag_custom:
                     logger.warning(f"Max lag found in fetched data ({max_lag_fetched}) is less than initially determined ({max_lag_custom}). Adjusting.")
                     max_lag_custom = max_lag_fetched
                if max_lag_custom <= 0:
                    raise ValueError("No valid correlation data found for lag 1 or higher.")

                custom_data_ok = True

            except Exception as e:
                logger.error(f"Error preparing data for Custom Mode: {e}", exc_info=True)
                print(f"\nError: Could not load necessary data from '{db_path_custom.name}'.")
                print("Ensure this DB has been successfully processed by the Analysis path previously.")
            finally:
                if conn_custom: conn_custom.close()

            if not custom_data_ok:
                print("Cannot proceed with Custom Mode.")
                continue

            # --- Custom Mode Action Selection ---
            while True:
                print("\n--- Custom Mode Actions ---")
                print(f"Using Data: {symbol_custom} ({timeframe_custom}) | Max Lag: {max_lag_custom} | Configs: {len(indicator_configs_processed_custom)}")
                print("[P]redict price path using leaderboard")
                print("[V]isualize correlations (Heatmap, Combined, Lines, etc.)")
                print("[B]oth Predict and Visualize")
                print("[Q]uit Custom Mode")
                custom_action = input("Select action [P/v/b/q]: ").strip().lower() or 'b'

                run_predict = custom_action in ['p', 'b']
                run_visualize = custom_action in ['v', 'b']

                if custom_action == 'q':
                    logger.info("Exiting Custom Mode.")
                    break

                if not run_predict and not run_visualize:
                    print("Invalid selection.")
                    continue

                # --- Execute Prediction (if chosen) ---
                if run_predict:
                    print("\n--- Custom Prediction ---")
                    print(f"Predicting path up to determined max lag: {max_lag_custom}")
                    try:
                        # Temporarily set console to WARNING for prediction process
                        logging_setup.set_console_log_level(logging.WARNING)
                        predictor.predict_price(db_path_custom, symbol_custom, timeframe_custom, max_lag_custom)
                    except Exception as pred_err:
                        logger.error(f"Custom Mode prediction error: {pred_err}", exc_info=True)
                        print("\nPrediction encountered an error. Check logs.")
                    finally:
                        logging_setup.reset_console_log_level() # Reset console level

                # --- Execute Visualization (if chosen) ---
                if run_visualize:
                    print("\n--- Custom Visualization ---")
                    viz_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                    file_prefix_custom = f"{viz_timestamp}_{symbol_custom}_{timeframe_custom}_CUSTOM"
                    logger.info(f"Generating visualizations with prefix: {file_prefix_custom}")

                    try:
                        vis_success = 0; vis_total = 4
                        num_configs = len(indicator_configs_processed_custom)
                        if not indicator_configs_processed_custom:
                             logger.warning("No configs for viz."); vis_total = 0
                        else:
                            is_tweak_path_custom = False # Assume False for custom mode viz
                            configs_for_viz_custom = indicator_configs_processed_custom
                            corrs_for_viz_custom = correlations_by_config_id_custom
                            configs_limited_custom = configs_for_viz_custom
                            corrs_limited_custom = corrs_for_viz_custom
                            limit_viz = config.DEFAULTS["heatmap_max_configs"]

                            if len(configs_for_viz_custom) > limit_viz:
                                logger.info(f"Custom Viz: Config count ({len(configs_for_viz_custom)}) exceeds limit ({limit_viz}). Limiting some plots.") # Changed to INFO
                                perf_data_custom = []
                                for cfg in configs_for_viz_custom:
                                    cfg_id = cfg.get('config_id')
                                    if cfg_id is None: continue
                                    corrs = corrs_for_viz_custom.get(cfg_id, [])
                                    if corrs and len(corrs) >= max_lag_custom:
                                        try:
                                            corr_arr = np.array(corrs[:max_lag_custom], dtype=float)
                                            if not np.isnan(corr_arr).all(): peak_abs = np.nanmax(np.abs(corr_arr)); perf_data_custom.append({'config_id': cfg_id, 'peak_abs': peak_abs})
                                        except Exception as peak_err: logger.warning(f"Error calc peak abs for {cfg_id} in custom viz: {peak_err}")
                                if perf_data_custom:
                                    perf_data_custom.sort(key=lambda x: x.get('peak_abs', 0), reverse=True)
                                    top_ids_custom = {item['config_id'] for item in perf_data_custom[:limit_viz]}
                                    configs_limited_custom = [cfg for cfg in configs_for_viz_custom if cfg.get('config_id') in top_ids_custom]
                                    corrs_limited_custom = {cfg_id: corrs for cfg_id, corrs in corrs_for_viz_custom.items() if cfg_id in top_ids_custom}
                                    logger.info(f"Custom Viz: Limited viz set contains {len(configs_limited_custom)} configs.")
                                else: logger.error("Custom Viz: Could not filter configs for viz. Using full set.")

                            num_limited = len(configs_limited_custom)
                            if not configs_limited_custom:
                                logger.warning("No configs left after limiting for custom viz plots."); vis_total -= 3
                            else:
                                logger.info(f"Generating line charts ({num_limited} configs)..."); visualization_generator.plot_correlation_lines(corrs_limited_custom, configs_limited_custom, max_lag_custom, config.LINE_CHARTS_DIR, file_prefix_custom); vis_success += 1
                                logger.info(f"Generating combined chart ({num_limited} configs)..."); visualization_generator.generate_combined_correlation_chart(corrs_limited_custom, configs_limited_custom, max_lag_custom, config.COMBINED_CHARTS_DIR, file_prefix_custom); vis_success += 1
                                logger.info(f"Generating heatmap ({num_limited} configs)..."); visualization_generator.generate_enhanced_heatmap(corrs_limited_custom, configs_limited_custom, max_lag_custom, config.HEATMAPS_DIR, file_prefix_custom, is_tweak_path_custom); vis_success += 1

                            if not corrs_for_viz_custom:
                                logger.warning("No configs for Custom Envelope chart."); vis_total -= 1
                            else:
                                logger.info(f"Generating envelope chart ({len(configs_for_viz_custom)} configs)..."); visualization_generator.generate_correlation_envelope_chart(corrs_for_viz_custom, configs_for_viz_custom, max_lag_custom, config.REPORTS_DIR, file_prefix_custom, is_tweak_path_custom); vis_success += 1

                        logger.info(f"Custom Viz generation finished ({vis_success}/{vis_total} types attempted).")
                        print(f"Visualizations saved to '{config.REPORTS_DIR}' subfolders.")
                    except Exception as vis_err:
                        logger.error(f"Error during custom viz generation: {vis_err}", exc_info=True)
                        print("\nError generating visualizations. Check logs.")

                # After completing action(s), exit custom mode
                logger.info("Finished custom action(s). Exiting.")
                sys.exit(0)

            continue # Go back to Custom Mode Action Selection if Q wasn't chosen

        elif mode_choice == 'b':
            logger.info("--- Entering Backtest Mode ---")
            print("\n--- Backtest Mode ---")
            print("Select the database containing historical price data for backtesting.")
            db_path_bt = data_manager.select_existing_database()

            if not db_path_bt:
                print("No database selected. Exiting Backtest Mode.")
                continue # Go back to Mode Selection

            logger.info(f"Backtest Mode using DB: {db_path_bt.name}")

            # Extract symbol/timeframe
            try:
                base_name = db_path_bt.stem
                parts = base_name.split('_', 1)
                if len(parts) == 2:
                    symbol_bt, timeframe_bt = parts
                    symbol_bt = symbol_bt.upper()
                    logger.info(f"Extracted context: Symbol={symbol_bt}, Timeframe={timeframe_bt}")
                else:
                    raise ValueError("Invalid filename format")
            except Exception as e:
                logger.error(f"Cannot parse symbol/timeframe from DB name '{db_path_bt.name}': {e}")
                print("Error: Invalid database filename format. Cannot proceed.")
                continue

            # Validate data
            if not data_manager.validate_data(db_path_bt):
                print(f"Selected database '{db_path_bt.name}' is empty or invalid. Cannot backtest.")
                continue

            # Get Backtest Parameters
            max_lag_bt = 0
            while max_lag_bt <= 0:
                try:
                    prompt = f"Enter max lag to backtest (e.g., 7): "
                    lag_input = input(prompt).strip()
                    max_lag_bt = int(lag_input) if lag_input else config.DEFAULTS.get("max_lag", 7)
                    if max_lag_bt <= 0: print("Max lag must be positive.")
                except ValueError: print("Invalid input. Please enter an integer.")
            logger.info(f"Backtest max lag set to {max_lag_bt}")

            num_points_bt = 0
            while num_points_bt <= 0:
                try:
                    prompt = f"Enter number of historical points per lag to test (e.g., 50): "
                    points_input = input(prompt).strip()
                    num_points_bt = int(points_input) if points_input else 50
                    if num_points_bt <= 0: print("Number of points must be positive.")
                except ValueError: print("Invalid input. Please enter an integer.")
            logger.info(f"Backtest num points set to {num_points_bt}")

            # Execute Backtest
            print("\nStarting backtest...")
            try:
                # Temporarily set console to INFO to see backtest progress/summary
                logging_setup.set_console_log_level(logging.INFO)
                backtester.run_backtest(
                    db_path=db_path_bt,
                    symbol=symbol_bt,
                    timeframe=timeframe_bt,
                    max_lag_backtest=max_lag_bt,
                    num_backtest_points=num_points_bt
                )
            except Exception as bt_err:
                logger.error(f"Backtest error: {bt_err}", exc_info=True)
                print("\nBacktest encountered an error. Check logs.")
            finally:
                logging_setup.reset_console_log_level() # Reset console level

            logger.info("Finished backtest action. Exiting.")
            sys.exit(0) # Exit after backtest

        elif mode_choice == 'a':
             logger.info("--- Entering Analysis Mode ---")
             break
        else:
            print("Invalid mode selection.")

    # --- Analysis Mode Continues ---
    db_path = None; symbol = None; timeframe = None
    if mode_choice == 'a':
        data_source_info = data_manager.manage_data_source()
        if data_source_info is None: logger.info("Exiting: No data source selected."); sys.exit(0)
        db_path, symbol, timeframe = data_source_info
        logger.info(f"Using data source: {db_path.name} ({symbol}, {timeframe})")

    if db_path is None or symbol is None or timeframe is None:
        logger.critical("Data source not defined. Exiting.")
        sys.exit(1)

    try:
        indicator_factory._load_indicator_definitions()
        indicator_definitions = indicator_factory._INDICATOR_DEFS
        if not indicator_definitions: raise ValueError("Indicator definitions empty.")
        logger.info(f"Accessed {len(indicator_definitions)} indicator definitions.")
    except Exception as e: logger.critical(f"Failed load indicator definitions: {e}", exc_info=True); sys.exit(1)

    logger.info("--- Loading initial data for date range calculation ---")
    data = data_manager.load_data(db_path)
    if data is None or data.empty:
        logger.error(f"Failed load data from {db_path}. Exiting."); sys.exit(1)

    logger.info(f"--- Initial data loaded. Shape: {data.shape}. ---")
    if 'date' not in data.columns:
        logger.error("CRITICAL: 'date' column MISSING after initial load_data! Cannot proceed.")
        sys.exit(1)

    data_daterange_str = "Unknown"
    try:
        min_date_dt = data['date'].min(); max_date_dt = data['date'].max()
        logger.info(f"--- Data Date Range (Raw): {min_date_dt} to {max_date_dt} ---")
        if pd.isna(min_date_dt) or pd.isna(max_date_dt):
            logger.error("NaN found in min/max date during initial calc.")
            data_daterange_str = "Invalid Dates"
        else:
            if min_date_dt.tzinfo is None: min_date_dt = min_date_dt.tz_localize('UTC')
            if max_date_dt.tzinfo is None: max_date_dt = max_date_dt.tz_localize('UTC')
            data_daterange_str = f"{min_date_dt.strftime('%Y%m%d')}-{max_date_dt.strftime('%Y%m%d')}"
            logger.info(f"--- Calculated data_daterange_str = {data_daterange_str} ---")
            if min_date_dt < data_manager.EARLIEST_VALID_DATE:
                 logger.error(f"--- Calculated min_date {min_date_dt} is before {data_manager.EARLIEST_VALID_DATE}! Check data source integrity. ---")
                 sys.exit(f"Exiting due to suspected invalid date range in source data (Min: {min_date_dt}).")
    except Exception as e:
        logger.error(f"Error determining date range: {e}", exc_info=True)
        data_daterange_str = "Error"

    if data_daterange_str in ["Unknown", "Invalid Dates", "Error", "Load Failed"]:
        logger.critical(f"Could not determine a valid data date range ({data_daterange_str}). Exiting analysis.")
        sys.exit(1)
    logger.info(f"Final data_daterange_str to be used: {data_daterange_str}")

    min_points_needed = config.DEFAULTS["min_data_points_for_lag"]
    estimated_nan_rows = min(100, int(len(data) * 0.05))
    effective_data_len = max(0, len(data) - estimated_nan_rows)
    max_possible_lag = max(0, effective_data_len - min_points_needed - 1)

    if max_possible_lag <= 0:
        logger.critical(f"Insufficient data rows ({len(data)}, estimated effective {effective_data_len}) for lag calculation. Need > {min_points_needed + 1 + estimated_nan_rows}.");
        sys.exit(1)

    default_lag_config = config.DEFAULTS["max_lag"]
    reasonable_max_suggested = min(max_possible_lag, max(30, int(effective_data_len * 0.1)), 500)
    suggested_lag = min(reasonable_max_suggested, default_lag_config)

    max_lag = 0
    while max_lag <= 0:
        try:
            prompt = (f"\nEnter max correlation lag ({timeframe}).\n"
                      f"  - Config Default: {default_lag_config}\n"
                      f"  - Suggested Max (capped): {suggested_lag}\n"
                      f"  - Absolute Max Possible (estimated): {max_possible_lag}\n"
                      f"Enter lag (e.g., {suggested_lag}): ")
            lag_input = input(prompt).strip()
            max_lag = suggested_lag if not lag_input else int(lag_input)

            if max_lag <= 0: print("Lag must be positive."); max_lag = 0
            elif max_lag > max_possible_lag:
                print(f"WARNING: Lag {max_lag} exceeds estimated max possible ({max_possible_lag}).")
                if input(f"This may fail. Continue anyway? [y/N]: ").strip().lower() != 'y': max_lag = 0
                else: print(f"Proceeding with potentially excessive lag: {max_lag}"); break
            elif max_lag > reasonable_max_suggested:
                print(f"WARNING: Lag {max_lag} is higher than the suggested cap ({reasonable_max_suggested}).")
                print(f"Using a very high lag can significantly increase computation time and may lead to errors.")
                if input(f"Continue with lag {max_lag}? [y/N]: ").strip().lower() != 'y': max_lag = 0
                else: print(f"Proceeding with high lag: {max_lag}"); break
            else: print(f"Using lag: {max_lag}"); break
        except ValueError: print("Invalid input. Please enter an integer.")
        except Exception as e: logger.error(f"Error getting max_lag: {e}", exc_info=True); print("An input error occurred. Exiting."); sys.exit(1)

    logger.info(f"Using final max_lag = {max_lag}.")
    final_min_required_len = max_lag + min_points_needed
    if effective_data_len < final_min_required_len:
        logger.critical(f"Estimated effective data length ({effective_data_len}) insufficient for chosen max_lag={max_lag}. Need {final_min_required_len}. Exiting.");
        sys.exit(1)

    # --- Analysis State ---
    conn = None; indicator_configs_to_process: List[Dict] = []; is_tweak_path = False
    selected_indicator_for_tweak = None; analysis_path_successful = False
    best_config_per_lag_result = {}; all_indicators_optimized_results = {}; all_evaluated_configs_aggregated = []

    try:
        conn = sqlite_manager.create_connection(str(db_path)); assert conn is not None
        conn.execute("BEGIN;"); symbol_id = sqlite_manager._get_or_create_id(conn, 'symbols', 'symbol', symbol)
        timeframe_id = sqlite_manager._get_or_create_id(conn, 'timeframes', 'timeframe', timeframe); conn.commit()
        logger.debug(f"Retrieved DB IDs - Symbol: {symbol_id}, Timeframe: {timeframe_id}")
        conn.close(); conn = None

        # --- Analysis Path Selection ---
        while not analysis_path_successful:
            print("\n--- Analysis Path ---\n[D]efault: Use default parameters.\n[T]weak:   Optimize parameters (Bayesian).")
            choice = input("Select path [D/t]: ").strip().lower() or 'd'

            if choice == 'd':
                logger.info("Processing Default Path...")
                is_tweak_path = False; temp_config_list = []
                conn = sqlite_manager.create_connection(str(db_path)); assert conn is not None
                try:
                    for name, definition in indicator_definitions.items():
                        params_def = definition.get('parameters', {})
                        actual_defaults = {k: v.get('default') for k, v in params_def.items() if 'default' in v}
                        conditions = definition.get('conditions', [])
                        if parameter_generator.evaluate_conditions(actual_defaults, conditions):
                            try:
                                config_id = sqlite_manager.get_or_create_indicator_config_id(conn, name, actual_defaults)
                                temp_config_list.append({'indicator_name': name, 'params': actual_defaults, 'config_id': config_id})
                            except Exception as cfg_err: logger.error(f"Failed get/create config ID for default '{name}': {cfg_err}")
                        else: logger.warning(f"Default params invalid for '{name}'. Skipping.")
                finally: conn.close(); conn = None
                estimated_outputs = 1.5; num_configs = len(temp_config_list)
                actual_total_corrs = num_configs * estimated_outputs * max_lag
                limit = config.DEFAULTS["target_max_correlations"]
                logger.info(f"Default path: {num_configs} configs. Est. correlations: ~{int(actual_total_corrs)} (Limit: {limit})")
                if actual_total_corrs > limit * 1.1:
                     if input(f"WARNING: High estimate ({int(actual_total_corrs)} > {limit}). Proceed? [y/N]: ").strip().lower() != 'y':
                         logger.info("User aborted default path."); continue
                indicator_configs_to_process = temp_config_list; analysis_path_successful = True; break

            elif choice == 't':
                if not parameter_optimizer.SKOPT_AVAILABLE:
                     logger.error("skopt required for Tweak path."); print("\nError: scikit-optimize not installed."); continue
                is_tweak_path = True
                available_indicators = sorted([ name for name, definition in indicator_definitions.items()
                    if any( ('min' in d and 'max' in d) or isinstance(d.get('default'), (int, float)) for p, d in definition.get('parameters', {}).items()) ])
                if not available_indicators: logger.error("No indicators with tunable params."); print("\nError: No tunable indicators found."); continue

                while True: # Tweak sub-choice loop
                    print("\n--- Bayesian Optimization Options ---\n[S]ingle Indicator\n[A]ll Available (default)\n[E]xclude Indicators\n[B]ack")
                    tweak_choice = input("Select mode [A/s/e/b]: ").strip().lower() or 'a'
                    indicators_to_optimize = []

                    if tweak_choice == 's':
                        print("\nAvailable Indicators to Optimize:"); [print(f"{i+1}. {n}") for i, n in enumerate(available_indicators)]
                        while True:
                             try: idx = int(input(f"Select number (1-{len(available_indicators)}): ").strip()) - 1
                             except ValueError: print("Invalid input."); continue
                             if 0 <= idx < len(available_indicators): selected_indicator_for_tweak = available_indicators[idx]; indicators_to_optimize = [selected_indicator_for_tweak]; logger.info(f"Optimizing SINGLE: '{selected_indicator_for_tweak}'"); break
                             else: print("Invalid selection.")
                        break
                    elif tweak_choice == 'a': indicators_to_optimize = available_indicators; logger.info(f"Optimizing ALL {len(indicators_to_optimize)} indicators."); break
                    elif tweak_choice == 'e':
                        print("\nAvailable Indicators:"); [print(f"{i+1}. {n}") for i, n in enumerate(available_indicators)]
                        exclude_input = input("Enter numbers (comma-separated) to EXCLUDE: ").strip()
                        try: excluded_indices = {int(x.strip()) - 1 for x in exclude_input.split(',') if x.strip().isdigit()} if exclude_input else set()
                        except ValueError: print("Invalid format."); continue
                        indicators_to_optimize = [ind for i, ind in enumerate(available_indicators) if i not in excluded_indices and 0 <= i < len(available_indicators)]
                        excluded_names = {available_indicators[i] for i in excluded_indices if 0 <= i < len(available_indicators)}
                        logger.info(f"Excluding {len(excluded_names)}: {excluded_names}. Optimizing {len(indicators_to_optimize)} remaining.");
                        if not indicators_to_optimize: print("No indicators left."); continue
                        break
                    elif tweak_choice == 'b': break
                    else: print("Invalid mode.")
                if tweak_choice == 'b': continue

                # --- Execute Sequential Bayesian Optimization ---
                any_opt_failed = False
                all_evaluated_configs_aggregated = []
                all_indicators_optimized_results = {}
                best_config_per_lag_result = {}
                opt_n_calls = config.DEFAULTS["optimizer_n_calls"]; opt_n_initial = config.DEFAULTS["optimizer_n_initial_points"]

                print(f"\nStarting Bayesian Opt for {len(indicators_to_optimize)} indicator(s)...")
                # ** Temporarily reduce console logging for optimization loop **
                logging_setup.set_console_log_level(logging.WARNING)
                try:
                    for i, ind_name_opt in enumerate(indicators_to_optimize):
                        print(f"\n[{i+1}/{len(indicators_to_optimize)}] Optimizing: {ind_name_opt}...")
                        # --- LOG LEVEL CHANGE ---
                        logger.info(f"--- Starting Bayesian Opt for: {ind_name_opt} ({i+1}/{len(indicators_to_optimize)}) ---") # Changed to INFO
                        # --- END LOG LEVEL CHANGE ---
                        definition = indicator_definitions.get(ind_name_opt)
                        if not definition: logger.error(f"Definition missing for {ind_name_opt}. Skipping."); any_opt_failed = True; continue
                        required_cols = definition.get('required_inputs', []) + ['close']
                        if not all(col in data.columns for col in required_cols):
                            missing_c = [c for c in required_cols if c not in data.columns]
                            logger.error(f"Data missing cols {missing_c} for {ind_name_opt}. Skipping."); any_opt_failed = True; continue
                        print(f"(Evals={opt_n_calls}/lag, Init={opt_n_initial}, MaxLag={max_lag}, Acq={config.DEFAULTS['optimizer_acq_func']})")

                        # Clear caches *before each indicator* in the Tweak path
                        parameter_optimizer.indicator_series_cache.clear()
                        parameter_optimizer.single_correlation_cache.clear()
                        # --- LOG LEVEL CHANGE ---
                        logger.info(f"Cleared optimizer caches before optimizing {ind_name_opt}.") # Changed to INFO
                        # --- END LOG LEVEL CHANGE ---

                        try:
                            best_result_for_log, evaluated_configs_for_indicator = parameter_optimizer.optimize_parameters_bayesian_per_lag(
                                indicator_name=ind_name_opt, indicator_def=definition, base_data_with_required=data.copy(), max_lag=max_lag,
                                n_calls_per_lag=opt_n_calls, n_initial_points_per_lag=opt_n_initial,
                                db_path=str(db_path), symbol_id=symbol_id, timeframe_id=timeframe_id,
                                symbol=symbol, timeframe=timeframe, data_daterange=data_daterange_str, source_db_name=db_path.name
                                )
                            if not evaluated_configs_for_indicator:
                                logger.error(f"Optimization FAILED for {ind_name_opt}. No results returned.");
                                print(f"Opt FAILED for {ind_name_opt}.");
                                any_opt_failed = True;
                                continue

                            # --- LOG LEVEL CHANGE ---
                            logger.info(f"Opt complete for {ind_name_opt}. Adding {len(evaluated_configs_for_indicator)} unique configs.") # Changed to INFO
                            # --- END LOG LEVEL CHANGE ---
                            all_evaluated_configs_aggregated.extend(evaluated_configs_for_indicator)

                            all_indicators_optimized_results[ind_name_opt] = best_result_for_log
                            if len(indicators_to_optimize) == 1: best_config_per_lag_result = best_result_for_log

                        except Exception as opt_err:
                            logger.error(f"Optimization loop failed for '{ind_name_opt}': {opt_err}", exc_info=True);
                            any_opt_failed = True;
                            print(f"ERROR optimizing {ind_name_opt}. Continuing...")
                finally:
                    # ** Reset console logging level **
                    logging_setup.reset_console_log_level()

                if not all_evaluated_configs_aggregated:
                    logger.error("Tweak path finished, but NO configs evaluated across all indicators.");
                    print("\nOpt process completed, no results generated.");
                    continue

                seen_final_ids = set(); unique_final_configs = []
                for cfg in all_evaluated_configs_aggregated:
                    cfg_id = cfg.get('config_id')
                    if cfg_id is not None and cfg_id not in seen_final_ids:
                        unique_final_configs.append(cfg); seen_final_ids.add(cfg_id)
                    elif cfg_id is None: logger.warning(f"Found evaluated config with missing ID: {cfg}")
                indicator_configs_to_process = unique_final_configs
                logger.info(f"Tweak path complete. Using {len(indicator_configs_to_process)} unique configs from optimization for final processing.")
                analysis_path_successful = True; break

            else: print("Invalid choice."); logger.warning(f"Invalid path choice: '{choice}'")

    except Exception as prep_err: logger.critical(f"Error during analysis path prep: {prep_err}", exc_info=True); sys.exit(1)
    finally:
        if conn: conn.close()

    if not analysis_path_successful or not indicator_configs_to_process: logger.error("No configurations prepared. Exiting."); sys.exit(1)
    logger.info(f"Analysis path prep complete. Processing {len(indicator_configs_to_process)} final configurations.")

    # --- FIX: Add Default Custom Indicators If Missing ---
    conn_temp = sqlite_manager.create_connection(str(db_path))
    if conn_temp:
        try:
            all_custom_defaults = []
            for name, definition in indicator_definitions.items():
                if definition.get('type') == 'custom':
                    params_def = definition.get('parameters', {})
                    actual_defaults = {k: v.get('default') for k, v in params_def.items() if 'default' in v}
                    conditions = definition.get('conditions', [])
                    if parameter_generator.evaluate_conditions(actual_defaults, conditions):
                        try:
                            cfg_id = sqlite_manager.get_or_create_indicator_config_id(conn_temp, name, actual_defaults)
                            all_custom_defaults.append({'indicator_name': name, 'params': actual_defaults, 'config_id': cfg_id})
                        except Exception as e: logger.error(f"Error getting ID for custom default {name}: {e}")

            existing_cfg_ids = {cfg['config_id'] for cfg in indicator_configs_to_process if cfg.get('config_id') is not None}
            added_custom_defaults = 0
            for cfg in all_custom_defaults:
                if cfg.get('config_id') not in existing_cfg_ids:
                    indicator_configs_to_process.append(cfg)
                    existing_cfg_ids.add(cfg['config_id'])
                    added_custom_defaults += 1
            if added_custom_defaults > 0:
                logger.info(f"Added {added_custom_defaults} missing default custom indicator configs to processing list.")
                logger.info(f"Total configurations to process now: {len(indicator_configs_to_process)}")

        except Exception as custom_add_err:
            logger.error(f"Error ensuring custom default indicators are added: {custom_add_err}", exc_info=True)
        finally:
            conn_temp.close()
    else:
        logger.error("Cannot connect to DB to check/add custom default indicators.")
    # --- END FIX ---

    # --- Log Path Summary ---
    if is_tweak_path:
        if len(all_indicators_optimized_results) > 1:
             logger.info(f"Tweak path (Bayesian Sequential) used across {len(all_indicators_optimized_results)} indicator(s).")
             print("\n--- Best Config Per Lag (Summary Across Indicators) ---")
             for ind_name, results in all_indicators_optimized_results.items(): print(f"  - {ind_name}: Found best for {sum(1 for r in results.values() if r)}/{max_lag} lags.")
             print("-" * 60)
        elif selected_indicator_for_tweak:
            logger.info(f"Tweak path (Bayesian Single) used for: '{selected_indicator_for_tweak}'.")
            print("\n--- Best Config Per Lag (Bayesian Opt - Single Indicator) ---")
            found_any = False
            for lag in sorted(best_config_per_lag_result.keys()):
                 result = best_config_per_lag_result[lag]
                 if result: print(f"Lag {lag:>3}: Corr={result.get('correlation_at_lag', np.nan):.4f}, ID={result.get('config_id','N/A')}, Params={result.get('params',{})}"); found_any = True
                 else: print(f"Lag {lag:>3}: No suitable config found.")
            if not found_any: print("  (No best configs found)")
            print("-" * 60)

    # --- Indicator Calculation & Correlation Processing (Unified Logic) ---
    logger.info("--- Starting Final Indicator Calculation & Correlation ---")
    data_with_indicators = indicator_factory.compute_configured_indicators(data, indicator_configs_to_process)

    # --- FIX: Removed redundant apply_all_custom_indicators call ---
    # Custom defaults are now added to indicator_configs_to_process list above
    # if needed, and will be computed by compute_configured_indicators.
    # custom_configs_present = any(indicator_definitions.get(cfg.get('indicator_name', ''), {}).get('type') == 'custom' for cfg in indicator_configs_to_process)
    # if custom_configs_present:
    #      logger.info("Ensuring default custom indicators are applied for consistency if present...")
    #      data_with_indicators = custom_indicators.apply_all_custom_indicators(data_with_indicators)
    # --- END FIX ---

    logger.info("Dropping NaNs for correlation calculation...")
    data_final = data_with_indicators.dropna(how='any')
    logger.info(f"Shape for Correlation (NaNs dropped): {data_final.shape}")

    min_required_corr = max_lag + config.DEFAULTS["min_data_points_for_lag"]
    if len(data_final) < min_required_corr:
        logger.error(f"Insufficient rows ({len(data_final)}) after dropna for Correlation. Need {min_required_corr}. Exiting."); sys.exit(1)

    final_columns_in_df = set(data_final.columns); final_configs_for_corr = []; processed_stems = set()
    skipped_configs_count = 0
    for cfg in indicator_configs_to_process:
         name, cfg_id = cfg.get('indicator_name'), cfg.get('config_id')
         if name is None or cfg_id is None: skipped_configs_count+=1; continue
         stem = f"{name}_{cfg_id}"; found = False
         for col in final_columns_in_df:
              # Use a slightly more robust check to avoid partial matches
              # e.g., ensure it ends with the ID or ID_suffix
              if re.match(rf"^{re.escape(stem)}(_.*)?$", col):
                  found = True; break
         if found:
             if stem not in processed_stems:
                 final_configs_for_corr.append(cfg); processed_stems.add(stem)
         else:
             # ----- CHANGE WARNING TO INFO -----
             logger.info(f"Config {name} (ID: {cfg_id}) columns not found in final data after NaN drop. Excluding.")
             # ----- END CHANGE -----
             skipped_configs_count+=1

    if not final_configs_for_corr: logger.error("No valid configs remaining for correlation after NaN drop. Exiting."); sys.exit(1)
    # ----- CHANGE WARNING TO INFO -----
    if skipped_configs_count > 0: logger.info(f"Skipped {skipped_configs_count} configs due to missing columns post-NaN drop.")
    # ----- END CHANGE -----

    # --- Final PARALLEL Correlation Calculation ---
    logger.info(f"Proceeding with PARALLEL correlation calculation for {len(final_configs_for_corr)} configs...")
    corr_success = False
    # ** Temporarily reduce console logging for correlation calculation **
    logging_setup.set_console_log_level(logging.WARNING)
    try:
        corr_success = correlation_calculator.process_correlations(
            data_final, str(db_path), symbol_id, timeframe_id, final_configs_for_corr, max_lag
        )
    finally:
        # ** Reset console logging level **
        logging_setup.reset_console_log_level()

    if not corr_success:
        logger.error("Parallel correlation calculation failed. Exiting."); sys.exit(1)
    logger.info("Parallel correlation calculation complete.")

    # --- Fetch Final Correlations for Reporting ---
    conn = sqlite_manager.create_connection(str(db_path)); report_data_ok = False
    correlations_by_config_id: Dict[int, List[Optional[float]]] = {}
    if not conn: logger.error("Failed connect DB for fetching final correlations."); sys.exit(1)
    try:
        config_ids_to_fetch = [cfg['config_id'] for cfg in final_configs_for_corr if 'config_id' in cfg]
        if not config_ids_to_fetch: raise ValueError("No config IDs to fetch correlations.")
        logger.info(f"Fetching final correlations for {len(config_ids_to_fetch)} IDs (up to lag {max_lag})...")
        correlations_by_config_id = sqlite_manager.fetch_correlations(conn, symbol_id, timeframe_id, config_ids_to_fetch)
        if not correlations_by_config_id: logger.error("No correlation data structure returned.")
        else:
             actual_max_lag_in_data = 0; configs_with_data = 0
             for cfg_id, data_list in correlations_by_config_id.items():
                  if data_list and isinstance(data_list, list):
                      configs_with_data += 1; valid_indices = [i for i, v in enumerate(data_list) if pd.notna(v)]
                      if valid_indices: actual_max_lag_in_data = max(actual_max_lag_in_data, max(valid_indices) + 1)
             logger.info(f"Found correlation data for {configs_with_data} configs. Max lag found: {actual_max_lag_in_data}")
             reporting_max_lag = min(max_lag, actual_max_lag_in_data) if actual_max_lag_in_data > 0 else 0
             if reporting_max_lag <= 0: logger.error("No valid correlation data found for lag 1+. Reporting cannot proceed.")
             elif reporting_max_lag < max_lag:
                 logger.warning(f"Reporting max lag adjusted from {max_lag} to {reporting_max_lag} based on available data.")
                 max_lag = reporting_max_lag
             else: logger.info(f"Using effective max_lag = {max_lag} for reporting.")
             configs_valid_at_lag = sum(1 for cfg_id, dlist in correlations_by_config_id.items() if dlist and len(dlist) >= max_lag and any(pd.notna(x) for x in dlist[:max_lag]))
             if configs_valid_at_lag > 0: logger.info(f"{configs_valid_at_lag} configs have data up to lag {max_lag}."); report_data_ok = True
             else: logger.error(f"No configs have data up to adjusted lag {max_lag}.")
             # ----- CHANGE WARNING TO INFO -----
             if 0 < configs_valid_at_lag < len(correlations_by_config_id): logger.info(f"Retrieved valid data for only {configs_valid_at_lag}/{len(correlations_by_config_id)} requested configs.")
             # ----- END CHANGE -----
    except Exception as fetch_err: logger.error(f"Error fetching final correlations: {fetch_err}", exc_info=True)
    finally:
        if conn: conn.close()

    if not report_data_ok: logger.error("Cannot generate reports - no valid correlation data."); sys.exit(1)

    # --- Leaderboard Update Call (Batch Mode) ---
    logger.info("Running final leaderboard comparison and potential update (Batch Mode)...")
    try:
        leaderboard_manager.update_leaderboard(
            correlations_by_config_id, final_configs_for_corr,
            max_lag, symbol, timeframe, data_daterange_str, db_path.name )
    except Exception as lb_err: logger.error(f"Failed final leaderboard update/comparison: {lb_err}", exc_info=True)

    # --- Reporting & Visualization ---
    file_prefix = f"{timestamp_str}_{symbol}_{timeframe}"
    if is_tweak_path:
         suffix = ""
         if len(all_indicators_optimized_results) > 1: suffix = f"_TWEAK-BAYES_Seq_{len(all_indicators_optimized_results)}Inds"
         elif selected_indicator_for_tweak: safe_name = re.sub(r'[\\/*?:"<>|\s]+', '_', selected_indicator_for_tweak); suffix = f"_TWEAK-BAYES_Single_{safe_name}"
         else: suffix = "_TWEAK-BAYES_Unknown"
         file_prefix += suffix
    else: file_prefix += "_DEFAULT"

    try: logger.info("Generating peak correlation report..."); visualization_generator.generate_peak_correlation_report(correlations_by_config_id, final_configs_for_corr, max_lag, config.REPORTS_DIR, file_prefix)
    except Exception as report_err: logger.error(f"Error generating peak report: {report_err}", exc_info=True)

    configs_for_viz = final_configs_for_corr; corrs_for_viz = correlations_by_config_id
    configs_limited = configs_for_viz; corrs_limited = corrs_for_viz
    limit = config.DEFAULTS["heatmap_max_configs"]
    if len(configs_for_viz) > limit:
        # ----- CHANGE WARNING TO INFO -----
        logger.info(f"Config count ({len(configs_for_viz)}) exceeds limit ({limit}). Limiting some plots.")
        # ----- END CHANGE -----
        perf_data = []
        for cfg in configs_for_viz:
            cfg_id = cfg.get('config_id')
            if cfg_id is None: continue
            corrs = corrs_for_viz.get(cfg_id, [])
            if corrs and len(corrs) >= max_lag:
                 try:
                     corr_arr = np.array(corrs[:max_lag], dtype=float)
                     if not np.isnan(corr_arr).all(): peak_abs = np.nanmax(np.abs(corr_arr)); perf_data.append({'config_id': cfg_id, 'peak_abs': peak_abs})
                 except Exception as peak_err: logger.warning(f"Error calc peak abs for {cfg_id}: {peak_err}")
        if perf_data:
             perf_data.sort(key=lambda x: x.get('peak_abs', 0), reverse=True)
             top_ids = {item['config_id'] for item in perf_data[:limit]}
             configs_limited = [cfg for cfg in configs_for_viz if cfg.get('config_id') in top_ids]
             corrs_limited = {cfg_id: corrs for cfg_id, corrs in corrs_for_viz.items() if cfg_id in top_ids}
             logger.info(f"Limited viz set contains {len(configs_limited)} configs.")
        else: logger.error("Could not filter configs for viz. Using full set.")

    logger.info("Starting visualization generation...")
    try:
        vis_success = 0; vis_total = 4
        num_limited = len(configs_limited)
        if not configs_limited: logger.warning("No configs for limited plots."); vis_total -= 3
        else:
            logger.info(f"Generating line charts ({num_limited} configs)..."); visualization_generator.plot_correlation_lines(corrs_limited, configs_limited, max_lag, config.LINE_CHARTS_DIR, file_prefix); vis_success += 1
            logger.info(f"Generating combined chart ({num_limited} configs)..."); visualization_generator.generate_combined_correlation_chart(corrs_limited, configs_limited, max_lag, config.COMBINED_CHARTS_DIR, file_prefix); vis_success += 1
            logger.info(f"Generating heatmap ({num_limited} configs)..."); visualization_generator.generate_enhanced_heatmap(corrs_limited, configs_limited, max_lag, config.HEATMAPS_DIR, file_prefix, is_tweak_path); vis_success += 1
        if not corrs_for_viz: logger.warning("No configs for Envelope chart."); vis_total -= 1
        else:
            logger.info(f"Generating envelope chart ({len(configs_for_viz)} configs)..."); visualization_generator.generate_correlation_envelope_chart(corrs_for_viz, configs_for_viz, max_lag, config.REPORTS_DIR, file_prefix, is_tweak_path); vis_success += 1
        logger.info(f"Viz generation finished ({vis_success}/{vis_total} types attempted).")
    except Exception as vis_err: logger.error(f"Error during viz generation: {vis_err}", exc_info=True)

    # --- Final Summary ---
    end_time = datetime.now(timezone.utc); duration = end_time - start_time
    logger.info(f"--- Analysis Run Completed: {timestamp_str} ---")
    logger.info(f"Total execution time: {duration}")
    print(f"\nAnalysis complete (lag={max_lag}). Reports saved in '{config.REPORTS_DIR}'. Total time: {duration}")

    # --- Export Final Leaderboard (Text File) ---
    try:
        logger.info("Exporting final leaderboard state (text file)...");
        if not leaderboard_manager.export_leaderboard_to_text(): logger.error("Final leaderboard text export failed.")
    except Exception as ex_err: logger.error(f"Failed export final leaderboard text: {ex_err}", exc_info=True)

    # --- Generate Leading Indicator Report ---
    try:
        logger.info("Generating leading indicator tally report...")
        if not leaderboard_manager.generate_leading_indicator_report():
             logger.error("Failed to generate leading indicator report.")
    except Exception as lead_ind_err:
        logger.error(f"Failed to generate leading indicator report: {lead_ind_err}", exc_info=True)

    # --- AUTOMATIC Prediction Execution ---
    print("\n--- Automatic Price Prediction ---")
    logger.info(f"Running automatic prediction for {symbol} ({timeframe}) up to lag {max_lag}...")
    try:
        # Temporarily set console to WARNING for prediction process
        logging_setup.set_console_log_level(logging.WARNING)
        predictor.predict_price(db_path, symbol, timeframe, max_lag)
    except Exception as pred_err:
        logger.error(f"Automatic prediction error: {pred_err}", exc_info=True)
        print("\nAutomatic prediction encountered an error. Check logs.")
    finally:
        logging_setup.reset_console_log_level() # Reset console level


# --- Main Execution Block ---
if __name__ == "__main__":

    # --- Delete Existing Log File ---
    log_file_path = config.LOG_DIR / "logfile.txt"
    try:
        if log_file_path.exists():
            log_file_path.unlink()
            print(f"Deleted previous log file: {log_file_path}")
        else:
            print("No previous log file to delete.")
        # Ensure log directory exists after potential deletion attempt
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as log_del_err:
        print(f"WARNING: Could not delete previous log file '{log_file_path}': {log_del_err}")
    # --------------------------------

    # --- FIX: Initialize Logging AFTER potential deletion ---
    logging_setup.setup_logging()
    logger = logging.getLogger(__name__) # Get logger instance after setup
    # --------------------------

    try:
        run_analysis()
        logger.info("--- Analysis run finished normally. ---")
        sys.exit(0)
    except SystemExit as e:
         exit_code = e.code if isinstance(e.code, int) else 1
         log_msg = "Analysis finished successfully" if exit_code == 0 else f"Analysis exited prematurely (code {exit_code})"
         # Use logger safely, even if it might have been closed during cleanup
         try: logger.log(logging.INFO if exit_code == 0 else logging.ERROR, f"--- {log_msg}. ---")
         except Exception: print(f"--- {log_msg}. (Logger unavailable) ---")
         sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("--- Analysis interrupted by user. ---")
        print("\nAnalysis interrupted.")
        sys.exit(130)
    except Exception as e:
        try:
            logger.critical(f"--- Unhandled exception in main: {e} ---", exc_info=True)
            print(f"\nCRITICAL ERROR: {e}\nCheck log file ('{log_file_path}') for details.")
        except Exception as log_err:
            print(f"\nCRITICAL ERROR: {e}")
            print(f"Logging failed: {log_err}")
        sys.exit(1)