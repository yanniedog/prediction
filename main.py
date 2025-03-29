# main.py
import logging
import logging_setup

logging_setup.setup_logging() # Initialize logging first

import sys
import json
from datetime import datetime, timezone # Import timezone
import pandas as pd
import os
import shutil
from pathlib import Path
import numpy as np
import re
import math

# Import project modules
import utils
import config
import sqlite_manager
import data_manager
import parameter_generator
import parameter_optimizer # Import the updated optimizer
import indicator_factory
import correlation_calculator
import visualization_generator
import custom_indicators
import leaderboard_manager

logger = logging.getLogger(__name__)

# --- cleanup_previous_content unchanged ---
def cleanup_previous_content():
    """Deletes specified previous output files and directories."""
    logger.info("Starting cleanup of previous outputs...")
    deleted_count = 0; error_count = 0
    locations_to_clean = [
        (config.DB_DIR, "*.db"),
        (config.LOG_DIR, "*.log"),
        (config.LOG_DIR, "*.txt"),
        (config.REPORTS_DIR, "*.csv"),
        (config.REPORTS_DIR, "*.png"),
        config.HEATMAPS_DIR,
        config.LINE_CHARTS_DIR,
        config.COMBINED_CHARTS_DIR,
    ]
    for item in locations_to_clean:
        try:
            if isinstance(item, tuple):
                target_dir, pattern = item
                if target_dir.exists() and target_dir.is_dir():
                    logger.debug(f"Cleaning '{pattern}' in '{target_dir}'...")
                    for file_path in target_dir.glob(pattern):
                        if file_path.is_file():
                            try: file_path.unlink(); logger.info(f"Deleted: {file_path}"); deleted_count += 1
                            except OSError as e: logger.error(f"Error deleting {file_path}: {e}"); error_count += 1
            elif isinstance(item, Path):
                target_dir = item
                # Prevent deletion of root or leaderboard DB parent dir
                if target_dir != config.PROJECT_ROOT and target_dir != config.LEADERBOARD_DB_PATH.parent:
                    if target_dir.exists() and target_dir.is_dir():
                        logger.debug(f"Removing dir tree: '{target_dir}'...")
                        try: shutil.rmtree(target_dir); logger.info(f"Deleted dir tree: {target_dir}"); deleted_count += 1
                        except OSError as e: logger.error(f"Error deleting dir {target_dir}: {e}"); error_count += 1
                    else: logger.debug(f"Directory does not exist, skipping delete: {target_dir}")
                else: logger.debug(f"Skipping deletion of protected directory: {target_dir}")
            else: logger.debug(f"Skipping cleanup item: {item}")
        except Exception as e: logger.error(f"Error during cleanup: {e}"); error_count += 1
    logger.info(f"Cleanup finished. Deleted items/trees: {deleted_count}. Errors: {error_count}.")
    if error_count > 0: print("WARNING: Errors occurred during cleanup. Check logs.")


def run_analysis():
    """Main orchestration function for the analysis pipeline."""
    utils.clear_screen()

    start_time = datetime.now(timezone.utc); timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
    logger.info(f"--- Starting Analysis Run: {timestamp_str} ---")

    logger.info("Initializing leaderboard database...")
    if not leaderboard_manager.initialize_leaderboard_db():
        logger.error("Failed to initialize leaderboard database. Leaderboard updates will be skipped.")
    else: logger.info("Leaderboard database initialized successfully.")

    try:
        cleanup_choice = input("Delete previous analysis content (DBs, logs, charts - EXCLUDING Leaderboard)? [Y/n]: ").strip().lower() or 'y'
        logger.info(f"User choice for cleanup: '{cleanup_choice}'")
        if cleanup_choice == 'y':
            print("Proceeding with cleanup (excluding leaderboard)...")
            logging.shutdown()
            cleanup_previous_content()
            logging_setup.setup_logging() # Re-initialize after cleanup
            logger.info(f"Re-initialized logging after cleanup. Continuing run: {timestamp_str}")
            # Re-check leaderboard DB init as cleanup might have affected the directory
            logger.info("Re-checking leaderboard database initialization...")
            if not leaderboard_manager.initialize_leaderboard_db():
                 logger.error("Failed to initialize leaderboard database after cleanup. Updates will be skipped.")
            else: logger.info("Leaderboard database ok after cleanup.")
        else:
            print("Skipping cleanup.")
            logger.info("Skipping cleanup.")
    except Exception as e:
        logger.error(f"Error during cleanup prompt: {e}", exc_info=True)
        print("Error during cleanup, continuing...")

    # Ensure DB directory exists before initializing placeholder
    config.DB_DIR.mkdir(parents=True, exist_ok=True)
    placeholder_db_path = str(config.DB_DIR / "init_placeholder.db")
    if not sqlite_manager.initialize_database(placeholder_db_path):
        logger.warning("Placeholder DB init check failed.")
        # Don't exit here, maybe user selects existing DB

    data_source_info = data_manager.manage_data_source()
    if data_source_info is None:
        logger.info("Exiting: No data source selected.")
        sys.exit(0)

    db_path, symbol, timeframe = data_source_info
    logger.info(f"Using data source: {db_path.name} (Symbol: {symbol}, Timeframe: {timeframe})")

    # Load indicator definitions
    indicator_definitions = None
    try:
        indicator_factory._load_indicator_definitions()
        indicator_definitions = indicator_factory._INDICATOR_DEFS
        if not indicator_definitions: raise ValueError("Indicator definitions empty.")
        logger.info(f"Accessed {len(indicator_definitions)} indicator definitions.")
    except Exception as e:
        logger.critical(f"Failed load indicator definitions: {e}", exc_info=True)
        sys.exit(1)

    # Load data
    data = data_manager.load_data(db_path)
    if data is None or data.empty:
        logger.error(f"Failed load/empty data from {db_path}. Exiting.")
        sys.exit(1)
    logger.info(f"Initial data loaded. Shape: {data.shape}. Checking length...")

    time_interval = utils.determine_time_interval(data) or "periods"
    logger.info(f"Determined time interval: {time_interval}")
    data_daterange_str = "Unknown"
    try:
        min_date = data['date'].min().strftime('%Y%m%d')
        max_date = data['date'].max().strftime('%Y%m%d')
        data_daterange_str = f"{min_date}-{max_date}"
        logger.info(f"Dataset date range: {data_daterange_str}")
    except Exception as e:
        logger.warning(f"Could not determine dataset date range: {e}")

    # Determine max lag based on data length
    max_possible_lag = len(data) - config.MIN_DATA_POINTS_FOR_LAG - 1
    if max_possible_lag <= 0:
        logger.critical(f"Insufficient data ({len(data)} rows). Need > {config.MIN_DATA_POINTS_FOR_LAG + 1} rows.")
        sys.exit(1)

    suggested_lag = min(max_possible_lag, config.DEFAULT_MAX_LAG)
    max_lag = 0
    while max_lag <= 0:
        try:
            prompt = (f"\nEnter max correlation lag ({time_interval}).\n"
                      f"  - Default: {config.DEFAULT_MAX_LAG}\n"
                      f"  - Suggested: {suggested_lag}\n"
                      f"  - Max possible: {max_possible_lag}\n"
                      f"Enter lag (e.g., {suggested_lag}): ")
            lag_input = input(prompt).strip()
            if not lag_input:
                max_lag = suggested_lag
                print(f"Using suggested lag: {max_lag}")
            else:
                max_lag = int(lag_input)

            if max_lag <= 0:
                print("Max lag must be positive.")
                max_lag = 0 # Reset to loop again
            elif max_lag > max_possible_lag:
                print(f"Input lag {max_lag} exceeds max possible ({max_possible_lag}). Using max possible.")
                max_lag = max_possible_lag
                break # Exit loop with adjusted max lag
            else:
                print(f"Using user-defined lag: {max_lag}")
                break # Exit loop with valid user lag
        except ValueError:
            print("Invalid input. Please enter a positive integer.")
        except Exception as e:
            logger.error(f"Error getting max_lag input: {e}", exc_info=True)
            print("An error occurred. Exiting.")
            sys.exit(1)

    logger.info(f"Using final max_lag = {max_lag} for calculations.")
    min_required_len_for_lag = max_lag + config.MIN_DATA_POINTS_FOR_LAG
    if len(data) < min_required_len_for_lag:
        logger.critical(f"Insufficient rows ({len(data)}). Need {min_required_len_for_lag} for max_lag={max_lag}. Exiting.")
        sys.exit(1)
    logger.info(f"Data length ({len(data)}) sufficient for chosen max_lag={max_lag} (required: {min_required_len_for_lag}).")

    conn = None
    # indicator_configs_to_process is now populated AFTER optimization in Tweak path
    indicator_configs_to_process = [] # Holds the list used for final processing/reporting
    is_tweak_path = False
    selected_indicator_for_tweak = None
    analysis_path_successful = False
    best_config_per_lag_result = {} # Stores result from per-lag optimizer

    try:
        # Get symbol and timeframe IDs
        conn = sqlite_manager.create_connection(str(db_path))
        if not conn:
            logger.critical("Failed connect main DB pre-config.")
            sys.exit(1)
        symbol_id = sqlite_manager._get_or_create_id(conn, 'symbols', 'symbol', symbol)
        timeframe_id = sqlite_manager._get_or_create_id(conn, 'timeframes', 'timeframe', timeframe)
        logger.debug(f"DB IDs - Symbol: {symbol_id}, Timeframe: {timeframe_id}")
        conn.close() # Close after getting IDs
        conn = None

        # --- Analysis Path Selection ---
        while True: # Loop until a valid path is chosen and successfully prepared
            print("\n--- Analysis Path ---")
            choice = input("Select path: [D]efault (All Indicators), [T]weak (Optimize Single Indicator Per Lag): ").strip().lower()

            if choice == 'd':
                # --- Default Path Logic ---
                logger.info("Processing Default Path: Preparing default configs for ALL defined indicators.")
                is_tweak_path = False
                temp_config_list = []
                conn = sqlite_manager.create_connection(str(db_path))
                if not conn:
                    logger.critical("Failed reconnect DB for default config IDs.")
                    # Cannot proceed without DB connection here
                    raise ConnectionError("Failed to connect to database for default config processing.")

                try:
                    for name, definition in indicator_definitions.items():
                        params = definition.get('parameters', {})
                        # Use .get('default') safely
                        actual_defaults = {k: v.get('default') for k, v in params.items() if 'default' in v}
                        conditions = definition.get('conditions', [])
                        # Validate default parameters
                        if parameter_generator.evaluate_conditions(actual_defaults, conditions):
                            try:
                                config_id = sqlite_manager.get_or_create_indicator_config_id(conn, name, actual_defaults)
                                temp_config_list.append({'indicator_name': name, 'params': actual_defaults, 'config_id': config_id})
                            except Exception as cfg_err:
                                logger.error(f"Failed get/create config ID for default {name}: {cfg_err}")
                        else:
                            logger.warning(f"Default params for '{name}' ({actual_defaults}) invalid based on conditions. Skipping.")
                finally:
                    if conn: conn.close()
                    conn = None # Reset conn

                # Check estimated correlations
                estimated_outputs_per_config = 1.5 # Rough estimate
                num_default_configs = len(temp_config_list)
                actual_total_correlations = num_default_configs * estimated_outputs_per_config * max_lag
                logger.info(f"Default path generated {num_default_configs} default configurations.")
                logger.info(f"Estimated total correlations: ~{int(actual_total_correlations)} (Target: <= {config.TARGET_MAX_CORRELATIONS})")

                if actual_total_correlations > config.TARGET_MAX_CORRELATIONS * 1.1 :
                     proceed_choice = input(f"WARNING: Approx {int(actual_total_correlations)} correlations estimated. Proceed anyway? [y/N]: ").strip().lower() or 'n'
                     if proceed_choice != 'y':
                         logger.info("User aborted default path due to high correlation estimate.")
                         continue # Go back to path selection

                # Set the list for processing and mark success
                indicator_configs_to_process = temp_config_list
                analysis_path_successful = True
                break # Exit the loop

            elif choice == 't':
                # --- Tweak Path Logic (Per-Lag Optimization) ---
                logger.info("Processing Tweak Path: Selecting indicator for PER-LAG optimization.")
                is_tweak_path = True
                # Filter indicators that actually have parameters to optimize
                available_indicators = sorted([name for name, definition in indicator_definitions.items() if definition.get('parameters')])
                if not available_indicators:
                    logger.error("No indicators with parameters defined in JSON. Cannot run Tweak path.")
                    print("Error: No indicators found with tunable parameters.")
                    continue # Go back to path selection

                print("\nAvailable Indicators to Optimize:")
                for i, n in enumerate(available_indicators): print(f"{i+1}. {n}")

                selected_indicator_for_tweak = None
                while True: # Loop for indicator selection
                    try:
                        idx_str = input(f"Select indicator number (1-{len(available_indicators)}): ").strip()
                        idx = int(idx_str) - 1
                        if 0 <= idx < len(available_indicators):
                            selected_indicator_for_tweak = available_indicators[idx]
                            logger.info(f"User selected indicator for per-lag optimization: '{selected_indicator_for_tweak}'")
                            break # Exit indicator selection loop
                        else:
                            print("Invalid selection.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")

                definition = indicator_definitions[selected_indicator_for_tweak]

                # Confirm required columns are present
                required_for_opt = definition.get('required_inputs', []) + ['close']
                if not all(col in data.columns for col in required_for_opt):
                    logger.critical(f"Base data is missing columns required for optimizing {selected_indicator_for_tweak}: {required_for_opt}. Exiting.")
                    sys.exit(1)

                # Call the per-lag optimizer
                print(f"\nOptimizing parameters for '{selected_indicator_for_tweak}' for each lag...")
                print(f"(Using {config.OPTIMIZER_ITERATIONS} iterations PER lag, Max Lag={max_lag})")
                # print(f"(Scoring method within lag: 'max_abs')") # Scoring is implicit now

                best_config_per_lag_result, all_evaluated_configs_list = parameter_optimizer.optimize_parameters_per_lag(
                    indicator_name=selected_indicator_for_tweak,
                    indicator_def=definition,
                    base_data_with_required=data.copy(), # Pass a copy
                    max_lag=max_lag,
                    num_iterations_per_lag=config.OPTIMIZER_ITERATIONS,
                    db_path=str(db_path),
                    symbol_id=symbol_id,
                    timeframe_id=timeframe_id
                )

                # Check results
                if not best_config_per_lag_result and not all_evaluated_configs_list:
                     logger.error(f"Per-lag optimization failed for {selected_indicator_for_tweak}. No results returned.")
                     print("Optimization did not yield results. Check logs.")
                     continue # Go back to path selection
                else:
                     logger.info(f"Per-lag optimization completed for {selected_indicator_for_tweak}.")
                     logger.info(f"Found {len(all_evaluated_configs_list)} unique configurations evaluated across all lags.")
                     # Use the list of ALL evaluated configs for subsequent reporting/visualization
                     indicator_configs_to_process = all_evaluated_configs_list
                     analysis_path_successful = True
                     break # Exit the loop
            else:
                print("Invalid choice. Please enter 'd' or 't'.")
                logger.warning(f"Invalid analysis path choice: '{choice}'")

    # Handle potential ConnectionError from Default path DB failure
    except ConnectionError as ce:
        logger.critical(f"Database connection failed during analysis path setup: {ce}")
        sys.exit(1)
    except Exception as prep_err:
        logger.critical(f"Error during analysis path preparation: {prep_err}", exc_info=True)
        if conn: conn.close() # Ensure connection closed on error
        sys.exit(1)
    finally:
        if conn: conn.close() # Ensure connection closed normally

    # --- Exit if Analysis Path Failed ---
    if not analysis_path_successful or not indicator_configs_to_process:
        logger.error("No configurations prepared or analysis path failed. Exiting.")
        sys.exit(1)

    # --- Log Path Summary ---
    logger.info(f"Processing {len(indicator_configs_to_process)} final configurations.")
    if is_tweak_path:
        logger.info(f"Tweak path used: Configurations are all unique ones evaluated during per-lag optimization for '{selected_indicator_for_tweak}'.")
        # Print the best config found for each lag
        print("\n--- Best Configuration Found Per Lag ---")
        found_any_best = False
        for lag, result in sorted(best_config_per_lag_result.items()):
            if result:
                print(f"Lag {lag:>3}: Corr={result.get('correlation_at_lag', float('nan')):.4f}, ConfigID={result.get('config_id', 'N/A')}, Params={result.get('params', {})}")
                found_any_best = True
            else:
                print(f"Lag {lag:>3}: No suitable configuration found.")
        if not found_any_best: print("  (No best configurations found for any lag)")
        print("-" * 40)

    # --- Post-Optimization Processing ---
    data_with_indicators = data.copy() # Start with original data
    data_final = None # Will hold the final DataFrame used for correlation fetching/reporting base

    if not is_tweak_path:
        # --- Default Path Calculation ---
        logger.info("Starting standard indicator computation for Default path...")
        standard_configs = []
        custom_configs = []
        for cfg in indicator_configs_to_process:
            def_type = indicator_definitions.get(cfg['indicator_name'], {}).get('type')
            if def_type == 'custom':
                custom_configs.append(cfg)
            elif def_type: # ta-lib or pandas-ta
                standard_configs.append(cfg)
            else:
                logger.warning(f"Config ID {cfg.get('config_id','N/A')} for '{cfg.get('indicator_name','Unknown')}' has unknown type. Skipping.")

        logger.info(f"Processing {len(standard_configs)} standard configurations.")
        data_with_std_indicators = indicator_factory.compute_configured_indicators(data, standard_configs)
        logger.info("Standard indicator computation phase complete.")
        data_with_indicators = data_with_std_indicators # Update base

        # Apply custom indicators if any were selected
        if custom_configs:
            logger.info(f"Applying {len(custom_configs)} custom indicator configurations...")
            custom_names_to_run = {cfg['indicator_name'] for cfg in custom_configs}
            logger.warning(f"Custom indicator params might not be used by apply_all. Running based on names: {custom_names_to_run}")
            # Apply custom indicators on top of standard ones
            data_with_indicators = custom_indicators.apply_all_custom_indicators(data_with_indicators) # Apply to the result
            logger.info("Custom indicator application phase complete.")
        else:
            logger.info("No custom indicators to apply for Default path.")

        # --- Default Path Correlation Calculation ---
        logger.info("Calling correlation calculation for Default path...")
        logger.info(f"DataFrame shape before final dropna (Default Path): {data_with_indicators.shape}")
        # Drop rows with NaN in *any* column before correlation
        data_final = data_with_indicators.dropna(how='any')
        logger.info(f"Dropped {len(data_with_indicators) - len(data_final)} rows. Final shape for Default Correlation: {data_final.shape}")

        if len(data_final) < min_required_len_for_lag:
            logger.error(f"Insufficient rows ({len(data_final)}) after dropna for Default Path. Need {min_required_len_for_lag} for max_lag={max_lag}. Exiting.")
            sys.exit(1)

        # Calculate and store correlations
        corr_success = correlation_calculator.process_correlations(
            data_final, str(db_path), symbol_id, timeframe_id, indicator_configs_to_process, max_lag
        )
        if not corr_success:
            logger.error("Default path correlation phase failed. Exiting.")
            sys.exit(1)
        logger.info("Default path correlation calculation and storage complete.")

    else:
        # --- Tweak Path Post-Processing ---
        logger.info("Skipping main correlation calculation for Tweak path (done in optimizer).")
        logger.info("Re-calculating indicators for reporting base using ALL evaluated configs...")
        # Recompute indicators for the full set of evaluated configs for consistent reporting/visualization base
        data_with_indicators = indicator_factory.compute_configured_indicators(data, indicator_configs_to_process)
        # For tweak path reporting, we might not need to dropna aggressively,
        # correlations are already in DB. Let visualization handle NaNs.
        data_final = data_with_indicators # Keep NaNs for reporting base
        logger.info(f"Reporting base for Tweak Path created. Shape (with NaNs): {data_final.shape}")

    # --- Fetch Correlations & Generate Reports/Visualizations ---
    # This part now uses the full list of configs (`indicator_configs_to_process`)
    # determined by the chosen path (Default or Tweak).
    conn = sqlite_manager.create_connection(str(db_path))
    report_data_ok = False
    correlations_by_config_id = {}
    if not conn:
        logger.error("Failed connect DB for fetching final correlations.")
        sys.exit(1)
    try:
        config_ids_to_fetch = [cfg['config_id'] for cfg in indicator_configs_to_process if 'config_id' in cfg]
        if not config_ids_to_fetch:
             logger.error("No valid config IDs found to fetch correlations.")
             raise ValueError("No config IDs available for fetching.")

        logger.info(f"Fetching final correlations from DB for {len(config_ids_to_fetch)} config IDs (up to lag {max_lag})...")

        # Fetch correlations using the collected config IDs
        correlations_by_config_id = sqlite_manager.fetch_correlations(conn, symbol_id, timeframe_id, config_ids_to_fetch)

        if not correlations_by_config_id:
            logger.error("No correlation data structure returned from DB fetch.")
        else:
             # Determine the actual max lag WITH data across all fetched configs
             actual_max_lag_in_data = 0
             configs_with_any_data = 0
             for cfg_id, data_list in correlations_by_config_id.items():
                  if data_list and isinstance(data_list, list): # Check if list is not empty/None
                      configs_with_any_data += 1
                      try:
                          # Find the highest index (lag-1) with a non-NaN value
                          valid_indices = [i for i, v in enumerate(data_list) if pd.notna(v)]
                          if valid_indices:
                              actual_max_lag_in_data = max(actual_max_lag_in_data, max(valid_indices) + 1)
                      except (ValueError, TypeError) as e:
                           logger.warning(f"Error processing data list for config {cfg_id}: {e}")
             logger.info(f"Found correlation data for {configs_with_any_data} configurations in DB.")

             # Handle case where data exists but only for lag 0 or less (unlikely)
             if actual_max_lag_in_data == 0 and configs_with_any_data > 0:
                 logger.warning("Data found, but actual max lag with valid data is 0.")
                 # If requested max_lag > 0, maybe only lag 0 was valid? Use requested max_lag for reporting.
                 if max_lag > 0:
                     logger.warning(f"Proceeding with requested max_lag={max_lag} for reporting, results may be empty/NaN.")
                     actual_max_lag_in_data = max_lag # Use user requested lag
                 else:
                      logger.error("No valid positive lag data found and requested max_lag is 0. Reporting cannot proceed.")
                      actual_max_lag_in_data = 0 # Ensure it remains 0

             # Determine the final max_lag to use for reporting (minimum of requested and available)
             reporting_max_lag = min(max_lag, actual_max_lag_in_data) if actual_max_lag_in_data > 0 else 0

             if reporting_max_lag < max_lag:
                 logger.warning(f"Reporting max lag adjusted from {max_lag} down to {reporting_max_lag} based on available data.")

             if reporting_max_lag <= 0:
                 logger.error("No valid positive lag data found in DB (reporting_max_lag=0). Reporting cannot proceed.")
             else:
                 # Use adjusted lag for subsequent steps
                 max_lag = reporting_max_lag # IMPORTANT: Update max_lag for reporting

                 # Final check: are there *any* configs with *any* valid data up to the *adjusted* max_lag?
                 configs_with_valid_data_at_lag = 0
                 for cfg_id, data_list in correlations_by_config_id.items():
                     if data_list and len(data_list) >= max_lag: # Check length against adjusted max_lag
                         if any(pd.notna(x) for x in data_list[:max_lag]): # Check for any non-NaN up to adjusted max_lag
                             configs_with_valid_data_at_lag += 1

                 if configs_with_valid_data_at_lag > 0:
                     logger.info(f"Found data for {configs_with_valid_data_at_lag} configs up to adjusted lag {max_lag}.")
                     report_data_ok = True # Data is okay for reporting
                 else:
                     logger.error(f"ALL configurations lack valid correlation data up to adjusted lag {max_lag}.")

                 # Log if only partial data was retrieved
                 fetched_ids_count = len(correlations_by_config_id)
                 if 0 < configs_with_valid_data_at_lag < fetched_ids_count:
                     logger.warning(f"Retrieved valid data for only {configs_with_valid_data_at_lag}/{fetched_ids_count} requested configurations up to lag {max_lag}.")

    except ValueError as ve: # Catch specific error for no config IDs
        logger.error(f"Value error during correlation fetching setup: {ve}")
    except Exception as fetch_err:
        logger.error(f"Error fetching correlations: {fetch_err}", exc_info=True)
    finally:
        if conn: conn.close()

    # --- Exit if Correlation Fetching Failed ---
    if not report_data_ok:
        logger.error("Cannot generate reports/visualizations - no valid correlation data retrieved or max_lag is zero.")
        sys.exit(1)

    # --- Leaderboard Update ---
    logger.info("Attempting to update persistent leaderboard...")
    try:
        # Use the fetched full correlation data and the full list of processed configs
        leaderboard_manager.update_leaderboard(
            current_run_correlations=correlations_by_config_id,
            indicator_configs=indicator_configs_to_process,
            max_lag=max_lag, # Use the (potentially adjusted) max lag
            symbol=symbol,
            timeframe=timeframe,
            data_daterange=data_daterange_str,
            source_db_name=db_path.name
        )
    except Exception as lb_err:
        logger.error(f"Failed to update leaderboard: {lb_err}", exc_info=True)

    # --- Reporting & Visualization ---
    file_prefix = f"{timestamp_str}_{symbol}_{timeframe}"
    if is_tweak_path and selected_indicator_for_tweak:
         # Sanitize indicator name for filename
         safe_indicator_name = re.sub(r'[\\/*?:"<>|\s]+', '_', selected_indicator_for_tweak)
         file_prefix += f"_TWEAK_{safe_indicator_name}"

    # Generate Peak Correlation Report
    try:
        logger.info("Generating peak correlation report...")
        visualization_generator.generate_peak_correlation_report(
            correlations_by_config_id, indicator_configs_to_process, max_lag, config.REPORTS_DIR, file_prefix
        )
    except Exception as report_err:
        logger.error(f"Error generating peak correlation report: {report_err}", exc_info=True)

    # --- Visualization Preparation ---
    configs_for_viz = indicator_configs_to_process
    correlations_for_viz = correlations_by_config_id
    num_configs_for_viz = len(configs_for_viz)

    configs_for_limited_viz = configs_for_viz
    correlations_for_limited_viz = correlations_for_viz

    # Filter configs for visualization if necessary (e.g., too many from Tweak path)
    if num_configs_for_viz > config.HEATMAP_MAX_CONFIGS:
        logger.warning(f"Too many configurations ({num_configs_for_viz}) for detailed visualization. Limiting plots (Heatmap, Combined Chart, Lines) to top {config.HEATMAP_MAX_CONFIGS} based on peak absolute correlation.")

        # Calculate peak absolute correlation for sorting/filtering
        perf_data = []
        for cfg in configs_for_viz:
            cfg_id = cfg.get('config_id')
            if cfg_id is None: continue # Skip configs without ID
            corrs = correlations_for_viz.get(cfg_id, [])
            if corrs and len(corrs) >= max_lag: # Check against adjusted max_lag
                 corr_array = np.array(corrs[:max_lag], dtype=float)
                 if not np.isnan(corr_array).all():
                      peak_abs = np.nanmax(np.abs(corr_array))
                      if pd.notna(peak_abs):
                           perf_data.append({'config_id': cfg_id, 'peak_abs': peak_abs})

        if perf_data:
             perf_data.sort(key=lambda x: x['peak_abs'], reverse=True)
             top_config_ids_for_viz = {item['config_id'] for item in perf_data[:config.HEATMAP_MAX_CONFIGS]}

             # Ensure default config is included if relevant (Tweak path)
             if is_tweak_path and selected_indicator_for_tweak:
                 # Find default config ID from the full list processed
                 default_config_details = None
                 try:
                     indicator_def_viz = indicator_factory._get_indicator_definition(selected_indicator_for_tweak)
                     if indicator_def_viz:
                         default_params_viz = {k: v.get('default') for k, v in indicator_def_viz.get('parameters', {}).items() if 'default' in v}
                         # Search the full list used for processing
                         for cfg_full in indicator_configs_to_process:
                             if utils.compare_param_dicts(cfg_full.get('params',{}), default_params_viz):
                                 default_config_details = cfg_full
                                 break
                 except Exception as e:
                     logger.warning(f"Could not get default config details for viz check: {e}")

                 if default_config_details:
                     default_id_viz = default_config_details.get('config_id')
                     if default_id_viz is not None:
                         # If default is not in top N and list is full, replace worst
                         if default_id_viz not in top_config_ids_for_viz and len(top_config_ids_for_viz) >= config.HEATMAP_MAX_CONFIGS:
                             worst_in_top_id = perf_data[config.HEATMAP_MAX_CONFIGS-1]['config_id']
                             logger.warning(f"Replacing config {worst_in_top_id} with default {default_id_viz} in visualization set.")
                             top_config_ids_for_viz.discard(worst_in_top_id) # Use discard to avoid error if not present
                             top_config_ids_for_viz.add(default_id_viz)
                         # If default not in top N and list is not full, just add it
                         elif default_id_viz not in top_config_ids_for_viz:
                              top_config_ids_for_viz.add(default_id_viz)

             # Filter the main lists based on the final set of IDs for limited viz
             configs_for_limited_viz = [cfg for cfg in configs_for_viz if cfg.get('config_id') in top_config_ids_for_viz]
             correlations_for_limited_viz = {cfg_id: corrs for cfg_id, corrs in correlations_for_viz.items() if cfg_id in top_config_ids_for_viz}
             logger.info(f"Using filtered set of {len(configs_for_limited_viz)} configurations for limited visualizations.")
        else:
             logger.error("Could not determine performance to filter configurations for visualization. Using full set.")
             # Fallback: Use the original potentially large lists
             configs_for_limited_viz = configs_for_viz
             correlations_for_limited_viz = correlations_for_viz
    else:
        # If within limit, use the full set for all visualizations
        logger.info(f"Using full set of {num_configs_for_viz} configurations for all visualizations.")
        configs_for_limited_viz = configs_for_viz
        correlations_for_limited_viz = correlations_for_viz

    # --- Generate Visualizations ---
    logger.info("Starting visualization generation...")
    try:
        vis_success_count = 0
        vis_total_count = 4 # Number of visualization types attempted

        # Use the potentially filtered lists for plots with limits
        limited_viz_configs = configs_for_limited_viz
        limited_viz_corrs = correlations_for_limited_viz
        num_limited_configs = len(limited_viz_configs)

        # Generate Line Charts (Limited Set)
        logger.info(f"Generating separate line charts with CI (using {num_limited_configs} configs)...")
        visualization_generator.plot_correlation_lines(
            limited_viz_corrs, limited_viz_configs, max_lag, config.LINE_CHARTS_DIR, file_prefix
        )
        vis_success_count += 1

        # Generate Combined Chart (Limited Set)
        logger.info(f"Generating combined correlation chart (using {num_limited_configs} configs)...")
        visualization_generator.generate_combined_correlation_chart(
            limited_viz_corrs, limited_viz_configs, max_lag, config.COMBINED_CHARTS_DIR, file_prefix
        )
        vis_success_count += 1

        # Generate Heatmap (Limited Set)
        logger.info(f"Generating enhanced heatmap (using {num_limited_configs} configs)...")
        visualization_generator.generate_enhanced_heatmap(
            limited_viz_corrs, limited_viz_configs, max_lag, config.HEATMAPS_DIR, file_prefix, is_tweak_path
        )
        vis_success_count += 1

        # Generate Envelope Chart (Full Set - shows bounds across *all* evaluated)
        logger.info(f"Generating correlation envelope chart (using {num_configs_for_viz} configs)...")
        visualization_generator.generate_correlation_envelope_chart(
            correlations_for_viz, configs_for_viz, max_lag, config.REPORTS_DIR, file_prefix, is_tweak_path
        )
        vis_success_count += 1

        logger.info(f"Visualization generation finished ({vis_success_count}/{vis_total_count} types attempted).")

    except Exception as vis_err:
        logger.error(f"Error during visualization: {vis_err}", exc_info=True)

    # --- Final Summary ---
    end_time = datetime.now(timezone.utc)
    duration = end_time - start_time
    logger.info(f"--- Analysis Run Completed: {timestamp_str} ---")
    logger.info(f"Total execution time: {duration}")
    print(f"\nAnalysis complete using max_lag={max_lag}. Reports saved in '{config.REPORTS_DIR}'. Total time: {duration}")

    # --- Export Leaderboard ---
    try:
        logger.info("Exporting leaderboard...")
        leaderboard_manager.export_leaderboard_to_text() # Uses fixed filename
    except Exception as export_err:
        logger.error(f"Failed to export leaderboard to text: {export_err}", exc_info=True)

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        run_analysis()
        sys.exit(0) # Explicit success exit code
    except SystemExit as e:
         # Log non-zero exits as errors, zero exits as info
         if e.code != 0:
             logger.error(f"Analysis exited with code {e.code}.")
         else:
             logger.info("Analysis finished successfully.")
         sys.exit(e.code) # Propagate the exit code
    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user (Ctrl+C).")
        print("\nAnalysis interrupted by user.")
        sys.exit(130) # Standard exit code for Ctrl+C
    except Exception as e:
        # Log any other unhandled exception as critical
        logger.critical(f"An unhandled exception occurred in the main execution block: {e}", exc_info=True)
        print(f"\nAn critical error occurred: {e}")
        print("Please check the log file for details.")
        sys.exit(1) # General error exit code