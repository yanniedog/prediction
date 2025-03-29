# main.py
import logging
import logging_setup

logging_setup.setup_logging() # Initialize logging first

import sys
import json
from datetime import datetime
import pandas as pd
import os
import shutil
from pathlib import Path
import numpy as np
import re
import math # Import math for power/root calculation

# Import project modules
import utils
import config
import sqlite_manager
import data_manager
import parameter_generator # Still needed for condition evaluation, maybe random start points
import parameter_optimizer # NEW: For tweak path optimization
import indicator_factory
import correlation_calculator
import visualization_generator
import custom_indicators

logger = logging.getLogger(__name__)

# --- cleanup_previous_content function remains unchanged ---
def cleanup_previous_content():
    """Deletes specified previous output files and directories."""
    logger.info("Starting cleanup of previous outputs...")
    deleted_count = 0; error_count = 0
    locations_to_clean = [
        (config.DB_DIR, "*.db"),
        (config.LOG_DIR, "*.log"),
        (config.LOG_DIR, "*.txt"), # Also clean txt log files
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
                if target_dir.exists() and target_dir.is_dir():
                    logger.debug(f"Removing dir tree: '{target_dir}'...")
                    try: shutil.rmtree(target_dir); logger.info(f"Deleted dir tree: {target_dir}"); deleted_count += 1
                    except OSError as e: logger.error(f"Error deleting dir {target_dir}: {e}"); error_count += 1
            else: logger.debug(f"Skipping cleanup item: {item}")
        except Exception as e: logger.error(f"Error during cleanup: {e}"); error_count += 1
    logger.info(f"Cleanup finished. Deleted items/trees: {deleted_count}. Errors: {error_count}.")
    if error_count > 0: print("WARNING: Errors occurred during cleanup. Check logs.")


def run_analysis():
    """Main orchestration function for the analysis pipeline."""
    utils.clear_screen()

    start_time = datetime.now(); timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
    logger.info(f"--- Starting Analysis Run: {timestamp_str} ---")

    # --- Cleanup Prompt ---
    try:
        cleanup_choice = input("Delete all previously generated content (DBs, logs, charts, reports)? [Y/n]: ").strip().lower() or 'y'
        logger.info(f"User choice for cleanup: '{cleanup_choice}'")
        if cleanup_choice == 'y':
            print("Proceeding with cleanup...")
            # Ensure logging is shut down *before* deleting the log file
            logging.shutdown()
            cleanup_previous_content()
            # Re-initialize logging *after* cleanup
            logging_setup.setup_logging()
            logger.info(f"Re-initialized logging after cleanup. Continuing run: {timestamp_str}")
        else: print("Skipping cleanup."); logger.info("Skipping cleanup.")
    except Exception as e: logger.error(f"Error during cleanup prompt: {e}", exc_info=True); print("Error during cleanup, continuing...")


    # --- DB Init, Data Source, Indicator Def Loading ---
    placeholder_db_path = str(config.DB_DIR / "init_placeholder.db")
    if not sqlite_manager.initialize_database(placeholder_db_path): logger.warning("Placeholder DB init check failed.")
    data_source_info = data_manager.manage_data_source()
    if data_source_info is None: logger.info("Exiting: No data source selected."); sys.exit(0)
    db_path, symbol, timeframe = data_source_info
    logger.info(f"Using data source: {db_path.name} (Symbol: {symbol}, Timeframe: {timeframe})")
    indicator_definitions = None
    try:
        indicator_factory._load_indicator_definitions(); indicator_definitions = indicator_factory._INDICATOR_DEFS
        if not indicator_definitions: raise ValueError("Indicator definitions empty.")
        logger.info(f"Accessed {len(indicator_definitions)} indicator definitions.")
    except Exception as e: logger.critical(f"Failed load indicator definitions: {e}", exc_info=True); sys.exit(1)

    # --- Load Data EARLY --- Needed for optimizer and default path check
    data = data_manager.load_data(db_path)
    if data is None or data.empty: logger.error(f"Failed load/empty data from {db_path}. Exiting."); sys.exit(1)
    logger.info(f"Initial data loaded. Shape: {data.shape}. Checking length...")
    time_interval = utils.determine_time_interval(data) or "periods"; logger.info(f"Determined time interval: {time_interval}")

    # --- Get max_lag and check data length ---
    # For now, use a fixed value. Could potentially be dynamic based on data length.
    max_lag = 300 # Keep this fixed for now, or adjust as needed
    logger.info(f"Using fixed max_lag = {max_lag} for calculations.")
    min_required_len_for_lag = max_lag + config.MIN_DATA_POINTS_FOR_LAG # Need sufficient points *beyond* the max lag
    if len(data) < min_required_len_for_lag:
        logger.critical(f"Insufficient rows ({len(data)}) in initial data load. Need at least {min_required_len_for_lag} for max_lag={max_lag}. Exiting.")
        print(f"Error: Not enough historical data ({len(data)} rows) to perform analysis with max_lag={max_lag}. Need at least {min_required_len_for_lag} rows.")
        print(f"Try downloading more data or reducing max_lag.")
        sys.exit(1)
    logger.info(f"Data length ({len(data)}) sufficient for max_lag={max_lag} (required: {min_required_len_for_lag}).")


    # --- Config Prep ---
    conn = None; indicator_configs_to_process = []; is_tweak_path = False; selected_indicator_for_tweak = None
    analysis_path_successful = False # Flag to track if a path was chosen and prepared

    try:
        conn = sqlite_manager.create_connection(str(db_path))
        if not conn: logger.critical("Failed connect main DB pre-config."); sys.exit(1)
        symbol_id = sqlite_manager._get_or_create_id(conn, 'symbols', 'symbol', symbol)
        timeframe_id = sqlite_manager._get_or_create_id(conn, 'timeframes', 'timeframe', timeframe)
        logger.debug(f"DB IDs - Symbol: {symbol_id}, Timeframe: {timeframe_id}")
        # Close connection here, optimizer/correlation calc will reopen as needed
        conn.close()
        conn = None

        while True: # Main loop for analysis path selection
            print("\n--- Analysis Path ---")
            choice = input("Select path: [D]efault (All Indicators), [T]weak (Single Indicator, Optimized): ").strip().lower() # Removed default 'd'

            if choice == 'd':
                logger.info("Processing Default Path: Preparing default configs for ALL defined indicators.")
                is_tweak_path = False
                temp_config_list = []
                # Reopen connection to get config IDs
                conn = sqlite_manager.create_connection(str(db_path))
                if not conn: logger.critical("Failed reconnect DB for default config IDs."); break # Exit loop
                try:
                    for name, definition in indicator_definitions.items():
                        params = definition.get('parameters', {}); actual_defaults = {k: v.get('default') for k, v in params.items() if 'default' in v}
                        conditions = definition.get('conditions', [])
                        if parameter_generator.evaluate_conditions(actual_defaults, conditions):
                            try:
                                config_id = sqlite_manager.get_or_create_indicator_config_id(conn, name, actual_defaults)
                                temp_config_list.append({ 'indicator_name': name, 'params': actual_defaults, 'config_id': config_id })
                            except Exception as cfg_err:
                                logger.error(f"Failed get/create config ID for default {name}: {cfg_err}")
                        else:
                            logger.warning(f"Default params for '{name}' invalid based on conditions. Skipping.")
                finally:
                    if conn: conn.close(); conn = None

                # Check total correlations for default path
                num_default_configs = len(temp_config_list)
                # Estimate requires knowledge of outputs per indicator, * 1 is a placeholder
                # A more accurate estimate needs parsing indicator_params.json outputs
                estimated_outputs_per_config = 1.5 # Average guess
                actual_total_correlations = num_default_configs * estimated_outputs_per_config * max_lag
                logger.info(f"Default path generated {num_default_configs} default configurations.")
                logger.info(f"Estimated total correlations: ~{int(actual_total_correlations)} (Target: <= {config.TARGET_MAX_CORRELATIONS})")
                if actual_total_correlations > config.TARGET_MAX_CORRELATIONS * 1.1 : # Allow some buffer
                     proceed_choice = input(f"WARNING: This may generate significantly more than {config.TARGET_MAX_CORRELATIONS} correlations ({int(actual_total_correlations)} estimated). Proceed anyway? [y/N]: ").strip().lower() or 'n'
                     if proceed_choice != 'y':
                         logger.info("User aborted default path due to potentially high correlation count.")
                         continue # Go back to path selection
                indicator_configs_to_process = temp_config_list
                analysis_path_successful = True
                break # Exit path selection loop

            elif choice == 't':
                logger.info("Processing Tweak Path: Selecting indicator for optimization.")
                is_tweak_path = True; available_indicators = sorted(list(indicator_definitions.keys()))
                print("\nAvailable Indicators to Optimize:")
                [print(f"{i+1}. {n}") for i, n in enumerate(available_indicators)]
                selected_indicator_for_tweak = None
                while True: # Loop for indicator selection
                    try:
                        idx_str = input(f"Select indicator number (1-{len(available_indicators)}): ").strip(); idx = int(idx_str) - 1
                        if 0 <= idx < len(available_indicators):
                             selected_indicator_for_tweak = available_indicators[idx]
                             logger.info(f"User selected indicator for optimization: '{selected_indicator_for_tweak}'")
                             break
                        else: print("Invalid selection.")
                    except ValueError: print("Invalid input.")

                definition = indicator_definitions[selected_indicator_for_tweak]

                # --- Run the Optimizer ---
                print(f"\nOptimizing parameters for '{selected_indicator_for_tweak}'...")
                print(f"(Targeting top {config.HEATMAP_MAX_CONFIGS} configurations over ~{config.OPTIMIZER_ITERATIONS} iterations)")

                # Pass the initially loaded data (make sure it includes necessary base columns like OHLCV)
                # The optimizer needs the base data to calculate indicators internally.
                # Ensure required columns are present before calling optimizer
                required_for_opt = definition.get('required_inputs', []) + ['close'] # Optimizer needs 'close' too
                if not all(col in data.columns for col in required_for_opt):
                    logger.critical(f"Base data is missing columns required for optimizing {selected_indicator_for_tweak}: {required_for_opt}. Exiting.")
                    sys.exit(1)

                # The optimizer now returns the list including score and correlations
                optimizer_results = parameter_optimizer.optimize_parameters(
                    indicator_name=selected_indicator_for_tweak,
                    indicator_def=definition,
                    base_data_with_required=data.copy(), # Pass a copy
                    max_lag=max_lag,
                    num_iterations=config.OPTIMIZER_ITERATIONS,
                    target_configs=config.HEATMAP_MAX_CONFIGS,
                    db_path=str(db_path),
                    symbol_id=symbol_id, # Fetched earlier
                    timeframe_id=timeframe_id # Fetched earlier
                )

                # Reformat optimizer results to match the structure expected by downstream functions
                # (which is List[Dict[str, Any]] with 'indicator_name', 'params', 'config_id')
                indicator_configs_to_process = [
                    {'indicator_name': r['indicator_name'], 'params': r['params'], 'config_id': r['config_id']}
                    for r in optimizer_results # Extract needed keys
                ]
                # Store the full optimizer results separately if needed later, but indicator_configs_to_process is used for consistency
                # We also need the correlations fetched/calculated by the optimizer for the report
                correlations_by_config_id_from_opt = {r['config_id']: r.get('correlations', []) for r in optimizer_results}


                if not indicator_configs_to_process:
                     logger.error(f"Optimization failed to return any valid configurations for {selected_indicator_for_tweak}.")
                     print("Optimization did not yield results. Please check logs.")
                     continue # Loop back to allow user to choose another indicator or path
                else:
                    logger.info(f"Optimization completed. Found {len(indicator_configs_to_process)} configurations for {selected_indicator_for_tweak}.")
                    analysis_path_successful = True
                    break # Exit path selection loop

            else: print("Invalid choice. Please enter 'd' or 't'."); logger.warning(f"Invalid analysis path choice: '{choice}'")

    except Exception as prep_err:
        logger.critical(f"Error during analysis path preparation: {prep_err}", exc_info=True)
        if conn: conn.close() # Ensure connection closed on error
        sys.exit(1)
    finally:
        if conn: conn.close() # Final check

    if not analysis_path_successful or not indicator_configs_to_process:
        logger.error("No configurations prepared or analysis path failed. Exiting.")
        sys.exit(1)
    logger.info(f"Prepared {len(indicator_configs_to_process)} configurations for {'optimization' if is_tweak_path else 'default'} analysis.")

    # --- Indicator Computation (Standard Path Only) ---
    data_with_indicators = data.copy() # Start with original data
    if not is_tweak_path:
        logger.info("Starting standard indicator computation for Default path...")
        # Filter configs - separate standard and custom based on definitions
        standard_configs = []
        custom_configs = []
        for cfg in indicator_configs_to_process:
            def_type = indicator_definitions.get(cfg['indicator_name'], {}).get('type')
            if def_type == 'custom':
                custom_configs.append(cfg)
            elif def_type: # Treat TA-Lib, Pandas-TA, etc. as standard
                standard_configs.append(cfg)
            else:
                logger.warning(f"Config ID {cfg['config_id']} for '{cfg['indicator_name']}' has unknown type. Skipping.")

        logger.info(f"Processing {len(standard_configs)} standard configurations.")
        data_with_std_indicators = indicator_factory.compute_configured_indicators(data, standard_configs)
        logger.info("Standard indicator computation phase complete.")

        data_with_indicators = data_with_std_indicators # Start with standard results
        if custom_configs:
            logger.info(f"Applying {len(custom_configs)} custom indicator configurations...")
            custom_names_to_run = {cfg['indicator_name'] for cfg in custom_configs}
            logger.warning(f"Custom indicator params might not be used by apply_all. Running based on names: {custom_names_to_run}")
            data_with_indicators = custom_indicators.apply_all_custom_indicators(data_with_std_indicators) # Apply custom on top
            logger.info("Custom indicator application phase complete.")
        else:
            logger.info("No custom indicators to apply for Default path.")

        # --- Calculate & Store Correlations (Default Path Only) ---
        logger.info("Calling correlation calculation for Default path...")
        # Perform final dropna *before* correlation calculation for default path
        logger.info(f"DataFrame shape before final dropna (Default Path): {data_with_indicators.shape}")
        data_final = data_with_indicators.dropna(how='any')
        logger.info(f"Dropped {len(data_with_indicators) - len(data_final)} rows. Final shape for Default Correlation: {data_final.shape}")
        if len(data_final) < min_required_len_for_lag:
             logger.error(f"Insufficient rows ({len(data_final)}) remaining after final dropna for Default Path. Need at least {min_required_len_for_lag}. Exiting.")
             sys.exit(1)

        corr_success = correlation_calculator.process_correlations(data_final, str(db_path), symbol_id, timeframe_id, indicator_configs_to_process, max_lag)
        if not corr_success: logger.error("Default path correlation phase failed. Exiting."); sys.exit(1)
        logger.info("Default path correlation calculation and storage complete.")

    else: # Tweak Path
        logger.info("Skipping main indicator computation and correlation steps for Tweak path (done within optimizer).")
        # Need data_final with the *optimized* indicators for reporting/visualization consistency
        logger.info("Re-calculating indicators for the final optimized configurations for reporting...")
        data_with_indicators = indicator_factory.compute_configured_indicators(data, indicator_configs_to_process)
        logger.info(f"DataFrame shape before final dropna (Tweak Path): {data_with_indicators.shape}")
        data_final = data_with_indicators.dropna(how='any')
        logger.info(f"Dropped {len(data_with_indicators) - len(data_final)} rows. Final shape for Tweak Reporting: {data_final.shape}")
        if len(data_final) < min_required_len_for_lag:
             logger.error(f"Insufficient rows ({len(data_final)}) remaining after final dropna for Tweak Path. Need at least {min_required_len_for_lag}. Exiting.")
             sys.exit(1)


    # --- Retrieve Correlations (Common for both paths) ---
    # In Tweak path, we already have them from optimizer_results, but fetching ensures consistency and gets data if run was interrupted.
    conn = sqlite_manager.create_connection(str(db_path)); report_data_ok = False
    correlations_by_config_id = {} # Initialize
    if not conn: logger.error("Failed connect DB for fetching correlations."); sys.exit(1)
    try:
        config_ids_to_fetch = [cfg['config_id'] for cfg in indicator_configs_to_process]
        logger.info(f"Fetching final correlations from DB for {len(config_ids_to_fetch)} config IDs (up to lag {max_lag})...")
        correlations_by_config_id = sqlite_manager.fetch_correlations(conn, symbol_id, timeframe_id, config_ids_to_fetch)

        if not correlations_by_config_id:
             logger.error("No correlation data structure returned from DB fetch.")
        else:
             # Find the actual max lag present in the fetched data for *any* config ID
             actual_max_lag_in_data = 0
             configs_with_any_data = 0
             for cfg_id, data_list in correlations_by_config_id.items():
                  if data_list: # Check if list is not empty
                      configs_with_any_data += 1
                      try:
                          # Find the last non-None value's index (+1 for lag)
                          valid_indices = [i for i, v in enumerate(data_list) if pd.notna(v)]
                          if valid_indices:
                              actual_max_lag_in_data = max(actual_max_lag_in_data, max(valid_indices) + 1)
                      except ValueError: pass # Handle lists with only Nones
             logger.info(f"Found correlation data for {configs_with_any_data} configurations in DB.")
             logger.info(f"Actual maximum lag found in fetched data: {actual_max_lag_in_data}")

             # Adjust max_lag for reporting if necessary
             # Use the *minimum* of the originally set max_lag and the actual max lag found
             # This prevents errors if the DB contains less data than initially requested.
             reporting_max_lag = min(max_lag, actual_max_lag_in_data) if actual_max_lag_in_data > 0 else 0
             if reporting_max_lag < max_lag:
                  logger.warning(f"Reporting max lag adjusted from {max_lag} to {reporting_max_lag} based on available data.")
             if reporting_max_lag <= 0:
                  logger.error("No valid positive lag data found in DB. Reporting cannot proceed.")
             else:
                 max_lag = reporting_max_lag # Use the adjusted lag for reports/plots
                 # Check if *any* config has valid data up to the reporting max_lag
                 configs_with_valid_data_at_lag = sum(1 for data_list in correlations_by_config_id.values() if data_list and len(data_list) >= max_lag and any(pd.notna(x) for x in data_list[:max_lag]))
                 if configs_with_valid_data_at_lag > 0:
                     logger.info(f"Sufficient data found for {configs_with_valid_data_at_lag} configurations up to adjusted lag {max_lag} for reporting.")
                     report_data_ok = True
                 else:
                     logger.error(f"Retrieved structure, but ALL configurations lack valid correlation data up to adjusted lag {max_lag}.")

                 if 0 < configs_with_valid_data_at_lag < len(config_ids_to_fetch):
                     logger.warning(f"Retrieved valid data for only {configs_with_valid_data_at_lag}/{len(config_ids_to_fetch)} requested configurations up to lag {max_lag}.")

    except Exception as fetch_err:
        logger.error(f"Error fetching correlations: {fetch_err}", exc_info=True)
    finally:
        if conn: conn.close()

    if not report_data_ok:
        logger.error("Cannot generate reports/visualizations - no valid correlation data retrieved or max_lag is zero.")
        print("Error: Could not retrieve valid correlation data from the database. Cannot generate reports.")
        sys.exit(1)

    # --- Generate Peak Correlation Report (CSV and Console) ---
    file_prefix = f"{timestamp_str}_{symbol}_{timeframe}"
    try:
        logger.info("Generating peak correlation report...")
        visualization_generator.generate_peak_correlation_report(
            correlations_by_config_id,
            indicator_configs_to_process, # Use the final list of configs
            max_lag, # Use the potentially adjusted max_lag
            config.REPORTS_DIR,
            file_prefix
        )
    except Exception as report_err:
        logger.error(f"Error generating peak correlation report: {report_err}", exc_info=True)


    # --- Visualization ---
    logger.info("Starting visualization generation...")
    try:
        vis_success_count = 0; vis_total_count = 4 # Assuming 4 visualization types

        logger.info("Generating separate line charts with CI..."); visualization_generator.plot_correlation_lines(correlations_by_config_id, indicator_configs_to_process, max_lag, config.LINE_CHARTS_DIR, file_prefix); vis_success_count +=1
        logger.info("Generating combined correlation chart..."); visualization_generator.generate_combined_correlation_chart(correlations_by_config_id, indicator_configs_to_process, max_lag, config.COMBINED_CHARTS_DIR, file_prefix); vis_success_count +=1
        logger.info("Generating enhanced heatmap..."); visualization_generator.generate_enhanced_heatmap(correlations_by_config_id, indicator_configs_to_process, max_lag, config.HEATMAPS_DIR, file_prefix, is_tweak_path); vis_success_count +=1
        logger.info("Generating correlation envelope chart..."); visualization_generator.generate_correlation_envelope_chart(correlations_by_config_id, indicator_configs_to_process, max_lag, config.REPORTS_DIR, file_prefix); vis_success_count +=1

        logger.info(f"Visualization generation finished ({vis_success_count}/{vis_total_count} types attempted).")
    except Exception as vis_err: logger.error(f"Error during visualization: {vis_err}", exc_info=True)

    # --- Finalization ---
    end_time = datetime.now(); duration = end_time - start_time
    logger.info(f"--- Analysis Run Completed: {timestamp_str} ---")
    logger.info(f"Total execution time: {duration}")
    print(f"\nAnalysis complete. Reports saved in '{config.REPORTS_DIR}'. Total time: {duration}")

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        run_analysis()
        sys.exit(0)
    except SystemExit as e:
         # Only log error if exit code is non-zero
         if e.code != 0: logger.error(f"Analysis exited with code {e.code}.")
         else: logger.info("Analysis finished successfully.")
         sys.exit(e.code) # Propagate the exit code
    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user (Ctrl+C).")
        print("\nAnalysis interrupted by user.")
        sys.exit(130) # Standard exit code for Ctrl+C
    except Exception as e:
        logger.critical(f"An unhandled exception occurred in the main execution block: {e}", exc_info=True)
        print(f"\nAn critical error occurred: {e}")
        print("Please check the log file for details.")
        sys.exit(1)