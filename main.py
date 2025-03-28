# main.py
import logging
import logging_setup

logging_setup.setup_logging()

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

# Import utils first for clear_screen
import utils
import config # Import config to get TARGET_MAX_CORRELATIONS etc.
import sqlite_manager
import data_manager
import parameter_generator
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
            cleanup_previous_content()
            logging.shutdown()
            logging_setup.setup_logging()
            logger.info(f"Re-initialized logging after cleanup. Continuing run: {timestamp_str}")
        else: print("Skipping cleanup."); logger.info("Skipping cleanup.")
    except Exception as e: logger.error(f"Error during cleanup: {e}", exc_info=True); print("Error during cleanup, continuing...")

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

    # --- Config Prep ---
    conn = None; indicator_configs_to_process = []; is_tweak_path = False; selected_indicator_for_tweak = None

    # --- Get max_lag early for calculation ---
    # For now, use a fixed value, but this could be made dynamic later
    # based on data length if needed, although the user now controls complexity
    # via the range factor.
    max_lag = 300 # Keep this fixed for now, or adjust as needed
    logger.info(f"Using fixed max_lag = {max_lag} for calculations.")

    try:
        conn = sqlite_manager.create_connection(str(db_path))
        if not conn: logger.critical("Failed connect main DB pre-config."); sys.exit(1)
        symbol_id = sqlite_manager._get_or_create_id(conn, 'symbols', 'symbol', symbol)
        timeframe_id = sqlite_manager._get_or_create_id(conn, 'timeframes', 'timeframe', timeframe)
        logger.debug(f"DB IDs - Symbol: {symbol_id}, Timeframe: {timeframe_id}")

        while True: # Main loop for analysis path selection
            print("\n--- Analysis Path ---")
            choice = input("Select path: [D]efault (All Indicators, default), [T]weak (Single Indicator): ").strip().lower() or 'd'
            logger.info(f"User selected analysis path: '{choice}'")

            if choice == 'd':
                logger.info("Processing Default Path: Preparing default configs for ALL defined indicators.")
                is_tweak_path = False
                temp_config_list = []
                for name, definition in indicator_definitions.items():
                    params = definition.get('parameters', {}); actual_defaults = {k: v.get('default') for k, v in params.items() if 'default' in v}
                    conditions = definition.get('conditions', [])
                    if parameter_generator.evaluate_conditions(actual_defaults, conditions):
                         try: config_id = sqlite_manager.get_or_create_indicator_config_id(conn, name, actual_defaults); temp_config_list.append({ 'indicator_name': name, 'params': actual_defaults, 'config_id': config_id })
                         except Exception as cfg_err: logger.error(f"Failed get/create config ID for {name}: {cfg_err}")
                    else: logger.warning(f"Default params for '{name}' invalid. Skipping.")

                # Check total correlations for default path
                num_default_configs = len(temp_config_list)
                actual_total_correlations = num_default_configs * max_lag
                logger.info(f"Default path will generate {num_default_configs} configurations.")
                logger.info(f"Estimated total correlations: {actual_total_correlations} (Target: <= {config.TARGET_MAX_CORRELATIONS})")
                if actual_total_correlations > config.TARGET_MAX_CORRELATIONS:
                     proceed_choice = input(f"WARNING: This will generate > {config.TARGET_MAX_CORRELATIONS} correlations. Proceed anyway? [y/N]: ").strip().lower() or 'n'
                     if proceed_choice != 'y':
                         logger.info("User aborted default path due to high correlation count.")
                         continue # Go back to path selection
                indicator_configs_to_process = temp_config_list
                break # Exit path selection loop

            elif choice == 't':
                logger.info("Processing Tweak Path: Selecting indicator for tweaking.")
                is_tweak_path = True; available_indicators = sorted(list(indicator_definitions.keys()))
                print("\nAvailable Indicators to Tweak:"); [print(f"{i+1}. {n}") for i, n in enumerate(available_indicators)]
                selected_indicator_for_tweak = None
                while True:
                    try:
                        idx_str = input(f"Select indicator number (1-{len(available_indicators)}): ").strip(); idx = int(idx_str) - 1
                        if 0 <= idx < len(available_indicators): selected_indicator_for_tweak = available_indicators[idx]; logger.info(f"User selected tweak: '{selected_indicator_for_tweak}'"); break
                        else: print("Invalid selection.")
                    except ValueError: print("Invalid input.")

                definition = indicator_definitions[selected_indicator_for_tweak]
                param_defs = definition.get('parameters', {}); conditions = definition.get('conditions', [])

                # --- Calculate Suggested Range Steps ---
                tweakable_params = {p: d for p, d in param_defs.items() if d.get('default') is not None and isinstance(d.get('default'), (int, float))}
                num_params_to_tweak = len(tweakable_params)
                suggested_range_steps = 2 # Default suggestion (5 values)

                if num_params_to_tweak > 0 and max_lag > 0:
                    try:
                        # Calculate ideal number of values per parameter to stay within budget
                        target_configs = config.TARGET_MAX_CORRELATIONS / max_lag
                        # Handle potential for target_configs being < 1 if TARGET_MAX_CORRELATIONS is very low or max_lag very high
                        if target_configs < 1: target_configs = 1
                        target_values_per_param = target_configs ** (1 / num_params_to_tweak)
                        # Convert number of values to range_steps (num_values = 2*steps + 1 => steps = (num_values - 1) / 2)
                        calculated_steps = (target_values_per_param - 1) / 2
                        # Round and ensure minimum of 1 step (3 values total)
                        suggested_range_steps = max(1, int(round(calculated_steps)))
                        logger.info(f"Calculated suggested range_steps = {suggested_range_steps} based on {num_params_to_tweak} tweakable params, max_lag={max_lag}, target={config.TARGET_MAX_CORRELATIONS}.")
                    except Exception as calc_err:
                        logger.warning(f"Could not calculate suggested range steps: {calc_err}. Using default suggestion: {suggested_range_steps}.")
                elif num_params_to_tweak == 0:
                     logger.warning(f"Indicator '{selected_indicator_for_tweak}' has no tweakable numeric parameters with defaults. Only default config will be run.")
                     suggested_range_steps = 0 # Indicate only default
                else:
                     logger.warning("Cannot calculate suggested range steps (num_params_to_tweak or max_lag is zero). Using default suggestion.")

                # --- Loop for User Input and Confirmation ---
                while True:
                    print(f"\n--- Tweaking '{selected_indicator_for_tweak}' ---")
                    print(f"Indicator has {num_params_to_tweak} tweakable numeric parameter(s).")
                    print(f"Max Lag for correlation is {max_lag}.")
                    print(f"Target max correlations: {config.TARGET_MAX_CORRELATIONS}.")
                    if num_params_to_tweak > 0:
                         print(f"Suggested 'Range Steps': {suggested_range_steps} (This means 2*{suggested_range_steps}+1 = {2*suggested_range_steps+1} values per parameter).")
                         prompt_text = f"Enter desired Range Steps (integer >= 1, default={suggested_range_steps}): "
                         min_steps = 1
                    else:
                         print("Only the default parameter set will be used.")
                         prompt_text = "Enter 0 to confirm running only the default: "
                         min_steps = 0
                         suggested_range_steps = 0 # Force 0 if no params

                    try:
                        user_steps_str = input(prompt_text).strip()
                        if not user_steps_str: # User pressed Enter
                            user_range_steps = suggested_range_steps
                        else:
                            user_range_steps = int(user_steps_str)

                        if user_range_steps < min_steps:
                            print(f"Range Steps must be >= {min_steps}.")
                            logger.warning(f"Invalid user input for range steps: {user_steps_str}")
                            continue

                        logger.info(f"User chose range_steps = {user_range_steps}.")

                        # Generate configs with the chosen steps
                        tweaked_param_sets = parameter_generator.generate_configurations(
                            param_defs, conditions, range_steps=user_range_steps
                        )
                        num_generated_configs = len(tweaked_param_sets)
                        actual_total_correlations = num_generated_configs * max_lag

                        print(f"\nThis will generate {num_generated_configs} configurations.")
                        print(f"Estimated total correlations: {actual_total_correlations}")

                        if actual_total_correlations > config.TARGET_MAX_CORRELATIONS:
                            warn_choice = input(f"WARNING: Exceeds target of {config.TARGET_MAX_CORRELATIONS}. [P]roceed, [R]e-adjust steps, [A]bort? [P/r/a]: ").strip().lower() or 'p'
                            if warn_choice == 'r':
                                logger.info("User chose to re-adjust range steps.")
                                continue # Loop back to ask for steps again
                            elif warn_choice == 'a':
                                logger.info("User aborted tweak path.")
                                indicator_configs_to_process = [] # Ensure empty list
                                break # Break inner loop, will go back to main path choice
                            elif warn_choice == 'p':
                                logger.info("User chose to proceed despite high correlation count.")
                                # Fall through to generate config IDs
                            else:
                                print("Invalid choice.")
                                continue # Ask warning question again
                        # If within budget or user chose 'p', proceed to generate IDs

                        if not tweaked_param_sets:
                             logger.error(f"No valid configs generated for '{selected_indicator_for_tweak}'. Try different range steps or check conditions.");
                             # Allow user to retry
                             retry_choice = input("No configurations generated. [R]etry steps, [A]bort? [R/a]: ").strip().lower() or 'r'
                             if retry_choice == 'r':
                                 continue
                             else:
                                 indicator_configs_to_process = []
                                 break # Exit inner loop, return to main path choice

                        # Generate Config IDs
                        temp_config_list = []
                        failed_id_count = 0
                        for params in tweaked_param_sets:
                             try:
                                 config_id = sqlite_manager.get_or_create_indicator_config_id(conn, selected_indicator_for_tweak, params)
                                 temp_config_list.append({ 'indicator_name': selected_indicator_for_tweak, 'params': params, 'config_id': config_id })
                             except Exception as cfg_err:
                                 logger.error(f"Failed get/create config ID for {selected_indicator_for_tweak} params {params}: {cfg_err}")
                                 failed_id_count += 1

                        if failed_id_count > 0:
                             logger.error(f"Failed to prepare {failed_id_count} config IDs. The analysis might be incomplete.")
                             # Decide whether to proceed or abort based on severity? For now, proceed.

                        if not temp_config_list:
                             logger.error(f"No config IDs successfully generated for '{selected_indicator_for_tweak}'.")
                             # Allow user to retry?
                             retry_choice = input("No config IDs generated. [R]etry steps, [A]bort? [R/a]: ").strip().lower() or 'r'
                             if retry_choice == 'r':
                                 continue
                             else:
                                 indicator_configs_to_process = []
                                 break # Exit inner loop, return to main path choice
                        else:
                             indicator_configs_to_process = temp_config_list
                             break # Exit inner loop (user steps confirmed)

                    except ValueError:
                        print("Invalid input. Please enter an integer.")
                        logger.warning("Invalid non-integer input for range steps.")
                # If we broke from the inner loop without setting indicator_configs_to_process,
                # the outer loop will continue (asking for D/T again).
                # If we successfully set indicator_configs_to_process, break the outer loop.
                if indicator_configs_to_process:
                    break # Exit outer path selection loop

            else: print("Invalid choice."); logger.warning(f"Invalid analysis path choice: '{choice}'")

    except Exception as prep_err: logger.critical(f"Error during config prep: {prep_err}", exc_info=True); sys.exit(1)
    finally:
        if conn: conn.close(); logger.debug("Closed DB connection after config prep.")

    if not indicator_configs_to_process: logger.error("No configs prepared. Exiting."); sys.exit(1)
    logger.info(f"Prepared {len(indicator_configs_to_process)} configurations for analysis.")

    # 5. Load Data
    data = data_manager.load_data(db_path)
    if data is None or data.empty: logger.error(f"Failed load/empty data from {db_path}. Exiting."); sys.exit(1)
    logger.info(f"Data sample entering indicator calculation (first 5 rows):\n{data.head(5).to_string()}")
    time_interval = utils.determine_time_interval(data) or "periods"; logger.info(f"Determined time interval: {time_interval}")

    # 6. Compute Indicators
    logger.info("Starting standard indicator computation...")
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

    data_with_all_indicators = data_with_std_indicators # Start with standard results
    if custom_configs:
         logger.info(f"Applying {len(custom_configs)} custom indicator configurations (note: params might not be fully dynamic yet)...")
         # NOTE: Custom indicators apply_all currently uses default params from within the functions.
         # If custom indicators need to use the generated `params` from `custom_configs`,
         # `apply_all_custom_indicators` and the individual custom functions would need significant changes.
         # For now, it just runs the custom indicators defined in the list.
         custom_names_to_run = {cfg['indicator_name'] for cfg in custom_configs}
         logger.warning(f"Custom indicator params from tweak path might not be used by apply_all. Running based on names: {custom_names_to_run}")
         data_with_all_indicators = custom_indicators.apply_all_custom_indicators(data_with_std_indicators) # Apply custom on top
         logger.info("Custom indicator application phase complete.")
    else:
         logger.info("No custom indicators to apply.")

    logger.info(f"DataFrame shape before final dropna: {data_with_all_indicators.shape}")
    if data_with_all_indicators.empty: logger.error("DataFrame empty before final dropna."); sys.exit(1)

    # NaN Checks and Drop
    indicator_cols_added = list(data_with_all_indicators.columns.difference(data.columns))
    logger.info(f"Columns added by indicator factory: {len(indicator_cols_added)}")
    if not indicator_cols_added:
         logger.error("CRITICAL: No indicator columns were added to the DataFrame after computation. Exiting.")
         sys.exit(1)

    nan_check = data_with_all_indicators[indicator_cols_added].isnull().all(); all_nan_columns = nan_check[nan_check].index.tolist()
    if all_nan_columns:
         logger.error(f"CRITICAL: Indicator columns with ONLY NaNs before final dropna: {all_nan_columns}")
         logger.warning(f"Dropping all-NaN columns: {all_nan_columns}")
         try:
            data_with_all_indicators = data_with_all_indicators.drop(columns=all_nan_columns)
            logger.info(f"Shape after dropping all-NaN columns: {data_with_all_indicators.shape}")
            if data_with_all_indicators.empty: logger.error("DataFrame empty after dropping all-NaN cols."); sys.exit(1)
         except Exception as drop_err: logger.error(f"Failed drop all-NaN columns: {drop_err}.")

    nan_percentage = (data_with_all_indicators.isnull().sum()/len(data_with_all_indicators))*100; high_nan_cols = nan_percentage[nan_percentage > 99.0].index.tolist()
    if high_nan_cols: logger.warning(f"Columns >99% NaNs before final dropna: {high_nan_cols}")

    initial_len = len(data_with_all_indicators)
    # Drop rows with ANY NaN across the whole DataFrame AFTER indicators are calculated
    data_final = data_with_all_indicators.dropna(how='any')
    dropped_rows = initial_len - len(data_final)
    logger.info(f"Dropped {dropped_rows} rows with NaNs after all indicators. Final data shape: {data_final.shape}")

    # --- 7. Check data length against max_lag ---
    min_required_len_for_lag = max_lag + 3 # Need at least 3 points beyond the max lag for stable correlation
    if len(data_final) < min_required_len_for_lag:
        logger.error(f"Insufficient rows ({len(data_final)}) remaining after final dropna. Need at least {min_required_len_for_lag} for max_lag={max_lag}. Exiting.")
        sys.exit(1)

    logger.info(f"Data sample entering correlation calculation (data_final, first 5 rows):\n{data_final.head(5).to_string()}")

    # --- 7b. Calculate & Store Correlations ---
    try:
        conn = sqlite_manager.create_connection(str(db_path));
        if not conn: raise ConnectionError("Failed reconnect pre-correlation IDs.")
        # Re-fetch IDs just in case connection dropped (shouldn't happen here but safe)
        symbol_id = sqlite_manager._get_or_create_id(conn, 'symbols', 'symbol', symbol)
        timeframe_id = sqlite_manager._get_or_create_id(conn, 'timeframes', 'timeframe', timeframe)
    except Exception as db_id_err: logger.critical(f"Failed get DB IDs pre-correlation: {db_id_err}"); sys.exit(1)
    finally:
         if conn: conn.close()

    # Pass the list of config dicts that were *actually* processed (had IDs)
    corr_success = correlation_calculator.process_correlations(data_final, str(db_path), symbol_id, timeframe_id, indicator_configs_to_process, max_lag)
    if not corr_success: logger.error("Correlation phase failed. Exiting."); sys.exit(1)
    logger.info("Correlation calculation and storage complete.")

    # --- 8. Retrieve Correlations ---
    conn = sqlite_manager.create_connection(str(db_path)); report_data_ok = False
    if not conn: logger.error("Failed connect DB for fetching correlations."); sys.exit(1)
    try:
        config_ids_processed = [cfg['config_id'] for cfg in indicator_configs_to_process]
        logger.info(f"Fetching correlations for {len(config_ids_processed)} config IDs (up to lag {max_lag})...")
        correlations_by_config_id = sqlite_manager.fetch_correlations(conn, symbol_id, timeframe_id, config_ids_processed)
        if not correlations_by_config_id:
             logger.error("No correlation data structure returned from fetch.")
        else:
             actual_max_lag_in_data = 0
             for cfg_id, data_list in correlations_by_config_id.items():
                 if data_list:
                     actual_max_lag_in_data = max(actual_max_lag_in_data, len(data_list))

             # Adjust max_lag if DB data has less than requested, but only if it's positive
             if 0 < actual_max_lag_in_data < max_lag:
                 logger.warning(f"Max lag found in fetched data ({actual_max_lag_in_data}) is less than requested max_lag ({max_lag}). Using {actual_max_lag_in_data} for reporting.")
                 max_lag = actual_max_lag_in_data
             elif actual_max_lag_in_data == 0 and max_lag > 0:
                  logger.error("Fetched correlation data appears empty or has zero lag. Cannot proceed with reporting.")
                  max_lag = 0 # Prevent further processing if no valid lag data

             configs_with_data = 0
             if max_lag > 0:
                 configs_with_data = sum(1 for data_list in correlations_by_config_id.values() if data_list and len(data_list) >= max_lag and any(pd.notna(x) for x in data_list[:max_lag]))
                 if configs_with_data > 0:
                     logger.info(f"Successfully retrieved valid correlation data for {configs_with_data} configurations up to lag {max_lag}.")
                     report_data_ok = True
                 else:
                     logger.error(f"Retrieved structure, but ALL configurations lack valid correlation data up to lag {max_lag}.")

                 if 0 < configs_with_data < len(config_ids_processed):
                     logger.warning(f"Retrieved valid data for only {configs_with_data}/{len(config_ids_processed)} requested configurations.")
             else:
                 logger.error("Max lag is 0 after fetching data. Reporting cannot proceed.")

    except Exception as fetch_err: logger.error(f"Error fetching correlations: {fetch_err}", exc_info=True); correlations_by_config_id = {}
    finally: conn.close()

    if not report_data_ok: logger.error("Cannot generate visualizations - no valid correlation data retrieved or max_lag is zero."); sys.exit(1)

    # --- 9. Visualization ---
    file_prefix = f"{timestamp_str}_{symbol}_{timeframe}"; logger.info("Starting visualization generation...")
    try:
        vis_success_count = 0; vis_total_count = 4 # Assuming 4 visualization types

        logger.info("Generating separate line charts with CI..."); visualization_generator.plot_correlation_lines(correlations_by_config_id, indicator_configs_to_process, max_lag, config.LINE_CHARTS_DIR, file_prefix); vis_success_count +=1
        logger.info("Generating combined correlation chart..."); visualization_generator.generate_combined_correlation_chart(correlations_by_config_id, indicator_configs_to_process, max_lag, config.COMBINED_CHARTS_DIR, file_prefix); vis_success_count +=1
        logger.info("Generating enhanced heatmap..."); visualization_generator.generate_enhanced_heatmap(correlations_by_config_id, indicator_configs_to_process, max_lag, config.HEATMAPS_DIR, file_prefix, is_tweak_path); vis_success_count +=1
        logger.info("Generating correlation envelope chart..."); visualization_generator.generate_correlation_envelope_chart(correlations_by_config_id, indicator_configs_to_process, max_lag, config.REPORTS_DIR, file_prefix); vis_success_count +=1

        logger.info(f"Visualization generation finished ({vis_success_count}/{vis_total_count} types attempted).")
    except Exception as vis_err: logger.error(f"Error during visualization: {vis_err}", exc_info=True)

    # --- 10. Finalization ---
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
         if e.code != 0: logger.error(f"Analysis exited with code {e.code}.")
         else: logger.info("Analysis finished successfully.")
         sys.exit(e.code)
    except KeyboardInterrupt: logger.warning("Analysis interrupted by user (Ctrl+C)."); sys.exit(130)
    except Exception as e: logger.critical(f"An unhandled exception occurred: {e}", exc_info=True); sys.exit(1)