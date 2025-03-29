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
import parameter_optimizer
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
            logging_setup.setup_logging()
            logger.info(f"Re-initialized logging after cleanup. Continuing run: {timestamp_str}")
            logger.info("Re-checking leaderboard database initialization...")
            if not leaderboard_manager.initialize_leaderboard_db(): logger.error("Failed to initialize leaderboard database after cleanup. Updates will be skipped.")
            else: logger.info("Leaderboard database ok after cleanup.")
        else: print("Skipping cleanup."); logger.info("Skipping cleanup.")
    except Exception as e: logger.error(f"Error during cleanup prompt: {e}", exc_info=True); print("Error during cleanup, continuing...")

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

    data = data_manager.load_data(db_path)
    if data is None or data.empty: logger.error(f"Failed load/empty data from {db_path}. Exiting."); sys.exit(1)
    logger.info(f"Initial data loaded. Shape: {data.shape}. Checking length...")
    time_interval = utils.determine_time_interval(data) or "periods"; logger.info(f"Determined time interval: {time_interval}")
    data_daterange_str = "Unknown"
    try:
        min_date = data['date'].min().strftime('%Y%m%d')
        max_date = data['date'].max().strftime('%Y%m%d')
        data_daterange_str = f"{min_date}-{max_date}"
        logger.info(f"Dataset date range: {data_daterange_str}")
    except Exception as e: logger.warning(f"Could not determine dataset date range: {e}")

    max_possible_lag = len(data) - config.MIN_DATA_POINTS_FOR_LAG - 1
    if max_possible_lag <= 0: logger.critical(f"Insufficient data ({len(data)} rows). Need > {config.MIN_DATA_POINTS_FOR_LAG + 1} rows."); sys.exit(1)
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
            if not lag_input: max_lag = suggested_lag; print(f"Using suggested lag: {max_lag}")
            else: max_lag = int(lag_input)
            if max_lag <= 0: print("Max lag must be positive."); max_lag = 0
            elif max_lag > max_possible_lag: print(f"Input lag {max_lag} exceeds max possible ({max_possible_lag}). Using max possible."); max_lag = max_possible_lag
            else: print(f"Using user-defined lag: {max_lag}"); break
        except ValueError: print("Invalid input. Please enter a positive integer.")
        except Exception as e: logger.error(f"Error getting max_lag input: {e}", exc_info=True); print("An error occurred. Exiting."); sys.exit(1)

    logger.info(f"Using final max_lag = {max_lag} for calculations.")
    min_required_len_for_lag = max_lag + config.MIN_DATA_POINTS_FOR_LAG
    if len(data) < min_required_len_for_lag: logger.critical(f"Insufficient rows ({len(data)}). Need {min_required_len_for_lag} for max_lag={max_lag}. Exiting."); sys.exit(1)
    logger.info(f"Data length ({len(data)}) sufficient for chosen max_lag={max_lag} (required: {min_required_len_for_lag}).")

    conn = None; indicator_configs_to_process = []; is_tweak_path = False; selected_indicator_for_tweak = None
    analysis_path_successful = False
    try:
        conn = sqlite_manager.create_connection(str(db_path))
        if not conn: logger.critical("Failed connect main DB pre-config."); sys.exit(1)
        symbol_id = sqlite_manager._get_or_create_id(conn, 'symbols', 'symbol', symbol)
        timeframe_id = sqlite_manager._get_or_create_id(conn, 'timeframes', 'timeframe', timeframe)
        logger.debug(f"DB IDs - Symbol: {symbol_id}, Timeframe: {timeframe_id}")
        conn.close(); conn = None

        while True:
            print("\n--- Analysis Path ---")
            choice = input("Select path: [D]efault (All Indicators), [T]weak (Single Indicator, Optimized): ").strip().lower()
            if choice == 'd':
                logger.info("Processing Default Path: Preparing default configs for ALL defined indicators.")
                is_tweak_path = False
                temp_config_list = []
                conn = sqlite_manager.create_connection(str(db_path))
                if not conn: logger.critical("Failed reconnect DB for default config IDs."); break
                try:
                    for name, definition in indicator_definitions.items():
                        params = definition.get('parameters', {}); actual_defaults = {k: v.get('default') for k, v in params.items() if 'default' in v}
                        conditions = definition.get('conditions', [])
                        if parameter_generator.evaluate_conditions(actual_defaults, conditions):
                            try:
                                config_id = sqlite_manager.get_or_create_indicator_config_id(conn, name, actual_defaults)
                                temp_config_list.append({ 'indicator_name': name, 'params': actual_defaults, 'config_id': config_id })
                            except Exception as cfg_err: logger.error(f"Failed get/create config ID for default {name}: {cfg_err}")
                        else: logger.warning(f"Default params for '{name}' invalid based on conditions. Skipping.")
                finally:
                    if conn: conn.close(); conn = None
                estimated_outputs_per_config = 1.5; num_default_configs = len(temp_config_list)
                actual_total_correlations = num_default_configs * estimated_outputs_per_config * max_lag
                logger.info(f"Default path generated {num_default_configs} default configurations.")
                logger.info(f"Estimated total correlations: ~{int(actual_total_correlations)} (Target: <= {config.TARGET_MAX_CORRELATIONS})")
                if actual_total_correlations > config.TARGET_MAX_CORRELATIONS * 1.1 :
                     proceed_choice = input(f"WARNING: Approx {int(actual_total_correlations)} correlations estimated. Proceed anyway? [y/N]: ").strip().lower() or 'n'
                     if proceed_choice != 'y': logger.info("User aborted default path."); continue
                indicator_configs_to_process = temp_config_list
                analysis_path_successful = True; break
            elif choice == 't':
                logger.info("Processing Tweak Path: Selecting indicator for optimization.")
                is_tweak_path = True; available_indicators = sorted(list(indicator_definitions.keys()))
                print("\nAvailable Indicators to Optimize:"); [print(f"{i+1}. {n}") for i, n in enumerate(available_indicators)]
                selected_indicator_for_tweak = None
                while True:
                    try:
                        idx_str = input(f"Select indicator number (1-{len(available_indicators)}): ").strip(); idx = int(idx_str) - 1
                        if 0 <= idx < len(available_indicators): selected_indicator_for_tweak = available_indicators[idx]; logger.info(f"User selected indicator for optimization: '{selected_indicator_for_tweak}'"); break
                        else: print("Invalid selection.")
                    except ValueError: print("Invalid input.")
                definition = indicator_definitions[selected_indicator_for_tweak]
                print(f"\nOptimizing parameters for '{selected_indicator_for_tweak}'...")
                print(f"(Targeting top {config.HEATMAP_MAX_CONFIGS} over ~{config.OPTIMIZER_ITERATIONS} iterations, Lag={max_lag})")
                print(f"(Using scoring method: '{config.OPTIMIZER_SCORING_METHOD}')")
                required_for_opt = definition.get('required_inputs', []) + ['close']
                if not all(col in data.columns for col in required_for_opt): logger.critical(f"Base data is missing columns for optimizing {selected_indicator_for_tweak}: {required_for_opt}. Exiting."); sys.exit(1)
                optimizer_results = parameter_optimizer.optimize_parameters(
                    indicator_name=selected_indicator_for_tweak, indicator_def=definition, base_data_with_required=data.copy(),
                    max_lag=max_lag, num_iterations=config.OPTIMIZER_ITERATIONS, target_configs=config.HEATMAP_MAX_CONFIGS,
                    db_path=str(db_path), symbol_id=symbol_id, timeframe_id=timeframe_id, scoring_method=config.OPTIMIZER_SCORING_METHOD
                )
                indicator_configs_to_process = [{'indicator_name': r['indicator_name'], 'params': r['params'], 'config_id': r['config_id']} for r in optimizer_results]
                if not indicator_configs_to_process: logger.error(f"Optimization failed for {selected_indicator_for_tweak}."); print("Optimization did not yield results. Check logs."); continue
                else: logger.info(f"Optimization completed. Found {len(indicator_configs_to_process)} configs (incl. default) for {selected_indicator_for_tweak}."); analysis_path_successful = True; break
            else: print("Invalid choice. Please enter 'd' or 't'."); logger.warning(f"Invalid analysis path choice: '{choice}'")
    except Exception as prep_err: logger.critical(f"Error during analysis path preparation: {prep_err}", exc_info=True); sys.exit(1)
    finally:
        if conn: conn.close()

    if not analysis_path_successful or not indicator_configs_to_process: logger.error("No configurations prepared or analysis path failed. Exiting."); sys.exit(1)
    logger.info(f"Prepared {len(indicator_configs_to_process)} configurations for {'optimization' if is_tweak_path else 'default'} analysis using max_lag={max_lag}.")

    data_with_indicators = data.copy(); data_final = None
    if not is_tweak_path: # Default Path
        logger.info("Starting standard indicator computation for Default path...")
        standard_configs = []; custom_configs = []
        for cfg in indicator_configs_to_process:
            def_type = indicator_definitions.get(cfg['indicator_name'], {}).get('type')
            if def_type == 'custom': custom_configs.append(cfg)
            elif def_type: standard_configs.append(cfg)
            else: logger.warning(f"Config ID {cfg['config_id']} for '{cfg['indicator_name']}' has unknown type. Skipping.")
        logger.info(f"Processing {len(standard_configs)} standard configurations.")
        data_with_std_indicators = indicator_factory.compute_configured_indicators(data, standard_configs)
        logger.info("Standard indicator computation phase complete.")
        data_with_indicators = data_with_std_indicators
        if custom_configs:
            logger.info(f"Applying {len(custom_configs)} custom indicator configurations...")
            custom_names_to_run = {cfg['indicator_name'] for cfg in custom_configs}
            logger.warning(f"Custom indicator params might not be used by apply_all. Running based on names: {custom_names_to_run}")
            data_with_indicators = custom_indicators.apply_all_custom_indicators(data_with_std_indicators)
            logger.info("Custom indicator application phase complete.")
        else: logger.info("No custom indicators to apply for Default path.")
        logger.info("Calling correlation calculation for Default path...")
        logger.info(f"DataFrame shape before final dropna (Default Path): {data_with_indicators.shape}")
        data_final = data_with_indicators.dropna(how='any')
        logger.info(f"Dropped {len(data_with_indicators) - len(data_final)} rows. Final shape for Default Correlation: {data_final.shape}")
        if len(data_final) < min_required_len_for_lag: logger.error(f"Insufficient rows ({len(data_final)}) after dropna for Default Path. Need {min_required_len_for_lag} for max_lag={max_lag}. Exiting."); sys.exit(1)
        corr_success = correlation_calculator.process_correlations(data_final, str(db_path), symbol_id, timeframe_id, indicator_configs_to_process, max_lag)
        if not corr_success: logger.error("Default path correlation phase failed. Exiting."); sys.exit(1)
        logger.info("Default path correlation calculation and storage complete.")
    else: # Tweak Path
        logger.info("Skipping main correlation calculation for Tweak path (done in optimizer).")
        logger.info("Re-calculating indicators for reporting base (optional)...")
        reporting_configs = indicator_configs_to_process[:config.HEATMAP_MAX_CONFIGS]
        data_with_indicators = indicator_factory.compute_configured_indicators(data, reporting_configs)
        data_final = data_with_indicators # Keep NaNs
        logger.info(f"Skipping dropna for Tweak Path reporting base. Shape (with NaNs): {data_final.shape}")

    conn = sqlite_manager.create_connection(str(db_path)); report_data_ok = False
    correlations_by_config_id = {}
    if not conn: logger.error("Failed connect DB for fetching correlations."); sys.exit(1)
    try:
        config_ids_to_fetch = [cfg['config_id'] for cfg in indicator_configs_to_process]
        logger.info(f"Fetching final correlations from DB for {len(config_ids_to_fetch)} config IDs (up to lag {max_lag})...")
        correlations_by_config_id = sqlite_manager.fetch_correlations(conn, symbol_id, timeframe_id, config_ids_to_fetch)
        if not correlations_by_config_id: logger.error("No correlation data structure returned from DB fetch.")
        else:
             actual_max_lag_in_data = 0; configs_with_any_data = 0
             for cfg_id, data_list in correlations_by_config_id.items():
                  if data_list:
                      configs_with_any_data += 1
                      try:
                          valid_indices = [i for i, v in enumerate(data_list) if pd.notna(v)]
                          if valid_indices: actual_max_lag_in_data = max(actual_max_lag_in_data, max(valid_indices) + 1)
                      except ValueError: pass
             logger.info(f"Found correlation data for {configs_with_any_data} configurations in DB.")
             logger.info(f"Actual maximum lag found in fetched data: {actual_max_lag_in_data}")
             reporting_max_lag = min(max_lag, actual_max_lag_in_data) if actual_max_lag_in_data > 0 else 0
             if reporting_max_lag < max_lag: logger.warning(f"Reporting max lag adjusted from {max_lag} down to {reporting_max_lag} based on available data.")
             if reporting_max_lag <= 0: logger.error("No valid positive lag data found in DB. Reporting cannot proceed.")
             else:
                 max_lag = reporting_max_lag
                 configs_with_valid_data_at_lag = sum(1 for data_list in correlations_by_config_id.values() if data_list and len(data_list) >= max_lag and any(pd.notna(x) for x in data_list[:max_lag]))
                 if configs_with_valid_data_at_lag > 0: logger.info(f"Sufficient data found for {configs_with_valid_data_at_lag} configs up to adjusted lag {max_lag}."); report_data_ok = True
                 else: logger.error(f"ALL configurations lack valid correlation data up to adjusted lag {max_lag}.")
                 if 0 < configs_with_valid_data_at_lag < len(config_ids_to_fetch): logger.warning(f"Retrieved valid data for only {configs_with_valid_data_at_lag}/{len(config_ids_to_fetch)} requested configurations.")
    except Exception as fetch_err: logger.error(f"Error fetching correlations: {fetch_err}", exc_info=True)
    finally:
        if conn: conn.close()

    if not report_data_ok: logger.error("Cannot generate reports/visualizations - no valid correlation data retrieved or max_lag is zero."); sys.exit(1)

    logger.info("Attempting to update persistent leaderboard...")
    try:
        leaderboard_manager.update_leaderboard(
            current_run_correlations=correlations_by_config_id, indicator_configs=indicator_configs_to_process,
            max_lag=max_lag, symbol=symbol, timeframe=timeframe, data_daterange=data_daterange_str, source_db_name=db_path.name
        )
    except Exception as lb_err: logger.error(f"Failed to update leaderboard: {lb_err}", exc_info=True)

    file_prefix = f"{timestamp_str}_{symbol}_{timeframe}"
    try:
        logger.info("Generating peak correlation report...")
        visualization_generator.generate_peak_correlation_report(correlations_by_config_id, indicator_configs_to_process, max_lag, config.REPORTS_DIR, file_prefix)
    except Exception as report_err: logger.error(f"Error generating peak correlation report: {report_err}", exc_info=True)

    # --- Visualization Preparation ---
    configs_for_viz = indicator_configs_to_process
    correlations_for_viz = correlations_by_config_id
    # *Only* slice down the list for visualization if necessary (esp. for Default path)
    if len(indicator_configs_to_process) > config.HEATMAP_MAX_CONFIGS:
        logger.info(f"Limiting visualizations to top {config.HEATMAP_MAX_CONFIGS} configurations based on final fetch order (usually score).")
        configs_for_viz_sliced = indicator_configs_to_process[:config.HEATMAP_MAX_CONFIGS]
        # <<< FIX for Default Config in Viz >>>
        # Check if default config exists in the original full list
        default_config_details = None
        if is_tweak_path and indicator_configs_to_process:
            try:
                tweak_name = indicator_configs_to_process[0]['indicator_name']
                indicator_def = indicator_definitions.get(tweak_name)
                if indicator_def:
                    default_params = {k: v.get('default') for k, v in indicator_def.get('parameters', {}).items() if 'default' in v}
                    for cfg in indicator_configs_to_process: # Search full list
                        if utils.compare_param_dicts(cfg.get('params',{}), default_params):
                            default_config_details = cfg
                            break
            except Exception as e:
                logger.warning(f"Could not get default config details for viz check: {e}")

        # Ensure default is included in the sliced list if it existed and is tweak path
        if is_tweak_path and default_config_details:
            default_id = default_config_details.get('config_id')
            is_default_in_sliced = any(cfg['config_id'] == default_id for cfg in configs_for_viz_sliced)
            if not is_default_in_sliced:
                # Add default, potentially replacing the last one if list is already full
                if len(configs_for_viz_sliced) >= config.HEATMAP_MAX_CONFIGS:
                    logger.warning(f"Replacing last config in viz list to ensure default (ID: {default_id}) is included.")
                    configs_for_viz_sliced[-1] = default_config_details
                else:
                    configs_for_viz_sliced.append(default_config_details)
                # Re-sort might be needed if order matters beyond just inclusion
                # configs_for_viz_sliced.sort(...) # Add sorting if required

        configs_for_viz = configs_for_viz_sliced # Use the potentially modified sliced list
        # <<< END FIX >>>
        correlations_for_viz = {cfg['config_id']: correlations_by_config_id.get(cfg['config_id'], []) for cfg in configs_for_viz}

    logger.info("Starting visualization generation...")
    try:
        vis_success_count = 0; vis_total_count = 4
        logger.info("Generating separate line charts with CI..."); visualization_generator.plot_correlation_lines(correlations_for_viz, configs_for_viz, max_lag, config.LINE_CHARTS_DIR, file_prefix); vis_success_count +=1
        logger.info("Generating combined correlation chart..."); visualization_generator.generate_combined_correlation_chart(correlations_for_viz, configs_for_viz, max_lag, config.COMBINED_CHARTS_DIR, file_prefix); vis_success_count +=1
        logger.info("Generating enhanced heatmap..."); visualization_generator.generate_enhanced_heatmap(correlations_for_viz, configs_for_viz, max_lag, config.HEATMAPS_DIR, file_prefix, is_tweak_path); vis_success_count +=1
        logger.info("Generating correlation envelope chart..."); visualization_generator.generate_correlation_envelope_chart(correlations_for_viz, configs_for_viz, max_lag, config.REPORTS_DIR, file_prefix, is_tweak_path); vis_success_count +=1
        logger.info(f"Visualization generation finished ({vis_success_count}/{vis_total_count} types attempted).")
    except Exception as vis_err: logger.error(f"Error during visualization: {vis_err}", exc_info=True)

    end_time = datetime.now(timezone.utc); duration = end_time - start_time
    logger.info(f"--- Analysis Run Completed: {timestamp_str} ---")
    logger.info(f"Total execution time: {duration}")
    print(f"\nAnalysis complete using max_lag={max_lag}. Reports saved in '{config.REPORTS_DIR}'. Total time: {duration}")

    # <<< --- Export Leaderboard --- >>>
    try:
        # Use fixed filename "leaderboard.txt"
        leaderboard_manager.export_leaderboard_to_text()
    except Exception as export_err:
        logger.error(f"Failed to export leaderboard to text: {export_err}", exc_info=True)
    # <<< --- End Export --- >>>


if __name__ == "__main__":
    try:
        run_analysis()
        sys.exit(0)
    except SystemExit as e:
         if e.code != 0: logger.error(f"Analysis exited with code {e.code}.")
         else: logger.info("Analysis finished successfully.")
         sys.exit(e.code)
    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user (Ctrl+C).")
        print("\nAnalysis interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"An unhandled exception occurred in the main execution block: {e}", exc_info=True)
        print(f"\nAn critical error occurred: {e}")
        print("Please check the log file for details.")
        sys.exit(1)