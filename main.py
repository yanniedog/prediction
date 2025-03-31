# main.py

import logging
import logging_setup # Import the setup module

import sys
import json
from datetime import datetime, timezone, timedelta
import pandas as pd
import os
import shutil
from pathlib import Path
import numpy as np
import re
import math
import sqlite3
import time
from typing import Dict, List, Optional, Tuple, Any, Set

# Import project modules
import utils
import config # Import config first
import sqlite_manager
import data_manager
import parameter_generator
import parameter_optimizer
import indicator_factory
import correlation_calculator
import visualization_generator
import custom_indicators
import leaderboard_manager
import predictor
import backtester

# Logger initialization moved inside main execution block

# --- Configuration Constants (Fetched from config) ---
ETA_UPDATE_INTERVAL_SECONDS = config.DEFAULTS.get("eta_update_interval_seconds", 15)
INTERIM_REPORT_FREQUENCY = config.DEFAULTS.get("interim_report_frequency", 10)
TOTAL_ANALYSIS_STEPS = config.DEFAULTS.get("total_analysis_steps_estimate", 10) # Adjust this based on workload
DEFAULT_ANALYSIS_PATH = config.DEFAULTS.get("default_analysis_path", 'tweak') # 'tweak' (Bayesian) or 'classical'

# --- Global State for Timing and Progress ---
# These will be managed within run_analysis() context
_confirmed_analysis_start_time: Optional[float] = None
_last_periodic_report_time: Optional[float] = None
_periodic_report_interval_seconds: int = 30 # How often to generate tally report

# --- Phase Functions ---

def _setup_and_select_mode(timestamp_str: str) -> Optional[str]:
    """Handles initial setup, cleanup prompt, and mode selection."""
    logger = logging.getLogger(__name__)
    logger.info("Initializing leaderboard database (pre-cleanup)...")
    if not leaderboard_manager.initialize_leaderboard_db():
        logger.error("Failed initialize leaderboard DB.")
        print("\nERROR: Could not initialize leaderboard database. Exiting.")
        sys.exit(1)

    try:
        cleanup_choice = input("Delete previous analysis content? (Reports, Logs, Leaderboard DB) [Y/n]: ").strip().lower() # Clarified prompt
        if cleanup_choice == '' or cleanup_choice == 'y':
            print("Proceeding with cleanup (Reports, Logs, Leaderboard DB)...");
            handlers = logging.getLogger().handlers[:]
            for handler in handlers: handler.close(); logging.getLogger().removeHandler(handler)
            logging.shutdown()

            # --- Define files to KEEP during cleanup ---
            # Exclude only essential config/source files, NOT the leaderboard DB
            files_to_keep_during_cleanup = [
                config.INDICATOR_PARAMS_PATH.name,
                '.gitignore',
                # Add any other critical config files here if needed
            ]
            logger.info(f"Cleanup selected. Keeping: {files_to_keep_during_cleanup}")
            # Call cleanup, explicitly passing the files to keep (overriding the default)
            # clean_db=False ensures user data DBs aren't touched unless explicitly requested later
            utils.cleanup_previous_content(
                clean_reports=True,
                clean_logs=True,
                clean_db=False, # Keep user data DBs safe by default
                exclude_files=files_to_keep_during_cleanup
            )
            # ---------------------------------------------

            logging_setup.setup_logging(); logger = logging.getLogger(__name__)
            logger.info(f"Re-initialized logging after cleanup: {timestamp_str}")
            # Re-initialize leaderboard DB (it was just deleted)
            if not leaderboard_manager.initialize_leaderboard_db():
                logger.critical("Failed re-initialize leaderboard DB after cleanup!")
                print("\nCRITICAL ERROR: Could not re-initialize leaderboard DB. Exiting.")
                sys.exit(1)
            logger.info("Leaderboard DB re-initialized after cleanup.")
        else:
            print("Skipping cleanup."); logger.info("Skipping cleanup.")
    except Exception as e:
        logger.error(f"Cleanup error: {e}", exc_info=True)
        print("\nERROR during cleanup process. Exiting.")
        sys.exit(1)

    # Mode Selection
    while True:
        print("\n--- Mode Selection ---")
        print("[A]nalysis: Run full analysis (Bayesian/Classical Path)")
        print("[C]ustom:   Run reports/predictions on existing DB data")
        print("[B]acktest Check: Run historical predictor check (uses final leaderboard)")
        print("[Q]uit")
        mode_choice = input("Select Mode [A/c/b/q]: ").strip().lower() or 'a'
        if mode_choice in ['a', 'c', 'b']:
            logger.info(f"Mode selected: {mode_choice}")
            return mode_choice
        elif mode_choice == 'q':
            logger.info("User quit at mode selection.")
            return None
        else:
            print("Invalid mode selection.")

# Custom/Backtest modes remain largely unchanged in their core logic
def _run_custom_mode(timestamp_str: str):
    """Runs actions on an existing database without recalculating."""
    logger = logging.getLogger(__name__); logger.info("--- Entering Custom Mode ---"); print("\n--- Custom Mode ---")
    db_path_custom = data_manager.select_existing_database()
    if not db_path_custom: print("No database selected for Custom Mode."); return
    logger.info(f"Custom Mode using DB: {db_path_custom.name}")
    try:
        base_name = db_path_custom.stem; match = re.match(r'^([A-Z0-9]+)[_-]([a-zA-Z0-9]+)$', base_name, re.IGNORECASE)
        if not match: raise ValueError("Filename does not match SYMBOL_TIMEFRAME format.")
        symbol_custom, timeframe_custom = match.groups(); symbol_custom = symbol_custom.upper()
        logger.info(f"Parsed DB: Symbol={symbol_custom}, Timeframe={timeframe_custom}")
    except Exception as e: logger.error(f"Cannot parse DB name '{db_path_custom.name}': {e}"); print(f"Error: Invalid DB filename format."); return

    conn_custom = None; max_lag_custom = None; configs_custom = []; corrs_custom = {}; custom_data_ok = False; symbol_id_custom = -1; timeframe_id_custom = -1
    try:
        conn_custom = sqlite_manager.create_connection(str(db_path_custom)); assert conn_custom, "Failed connect custom DB."
        cursor = conn_custom.cursor()
        cursor.execute("SELECT id FROM symbols WHERE LOWER(symbol) = ?", (symbol_custom.lower(),)); res_sym = cursor.fetchone(); symbol_id_custom = res_sym[0] if res_sym else None
        cursor.execute("SELECT id FROM timeframes WHERE LOWER(timeframe) = ?", (timeframe_custom.lower(),)); res_tf = cursor.fetchone(); timeframe_id_custom = res_tf[0] if res_tf else None
        assert symbol_id_custom is not None and timeframe_id_custom is not None, "Symbol/Timeframe ID not found in DB."
        logger.debug(f"Custom Mode DB IDs: Sym={symbol_id_custom}, TF={timeframe_id_custom}")
        max_lag_custom = sqlite_manager.get_max_lag_for_pair(conn_custom, symbol_id_custom, timeframe_id_custom); assert max_lag_custom and max_lag_custom > 0, f"No correlation data found for {symbol_custom}/{timeframe_custom}."
        config_ids_custom = sqlite_manager.get_distinct_config_ids_for_pair(conn_custom, symbol_id_custom, timeframe_id_custom); assert config_ids_custom, "No distinct config IDs found."
        configs_custom = sqlite_manager.get_indicator_configs_by_ids(conn_custom, config_ids_custom); assert configs_custom, "Failed fetch config details."
        # Fetch correlations once for all custom actions
        corrs_custom = sqlite_manager.fetch_correlations(conn_custom, symbol_id_custom, timeframe_id_custom, config_ids_custom); assert corrs_custom, "Failed fetch correlations."
        max_lag_fetched = max((len(v) for v in corrs_custom.values() if v), default=0)
        if max_lag_fetched < max_lag_custom: logger.warning(f"Fetched lag ({max_lag_fetched}) < DB max lag ({max_lag_custom}). Using {max_lag_fetched}."); max_lag_custom = max_lag_fetched
        if max_lag_custom <= 0: raise ValueError("No valid lag data loaded.")
        custom_data_ok = True
    except Exception as e: logger.error(f"Error preparing Custom Mode data: {e}", exc_info=True); print(f"\nError loading data from '{db_path_custom.name}'.")
    finally:
        if conn_custom: conn_custom.close()
    if not custom_data_ok: print("Cannot proceed with Custom Mode."); return

    while True:
        print("\n--- Custom Mode Actions ---"); print(f"Data Source: {symbol_custom} ({timeframe_custom}), Max Lag: {max_lag_custom}, Configs: {len(configs_custom)}")
        print("\n[P]redict\n[V]isualize\n[R]eports\n[A]ll\n[Q]uit Custom Mode")
        custom_action = input("Select Action [P/v/r/a/q]: ").strip().lower() or 'p'
        run_predict = custom_action in ['p', 'a']; run_visualize = custom_action in ['v', 'a']; run_reports = custom_action in ['r', 'a']
        if custom_action == 'q': logger.info("Exiting Custom Mode."); break
        if not run_predict and not run_visualize and not run_reports: print("Invalid selection."); continue
        custom_file_prefix = f"{timestamp_str}_{symbol_custom}_{timeframe_custom}_CUSTOM"
        if run_reports:
             print("\n--- Custom Reports ---"); logger.info(f"Generating custom reports: {custom_file_prefix}")
             try:
                 # Pass fetched correlation_data directly
                 utils.run_interim_reports(db_path_custom, symbol_id_custom, timeframe_id_custom, configs_custom, max_lag_custom, custom_file_prefix, "Custom", correlation_data=corrs_custom)
                 print("Custom reports generated.")
             except Exception as report_err: logger.error(f"Custom report error: {report_err}", exc_info=True); print("\nError generating custom reports.")
        if run_predict:
             print("\n--- Custom Prediction ---")
             try: predictor.predict_price(db_path_custom, symbol_custom, timeframe_custom, max_lag_custom)
             except Exception as pred_err: logger.error(f"Custom prediction error: {pred_err}", exc_info=True); print("\nError running prediction.")
        if run_visualize:
             print("\n--- Custom Visualization ---"); logger.info(f"Generating custom visualizations: {custom_file_prefix}")
             try:
                 # Pass fetched correlation_data directly to avoid re-read
                 visualization_generator.generate_peak_correlation_report(corrs_custom, configs_custom, max_lag_custom, config.REPORTS_DIR, custom_file_prefix)
                 limit_viz = config.DEFAULTS.get("heatmap_max_configs", 50); configs_limited, corrs_limited = configs_custom, corrs_custom
                 if len(configs_custom) > limit_viz:
                     logger.info(f"Limiting custom visualizations to top {limit_viz} configs.")
                     try:
                         perf_data = [{'config_id': cfg_id, 'peak_abs': np.nanmax(np.abs(np.array(corrs[:max_lag_custom], dtype=float)))} for cfg_id, corrs in corrs_custom.items() if corrs and len(corrs) >= max_lag_custom]
                         perf_data = [p for p in perf_data if pd.notna(p['peak_abs'])]
                         perf_data.sort(key=lambda x: x['peak_abs'], reverse=True)
                         if perf_data:
                             top_ids = {item['config_id'] for item in perf_data[:limit_viz]}
                             configs_limited = [cfg for cfg in configs_custom if cfg.get('config_id') in top_ids]
                             corrs_limited = {cfg_id: corrs for cfg_id, corrs in corrs_custom.items() if cfg_id in top_ids}
                     except Exception as filter_err:
                         logger.error(f"Error filtering viz: {filter_err}. Using full set.")
                         # Implicitly uses full set if error occurs or perf_data is empty
                 if configs_limited: # Plot using limited (or full if not filtered) set
                     visualization_generator.plot_correlation_lines(corrs_limited, configs_limited, max_lag_custom, config.LINE_CHARTS_DIR, custom_file_prefix)
                     visualization_generator.generate_combined_correlation_chart(corrs_limited, configs_limited, max_lag_custom, config.COMBINED_CHARTS_DIR, custom_file_prefix)
                     visualization_generator.generate_enhanced_heatmap(corrs_limited, configs_limited, max_lag_custom, config.HEATMAPS_DIR, custom_file_prefix)
                 if corrs_custom: # Envelope always uses full data
                      visualization_generator.generate_correlation_envelope_chart(corrs_custom, configs_custom, max_lag_custom, config.REPORTS_DIR, custom_file_prefix)
                 print(f"Custom visualizations potentially saved.")
             except Exception as vis_err: logger.error(f"Custom visualization error: {vis_err}", exc_info=True); print("\nError generating visualizations.")
        # Exit after one action set as before
        if custom_action != 'q':
            break


def _run_backtest_check_mode(timestamp_str: str):
    """Runs the historical predictor check (simplified backtest)."""
    logger = logging.getLogger(__name__); logger.info("--- Entering Historical Predictor Check Mode ---"); print("\n--- Historical Predictor Check ---");
    print("WARNING: This mode uses the FINAL leaderboard, introducing LOOKAHEAD BIAS."); print("         Results DO NOT represent realistic trading performance.")
    db_path_bt = data_manager.select_existing_database();
    if not db_path_bt: print("No database selected for Backtest Check."); return
    logger.info(f"Historical Check Mode using DB: {db_path_bt.name}")
    try:
        base_name = db_path_bt.stem; match = re.match(r'^([A-Z0-9]+)[_-]([a-zA-Z0-9]+)$', base_name, re.IGNORECASE);
        if not match: raise ValueError("Filename format error.")
        symbol_bt, timeframe_bt = match.groups(); symbol_bt = symbol_bt.upper()
        logger.info(f"Parsed DB: Symbol={symbol_bt}, Timeframe={timeframe_bt}")
    except Exception as e: logger.error(f"Cannot parse BT DB name '{db_path_bt.name}': {e}"); print(f"Error: Invalid DB filename format."); return
    if not data_manager.validate_data(db_path_bt): print(f"Selected DB '{db_path_bt.name}' empty/invalid."); return
    max_lag_bt = 0; num_points_bt = 0
    while max_lag_bt <= 0:
        try: default_lag_bt = config.DEFAULTS.get("max_lag", 7); max_lag_bt = int(input(f"Enter max lag to check [default: {default_lag_bt}]: ").strip() or default_lag_bt)
        except ValueError: print("Invalid input."); max_lag_bt = 0
        if max_lag_bt <= 0: print("Max lag must be positive.")
    while num_points_bt <= 0:
        try: default_points_bt = config.DEFAULTS.get("backtester_default_points", 50); num_points_bt = int(input(f"Enter points per lag [default: {default_points_bt}]: ").strip() or default_points_bt)
        except ValueError: print("Invalid input."); num_points_bt = 0
        if num_points_bt <= 0: print("Points must be positive.")
    print(f"\nStarting historical check for {symbol_bt} ({timeframe_bt}), Lag={max_lag_bt}, Points={num_points_bt}...")
    try: backtester.run_backtest(db_path_bt, symbol_bt, timeframe_bt, max_lag_bt, num_points_bt)
    except Exception as bt_err: logger.error(f"Historical check run failed: {bt_err}", exc_info=True); print("\nError during historical check.")
    logger.info("Finished historical check.")


def _select_data_source_and_lag() -> Tuple[Path, str, str, pd.DataFrame, int, int, int, str]:
    """Handles data source selection/download and max lag input."""
    # ... (logic remains the same) ...
    logger = logging.getLogger(__name__); data_source_info = data_manager.manage_data_source();
    if data_source_info is None: logger.info("Exiting: No data source."); sys.exit(0)
    db_path, symbol, timeframe = data_source_info; logger.info(f"Using data source: {db_path.name}")
    if not (db_path and symbol and timeframe):
        logger.critical("Data source info invalid.")
        raise ValueError("Data source information missing after selection.")
    data = data_manager.load_data(db_path)
    if data is None or data.empty:
        logger.critical(f"Failed load data {db_path}.")
        raise ValueError(f"Failed to load data from {db_path}")
    if not ('date' in data.columns and 'close' in data.columns):
        logger.critical("Data missing required 'date' or 'close' columns.")
        raise ValueError("Dataframe must contain 'date' and 'close' columns.")
    try:
        min_date_dt=data['date'].min(); max_date_dt=data['date'].max()
        if pd.isna(min_date_dt) or pd.isna(max_date_dt):
             raise ValueError("Invalid min/max dates in data.")
        if min_date_dt.tzinfo is None: min_date_dt = min_date_dt.tz_localize('UTC')
        if max_date_dt.tzinfo is None: max_date_dt = max_date_dt.tz_localize('UTC')
        data_daterange_str = f"{min_date_dt.strftime('%Y%m%d')}-{max_date_dt.strftime('%Y%m%d')}"
        if min_date_dt < data_manager.EARLIEST_VALID_DATE:
            raise ValueError(f"Data starts too early ({min_date_dt}). Minimum allowed is {data_manager.EARLIEST_VALID_DATE}.")
    except Exception as e: logger.critical(f"Invalid date range: {e}", exc_info=True); sys.exit(1)
    logger.info(f"Data date range: {data_daterange_str}")
    min_points_needed = max(config.DEFAULTS["min_data_points_for_lag"], config.DEFAULTS["min_regression_points"])
    estimated_nan_rows = min(100, int(len(data)*0.05)); effective_data_len = max(0, len(data)-estimated_nan_rows)
    max_possible_lag = max(0, effective_data_len - min_points_needed - 1)
    if max_possible_lag <= 0: logger.critical(f"Insufficient data ({len(data)} rows)."); print(f"\nERROR: Insufficient data points ({effective_data_len} effective) for analysis."); sys.exit(1)
    default_lag = config.DEFAULTS.get("max_lag", 7); suggested_lag = min(max_possible_lag, max(30, int(effective_data_len*0.1)), 500); suggested_lag = min(suggested_lag, default_lag)
    max_lag = 0
    while max_lag <= 0:
        try:
            lag_input = input(f"\nEnter max correlation lag ({timeframe}) [Suggest: {suggested_lag}, Max: {max_possible_lag}]: ").strip(); max_lag = suggested_lag if not lag_input else int(lag_input)
            if max_lag <= 0: print("Lag must be positive."); max_lag=0; continue
            if max_lag > max_possible_lag:
                print(f"Warning: Requested lag {max_lag} exceeds maximum possible ({max_possible_lag}) based on data length and minimum points needed.")
                if input("Continue anyway? [y/N]: ").strip().lower() != 'y':
                    max_lag=0;
                    continue
            elif max_lag > suggested_lag: print(f"Note: Lag {max_lag} > Suggested {suggested_lag}.")
            print(f"Using Max Lag = {max_lag}"); break
        except ValueError: print("Invalid input. Please enter a number."); max_lag=0
    if effective_data_len < (max_lag + min_points_needed): logger.critical(f"Insufficient effective data length ({effective_data_len}) for lag {max_lag} and min points {min_points_needed}."); print(f"\nERROR: Not enough data rows ({effective_data_len} effective) to support the chosen lag {max_lag}."); sys.exit(1)
    conn = None; symbol_id = -1; timeframe_id = -1
    try:
        conn = sqlite_manager.create_connection(str(db_path))
        if not conn: raise ConnectionError("DB connect fail.")
        conn.execute("BEGIN;"); symbol_id = sqlite_manager._get_or_create_id(conn, 'symbols', 'symbol', symbol); timeframe_id = sqlite_manager._get_or_create_id(conn, 'timeframes', 'timeframe', timeframe); conn.commit()
    except Exception as id_err:
        logger.critical(f"Failed get/create DB IDs: {id_err}", exc_info=True)
        if conn:
            try: conn.rollback(); logger.warning("Rolled back DB transaction due to ID error.")
            except Exception as rb_err: logger.error(f"Rollback attempt failed: {rb_err}")
        print("\nCRITICAL ERROR: Could not obtain database IDs for symbol/timeframe. Exiting."); sys.exit(1)
    finally:
        if conn: conn.close()
    logger.info(f"Using DB IDs - Symbol: {symbol_id}, Timeframe: {timeframe_id}")
    return db_path, symbol, timeframe, data, max_lag, symbol_id, timeframe_id, data_daterange_str


def _prepare_configurations(
    _display_progress_func: callable, # <-- Pass the progress function
    current_step: int,
    db_path: Path, symbol: str, timeframe: str, max_lag: int, data: pd.DataFrame,
    indicator_definitions: Dict, symbol_id: int, timeframe_id: int, data_daterange_str: str,
    timestamp_str: str,
    # --- New args for global timing ---
    analysis_start_time_global: float,
    total_analysis_steps_global: int
) -> Tuple[List[Dict], bool, int]:
    """Handles analysis path selection (Bayesian default) and configuration generation/optimization."""
    global _confirmed_analysis_start_time, _last_periodic_report_time # Use global for start time confirmation

    logger = logging.getLogger(__name__)
    indicator_configs_to_process: List[Dict] = []; is_bayesian_path = False; analysis_path_successful = False
    interim_correlations_accumulator: Dict[int, List[Optional[float]]] = {}

    if not indicator_definitions: logger.critical("Indicator definitions missing/empty."); sys.exit(1)

    try:
        while not analysis_path_successful:
            print("\n--- Analysis Path ---"); print("[B]ayesian Optimization (Default)"); print("[C]lassical")
            default_prompt = "B" if DEFAULT_ANALYSIS_PATH == 'tweak' else "C"; choice = input(f"Select Path [{default_prompt}/c]: ").strip().lower()
            if not choice: choice = 'b' if DEFAULT_ANALYSIS_PATH == 'tweak' else 'c'

            if choice == 'c': # Classical Path
                 logger.info("Processing Classical Path..."); is_bayesian_path = False; temp_config_list = []
                 # -- Classical Config Gen (no major timing changes needed here, relatively fast) --
                 conn = sqlite_manager.create_connection(str(db_path))
                 if not conn: raise ConnectionError("Failed DB connect Classical.")
                 try:
                     gen_start_time = time.time(); all_defs = list(indicator_definitions.items()); total_defs = len(all_defs)
                     print("Generating Classical configurations...");
                     for i, (name, definition) in enumerate(all_defs):
                         print(f"\rGenerating Classical Configs: {i+1}/{total_defs} ({name[:20]}...)      ", end="")
                         if not isinstance(definition, dict): logger.warning(f"Skip invalid def '{name}'"); continue
                         generated_configs = parameter_generator.generate_configurations(definition)
                         for params in generated_configs:
                             try: config_id = sqlite_manager.get_or_create_indicator_config_id(conn, name, params); temp_config_list.append({'indicator_name': name, 'params': params, 'config_id': config_id})
                             except Exception as cfg_err: logger.error(f"Failed get/create ID classical '{name}' {params}: {cfg_err}")
                     print(); logger.info(f"Classical Path: Config gen took: {utils.format_duration(timedelta(seconds=time.time() - gen_start_time))}")
                 finally:
                     if conn: conn.close()
                 num_configs = len(temp_config_list)
                 if num_configs == 0: print("\nError: No configs generated Classical Path."); continue
                 estimated_duration = utils.estimate_duration(num_configs, max_lag, 'classical')
                 logger.info(f"Classical path: {num_configs} configs."); print(f"\nEstimate (Classical): ~{utils.format_duration(estimated_duration)}")
                 # --- Confirmation Point ---
                 if input("Proceed? [Y/n]: ").strip().lower() == 'n': continue
                 _confirmed_analysis_start_time = time.time() # *** START GLOBAL TIMER HERE ***
                 _last_periodic_report_time = _confirmed_analysis_start_time # Init periodic timer
                 # --------------------------
                 indicator_configs_to_process = temp_config_list; analysis_path_successful = True; break

            elif choice == 'b': # Bayesian Path
                 if not parameter_optimizer.SKOPT_AVAILABLE: print("\nError: scikit-optimize needed for Bayesian path. Install with: pip install scikit-optimize"); continue
                 is_bayesian_path = True; logger.info("Processing Bayesian Path...")
                 if not indicator_definitions: logger.error("Indicator defs empty Bayesian."); continue
                 available_indicators = sorted([
                     n for n, d in indicator_definitions.items()
                     if isinstance(d, dict) and 'parameters' in d and isinstance(d['parameters'], dict) and
                     any(isinstance(v, dict) and 'min' in v and 'max' in v and isinstance(v.get('default'), (int, float))
                         for p, v in d['parameters'].items())
                 ])
                 if not available_indicators: print("\nError: No indicators with tunable parameters (min/max/default) found."); continue

                 indicators_to_optimize = []
                 # -- Scope Selection (remains same) --
                 while True: # Inner loop for scope selection
                     print("\n--- Bayesian Scope ---"); print("[A]ll tunable indicators (default)\n[S]elect single indicator\n[E]xclude specific indicators\n[B]ack to Path Selection")
                     scope_choice = input("Select Scope [A/s/e/b]: ").strip().lower() or 'a'
                     if scope_choice == 's':
                          print("\nAvailable Tunable Indicators:"); [print(f" {i+1}. {n}") for i, n in enumerate(available_indicators)];
                          try:
                              idx_input = input(f"Select number (1-{len(available_indicators)}): ").strip()
                              if not idx_input.isdigit(): raise ValueError("Input must be a number.")
                              idx = int(idx_input) - 1
                              if not (0 <= idx < len(available_indicators)): raise IndexError("Selection out of range.")
                              indicators_to_optimize = [available_indicators[idx]]
                          except (ValueError, IndexError) as sel_err:
                              print(f"Invalid selection: {sel_err}. Please try again."); continue
                          logger.info(f"Optimizing SINGLE indicator: '{indicators_to_optimize[0]}'"); break
                     elif scope_choice == 'a':
                         indicators_to_optimize = available_indicators; logger.info(f"Optimizing ALL {len(indicators_to_optimize)} tunable indicators."); break
                     elif scope_choice == 'e':
                          print("\nAvailable Tunable Indicators:"); [print(f" {i+1}. {n}") for i, n in enumerate(available_indicators)]
                          exclude_input = input("Enter number(s) (comma-separated) to EXCLUDE: ").strip()
                          try:
                              if not exclude_input: excluded_indices = set()
                              else: excluded_indices = {int(x.strip()) - 1 for x in exclude_input.split(',') if x.strip().isdigit()}
                              indicators_to_optimize = [ind for i, ind in enumerate(available_indicators) if i not in excluded_indices and 0 <= i < len(available_indicators)]
                              if not indicators_to_optimize: print("Error: Excluding these leaves no indicators to optimize."); continue
                              if len(excluded_indices) > len(available_indicators): print("Warning: Some exclusion numbers were invalid/out of range.")
                              valid_exclusions = {i+1 for i in excluded_indices if 0 <= i < len(available_indicators)}
                              logger.info(f"Optimizing {len(indicators_to_optimize)} indicators (Excluded #{valid_exclusions if valid_exclusions else 'None'})."); break
                          except ValueError:
                              print("Invalid format. Enter numbers separated by commas (e.g., 1, 3, 5)."); continue
                     elif scope_choice == 'b':
                         indicators_to_optimize = []
                         break
                     else: print("Invalid selection.")

                 if not indicators_to_optimize: continue

                 num_indicators_to_opt = len(indicators_to_optimize); estimated_duration = utils.estimate_duration(num_indicators_to_opt, max_lag, 'tweak')
                 print(f"\nEstimate (Bayesian - {num_indicators_to_opt} indicator(s)): ~{utils.format_duration(estimated_duration)}")
                 # --- Confirmation Point ---
                 if input("Proceed? [Y/n]: ").strip().lower() == 'n': continue
                 _confirmed_analysis_start_time = time.time() # *** START GLOBAL TIMER HERE ***
                 _last_periodic_report_time = _confirmed_analysis_start_time # Init periodic timer
                 # --------------------------

                 all_evaluated_configs_aggregated = []; any_opt_failed = False; all_indicators_optimized_results = {}
                 opt_n_calls = config.DEFAULTS["optimizer_n_calls"]; opt_n_initial = config.DEFAULTS["optimizer_n_initial_points"]
                 print(f"\nStarting Bayesian Optimization ({opt_n_calls} evals per indicator/lag)..."); interim_correlations_accumulator.clear()

                 # Calculate approx steps for this phase
                 config_prep_step_start = current_step
                 config_prep_step_end = current_step + 3 # Allocate ~3 steps for config prep/opt

                 for i, ind_name_opt in enumerate(indicators_to_optimize):
                     # --- Calculate dynamic progress step based on overall analysis ---
                     fraction_done_opt = (i + 0.5) / num_indicators_to_opt # Mid-point progress through indicators
                     current_overall_step = config_prep_step_start + (config_prep_step_end - config_prep_step_start) * fraction_done_opt
                     _display_progress_func(f"Optimizing {ind_name_opt} [{i+1}/{num_indicators_to_opt}]", current_overall_step, total_analysis_steps_global)
                     # -----------------------------------------------------------------

                     logger.info(f"--- Starting Opt: {ind_name_opt} ({i+1}/{num_indicators_to_opt}) ---")
                     definition = indicator_definitions.get(ind_name_opt)
                     req_inputs = definition.get('required_inputs', []) if isinstance(definition, dict) else []
                     if not isinstance(definition, dict) or not all(col in data.columns for col in req_inputs + ['close']):
                         logger.error(f"Missing definition or required columns '{req_inputs + ['close']}' for indicator '{ind_name_opt}'. Skipping optimization."); any_opt_failed=True; continue

                     parameter_optimizer.indicator_series_cache.clear(); parameter_optimizer.single_correlation_cache.clear()
                     try:
                         # --- Pass global timing info down ---
                         best_res_log, eval_configs_ind = parameter_optimizer.optimize_parameters_bayesian_per_lag(
                             indicator_name=ind_name_opt, indicator_def=definition,
                             base_data_with_required=data.copy(), max_lag=max_lag,
                             n_calls_per_lag=opt_n_calls, n_initial_points_per_lag=opt_n_initial,
                             db_path=str(db_path), symbol_id=symbol_id, timeframe_id=timeframe_id,
                             symbol=symbol, timeframe=timeframe, data_daterange=data_daterange_str, # Passed correct variable
                             source_db_name=db_path.name,
                             interim_correlations_accumulator=interim_correlations_accumulator,
                             # --- New Args ---
                             analysis_start_time_global=_confirmed_analysis_start_time, # Pass confirmed start
                             total_analysis_steps_global=total_analysis_steps_global,
                             current_step_base = config_prep_step_start, # Pass the base step for this phase
                             total_steps_in_phase = config_prep_step_end - config_prep_step_start, # Pass steps allocated
                             indicator_index = i, # Pass current indicator index
                             total_indicators_in_phase = num_indicators_to_opt, # Pass total indicators
                             display_progress_func=_display_progress_func # Pass the main display func
                         )
                         # ------------------------------------

                         if not eval_configs_ind: logger.error(f"Optimization FAILED for {ind_name_opt} (returned no evaluated configs)."); any_opt_failed = True
                         else: all_evaluated_configs_aggregated.extend(eval_configs_ind); all_indicators_optimized_results[ind_name_opt] = best_res_log

                         # --- Interim Reporting Logic (Stays Similar) ---
                         if (i + 1) % INTERIM_REPORT_FREQUENCY == 0 or (i + 1) == num_indicators_to_opt:
                             # Recalculate overall step for report stage display
                              report_frac_done = (i + 1) / num_indicators_to_opt
                              report_overall_step = config_prep_step_start + (config_prep_step_end - config_prep_step_start) * report_frac_done
                              _display_progress_func(f"Interim Report {i+1}/{num_indicators_to_opt}", report_overall_step, total_analysis_steps_global)

                              evaluated_ids_in_accumulator = set(interim_correlations_accumulator.keys()); seen_report_ids = set(); current_unique_configs_interim = []
                              for cfg_eval in all_evaluated_configs_aggregated:
                                  cfg_id_eval = cfg_eval.get('config_id');
                                  if cfg_id_eval is not None and cfg_id_eval in evaluated_ids_in_accumulator and cfg_id_eval not in seen_report_ids:
                                      current_unique_configs_interim.append(cfg_eval); seen_report_ids.add(cfg_id_eval)
                              if not current_unique_configs_interim: logger.warning(f"No unique configs found for interim report after '{ind_name_opt}'."); continue

                              interim_file_prefix = f"{timestamp_str}_{symbol}_{timeframe}_BAYESIAN_INTERIM_{i+1}"
                              stage_name = f"Interim_Post_{ind_name_opt}"; logger.info(f"Running {stage_name} Reports for {len(current_unique_configs_interim)} unique configs...")
                              # Pass accumulator explicitly to avoid re-fetch
                              utils.run_interim_reports(db_path, symbol_id, timeframe_id, current_unique_configs_interim, max_lag, interim_file_prefix, stage_name, correlation_data=interim_correlations_accumulator)
                              # Export leaderboard text (already frequent) and tally report
                              try: leaderboard_manager.generate_leading_indicator_report()
                              except Exception as lte: logger.error(f"Error generating interim tally report: {lte}", exc_info=True)
                              _last_periodic_report_time = time.time() # Reset timer after reports

                     except Exception as opt_err: logger.error(f"Optimization loop failed for indicator '{ind_name_opt}': {opt_err}", exc_info=True); any_opt_failed = True; print(f"\nERROR during optimization of {ind_name_opt}. See log.")

                     # --- Periodic Tally Report Generation (if interval passed) ---
                     now = time.time()
                     if _last_periodic_report_time is not None and (now - _last_periodic_report_time > _periodic_report_interval_seconds):
                          try:
                              # Use current progress step for display context
                              periodic_overall_step = config_prep_step_start + (config_prep_step_end - config_prep_step_start) * ((i + 0.8) / num_indicators_to_opt) # Estimate slightly ahead
                              _display_progress_func(f"Periodic Tally Report", periodic_overall_step, total_analysis_steps_global)
                              logger.info("Generating periodic leading indicator tally report...")
                              leaderboard_manager.generate_leading_indicator_report()
                              _last_periodic_report_time = now
                          except Exception as tally_err:
                              logger.error(f"Error generating periodic tally report: {tally_err}", exc_info=True)
                 # --- End Bayesian Indicator Loop ---

                 if not all_evaluated_configs_aggregated: print("\nOptimization completed, but no configurations were evaluated."); continue

                 seen_final_ids = set(); unique_final_configs = [];
                 for cfg in all_evaluated_configs_aggregated:
                     cfg_id = cfg.get('config_id');
                     if isinstance(cfg_id, int) and cfg_id not in seen_final_ids:
                         unique_final_configs.append(cfg); seen_final_ids.add(cfg_id)
                 indicator_configs_to_process = unique_final_configs
                 if not indicator_configs_to_process: print("\nOptimization completed, but no valid final configurations."); continue

                 logger.info(f"Bayesian path finished. {len(indicator_configs_to_process)} unique configs."); analysis_path_successful = True; break
            else: print("Invalid choice. Please select 'b', 'c', or press Enter for default.")
    except Exception as prep_err: logger.critical(f"Configuration preparation phase failed: {prep_err}", exc_info=True); sys.exit(1)

    if not analysis_path_successful or not indicator_configs_to_process: logger.error("No configurations prepared."); sys.exit(1)

    # NOTE: current_step is now returned from this function
    # The update happens within the calling function 'run_analysis' based on the returned value
    # config_prep_step_end = current_step # No longer needed here
    logger.info(f"Config phase complete. Path: {'Bayesian' if is_bayesian_path else 'Classical'}. Configs: {len(indicator_configs_to_process)}")
    return indicator_configs_to_process, is_bayesian_path, config_prep_step_end # Return the calculated end step


def _calculate_indicators_and_correlations(
        _display_progress_func: callable, # <-- Pass the progress function
        current_step: int,
        db_path: Path, symbol_id: int, timeframe_id: int, max_lag: int, data: pd.DataFrame,
        indicator_configs_to_process: List[Dict[str, Any]],
        # --- New args for global timing ---
        analysis_start_time_global: float,
        total_analysis_steps_global: int
) -> Tuple[List[Dict], Dict[int, List[Optional[float]]], int, int]:
    """Calculates indicators, filters failed ones, calculates correlations."""
    global _last_periodic_report_time # Allow access to update periodic timer

    logger = logging.getLogger(__name__)

    # --- Indicator Calculation (Relatively Fast - Use single step update) ---
    indicator_step_start = current_step
    indicator_step_end = current_step + 2 # Allocate 2 steps
    _display_progress_func("Calculating Indicators...", indicator_step_start + 0.5, total_analysis_steps_global) # Show midpoint
    indicator_calc_start = time.time(); data_with_indicators, failed_calc_ids = indicator_factory.compute_configured_indicators(data.copy(), indicator_configs_to_process); logger.info(f"Indicator calc took: {utils.format_duration(timedelta(seconds=time.time() - indicator_calc_start))}")
    if failed_calc_ids: logger.warning(f"Indicator calculation failed for {len(failed_calc_ids)} config IDs: {failed_calc_ids}")
    _display_progress_func("Indicators Calculated", indicator_step_end, total_analysis_steps_global)
    current_step = indicator_step_end
    # ------------------------------------------------------------------------

    _display_progress_func("Preparing Data for Correlation", current_step, total_analysis_steps_global)
    configs_for_corr_attempt = [cfg for cfg in indicator_configs_to_process if cfg.get('config_id') not in failed_calc_ids]
    num_skipped_calc = len(indicator_configs_to_process) - len(configs_for_corr_attempt);
    if num_skipped_calc > 0: logger.info(f"Skipped {num_skipped_calc} configs due to calculation failures.")
    if not configs_for_corr_attempt: logger.error("No configurations remaining."); print("\nERROR: All indicators failed."); sys.exit(1)

    logger.info("Dropping rows with any NaNs for correlation analysis..."); data_final = data_with_indicators.dropna(how='any'); logger.info(f"Shape after NaN drop (for Correlation): {data_final.shape}")
    min_points_needed = max(config.DEFAULTS["min_data_points_for_lag"], config.DEFAULTS["min_regression_points"]); min_required_corr_len = max_lag + min_points_needed
    if len(data_final) < min_required_corr_len:
        logger.error(f"Insufficient rows ({len(data_final)}) for max_lag={max_lag} (need {min_required_corr_len}).");
        print(f"\nERROR: Insufficient data rows ({len(data_final)}) after NaN drop. Reduce max_lag?"); sys.exit(1)

    final_configs_for_corr = []; final_columns_in_df = set(data_final.columns); processed_stems = set(); skipped_post_nan = 0
    for cfg in configs_for_corr_attempt:
        name = cfg['indicator_name']; cfg_id = cfg['config_id']; stem = f"{name}_{cfg_id}"
        found_columns = any(col.startswith(stem) for col in final_columns_in_df)
        if found_columns and stem not in processed_stems:
            final_configs_for_corr.append(cfg); processed_stems.add(stem); logger.debug(f"Config {cfg_id} ({name}) included for correlation.")
        elif not found_columns and stem not in processed_stems:
            logger.info(f"All output for {name} (ID: {cfg_id}) removed during NaN drop."); skipped_post_nan += 1; processed_stems.add(stem)
    if not final_configs_for_corr: logger.error("No valid configurations remaining after NaN drop."); print("\nERROR: No indicators suitable for correlation."); sys.exit(1)
    if skipped_post_nan > 0: logger.info(f"Skipped {skipped_post_nan} additional config(s) post NaN drop.")
    logger.info(f"Proceeding to correlation with {len(final_configs_for_corr)} final configurations.");
    _display_progress_func("Data Prepared for Correlation", current_step, total_analysis_steps_global)

    # --- Correlation Calculation (Potentially Long - Uses Internal Progress) ---
    corr_step_start = current_step
    corr_step_end = current_step + 3 # Allocate ~3 steps
    # --- Pass global timing info down ---
    corr_success = correlation_calculator.process_correlations(
        data=data_final, db_path=str(db_path), symbol_id=symbol_id, timeframe_id=timeframe_id,
        indicator_configs_processed=final_configs_for_corr, max_lag=max_lag,
        # --- New Args ---
        analysis_start_time_global=analysis_start_time_global,
        total_analysis_steps_global=total_analysis_steps_global,
        current_step_base=corr_step_start, # Pass the base step for this phase
        total_steps_in_phase=corr_step_end - corr_step_start, # Pass steps allocated
        display_progress_func=_display_progress_func, # Pass the main display func
        periodic_report_func=leaderboard_manager.generate_leading_indicator_report # Pass tally report func
    )
    # ------------------------------------
    current_step = corr_step_end
    if not corr_success: logger.error("Correlation calculation reported failure."); print("\nERROR: Correlation failed."); sys.exit(1)
    _display_progress_func("Correlations Calculated", current_step, total_analysis_steps_global)
    # Reset periodic timer after this potentially long phase
    _last_periodic_report_time = time.time()
    # -----------------------------------------------------------------------------

    # Fetch Final Correlations (Quick)
    _display_progress_func("Fetching Final Correlations", current_step, total_analysis_steps_global)
    conn = None; report_data_ok = False; correlations_by_config_id = {}; adjusted_max_lag = max_lag
    try:
        conn = sqlite_manager.create_connection(str(db_path))
        if not conn: raise ConnectionError("DB connect fail.")
        config_ids_to_fetch = [cfg['config_id'] for cfg in final_configs_for_corr]
        if not config_ids_to_fetch: raise ValueError("No config IDs to fetch.")
        correlations_by_config_id = sqlite_manager.fetch_correlations(conn, symbol_id, timeframe_id, config_ids_to_fetch)
        if not correlations_by_config_id: raise ValueError("Fetch correlations returned no data.")

        actual_max_lag_in_data = max((len(v) for v in correlations_by_config_id.values() if v), default=0)
        if actual_max_lag_in_data <= 0: raise ValueError("No valid lag data found.")
        if actual_max_lag_in_data < adjusted_max_lag:
            logger.warning(f"Adjusting max lag based on fetched data: {adjusted_max_lag} -> {actual_max_lag_in_data}")
            adjusted_max_lag = actual_max_lag_in_data
        report_data_ok = True
    except Exception as fetch_err: logger.error(f"Error fetching final correlations: {fetch_err}", exc_info=True)
    finally:
        if conn: conn.close()

    if not report_data_ok: logger.error("Cannot proceed - failed fetch final correlations."); print("\nERROR: Failed fetch final correlations."); sys.exit(1)
    _display_progress_func("Correlations Fetched", current_step, total_analysis_steps_global)
    return final_configs_for_corr, correlations_by_config_id, adjusted_max_lag, current_step


def _generate_final_reports_and_predict(
    _display_progress_func: callable, # <-- Pass the progress function
    current_step: int, # <<< --- ACCEPT current_step AS ARGUMENT ---
    db_path: Path, symbol: str, timeframe: str, max_lag: int, symbol_id: int, timeframe_id: int,
    data_daterange_str: str, timestamp_str: str,
    is_bayesian_path: bool,
    final_configs_for_corr: List[Dict], correlations_by_config_id: Dict[int, List[Optional[float]]],
    # --- New args for global timing ---
    analysis_start_time_global: float,
    total_analysis_steps_global: int
):
    """Handles final leaderboard update, reporting, prediction, and backtest prompt."""
    logger = logging.getLogger(__name__)

    # --- Leaderboard Update (Quick) ---
    leaderboard_step_start = current_step
    leaderboard_step_end = current_step + 1
    _display_progress_func("Updating Leaderboard", leaderboard_step_start + 0.5, total_analysis_steps_global)
    try:
        leaderboard_manager.update_leaderboard(correlations_by_config_id, final_configs_for_corr, max_lag, symbol, timeframe, data_daterange_str, db_path.name)
    except Exception as lb_err: logger.error(f"Final Leaderboard update failed: {lb_err}", exc_info=True); print("\nWarning: Error final leaderboard update.")
    _display_progress_func("Leaderboard Updated", leaderboard_step_end, total_analysis_steps_global)
    current_step = leaderboard_step_end
    # -----------------------------------

    # --- Final Reports & Viz (Can take time) ---
    report_step_start = current_step
    report_step_end = current_step + 2 # Allocate 2 steps
    _display_progress_func("Generating Final Reports/Viz", report_step_start + 1.0, total_analysis_steps_global)
    file_prefix = f"{timestamp_str}_{symbol}_{timeframe}"
    file_prefix += "_BAYESIAN" if is_bayesian_path else "_CLASSICAL"
    logger.info(f"Using final report/visualization prefix: {file_prefix}")

    try:
        # Run general reports (use fetched correlations)
        utils.run_interim_reports(db_path, symbol_id, timeframe_id, final_configs_for_corr, max_lag, file_prefix, "Final", correlation_data=correlations_by_config_id)
        # Final export of leaderboard text and tally report
        try: leaderboard_manager.export_leaderboard_to_text()
        except Exception as lbe: logger.error(f"Error final leaderboard export: {lbe}", exc_info=True)
        try: leaderboard_manager.generate_leading_indicator_report()
        except Exception as lte: logger.error(f"Error final tally export: {lte}", exc_info=True)

        # Prepare data for visualization
        configs_for_viz = final_configs_for_corr; corrs_for_viz = correlations_by_config_id
        configs_limited = configs_for_viz; corrs_limited = corrs_for_viz
        limit = config.DEFAULTS.get("heatmap_max_configs", 50)
        if len(configs_for_viz) > limit:
             logger.info(f"Limiting final visualizations to top {limit} configs.")
             try:
                  perf_data = []
                  for cfg_id, corrs in corrs_for_viz.items():
                      if corrs and len(corrs) >= max_lag:
                           try: valid_corrs = np.array(corrs[:max_lag], dtype=float); peak = np.nanmax(np.abs(valid_corrs));
                           except (ValueError, TypeError): peak = np.nan
                           if pd.notna(peak): perf_data.append({'config_id': cfg_id, 'peak_abs': peak})
                      # else: logger.warning(f"Config {cfg_id} has insufficient data for viz ranking.")
                  perf_data.sort(key=lambda x: x['peak_abs'], reverse=True)
                  if perf_data:
                      top_ids = {item['config_id'] for item in perf_data[:limit]}
                      configs_limited = [cfg for cfg in configs_for_viz if cfg.get('config_id') in top_ids]
                      corrs_limited = {cfg_id: corrs for cfg_id, corrs in corrs_for_viz.items() if cfg_id in top_ids}
                      logger.info(f"Successfully limited viz data to {len(configs_limited)} configs.")
                  else: logger.warning("Could not rank configs for viz limiting."); # Keep full set
             except Exception as filter_err: logger.error(f"Error filtering viz data: {filter_err}."); # Keep full set

        # Generate visualizations (can be slow, only update progress after)
        if configs_limited and corrs_limited:
            logger.info(f"Generating line plots for {len(configs_limited)} configs...")
            visualization_generator.plot_correlation_lines(corrs_limited, configs_limited, max_lag, config.LINE_CHARTS_DIR, file_prefix)
            logger.info(f"Generating combined chart for {len(configs_limited)} configs...")
            visualization_generator.generate_combined_correlation_chart(corrs_limited, configs_limited, max_lag, config.COMBINED_CHARTS_DIR, file_prefix)
            logger.info(f"Generating heatmap for {len(configs_limited)} configs...")
            visualization_generator.generate_enhanced_heatmap(corrs_limited, configs_limited, max_lag, config.HEATMAPS_DIR, file_prefix)
        else: logger.warning("No configurations available after filtering for main visualizations.")
        if corrs_for_viz and final_configs_for_corr: # Envelope uses full data
            logger.info(f"Generating correlation envelope for {len(final_configs_for_corr)} configs...")
            visualization_generator.generate_correlation_envelope_chart(corrs_for_viz, final_configs_for_corr, max_lag, config.REPORTS_DIR, file_prefix)
        else: logger.warning("Skipping envelope chart.")

        logger.info("Final visualization generation attempt complete."); print("\nFinal reports and visualizations generated.")
    except Exception as final_report_vis_err: logger.error(f"Error final reporting/vis phase: {final_report_vis_err}", exc_info=True); print("\nWarning: Error during final report/vis generation.")
    _display_progress_func("Reports & Visualizations Complete", report_step_end, total_analysis_steps_global)
    current_step = report_step_end
    # -------------------------------------------

    # --- Automatic Prediction (Quick) ---
    predict_step_start = current_step
    predict_step_end = current_step + 1
    _display_progress_func("Running Prediction", predict_step_start + 0.5, total_analysis_steps_global)
    print("\n--- Automatic Price Prediction ---"); logger.info(f"Running prediction up to lag {max_lag}...")
    try: predictor.predict_price(db_path, symbol, timeframe, max_lag)
    except Exception as pred_err: logger.error(f"Automatic prediction failed: {pred_err}", exc_info=True); print("\nError running automatic prediction.")
    _display_progress_func("Prediction Attempted", predict_step_end, total_analysis_steps_global)
    current_step = predict_step_end
    # -------------------------------------

    # Optional Backtest Check Prompt (Doesn't affect main ETA)
    if input("\nRun Historical Predictor Check (uses final leaderboard - LOOKAHEAD BIAS!)? [y/N]: ").strip().lower() == 'y':
        print("\n--- Launching Historical Predictor Check ---")
        try:
             num_points_bt = 0
             while num_points_bt <= 0:
                try: default_points_bt = config.DEFAULTS.get("backtester_default_points", 50); points_input = input(f"Points per lag [default: {default_points_bt}]: ").strip(); num_points_bt = int(points_input or default_points_bt)
                except ValueError: print("Invalid input."); num_points_bt = 0
                if num_points_bt <= 0: print("Points must be positive.")
             logger.info(f"Starting historical predictor check: lag={max_lag}, points={num_points_bt}")
             backtester.run_backtest(db_path, symbol, timeframe, max_lag, num_points_bt)
        except Exception as bt_err: logger.error(f"Post-analysis historical check failed: {bt_err}", exc_info=True); print("\nError running historical check.")
        logger.info("Finished optional historical check.")
    else: logger.info("Skipped optional historical predictor check.")


# --- Main Analysis Orchestration ---
def run_analysis():
    """Main orchestrating function, calling phase functions."""
    global _confirmed_analysis_start_time # Use global
    logger = logging.getLogger(__name__)
    initial_script_start_time = time.time() # Log start even before confirmation
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger.info(f"--- Starting Analysis Run Attempt: {timestamp_str} ---")
    last_eta_update_time = time.time()
    current_analysis_step = 0
    _confirmed_analysis_start_time = None # Reset global timer state

    def _display_progress(stage_name: str, current_step: float, total_steps: int):
        """Displays progress and GLOBAL estimated time remaining."""
        nonlocal last_eta_update_time
        global _confirmed_analysis_start_time # Access global start time

        # Use the confirmed start time if available, otherwise estimate from script start
        start_time_for_calc = _confirmed_analysis_start_time if _confirmed_analysis_start_time is not None else initial_script_start_time
        # Only show ETA after confirmation
        show_eta = _confirmed_analysis_start_time is not None

        now = time.time(); update_interval = ETA_UPDATE_INTERVAL_SECONDS
        force_update = ("Complete" in stage_name) or ("Finished" in stage_name) or (current_step >= total_steps)
        time_elapsed_since_last = now - last_eta_update_time
        should_update = (time_elapsed_since_last > update_interval) or (current_step == 1) or force_update

        if should_update:
            elapsed_td = timedelta(seconds=now - start_time_for_calc); elapsed_str = utils.format_duration(elapsed_td);
            percent = min(100.0, (current_step / total_steps * 100) if total_steps > 0 else 0);
            eta_str = "Pending User Confirm"
            if show_eta:
                if percent > 0.1 and current_step < total_steps : # Start estimating after tiny progress
                    elapsed_seconds = elapsed_td.total_seconds()
                    # Base ETA on progress since confirmation
                    rate = current_step / elapsed_seconds if elapsed_seconds > 1 else 0
                    if rate > 0:
                        remaining_steps = total_steps - current_step
                        eta_seconds = remaining_steps / rate
                        eta_td = timedelta(seconds=eta_seconds)
                        eta_str = utils.format_duration(eta_td)
                    else: eta_str = "Calculating..."
                elif current_step >= total_steps: eta_str = "Done"
                else: eta_str = "Calculating..." # Initial state after confirm

            current_step_display = min(current_step, total_steps)
            print(f"\rStage: {stage_name:<35} | Overall: {current_step_display:.1f}/{total_steps} ({percent: >5.1f}%) | Elapsed: {elapsed_str: <15} | ETA: {eta_str: <15}", end="")
            last_eta_update_time = now
            if force_update: print()
            # Ensure log file gets frequent updates if needed (optional flush)
            # logging.getLogger().handlers[0].flush() # Example: Flush first handler (likely file) - Use with caution

    # --- Analysis Flow ---
    _display_progress("Initializing...", 0, TOTAL_ANALYSIS_STEPS)
    mode_choice = _setup_and_select_mode(timestamp_str);
    if mode_choice is None: return

    if mode_choice == 'c': _run_custom_mode(timestamp_str); return
    if mode_choice == 'b': _run_backtest_check_mode(timestamp_str); return

    # --- Full Analysis Path ('a') ---
    current_analysis_step = 1; _display_progress("Setup Complete", current_analysis_step, TOTAL_ANALYSIS_STEPS)

    db_path, symbol, timeframe, data, max_lag, symbol_id, timeframe_id, data_daterange_str = _select_data_source_and_lag();
    current_analysis_step = 2; _display_progress("Data Source & Lag Confirmed", current_analysis_step, TOTAL_ANALYSIS_STEPS)

    indicator_factory._load_indicator_definitions(); indicator_definitions = indicator_factory._INDICATOR_DEFS
    # Pass global timing args here
    indicator_configs_to_process, is_bayesian_path, current_analysis_step = _prepare_configurations(
        _display_progress, current_analysis_step, db_path, symbol, timeframe, max_lag, data,
        indicator_definitions, symbol_id, timeframe_id, data_daterange_str, timestamp_str,
        analysis_start_time_global=_confirmed_analysis_start_time, # Will be None until confirmed inside
        total_analysis_steps_global=TOTAL_ANALYSIS_STEPS
    )
    # current_analysis_step is updated inside
    _display_progress("Configuration Prep Complete", current_analysis_step, TOTAL_ANALYSIS_STEPS)

    # Ensure start time is set before proceeding if not already
    if _confirmed_analysis_start_time is None:
        logger.critical("Analysis start timer was not set after config preparation! Exiting.")
        sys.exit(1)

    # Pass global timing args here
    final_configs_for_corr, correlations_by_config_id, max_lag, current_analysis_step = _calculate_indicators_and_correlations(
        _display_progress, current_analysis_step, db_path, symbol_id, timeframe_id, max_lag, data, indicator_configs_to_process,
        analysis_start_time_global=_confirmed_analysis_start_time,
        total_analysis_steps_global=TOTAL_ANALYSIS_STEPS
    )
    # current_analysis_step is updated inside
    _display_progress("Indicator/Correlation Complete", current_analysis_step, TOTAL_ANALYSIS_STEPS)

    # Pass global timing args here (though less critical for ETA in final phase)
    # --- Pass the updated current_analysis_step ---
    _generate_final_reports_and_predict(
        _display_progress, current_analysis_step, db_path, symbol, timeframe, max_lag, symbol_id, timeframe_id,
        data_daterange_str, timestamp_str, is_bayesian_path,
        final_configs_for_corr, correlations_by_config_id,
        analysis_start_time_global=_confirmed_analysis_start_time,
        total_analysis_steps_global=TOTAL_ANALYSIS_STEPS
    )
    # --- Progress display handled within the function ---

    # Ensure final progress shows 100%
    _display_progress("Run Finished", TOTAL_ANALYSIS_STEPS, TOTAL_ANALYSIS_STEPS)

    end_time = time.time();
    # Calculate duration based on confirmed start if available
    actual_start_time = _confirmed_analysis_start_time if _confirmed_analysis_start_time is not None else initial_script_start_time
    duration_td = timedelta(seconds=end_time - actual_start_time)
    duration_str = utils.format_duration(duration_td)
    logger.info(f"--- Analysis Run Completed: {timestamp_str} ---"); logger.info(f"Total execution time (post-confirmation): {duration_str}")
    print(f"\nAnalysis complete (lag={max_lag}). Reports saved in '{config.REPORTS_DIR}'. Total time: {duration_str}")


# --- Main Execution Block ---
if __name__ == "__main__":
    logging_setup.setup_logging(); logger = logging.getLogger(__name__)
    exit_code = 0
    try: run_analysis()
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else 1
        log_msg = "Analysis exited via sys.exit()" if exit_code == 0 else f"Analysis exited prematurely via sys.exit() (code {exit_code})"
        try: logger.log(logging.INFO if exit_code == 0 else logging.WARNING, f"--- {log_msg}. ---")
        except Exception: pass
        print(f"\n--- {log_msg} ---");
    except KeyboardInterrupt:
        exit_code = 130
        log_msg = "Analysis interrupted by user (Ctrl+C)."
        try: logger.warning(f"--- {log_msg} ---")
        except Exception: pass
        print(f"\n--- {log_msg} ---");
    except Exception as e:
        exit_code = 1
        log_msg = f"UNHANDLED EXCEPTION: {e}"
        try: logger.critical(f"--- {log_msg} ---", exc_info=True)
        except Exception: pass
        log_file_path = config.LOG_DIR / "logfile.txt"
        print(f"\nCRITICAL ERROR: {e}\nCheck log file ('{log_file_path}') for details.")
    finally:
        if exit_code != 0:
            try: logger.info("--- Analysis run concluded (with errors or interruption). ---")
            except Exception: pass
        logging.shutdown()
        sys.exit(exit_code)