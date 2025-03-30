# parameter_optimizer.py
import logging
import random
import json
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import sqlite3
import time
import warnings
from functools import partial # Import partial
from datetime import datetime, timedelta # For ETA calculation

# --- Bayesian Optimization Imports ---
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical # Import dimension types
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    gp_minimize = Real = Integer = Categorical = use_named_args = None
    warnings.warn("scikit-optimize (skopt) not installed. Bayesian Optimization path will not be available.", ImportWarning)
# --- End Bayesian Optimization Imports ---

import config as app_config # Import app_config
import indicator_factory
import sqlite_manager
import utils # Import utils for format_duration
import parameter_generator
import leaderboard_manager # Import leaderboard manager

logger = logging.getLogger(__name__)

# --- Caches ---
# These caches are intended to be cleared *before* each indicator optimization run in main.py
indicator_series_cache: Dict[str, pd.DataFrame] = {} # param_hash -> indicator_output_df
single_correlation_cache: Dict[Tuple[str, int], Optional[float]] = {} # (param_hash, lag) -> correlation_value

# --- Constants ---
# Get threshold from config, provide a default fallback
WEAK_CORR_THRESHOLD_SKIP = app_config.DEFAULTS.get("weak_corr_threshold_skip", 0.15)
ETA_UPDATE_INTERVAL_SECONDS = app_config.DEFAULTS.get("eta_update_interval_seconds", 15)


# --- Helper functions ---
def _get_config_hash(params: Dict[str, Any]) -> str:
    """Generates a stable SHA256 hash for a parameter dictionary."""
    # Uses the more robust hashing from utils
    return utils.get_config_hash(params)

# --- Core Calculation Helpers (Revised Caching) ---
def _calculate_correlation_at_target_lag(
    params: Dict[str, Any], config_id: int, indicator_name: str, target_lag: int,
    base_data_with_required: pd.DataFrame, shifted_close_at_lag: Optional[pd.Series]
) -> Optional[float]:
    """Computes indicator (using cache) and correlation ONLY for the target_lag."""
    param_hash = _get_config_hash(params)
    # Reduced log frequency for this specific step
    # logger.debug(f"Calculating single corr: ID {config_id} (Hash: {param_hash[:8]}), Lag {target_lag}")

    # 1. Get Indicator Series (Cache check)
    indicator_output_df = None; cache_hit = False
    if param_hash in indicator_series_cache:
        indicator_output_df = indicator_series_cache[param_hash]
        if not isinstance(indicator_output_df, pd.DataFrame):
             logger.warning(f"Invalid cached data for {param_hash[:8]}. Recomputing."); indicator_output_df = None; del indicator_series_cache[param_hash]
        elif indicator_output_df.empty:
             logger.debug(f"Cached indicator DF is empty for {param_hash[:8]}. Recomputing."); indicator_output_df = None; del indicator_series_cache[param_hash] # Allow recompute if cached empty
        else: cache_hit = True; # logger.debug(f"Indicator cache HIT: ID {config_id} / hash {param_hash[:8]}.") # Reduce noise

    if indicator_output_df is None: # Cache miss or invalid
        # logger.debug(f"Indicator cache MISS: ID {config_id} / hash {param_hash[:8]}. Computing...") # Reduce noise
        config_details = {'indicator_name': indicator_name, 'params': params, 'config_id': config_id}
        # Pass copy to avoid modification issues
        indicator_output_df = indicator_factory._compute_single_indicator(base_data_with_required.copy(), config_details)
        indicator_series_cache[param_hash] = indicator_output_df if (indicator_output_df is not None and not indicator_output_df.empty) else pd.DataFrame() # Store empty DF if failed

    # Check result
    if indicator_output_df is None or indicator_output_df.empty:
        logger.debug(f"Indicator {'cache' if cache_hit else 'compute'} failed/empty for ID {config_id}.") # Keep as debug
        return None

    # 2. Calculate Correlation
    output_columns = list(indicator_output_df.columns)
    if not output_columns: logger.warning(f"Indicator DF empty columns: ID {config_id}."); return None
    if shifted_close_at_lag is None: logger.error(f"Shifted close missing for lag {target_lag}! Cannot calc corr for ID {config_id}."); return None

    max_abs_corr = -1.0; best_signed_corr_at_lag = np.nan
    for col in output_columns:
        if col not in indicator_output_df.columns: continue
        try: indicator_series = indicator_output_df[col].astype(float)
        except (ValueError, TypeError) as e: logger.warning(f"Cannot convert '{col}' to float (ID {config_id}). Skipping. Error: {e}"); continue
        if indicator_series.isnull().all() or indicator_series.nunique(dropna=True) <= 1: logger.debug(f"Skipping const/NaN col '{col}' (ID {config_id}) lag {target_lag}."); continue

        try:
            # Ensure alignment and handle NaNs properly during correlation
            combined = pd.concat([indicator_series, shifted_close_at_lag], axis=1).dropna()
            if len(combined) < 2: continue # Not enough pairs to correlate
            correlation = combined.iloc[:, 0].corr(combined.iloc[:, 1])

            if pd.notna(correlation):
                current_abs = abs(float(correlation))
                if current_abs > max_abs_corr: max_abs_corr = current_abs; best_signed_corr_at_lag = float(correlation)
        except Exception as e: logger.error(f"Error calculating single corr for {col}, lag {target_lag}: {e}", exc_info=True)

    # logger.debug(f"Single corr calculated: ID {config_id}, Lag {target_lag}. Result: {best_signed_corr_at_lag}") # Reduce noise
    return best_signed_corr_at_lag if max_abs_corr > -1.0 else np.nan

# --- Objective Function ---
def _objective_function(params_list: List[Any], *, # Use keyword-only args after params_list
                         param_names: List[str], target_lag: int, indicator_name: str,
                         indicator_def: Dict, base_data_with_required: pd.DataFrame,
                         db_path: str, symbol_id: int, timeframe_id: int,
                         master_evaluated_configs: Dict[str, Dict[str, Any]],
                         progress_state: Dict[str, Any],
                         shifted_closes_global_cache: Dict[int, pd.Series],
                         # ----> NEW PARAMETERS FOR LEADERBOARD <----
                         symbol: str, timeframe: str, data_daterange: str, source_db_name: str
                         ) -> float:
    """Objective function for Bayesian opt, using FAST single-lag evaluation, caches, and REAL-TIME leaderboard update."""
    # Convert/Clean params
    params_dict = dict(zip(param_names, params_list))
    for k, v in params_dict.items():
        if isinstance(v, np.integer): params_dict[k] = int(v)
        elif isinstance(v, np.floating): params_dict[k] = float(v)

    # Validate Params
    full_params = params_dict.copy()
    param_defs = indicator_def.get('parameters', {})
    for p_name, p_details in param_defs.items():
         if p_name not in full_params and 'default' in p_details: full_params[p_name] = p_details['default']
    if not parameter_generator.evaluate_conditions(full_params, indicator_def.get('conditions', [])):
        logger.debug(f"ObjFn ({indicator_name}): Invalid params: {params_dict} (Full: {full_params}). High cost.")
        return 1e6

    # --- Cache Check / Calculation ---
    params_to_store = full_params; param_hash = _get_config_hash(params_to_store)
    cache_key = (param_hash, target_lag); correlation_value = None; cache_hit = False; config_id = None

    if cache_key in single_correlation_cache:
        correlation_value = single_correlation_cache[cache_key]; cache_hit = True
        config_id = master_evaluated_configs.get(param_hash, {}).get('config_id')
        if config_id is None: logger.error(f"CRITICAL Cache Inconsistency: Hash {param_hash[:8]} lag {target_lag} in single cache but not master list!")
        # logger.debug(f"ObjFn ({indicator_name} Lag {target_lag}): Cached single corr: ID {config_id} / hash {param_hash[:8]}. Val: {correlation_value}") # Reduce noise
    else: # Cache Miss
        cache_hit = False; # logger.debug(f"ObjFn ({indicator_name} Lag {target_lag}): Single corr cache MISS: hash {param_hash[:8]}.") # Reduce noise
        # Get Config ID
        if param_hash in master_evaluated_configs: config_id = master_evaluated_configs[param_hash]['config_id']
        else:
            # >>> Critical Section: Need to ensure DB access is thread-safe if multi-processing objective <<<
            # For now, assuming sequential execution per indicator. If parallelizing lags, need locking here.
            conn_temp = sqlite_manager.create_connection(db_path)
            if conn_temp:
                try:
                    config_id = sqlite_manager.get_or_create_indicator_config_id(conn_temp, indicator_name, params_to_store)
                    master_evaluated_configs[param_hash] = {'params': params_to_store, 'config_id': config_id}
                    logger.debug(f"Added new config to master: '{indicator_name}' (ID: {config_id}), Params: {params_to_store}") # Changed to debug
                except Exception as e: logger.error(f"ObjFn ({indicator_name}): Failed get/create config ID: {e}", exc_info=True)
                finally: conn_temp.close()
            else: logger.error(f"ObjFn ({indicator_name}): Cannot connect DB for config ID.")
            # <<< End Critical Section >>>
        if config_id is None: logger.error(f"ObjFn ({indicator_name}): Failed get config ID. High cost."); return 1e6

        # Calculate Single Correlation
        shifted_close = shifted_closes_global_cache.get(target_lag)
        if shifted_close is None: logger.error(f"ObjFn ({indicator_name} Lag {target_lag}): Shifted close missing!"); correlation_value = None
        else: correlation_value = _calculate_correlation_at_target_lag(params_to_store, config_id, indicator_name, target_lag, base_data_with_required, shifted_close)
        single_correlation_cache[cache_key] = correlation_value # Cache result

    # --- Progress Reporting with ETA ---
    progress_state['calls_completed'] += 1; calls_done = progress_state['calls_completed']
    calls_total = progress_state['total_expected_calls']; current_time = time.time()
    elapsed_td = timedelta(seconds=current_time - progress_state['start_time'])
    rate = (calls_done / elapsed_td.total_seconds()) if elapsed_td.total_seconds() > 1 else 0
    percent = (calls_done / calls_total * 100) if calls_total > 0 else 0
    eta_td = timedelta(seconds=(calls_total - calls_done) / rate) if rate > 0 and calls_done < calls_total else timedelta(seconds=0)

    # Log less frequently based on time interval
    if current_time - progress_state.get('last_report_time', 0) > ETA_UPDATE_INTERVAL_SECONDS or calls_done == 1 or calls_done == calls_total:
        eta_str = utils.format_duration(eta_td) if calls_done < calls_total else "Done"
        elapsed_str = utils.format_duration(elapsed_td)
        # Show progress per indicator/lag combo clearly
        print(f"\rOpt ({indicator_name} L{target_lag}): {calls_done}/{calls_total} ({percent:.1f}%) | Elapsed: {elapsed_str} | ETA: {eta_str} ", end="")
        # Log full details less often
        if calls_done % (progress_state['log_frequency']) == 0 or calls_done == 1 or calls_done == calls_total:
             logger.info(f"Progress ({indicator_name} L{target_lag}): Eval {calls_done}/{calls_total} ({percent:.1f}%) | Rate: {rate:.1f} evals/s | CacheHit: {'Yes' if cache_hit else 'No'} | Elapsed: {elapsed_str} | ETA: {eta_str}")
        progress_state['last_report_time'] = current_time
        # Ensure the print buffer is flushed
        if calls_done == calls_total: print() # Newline at the end


    # --- Check & Update Leaderboard IMMEDIATELY ---
    if config_id is not None and pd.notna(correlation_value):
        try:
            # This function now handles its own DB connection and export trigger
            updated = leaderboard_manager.check_and_update_single_lag(
                lag=target_lag,
                correlation_value=float(correlation_value),
                indicator_name=indicator_name,
                params=params_to_store, # Pass the actual parameters used
                config_id=config_id,
                symbol=symbol,
                timeframe=timeframe,
                data_daterange=data_daterange,
                source_db_name=source_db_name
            )
            # Optional: Add logic here if something needs to happen *immediately* after update
            # if updated: logger.debug(f"Real-time leaderboard update triggered for Lag {target_lag} by ID {config_id}")

        except Exception as lb_err:
            logger.error(f"Error during immediate leaderboard update check (Lag {target_lag}, ID {config_id}): {lb_err}", exc_info=True)

    # --- Determine Objective Score ---
    if pd.notna(correlation_value) and isinstance(correlation_value, (float, int, np.number)):
         score = abs(float(correlation_value)); objective_value = -score
         log_id = config_id if config_id is not None else "N/A"
         # logger.debug(f"  -> ObjFn ({indicator_name} L{target_lag}): ID {log_id} -> Score={score:.4f} (Corr={correlation_value:.4f}). Objective={objective_value:.4f}") # Reduce noise
         return objective_value
    else:
         log_id = config_id if config_id is not None else "N/A"
         logger.debug(f"  -> ObjFn ({indicator_name} L{target_lag}): ID {log_id} ({params_to_store}) -> Failed/NaN corr. High cost.")
         return 1e6

# --- Main Bayesian Optimization Function ---
def optimize_parameters_bayesian_per_lag(
    indicator_name: str, indicator_def: Dict, base_data_with_required: pd.DataFrame,
    max_lag: int, n_calls_per_lag: int, n_initial_points_per_lag: int,
    db_path: str, symbol_id: int, timeframe_id: int,
    # ----> NEW PARAMETERS FOR CONTEXT <----
    symbol: str, timeframe: str, data_daterange: str, source_db_name: str
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]: # Return best_per_lag (mostly for logging) and evaluated_configs
    """Optimizes parameters using Bayesian Optimization per lag. Returns evaluated configs."""
    if not SKOPT_AVAILABLE: logger.error("skopt not installed."); return {}, []
    logger.info(f"--- Starting Bayesian Opt Per-Lag for: {indicator_name} ---")
    logger.info(f"Lags: 1-{max_lag}, Evals/Lag: {n_calls_per_lag}, Initials: {n_initial_points_per_lag}")

    search_space, param_names, fixed_params, has_tunable = _define_search_space(indicator_def.get('parameters', {}))
    if not has_tunable: logger.error(f"'{indicator_name}' has no tunable params."); return {}, []
    logger.info(f"Opt Space Dimensions ({len(param_names)}): {param_names}, Fixed: {fixed_params}")

    # --- Global Tracking (within this function's scope) ---
    master_evaluated_configs: Dict[str, Dict[str, Any]] = {} # Hash -> {'params': dict, 'config_id': int}
    best_config_per_lag: Dict[int, Dict[str, Any]] = {} # Lag -> best result dict (still useful for logging)

    # Total expected calls for *this indicator's* optimization phase
    total_expected_calls_indicator = max_lag * n_calls_per_lag
    # Log frequency based on total calls
    log_freq = max(1, total_expected_calls_indicator // 10) if total_expected_calls_indicator > 20 else 5
    progress_state = {'calls_completed': 0, 'start_time': time.time(),
                      'last_report_time': 0, 'total_expected_calls': total_expected_calls_indicator,
                      'log_frequency': log_freq}

    # --- Pre-calculate Shifted Closes ---
    logger.info(f"Pre-calc {max_lag} shifted 'close' series..."); start_shift = time.time()
    if 'close' not in base_data_with_required.columns: logger.error("'close' missing."); return {}, []
    close_series = base_data_with_required['close'].astype(float)
    # Align shifted series with the original index for safety
    shifted_closes_global_cache = {lag: close_series.shift(-lag).reindex(base_data_with_required.index) for lag in range(1, max_lag + 1)}
    logger.info(f"Shifted closes pre-calc complete: {time.time() - start_shift:.2f}s.")

    # --- Evaluate Default Config (Fast Eval) ---
    # This adds the default config to master_evaluated_configs and single_correlation_cache if valid
    _ = _process_default_config_fast_eval( # Don't need the ID returned here anymore
        indicator_name, indicator_def, fixed_params, param_names, db_path,
        master_evaluated_configs, base_data_with_required, max_lag,
        shifted_closes_global_cache,
        symbol, timeframe, data_daterange, source_db_name
    )

    # --- Optimization Loop per Lag ---
    total_lags_skipped = 0
    for target_lag in range(1, max_lag + 1):
        print(f"\rOptimizing {indicator_name} - Lag {target_lag}/{max_lag}...", end="")

        # --- Initial Point Evaluation & Weak Correlation Check ---
        logger.debug(f"Lag {target_lag}: Evaluating initial points...")
        initial_x = []
        initial_y = []
        initial_configs_evaluated_this_lag = 0
        max_abs_corr_initial = 0.0

        # Generate initial random points (ensure n_initial_points_per_lag > 0)
        num_random_points = max(0, n_initial_points_per_lag)
        for i in range(num_random_points):
            # Generate a *random* valid config using the utility
            # Pass only the tunable part of the definition
            tunable_param_defs = {name: indicator_def['parameters'][name] for name in param_names if name in indicator_def.get('parameters', {})}
            random_params_dict = parameter_generator._generate_random_valid_config(tunable_param_defs, indicator_def.get('conditions', []))

            if random_params_dict is None:
                logger.warning(f"Lag {target_lag}, Initial Point {i+1}: Failed to generate a random valid config. Skipping this point.")
                continue

            # Objective function handles getting ID, calculating corr, updating leaderboard
            obj_value = _objective_function(
                params_list=[random_params_dict.get(p_name) for p_name in param_names], # Extract values in correct order
                param_names=param_names, target_lag=target_lag, indicator_name=indicator_name,
                indicator_def=indicator_def, base_data_with_required=base_data_with_required,
                db_path=db_path, symbol_id=symbol_id, timeframe_id=timeframe_id,
                master_evaluated_configs=master_evaluated_configs, progress_state=progress_state,
                shifted_closes_global_cache=shifted_closes_global_cache,
                symbol=symbol, timeframe=timeframe, data_daterange=data_daterange, source_db_name=source_db_name
            )

            initial_configs_evaluated_this_lag += 1
            if obj_value < (1e6 - 1.0): # Check if valid result
                 initial_x.append([random_params_dict.get(p_name) for p_name in param_names])
                 initial_y.append(obj_value)
                 current_abs_corr = -obj_value # Objective is -abs(corr)
                 max_abs_corr_initial = max(max_abs_corr_initial, current_abs_corr)
                 logger.debug(f"Lag {target_lag}, Initial Point {i+1}: Params={random_params_dict}, Corr={current_abs_corr:.4f}")
            else:
                 logger.debug(f"Lag {target_lag}, Initial Point {i+1}: Params={random_params_dict}, Invalid objective value ({obj_value}).")
        # --- End Initial Point Evaluation ---

        # --- Skip Lag if Initial Points are Weak ---
        if max_abs_corr_initial < WEAK_CORR_THRESHOLD_SKIP:
             logger.warning(f"Skipping Optimization for Lag {target_lag} ({indicator_name}): Max abs corr in initial {initial_configs_evaluated_this_lag} points ({max_abs_corr_initial:.4f}) < threshold ({WEAK_CORR_THRESHOLD_SKIP}).")
             total_lags_skipped += 1
             # Ensure progress counter reflects skipped evals for this lag if needed for overall ETA
             calls_to_skip = n_calls_per_lag - initial_configs_evaluated_this_lag
             progress_state['calls_completed'] += calls_to_skip
             continue # Skip to the next lag
        # --- End Skip Check ---

        # Proceed with Bayesian Optimization using evaluated initial points
        logger.info(f"Lag {target_lag}: Proceeding with guided optimization ({len(initial_x)} valid initial points).")

        # Define the objective function partially applied with context for gp_minimize
        objective_partial = partial(_objective_function,
                                    param_names=param_names, target_lag=target_lag, indicator_name=indicator_name,
                                    indicator_def=indicator_def, base_data_with_required=base_data_with_required,
                                    db_path=db_path, symbol_id=symbol_id, timeframe_id=timeframe_id,
                                    master_evaluated_configs=master_evaluated_configs, progress_state=progress_state,
                                    shifted_closes_global_cache=shifted_closes_global_cache,
                                    symbol=symbol, timeframe=timeframe, data_daterange=data_daterange, source_db_name=source_db_name
                                    )
        try:
             with warnings.catch_warnings():
                  warnings.simplefilter("ignore", category=UserWarning) # Filter skopt warnings
                  warnings.simplefilter("ignore", category=RuntimeWarning) # Filter potential numpy warnings

                  # Adjust n_calls: total desired calls MINUS initial points already evaluated
                  num_guided_calls = max(0, n_calls_per_lag - initial_configs_evaluated_this_lag)

                  if num_guided_calls > 0:
                      result = gp_minimize(func=objective_partial, dimensions=search_space,
                                          acq_func=app_config.DEFAULTS.get("optimizer_acq_func", 'gp_hedge'), # Get from config
                                          n_calls=num_guided_calls, # Only the *remaining* calls
                                          n_initial_points=0, # No *additional* random points needed by gp_minimize
                                          x0=initial_x if initial_x else None, # Provide evaluated initial points
                                          y0=initial_y if initial_y else None,
                                          random_state=None, # Use default random state behavior
                                          noise='gaussian') # Assume some noise

                      best_params_list = result.x; best_objective_value = result.fun
                  elif initial_y: # Only initial points were run
                      best_initial_idx = np.argmin(initial_y)
                      best_params_list = initial_x[best_initial_idx]
                      best_objective_value = initial_y[best_initial_idx]
                      logger.info(f"Lag {target_lag}: Using best from initial points only (no guided calls).")
                  else: # No valid initial points AND no guided calls -> skip
                       logger.warning(f"Lag {target_lag}: No valid initial points and no guided calls requested. No result.")
                       best_config_per_lag[target_lag] = None
                       continue # Skip to next lag

             # Process the best result found (either from guided opt or initial points)
             if best_objective_value < (1e6 - 1.0): # Found valid solution
                 best_score = -best_objective_value
                 best_params_opt = dict(zip(param_names, best_params_list))
                 for k, v in best_params_opt.items(): # Convert numpy types
                     if isinstance(v, np.integer): best_params_opt[k] = int(v)
                     elif isinstance(v, np.floating): best_params_opt[k] = float(v)
                 best_params_full = {**fixed_params, **best_params_opt}
                 best_hash = _get_config_hash(best_params_full)
                 best_config_info = master_evaluated_configs.get(best_hash)

                 if best_config_info and 'config_id' in best_config_info:
                     best_config_id = best_config_info['config_id']
                     # Retrieve correlation from cache (should exist from objective call)
                     best_corr = single_correlation_cache.get((best_hash, target_lag))
                     logger.info(f"Lag {target_lag} Best (Opt {indicator_name}): ID={best_config_id}, Params={best_params_full}, Score={best_score:.6f}, Corr={best_corr if best_corr is not None else 'N/A'}")
                     best_config_per_lag[target_lag] = {'params': best_params_full, 'config_id': best_config_id, 'correlation_at_lag': best_corr, 'score_at_lag': best_score}
                 else:
                     logger.error(f"Lag {target_lag}: Could not find config ID for best params {best_params_full} (Hash: {best_hash[:8]}). Opt score: {best_objective_value}. Master keys: {list(master_evaluated_configs.keys())[:5]}...")
                     best_config_per_lag[target_lag] = None
             else:
                  logger.warning(f"Lag {target_lag}: Optimization failed for {indicator_name} (Objective={best_objective_value}). No valid solution found.")
                  best_config_per_lag[target_lag] = None

        except Exception as opt_err:
            logger.error(f"Error during opt for {indicator_name} lag {target_lag}: {opt_err}", exc_info=True)
            best_config_per_lag[target_lag] = None
    # End lag loop
    print() # Newline after finishing all lags for the indicator

    if total_lags_skipped > 0:
        logger.warning(f"{indicator_name}: Skipped optimization for {total_lags_skipped}/{max_lag} lags due to weak initial correlation.")

    _log_final_progress(indicator_name, progress_state)

    final_all_evaluated_configs = _format_final_evaluated_configs(indicator_name, master_evaluated_configs)
    _log_optimization_summary(indicator_name, max_lag, best_config_per_lag, final_all_evaluated_configs)
    return best_config_per_lag, final_all_evaluated_configs


# --- Helper Functions for Bayesian Opt ---
def _define_search_space(param_defs: Dict) -> Tuple[List, List[str], Dict, bool]:
    """Helper to define the search space for skopt."""
    search_space = []; param_names = []; fixed_params = {}; has_tunable = False
    period_params = ['fast', 'slow', 'fastperiod', 'slowperiod', 'signalperiod', 'timeperiod', 'timeperiod1', 'timeperiod2', 'timeperiod3', 'length', 'window', 'obv_period', 'price_period', 'fastk_period', 'slowk_period', 'slowd_period', 'fastd_period', 'tenkan', 'kijun', 'senkou']
    factor_params = ['fastlimit','slowlimit','acceleration','maximum','vfactor','smoothing']
    dev_scalar_params = ['scalar','nbdev','nbdevup','nbdevdn']

    # --- Add MAType as Categorical if present ---
    if 'matype' in param_defs:
        # TA-Lib MATypes 0-8
        # Note: Pandas-TA uses strings ('sma', 'ema'...). This currently only handles TA-Lib style.
        # If optimizing Pandas-TA indicators with mamode, would need Categorical(['sma', 'ema', ...])
        if param_defs['matype'].get('default', -1) >= 0: # Check if it looks like TA-Lib style
             matype_values = list(range(9)) # 0 to 8
             search_space.append(Categorical(categories=matype_values, name='matype'))
             param_names.append('matype')
             has_tunable = True # Treat matype as tunable
             logger.debug("Space 'matype': Categorical(0..8)")
        elif 'default' in param_defs['matype']: # If not 0-8 default, treat as fixed
            fixed_params['matype'] = param_defs['matype']['default']
            logger.debug(f"Param 'matype': Fixed @ '{fixed_params['matype']}' (non-standard default).")
    # --- End MAType Handling ---

    for name, details in sorted(param_defs.items()):
        if name == 'matype': continue # Already handled

        default = details.get('default'); min_j = details.get('min'); max_j = details.get('max')
        # Infer type more robustly
        p_type = type(default) if default is not None else (type(min_j) if min_j is not None else (type(max_j) if max_j is not None else None))
        is_num_tune = p_type in [int, float] and (min_j is not None and max_j is not None)

        if not is_num_tune:
            if default is not None: fixed_params[name] = default; logger.debug(f"Param '{name}': Fixed @ '{default}'.")
            else: logger.warning(f"Param '{name}': No default/bounds for fixing. Skipping."); continue
            continue

        # Use JSON bounds directly if valid
        min_v, max_v = min_j, max_j; b_src = "JSON"

        # Validate bounds
        if not isinstance(min_v, (int, float)) or not isinstance(max_v, (int, float)):
            logger.warning(f"Param '{name}': Invalid min/max types ({type(min_v)}, {type(max_v)}). Fixing @ default.");
            if default is not None: fixed_params[name] = default
            continue

        if min_v >= max_v - 1e-9:
             fix_val = default if default is not None else min_v
             if p_type is int: fix_val = int(round(fix_val))
             logger.warning(f"Param '{name}': min ({min_v}) >= max ({max_v}). Fixing @ {fix_val}.");
             fixed_params[name] = fix_val; continue

        low, high = min_v, max_v
        if p_type is int:
            # Ensure integer bounds are distinct after potential rounding/conversion
            low_int, high_int = int(np.floor(low)), int(np.ceil(high))
            if low_int >= high_int: high_int = low_int + 1
            # Check strict minimums (e.g., period >= 2)
            strict_min_2 = name in period_params and name.lower() not in ['mom', 'roc', 'rocp', 'rocr', 'rocr100', 'atr', 'natr', 'beta', 'correl', 'signalperiod', 'fastk_period', 'slowk_period', 'slowd_period', 'fastd_period', 'tenkan', 'kijun', 'senkou', 'timeperiod1', 'timeperiod2', 'timeperiod3']
            min_required = 2 if strict_min_2 else 1
            low_int = max(min_required, low_int)
            if low_int >= high_int: high_int = low_int + 1 # Re-check after applying min_required
            search_space.append(Integer(low=low_int, high=high_int, name=name))
            param_names.append(name); has_tunable = True; logger.debug(f"Space '{name}': Int({low_int},{high_int}) ({b_src})")
        elif p_type is float:
             # Apply specific bounds based on param type if min/max were not defined (less likely now)
             if name in factor_params: low = max(0.01, low)
             elif name in dev_scalar_params: low = max(0.1, low)
             # Ensure low < high after applying constraints
             if low >= high: high = low + 0.01
             search_space.append(Real(low=float(low), high=float(high), prior='uniform', name=name))
             param_names.append(name); has_tunable = True; logger.debug(f"Space '{name}': Real({float(low):.4f},{float(high):.4f}) ({b_src})")

    return search_space, param_names, fixed_params, has_tunable

def _process_default_config_fast_eval(
    indicator_name, indicator_def, fixed_params, param_names, db_path,
    master_evaluated_configs, base_data, max_lag, shifted_closes_cache,
    # --> Context <--
    symbol: str, timeframe: str, data_daterange: str, source_db_name: str
) -> Optional[int]: # Return config ID or None
    """Evaluates default config using FAST eval, populates caches, AND checks leaderboard."""
    param_defs = indicator_def.get('parameters', {})
    # Get defaults for tunable params + fixed params
    defaults_opt = {p: param_defs[p]['default'] for p in param_names if 'default' in param_defs.get(p, {})}
    defaults_full = {**fixed_params, **defaults_opt}; default_id = None

    # Check if default is valid according to conditions
    if not defaults_full or not parameter_generator.evaluate_conditions(defaults_full, indicator_def.get('conditions', [])):
        logger.warning(f"Default params missing/invalid for {indicator_name}. Cannot pre-evaluate."); return None

    default_hash = _get_config_hash(defaults_full)
    # Check if already processed (e.g., by previous lag's objective call)
    if default_hash in master_evaluated_configs:
         default_id = master_evaluated_configs[default_hash]['config_id']
         logger.debug(f"Default config for {indicator_name} already in master list (ID: {default_id}).")
         # Still ensure its correlations are cached and leaderboard checked
    else:
        # Get or create ID if not in master list yet
        conn = sqlite_manager.create_connection(db_path)
        if not conn: logger.error("Cannot connect DB for default config."); return None
        try:
            default_id = sqlite_manager.get_or_create_indicator_config_id(conn, indicator_name, defaults_full)
            if default_id is not None:
                 master_evaluated_configs[default_hash] = {'params': defaults_full, 'config_id': default_id}
                 logger.debug(f"Default added to master (ID: {default_id}): {defaults_full}")
            else: logger.error(f"Failed to create DB entry for default config: {defaults_full}"); return None
        except Exception as e:
            logger.error(f"Failed process default for {indicator_name}: {e}", exc_info=True); return None
        finally: conn.close()

    # --- Pre-calculate single correlations for default AND check leaderboard ---
    if default_id is None: return None # Should not happen if logic above is correct
    logger.debug(f"Pre-calculating single corrs & checking leaderboard for default config ID {default_id}...")
    any_updated_leaderboard = False
    for lag in range(1, max_lag + 1):
         cache_key = (default_hash, lag); corr = None
         if cache_key in single_correlation_cache:
             corr = single_correlation_cache[cache_key]
             # logger.debug(f"Default (ID:{default_id}) Lag {lag}: Found in single cache: {corr}") # Reduce noise
         else:
             shifted_close = shifted_closes_cache.get(lag)
             if shifted_close is not None:
                 corr = _calculate_correlation_at_target_lag(defaults_full, default_id, indicator_name, lag, base_data, shifted_close)
                 single_correlation_cache[cache_key] = corr # Cache result (even if None)
                 # logger.debug(f"Default (ID:{default_id}) Lag {lag}: Calculated single corr: {corr}") # Reduce noise
             else:
                 single_correlation_cache[cache_key] = None
                 logger.warning(f"Shifted close missing for default lag {lag}.")

         # Check leaderboard regardless of whether it was cached or calculated now
         if corr is not None and pd.notna(corr):
             try:
                 updated = leaderboard_manager.check_and_update_single_lag(
                     lag=lag, correlation_value=float(corr), indicator_name=indicator_name,
                     params=defaults_full, config_id=default_id, symbol=symbol,
                     timeframe=timeframe, data_daterange=data_daterange, source_db_name=source_db_name
                 )
                 if updated: any_updated_leaderboard = True
             except Exception as lb_err:
                 logger.error(f"Error checking leaderboard for default (Lag {lag}, ID {default_id}): {lb_err}", exc_info=True)

    # Trigger export *once* after checking all lags for default if any update occurred
    if any_updated_leaderboard:
        try:
            if leaderboard_manager.export_leaderboard_to_text(): logger.debug("Leaderboard exported after default eval update.")
            else: logger.error("Failed export leaderboard after default eval.")
        except Exception as ex_err: logger.error(f"Error exporting leaderboard: {ex_err}", exc_info=True)

    logger.debug(f"Finished pre-calc/leaderboard check for default ID {default_id}.")
    return default_id


# --- Deprecated --- -> No longer used as initial points are evaluated directly
# def _prepare_initial_point_fast_eval(default_config_id, default_params_full, param_names, target_lag):
#     """Prepares x0, y0 using the single_correlation_cache."""
#     x0, y0 = None, None
#     if default_config_id is not None and default_params_full:
#         default_hash = _get_config_hash(default_params_full)
#         cache_key = (default_hash, target_lag)
#         if cache_key in single_correlation_cache:
#             corr = single_correlation_cache[cache_key]
#             if corr is not None and pd.notna(corr):
#                  obj = -abs(corr)
#                  if all(p in default_params_full for p in param_names):
#                      try:
#                          x0_vals = [default_params_full[p] for p in param_names]
#                          x0 = [x0_vals]
#                          y0 = [obj]
#                      except KeyError as ke:
#                          logger.error(f"KeyError preparing x0 for default lag {target_lag}: {ke}. Params: {default_params_full}, Names: {param_names}")
#                          x0, y0 = None, None
#                  else: logger.warning(f"Lag {target_lag}: Default missing tunable params for x0.")
#             else: logger.debug(f"Lag {target_lag}: Default corr NaN/None in single cache. No x0.")
#         else: logger.debug(f"Lag {target_lag}: Default corr not found in single cache. No x0.")
#     return x0, y0


def _log_final_progress(indicator_name, progress_state):
    """Logs final optimization progress summary."""
    elapsed_td = timedelta(seconds=time.time() - progress_state['start_time'])
    rate = (progress_state['calls_completed'] / elapsed_td.total_seconds()) if elapsed_td.total_seconds() > 1 else 0
    elapsed_str = utils.format_duration(elapsed_td)
    logger.info(f"Progress ({indicator_name}): Opt finished. {progress_state['calls_completed']}/{progress_state['total_expected_calls']} evals. Avg Rate: {rate:.1f} evals/s. Total Time: {elapsed_str}.")

def _format_final_evaluated_configs(indicator_name, master_evaluated_configs):
    """Formats evaluated configs into the expected list structure."""
    final_list = []
    for cfg_hash, cfg_data in master_evaluated_configs.items():
         if isinstance(cfg_data, dict) and 'params' in cfg_data and 'config_id' in cfg_data:
              # Ensure config_id is an integer
              cfg_id = cfg_data['config_id']
              if isinstance(cfg_id, int):
                   final_list.append({'indicator_name': indicator_name, 'params': cfg_data['params'], 'config_id': cfg_id})
              else:
                   logger.warning(f"Invalid config_id type ({type(cfg_id)}) for hash {cfg_hash}. Skipping.")
         else: logger.warning(f"Master config incomplete/invalid: hash {cfg_hash}. Skipping.")
    return final_list

def _log_optimization_summary(indicator_name, max_lag, best_config_per_lag, final_all_evaluated_configs):
    """Logs summary of optimization results."""
    logger.info(f"--- Bayesian Opt Per-Lag Finished: {indicator_name} ---")
    logger.info(f"Evaluated {len(final_all_evaluated_configs)} unique configs in total for this indicator.")
    found_count = sum(1 for v in best_config_per_lag.values() if v is not None)
    logger.info(f"Found best configs for {found_count}/{max_lag} lags.")
    # Optional: Log details of the best configs found per lag if needed (can be verbose)
    # for lag, best_info in sorted(best_config_per_lag.items()):
    #     if best_info:
    #          logger.debug(f"  Lag {lag}: ID={best_info['config_id']}, Corr={best_info.get('correlation_at_lag'):.4f}, Params={best_info['params']}")
    #     else:
    #          logger.debug(f"  Lag {lag}: No best config found/valid.")