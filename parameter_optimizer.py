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

# --- Bayesian Optimization Imports ---
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    gp_minimize = Real = Integer = Categorical = use_named_args = None
    warnings.warn("scikit-optimize (skopt) not installed. Bayesian Optimization path will not be available.", ImportWarning)
# --- End Bayesian Optimization Imports ---

import config as app_config
import indicator_factory
# Removed correlation_calculator import as the final calc moves to main
import sqlite_manager
import utils
import parameter_generator
import leaderboard_manager # Import leaderboard manager

logger = logging.getLogger(__name__)

# --- Caches ---
# These caches are now intended to be cleared *before* each indicator optimization run in main.py
indicator_series_cache: Dict[str, pd.DataFrame] = {} # param_hash -> indicator_output_df
single_correlation_cache: Dict[Tuple[str, int], Optional[float]] = {} # (param_hash, lag) -> correlation_value

# --- Helper functions ---
def _get_config_hash(params: Dict[str, Any]) -> str:
    """Generates a stable SHA256 hash for a parameter dictionary."""
    config_str = json.dumps(params, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()

# --- Core Calculation Helpers (Revised Caching) ---
def _calculate_correlation_at_target_lag(
    params: Dict[str, Any], config_id: int, indicator_name: str, target_lag: int,
    base_data_with_required: pd.DataFrame, shifted_close_at_lag: Optional[pd.Series]
) -> Optional[float]:
    """Computes indicator (using cache) and correlation ONLY for the target_lag."""
    param_hash = _get_config_hash(params)
    logger.debug(f"Calculating single corr: ID {config_id} (Hash: {param_hash[:8]}), Lag {target_lag}")

    # 1. Get Indicator Series (Cache check)
    indicator_output_df = None; cache_hit = False
    if param_hash in indicator_series_cache:
        indicator_output_df = indicator_series_cache[param_hash]
        if not isinstance(indicator_output_df, pd.DataFrame):
             logger.warning(f"Invalid indicator cache item for {param_hash[:8]}. Recomputing."); indicator_output_df = None; del indicator_series_cache[param_hash]
        else: cache_hit = True; logger.debug(f"Indicator cache HIT: ID {config_id} / hash {param_hash[:8]}.")
    if indicator_output_df is None: # Cache miss or invalid
        logger.debug(f"Indicator cache MISS: ID {config_id} / hash {param_hash[:8]}. Computing...")
        config_details = {'indicator_name': indicator_name, 'params': params, 'config_id': config_id}
        # Pass a copy here to be safe inside the optimizer loop, although factory should handle it
        indicator_output_df = indicator_factory._compute_single_indicator(base_data_with_required.copy(), config_details)
        indicator_series_cache[param_hash] = indicator_output_df if (indicator_output_df is not None and not indicator_output_df.empty) else pd.DataFrame() # Store empty DF if failed

    # Check result
    if indicator_output_df is None or indicator_output_df.empty:
        logger.warning(f"Indicator {'cache' if cache_hit else 'compute'} failed/empty for ID {config_id}.")
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
            # Use pandas corr method directly
            correlation = indicator_series.corr(shifted_close_at_lag)
            if pd.notna(correlation):
                current_abs = abs(float(correlation))
                if current_abs > max_abs_corr: max_abs_corr = current_abs; best_signed_corr_at_lag = float(correlation)
        except Exception as e: logger.error(f"Error calculating single corr for {col}, lag {target_lag}: {e}", exc_info=True)

    logger.debug(f"Single corr calculated: ID {config_id}, Lag {target_lag}. Result: {best_signed_corr_at_lag}")
    # Return NaN if no valid correlation found (max_abs_corr remains -1.0)
    return best_signed_corr_at_lag if max_abs_corr > -1.0 else np.nan

# --- REMOVED: _calculate_all_correlations_for_config_post_opt ---
# This calculation will now happen once in main.py using the parallel processor

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
        logger.warning(f"ObjFn ({indicator_name}): Invalid params: {params_dict} (Full: {full_params}). High cost."); return 1e6

    # --- Cache Check / Calculation ---
    params_to_store = full_params; param_hash = _get_config_hash(params_to_store)
    cache_key = (param_hash, target_lag); correlation_value = None; cache_hit = False; config_id = None

    if cache_key in single_correlation_cache:
        correlation_value = single_correlation_cache[cache_key]; cache_hit = True
        config_id = master_evaluated_configs.get(param_hash, {}).get('config_id')
        if config_id is None: logger.error(f"CRITICAL Cache Inconsistency: Hash {param_hash[:8]} lag {target_lag} in single cache but not master list!")
        logger.debug(f"ObjFn ({indicator_name} Lag {target_lag}): Cached single corr: ID {config_id} / hash {param_hash[:8]}. Val: {correlation_value}")
    else: # Cache Miss
        cache_hit = False; logger.debug(f"ObjFn ({indicator_name} Lag {target_lag}): Single corr cache MISS: hash {param_hash[:8]}.")
        # Get Config ID
        if param_hash in master_evaluated_configs: config_id = master_evaluated_configs[param_hash]['config_id']
        else:
            conn_temp = sqlite_manager.create_connection(db_path)
            if conn_temp:
                try:
                    # Ensure ID creation happens within a transaction for safety
                    config_id = sqlite_manager.get_or_create_indicator_config_id(conn_temp, indicator_name, params_to_store)
                    master_evaluated_configs[param_hash] = {'params': params_to_store, 'config_id': config_id}
                    logger.info(f"Added new config to master: '{indicator_name}' (ID: {config_id}), Params: {params_to_store}")
                except Exception as e: logger.error(f"ObjFn ({indicator_name}): Failed get/create config ID: {e}", exc_info=True)
                finally: conn_temp.close()
            else: logger.error(f"ObjFn ({indicator_name}): Cannot connect DB for config ID.")
        if config_id is None: logger.error(f"ObjFn ({indicator_name}): Failed get config ID. High cost."); return 1e6

        # Calculate Single Correlation
        shifted_close = shifted_closes_global_cache.get(target_lag)
        if shifted_close is None: logger.error(f"ObjFn ({indicator_name} Lag {target_lag}): Shifted close missing!"); correlation_value = None
        else: correlation_value = _calculate_correlation_at_target_lag(params_to_store, config_id, indicator_name, target_lag, base_data_with_required, shifted_close)
        single_correlation_cache[cache_key] = correlation_value # Cache result

    # --- Progress Reporting ---
    progress_state['calls_completed'] += 1; calls_done = progress_state['calls_completed']
    calls_total = progress_state['total_expected_calls']; current_time = time.time()
    elapsed = current_time - progress_state['start_time']; rate = (calls_done / elapsed) if elapsed > 0 else 0
    percent = (calls_done / calls_total * 100) if calls_total > 0 else 0
    # Log less frequently
    if calls_done % 10 == 0 or calls_done == 1 or calls_done == calls_total:
        logger.info(f"Progress ({indicator_name} L{target_lag}): Eval {calls_done}/{calls_total} ({percent:.1f}%) | Rate: {rate:.1f} evals/s | SingleCorrCacheHit: {'Yes' if cache_hit else 'No'} | Elapsed: {elapsed:.1f}s")
    progress_state['last_report_time'] = current_time

    # --- Check & Update Leaderboard IMMEDIATELY ---
    # This is the key change for real-time updates during Tweak path
    if config_id is not None and pd.notna(correlation_value):
        try:
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
            # Optional: Trigger text file export less frequently if needed
            # Add a counter or timer check here if export needs throttling
            # Example: Export every 50 updates or every 10 seconds
            # For simplicity now, export only if updated (could be frequent)
            if updated:
                try:
                    if leaderboard_manager.export_leaderboard_to_text():
                        logger.debug(f"Leaderboard text exported after update (Lag {target_lag}, ID {config_id}).")
                    else:
                        logger.error("Failed to export leaderboard text after update.")
                except Exception as ex_err:
                    logger.error(f"Error exporting leaderboard text: {ex_err}", exc_info=True)

        except Exception as lb_err:
            logger.error(f"Error during immediate leaderboard update check (Lag {target_lag}, ID {config_id}): {lb_err}", exc_info=True)

    # --- Determine Objective Score ---
    if pd.notna(correlation_value) and isinstance(correlation_value, (float, int, np.number)):
         score = abs(float(correlation_value)); objective_value = -score
         log_id = config_id if config_id is not None else "N/A"
         logger.debug(f"  -> ObjFn ({indicator_name} L{target_lag}): ID {log_id} -> Score={score:.4f} (Corr={correlation_value:.4f}). Objective={objective_value:.4f}")
         return objective_value
    else:
         log_id = config_id if config_id is not None else "N/A"
         logger.warning(f"  -> ObjFn ({indicator_name} L{target_lag}): ID {log_id} ({params_to_store}) -> Failed/NaN corr. High cost.")
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
    # REMOVED: final_correlations_to_batch_insert (will be done in main)
    best_config_per_lag: Dict[int, Dict[str, Any]] = {} # Lag -> best result dict (still useful for logging)

    progress_state = {'calls_completed': 0, 'start_time': time.time(), 'last_report_time': time.time(), 'total_expected_calls': max_lag * n_calls_per_lag}

    # --- Pre-calculate Shifted Closes ---
    logger.info(f"Pre-calc {max_lag} shifted 'close' series..."); start_shift = time.time()
    if 'close' not in base_data_with_required.columns: logger.error("'close' missing."); return {}, []
    close_series = base_data_with_required['close'].astype(float)
    shifted_closes_global_cache = {lag: close_series.shift(-lag) for lag in range(1, max_lag + 1)}
    logger.info(f"Shifted closes pre-calc complete: {time.time() - start_shift:.2f}s.")

    # --- Clear Caches ---
    # Clearing is now done in main.py *before* calling this function for an indicator
    # logger.info(f"Caches cleared before optimizing {indicator_name}.")

    # --- Evaluate Default Config (Fast Eval) ---
    # This pre-populates caches and ensures default is considered
    default_config_id = _process_default_config_fast_eval(
        indicator_name, indicator_def, fixed_params, param_names, db_path,
        master_evaluated_configs, base_data_with_required, max_lag,
        shifted_closes_global_cache,
        # --> Pass Context <--
        symbol, timeframe, data_daterange, source_db_name
    )
    default_params_full = None
    if default_config_id is not None: # Find params for default ID
        for cfg_data in master_evaluated_configs.values():
            if cfg_data.get('config_id') == default_config_id: default_params_full = cfg_data.get('params'); break

    # --- Optimization Loop per Lag ---
    for target_lag in range(1, max_lag + 1):
        logger.info(f"--- Bayesian Opt for Lag: {target_lag}/{max_lag} ---")
        # Use partial to bind arguments that don't change per call
        objective_partial = partial(_objective_function,
                                    param_names=param_names, target_lag=target_lag, indicator_name=indicator_name,
                                    indicator_def=indicator_def, base_data_with_required=base_data_with_required,
                                    db_path=db_path, symbol_id=symbol_id, timeframe_id=timeframe_id,
                                    master_evaluated_configs=master_evaluated_configs, progress_state=progress_state,
                                    shifted_closes_global_cache=shifted_closes_global_cache,
                                    # ----> BIND NEW PARAMETERS <----
                                    symbol=symbol, timeframe=timeframe, data_daterange=data_daterange, source_db_name=source_db_name
                                    )
        try:
             with warnings.catch_warnings():
                  warnings.simplefilter("ignore", category=UserWarning)
                  warnings.simplefilter("ignore", category=RuntimeWarning)
                  x0, y0 = _prepare_initial_point_fast_eval(default_config_id, default_params_full, param_names, target_lag)
                  if x0: logger.debug(f"Lag {target_lag}: Using default as x0={x0}, y0=[{y0[0]:.4f}]")
                  else: logger.debug(f"Lag {target_lag}: No valid initial point (x0) from default.")

                  # Ensure n_initial_points is not larger than n_calls
                  current_n_initial = min(n_initial_points_per_lag, n_calls_per_lag)
                  if x0 is not None and current_n_initial > 0:
                      current_n_initial = max(0, current_n_initial - 1) # Reduce random points if x0 is used

                  result = gp_minimize(func=objective_partial, dimensions=search_space,
                                       acq_func=app_config.DEFAULTS["optimizer_acq_func"],
                                       n_calls=n_calls_per_lag, n_initial_points=current_n_initial, # Use adjusted initial points
                                       x0=x0, y0=y0, random_state=None, noise='gaussian') # Consider noise='gaussian'

             best_params_list = result.x; best_objective_value = result.fun
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
                     best_corr = single_correlation_cache.get((best_hash, target_lag)) # Get from cache
                     logger.info(f"--- Lag {target_lag} Best (Opt {indicator_name}) --- ID: {best_config_id}, Params: {best_params_full}, Score@Lag: {best_score:.6f}, Corr@Lag: {best_corr if best_corr is not None else 'N/A'}")
                     # Store best config found for this lag (useful for logging/summary)
                     best_config_per_lag[target_lag] = {'params': best_params_full, 'config_id': best_config_id, 'correlation_at_lag': best_corr, 'score_at_lag': best_score}
                 else: logger.error(f"Lag {target_lag}: Could not find config ID for best params {best_params_full}."); best_config_per_lag[target_lag] = None
             else: # Opt failed, check default
                 logger.warning(f"Lag {target_lag}: Opt failed for {indicator_name}. Checking default.")
                 if default_config_id is not None and default_params_full:
                     default_hash = _get_config_hash(default_params_full)
                     default_corr = single_correlation_cache.get((default_hash, target_lag))
                     if default_corr is not None and pd.notna(default_corr):
                         logger.info(f"Lag {target_lag}: Falling back to default config ID {default_config_id} for {indicator_name}.")
                         best_config_per_lag[target_lag] = {'params': default_params_full, 'config_id': default_config_id, 'correlation_at_lag': default_corr, 'score_at_lag': abs(default_corr)}
                     else: logger.warning(f"Lag {target_lag}: Default ({default_config_id}) for {indicator_name} had NaN/None corr. No solution."); best_config_per_lag[target_lag] = None
                 else: logger.warning(f"Lag {target_lag}: No default config available/evaluated for {indicator_name}. No solution."); best_config_per_lag[target_lag] = None
        except Exception as opt_err: logger.error(f"Error during opt for {indicator_name} lag {target_lag}: {opt_err}", exc_info=True); best_config_per_lag[target_lag] = None
    # End lag loop

    _log_final_progress(indicator_name, progress_state)

    # --- POST-OPTIMIZATION Calculation REMOVED ---
    # The final calculation of all correlations for all evaluated configs will happen in main.py

    # Format and return evaluated configurations
    final_all_evaluated_configs = _format_final_evaluated_configs(indicator_name, master_evaluated_configs)
    _log_optimization_summary(indicator_name, max_lag, best_config_per_lag, final_all_evaluated_configs)
    # Return best per lag mainly for logging, and the crucial list of evaluated configs
    return best_config_per_lag, final_all_evaluated_configs


# --- Helper Functions for Bayesian Opt ---
def _define_search_space(param_defs: Dict) -> Tuple[List, List[str], Dict, bool]:
    """Helper to define the search space for skopt."""
    search_space = []; param_names = []; fixed_params = {}; has_tunable = False
    period_params = ['fast', 'slow', 'fastperiod', 'slowperiod', 'signalperiod', 'timeperiod', 'timeperiod1', 'timeperiod2', 'timeperiod3', 'length', 'window', 'obv_period', 'price_period', 'fastk_period', 'slowk_period', 'slowd_period', 'fastd_period', 'tenkan', 'kijun', 'senkou']
    factor_params = ['fastlimit','slowlimit','acceleration','maximum','vfactor','smoothing']
    dev_scalar_params = ['scalar','nbdev','nbdevup','nbdevdn']
    for name, details in sorted(param_defs.items()):
        default = details.get('default'); min_j = details.get('min'); max_j = details.get('max')
        p_type = type(default) if default is not None else (type(min_j) if min_j is not None else None)
        is_num_tune = isinstance(p_type, type) and issubclass(p_type, (int, float)) and ((min_j is not None and max_j is not None) or default is not None)
        if not is_num_tune:
            if default is not None: fixed_params[name] = default; logger.debug(f"Param '{name}': Fixed @ '{default}'.")
            else: logger.warning(f"Param '{name}': No default/bounds. Skipping."); continue
            continue
        min_v = min_j; max_v = max_j; b_src = "JSON"
        if min_v is None:
            b_src += "/Fallback(min)"; min_v = 1 # Generic fallback
            if issubclass(p_type, int): strict_min_2 = name in period_params and name.lower() not in ['mom', 'roc', 'rocp', 'rocr', 'rocr100', 'atr', 'natr', 'beta', 'correl', 'signalperiod', 'fastk_period', 'slowk_period', 'slowd_period', 'fastd_period', 'tenkan', 'kijun', 'senkou', 'timeperiod1', 'timeperiod2', 'timeperiod3']; min_v = 2 if strict_min_2 else 1
            elif issubclass(p_type, float): min_v = 0.01 if name in factor_params else (0.1 if name in dev_scalar_params else (2.0 if name in period_params else 0.0))
        if max_v is None:
            b_src += "/Fallback(max)"; max_v = 100 # Generic fallback
            if default is None: logger.warning(f"Param '{name}': No default for fallback max. Fixing @ min {min_v}."); fixed_params[name] = min_v; continue
            if issubclass(p_type, int): max_v = max(min_v + 1, int(default * 3.0), 200)
            elif issubclass(p_type, float): max_v = 1.0 if name in factor_params else (max(min_v + 0.1, default * 3.0, 5.0) if name in dev_scalar_params else (max(min_v + 1.0, default * 3.0, 200.0) if name in period_params else max(min_v + 0.1, default * 3.0)))
        if min_v >= max_v - 1e-9: logger.warning(f"Param '{name}': min ({min_v}) >= max ({max_v}). Fixing @ min."); fixed_params[name] = min_v; continue
        low, high = min_v, max_v
        if issubclass(p_type, int): search_space.append(Integer(low=int(low), high=int(high), name=name)); param_names.append(name); has_tunable = True; logger.debug(f"Space '{name}': Int({int(low)},{int(high)}) ({b_src})")
        elif issubclass(p_type, float): search_space.append(Real(low=float(low), high=float(high), prior='uniform', name=name)); param_names.append(name); has_tunable = True; logger.debug(f"Space '{name}': Real({float(low):.4f},{float(high):.4f}) ({b_src})")
    return search_space, param_names, fixed_params, has_tunable

def _process_default_config_fast_eval(
    indicator_name, indicator_def, fixed_params, param_names, db_path,
    master_evaluated_configs, base_data, max_lag, shifted_closes_cache,
    # --> Context <--
    symbol: str, timeframe: str, data_daterange: str, source_db_name: str
) -> Optional[int]:
    """Evaluates default config using FAST eval, populates caches, AND checks leaderboard."""
    param_defs = indicator_def.get('parameters', {})
    defaults_opt = {p: param_defs[p]['default'] for p in param_names if 'default' in param_defs.get(p, {})}
    defaults_full = {**fixed_params, **defaults_opt}; default_id = None
    if defaults_full and parameter_generator.evaluate_conditions(defaults_full, indicator_def.get('conditions', [])):
        default_hash = _get_config_hash(defaults_full); conn = sqlite_manager.create_connection(db_path)
        if conn:
            try:
                if default_hash not in master_evaluated_configs:
                    default_id = sqlite_manager.get_or_create_indicator_config_id(conn, indicator_name, defaults_full)
                    master_evaluated_configs[default_hash] = {'params': defaults_full, 'config_id': default_id}
                    logger.info(f"Default added to master (ID: {default_id}): {defaults_full}")
                else: default_id = master_evaluated_configs[default_hash]['config_id']
                if default_id is not None: # Pre-calc single corrs for default AND check leaderboard
                    logger.debug("Pre-calc single corrs for default and check leaderboard...")
                    any_updated = False
                    for lag in range(1, max_lag + 1):
                         cache_key = (default_hash, lag); corr = None
                         if cache_key in single_correlation_cache:
                             corr = single_correlation_cache[cache_key]
                         else:
                             shifted_close = shifted_closes_cache.get(lag)
                             if shifted_close is not None:
                                 corr = _calculate_correlation_at_target_lag(defaults_full, default_id, indicator_name, lag, base_data, shifted_close)
                                 single_correlation_cache[cache_key] = corr
                             else:
                                 single_correlation_cache[cache_key] = None # Ensure cache entry exists

                         # Check leaderboard for this default correlation
                         if corr is not None and pd.notna(corr):
                             try:
                                 updated = leaderboard_manager.check_and_update_single_lag(
                                     lag=lag, correlation_value=float(corr), indicator_name=indicator_name,
                                     params=defaults_full, config_id=default_id, symbol=symbol,
                                     timeframe=timeframe, data_daterange=data_daterange, source_db_name=source_db_name
                                 )
                                 if updated: any_updated = True
                             except Exception as lb_err:
                                 logger.error(f"Error checking leaderboard for default (Lag {lag}, ID {default_id}): {lb_err}", exc_info=True)
                    # Export leaderboard text if default caused any update
                    if any_updated:
                        try:
                            if leaderboard_manager.export_leaderboard_to_text(): logger.debug("Leaderboard exported after default eval.")
                            else: logger.error("Failed export leaderboard after default eval.")
                        except Exception as ex_err: logger.error(f"Error exporting leaderboard: {ex_err}", exc_info=True)

                    logger.debug("Finished pre-calc/leaderboard check for default.")
            except Exception as e: logger.error(f"Failed process default for {indicator_name}: {e}", exc_info=True); default_id = None
            finally: conn.close()
        else: logger.error("Cannot connect DB for default config.")
    else: logger.warning(f"Default params missing/invalid for {indicator_name}.")
    return default_id


def _prepare_initial_point_fast_eval(default_config_id, default_params_full, param_names, target_lag):
    """Prepares x0, y0 using the single_correlation_cache."""
    x0, y0 = None, None
    if default_config_id is not None and default_params_full:
        default_hash = _get_config_hash(default_params_full)
        cache_key = (default_hash, target_lag)
        if cache_key in single_correlation_cache:
            corr = single_correlation_cache[cache_key]
            if corr is not None and pd.notna(corr):
                 obj = -abs(corr)
                 if all(p in default_params_full for p in param_names):
                     # Ensure correct order and handle potential missing tunable params (shouldn't happen if default_params_full is correct)
                     try:
                         x0_vals = [default_params_full[p] for p in param_names]
                         x0 = [x0_vals] # Needs to be list of lists
                         y0 = [obj] # Needs to be a list
                     except KeyError as ke:
                         logger.error(f"KeyError preparing x0 for default lag {target_lag}: {ke}. Params: {default_params_full}, Names: {param_names}")
                         x0, y0 = None, None # Fail safe
                 else: logger.warning(f"Lag {target_lag}: Default missing tunable params for x0.")
            else: logger.debug(f"Lag {target_lag}: Default corr NaN/None in single cache. No x0.")
        else: logger.debug(f"Lag {target_lag}: Default corr not found in single cache. No x0.")
    return x0, y0

def _log_final_progress(indicator_name, progress_state):
    """Logs final optimization progress summary."""
    elapsed = time.time() - progress_state['start_time']; rate = (progress_state['calls_completed'] / elapsed) if elapsed > 0 else 0
    logger.info(f"Progress ({indicator_name}): Opt finished. {progress_state['calls_completed']}/{progress_state['total_expected_calls']} evals. Avg Rate: {rate:.1f} evals/s. Time: {elapsed:.1f}s.")

# REMOVED: _perform_final_batch_insert (will be done in main)

def _format_final_evaluated_configs(indicator_name, master_evaluated_configs):
    """Formats evaluated configs into the expected list structure."""
    final_list = []
    for cfg_hash, cfg_data in master_evaluated_configs.items():
         if isinstance(cfg_data, dict) and 'params' in cfg_data and 'config_id' in cfg_data:
              final_list.append({'indicator_name': indicator_name, 'params': cfg_data['params'], 'config_id': cfg_data['config_id']})
         else: logger.warning(f"Master config incomplete/invalid: hash {cfg_hash}. Skipping.")
    return final_list

def _log_optimization_summary(indicator_name, max_lag, best_config_per_lag, final_all_evaluated_configs):
    """Logs summary of optimization results."""
    logger.info(f"--- Bayesian Opt Per-Lag Finished: {indicator_name} ---")
    logger.info(f"Evaluated {len(final_all_evaluated_configs)} unique configs.")
    found_count = sum(1 for v in best_config_per_lag.values() if v is not None)
    logger.info(f"Found best configs for {found_count}/{max_lag} lags.")