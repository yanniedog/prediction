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
from functools import partial
from datetime import datetime, timedelta
from collections import defaultdict

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

import config as app_config
import indicator_factory
import sqlite_manager
import utils
import parameter_generator
import leaderboard_manager
# Import main is no longer needed as we pass the display function directly
# import main

logger = logging.getLogger(__name__)

# --- Caches ---
indicator_series_cache: Dict[str, pd.DataFrame] = {} # param_hash -> indicator_output_df
single_correlation_cache: Dict[Tuple[str, int], Optional[float]] = {} # (param_hash, lag) -> correlation_value
indicator_factory_instance = indicator_factory.IndicatorFactory() # Create singleton instance

# --- Constants ---
WEAK_CORR_THRESHOLD_SKIP = app_config.DEFAULTS.get("weak_corr_threshold_skip", 0.15)
ETA_UPDATE_INTERVAL_SECONDS = app_config.DEFAULTS.get("eta_update_interval_seconds", 15)
MAX_TOTAL_OPT_FAILURES_INDICATOR = app_config.DEFAULTS.get("max_total_opt_failures_indicator", 50)


# --- Helper functions ---
def _get_config_hash(params: Dict[str, Any]) -> str:
    """Generates a stable SHA256 hash for a parameter dictionary."""
    return utils.get_config_hash(params)


# --- Objective Function (Accepts global timing info) ---
def _objective_function(params_list: List[Any], *, # Use keyword-only args after params_list
                         param_names: List[str], target_lag: int, indicator_name: str,
                         indicator_def: Dict, base_data_with_required: pd.DataFrame,
                         db_path: str, symbol_id: int, timeframe_id: int,
                         master_evaluated_configs: Dict[str, Dict[str, Any]],
                         # --- Modified progress state ---
                         progress_info: Dict[str, Any], # Now contains timing, steps etc.
                         # ------------------------------
                         shifted_closes_global_cache: Dict[int, pd.Series],
                         failure_tracker: Dict[str, int],
                         interim_correlations_accumulator: Dict[int, List[Optional[float]]],
                         max_lag_for_accumulator: int,
                         symbol: str, timeframe: str, data_daterange: str, source_db_name: str
                         ) -> float:
    """
    Objective function for Bayesian opt.
    Calculates indicator, correlation for target lag, updates caches & accumulator,
    checks leaderboard, updates GLOBAL progress, and returns negative absolute correlation.
    """
    # 1. Convert/Clean/Validate Params
    try:
        params_dict = dict(zip(param_names, params_list))
        for k, v in params_dict.items():
            if isinstance(v, np.integer): params_dict[k] = int(v)
            elif isinstance(v, np.floating): params_dict[k] = float(v)

        full_params = params_dict.copy()
        param_defs = indicator_def.get('parameters', {})
        if not param_defs:
            raise ValueError(f"Indicator '{indicator_name}' has no parameter definitions. Please check indicator_params.json.")

        for p_name, p_details in param_defs.items():
            if p_name not in full_params and 'default' in p_details:
                full_params[p_name] = p_details['default']
            elif p_name not in full_params and 'default' not in p_details:
                raise ValueError(f"Required parameter '{p_name}' missing for indicator '{indicator_name}' and no default value provided.")

        if not parameter_generator.evaluate_conditions(full_params, indicator_def.get('conditions', [])):
            invalid_params = []
            for condition_group in indicator_def.get('conditions', []):
                for param_name, ops_dict in condition_group.items():
                    if param_name in full_params:
                        param_value = full_params[param_name]
                        for op, value in ops_dict.items():
                            if not parameter_generator._evaluate_single_condition(param_value, op, value):
                                invalid_params.append(f"{param_name} {op} {value} (got {param_value})")
            raise ValueError(f"Parameter conditions not met for indicator '{indicator_name}': {', '.join(invalid_params)}")

    except Exception as e:
        failure_tracker[indicator_name] += 1
        logger.error(f"Parameter validation failed for {indicator_name}: {str(e)}")
        return 1e6  # High cost for invalid parameters

    # 2. Cache Check / Get Config ID
    try:
        params_to_store = full_params
        param_hash = _get_config_hash(params_to_store)
        cache_key = (param_hash, target_lag)
        correlation_value: Optional[float] = None
        cache_hit = False
        config_id: Optional[int] = None

        if cache_key in single_correlation_cache:
            correlation_value = single_correlation_cache[cache_key]
            cache_hit = True
            config_id = master_evaluated_configs.get(param_hash, {}).get('config_id')
            if config_id is None:
                logger.error(f"Cache inconsistency detected: Hash {param_hash[:8]} for lag {target_lag} has cached correlation but missing config ID. Attempting recovery...")
                conn_temp_rec = sqlite_manager.create_connection(db_path)
                if conn_temp_rec:
                    try:
                        config_id = sqlite_manager.get_or_create_indicator_config_id(conn_temp_rec, indicator_name, params_to_store)
                        if config_id is None:
                            raise ValueError(f"Failed to recover config ID for hash {param_hash[:8]}")
                    except Exception as e:
                        logger.error(f"Failed to recover config ID: {str(e)}")
                        raise
                    finally:
                        conn_temp_rec.close()
                else:
                    raise ValueError(f"Could not establish database connection to recover config ID for hash {param_hash[:8]}")

    except Exception as e:
        logger.error(f"Cache/database operation failed for {indicator_name}: {str(e)}")
        failure_tracker[indicator_name] += 1
        return 1e6

    # 3. Update Interim Accumulator
    if config_id not in interim_correlations_accumulator:
        # Ensure list is initialized with the correct length based on max lag for the entire run
        interim_correlations_accumulator[config_id] = [None] * max_lag_for_accumulator
    if 0 <= target_lag - 1 < max_lag_for_accumulator:
        interim_correlations_accumulator[config_id][target_lag - 1] = correlation_value
    else:
        # This case should be rare if max_lag_for_accumulator is correct
        logger.warning(f"Target lag {target_lag} out of bounds for accumulator (size {max_lag_for_accumulator}). Expanding list.")
        needed_len = target_lag
        current_len = len(interim_correlations_accumulator[config_id])
        if needed_len > current_len:
            interim_correlations_accumulator[config_id].extend([None] * (needed_len - current_len))
        if 0 <= target_lag - 1 < len(interim_correlations_accumulator[config_id]):
             interim_correlations_accumulator[config_id][target_lag - 1] = correlation_value
        else:
             logger.error(f"Failed to place correlation for lag {target_lag} in accumulator even after expansion.")


    # 4. Progress Reporting using GLOBAL context passed via progress_info
    progress_info['calls_completed_current_indicator'] += 1
    calls_done_ind = progress_info['calls_completed_current_indicator']
    calls_total_ind = progress_info['total_expected_calls_indicator']
    # Increment calls for the current lag specifically
    progress_info['calls_completed_current_lag'] = progress_info.get('calls_completed_current_lag', 0) + 1
    calls_done_lag = progress_info['calls_completed_current_lag']
    calls_total_lag = progress_info['n_calls_per_lag']

    current_time = time.time()
    # Throttle progress updates based on time interval or completion
    if current_time - progress_info.get('last_report_time', 0) > ETA_UPDATE_INTERVAL_SECONDS or calls_done_lag == 1 or calls_done_lag == calls_total_lag or calls_done_ind == calls_total_ind:
        # --- Calculate OVERALL Progress Step ---
        # Fraction done with current indicator (including progress within the current lag)
        frac_lag_done = calls_done_lag / calls_total_lag if calls_total_lag > 0 else 1.0
        frac_indicator = (progress_info['indicator_index'] + frac_lag_done) / progress_info['total_indicators_in_phase']
        # Map to overall step range for the optimization phase
        current_overall_step = progress_info['current_step_base'] + progress_info['total_steps_in_phase'] * frac_indicator
        # --- Call the main display function passed in progress_info ---
        stage_desc = f"Opt {indicator_name} L{target_lag} ({calls_done_lag}/{calls_total_lag})"
        progress_info['display_progress_func'](stage_desc, current_overall_step, progress_info['total_analysis_steps_global'])
        # --------------------------------------------------------------

        # Log details less frequently to file
        if calls_done_ind % progress_info.get('log_frequency', 50) == 0 or calls_done_lag == 1 or calls_done_lag == calls_total_lag:
             logger.info(f"Progress ({indicator_name} L{target_lag}): Eval {calls_done_lag}/{calls_total_lag} | Total Evals: {calls_done_ind}/{calls_total_ind} | CorrCacheHit: {'Yes' if cache_hit else 'No'}")
        progress_info['last_report_time'] = current_time

    # 5. Check & Update Leaderboard IMMEDIATELY
    if pd.notna(correlation_value):
        try:
            # Ensure config_id is valid before calling
            if config_id is not None:
                leaderboard_manager.check_and_update_single_lag(
                    lag=target_lag, correlation_value=float(correlation_value),
                    indicator_name=indicator_name, params=params_to_store, config_id=config_id,
                    symbol=symbol, timeframe=timeframe, data_daterange=data_daterange, source_db_name=source_db_name
                )
            else:
                logger.error(f"Cannot update leaderboard for lag {target_lag}, config ID is None.")
        except Exception as lb_err:
            logger.error(f"Error immediate LB update (Lag {target_lag}, ID {config_id}): {lb_err}", exc_info=True)

    # 6. Determine Objective Score (minimize negative absolute correlation)
    if pd.notna(correlation_value) and isinstance(correlation_value, (float, int, np.number)):
         # Ensure it's a standard float before taking abs
         score = abs(float(correlation_value))
         # Objective is to minimize the negative absolute correlation (maximizes absolute correlation)
         objective_value = -score
         return objective_value
    else:
         # Failed correlation calculation or NaN result
         failure_tracker[indicator_name] += 1
         return 1e6 # High cost for failure


# --- Main Bayesian Optimization Function (Accepts global timing info) ---
def optimize_parameters_bayesian_per_lag(
    indicator_name: str, indicator_def: Dict, base_data_with_required: pd.DataFrame,
    max_lag: int, n_calls_per_lag: int, n_initial_points_per_lag: int,
    db_path: str, symbol_id: int, timeframe_id: int,
    symbol: str, timeframe: str, data_daterange: str, source_db_name: str,
    interim_correlations_accumulator: Dict[int, List[Optional[float]]],
    # --- New Args ---
    analysis_start_time_global: float, # Global start time
    total_analysis_steps_global: int, # Total steps for whole run
    current_step_base: float,        # Base step for this optimization phase
    total_steps_in_phase: float,     # Total steps allocated to this phase
    indicator_index: int,           # Index of current indicator being optimized
    total_indicators_in_phase: int, # Total indicators in this phase
    display_progress_func: callable   # Main progress display function from main.py
    # ----------------
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    """Optimizes parameters using Bayesian Optimization per lag. Returns evaluated configs."""
    if not SKOPT_AVAILABLE:
        logger.error("skopt not installed. Bayesian optimization unavailable.")
        return {}, []
    logger.info(f"--- Starting Bayesian Opt Per-Lag for: {indicator_name} ---")
    logger.info(f"Lags: 1-{max_lag}, Evals/Lag: {n_calls_per_lag}, Initials: {n_initial_points_per_lag}")

    search_space, param_names, param_bounds, has_tunable = _define_search_space(indicator_def.get('parameters', {}))
    if not has_tunable:
        logger.warning(f"'{indicator_name}' has no tunable parameters defined for optimization. Evaluating default only.")
        master_evaluated_configs_temp: Dict[str, Dict[str, Any]] = {} # Need a temp map for default processing
        shifted_closes_global_cache_temp = {} # Need shifted cache even for default
        if 'close' in base_data_with_required.columns:
             close_series = base_data_with_required['close'].astype(float)
             shifted_closes_global_cache_temp = {lag: close_series.shift(-lag).reindex(base_data_with_required.index) for lag in range(1, max_lag + 1)}
        else: logger.error("Cannot evaluate default: 'close' column missing.")

        default_id = _process_default_config_fast_eval(
            indicator_name, indicator_def, param_bounds, param_names, db_path,
            master_evaluated_configs_temp, base_data_with_required, max_lag,
            shifted_closes_global_cache_temp,
            interim_correlations_accumulator, max_lag, # Pass max_lag as max_lag_for_accumulator
            symbol, timeframe, data_daterange, source_db_name
        )
        default_configs = []
        if default_id is not None:
             default_params = {k: v['default'] for k,v in indicator_def.get('parameters',{}).items() if 'default' in v}
             default_configs.append({'indicator_name': indicator_name, 'params': default_params, 'config_id': default_id})
        return {}, default_configs # Return empty best_per_lag, possibly default config list

    logger.info(f"Opt Space Dimensions ({len(param_names)}): {param_names}, Fixed: {param_bounds}")

    # --- Global Tracking ---
    master_evaluated_configs: Dict[str, Dict[str, Any]] = {} # Tracks all unique configs evaluated for this indicator
    best_config_per_lag: Dict[int, Dict[str, Any]] = {} # Tracks the best result found for each lag
    total_expected_calls_indicator = max_lag * n_calls_per_lag
    log_freq = max(1, total_expected_calls_indicator // 10) if total_expected_calls_indicator > 20 else 5
    # --- Setup progress_info dict to be passed to objective function ---
    progress_info = {
        'calls_completed_current_indicator': 0, # Tracks total calls for *this* indicator across all lags
        'last_report_time': time.time(),
        'total_expected_calls_indicator': total_expected_calls_indicator,
        'log_frequency': log_freq,
        'n_calls_per_lag': n_calls_per_lag,
        # --- Global context ---
        'analysis_start_time_global': analysis_start_time_global,
        'total_analysis_steps_global': total_analysis_steps_global,
        'current_step_base': current_step_base,
        'total_steps_in_phase': total_steps_in_phase,
        'indicator_index': indicator_index,
        'total_indicators_in_phase': total_indicators_in_phase,
        'display_progress_func': display_progress_func
    }
    # ------------------------------------------------------------------
    failure_tracker = defaultdict(int) # Tracks failures per indicator

    # --- Pre-calculate Shifted Closes ---
    logger.info(f"Pre-calculating {max_lag} shifted 'close' price series...")
    start_shift = time.time()
    if 'close' not in base_data_with_required.columns:
        logger.error("'close' column missing from base data. Cannot optimize.")
        return {}, []
    close_series = base_data_with_required['close'].astype(float)
    shifted_closes_global_cache = {lag: close_series.shift(-lag).reindex(base_data_with_required.index) for lag in range(1, max_lag + 1)}
    logger.info(f"Pre-calculation of shifted closes complete. Time: {time.time() - start_shift:.2f}s.")

    # --- Evaluate Default Config First ---
    # This ensures the default is evaluated and its correlations are in the accumulator and leaderboard checked
    default_id = _process_default_config_fast_eval(
        indicator_name, indicator_def, param_bounds, param_names, db_path,
        master_evaluated_configs, base_data_with_required, max_lag,
        shifted_closes_global_cache,
        interim_correlations_accumulator, max_lag, # Pass max_lag as max_lag_for_accumulator
        symbol, timeframe, data_daterange, source_db_name
    )

    # --- Optimization Loop per Lag ---
    total_lags_skipped_weak = 0
    aborted_indicator = False
    for target_lag in range(1, max_lag + 1):
        # Update progress display *at the start* of each lag's processing
        frac_lag_start = (target_lag - 1) / max_lag
        frac_indicator_progress = (progress_info['indicator_index'] + frac_lag_start) / progress_info['total_indicators_in_phase']
        current_overall_step = progress_info['current_step_base'] + progress_info['total_steps_in_phase'] * frac_indicator_progress
        progress_info['display_progress_func'](f"Optimizing {indicator_name} L{target_lag}", current_overall_step, progress_info['total_analysis_steps_global'])

        # Reset calls completed for the current lag before starting its evaluations
        progress_info['calls_completed_current_lag'] = 0

        # Check for excessive total failures across previous lags
        if failure_tracker[indicator_name] > MAX_TOTAL_OPT_FAILURES_INDICATOR:
            logger.warning(f"Aborting optimization for '{indicator_name}' due to excessive failures ({failure_tracker[indicator_name]} > {MAX_TOTAL_OPT_FAILURES_INDICATOR}) across previous lags.")
            aborted_indicator = True
            # Estimate remaining calls to skip to update overall progress accurately
            # remaining_lags = max_lag - target_lag + 1
            # calls_to_skip_total = remaining_lags * n_calls_per_lag
            # progress_info['calls_completed_current_indicator'] += calls_to_skip_total # Increment total count
            break # Exit the lag loop

        # --- Initial Point Evaluation & Weak Correlation Check ---
        initial_x = []; initial_y = []; max_abs_corr_initial = 0.0; initial_configs_evaluated_this_lag = 0
        num_random_points = max(0, n_initial_points_per_lag)
        for i in range(num_random_points):
            # Check if max failures reached during initial points
            if failure_tracker[indicator_name] > MAX_TOTAL_OPT_FAILURES_INDICATOR:
                logger.warning(f"Aborting '{indicator_name}' during initial points (Lag {target_lag}) due to failures.")
                aborted_indicator = True; break

            tunable_param_defs = {name: indicator_def['parameters'][name] for name in param_names if name in indicator_def.get('parameters', {})}
            random_params_dict = parameter_generator._generate_random_valid_config(tunable_param_defs, indicator_def.get('conditions', []))
            if random_params_dict is None: continue # Could not generate a valid random config

            # Call objective function for the random point
            obj_value = _objective_function(params_list=[random_params_dict.get(p_name) for p_name in param_names],
                                            param_names=param_names, target_lag=target_lag, indicator_name=indicator_name, indicator_def=indicator_def,
                                            base_data_with_required=base_data_with_required, db_path=db_path, symbol_id=symbol_id, timeframe_id=timeframe_id,
                                            master_evaluated_configs=master_evaluated_configs,
                                            progress_info=progress_info, # Pass the shared progress dict
                                            shifted_closes_global_cache=shifted_closes_global_cache,
                                            failure_tracker=failure_tracker,
                                            interim_correlations_accumulator=interim_correlations_accumulator,
                                            max_lag_for_accumulator=max_lag,
                                            symbol=symbol, timeframe=timeframe, data_daterange=data_daterange, source_db_name=source_db_name)

            initial_configs_evaluated_this_lag += 1 # Count attempt even if failed
            # Store result if it was valid (not the high cost failure value)
            if obj_value < (1e6 - 1.0):
                 initial_x.append([random_params_dict.get(p_name) for p_name in param_names])
                 initial_y.append(obj_value)
                 # Track max abs correlation found during initial phase
                 max_abs_corr_initial = max(max_abs_corr_initial, -obj_value) # obj_value is negative abs correlation

        if aborted_indicator: break # Exit lag loop if aborted during initial points

        # --- Skip Guided Optimization for this Lag if Initial Points are Weak ---
        if max_abs_corr_initial < WEAK_CORR_THRESHOLD_SKIP:
             total_lags_skipped_weak += 1
             # Ensure progress counter reflects skipped guided calls
             calls_to_skip_lag = max(0, n_calls_per_lag - initial_configs_evaluated_this_lag)
             progress_info['calls_completed_current_indicator'] += calls_to_skip_lag
             continue # Skip to the next lag

        # --- Proceed with Guided Optimization ---
        # Partial function to pass fixed arguments to the objective function
        objective_partial = partial(_objective_function,
                                    param_names=param_names, target_lag=target_lag, indicator_name=indicator_name,
                                    indicator_def=indicator_def, base_data_with_required=base_data_with_required,
                                    db_path=db_path, symbol_id=symbol_id, timeframe_id=timeframe_id,
                                    master_evaluated_configs=master_evaluated_configs,
                                    progress_info=progress_info, # Pass the shared progress dict
                                    shifted_closes_global_cache=shifted_closes_global_cache, failure_tracker=failure_tracker,
                                    interim_correlations_accumulator=interim_correlations_accumulator,
                                    max_lag_for_accumulator=max_lag,
                                    symbol=symbol, timeframe=timeframe, data_daterange=data_daterange, source_db_name=source_db_name)
        try:
             with warnings.catch_warnings(): # Suppress skopt warnings
                  warnings.simplefilter("ignore", category=UserWarning); warnings.simplefilter("ignore", category=RuntimeWarning)
                  # Number of calls remaining for the guided optimization phase
                  num_guided_calls = max(0, n_calls_per_lag - initial_configs_evaluated_this_lag)

                  best_params_list = None; best_objective_value = 1e6
                  if num_guided_calls > 0:
                      # Run the Bayesian optimization
                      result = gp_minimize(
                          func=objective_partial,
                          dimensions=search_space,
                          acq_func=app_config.DEFAULTS.get("optimizer_acq_func", 'gp_hedge'),
                          n_calls=num_guided_calls,
                          n_initial_points=0, # Initial points already evaluated
                          x0=initial_x if initial_x else None, # Provide initial points
                          y0=initial_y if initial_y else None, # Provide their results
                          random_state=None, # Use default random state behavior
                          noise='gaussian' # Assume some noise in observations
                      )
                      # Check if optimization found a result
                      if result and hasattr(result, 'x') and hasattr(result, 'fun'):
                         best_params_list = result.x
                         best_objective_value = result.fun
                      else:
                          logger.warning(f"gp_minimize returned invalid result for Lag {target_lag}.")
                          # If opt failed but we had initial points, use best of those
                          if initial_y:
                              best_initial_idx = np.argmin(initial_y)
                              best_params_list = initial_x[best_initial_idx]
                              best_objective_value = initial_y[best_initial_idx]
                              logger.info(f"Lag {target_lag}: Using best from initial points as gp_minimize failed.")


                  elif initial_y: # Only initial points were evaluated (e.g., n_calls <= n_initial)
                      best_initial_idx = np.argmin(initial_y)
                      best_params_list = initial_x[best_initial_idx]
                      best_objective_value = initial_y[best_initial_idx]
                  else:
                      # This should only happen if n_calls=0 or all initial points failed
                      logger.warning(f"Lag {target_lag}: No valid initial points or guided calls conducted.")
                      best_config_per_lag[target_lag] = None; continue # Skip to next lag

             # Process the best result found (either from guided opt or initial points)
             if best_params_list is not None and best_objective_value < (1e6 - 1.0):
                 best_score = -best_objective_value # Convert back to positive correlation score
                 # Clean parameter types (skopt might return numpy types)
                 best_params_opt = dict(zip(param_names, best_params_list))
                 for k, v in best_params_opt.items():
                     if isinstance(v, np.integer): best_params_opt[k] = int(v)
                     elif isinstance(v, np.floating): best_params_opt[k] = float(v)

                 # Combine with fixed parameters and get hash/ID
                 best_params_full = {**param_bounds, **best_params_opt}
                 best_hash = _get_config_hash(best_params_full)
                 best_config_info = master_evaluated_configs.get(best_hash)

                 if best_config_info and 'config_id' in best_config_info:
                     best_config_id = best_config_info['config_id']
                     # Retrieve the actual correlation value from cache (objective was -abs(corr))
                     best_corr = single_correlation_cache.get((best_hash, target_lag))
                     # Store the best result found *for this specific lag*
                     best_config_per_lag[target_lag] = {
                         'params': best_params_full,
                         'config_id': best_config_id,
                         'correlation_at_lag': best_corr,
                         'score_at_lag': best_score # Store positive score
                     }
                 else:
                     # This indicates an issue if the config wasn't stored correctly
                     logger.error(f"Lag {target_lag}: Could not find config ID for best params hash {best_hash}. Params: {best_params_full}. Opt score: {best_objective_value}.")
                     best_config_per_lag[target_lag] = None
             else:
                 # Optimization failed to find a valid solution better than the failure cost
                 logger.warning(f"Lag {target_lag}: Opt failed for {indicator_name}. No valid solution found (Best Obj: {best_objective_value}).")
                 best_config_per_lag[target_lag] = None
        except Exception as opt_err:
            logger.error(f"Error during optimization process for {indicator_name} lag {target_lag}: {opt_err}", exc_info=True)
            best_config_per_lag[target_lag] = None
    # End lag loop

    print() # Newline after lags finish or abortion

    if aborted_indicator:
         logger.warning(f"Optimization aborted for '{indicator_name}' after lag {target_lag-1 if 'target_lag' in locals() and target_lag > 1 else 'start'} due to excessive failures.")
    if total_lags_skipped_weak > 0:
        logger.info(f"{indicator_name}: Skipped optimization for {total_lags_skipped_weak}/{max_lag} lags due to weak initial correlation.")

    # Log final state using the last calculated overall step
    # Estimate final progress fraction for this indicator
    final_frac_indicator = (progress_info['indicator_index'] + 1) / progress_info['total_indicators_in_phase']
    final_overall_step = progress_info['current_step_base'] + progress_info['total_steps_in_phase'] * final_frac_indicator
    progress_info['display_progress_func'](f"Opt Finished: {indicator_name}", final_overall_step, progress_info['total_analysis_steps_global'])

    # Format the list of all unique configurations evaluated during this indicator's optimization
    final_all_evaluated_configs = _format_final_evaluated_configs(indicator_name, master_evaluated_configs)
    _log_optimization_summary(indicator_name, max_lag, best_config_per_lag, final_all_evaluated_configs)

    # Return the dictionary of best configs found per lag, and the list of all unique configs evaluated
    return best_config_per_lag, final_all_evaluated_configs


# --- Helper Functions for Bayesian Opt ---

def _define_search_space(param_defs: Dict) -> Tuple[List, List[str], Dict, bool]:
    """Creates the search space definition for skopt based on parameter definitions."""
    search_space = []
    param_names = []
    param_bounds = {}  # Initialize param_bounds
    fixed_params = {}
    has_tunable = False
    # Define common period parameter names
    period_params = ['fast', 'slow', 'fastperiod', 'slowperiod', 'signalperiod', 'timeperiod', 'timeperiod1', 'timeperiod2', 'timeperiod3', 'length', 'window', 'obv_period', 'price_period', 'fastk_period', 'slowk_period', 'slowd_period', 'fastd_period', 'tenkan', 'kijun', 'senkou']
    # Define parameters usually treated as factors or small floats
    factor_params = ['fastlimit','slowlimit','acceleration','maximum','vfactor','smoothing']
    # Define deviation/scalar parameters
    dev_scalar_params = ['scalar','nbdev','nbdevup','nbdevdn']

    # Handle matype separately if present
    if 'matype' in param_defs:
        matype_def = param_defs['matype']
        default = matype_def.get('default')
        # Only make matype tunable if it has a valid default >= 0
        if isinstance(default, (int, float)) and default >= 0:
            matype_values = list(range(9))  # TA-Lib MA types 0-8
            search_space.append(Categorical(categories=matype_values, name='matype'))
            param_names.append('matype')
            param_bounds['matype'] = {'min': 0, 'max': 8, 'default': default}  # Add to param_bounds
            has_tunable = True
            logger.debug("Space 'matype': Categorical(0..8)")
        elif default is not None:  # If default is present but not >= 0, treat as fixed
            fixed_params['matype'] = int(default) if isinstance(default, (int, float)) else default
            logger.debug(f"Fixed 'matype' to {fixed_params['matype']}")

    # Process other parameters
    for name, details in sorted(param_defs.items()):
        if name == 'matype': continue # Already handled

        default = details.get('default')
        min_j = details.get('min')
        max_j = details.get('max')
        # Infer type based on default or min/max if default is None
        p_type = type(default) if default is not None else (type(min_j) if min_j is not None else (type(max_j) if max_j is not None else None))

        is_numeric = p_type in [int, float]
        # Check if JSON defines valid numeric bounds
        has_json_bounds = min_j is not None and max_j is not None and isinstance(min_j, (int, float)) and isinstance(max_j, (int, float))

        if is_numeric and has_json_bounds:
            min_v, max_v = min_j, max_j
            # Check if bounds are effectively identical or invalid
            if min_v >= max_v - 1e-9: # Allow for small float differences
                # If bounds invalid, fix the parameter at default if possible, otherwise use min
                fix_val = default if default is not None else min_v
                # Ensure correct type for fixed value
                fixed_params[name] = int(round(fix_val)) if p_type is int else float(fix_val)
                logger.warning(f"Param '{name}': min >= max ({min_v}, {max_v}). Fixing @ {fixed_params[name]}.")
                continue # Skip adding to search space

            # Valid bounds defined, add to search space
            low, high = min_v, max_v
            bound_source = "JSON" # Indicate bounds came from definition file

            if p_type is int:
                # Convert potentially float bounds from JSON to int for Integer space
                low_int, high_int = int(np.floor(low)), int(np.ceil(high))
                # Ensure high is at least low + 1
                high_int = max(low_int + 1, high_int)

                # Apply minimum bounds for certain parameter types (e.g., periods usually >= 1 or 2)
                # List of params that are periods but *can* be 1
                period_exceptions_allow_1 = [
                    'mom', 'roc', 'rocp', 'rocr', 'rocr100', 'atr', 'natr', 'beta', 'correl',
                    'signalperiod', 'fastk_period', 'slowk_period', 'slowd_period', 'fastd_period',
                    'tenkan', 'kijun', 'senkou', 'timeperiod1', 'timeperiod2', 'timeperiod3'
                ]
                strict_min_2 = name in period_params and name.lower() not in period_exceptions_allow_1
                min_required = 2 if strict_min_2 else 1
                low_int = max(min_required, low_int)
                # Ensure high is still valid after adjusting low
                high_int = max(low_int + 1, high_int)

                search_space.append(Integer(low=low_int, high=high_int, name=name))
                param_bounds[name] = {'min': low_int, 'max': high_int, 'default': default}  # Add to param_bounds
                logger.debug(f"Space '{name}': Int({low_int},{high_int}) ({bound_source})")
            else: # Float type
                # Apply specific semantic lower bounds if applicable
                if name in factor_params: low = max(0.01, low) # Factors usually > 0
                elif name in dev_scalar_params: low = max(0.1, low) # Std Dev multipliers usually > 0

                # Ensure high > low after adjustments
                if low >= high: high = low + 0.01 # Add minimal separation

                search_space.append(Real(low=float(low), high=float(high), prior='uniform', name=name))
                param_bounds[name] = {'min': float(low), 'max': float(high), 'default': default}  # Add to param_bounds
                logger.debug(f"Space '{name}': Real({float(low):.4f},{float(high):.4f}) ({bound_source})")

            param_names.append(name)
            has_tunable = True

        elif default is not None: # Fix parameter if no valid bounds or non-numeric
            fixed_params[name] = default

        else: # No default and no bounds - cannot process
            logger.warning(f"Param '{name}': Skipping - no default value and no valid min/max bounds provided.")

    return search_space, param_names, param_bounds, has_tunable


def _process_default_config_fast_eval(
    indicator_name: str, indicator_def: Dict, fixed_params: Dict, param_names: List[str], db_path: str,
    master_evaluated_configs: Dict[str, Dict[str, Any]], base_data: pd.DataFrame, max_lag: int,
    shifted_closes_cache: Dict[int, pd.Series],
    interim_correlations_accumulator: Dict[int, List[Optional[float]]],
    max_lag_for_accumulator: int,
    symbol: str, timeframe: str, data_daterange: str, source_db_name: str
) -> Optional[int]:
    """Evaluates default config, updates caches, accumulator, and checks leaderboard."""
    param_defs = indicator_def.get('parameters', {})
    # Construct default parameters dict from tunable and fixed parts
    defaults_opt = {p: param_defs[p]['default'] for p in param_names if p in param_defs and 'default' in param_defs[p]}
    defaults_full = {**fixed_params, **defaults_opt}

    # Validate the combined default parameters against conditions
    if not defaults_full or not parameter_generator.evaluate_conditions(defaults_full, indicator_def.get('conditions', [])):
        logger.warning(f"Default parameters invalid or incomplete for {indicator_name}. Cannot fast-evaluate default.")
        return None

    default_hash = _get_config_hash(defaults_full)
    default_id: Optional[int] = None

    # Check if config ID is already known (e.g., from previous evaluations in this run)
    if default_hash in master_evaluated_configs:
        default_id = master_evaluated_configs[default_hash]['config_id']
    else:
        # If not known, get/create it from DB
        conn = sqlite_manager.create_connection(db_path)
        if not conn:
            logger.error(f"Cannot connect to DB for default config ID {indicator_name}.")
            return None
        try:
            default_id = sqlite_manager.get_or_create_indicator_config_id(conn, indicator_name, defaults_full)
            # Store it in the master map for this run
            master_evaluated_configs[default_hash] = {'params': defaults_full, 'config_id': default_id}
        except Exception as e:
            logger.error(f"Failed to process default config ID for {indicator_name}: {e}", exc_info=True)
            if conn: conn.close() # Ensure connection closed on error
            return None
        finally:
            if conn: conn.close()

    if default_id is None: return None # Could not get/create ID

    any_updated_leaderboard = False
    # Ensure accumulator list exists for this ID
    if default_id not in interim_correlations_accumulator:
        interim_correlations_accumulator[default_id] = [None] * max_lag_for_accumulator

    # Check indicator cache or calculate if necessary
    if default_hash not in indicator_series_cache:
        config_details = {'indicator_name': indicator_name, 'params': defaults_full, 'config_id': default_id}
        # Get indicator config from factory
        indicator_config = indicator_factory_instance.indicator_params.get(indicator_name)
        if indicator_config is None:
            logger.error(f"Unknown indicator: {indicator_name}")
            indicator_df = None
        else:
            indicator_df = indicator_factory_instance._compute_single_indicator(
                data=base_data.copy(),
                name=indicator_name,
                config=indicator_config
            )
        # Cache result (or empty DF on failure)
        indicator_series_cache[default_hash] = indicator_df if (indicator_df is not None and not indicator_df.empty) else pd.DataFrame()

    default_indicator_df = indicator_series_cache[default_hash]
    output_columns = list(default_indicator_df.columns) if default_indicator_df is not None else []

    if not output_columns:
        logger.warning(f"Default config ID {default_id} ('{indicator_name}') produced empty indicator output.")
    else:
        # Calculate/cache correlation for each lag
        for lag in range(1, max_lag + 1):
            cache_key = (default_hash, lag)
            corr: Optional[float] = None

            # Check correlation cache
            if cache_key in single_correlation_cache:
                corr = single_correlation_cache[cache_key]
            else:
                # Calculate correlation if not cached
                shifted_close = shifted_closes_cache.get(lag)
                if shifted_close is not None:
                    max_abs_corr_lag = -1.0
                    best_signed_corr_lag = np.nan
                    # Find best correlation across potentially multiple output columns
                    for col in output_columns:
                        if col not in default_indicator_df.columns: continue
                        try: indicator_series = default_indicator_df[col].astype(float)
                        except (ValueError, TypeError): continue
                        if indicator_series.isnull().all() or indicator_series.nunique(dropna=True) <= 1: continue
                        try:
                            combined = pd.concat([indicator_series, shifted_close], axis=1).dropna()
                            if len(combined) >= 2:
                                current_corr = combined.iloc[:, 0].corr(combined.iloc[:, 1])
                                if pd.notna(current_corr):
                                    current_abs = abs(float(current_corr))
                                    if current_abs > max_abs_corr_lag:
                                        max_abs_corr_lag = current_abs
                                        best_signed_corr_lag = float(current_corr)
                        except Exception as e:
                            logger.error(f"Error calc default corr for col '{col}', lag {lag}: {e}", exc_info=False)
                    corr = best_signed_corr_lag if max_abs_corr_lag > -1.0 else np.nan
                # Cache the calculated correlation (or None if failed)
                single_correlation_cache[cache_key] = corr

            # Store correlation in accumulator
            if 0 <= lag - 1 < max_lag_for_accumulator:
                 interim_correlations_accumulator[default_id][lag - 1] = corr
            else: logger.warning(f"Default eval: Lag {lag} out of bounds for accumulator ID {default_id}.")

            # Check leaderboard immediately
            if corr is not None and pd.notna(corr):
                try:
                    updated = leaderboard_manager.check_and_update_single_lag(
                        lag=lag, correlation_value=float(corr), indicator_name=indicator_name,
                        params=defaults_full, config_id=default_id, symbol=symbol, timeframe=timeframe,
                        data_daterange=data_daterange, source_db_name=source_db_name
                    )
                    any_updated_leaderboard |= updated # Track if any update occurred
                except Exception as lb_err:
                    logger.error(f"Error checking LB for default (Lag {lag}, ID {default_id}): {lb_err}", exc_info=True)

    # Export leaderboard text file if any update happened during default eval
    if any_updated_leaderboard:
        try:
            leaderboard_manager.export_leaderboard_to_text()
        except Exception as ex_err:
            logger.error(f"Error exporting LB after default eval: {ex_err}", exc_info=True)

    return default_id


def _format_final_evaluated_configs(indicator_name: str, master_evaluated_configs: Dict) -> List[Dict[str, Any]]:
    """Format final evaluated configurations for output."""
    formatted_configs = []
    for config_hash, config_data in master_evaluated_configs.items():
        formatted_config = {
            'indicator_name': indicator_name,
            'params': config_data.get('params', {}),
            'config_id': config_data.get('config_id'),
            'correlations': config_data.get('correlations', {})  # Include correlations
        }
        formatted_configs.append(formatted_config)
    return formatted_configs


def _log_optimization_summary(indicator_name: str, max_lag: int, best_config_per_lag: Dict, final_all_evaluated_configs: List):
    """Log optimization summary."""
    logger.info(f"Optimization summary for {indicator_name}")
    logger.info(f"Max lag: {max_lag}")
    logger.info(f"Best configs per lag: {best_config_per_lag}")
    logger.info(f"Total evaluated configs: {len(final_all_evaluated_configs)}")
    
    # Log details of best configs
    for lag, config_info in best_config_per_lag.items():
        if isinstance(config_info, dict):
            config_id = config_info.get('config_id', 'N/A')
            correlation = config_info.get('correlation', 'N/A')
            logger.info(f"Lag {lag}: Config {config_id}, Correlation: {correlation}")
        else:
            logger.info(f"Lag {lag}: {config_info}")
    
    # Log summary of all evaluated configs
    if final_all_evaluated_configs:
        logger.info(f"Evaluated {len(final_all_evaluated_configs)} configurations")
        for i, config in enumerate(final_all_evaluated_configs[:5]):  # Log first 5
            config_id = config.get('config_id', 'N/A')
            params = config.get('params', {})
            logger.info(f"Config {i+1}: ID {config_id}, Params: {params}")
        if len(final_all_evaluated_configs) > 5:
            logger.info(f"... and {len(final_all_evaluated_configs) - 5} more configurations")

def optimize_parameters_bayesian(data, indicator_def, n_trials=10):
    """Stub for Bayesian optimization. Returns best config and score using grid search for now."""
    if data is None or data.empty:
        raise ValueError("Input data is empty or None. Please provide valid market data.")

    if not indicator_def or not isinstance(indicator_def, dict):
        raise ValueError("Invalid indicator definition. Expected a dictionary with 'name' and 'parameters' keys.")

    # Handle the case where indicator_def is a dict with indicator name as key
    if len(indicator_def) == 1:
        # Extract the actual indicator definition
        indicator_name = next(iter(indicator_def))
        actual_def = indicator_def[indicator_name]
        param_defs = actual_def.get("params") or actual_def.get("parameters")
    else:
        # Direct indicator definition
        param_defs = indicator_def.get("params") or indicator_def.get("parameters")
    
    if not param_defs or not isinstance(param_defs, dict):
        raise ValueError(f"Indicator '{indicator_def.get('name', 'UNKNOWN')}' must have 'params' or 'parameters' as a dictionary.")
    
    # Validate required fields
    missing_fields = []
    if len(indicator_def) == 1:
        # For nested format, check the actual definition
        if 'name' not in actual_def:
            missing_fields.append("'name'")
        if 'type' not in actual_def:
            missing_fields.append("'type'")
    else:
        # For direct format
        if 'name' not in indicator_def:
            missing_fields.append("'name'")
        if 'type' not in indicator_def:
            missing_fields.append("'type'")
    
    if missing_fields:
        raise ValueError(f"Indicator definition missing required fields: {', '.join(missing_fields)}")
    
    # Validate parameter definitions
    invalid_params = []
    for param_name, spec in param_defs.items():
        if not isinstance(spec, dict):
            invalid_params.append(f"'{param_name}' (not a dictionary)")
            continue
            
        if "min" in spec and "max" in spec:
            try:
                min_val = float(spec["min"])
                max_val = float(spec["max"])
                if min_val >= max_val:
                    invalid_params.append(f"'{param_name}' (min >= max: {min_val} >= {max_val})")
            except (ValueError, TypeError):
                invalid_params.append(f"'{param_name}' (invalid min/max values)")
                
        if "default" in spec:
            try:
                default = float(spec["default"])
                if "min" in spec and default < float(spec["min"]):
                    invalid_params.append(f"'{param_name}' (default < min: {default} < {spec['min']})")
                if "max" in spec and default > float(spec["max"]):
                    invalid_params.append(f"'{param_name}' (default > max: {default} > {spec['max']})")
            except (ValueError, TypeError):
                invalid_params.append(f"'{param_name}' (invalid default value)")
    
    if invalid_params:
        indicator_name = next(iter(indicator_def)) if len(indicator_def) == 1 else indicator_def.get('name', 'UNKNOWN')
        raise ValueError(f"Invalid parameter definitions in indicator '{indicator_name}': {', '.join(invalid_params)}")

    # Get the indicator name for computing
    if len(indicator_def) == 1:
        indicator_name = next(iter(indicator_def))
    else:
        indicator_name = indicator_def.get("name", "IND")

    # Simple grid search implementation for now
    factory = indicator_factory.IndicatorFactory()
    param_names = list(param_defs.keys())
    param_ranges = []
    
    for p, spec in param_defs.items():
        if "min" in spec and "max" in spec and "default" in spec:
            if isinstance(spec["default"], int):
                param_ranges.append([spec["min"], spec["max"], spec["default"]])
            elif isinstance(spec["default"], float):
                param_ranges.append([spec["min"], spec["max"], spec["default"]])
            else:
                param_ranges.append([spec["default"]])
        else:
            param_ranges.append([spec.get("default")])
    
    from itertools import product
    best_params = None
    best_score = float('-inf')
    
    for values in product(*param_ranges):
        params = dict(zip(param_names, values))
        try:
            indicators = factory.compute_indicators(data, {indicator_name: params})
            if indicator_name in indicators:
                score = indicators[indicator_name].mean()
                if score > best_score:
                    best_score = score
                    best_params = params
        except Exception:
            continue
    
    if best_params is None:
        raise ValueError("No valid parameter set found")
    
    return best_params, best_score

def optimize_parameters_classical(data, indicator_def):
    """Classical optimization: grid search for best config by mean indicator value."""
    if data is None or data.empty:
        raise ValueError("Input data is empty.")
    if not indicator_def or not isinstance(indicator_def, dict):
        raise ValueError("Invalid indicator definition.")
    
    # Handle the case where indicator_def is a dict with indicator name as key
    if len(indicator_def) == 1:
        # Extract the actual indicator definition
        indicator_name = next(iter(indicator_def))
        actual_def = indicator_def[indicator_name]
        param_defs = actual_def.get("params") or actual_def.get("parameters")
    else:
        # Direct indicator definition
        param_defs = indicator_def.get("params") or indicator_def.get("parameters")
    
    if not param_defs or not isinstance(param_defs, dict):
        raise ValueError("Indicator definition must have 'params' or 'parameters' as a dict.")
    
    # Get the indicator name for computing
    if len(indicator_def) == 1:
        indicator_name = next(iter(indicator_def))
    else:
        indicator_name = indicator_def.get("name", "IND")
    
    factory = indicator_factory.IndicatorFactory()
    param_names = list(param_defs.keys())
    param_ranges = []
    for p, spec in param_defs.items():
        if "min" in spec and "max" in spec and "default" in spec:
            if isinstance(spec["default"], int):
                param_ranges.append([spec["min"], spec["max"], spec["default"]])
            elif isinstance(spec["default"], float):
                param_ranges.append([spec["min"], spec["max"], spec["default"]])
            else:
                param_ranges.append([spec["default"]])
        else:
            param_ranges.append([spec.get("default")])
    from itertools import product
    best_params = None
    best_score = float('-inf')
    for values in product(*param_ranges):
        params = dict(zip(param_names, values))
        try:
            indicators = factory.compute_indicators(data, {indicator_name: params})
            if indicator_name in indicators:
                score = indicators[indicator_name].mean()
                if score > best_score:
                    best_score = score
                    best_params = params
        except Exception:
            continue
    if best_params is None:
        raise ValueError("No valid parameter set found")
    return best_params, best_score

def optimize_parameters(data, indicator_def, method="bayesian", n_trials=10):
    """Main optimization function. Dispatches to Bayesian or classical optimization."""
    if method == "bayesian":
        return optimize_parameters_bayesian(data, indicator_def, n_trials=n_trials)
    elif method == "classical":
        return optimize_parameters_classical(data, indicator_def)
    else:
        raise ValueError(f"Unknown optimization method: {method}")

def _prepare_optimization_data(data: pd.DataFrame, indicator_definition: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Prepares and validates optimization data and indicator definition for tests.
    Ensures indicator definition is loaded from indicator_params.json if not already in correct format.
    Returns (data, indicator_def) where indicator_def is a dict with indicator name as key.
    """
    import json
    import os
    # If indicator_definition is already in correct format, return as is
    if isinstance(indicator_definition, dict) and any(
        isinstance(v, dict) and ("params" in v or "parameters" in v)
        for v in indicator_definition.values()
    ):
        return data, indicator_definition
    # Otherwise, try to load from indicator_params.json
    params_path = os.path.join(os.path.dirname(__file__), "indicator_params.json")
    with open(params_path, "r", encoding="utf-8") as f:
        params_json = json.load(f)
    # Find the indicator by name
    name = indicator_definition.get("name") if isinstance(indicator_definition, dict) else None
    if not name or name not in params_json:
        raise ValueError(f"Indicator definition for '{name}' not found in indicator_params.json")
    indicator_def = {name: params_json[name]}
    return data, indicator_def

def objective_function(config, data, indicator_def):
    """Public objective function for optimization tests. Returns mean indicator value as score."""
    if not isinstance(config, dict):
        raise ValueError("Config must be a dict.")
    if data is None or data.empty:
        raise ValueError("Input data is empty.")
    if not indicator_def or not isinstance(indicator_def, dict):
        raise ValueError("Invalid indicator definition.")
    
    # Handle the case where indicator_def is a dict with indicator name as key
    if len(indicator_def) == 1:
        # Extract the actual indicator definition
        indicator_name = next(iter(indicator_def))
        actual_def = indicator_def[indicator_name]
        param_defs = actual_def.get("params") or actual_def.get("parameters")
    else:
        # Direct indicator definition
        param_defs = indicator_def.get("params") or indicator_def.get("parameters")
    
    if not param_defs or not isinstance(param_defs, dict):
        raise ValueError("Indicator definition must have 'params' or 'parameters' as a dict.")
    
    # Validate config against param_defs
    for param_name, spec in param_defs.items():
        if param_name not in config:
            if "default" in spec:
                config[param_name] = spec["default"]
            else:
                raise ValueError(f"Missing required parameter: {param_name}")
        
        val = config[param_name]
        if "min" in spec and val < spec["min"]:
            raise ValueError(f"Parameter '{param_name}' below min: {val} < {spec['min']}")
        if "max" in spec and val > spec["max"]:
            raise ValueError(f"Parameter '{param_name}' above max: {val} > {spec['max']}")
    
    factory = indicator_factory.IndicatorFactory()
    
    # Get the indicator name for computing
    if len(indicator_def) == 1:
        indicator_name = next(iter(indicator_def))
    else:
        indicator_name = indicator_def.get("name", "IND")
    
    try:
        indicators = factory.compute_indicators(data, {indicator_name: config})
        if indicator_name in indicators:
            score = indicators[indicator_name].mean()
            return score
        else:
            # Try to find any numeric column
            for col in indicators.columns:
                if pd.api.types.is_numeric_dtype(indicators[col]):
                    return indicators[col].mean()
            raise ValueError("No numeric indicator output found")
    except Exception as e:
        logger.error(f"Error computing indicator {indicator_name}: {e}")
        return float('-inf')  # Return worst possible score on error

def _evaluate_configuration(config: dict, data: pd.DataFrame, indicator_def: dict) -> Tuple[float, pd.DataFrame]:
    """
    Evaluates a configuration: runs the indicator and returns (score, output_df).
    Score is the mean of the indicator output (for test compatibility).
    """
    if not isinstance(config, dict):
        raise ValueError("Config must be a dict.")
    if data is None or data.empty:
        raise ValueError("Input data is empty.")
    if not indicator_def or not isinstance(indicator_def, dict):
        raise ValueError("Invalid indicator definition.")
    
    # Handle the case where indicator_def is a dict with indicator name as key
    if len(indicator_def) == 1:
        # Extract the actual indicator definition
        indicator_name = next(iter(indicator_def))
        actual_def = indicator_def[indicator_name]
        param_defs = actual_def.get("params") or actual_def.get("parameters")
    else:
        # Direct indicator definition
        param_defs = indicator_def.get("params") or indicator_def.get("parameters")
    
    if not param_defs or not isinstance(param_defs, dict):
        raise ValueError("Indicator definition must have 'params' or 'parameters' as a dict.")
    
    # Validate config against param_defs
    for param_name, spec in param_defs.items():
        if param_name not in config:
            if "default" in spec:
                config[param_name] = spec["default"]
            else:
                raise ValueError(f"Missing required parameter: {param_name}")
        
        val = config[param_name]
        if "min" in spec and val < spec["min"]:
            raise ValueError(f"Parameter '{param_name}' below min: {val} < {spec['min']}")
        if "max" in spec and val > spec["max"]:
            raise ValueError(f"Parameter '{param_name}' above max: {val} > {spec['max']}")
    
    # Get the indicator name for computing
    if len(indicator_def) == 1:
        indicator_name = next(iter(indicator_def))
    else:
        indicator_name = indicator_def.get("name", "IND")
    
    factory = indicator_factory.IndicatorFactory()
    
    try:
        indicators = factory.compute_indicators(data, {indicator_name: config})
        if indicator_name in indicators:
            score = indicators[indicator_name].mean()
            return score, indicators
        else:
            # Try to find any numeric column
            for col in indicators.columns:
                if pd.api.types.is_numeric_dtype(indicators[col]):
                    return indicators[col].mean(), indicators
            raise ValueError("No numeric indicator output found")
    except Exception as e:
        logger.error(f"Error computing indicator {indicator_name}: {e}")
        raise ValueError(f"Error computing indicator {indicator_name}: {e}")
