# parameter_optimizer.py
import logging
import random
import json
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import sqlite3
import time # Import time for potential delays/debugging

import config as app_config
import indicator_factory
import correlation_calculator
import sqlite_manager
import utils
import parameter_generator

logger = logging.getLogger(__name__)

# --- Helper Functions (_get_config_hash, _generate_candidate) ---
# (Keep these as they are)
def _get_config_hash(params: Dict[str, Any]) -> str:
    """Generates a stable SHA256 hash for a parameter dictionary."""
    config_str = json.dumps(params, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()

def _generate_candidate(
    base_config: Optional[Dict[str, Any]],
    param_defs: Dict[str, Dict],
    conditions: List[Dict],
    indicator_name: str,
    evaluated_hashes: set # Hashes evaluated *within the current target_lag optimization*
) -> Optional[Dict[str, Any]]:
    """
    Generates a new candidate configuration.
    If base_config is provided, it perturbs it.
    Otherwise, it calls _generate_random_valid_config for exploration.
    Ensures the generated config hash is not in evaluated_hashes for the current lag.
    Returns None if it can't generate a new valid config after several tries.
    """
    max_tries = 30 # Tries to find a *new* valid config for *this lag's* optimization
    attempt = 0
    period_params = [
        'fast', 'slow', 'fastperiod', 'slowperiod', 'signalperiod',
        'timeperiod', 'timeperiod1', 'timeperiod2', 'timeperiod3',
        'length', 'window', 'obv_period', 'price_period',
        'fastk_period', 'slowk_period', 'slowd_period', 'fastd_period',
        'tenkan', 'kijun', 'senkou'
    ]
    factor_params = ['fastlimit','slowlimit','acceleration','maximum','vfactor','smoothing']
    dev_scalar_params = ['scalar','nbdev','nbdevup','nbdevdn']

    while attempt < max_tries:
        attempt += 1
        candidate_params = None
        log_prefix = f"Candidate Gen (Try {attempt}/{max_tries}) for {indicator_name}:"

        if base_config:
            # --- Perturbation Logic ---
            candidate_params = base_config.copy()
            tweakable_params = [p for p, d in param_defs.items() if isinstance(d.get('default'), (int, float))]

            if not tweakable_params:
                 logger.warning(f"{log_prefix} No numeric tweakable parameters found for perturbation.")
                 # If no numeric params, perturbation yields the same config.
                 # Force random generation instead.
                 base_config = None
                 continue # Go to next attempt, will trigger random generation

            param_to_tweak = random.choice(tweakable_params)
            details = param_defs[param_to_tweak]
            current_value = candidate_params[param_to_tweak]

            if isinstance(current_value, int):
                min_val = 2 if param_to_tweak in period_params else 1
                step_type = random.choice(['step', 'gauss'])
                if step_type == 'step':
                    step_size = max(1, int(current_value * 0.1))
                    change = random.choice([-step_size, step_size])
                else: # gauss
                    sigma = max(1, int(current_value * 0.15))
                    change = int(round(random.gauss(0, sigma)))

                new_value = max(min_val, current_value + change)
                if new_value == current_value and change != 0: new_value = max(min_val, current_value + (1 if change > 0 else -1))
                elif new_value == current_value: new_value = max(min_val, current_value + random.choice([-1,1]))
                candidate_params[param_to_tweak] = new_value

            elif isinstance(current_value, float):
                min_val = 0.01 if param_to_tweak in factor_params else (0.1 if param_to_tweak in dev_scalar_params else 0.0)
                sigma = max(0.01, abs(current_value * 0.15))
                change = round(random.gauss(0, sigma), 4)
                new_value = round(max(min_val, current_value + change), 4)
                if np.isclose(new_value, current_value) and not np.isclose(change, 0): new_value = round(max(min_val, current_value + (0.01 if change > 0 else -0.01)), 4)
                elif np.isclose(new_value, current_value): new_value = round(max(min_val, current_value + random.choice([-0.01, 0.01])), 4)
                candidate_params[param_to_tweak] = new_value
            else:
                 # This case should not be reached if tweakable_params only includes int/float
                 logger.warning(f"{log_prefix} Cannot perturb non-numeric param '{param_to_tweak}'. Trying random.")
                 base_config = None
                 continue

            logger.debug(f"{log_prefix} Perturbed '{param_to_tweak}' from {base_config.get(param_to_tweak)} to {candidate_params[param_to_tweak]}")

        else:
            # --- Random Generation Logic ---
            logger.debug(f"{log_prefix} Generating random candidate using parameter_generator.")
            candidate_params = parameter_generator._generate_random_valid_config(param_defs, conditions)
            if candidate_params is None:
                logger.warning(f"{log_prefix} Failed to generate a random valid candidate via helper.")
                continue


        # --- Validation and Uniqueness Check ---
        if candidate_params is None: continue

        candidate_hash = _get_config_hash(candidate_params)
        is_valid = parameter_generator.evaluate_conditions(candidate_params, conditions) # Always re-validate
        is_new_for_this_lag = candidate_hash not in evaluated_hashes # Check against hashes for current lag optimization

        if is_valid and is_new_for_this_lag:
            logger.debug(f"{log_prefix} Found new valid candidate for this lag: {candidate_params}")
            return candidate_params
        elif not is_valid:
             logger.debug(f"{log_prefix} Generated candidate {candidate_params} is invalid. Retrying.")
             base_config = None # Force random if invalid
        elif not is_new_for_this_lag:
             logger.debug(f"{log_prefix} Candidate {candidate_params} hash {candidate_hash[:8]}... already evaluated for this lag. Retrying.")
             base_config = None # Force random if already seen for this lag

    logger.warning(f"Could not generate a new, valid, unique configuration for {indicator_name} for this lag after {max_tries} tries.")
    return None

# --- NEW/REVISED Evaluation Function ---
def _evaluate_config_and_get_correlations(
    params: Dict[str, Any],
    config_id: int,
    indicator_name: str,
    base_data_with_required: pd.DataFrame,
    max_lag: int,
) -> Tuple[bool, Dict[str, List[Optional[float]]]]:
    """
    Computes indicator and ALL correlations (1 to max_lag) for all output columns.

    Returns:
        Tuple[bool, Dict[str, List[Optional[float]]]]
        - bool: True if calculation was successful (at least one correlation calculated), False otherwise.
        - Dict: Maps output column name to its list of correlations [corr_lag1, corr_lag2, ...]. Empty dict on failure.
    """
    logger.debug(f"Evaluating all correlations for Config ID {config_id} ({indicator_name}, {params}) up to lag {max_lag}")

    # 1. Compute Indicator
    config_details = {'indicator_name': indicator_name, 'params': params, 'config_id': config_id}
    indicator_output_df = indicator_factory._compute_single_indicator(base_data_with_required.copy(), config_details)

    if indicator_output_df is None or indicator_output_df.empty:
        logger.warning(f"Indicator computation failed or returned empty for Config ID {config_id}.")
        return False, {}
    nan_check = indicator_output_df.isnull().all()
    all_nan_cols = nan_check[nan_check].index.tolist()
    if all_nan_cols:
        logger.warning(f"Indicator {indicator_name} (Cfg {config_id}) produced all-NaN columns: {all_nan_cols}.")
        return False, {}

    # 2. Combine with 'close' and drop NaNs for correlation input
    if 'close' not in base_data_with_required.columns:
        logger.error("Base data missing 'close' column for correlation calculation.")
        return False, {}
    data_for_corr = pd.concat([base_data_with_required[['close']], indicator_output_df.reindex(base_data_with_required.index)], axis=1)
    initial_len = len(data_for_corr)
    data_for_corr.dropna(inplace=True)
    dropped_rows = initial_len - len(data_for_corr)
    if dropped_rows > 0: logger.debug(f"Dropped {dropped_rows} rows with NaNs before correlation for Config ID {config_id}.")

    min_required_len = max_lag + 3 # Need at least 3 points for correlation calculation after shifting
    if len(data_for_corr) < min_required_len:
        logger.warning(f"Insufficient data ({len(data_for_corr)} rows) for correlation (max_lag={max_lag}) for Config ID {config_id}. Need {min_required_len}.")
        return False, {}

    # 3. Calculate Correlations for each output column
    all_output_correlations: Dict[str, List[Optional[float]]] = {}
    output_columns = list(indicator_output_df.columns)
    calculation_successful = False
    logger.debug(f"Config ID {config_id}: Calculating correlations for columns: {output_columns}")

    for indicator_col in output_columns:
        col_correlations: List[Optional[float]] = [None] * max_lag # Pre-allocate with None
        if indicator_col not in data_for_corr.columns:
            logger.error(f"Internal error: Column '{indicator_col}' not in data_for_corr. Skipping.")
            continue
        if data_for_corr[indicator_col].nunique(dropna=True) <= 1:
            logger.warning(f"Indicator column '{indicator_col}' (Config ID {config_id}) has no variance. Setting correlations to NaN.")
            all_output_correlations[indicator_col] = [np.nan] * max_lag # Store NaNs
            continue

        any_col_corr_calculated = False
        for lag in range(1, max_lag + 1):
            correlation_value = correlation_calculator.calculate_correlation_indicator_vs_future_price( data_for_corr, indicator_col, lag )
            # Store float or None directly
            col_correlations[lag-1] = float(correlation_value) if pd.notna(correlation_value) else None
            if pd.notna(correlation_value):
                 any_col_corr_calculated = True

        all_output_correlations[indicator_col] = col_correlations
        if any_col_corr_calculated:
            calculation_successful = True # Mark success if at least one correlation was calculated for any column

    if not calculation_successful:
        logger.warning(f"Config ID {config_id}: No valid correlation results generated across all output columns.")
        return False, {}

    logger.debug(f"Config ID {config_id}: Successfully calculated correlation vectors.")
    return True, all_output_correlations


# --- NEW Main Optimization Function ---
def optimize_parameters_per_lag(
    indicator_name: str,
    indicator_def: Dict,
    base_data_with_required: pd.DataFrame,
    max_lag: int,
    num_iterations_per_lag: int, # Renamed for clarity
    db_path: str,
    symbol_id: int,
    timeframe_id: int,
    # scoring_method: str = 'max_abs' # Scoring method for selecting best *within* a lag
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Optimizes parameters for a single indicator, finding the best configuration *for each lag*.

    Args:
        indicator_name: Name of the indicator.
        indicator_def: Definition dictionary for the indicator.
        base_data_with_required: DataFrame containing necessary input columns ('close', etc.).
        max_lag: The maximum lag to optimize for.
        num_iterations_per_lag: Number of *new* configurations to evaluate for *each lag*.
        db_path: Path to the SQLite database.
        symbol_id: ID for the symbol in the database.
        timeframe_id: ID for the timeframe in the database.
        # scoring_method: How to determine the 'best' correlation at a specific lag ('max_abs', 'max_positive', 'max_negative').

    Returns:
        Tuple containing:
        1. Dict[int, Dict[str, Any]]: A dictionary mapping each lag (1 to max_lag) to the details
           of the best configuration found for that specific lag.
           Format: {lag: {'params': ..., 'config_id': ..., 'correlation_at_lag': ...}, ...}
           If no valid config is found for a lag, the entry might be missing or contain None.
        2. List[Dict[str, Any]]: A list of all unique configurations evaluated across all lags.
           Format: [{'indicator_name': ..., 'params': ..., 'config_id': ...}, ...]
    """
    logger.info(f"--- Starting Per-Lag Parameter Optimization for: {indicator_name} ---")
    logger.info(f"Target Lags: 1 to {max_lag}, Iterations per Lag: {num_iterations_per_lag}")

    param_defs = indicator_def.get('parameters', {})
    conditions = indicator_def.get('conditions', [])

    # --- Global Tracking ---
    # Stores all unique configs evaluated across all lags and their config IDs
    master_evaluated_configs: Dict[str, Dict[str, Any]] = {} # {hash: {'params': ..., 'config_id': ...}}
    # Stores all correlations calculated, ready for batch DB insert
    master_correlations_to_batch_insert: List[Tuple[int, int, int, int, Optional[float]]] = []
    # Stores the best result found specifically for each lag
    best_config_per_lag: Dict[int, Dict[str, Any]] = {} # {lag: {'params':..., 'config_id':..., 'correlation_at_lag':...}}
    # Cache calculated correlations to avoid recomputing within the same run
    # Map: {config_hash: {output_col_name: [corr_lag1, ...], ...}}
    correlation_cache: Dict[str, Dict[str, List[Optional[float]]]] = {}


    # --- Get Default Config Info ---
    default_params = {k: v.get('default') for k, v in param_defs.items() if 'default' in v}
    default_config_id = None
    default_hash = None
    if default_params and parameter_generator.evaluate_conditions(default_params, conditions):
        default_hash = _get_config_hash(default_params)
        conn = sqlite_manager.create_connection(db_path)
        if conn:
            try:
                default_config_id = sqlite_manager.get_or_create_indicator_config_id(conn, indicator_name, default_params)
                # Add default to master list immediately if valid
                master_evaluated_configs[default_hash] = {'params': default_params, 'config_id': default_config_id}
                logger.info(f"Default config prepared (ID: {default_config_id}): {default_params}")
            except Exception as e:
                logger.error(f"Failed to get/create config ID for default params: {e}", exc_info=True)
                default_config_id = None # Ensure it's None if DB interaction failed
            finally:
                conn.close()
        else:
            logger.error("Cannot connect to DB to get default config ID.")
    else:
        logger.error(f"Default parameters for {indicator_name} are missing or invalid. Cannot include in optimization base.")

    # --- Outer Loop: Iterate through each target lag ---
    for target_lag in range(1, max_lag + 1):
        logger.info(f"--- Optimizing for Lag: {target_lag}/{max_lag} ---")

        # --- Per-Lag Tracking ---
        evaluated_configs_results_this_lag: List[Dict[str, Any]] = [] # Stores {hash, score_at_lag}
        evaluated_hashes_this_lag: set = set() # Track hashes evaluated *for this lag's optimization*
        # correlations_from_this_lag_optimization = {} # Cache {hash: {col: [corrs]}} -> Use global correlation_cache instead

        best_score_for_this_lag = -float('inf') # Using max_abs score for optimization target
        best_config_params_for_this_lag = None
        best_config_id_for_this_lag = None
        best_correlation_at_this_lag = None

        # --- 1. Evaluate Default Config *at this lag* ---
        if default_config_id is not None and default_hash is not None:
            logger.debug(f"Lag {target_lag}: Evaluating default config (ID: {default_config_id}).")
            # Calculate correlations (use cache)
            calc_success, correlations_dict = False, {}
            if default_hash in correlation_cache:
                 logger.debug(f"Lag {target_lag}: Using cached correlations for default config.")
                 correlations_dict = correlation_cache[default_hash]
                 calc_success = True # Assume success if cached
            else:
                 logger.debug(f"Lag {target_lag}: Calculating correlations for default config.")
                 calc_success, correlations_dict = _evaluate_config_and_get_correlations(
                     default_params, default_config_id, indicator_name, base_data_with_required, max_lag
                 )
                 if calc_success:
                      correlation_cache[default_hash] = correlations_dict
                      # Add to batch insert list *only once*
                      # Check based on config_id to prevent duplicates if hash collision occurred (unlikely)
                      if not any(row[2] == default_config_id for row in master_correlations_to_batch_insert):
                           logger.debug(f"Lag {target_lag}: Scheduling correlations for default config ID {default_config_id} for batch insert.")
                           for col_name, corr_list in correlations_dict.items():
                                for lag_idx, corr_val in enumerate(corr_list):
                                     master_correlations_to_batch_insert.append((symbol_id, timeframe_id, default_config_id, lag_idx + 1, corr_val))

            if calc_success:
                 evaluated_hashes_this_lag.add(default_hash)
                 # Determine score *at this target lag* from *all* output columns
                 score_at_target_lag = -float('inf')
                 correlation_value_at_target_lag = None # The actual correlation value associated with the best score
                 for col, corrs in correlations_dict.items():
                      if len(corrs) >= target_lag:
                           corr_val = corrs[target_lag - 1]
                           if pd.notna(corr_val):
                                current_score = abs(corr_val) # Score is absolute value for this lag
                                if current_score > score_at_target_lag:
                                     score_at_target_lag = current_score
                                     correlation_value_at_target_lag = corr_val # Store the actual value

                 if score_at_target_lag > -float('inf'):
                      logger.debug(f"Lag {target_lag}: Default config score = {score_at_target_lag:.4f} (Corr: {correlation_value_at_target_lag:.4f})")
                      evaluated_configs_results_this_lag.append({'hash': default_hash, 'score_at_lag': score_at_target_lag})
                      best_score_for_this_lag = score_at_target_lag
                      best_config_params_for_this_lag = default_params
                      best_config_id_for_this_lag = default_config_id
                      best_correlation_at_this_lag = correlation_value_at_target_lag
                 else: logger.debug(f"Lag {target_lag}: Default config had NaN correlation at this lag.")
            else: logger.warning(f"Lag {target_lag}: Failed to calculate correlations for default config.")


        # --- 2. Iterative Optimization Loop (for this target_lag) ---
        successful_evaluations_this_lag = 1 if best_config_params_for_this_lag is not None else 0
        iteration_this_lag = 0
        max_iterations_for_lag_opt = num_iterations_per_lag * 3 # Give some leeway

        while successful_evaluations_this_lag < num_iterations_per_lag and iteration_this_lag < max_iterations_for_lag_opt:
            iteration_this_lag += 1
            logger.debug(f"\n--- Lag {target_lag} - Opt Iteration {iteration_this_lag} (Evals: {successful_evaluations_this_lag}/{num_iterations_per_lag}) ---")

            # Determine base config for perturbation vs random exploration
            explore = random.random() < app_config.OPTIMIZER_RANDOM_EXPLORE_PROB
            base_for_candidate = None
            generation_method = "random"
            if not explore and best_config_params_for_this_lag is not None:
                base_for_candidate = best_config_params_for_this_lag
                generation_method = "perturbation"
            logger.debug(f"Lag {target_lag} - Gen method: {generation_method}")

            candidates_generated_this_iter = 0
            while candidates_generated_this_iter < app_config.OPTIMIZER_CANDIDATES_PER_ITERATION:
                 if successful_evaluations_this_lag >= num_iterations_per_lag: break

                 # Generate a candidate NOT already evaluated *for this lag's optimization*
                 candidate_params = _generate_candidate(
                     base_for_candidate, param_defs, conditions, indicator_name, evaluated_hashes_this_lag
                 )

                 if candidate_params is None:
                     logger.warning(f"Lag {target_lag}: Failed to generate a new valid candidate in this attempt.")
                     if generation_method == "perturbation": base_for_candidate = None; generation_method = "random (fallback)"; logger.debug("Switching to random generation for next attempt.") ; continue
                     else: logger.error(f"Lag {target_lag}: Failed to generate even a 'random' valid candidate. Breaking inner loop."); break

                 candidate_hash = _get_config_hash(candidate_params)
                 candidates_generated_this_iter += 1
                 evaluated_hashes_this_lag.add(candidate_hash) # Mark as evaluated for this lag

                 # Get/Create Config ID (add to master list if new)
                 candidate_config_id = None
                 if candidate_hash in master_evaluated_configs:
                      candidate_config_id = master_evaluated_configs[candidate_hash]['config_id']
                 else:
                      conn = sqlite_manager.create_connection(db_path)
                      if not conn: continue # Skip if cannot connect
                      try:
                           candidate_config_id = sqlite_manager.get_or_create_indicator_config_id(conn, indicator_name, candidate_params)
                           master_evaluated_configs[candidate_hash] = {'params': candidate_params, 'config_id': candidate_config_id}
                           logger.debug(f"Lag {target_lag}: Registered new master config ID {candidate_config_id} for hash {candidate_hash[:8]}")
                      except Exception as e: logger.error(f"Lag {target_lag}: Failed get/create config ID for {candidate_params}: {e}"); candidate_config_id = None
                      finally: conn.close()
                 if candidate_config_id is None: continue # Skip if failed to get ID

                 # Evaluate correlations (use cache if possible)
                 calc_success, correlations_dict = False, {}
                 if candidate_hash in correlation_cache:
                      logger.debug(f"Lag {target_lag}: Using cached correlations for candidate ID {candidate_config_id} / hash {candidate_hash[:8]}.")
                      correlations_dict = correlation_cache[candidate_hash]
                      calc_success = True
                 else:
                      logger.debug(f"Lag {target_lag}: Calculating correlations for candidate ID {candidate_config_id} / hash {candidate_hash[:8]}.")
                      calc_success, correlations_dict = _evaluate_config_and_get_correlations(
                           candidate_params, candidate_config_id, indicator_name, base_data_with_required, max_lag
                      )
                      if calc_success:
                           correlation_cache[candidate_hash] = correlations_dict
                           # Add to batch insert list *only once* per unique config ID
                           if not any(row[2] == candidate_config_id for row in master_correlations_to_batch_insert):
                                logger.debug(f"Lag {target_lag}: Scheduling correlations for config ID {candidate_config_id} for batch insert.")
                                for col_name, corr_list in correlations_dict.items():
                                     for lag_idx, corr_val in enumerate(corr_list):
                                         master_correlations_to_batch_insert.append((symbol_id, timeframe_id, candidate_config_id, lag_idx + 1, corr_val))
                           # else: logger.debug(f"Lag {target_lag}: Correlations for config ID {candidate_config_id} already scheduled for insert.")

                 if calc_success:
                      # Determine score *at this target lag*
                      score_at_target_lag = -float('inf')
                      correlation_value_at_target_lag = None
                      for col, corrs in correlations_dict.items():
                           if len(corrs) >= target_lag:
                               corr_val = corrs[target_lag - 1]
                               if pd.notna(corr_val):
                                    current_score = abs(corr_val) # Maximize absolute correlation at target lag
                                    if current_score > score_at_target_lag:
                                        score_at_target_lag = current_score
                                        correlation_value_at_target_lag = corr_val

                      if score_at_target_lag > -float('inf'):
                           logger.info(f"Lag {target_lag} Iter {iteration_this_lag}: Eval Config ID {candidate_config_id} ({candidate_params}). Score@Lag = {score_at_target_lag:.4f} (Corr: {correlation_value_at_target_lag:.4f})")
                           evaluated_configs_results_this_lag.append({'hash': candidate_hash, 'score_at_lag': score_at_target_lag})
                           successful_evaluations_this_lag += 1

                           # Update best for *this lag*
                           if score_at_target_lag > best_score_for_this_lag:
                               logger.info(f"***** Lag {target_lag}: New best score found: {score_at_target_lag:.4f} (Config ID {candidate_config_id}) *****")
                               best_score_for_this_lag = score_at_target_lag
                               best_config_params_for_this_lag = candidate_params
                               best_config_id_for_this_lag = candidate_config_id
                               best_correlation_at_this_lag = correlation_value_at_target_lag
                      else:
                           logger.warning(f"Lag {target_lag} Iter {iteration_this_lag}: Config ID {candidate_config_id} ({candidate_params}) had NaN correlation at this lag.")
                           # Still count as evaluation attempt, but don't increment success count
                 else:
                     logger.warning(f"Lag {target_lag} Iter {iteration_this_lag}: Correlation calculation failed for Config ID {candidate_config_id} ({candidate_params}).")

                 if successful_evaluations_this_lag >= num_iterations_per_lag: break # Exit inner candidate loop

            # End inner candidate generation loop
            if successful_evaluations_this_lag >= num_iterations_per_lag:
                 logger.info(f"Lag {target_lag}: Reached target evals ({num_iterations_per_lag}) in iteration {iteration_this_lag}.")
                 break
            if candidates_generated_this_iter == 0 and generation_method == "random (fallback)":
                 logger.warning(f"Lag {target_lag}: Could not generate any candidates in iteration {iteration_this_lag}.")
        # End optimization loop for this lag

        # --- Store Best Result for target_lag ---
        if best_config_id_for_this_lag is not None:
            logger.info(f"--- Lag {target_lag} Best Result ---")
            logger.info(f"  Config ID: {best_config_id_for_this_lag}")
            logger.info(f"  Params: {best_config_params_for_this_lag}")
            logger.info(f"  Correlation @ Lag {target_lag}: {best_correlation_at_this_lag:.6f}")
            logger.info(f"  Score (Abs Corr): {best_score_for_this_lag:.6f}")
            best_config_per_lag[target_lag] = {
                'params': best_config_params_for_this_lag,
                'config_id': best_config_id_for_this_lag,
                'correlation_at_lag': best_correlation_at_this_lag,
                'score_at_lag': best_score_for_this_lag
            }
        else:
            logger.warning(f"Lag {target_lag}: No valid configuration found with non-NaN correlation at this lag.")
            best_config_per_lag[target_lag] = None # Explicitly mark as None if no best found

    # End outer lag loop

    # --- Batch Insert All Collected Correlations ---
    if master_correlations_to_batch_insert:
        logger.info(f"Attempting to batch insert correlations for {len(master_evaluated_configs)} unique configs ({len(master_correlations_to_batch_insert)} total records)...")
        conn = sqlite_manager.create_connection(db_path)
        if conn:
             try:
                 success = sqlite_manager.batch_insert_correlations(conn, master_correlations_to_batch_insert)
                 if not success: logger.error("Batch insertion of correlations failed.")
             except Exception as db_err: logger.error(f"Error during batch correlation insert call: {db_err}", exc_info=True)
             finally: conn.close()
        else: logger.error("Could not connect to DB for batch correlation insert.")
    else: logger.info("No correlations generated to batch insert.")

    # --- Prepare Final Results ---
    # 1. The best config per lag
    final_best_per_lag = best_config_per_lag

    # 2. The list of all unique evaluated configs
    final_all_evaluated_configs = []
    for cfg_hash, cfg_data in master_evaluated_configs.items():
         final_all_evaluated_configs.append({
              'indicator_name': indicator_name,
              'params': cfg_data['params'],
              'config_id': cfg_data['config_id']
         })

    logger.info(f"--- Per-Lag Optimization Finished for {indicator_name} ---")
    logger.info(f"Evaluated {len(master_evaluated_configs)} unique configurations across all lags.")
    found_best_count = sum(1 for v in final_best_per_lag.values() if v is not None)
    logger.info(f"Found best configurations for {found_best_count} out of {max_lag} target lags.")

    return final_best_per_lag, final_all_evaluated_configs