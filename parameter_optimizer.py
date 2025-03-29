# parameter_optimizer.py
import logging
import random
import json
import hashlib # Using hashlib for more standard hashing
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import sqlite3 # Needed for type hinting and potential error catching

# Import config module correctly
import config as app_config # <<< RENAMED IMPORT TO AVOID CONFLICT <<<
import indicator_factory
import correlation_calculator
import sqlite_manager
import utils
import parameter_generator # Keep for condition evaluation and potentially initial random generation

logger = logging.getLogger(__name__)

def _get_config_hash(params: Dict[str, Any]) -> str:
    """Generates a stable SHA256 hash for a parameter dictionary."""
    # Use hashlib for a more standard and potentially less collision-prone hash
    # Use json dumps with sort_keys for deterministic serialization
    config_str = json.dumps(params, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()

def _calculate_config_performance(
    params: Dict[str, Any],
    config_id: int, # Pass config_id for logging/context
    indicator_name: str,
    base_data_with_required: pd.DataFrame, # Pass base data + required OHLCV etc.
    max_lag: int,
    db_path: str, # Needed to write correlations
    symbol_id: int,
    timeframe_id: int
) -> Tuple[Optional[float], List[Optional[float]]]:
    """
    Calculates indicator, correlations, stores them, and returns performance score.

    Returns:
        Tuple[Optional[float], List[Optional[float]]]: (score, correlation_values)
        Score is max absolute correlation over the lags. Returns (None, []) on failure.
    """
    logger.debug(f"Calculating performance for Config ID {config_id} ({indicator_name}, {params})")

    # 1. Compute Indicator for this specific config
    config_details = {'indicator_name': indicator_name, 'params': params, 'config_id': config_id}
    # Note: _compute_single_indicator returns ONLY the new columns
    indicator_output_df = indicator_factory._compute_single_indicator(base_data_with_required.copy(), config_details) # Pass copy

    if indicator_output_df is None or indicator_output_df.empty:
        logger.warning(f"Indicator computation failed or returned empty for Config ID {config_id}. Score: None.")
        return None, []

    # Check for all-NaN columns in the output (should be handled by factory, but double-check)
    nan_check = indicator_output_df.isnull().all()
    all_nan_cols = nan_check[nan_check].index.tolist()
    if all_nan_cols:
         logger.warning(f"Indicator {indicator_name} (Cfg {config_id}) produced all-NaN columns: {all_nan_cols}. Score: None.")
         # Clean up potentially created empty config entry? Maybe not necessary.
         return None, []

    # 2. Combine with necessary base data ('close') and drop NaNs for correlation input
    if 'close' not in base_data_with_required.columns:
        logger.error("Base data missing 'close' column for correlation calculation.")
        return None, []

    # Ensure alignment using the original index
    data_for_corr = pd.concat([base_data_with_required[['close']], indicator_output_df.reindex(base_data_with_required.index)], axis=1)

    initial_len = len(data_for_corr)
    data_for_corr.dropna(inplace=True) # Drop rows with NaNs *before* calculating correlations
    dropped_rows = initial_len - len(data_for_corr)
    if dropped_rows > 0:
        logger.debug(f"Dropped {dropped_rows} rows with NaNs before correlation for Config ID {config_id}.")

    # Check if enough data remains
    min_required_len = max_lag + 3 # Need at least 3 points for stable correlation calc
    if len(data_for_corr) < min_required_len:
        logger.warning(f"Insufficient data ({len(data_for_corr)} rows) for correlation (max_lag={max_lag}) for Config ID {config_id}. Need {min_required_len}. Score: None.")
        return None, []

    # 3. Calculate Correlations for each output column of this indicator config
    all_config_correlations = [] # List to store correlation lists for each output column
    best_config_score = -1.0 # Max abs correlation found for *this config* across all its outputs/lags
    conn = None
    output_columns = list(indicator_output_df.columns) # Get the specific columns computed
    logger.debug(f"Config ID {config_id}: Calculating correlations for columns: {output_columns}")

    try:
        conn = sqlite_manager.create_connection(db_path)
        if not conn:
            logger.error(f"Failed to connect to DB {db_path} for writing correlations.")
            return None, []

        for indicator_col in output_columns:
            col_correlations = [None] * max_lag # Initialize with Nones
            max_abs_corr_for_col = 0.0

            if indicator_col not in data_for_corr.columns:
                 logger.error(f"Internal error: Indicator column '{indicator_col}' not found in data_for_corr. Skipping correlation for this column.")
                 continue

            # Check variance of the indicator column *before* looping through lags
            if data_for_corr[indicator_col].nunique(dropna=True) <= 1:
                 logger.warning(f"Indicator column '{indicator_col}' (Config ID {config_id}) has no variance. Skipping correlations for this column.")
                 # Store NaNs for this column's correlations
                 for lag in range(1, max_lag + 1):
                     sqlite_manager.insert_correlation(conn, symbol_id, timeframe_id, config_id, lag, np.nan)
                 conn.commit() # Commit the NaNs
                 all_config_correlations.append([np.nan] * max_lag) # Add list of NaNs
                 continue # Move to next output column

            # Calculate correlations for this specific output column
            for lag in range(1, max_lag + 1):
                correlation_value = correlation_calculator.calculate_correlation_indicator_vs_future_price(
                    data_for_corr, indicator_col, lag
                )
                col_correlations[lag-1] = correlation_value # Store correlation for this lag

                # Store in DB immediately within the loop
                sqlite_manager.insert_correlation(conn, symbol_id, timeframe_id, config_id, lag, correlation_value)

                # Update max absolute correlation found so far for this column
                if correlation_value is not None and not np.isnan(correlation_value):
                    max_abs_corr_for_col = max(max_abs_corr_for_col, abs(correlation_value))

            # Commit correlations for this column/config_id after all lags are done
            conn.commit()
            all_config_correlations.append(col_correlations) # Append the list of correlations for this column

            # Update the overall best score for this configuration (across all its output columns)
            best_config_score = max(best_config_score, max_abs_corr_for_col)
            logger.debug(f"Config ID {config_id}, Column '{indicator_col}': Max Abs Corr = {max_abs_corr_for_col:.4f}")

        # Determine the final score and return correlations of the first output column
        # (or an empty list if no columns were processed)
        first_output_correlations = all_config_correlations[0] if all_config_correlations else []
        # If best_config_score remained -1.0, it means no valid correlations were found at all
        score = best_config_score if best_config_score >= 0 else None # Use >= 0 check
        if score is None:
            logger.warning(f"Config ID {config_id}: No valid non-zero correlations found across all output columns.")
        else:
             logger.debug(f"Config ID {config_id}: Final Score (Max Abs Corr across outputs) = {score:.4f}")
        return score, first_output_correlations

    except Exception as e:
        logger.error(f"Error during performance calculation or DB write for Config ID {config_id}: {e}", exc_info=True)
        if conn:
            try: conn.rollback()
            except Exception as rb_e: logger.error(f"Rollback failed: {rb_e}")
        return None, []
    finally:
        if conn: conn.close()


def _generate_candidate(
    base_config: Optional[Dict[str, Any]],
    param_defs: Dict[str, Dict],
    conditions: List[Dict],
    evaluated_hashes: set,
    indicator_name: str # for logging
) -> Optional[Dict[str, Any]]:
    """
    Generates a new candidate configuration by perturbing the base_config.
    If base_config is None, it generates a random valid config (currently uses defaults).
    Ensures the generated config hash is not in evaluated_hashes.
    Returns None if it can't generate a new valid config after several tries.
    """
    max_tries = 30 # Increased tries to find a new valid config
    attempt = 0

    while attempt < max_tries:
        attempt += 1
        candidate_params = {}
        log_prefix = f"Candidate Gen (Try {attempt}/{max_tries}) for {indicator_name}:"

        if base_config:
            # --- Perturbation Logic ---
            candidate_params = base_config.copy()
            tweakable_params = [p for p, d in param_defs.items() if isinstance(d.get('default'), (int, float))]

            if not tweakable_params:
                 logger.warning(f"{log_prefix} No numeric tweakable parameters found for perturbation.")
                 # If no tweakable params, we can't generate variations from the base
                 # Fallback to trying random generation? Or just return None?
                 # For now, let's signal failure for perturbation.
                 return None # Cannot perturb

            param_to_tweak = random.choice(tweakable_params)
            details = param_defs[param_to_tweak]
            current_value = candidate_params[param_to_tweak]

            # Simplified perturbation: random small step up or down
            if isinstance(current_value, int):
                # Define min_val based on common parameter types
                period_params = ['fast', 'slow', 'fastperiod', 'slowperiod', 'signalperiod', 'timeperiod', 'timeperiod1', 'timeperiod2', 'timeperiod3', 'length', 'window', 'obv_period', 'price_period']
                min_val = 2 if param_to_tweak in period_params else 1
                step = max(1, int(current_value * 0.1)) # Step relative to current value, minimum 1
                change = random.choice([-step, step, -step*2, step*2]) # Add wider steps occasionally
                new_value = max(min_val, current_value + change) # Enforce minimum
                candidate_params[param_to_tweak] = new_value
            elif isinstance(current_value, float):
                factor_params = ['fastlimit','slowlimit','acceleration','maximum','vfactor','scalar','nbdev','nbdevup','nbdevdn', 'smoothing']
                min_val = 0.01 if param_to_tweak in factor_params else 0.0 # Smallest typical value for factors
                step = max(0.01, abs(current_value * 0.1)) # Relative step, min 0.01
                change = random.choice([-step, step, -step*2, step*2]) * random.uniform(0.8, 1.2) # Add some randomness
                new_value = round(max(min_val, current_value + change), 4) # Enforce min, round
                candidate_params[param_to_tweak] = new_value
            else:
                 logger.warning(f"{log_prefix} Selected parameter '{param_to_tweak}' for perturbation is not numeric ({type(current_value)}). Trying another.")
                 continue # Try perturbing a different parameter

            logger.debug(f"{log_prefix} Perturbed '{param_to_tweak}' from {base_config.get(param_to_tweak)} to {candidate_params[param_to_tweak]}")

        else:
            # --- Random Generation Logic (Fallback/Exploration) ---
            # Currently uses defaults as the 'random' starting point.
            # A better approach would be to randomly select values within valid ranges.
            logger.debug(f"{log_prefix} Base config was None, generating 'random' candidate (using defaults).")
            candidate_params = {k: v.get('default') for k, v in param_defs.items() if 'default' in v}
            # If even the default is invalid (checked earlier), this will likely fail validation too.


        # --- Validation and Uniqueness Check ---
        candidate_hash = _get_config_hash(candidate_params)
        is_valid = parameter_generator.evaluate_conditions(candidate_params, conditions)
        is_new = candidate_hash not in evaluated_hashes

        if is_valid and is_new:
            logger.debug(f"{log_prefix} Found new valid candidate: {candidate_params}")
            return candidate_params
        elif not is_valid:
             logger.debug(f"{log_prefix} Candidate {candidate_params} is invalid according to conditions.")
        elif not is_new:
             logger.debug(f"{log_prefix} Candidate {candidate_params} hash {candidate_hash[:8]}... already evaluated.")
             # If we perturbed and hit an existing hash, maybe try perturbing a different param next time?

    logger.warning(f"Could not generate a new, valid configuration for {indicator_name} after {max_tries} tries.")
    return None


def optimize_parameters(
    indicator_name: str,
    indicator_def: Dict,
    base_data_with_required: pd.DataFrame,
    max_lag: int,
    num_iterations: int, # Max number of *new* evaluations to attempt
    target_configs: int, # How many top configs to return
    db_path: str,
    symbol_id: int,
    timeframe_id: int
) -> List[Dict[str, Any]]:
    """
    Optimizes parameters for a single indicator using an iterative feedback loop.
    Aims to find configurations with high absolute correlation scores.

    Args:
        indicator_name: Name of the indicator.
        indicator_def: Definition from indicator_params.json.
        base_data_with_required: DataFrame with OHLCV + date.
        max_lag: Maximum lag for correlation calculation.
        num_iterations: Max number of new configurations to evaluate.
        target_configs: The number of best configurations to return.
        db_path: Path to the SQLite database.
        symbol_id: ID of the symbol.
        timeframe_id: ID of the timeframe.

    Returns:
        List of dictionaries for the top 'target_configs' configurations found,
        sorted by score (desc), including score and correlations.
        Format: [{'indicator_name': ..., 'params': ..., 'config_id': ..., 'score': ..., 'correlations': [...]}, ...]
    """
    logger.info(f"--- Starting Parameter Optimization for: {indicator_name} ---")
    logger.info(f"Iterations: {num_iterations}, Target Configs: {target_configs}, Max Lag: {max_lag}")

    param_defs = indicator_def.get('parameters', {})
    conditions = indicator_def.get('conditions', [])

    # Store evaluated configs with details: {'params': {}, 'hash': '', 'config_id': 0, 'score': 0.0, 'correlations': []}
    evaluated_configs: List[Dict[str, Any]] = []
    evaluated_hashes: set = set() # Track hashes to avoid re-evaluating identical configs

    # --- 1. Evaluate Default Configuration ---
    default_params = {k: v.get('default') for k, v in param_defs.items() if 'default' in v}
    if not default_params:
        logger.error(f"Indicator {indicator_name} has no default parameters defined. Cannot optimize.")
        return []
    if not parameter_generator.evaluate_conditions(default_params, conditions):
        logger.error(f"Default parameters for {indicator_name} are invalid according to conditions. Cannot optimize.")
        # TODO: Could try generating a random valid config as starting point? For now, fail.
        return []

    logger.info(f"Evaluating default config: {default_params}")
    conn = sqlite_manager.create_connection(db_path)
    if not conn: return []
    try:
        default_config_id = sqlite_manager.get_or_create_indicator_config_id(conn, indicator_name, default_params)
    except Exception as e:
        logger.error(f"Failed to get/create config ID for default params: {e}", exc_info=True)
        conn.close(); return []
    finally: conn.close()

    # Calculate performance (this also stores correlations in DB)
    default_score, default_corrs = _calculate_config_performance(
        default_params, default_config_id, indicator_name, base_data_with_required, max_lag, db_path, symbol_id, timeframe_id
    )
    default_hash = _get_config_hash(default_params)

    best_config_params = default_params # Initialize best with default
    best_score = -1.0 # Initialize best score

    if default_score is not None:
        evaluated_configs.append({
            'params': default_params, 'hash': default_hash, 'config_id': default_config_id,
            'score': default_score, 'correlations': default_corrs
        })
        evaluated_hashes.add(default_hash)
        best_score = default_score # Update best score if default is valid
        logger.info(f"Default config score: {best_score:.4f}")
    else:
        logger.warning("Default configuration failed evaluation. Starting optimization from scratch/random.")
        best_config_params = None # Cannot perturb from default if it failed


    # --- 2. Iterative Optimization Loop ---
    successful_evaluations = 1 if default_score is not None else 0 # Count successful evals
    iteration = 0

    while successful_evaluations < num_iterations:
        iteration += 1
        logger.debug(f"\n--- Optimizer Iteration {iteration} (Evaluations: {successful_evaluations}/{num_iterations}) ---")

        # Determine base for generating candidates: Use current best if valid, else try random.
        # Use the aliased import 'app_config' here
        explore = random.random() < app_config.OPTIMIZER_RANDOM_EXPLORE_PROB # <<< FIXED LINE <<<
        base_for_candidate = None if explore or best_score < 0 else best_config_params # Perturb best if available and valid
        if base_for_candidate:
             logger.debug("Perturbing best known configuration.")
        else:
             logger.debug("Generating 'random' candidate (using defaults or attempting actual random if implemented).")

        candidates_evaluated_this_iter = 0
        # Use the aliased import 'app_config' here
        while candidates_evaluated_this_iter < app_config.OPTIMIZER_CANDIDATES_PER_ITERATION: # <<< FIXED LINE <<<
             if successful_evaluations >= num_iterations: break # Check budget before generating

             candidate_params = _generate_candidate(
                 base_for_candidate, param_defs, conditions, evaluated_hashes, indicator_name
             )

             if candidate_params is None:
                 logger.warning("Failed to generate a new valid candidate in this attempt.")
                 # If perturbation failed, try random explore next time within this iteration
                 if base_for_candidate is not None:
                      base_for_candidate = None
                      continue # Try again with random base
                 else:
                      logger.error("Failed to generate even a 'random' valid candidate. Stopping iteration.")
                      break # Stop trying for this iteration if random also fails

             candidate_hash = _get_config_hash(candidate_params)
             evaluated_hashes.add(candidate_hash) # Mark as evaluated
             candidates_evaluated_this_iter += 1

             # Get Config ID (must handle potential DB errors)
             conn = sqlite_manager.create_connection(db_path)
             if not conn: continue
             try:
                 config_id = sqlite_manager.get_or_create_indicator_config_id(conn, indicator_name, candidate_params)
             except Exception as e:
                 logger.error(f"Failed to get/create config ID for candidate {candidate_params}: {e}")
                 conn.close(); continue # Skip candidate
             finally: conn.close()

             # Evaluate performance
             score, correlations = _calculate_config_performance(
                 candidate_params, config_id, indicator_name, base_data_with_required, max_lag, db_path, symbol_id, timeframe_id
             )

             if score is not None:
                 logger.info(f"Iter {iteration}: Evaluated Config ID {config_id} ({candidate_params}). Score: {score:.4f}")
                 evaluated_configs.append({
                     'params': candidate_params, 'hash': candidate_hash, 'config_id': config_id,
                     'score': score, 'correlations': correlations
                 })
                 successful_evaluations += 1

                 # Update best score and params if improved
                 if score > best_score:
                     logger.info(f"***** New best score found: {score:.4f} (Config ID {config_id}) *****")
                     best_score = score
                     best_config_params = candidate_params # Update the base for next perturbation
             else:
                 logger.warning(f"Iter {iteration}: Evaluation failed for Config ID {config_id} ({candidate_params}).")

             # Immediate check after incrementing successful_evaluations
             if successful_evaluations >= num_iterations: break

        # Safety break if inner loop somehow doesn't exit but outer loop condition is met
        if successful_evaluations >= num_iterations:
            logger.info(f"Reached target number of {num_iterations} successful evaluations during iteration {iteration}.")
            break

        # Check if we seem stuck after many iterations without finding candidates
        if iteration >= num_iterations * 2 and successful_evaluations < num_iterations:
             logger.warning(f"Optimizer potentially stuck after {iteration} iterations with only {successful_evaluations} successful evals. Stopping early.")
             break


    # --- 3. Select Top Configurations ---
    if not evaluated_configs:
        logger.error(f"No configurations were successfully evaluated for {indicator_name}.")
        return []

    # Sort by score (descending) - higher score (max abs correlation) is better
    evaluated_configs.sort(key=lambda x: x.get('score', -1.0), reverse=True) # Use get with default for safety

    # Keep only the top N configurations specified by target_configs
    top_configs = evaluated_configs[:target_configs]

    logger.info(f"--- Optimization Finished for {indicator_name} ---")
    logger.info(f"Evaluated {len(evaluated_hashes)} unique configurations ({successful_evaluations} successful evaluations attempted).")
    logger.info(f"Best score found: {best_score:.4f}")
    logger.info(f"Returning top {len(top_configs)} configurations.")

    # Prepare list in the format expected by main.py (include score and correlations for potential reporting)
    results_for_main = []
    for result_config in top_configs: # Use a different variable name here to avoid confusion
        # Ensure correlations list is included, defaulting to empty list if missing
        corrs = result_config.get('correlations', [])
        results_for_main.append({
            'indicator_name': indicator_name,
            'params': result_config['params'],
            'config_id': result_config['config_id'],
            'score': result_config['score'], # Keep score
            'correlations': corrs # Keep correlations
        })

    return results_for_main